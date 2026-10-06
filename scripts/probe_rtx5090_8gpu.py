"""Eight-process toy sharded Ulysses correctness and representative PCIe collectives.

No production weight payloads or decoders are loaded. Launch via the bounded shell
wrapper, which checks idle GPUs before starting torchrun.
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist
from diffusers import MiniMaxH3Transformer3DModel
from src.inference.utils.weight_sharding import ResidentWeightShards
from src.inference.utils.ulysses import init_ulysses, install_ulysses
from src.models.hybrid_transform import apply_hybrid_attention_transform, set_layout
from src.models.sequence_layout import layout_from_indices


def toy(runtime, distribution="scatter"):
    device = runtime.device
    torch.manual_seed(42)
    # Eight heads, two blocks, text state and real distant video beyond the window.
    config = dict(num_attention_heads=8, attention_head_dim=16, hidden_size=32,
                  num_layers=2, num_refiner_layers=1, ffn_dim=64, in_channels=4,
                  audio_in_channels=4, patch_size=(1, 1, 1), text_dim=8,
                  freq_dim=16, time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=2)
    with torch.device("cpu" if runtime.is_main else "meta"):
        model = MiniMaxH3Transformer3DModel(**config)
        apply_hybrid_attention_transform(model, dict(enable_softmax_gate=True, anchor_frames="both",
            softmax_attention=dict(radius=1, chunk=0), linear_attention=dict(delta_rule="sana_scaled",
            linear_head_dim=16, bridge="alpha", a_fp32=True, short_conv=dict(targets=["k", "v"]),
            enable_text_state=True)))
        model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    # Preserve FP32 endpoints just as the real baseline loader does.
    for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope"):
        getattr(model, name).float()
    oracle = copy.deepcopy(model).to(device) if runtime.is_main else None
    control = ResidentWeightShards(model, runtime, scatter_chunk_bytes=1024,
                                   distribution=distribution)
    control.distribute()
    model._weight_shards = control
    install_ulysses(model, runtime, softmax_ranks=0)
    frames, spatial, text, audio = 9, 4, 3, 2
    video_start = text + audio
    seq = video_start + frames * spatial
    video_idx = torch.arange(video_start, seq, device=device)
    text_idx = torch.arange(text, device=device)
    layout = layout_from_indices(video_idx, frames, spatial, seq_len=seq, frame_size=(2, 2), text_indices=text_idx)
    set_layout(model, layout)
    if oracle is not None:
        set_layout(oracle, layout)
    # Explicitly verify FP32 global means for frames split across rank boundaries.
    runtime.configure(seq, 8)
    full = torch.arange(seq * 32, device=device, dtype=torch.float32).view(seq, 32).remainder(31)
    sums, work = runtime.video_frame_mean_async(full[runtime.local_start:runtime.local_end], layout)
    work.wait()
    torch.testing.assert_close(sums / spatial,
        full[video_start:].view(frames, spatial, 32).mean(1), rtol=0, atol=0)
    torch.manual_seed(91)
    positions = torch.zeros(seq, 3, dtype=torch.long, device=device)
    positions[video_start:, 0] = torch.arange(frames, device=device).repeat_interleave(spatial)
    inputs = dict(hidden_states=torch.randn(1, frames*spatial, 4, device=device),
        audio_hidden_states=torch.randn(1, audio, 4, device=device),
        encoder_hidden_states=torch.randn(1, text, 8, device=device),
        timestep=torch.tensor([0.4], device=device), timestep_indices=torch.zeros(seq, dtype=torch.long, device=device),
        token_tags=torch.cat([torch.ones(text, device=device), torch.full((audio,), 2, device=device),
                              torch.zeros(frames*spatial, device=device)]).long(),
        position_ids=positions, video_indices=video_idx, text_indices=text_idx,
        audio_indices=torch.arange(text, video_start, device=device), return_dict=False)
    outputs = []
    with torch.no_grad():
        for _ in range(2):
            outputs.append(model(**inputs))
        for a, b in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        expected = oracle(**inputs) if oracle is not None else None
        errors = []
        if expected is not None:
            for actual, reference in zip(outputs[0], expected):
                delta = (actual.float()-reference.float()).abs()
                errors.append(dict(max_abs=delta.max().item(), mean_abs=delta.mean().item()))
                torch.testing.assert_close(actual, reference, atol=0.035, rtol=0.04)
    assert all(p.is_meta for b in model.transformer_blocks for p in b.parameters())
    control.release(); control.release()
    assert all(p.is_meta for p in model.parameters())
    result = dict(repeated_bitwise=True, global_frame_mean_exact=True,
                  reference_errors=errors, weight_summary=control.summary(),
                  sequence_splits=runtime.splits, endpoint_text_shortconv=True)
    del oracle, model, outputs, control
    torch.cuda.empty_cache()
    return result


def collective_probe(runtime, repeats, output_bytes):
    device, world = runtime.device, runtime.world_size
    n = ((output_bytes + 2*world-1)//(2*world))
    shard = torch.full((n,), runtime.rank, device=device, dtype=torch.bfloat16)
    gathered = torch.empty(n*world, device=device, dtype=torch.bfloat16)
    measurements = []
    for index in range(repeats + 1):
        runtime.barrier(); torch.cuda.synchronize(device)
        start = time.perf_counter()
        dist.all_gather_into_tensor(gathered, shard)
        torch.cuda.synchronize(device)
        measurements.append(time.perf_counter() - start)
    for rank in range(world):
        assert (gathered[rank*n:(rank+1)*n] == rank).all().item()
    del gathered, shard
    # Representative 124-frame QKV/gate pack, 56 heads, width 770 BF16.
    from src.inference.utils.ulysses_runtime import UlyssesRuntime
    rt = UlyssesRuntime(runtime.rank, world, runtime.local_rank, device, "nccl")
    rt.standard_exchange = runtime.standard_exchange
    rt.configure(38003, 56)
    x = torch.full((rt.splits[rt.rank], 56, 770), rt.rank, dtype=torch.bfloat16, device=device)
    a2a = []
    for index in range(repeats + 1):
        runtime.barrier(); torch.cuda.synchronize(device)
        start = time.perf_counter()
        heads = rt.sequence_to_heads(x)
        back = rt.heads_to_sequence(heads)
        torch.cuda.synchronize(device)
        a2a.append(time.perf_counter()-start)
        assert torch.equal(back, x)
        del heads, back
    return dict(allgather_output_bytes=n*world*2, allgather_seconds=measurements,
                alltoall_roundtrip_seconds=a2a, alltoall_local_payload_bytes=x.numel()*2,
                sequence_splits=rt.splits, first_sample_is_cold=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-bytes", type=int, default=1376730080)
    parser.add_argument("--distribution", choices=["scatter", "broadcast"], default="scatter")
    parser.add_argument("--exchange", choices=["alltoall", "broadcast"], default="alltoall")
    parser.add_argument("--toy-only", action="store_true")
    args = parser.parse_args()
    runtime = init_ulysses(timeout_seconds=180, profile_enabled=True)
    runtime.standard_exchange = args.exchange
    if runtime.world_size != 8:
        raise ValueError("probe requires eight ranks")
    try:
        torch.set_grad_enabled(False)
        result = dict(rank=runtime.rank, standard_exchange=args.exchange,
                      toy=toy(runtime, args.distribution))
        runtime.barrier()
        torch.cuda.reset_peak_memory_stats(runtime.device)
        if not args.toy_only:
            result["collectives"] = collective_probe(runtime, args.repeats, args.output_bytes)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(runtime.device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(runtime.device)
        result["profile_ms"] = runtime.profile_milliseconds()
        records = [None]*runtime.world_size
        dist.all_gather_object(records, result)
        if runtime.is_main:
            with open(args.report, "x") as f:
                json.dump(records, f, indent=2)
            print(json.dumps(records, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
