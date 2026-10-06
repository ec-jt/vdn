"""Tiny runtime contracts. No checkpoint/decoder loads or full-model execution."""
import copy
import os
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from src.config.inference import InferenceConfig, PaperComputeConfig, paper_enabled, validate_memory
from src.inference.utils.paper_adaln import AdaLNSchedule, precompute_adaln
from src.inference.utils.paper_runtime import CoordinatedFiniteGuard, PaperRuntime, convert_paper_projections
from src.models.ops.mxfp8_linear import SUPPORTED_ROLES, MXFP8Linear
from src.models.sequence_layout import SequenceLayout


def config(**flags):
    cfg = OmegaConf.structured(InferenceConfig)
    cfg.checkpoint = "unused-local-checkpoint"
    cfg.memory.policy = "resident_weight_shards"
    cfg.memory.local_files_only = True
    cfg.kernels.inference_kernels = False
    cfg.kernels.softmax_backend = "flex"
    cfg.parallel.softmax_ranks = 0
    cfg.render.num_frames, cfg.render.num_steps = 124, 8
    cfg.render.prompt_file = "prompts/example_0.pt"
    for key, value in flags.items():
        setattr(cfg.compute, key, value)
    return cfg


def runtime(cfg, world_size=1):
    rt = SimpleNamespace(device=torch.device("cpu"), world_size=world_size, rank=0)
    return PaperRuntime(cfg, rt)


def test_original_defaults_unchanged_and_all_switches_independent():
    cfg = InferenceConfig()
    assert cfg.kernels.inference_kernels and not paper_enabled(cfg)
    for name, value in asdict(PaperComputeConfig()).items():
        if type(value) is bool:
            assert value is False
            candidate = config(**{name: True})
            validate_memory(candidate)
            assert paper_enabled(candidate)


@pytest.mark.parametrize("field,value", [
    ("render.num_frames", 345), ("render.num_steps", 50), ("render.seed", 1),
    ("render.prompt_file", "prompts/example_1.pt"), ("render.warmup_steps", 2),
    ("kernels.inference_kernels", True), ("kernels.vda_fused", True),
    ("precision.fp8.enabled", True), ("parallel.softmax_ranks", 6),
    ("memory.policy", "legacy"), ("compute.paper_max_query_rows", 0),
])
def test_experimental_admission_fails_closed(field, value):
    cfg = config(paper_adaln=True)
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError):
        validate_memory(cfg)


def projection_model(device="cpu", blocks=50):
    with torch.device(device):
        model = nn.Module()
        model.root = nn.Linear(4, 4, dtype=torch.float32)
        model.transformer_blocks = nn.ModuleList()
        for _ in range(blocks):
            block = nn.Module()
            for path in SUPPORTED_ROLES.values():
                parent = block
                for piece in path.split(".")[:-1]:
                    if piece not in parent._modules:
                        parent.add_module(piece, nn.Module())
                    parent = parent._modules[piece]
                parent.add_module(path.split(".")[-1], nn.Linear(4, 4, dtype=torch.bfloat16))
            model.transformer_blocks.append(block)
    return model.eval().requires_grad_(False)


def test_cpu_meta_350_conversion_after_merges_preserves_root_and_slots():
    cfg = config(paper_mxfp8=True)
    cpu, meta = projection_model(), projection_model("meta")
    root = cpu.root.weight.clone()
    cpu.transformer_blocks[0].attn.orig.to_q.weight.add_(1)
    records = convert_paper_projections(cpu, cfg)
    assert records == convert_paper_projections(meta, cfg)
    assert len(records) == 350
    assert {r["block"] for r in records} == set(range(50))
    assert torch.equal(cpu.root.weight, root) and cpu.root.weight.dtype == torch.float32
    for record in records:
        a, b = cpu.get_submodule(record["module"]), meta.get_submodule(record["module"])
        assert isinstance(a, MXFP8Linear) and not hasattr(a, "weight")
        assert a.check_finite and b.check_finite
        assert [(n, t.shape, t.dtype) for n, t in a.named_buffers()] == [
            (n, t.shape, t.dtype) for n, t in b.named_buffers()]


def test_conversion_disabled_and_truncation_rejected():
    toy = projection_model(blocks=1)
    assert convert_paper_projections(toy, config()) == []
    with pytest.raises(ValueError, match="50 blocks"):
        convert_paper_projections(toy, config(paper_mxfp8=True))


def test_finite_guard_accumulates_until_boundary_and_is_terminal():
    guard = CoordinatedFiniteGuard("cpu")
    guard.observe(torch.ones(3))
    guard.check("safe")
    assert guard.boundaries == 1
    guard.observe(torch.tensor([float("nan")]))
    guard.observe(torch.ones(3))
    with pytest.raises(FloatingPointError, match="all-rank"):
        guard.check("exchange")
    with pytest.raises(RuntimeError, match="terminal"):
        guard.check("later")


def test_encoding_guard_and_exception_vote():
    guard = CoordinatedFiniteGuard("cpu")
    guard.encoded(torch.tensor([127], dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8))
    with pytest.raises(FloatingPointError):
        guard.check("encoded")
    guard = CoordinatedFiniteGuard("cpu")
    with pytest.raises(FloatingPointError, match="injected"):
        with guard.local_phase("local"):
            raise ValueError("injected")
    with pytest.raises(ValueError, match="control group"):
        CoordinatedFiniteGuard("cpu", world_size=2)


def _finite_worker(rank, rendezvous, result, mismatch):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    guard = CoordinatedFiniteGuard("cpu", group=dist.group.WORLD, world_size=2)
    try:
        guard.observe(torch.tensor([float("inf") if rank and not mismatch else 1.]))
        try:
            guard.check("other_phase" if rank and mismatch else "before_payload")
        except FloatingPointError:
            result[rank] = 1
        else:
            # Must never reach the next application collective on either rank.
            result[rank] = -1
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", [False, True])
def test_real_two_rank_nonfinite_consensus_before_next_collective(tmp_path, mismatch):
    result = torch.zeros(2, dtype=torch.int32).share_memory_()
    mp.spawn(_finite_worker, args=("file://" + str(tmp_path / "gloo"), result, mismatch), nprocs=2, join=True)
    assert result.tolist() == [1, 1]


def adaln_model():
    from diffusers.models.embeddings import Timesteps, TimestepEmbedding
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AdaLayerNormModulation, MiniMaxH3AdaLayerNormOut
    model = nn.Module()
    model.time_proj = Timesteps(16, flip_sin_to_cos=True, downscale_freq_shift=0)
    model.time_embedder = TimestepEmbedding(16, 32, out_dim=12)
    model.transformer_blocks = nn.ModuleList([nn.Module() for _ in range(2)])
    for block in model.transformer_blocks:
        block.adaln_proj = MiniMaxH3AdaLayerNormModulation(12, 8).to(torch.bfloat16)
    model.norm_out = MiniMaxH3AdaLayerNormOut(8, 12, 1e-6).to(torch.bfloat16)
    return model.eval().requires_grad_(False)


def test_adaln_runtime_preparation_transport_step_close_and_charge():
    paper = runtime(config(paper_adaln=True))
    model = adaln_model()
    paper.schedule = AdaLNSchedule([torch.tensor([0.]), torch.tensor([.2, .7])])
    paper.steps = 2
    paper.prepare_adaln(model)
    paper.transport_adaln()
    paper.state = "active"
    for i in range(2):
        view = paper.step(paper.schedule.timestep(i))
        with torch.no_grad():
            expected = model.transformer_blocks[0].adaln_proj(view.embedding)
        for a, b in zip(view.block(0), expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert paper.preparation_charge() > 0
    assert paper.calls["adaln_precompute_projection"] == 6
    paper.finish()
    with pytest.raises(RuntimeError, match="closed"):
        view.block(0)
    paper.close()
    paper.close()


def _table_worker(rank, rendezvous, result):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    cfg = config(paper_adaln=True)
    rt = SimpleNamespace(device=torch.device("cpu"), rank=rank, world_size=2)
    paper = PaperRuntime(cfg, rt, control_group=dist.group.WORLD)
    paper.schedule = AdaLNSchedule([torch.tensor([0.]), torch.tensor([.2, .7])])
    try:
        if rank == 0:
            paper.prepare_adaln(adaln_model())
        paper.transport_adaln()
        table = paper.adaln.step(1, paper.schedule.timestep(1)).block(0)[0]
        tables = [None, None]
        dist.all_gather_object(tables, table)
        result[rank] = int(torch.equal(*tables) and paper.adaln.stats.steps == 2)
    finally:
        paper.close_tables()
        dist.destroy_process_group()


def test_real_two_rank_trusted_cpu_table_transport(tmp_path):
    result = torch.zeros(2, dtype=torch.int32).share_memory_()
    mp.spawn(_table_worker, args=("file://" + str(tmp_path / "tables"), result), nprocs=2, join=True)
    assert result.tolist() == [1, 1]


def test_fresh_request_and_same_shape_changed_schedule_rejected():
    cfg = config(paper_adaln=True)
    paper = runtime(cfg)
    layout = SequenceLayout(8, 2, 3, 2)
    video, audio = torch.tensor([0., .3]), torch.tensor([0., .6])
    paper.schedule = AdaLNSchedule.from_timesteps(video, audio)
    paper.request_key = (layout, 0)
    paper.steps = 2
    kwargs = dict(frames=124, steps=2, seed=42, shifts=(12., 3.))
    paper.begin(layout, 0, video, audio, **kwargs)
    with pytest.raises(FloatingPointError, match="exactly one"):
        paper.begin(layout, 0, video, audio, **kwargs)
    other = runtime(cfg)
    other.schedule, other.request_key, other.steps = paper.schedule, paper.request_key, 2
    with pytest.raises(FloatingPointError, match="schedule changed"):
        other.begin(layout, 0, video, torch.tensor([0., .7]), **kwargs)


def test_plan_key_includes_geometry_heads_device_bounds_and_budgets():
    paper = runtime(config(paper_window=True))
    layout = SequenceLayout(8, 2, 3, 2)
    attn = SimpleNamespace(layout=layout, anchor_frames="both", num_heads=56, head_dim=128,
                           _bounds=lambda l: [(0, 1)] * l.num_frames)
    key = paper.plan_key(attn, layout)
    paper.plans[key] = object()
    paper.state = "active"
    assert paper.window_plan(attn) is paper.plans[key]
    for name, value in (("num_heads", 48), ("anchor_frames", "none"), ("head_dim", 64)):
        old = getattr(attn, name)
        setattr(attn, name, value)
        with pytest.raises(ValueError, match="geometry changed"):
            paper.window_plan(attn)
        setattr(attn, name, old)
    paper.options["paper_max_query_rows"] += 1
    with pytest.raises(ValueError, match="geometry changed"):
        paper.window_plan(attn)


@pytest.mark.parametrize("fused", [False, True])
def test_explicit_adaln_eager_and_fused_consumers_without_projection(monkeypatch, fused):
    from src.models.ops import fused_block
    model = adaln_model()
    block = model.transformer_blocks[0]
    block.norm1 = nn.RMSNorm(8, dtype=torch.bfloat16)
    block.norm2 = nn.RMSNorm(8, dtype=torch.bfloat16)
    class Attention(nn.Module):
        def forward(self, x, rotary, mask):
            return x * .25
    block.attn = Attention()
    block.ff = nn.Linear(8, 8, dtype=torch.bfloat16)
    schedule = AdaLNSchedule([torch.tensor([0., .5])])
    request = precompute_adaln(schedule, time_proj=model.time_proj, time_embedder=model.time_embedder,
        block_projections=[block.adaln_proj], norm_out=model.norm_out)
    view = request.step(0, schedule.timestep(0))
    hidden = torch.randn(1, 6, 8).to(torch.bfloat16)
    indices = torch.arange(6)
    monkeypatch.setattr(fused_block, "_compiled", lambda name, body, **options: body)
    with torch.no_grad():
        expected = fused_block.paper_block_forward(block, hidden, view.embedding, indices, None)
    monkeypatch.setattr(block.adaln_proj, "forward", lambda *_: pytest.fail("lookup ran projection"))
    with torch.no_grad():
        actual = fused_block.paper_block_forward(block, hidden, view.embedding, indices, None,
                                                tables=view.block(0), fused=fused)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    request.close()


def test_native_inventory_counts_actual_quantized_calls_and_restores(monkeypatch):
    cfg = config(paper_mxfp8=True)
    model = projection_model()
    paper = runtime(cfg)
    paper.conversion = convert_paper_projections(model, cfg)
    def tiny_native(self, data, scales, *, rows=None, **kwargs):
        return torch.ones(rows, self.out_features, dtype=torch.bfloat16)
    monkeypatch.setattr(MXFP8Linear, "forward_quantized", tiny_native)
    paper.install(model)
    projection = model.transformer_blocks[0].attn.orig.to_q
    assert not projection.check_finite and not paper.calls["mxfp8.q"]
    projection.forward_quantized(torch.zeros(128, 128, dtype=torch.uint8),
                                 torch.zeros(512, dtype=torch.uint8), rows=3)
    assert paper.calls["mxfp8.q"] == 1
    paper.guard.check("test")
    paper.close()
    assert projection.check_finite and "forward_quantized" not in projection.__dict__


def test_sampler_failure_closes_owned_tables():
    from src.inference.render import _paper_request_cleanup
    paper = runtime(config(paper_adaln=True))
    paper.schedule = AdaLNSchedule([torch.tensor([0.])])
    paper.prepare_adaln(adaln_model())
    request = paper.adaln
    @_paper_request_cleanup
    def fail(transformer):
        raise RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        fail(SimpleNamespace(_paper_runtime=paper))
    assert request._closed and paper.adaln is None and paper.state == "failed"


@torch.no_grad()
def test_transformer_adaln_lookup_before_blocks_and_output_seam(monkeypatch):
    from src.inference.utils import ulysses
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    model = adaln_model()
    model.transformer_blocks = nn.ModuleList([MiniMaxH3TransformerBlock(8, 2, 4, 16, 12, 1e-6, 1e-6).bfloat16()])
    class Attention(nn.Module):
        num_heads = 2
        def forward(self, x, rotary, mask=None):
            return x * .125
    model.transformer_blocks[0].attn = Attention()
    for name in ("proj_in", "audio_proj_in", "context_embedder", "proj_out", "audio_proj_out"):
        setattr(model, name, nn.Linear(8, 8).bfloat16())
    model.token_refiner = nn.Identity()
    model.rope = lambda positions: (torch.ones(len(positions), 4), torch.zeros(len(positions), 4))
    rt = SimpleNamespace(configure=lambda *args: None, local_start=0, local_end=6,
                         profile_start=lambda: None, profile_end=lambda *args: None,
                         gather_sequence=lambda x: x)
    model._ulysses_runtime = rt
    model.eval().requires_grad_(False)
    monkeypatch.setattr(ulysses, "iter_hybrids", lambda m: iter([model.transformer_blocks[0].attn]))
    inputs = dict(hidden_states=torch.randn(1,2,8).bfloat16(), audio_hidden_states=torch.randn(1,2,8).bfloat16(),
        encoder_hidden_states=torch.randn(1,2,8).bfloat16(), timestep=torch.tensor([0., .6]),
        timestep_indices=torch.tensor([0,0,1,1,0,0]), token_tags=torch.tensor([1,1,2,2,0,0]),
        position_ids=torch.zeros(6,3), video_indices=torch.tensor([4,5]), audio_indices=torch.tensor([2,3]),
        text_indices=torch.tensor([0,1]), return_dict=False)
    expected = ulysses._ulysses_transformer_forward(model, **inputs)
    paper = runtime(config(paper_adaln=True))
    paper.schedule = AdaLNSchedule([inputs["timestep"]])
    paper.steps = 1
    paper.prepare_adaln(model)
    paper.transport_adaln()
    paper.state = "active"
    model._paper_runtime = paper
    monkeypatch.setattr(model.transformer_blocks[0].adaln_proj, "forward", lambda *_: pytest.fail("block projected"))
    monkeypatch.setattr(model.norm_out, "forward", lambda *_: pytest.fail("output projected"))
    actual = ulysses._ulysses_transformer_forward(model, **inputs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert paper.calls["adaln_block_lookup"] == paper.calls["adaln_output_lookup"] == 1
    paper.finish()
    paper.close()


def attention_fixture(device="cpu"):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention
    from src.models.hybrid_attention import HybridAttention
    from src.models.softmax_attention.paper_window import prepare_paper_window_plan
    cfg = config(paper_window=True, paper_qk_rope=True, paper_vda=True)
    paper = runtime(cfg)
    paper.runtime.device = torch.device(device)
    paper.runtime.world_size = 8
    paper.guard = CoordinatedFiniteGuard(device)
    layout = SequenceLayout(18, 4, 7, 2, 1, 2, 0, 2)
    original = MiniMaxH3Attention(128, heads=56, dim_head=128)
    attn = HybridAttention(original, 128, delta_rule="vdn_solve", radius=0, chunk=2,
                           anchor_frames="both", enable_text_state=True, a_fp32=True)
    attn = attn.eval().requires_grad_(False)
    attn.layout = layout
    branch = attn.linear_attention
    branch.paper_vda_enabled = True
    model = nn.Module()
    block = nn.Module()
    block.attn = attn
    # The seven-role native whitelist needs a SwiGLU-shaped FFN graph.
    up = nn.Module()
    up.proj = nn.Linear(128, 256)
    block.ff = nn.Module()
    block.ff.net = nn.ModuleList([up, nn.Identity(), nn.Linear(128, 128)])
    model.transformer_blocks = nn.ModuleList([block])
    model.eval().requires_grad_(False).to(dtype=torch.bfloat16)
    paper.state = "active"
    return paper, model, attn, layout, prepare_paper_window_plan


@torch.no_grad()
def test_attention_wiring_raw_only_out_of_place_and_prefetch_order(monkeypatch):
    from src.inference.utils import ulysses
    from src.models.softmax_attention import paper_qk_rope, paper_window
    paper, model, attn, layout, planner = attention_fixture()
    events = []
    raw_snapshot = []
    def exchange(x):
        events.append("input")
        assert x.shape == (layout.seq_len, 56, 4 * 128 + 2)
        raw_snapshot.extend(t.clone() for t in x[:, :7].split((128,128,128,1,1,128), -1)[:3])
        return x[:, :7].contiguous()
    def reverse(x):
        events.append("output")
        return x.repeat(1, 8, 1)
    rt = SimpleNamespace(rank=0, heads_per_rank=7, local_start=0, local_end=layout.seq_len,
        video_frame_mean_async=lambda x, l: (torch.zeros(l.num_frames, 128), SimpleNamespace(wait=lambda: None)),
        sequence_to_heads=exchange, heads_to_sequence=reverse,
        weight_shards=SimpleNamespace(prefetch=True, prefetch_next=lambda: events.append("prefetch")))
    attn._ulysses_runtime, attn._paper_runtime = rt, paper
    paper.rotary = (torch.ones(layout.seq_len, 96), torch.zeros(layout.seq_len, 96))
    paper.plans[paper.plan_key(attn, layout)] = planner(layout, attn._bounds(layout), "both")
    def qk(q, k, qw, kw, eps, cos, sin, **kwargs):
        assert q.shape == (layout.seq_len, 7, 128)
        assert cos.shape == (layout.seq_len, 96)
        return q.clone() + 1, k.clone() + 2
    def linear(a, raw, *args):
        for actual, expected in zip(raw, raw_snapshot):
            assert torch.equal(actual, expected)
        events.append("vda")
        return torch.zeros_like(raw[0])
    monkeypatch.setattr(paper_qk_rope, "paper_qk_rmsnorm_rope", qk)
    monkeypatch.setattr(paper_window, "paper_window_attention", lambda q,k,v,*args,**kw: v.clone())
    monkeypatch.setattr(ulysses, "_linear_branch_shard", linear)
    out = ulysses._paper_attention_forward(attn, torch.randn(1, layout.seq_len, 128).bfloat16())
    assert out.shape == (1, layout.seq_len, 128)
    assert events == ["input", "vda", "output", "prefetch"]
    assert paper.calls["qk_rope.fused"] == paper.calls["softmax.native_packed"] == 1
    paper.close()


GPU0 = os.environ.get("PAPER_RUNTIME_GPU0_TESTS") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"


@pytest.mark.skipif(not GPU0, reason="explicit idle GPU0 tiny integration only")
@pytest.mark.parametrize("native", [False, True])
@torch.no_grad()
def test_gpu0_real_qk_packed_vda_overlap_and_native_projections(native):
    from src.inference.utils.ulysses import _paper_attention_forward
    from src.models.ops.mxfp8_linear import convert_linear_to_mxfp8
    paper, model, attn, layout, planner = attention_fixture("cuda:0")
    paper.options["paper_mxfp8"] = native
    if native:
        paper.conversion = convert_linear_to_mxfp8(model, enabled=True)
    model.to("cuda:0")
    paper.install(model)
    rt = SimpleNamespace(rank=0, heads_per_rank=7, local_start=0, local_end=layout.seq_len,
        video_frame_mean_async=lambda x, l: (x[l.video_start:l.video_end].reshape(l.num_frames, l.tokens_per_frame, -1).sum(1, dtype=torch.float32), SimpleNamespace(wait=lambda: None)),
        sequence_to_heads=lambda x: x[:, :7].contiguous(), heads_to_sequence=lambda x: x.repeat(1,8,1))
    attn._ulysses_runtime = rt
    paper.rotary = (torch.ones(layout.seq_len, 96, device="cuda:0"),
                    torch.zeros(layout.seq_len, 96, device="cuda:0"))
    paper.plans[paper.plan_key(attn, layout)] = planner(layout, attn._bounds(layout), "both", device="cuda:0")
    hidden = torch.randn(1, layout.seq_len, 128, device="cuda:0", dtype=torch.bfloat16) * .1
    try:
        serial = _paper_attention_forward(attn, hidden)
        paper.options["paper_vda_overlap"] = True
        overlapped = _paper_attention_forward(attn, hidden)
        torch.cuda.synchronize()
        torch.testing.assert_close(overlapped, serial, atol=0, rtol=0)
        assert torch.isfinite(overlapped).all()
        assert paper.calls["softmax.native_packed"] == paper.calls["qk_rope.fused"] == 2
        assert paper.calls["vda.paper"] == 2 and paper.calls["vda.side_stream"] == 1
        if native:
            assert all(paper.calls["mxfp8." + role] == 2 for role in ("q", "k", "v", "softmax_out", "linear_out"))
            from src.models.ops.fused_block import fast_ff_forward
            ff_out = fast_ff_forward(model.transformer_blocks[0].ff, hidden)
            paper.guard.check("ffn_test", ff_out)
            assert ff_out.shape == hidden.shape
            assert paper.calls["mxfp8.ff_up"] == paper.calls["mxfp8.ff_down"] == 1
    finally:
        paper.close()
