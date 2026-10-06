"""Real saved-block comparison. No checkpoint modification or full render.

Run eight ranks. Both paths use the same captured full effective weights and
reconstructed global input, with two warmups and ten synchronized measurements.
The baseline includes actual resident-shard gathers, not only GEMMs.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import time
import types

import torch
from torch import nn
import torch.distributed as dist
from omegaconf import OmegaConf

from src.config.inference import InferenceConfig
from src.inference.utils.weight_sharding import build_meta_model, ResidentWeightShards
from src.inference.utils.paper_runtime import PaperRuntime, convert_paper_projections
from src.inference.utils.ulysses import init_ulysses, prebuild_prompt_masks, _paper_attention_forward
from src.inference.utils.tensor_parallel import shard_block, install_tp_block
from src.models.ops.fused_block import paper_block_forward
from src.inference.utils.nsight import nvtx_range


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("baseline","tp"), required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.capture)
    runtime = init_ulysses(timeout_seconds=180)
    runtime.standard_exchange = "allgather"
    paper = controller = None
    try:
        cfg = OmegaConf.merge(OmegaConf.structured(InferenceConfig), OmegaConf.load("configs/inference/8nfe_rtx5090_paper_stack.yaml"))
        model, art = build_meta_model(cfg)
        paper = PaperRuntime.create(cfg, runtime)
        paper.conversion = convert_paper_projections(model, cfg)
        prebuild_prompt_masks(cfg, model, runtime.device, paper=paper)
        layout, _ = paper.request_key
        runtime.configure(layout.seq_len, 56)
        block = model.transformer_blocks[0]
        # Count actual full 50-block tensors before freeing the unused meta graph.
        full_manifest = {n: dict(shape=list(t.shape), dtype=str(t.dtype), bytes=t.numel()*t.element_size())
                         for n,t in list(model.named_parameters())+list(model.named_buffers())}
        del model, art
        gc.collect()
        weights = torch.load(source/"trace-rank-0/000-weights.pt",map_location="cpu",weights_only=True,mmap=True)
        if args.mode == "tp" or runtime.is_main:
            block.load_state_dict(weights, assign=True, strict=True)
        del weights
        wrapper = nn.Module()
        wrapper.transformer_blocks = nn.ModuleList([block])
        wrapper.eval().requires_grad_(False)
        if args.mode == "tp":
            shard_block(block, runtime.rank, runtime.world_size)
            # Full AdaLN source is CPU-only; no block projection is executed.
            block.adaln_proj.to_empty(device="meta")
            for name, child in block.named_children():
                if name != "adaln_proj":
                    child.to(runtime.device)
        else:
            controller = ResidentWeightShards(wrapper,runtime,prefetch=True,distribution="broadcast")
            controller.distribute()
            runtime.weight_shards = controller
        block.attn.layout = layout
        paper.install(wrapper)
        paper.state = "active"
        if args.mode == "tp":
            install_tp_block(block, runtime, paper)
        else:
            block.attn._ulysses_runtime = runtime
            block.attn.forward = types.MethodType(_paper_attention_forward, block.attn)
        capture = torch.load(source/f"first-rank-{runtime.rank}.pt",map_location="cpu",weights_only=True,mmap=True)
        if args.mode == "tp":
            all_inputs = [torch.load(source/f"first-rank-{r}.pt",map_location="cpu",weights_only=True,mmap=True)["inputs"] for r in range(8)]
            inputs = [torch.cat([v[0] for v in all_inputs],dim=1),all_inputs[0][1],
                      *[torch.cat([v[i] for v in all_inputs]) for i in (2,3,4)],*all_inputs[0][5:]]
            del all_inputs
        else:
            inputs = capture["inputs"]
        hidden, temb, indices, cos, sin, *tables = [t.to(runtime.device) for t in inputs]
        del inputs
        # Q/K norm/RoPE always receives global rotary, even on baseline SP ranks.
        full_rotary = [torch.cat([torch.load(source/f"first-rank-{r}.pt",map_location="cpu",weights_only=True,mmap=True)["inputs"][i] for r in range(8)]).to(runtime.device) for i in (3,4)]
        paper.rotary = full_rotary
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        record = dict(mode=args.mode,rank=runtime.rank,geometry=dict(rows=layout.seq_len,heads=56,width=5376),
                      baseline=controller.summary() if controller else None,
                      permanent_bytes=sum(t.numel()*t.element_size() for t in list(block.parameters())+list(block.buffers()) if not t.is_meta),
                      iterations=[])
        if runtime.is_main:
            (output/"full-model-manifest.json").write_text(json.dumps(full_manifest,indent=2))
        first = None
        for i in range(12):
            runtime.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if args.profile and i == 2:
                torch.cuda.cudart().cudaProfilerStart()
            start = time.perf_counter()
            with nvtx_range(f"tp_probe.{args.mode}.{i}"), controller.forward() if controller else nullcontext():
                with controller.block(0) if controller else nullcontext():
                    actual = paper_block_forward(block,hidden,temb,indices,(cos,sin),tables=tables,fused=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            maximum = torch.tensor(seconds,dtype=torch.float64)
            dist.all_reduce(maximum,op=dist.ReduceOp.MAX,group=paper.group)
            if args.profile and i == 11:
                torch.cuda.cudart().cudaProfilerStop()
            local = actual[:,runtime.local_start:runtime.local_end] if args.mode == "tp" else actual
            cpu = local.cpu()
            if first is None:
                first = cpu.clone()
                delta = cpu.float()-capture["output"].float()
                record["historical_delta"] = dict(relative_l2=(delta.norm()/capture["output"].float().norm()).item(),maxabs=delta.abs().max().item())
                torch.save(cpu, output/f"output-rank-{runtime.rank}.pt")
            row = dict(iteration=i,warmup=i<2,seconds=seconds,max_rank_seconds=maximum.item(),
                       finite=bool(torch.isfinite(cpu).all()),byte_exact_repeat=torch.equal(cpu.view(torch.uint8),first.view(torch.uint8)),
                       allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved())
            record["iterations"].append(row)
            print(json.dumps(dict(rank=runtime.rank,mode=args.mode,**row)),flush=True)
            del actual,local,cpu
        record["median_max_rank_seconds"] = statistics.median(r["max_rank_seconds"] for r in record["iterations"][2:])
        record["paper"] = paper.inventory()
        (output/f"rank-{runtime.rank}.json").write_text(json.dumps(record,indent=2))
    finally:
        if controller is not None:
            controller.release()
        if paper is not None:
            paper.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
