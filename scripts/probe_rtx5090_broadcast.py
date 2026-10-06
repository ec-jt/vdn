"""Bounded, synthetic standard routing probe. No model weights are loaded."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_rtx5090_transport import production_pids, exclusive_json, transport_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exchange", choices=("broadcast", "alltoall"), default="broadcast")
    args = parser.parse_args()
    if production_pids():
        raise RuntimeError("Production is active")
    transport_configs(os.environ)
    import torch
    import torch.distributed as dist
    from src.inference.utils.ulysses_runtime import UlyssesRuntime
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 8:
        raise ValueError("Exactly eight local ranks required")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=150))
    try:
        torch.set_grad_enabled(False)
        rt = UlyssesRuntime(rank, 8, rank, torch.device("cuda", rank), "nccl",
                            standard_exchange=args.exchange)
        rt.configure(38003, 56)
        x = torch.empty((rt.splits[rank], 56, 770), dtype=torch.bfloat16, device=rt.device)
        samples = []
        forward, backward = [], []
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(4):
            # Exactly representable BF16 codes distinguish every source/head.
            codes = ((torch.arange(56, device=rt.device) + rank * 57 + iteration * 11) % 127).to(x.dtype)
            x.copy_(codes[None, :, None])
            dist.barrier(); torch.cuda.synchronize()
            start = time.perf_counter()
            heads = rt.sequence_to_heads(x)
            torch.cuda.synchronize()
            middle = time.perf_counter()
            back = rt.heads_to_sequence(heads)
            torch.cuda.synchronize()
            end = time.perf_counter()
            samples.append(end - start)
            forward.append(middle - start); backward.append(end - middle)
            offset = 0
            for source, rows in enumerate(rt.splits):
                expected = ((torch.arange(rank * 7, (rank + 1) * 7, device=rt.device)
                             + source * 57 + iteration * 11) % 127).to(x.dtype)
                for row in range(offset, offset + rows, 128):
                    assert bool((heads[row:min(row + 128, offset + rows)] == expected[None, :, None]).all().item())
                offset += rows
            assert torch.equal(back, x)
            del heads, back
        record = dict(rank=rank, exchange=args.exchange, rows=38003, heads=56, width=770,
                      sequence_splits=rt.splits, heads_per_rank=rt.heads_per_rank,
                      roundtrip_seconds=samples, forward_seconds=forward, backward_seconds=backward,
                      complete_values_exact=True, repetitions=4,
                      input_bytes=x.numel() * x.element_size(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      nccl_p2p_level=os.environ.get("NCCL_P2P_LEVEL"))
        exclusive_json(args.output_dir / f"rank-{rank}.json", record)
        records = [None] * 8
        dist.all_gather_object(records, record)
        if rank == 0:
            maxima = [max(r["roundtrip_seconds"][i] for r in records) for i in range(4)]
            result = dict(ranks=records, max_rank_per_sample=maxima,
                          warm_mean_max_rank=statistics.mean(maxima[1:]),
                          peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in records),
                          peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in records))
            exclusive_json(args.output_dir / "summary.json", result)
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
