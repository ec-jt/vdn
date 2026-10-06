"""Bounded exact attention-route probe, PyTorch/NCCL (not nccl-tests)."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--chunks', action='store_true', help='Probe-only bounded chunk alternatives')
    parser.add_argument('--stress-only', action='store_true', help='Small NCCL strided/dtype/lifetime checks')
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from src.inference.utils.ulysses_runtime import UlyssesRuntime
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 8
    assert os.environ.get('NCCL_P2P_LEVEL') == 'SYS'
    assert os.environ.get('NCCL_SHM_USE_CUDA_MEMCPY') == '0'
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    dist.init_process_group('nccl', timeout=timedelta(seconds=90))
    records = []

    def fill(tensor, row_start, head_start, width, digit):
        # Three base-2048 finite BF16 bit-pattern planes jointly identify every
        # global coordinate, including channels and nondivisible boundaries.
        for start in range(0, tensor.shape[0], 64):
            stop = min(start + 64, tensor.shape[0])
            rows = torch.arange(start + row_start, stop + row_start, device='cuda', dtype=torch.int64)
            heads = torch.arange(head_start, head_start + tensor.shape[1], device='cuda', dtype=torch.int64)
            cols = torch.arange(width, device='cuda', dtype=torch.int64)
            code = (rows[:, None, None] * 56 + heads[None, :, None]) * width + cols[None, None, :]
            tensor[start:stop].view(torch.int16).copy_(((code >> (11 * digit)) & 2047).to(torch.int16))

    try:
        dist.all_reduce(torch.ones(1, device='cuda'))
        if args.stress_only:
            for rows in (10, 67):
                rt = UlyssesRuntime(rank, 8, rank, torch.device('cuda', rank), 'nccl', standard_exchange='allgather')
                rt.configure(rows, 16)
                prior = []
                for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64):
                    for iteration in range(3):
                        full = (torch.arange(rows * 16 * 6, device='cuda').reshape(rows, 16, 6) + iteration * 7).to(dtype)
                        for method, source, expected in (
                            (rt.sequence_to_heads, full[rt.local_start:rt.local_end], full[:, rank * 2:(rank + 1) * 2]),
                            (rt.heads_to_sequence, full[:, rank * 2:(rank + 1) * 2], full[rt.local_start:rt.local_end]),
                        ):
                            storage = torch.empty((*source.shape[:-1], source.shape[-1] * 2), dtype=dtype, device='cuda')
                            storage[..., ::2].copy_(source)
                            before = storage.clone()
                            actual = method(storage[..., ::2])
                            # Leave outputs alive across calls and churn freed staging sizes.
                            prior.append((actual, expected.clone()))
                            assert torch.equal(storage[..., ::2], before[..., ::2])
                            del storage, before
                            churn = torch.empty((8, rows, 16, 6), dtype=dtype, device='cuda')
                            churn.fill_(13)
                            del churn
                        for actual, expected in prior:
                            assert torch.equal(actual, expected)
                # Raw floating bit patterns include signed zero, NaNs, infinities,
                # subnormals and finite values. Compare bytes, not float equality.
                bits = torch.tensor([0, -32768, 1, 127, 32640, -128, 32704, 16256], dtype=torch.int16, device='cuda')
                full = bits.repeat(rows * 16).reshape(rows, 16, 8).view(torch.bfloat16)
                for method, source, expected in (
                    (rt.sequence_to_heads, full[rt.local_start:rt.local_end], full[:, rank * 2:(rank + 1) * 2]),
                    (rt.heads_to_sequence, full[:, rank * 2:(rank + 1) * 2], full[rt.local_start:rt.local_end]),
                ):
                    assert torch.equal(method(source).view(torch.int16), expected.view(torch.int16))
            dist.barrier()
            if rank == 0:
                with (args.output_dir / 'nccl-stress.json').open('x') as f:
                    json.dump(dict(ranks=8, rows=[10, 67], dtypes=['float32', 'float16', 'bfloat16', 'int64'],
                                   strided=True, repetitions=3, retained_outputs=True,
                                   allocator_churn=True, special_bits_exact=True, passed=True), f, indent=2)
            return
        routes = ('chunk1024', 'chunk2048') if args.chunks else ('broadcast', 'allgather')
        for route in routes:
            rt = UlyssesRuntime(rank, 8, rank, torch.device('cuda', rank), 'nccl',
                                standard_exchange='allgather' if args.chunks else route)
            rt.configure(38510, 56)
            for direction, width in [('input', 770), ('output', 256)]:
                shape = (rt.splits[rank], 56, width) if direction == 'input' else (38510, 7, width)
                x = torch.empty(shape, dtype=torch.bfloat16, device='cuda')
                call = rt.sequence_to_heads if direction == 'input' else rt.heads_to_sequence
                if args.chunks:
                    def call(tensor):
                        # Experimental only: same bytes with bounded row chunks.
                        chunk = int(route.removeprefix('chunk'))
                        input_route = direction == 'input'
                        limit = max(rt.splits) if input_route else rt.sequence_length
                        out = tensor.new_empty((38510, 7, width) if input_route else (rt.splits[rank], 56, width))
                        for lo in range(0, limit, chunk):
                            count = min(chunk, limit - lo)
                            send = tensor.new_zeros((count, tensor.shape[1], width))
                            valid = max(0, min(count, tensor.shape[0] - lo))
                            send[:valid].copy_(tensor[lo:lo + valid])
                            gathered = tensor.new_empty((8 * count, tensor.shape[1], width))
                            dist.all_gather_into_tensor(gathered, send)
                            for peer in range(8):
                                if input_route:
                                    valid = max(0, min(count, rt.splits[peer] - lo))
                                    offset = sum(rt.splits[:peer]) + lo
                                    out[offset:offset + valid].copy_(gathered[peer * count:peer * count + valid, rank * 7:(rank + 1) * 7])
                                else:
                                    a, b = max(lo, rt.local_start), min(lo + count, rt.local_end)
                                    if b > a:
                                        out[a - rt.local_start:b - rt.local_start, peer * 7:(peer + 1) * 7].copy_(gathered[peer * count + a - lo:peer * count + b - lo])
                        return out
                fill(x, rt.local_start if direction == 'input' else 0,
                     0 if direction == 'input' else rank * 7, width, 0)
                samples = []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(7):
                    dist.barrier(); torch.cuda.synchronize()
                    start = time.perf_counter()
                    y = call(x)
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - start)
                    del y
                local = dict(seconds=samples, allocated=torch.cuda.max_memory_allocated(),
                             reserved=torch.cuda.max_memory_reserved())
                ranks = [None] * 8
                dist.all_gather_object(ranks, local)
                maxima = [max(r['seconds'][i] for r in ranks) for i in range(7)]
                record = dict(route=route, direction=direction, width=width, rows=38510,
                              heads=56, splits=rt.splits, ranks=ranks,
                              warmups=2, repetitions=5, maxrank_seconds=maxima,
                              median_maxrank_seconds=statistics.median(maxima[2:]))
                # Validate every value after timing, with independent direction oracles.
                for digit in range(3):
                    fill(x, rt.local_start if direction == 'input' else 0,
                         0 if direction == 'input' else rank * 7, width, digit)
                    y = call(x)
                    expected = torch.empty_like(y)
                    fill(expected, 0 if direction == 'input' else rt.local_start,
                         rank * 7 if direction == 'input' else 0, width, digit)
                    assert torch.equal(y.view(torch.int16), expected.view(torch.int16)), (rank, route, direction, digit)
                    del y, expected
                record['full_coordinate_identity_three_planes'] = True
                records.append(record)
                if rank == 0:
                    print(json.dumps(record), flush=True)
                del x
                torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            with (args.output_dir / 'routes.json').open('x') as f:
                json.dump(dict(tool='PyTorch NCCL, NOT nccl-tests', torch=torch.__version__,
                               nccl=torch.cuda.nccl.version(), records=records), f, indent=2)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
