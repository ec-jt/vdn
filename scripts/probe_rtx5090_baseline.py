"""Fixed baseline only: PyTorch/NCCL payload measurements, not nccl-tests."""
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
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from src.inference.utils.ulysses_runtime import UlyssesRuntime
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 8
    assert os.environ.get('NCCL_SHM_USE_CUDA_MEMCPY') == '0'
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    records = []

    def measure(name, operation, payload_bytes, **metadata):
        samples = []
        for _ in range(5):
            dist.barrier(); torch.cuda.synchronize()
            start = time.perf_counter()
            operation()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        times = [None] * 8
        dist.all_gather_object(times, samples)
        maxima = [max(t[i] for t in times) for i in range(5)]
        median = statistics.median(maxima[2:])
        record = dict(name=name, rank_seconds=times, maxrank_seconds=maxima,
                      warmups=2, measured_repetitions=3, median_maxrank_seconds=median,
                      payload_bytes=payload_bytes, payload_decimal_GB_s=payload_bytes / median / 1e9,
                      payload_GiB_s=payload_bytes / median / 2**30, **metadata)
        records.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)

    try:
        n = 256 * 2**20 // 4
        send = torch.arange(n, dtype=torch.int32, device='cuda') + rank
        recv = torch.empty_like(send)
        cases = [('PIX_01', [(0, 1)]), ('NODE_02', [(0, 2)]),
                 ('PIX_four_pairs', [(0, 1), (2, 3), (4, 5), (6, 7)]),
                 ('NODE_shared_uplink_two', [(0, 2), (1, 3)]),
                 ('NODE_four_pairs', [(0, 2), (1, 3), (4, 6), (5, 7)])]
        # All ranks enter the first collective before subset P2P use.
        dist.all_reduce(torch.ones(1, device='cuda'))
        for name, pairs in cases:
            for bidirectional in [False, True]:
                def transfer():
                    ops = []
                    for a, b in pairs:
                        if rank == a:
                            ops.append(dist.P2POp(dist.isend, send, b))
                            if bidirectional:
                                ops.append(dist.P2POp(dist.irecv, recv, b))
                        elif rank == b:
                            ops.append(dist.P2POp(dist.irecv, recv, a))
                            if bidirectional:
                                ops.append(dist.P2POp(dist.isend, send, a))
                    for work in dist.batch_isend_irecv(ops) if ops else []:
                        work.wait()
                directions = 2 if bidirectional else 1
                measure(name + ('_bidir' if bidirectional else '_oneway'), transfer,
                        n * 4 * len(pairs) * directions, pairs=pairs,
                        bytes_per_direction_per_pair=n * 4, directions=directions,
                        metric='aggregate delivered payload, bidir sums both directions')
                for a, b in pairs:
                    if rank == b or (bidirectional and rank == a):
                        source = a if rank == b else b
                        assert torch.equal(recv, send + source - rank)
        del send, recv
        torch.cuda.empty_cache()
        total_bytes = 1_376_730_080
        n = total_bytes // 4 // 8
        shard = torch.arange(n, device='cuda', dtype=torch.int32) + rank * n
        gathered = torch.empty(n * 8, device='cuda', dtype=torch.int32)
        measure('block_allgather', lambda: dist.all_gather_into_tensor(gathered, shard),
                total_bytes * 7 / 8, output_bytes_per_rank=total_bytes,
                metric='logical remote received bytes per rank; not wire/bidirectional bandwidth')
        for start in range(0, n * 8, 2**20):
            stop = min(start + 2**20, n * 8)
            assert torch.equal(gathered[start:stop], torch.arange(start, stop, dtype=torch.int32, device='cuda'))
        del shard, gathered
        torch.cuda.empty_cache()
        rt = UlyssesRuntime(rank, 8, rank, torch.device('cuda', rank), 'nccl', standard_exchange='broadcast')
        rt.configure(38510, 56)
        offset = sum(rt.splits[:rank])

        def fill(tensor, row_start, head_start, width, digit):
            # Three base-2048 digit planes jointly identify EVERY global element.
            # View finite BF16 bit patterns, avoiding BF16 numeric rounding.
            for start in range(0, tensor.shape[0], 64):
                stop = min(start + 64, tensor.shape[0])
                rows = torch.arange(start + row_start, stop + row_start, device='cuda', dtype=torch.int64)
                heads = torch.arange(head_start, head_start + tensor.shape[1], device='cuda', dtype=torch.int64)
                cols = torch.arange(width, device='cuda', dtype=torch.int64)
                code = ((rows[:, None, None] * 56 + heads[None, :, None]) * width + cols[None, None, :])
                bits = ((code >> (11 * digit)) & 2047).to(torch.int16)
                tensor[start:stop].view(torch.int16).copy_(bits)

        for direction, width in [('input', 770), ('output', 256)]:
            shape = (rt.splits[rank], 56, width) if direction == 'input' else (38510, 7, width)
            x = torch.empty(shape, dtype=torch.bfloat16, device='cuda')
            call = rt.sequence_to_heads if direction == 'input' else rt.heads_to_sequence
            fill(x, offset if direction == 'input' else 0, 0 if direction == 'input' else rank * 7, width, 0)
            def route():
                y = call(x)
                del y
            total = 38510 * 56 * width * 2
            measure('attention_' + direction + '_broadcast', route, total * 7 / 64,
                    rows=38510, heads=56, width=width, splits=rt.splits,
                    global_tensor_bytes=total,
                    metric='mean logical useful remote receive per rank, not broadcast traffic or wire rate')
            for digit in range(3):
                fill(x, offset if direction == 'input' else 0, 0 if direction == 'input' else rank * 7, width, digit)
                y = call(x)
                expected = torch.empty_like(y)
                fill(expected, 0 if direction == 'input' else offset, rank * 7 if direction == 'input' else 0, width, digit)
                assert torch.equal(y.view(torch.int16), expected.view(torch.int16)), (rank, direction, digit)
                del y, expected
            records[-1]['full_coordinate_identity_verified_three_digit_planes'] = True
            del x
            torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            with (args.output_dir / 'bandwidth.json').open('x') as f:
                json.dump(dict(tool='PyTorch NCCL, NOT nccl-tests', torch=torch.__version__,
                               nccl=torch.cuda.nccl.version(), records=records,
                               all_values_verified=True), f, indent=2)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
