"""Small startup scatter reproducer; parent must reserve eight idle GPUs."""
import argparse
from datetime import timedelta
import os
import time

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['scatter', 'broadcast', 'scatter-sync'], default='scatter')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=40))
    cpu = torch.arange(8 * 16 * 1024 * 1024, dtype=torch.float32).remainder_(17)
    result = torch.empty(16 * 1024 * 1024, dtype=torch.float32, device=device)
    start = time.perf_counter()
    for step in range(16):
        if args.mode == 'broadcast':
            for source in range(8):
                value = cpu[source * result.numel():(source + 1) * result.numel()].to(device) if rank == 0 else torch.empty_like(result)
                dist.broadcast(value, src=0)
                if rank == source:
                    result.copy_(value)
        else:
            sends = [cpu[r * result.numel():(r + 1) * result.numel()].to(device) for r in range(8)] if rank == 0 else None
            dist.scatter(result, scatter_list=sends, src=0)
            del sends
        if args.mode == 'scatter-sync':
            torch.cuda.synchronize()
        if rank == 0:
            print('submitted', step, flush=True)
    torch.cuda.synchronize()
    expected = cpu[rank * result.numel():(rank + 1) * result.numel()]
    assert torch.equal(result.cpu(), expected)
    print('rank', rank, 'seconds', time.perf_counter() - start, 'correct', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
