"""Actual controller startup probe with production-size mixed-dtype block."""
import os
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist
from src.inference.utils.weight_sharding import ResidentWeightShards


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=40))
    model = torch.nn.Module()
    with torch.device('cpu' if rank == 0 else 'meta'):
        model.register_parameter('root', torch.nn.Parameter(torch.full((200_000_000,), 2., dtype=torch.bfloat16)))
        model.transformer_blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        for block in model.transformer_blocks:
            block.register_parameter('large', torch.nn.Parameter(torch.full((680_000_000,), 3., dtype=torch.bfloat16)))
            block.register_parameter('small', torch.nn.Parameter(torch.full((1000,), 4., dtype=torch.float32)))
    model.eval().requires_grad_(False)
    runtime = SimpleNamespace(device=device, rank=rank, world_size=8, profile_start=lambda: None, profile_end=lambda *args: None)
    controller = ResidentWeightShards(model, runtime, distribution='broadcast')
    started = time.perf_counter()
    controller.distribute()
    torch.cuda.synchronize()
    print('rank', rank, 'distribution_seconds', time.perf_counter()-started, flush=True)
    assert (model.root == 2).all().item()
    for index, block in enumerate(model.transformer_blocks):
        with controller.block(index):
            assert (block.large == 3).all().item() and (block.small == 4).all().item()
    controller.release()
    print('rank', rank, 'correct', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
