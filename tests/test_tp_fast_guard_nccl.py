"""Bounded two-rank before-payload failure tests, including mismatched phases."""
from datetime import timedelta
import os
import sys
import pytest
import torch
import torch.distributed as dist
from src.inference.utils.paper_runtime import CoordinatedFiniteGuard


@pytest.mark.skipif(os.getenv('VDN_TEST_TP_GUARD') != '1',reason='explicit NCCL opt-in')
def test_guard_before_payload():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=30))
    group = dist.new_group(backend='gloo',timeout=timedelta(seconds=30))
    device = torch.device('cuda',rank)
    try:
        for failure in ('nan','cpu','nested'):
            guard = CoordinatedFiniteGuard(device,group=group,world_size=2,fast=True)
            payload = torch.ones(129,5376,device=device,dtype=torch.bfloat16)
            entered_payload = False
            with pytest.raises(FloatingPointError,match='all-rank'):
                with guard.local_phase('projection'):
                    if rank == 1:
                        if failure == 'nan':
                            payload[-1,-1] = float('nan')
                        else:
                            raise ValueError('injected CPU projection error')
                    guard.observe(payload)
                    if failure == 'nested':
                        guard.check('inner')
                entered_payload = True
                dist.all_reduce(payload)
            assert guard.failed and not entered_payload
            with pytest.raises(RuntimeError,match='terminal'):
                guard.check('retry')
        # Clean group after rejected payloads proves all ranks took the vote.
        x = torch.tensor(float(rank+1),device=device)
        dist.all_reduce(x)
        assert x.item() == 3
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__,*sys.argv[1:]]))
