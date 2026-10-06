"""Explicit two-rank resident lifetime stress, no model checkpoint."""
import os
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from types import SimpleNamespace

from src.inference.utils.weight_sharding import ResidentWeightShards
from src.models.ops.mxfp8_linear import MXFP8Linear


@pytest.mark.skipif(os.environ.get("VDN_TEST_PERSISTENT_NCCL") != "1", reason="explicit two-GPU test")
@torch.no_grad()
def test_repeated_requests_drain_pending_failure_keep_shards_and_group():
    assert int(os.environ.get("WORLD_SIZE", "0")) == 2
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    runtime = SimpleNamespace(device=torch.device(f"cuda:{rank}"), rank=rank, world_size=2,
        backend="nccl", is_main=rank == 0, profile_start=lambda: None, profile_end=lambda *a: None)
    def model(device):
        torch.manual_seed(1701)
        m = torch.nn.Module()
        with torch.device(device):
            m.root = torch.nn.Linear(37, 37, dtype=torch.bfloat16)
            m.transformer_blocks = torch.nn.ModuleList([
                MXFP8Linear(torch.nn.Linear(37, 37, dtype=torch.bfloat16)) for _ in range(3)])
        return m.eval().requires_grad_(False)
    controller = None
    try:
        reference = model("cpu").to(runtime.device)
        candidate = model("cpu" if rank == 0 else "meta")
        controller = ResidentWeightShards(candidate, runtime, distribution="broadcast", prefetch=True)
        controller.distribute()
        group = controller.weight_group
        pointers = [g.shard.data_ptr() for groups in controller.groups for g in groups]
        for request in range(8):
            # Different seed/prompt surrogates, exact same immutable model.
            torch.manual_seed(42 + request)
            x = torch.randn(129, 37, device=runtime.device, dtype=torch.bfloat16)
            if rank:
                torch.cuda._sleep(2_000_000)
            if request == 3:
                with pytest.raises(FloatingPointError, match="injected"), controller.forward():
                    with controller.block(0):
                        controller.prefetch_next()
                        candidate.transformer_blocks[0](x)
                        raise FloatingPointError("injected request finite failure")
                controller.finish_request()
            with controller.forward():
                for i, block in enumerate(candidate.transformer_blocks):
                    with controller.block(i):
                        controller.prefetch_next()
                        assert torch.equal(block(x), reference.transformer_blocks[i](x))
            controller.finish_request()
            assert controller.pending is controller.active is controller._active_gather is None
            assert controller.weight_group is group and not controller.released
            assert pointers == [g.shard.data_ptr() for groups in controller.groups for g in groups]
            churn = torch.empty(8 << 20, device=runtime.device, dtype=torch.uint8)
            churn.fill_(request)
            del churn, x
            dist.barrier()
        assert controller.max_live_gathered == 2
    finally:
        if controller is not None:
            controller.release()
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
