"""Explicit eight-rank tiny exact storage/collective/lifecycle coverage."""
import copy
from datetime import timedelta
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from src.inference.utils.tensor_parallel import PermanentTensorShards, shard_block, reduce_partial
from src.inference.utils.paper_runtime import CoordinatedFiniteGuard
from test_tensor_parallel import make_block


@pytest.mark.skipif(os.environ.get("VDN_TEST_TP_NCCL") != "1",reason="explicit eight-GPU test")
@pytest.mark.parametrize("packed", [False,True])
@torch.no_grad()
def test_exact_distribution_repeats_finite_vote_and_terminal_cleanup(packed):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl",timeout=timedelta(seconds=60))
    group = dist.new_group(backend="gloo",timeout=timedelta(seconds=60))
    device = torch.device("cuda",rank)
    guard = CoordinatedFiniteGuard(device,group=group,world_size=8)
    paper = SimpleNamespace(persistent=True,guard=guard,group=group,
        options={k:True for k in ("paper_mxfp8","paper_qk_rope","paper_window","paper_vda","paper_adaln","paper_fused_block")})
    runtime = SimpleNamespace(rank=rank,world_size=8,device=device,paper_runtime=paper)
    torch.manual_seed(94)
    m = nn.Module()
    m.root = nn.Linear(24,24)
    if packed:
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
        from src.models.hybrid_attention import HybridAttention
        from src.models.ops.mxfp8_linear import convert_linear_to_mxfp8
        def block():
            b = MiniMaxH3TransformerBlock(64,8,32,256,32,1e-6,1e-6)
            b.attn = HybridAttention(b.attn,64,delta_rule="vdn_solve",enable_text_state=True,short_conv=("k","v"))
            return b.to(torch.bfloat16)
    else:
        block = make_block
    m.transformer_blocks = nn.ModuleList([block() for _ in range(50)])
    m.register_buffer("tag",torch.arange(11))
    m.eval().requires_grad_(False)
    if packed:
        convert_linear_to_mxfp8(m,enabled=True)
    reference = copy.deepcopy(m)
    if rank:
        m.to("meta")
    owner = PermanentTensorShards(m,runtime,scatter_chunk_bytes=4096)
    try:
        owner.distribute()
        for original,local in zip(reference.transformer_blocks,m.transformer_blocks):
            shard_block(original,rank,8)
            a,b = original.state_dict(),local.state_dict()
            assert a.keys() == b.keys()
            for name,t in b.items():
                if name.startswith("adaln_proj."):
                    assert t.is_meta
                else:
                    assert torch.equal(t.cpu(),a[name]),name
                    assert t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()
        assert torch.equal(m.root.weight.cpu(),reference.root.weight)
        assert torch.equal(m.tag.cpu(),reference.tag)
        pointers = [t.data_ptr() for t in m.parameters() if not t.is_meta]
        for request in range(4):
            if rank == 5:
                torch.cuda._sleep(2_000_000)
            with owner.forward():
                for i in range(50):
                    with owner.block(i):
                        x = torch.full((3,24),rank+request+1.,device=device)
                        reduce_partial(x,runtime,"toy",guard)
                        assert torch.equal(x,torch.full_like(x,36+8*request))
            owner.finish_request()
            assert pointers == [t.data_ptr() for t in m.parameters() if not t.is_meta]
        # Every rank must fail BEFORE entering the NCCL payload collective.
        x = torch.ones(1,device=device)
        if rank == 3:
            x.fill_(float("nan"))
        with pytest.raises(FloatingPointError,match="all-rank"):
            with owner.forward():
                with owner.block(0):
                    reduce_partial(x,runtime,"injected_nonfinite",guard)
        assert owner._failed and guard.failed and owner.active is None
        with pytest.raises(RuntimeError,match="drain"):
            owner.finish_request()
    finally:
        owner.release()
        owner.release()
        assert all(t.is_meta for t in m.parameters())
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__,*sys.argv[1:]]))
