import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from src.inference.utils.tensor_parallel import shard_linear, shard_block, unswizzle_scales
from src.models.ops.mxfp8_linear import MXFP8Linear
from src.models.ops.mxfp8_quant import quantize_cpu, swizzle_scales


@pytest.mark.parametrize("k,start,end", [(7168, 896, 1792), (14336, 1792, 3584), (160, 32, 96)])
def test_mxfp8_k_slice_repacked_matches_requantization(k, start, end):
    torch.manual_seed(9)
    linear = nn.Linear(k, 129, bias=True, dtype=torch.bfloat16).eval()
    local = MXFP8Linear(linear)
    shard_linear(local, columns=(start, end), row_parallel=True)
    data, scales = quantize_cpu(linear.weight[:, start:end])
    assert torch.equal(local.weight_data, data)
    assert torch.equal(local.weight_scale, scales)
    assert torch.equal(local.tp_bias, linear.bias)
    assert local.bias is None
    assert local.weight_data.untyped_storage().nbytes() == local.weight_data.numel()


def test_swizzle_inverse_and_nonadjacent_rows():
    torch.manual_seed(1)
    x = torch.randint(0, 254, (256, 12), dtype=torch.uint8)
    assert torch.equal(unswizzle_scales(swizzle_scales(x), 256, 384), x)
    linear = nn.Linear(160, 512, dtype=torch.bfloat16).eval()
    rows = [*range(64, 128), *range(320, 384)]
    local = MXFP8Linear(linear)
    shard_linear(local, rows=rows)
    data, scales = quantize_cpu(linear.weight[rows])
    assert torch.equal(local.weight_data, data)
    assert torch.equal(local.weight_scale, scales)
    assert torch.equal(local.bias, linear.bias[rows])


def make_block():
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    from src.models.hybrid_attention import HybridAttention
    b = MiniMaxH3TransformerBlock(24, 8, 8, 32, 12, 1e-6, 1e-6)
    b.attn = HybridAttention(b.attn, 24, delta_rule="vdn_solve", radius=1,
                            anchor_frames="both", enable_text_state=True, short_conv=("k", "v"))
    # Exercise normally absent biases, too.
    for module in (b.attn.orig.to_out[0], b.attn.to_out_linear, b.ff.net[2]):
        module.bias = nn.Parameter(torch.randn(module.out_features))
    return b.eval().requires_grad_(False)


@pytest.mark.parametrize("extra_rows", [0, 1])
@pytest.mark.parametrize("anchors", ["none", "columns", "rows", "both"])
@torch.no_grad()
def test_fp32_full_block_partition_oracle(extra_rows, anchors):
    from src.models.sequence_layout import SequenceLayout
    from src.models.softmax_attention import window_softmax_reference
    from src.inference.utils.ulysses import _linear_branch_shard
    from src.models.ops.fused_block import _pre_ref, _post_ref, _swiglu_ref
    torch.manual_seed(71)
    block = make_block()
    block.attn.anchor_frames = anchors
    layout = SequenceLayout(seq_len=24+extra_rows, video_start=3, num_frames=5,
                            tokens_per_frame=4, frame_height=2, frame_width=2,
                            text_start=0, text_len=3)
    block.attn.layout = layout
    x = torch.randn(1, layout.seq_len, 24)
    tags = torch.arange(layout.seq_len) % 3
    tables = [torch.randn(3, 24)*.1 for _ in range(6)]
    a_shift, a_scale, a_gate, f_shift, f_scale, f_gate = tables
    normal = _pre_ref(x, block.norm1.weight, block.norm1.eps, a_scale, a_shift, tags)[0]
    def attention(b):
        a, branch = b.attn, b.attn.linear_attention
        heads, d = branch.num_heads, a.head_dim
        raw = tuple(p(normal).view(layout.seq_len, heads, d) for p in (a.orig.to_q,a.orig.to_k,a.orig.to_v))
        q, k = a.orig.norm_q(raw[0]), a.orig.norm_k(raw[1])
        soft = window_softmax_reference(q,k,raw[2],layout,a._bounds(layout),d**-.5,anchor_frames=a.anchor_frames)
        soft = soft * a.softmax_gate(normal)
        beta = torch.sigmoid(branch.beta_proj(normal))
        gate = branch.output_gate(normal)
        mean = normal[layout.video_start:layout.video_end].view(5,4,24).mean(1)
        far = _linear_branch_shard(a, raw, beta, gate, mean, 0, heads)
        assert torch.count_nonzero(far[:layout.video_start]) == 0
        assert torch.count_nonzero(far[layout.video_end:]) == 0
        if anchors == "both":
            assert torch.count_nonzero(far[layout.video_start:layout.video_start+4]) == 0
            assert torch.count_nonzero(far[layout.video_end-4:]) == 0
        out = a.orig.to_out[0](soft.reshape(layout.seq_len,-1))
        out[layout.video_start:layout.video_end] += a.to_out_linear(far[layout.video_start:layout.video_end].reshape(20,-1))
        return out
    expected_attention = attention(block)
    parts = [shard_block(copy.deepcopy(block), r, 8) for r in range(8)]
    actual_attention = sum(attention(b) for b in parts)
    actual_attention += block.attn.orig.to_out[0].bias
    actual_attention[layout.video_start:layout.video_end] += block.attn.to_out_linear.bias
    torch.testing.assert_close(actual_attention, expected_attention, rtol=2e-5, atol=2e-5)
    middle = _post_ref(x, a_gate, tags, expected_attention)
    hidden = _pre_ref(middle,block.norm2.weight,block.norm2.eps,f_scale,f_shift,tags)
    expected_ff = block.ff.net[2](_swiglu_ref(block.ff.net[0].proj(hidden)))
    actual_ff = sum(b.ff.net[2](_swiglu_ref(b.ff.net[0].proj(hidden))) for b in parts) + block.ff.net[2].bias
    torch.testing.assert_close(_post_ref(middle,f_gate,tags,actual_ff),
                               _post_ref(middle,f_gate,tags,expected_ff),rtol=2e-5,atol=2e-5)
    # A prompt omission changes recurrent video output; it is not a dead test.
    without_text = copy.deepcopy(block)
    without_text.attn.enable_text_state = False
    assert not torch.allclose(attention(without_text), expected_attention)
    for r,b in enumerate(parts):
        torch.testing.assert_close(b.attn.linear_attention.alpha.A_log, block.attn.linear_attention.alpha.A_log[r:r+1])
        assert b.attn.linear_attention.short_conv.k_sp.groups == 8


def test_invalid_slices_and_repeated_sharding_fail():
    b = make_block()
    shard_block(b, 0, 8)
    with pytest.raises(ValueError, match="already"):
        shard_block(b, 0, 8)
    with pytest.raises(ValueError, match="block32"):
        shard_linear(MXFP8Linear(nn.Linear(128,128).eval()),columns=(1,33))


def test_tp_meta_shapes_match_cpu_and_reject_aliases():
    original = make_block()
    meta = copy.deepcopy(original).to("meta")
    for rank in range(8):
        a, b = shard_block(copy.deepcopy(original),rank,8), shard_block(copy.deepcopy(meta),rank,8)
        assert [(n,t.shape,t.dtype) for n,t in a.state_dict().items()] == [(n,t.shape,t.dtype) for n,t in b.state_dict().items()]
        for name,t in a.state_dict().items():
            assert t.storage_offset() == 0
            assert t.untyped_storage().nbytes() == t.numel()*t.element_size()
    original.norm2.weight = original.norm1.weight
    with pytest.raises(ValueError,match="aliases"):
        shard_block(original,0,8)


def test_tp_lifecycle_poison_release_and_no_prefetch(monkeypatch):
    from types import SimpleNamespace
    from src.inference.utils.tensor_parallel import PermanentTensorShards
    m = nn.Module()
    m.transformer_blocks = nn.ModuleList([make_block()])
    c = PermanentTensorShards(m, SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(torch.cuda,"synchronize",lambda *a: None)
    with pytest.raises(RuntimeError,match="lifecycle"):
        with c.forward():
            pass
    c.initialized = True
    for _ in range(3):
        with c.forward():
            with c.block(0):
                with pytest.raises(RuntimeError,match="active"):
                    c.release()
        c.finish_request()
    with pytest.raises(RuntimeError,match="prefetch"):
        c.prefetch_next()
    with pytest.raises(ValueError,match="injected"):
        with c.forward():
            with c.block(0):
                raise ValueError("injected")
    assert c._failed and not c._in_forward and c.active is None
    with pytest.raises(RuntimeError,match="drain"):
        c.finish_request()
    c.release()
    c.release()
    assert all(t.is_meta for t in m.parameters())
