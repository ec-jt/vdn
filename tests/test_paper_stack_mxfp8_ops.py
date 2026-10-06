"""CPU contracts and explicitly gated SM120 shape/quantization/GEMM oracles."""
import os
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.models.ops.mxfp8_quant import quantize_cpu, quantize_activation, padded_shape, swizzle_scales
from src.models.ops.mxfp8_linear import (MXFP8Linear, SUPPORTED_ROLES, convert_linear_to_mxfp8,
                                        project_qkv, project_to_out, projection_features)


def unswizzle(s, r, k):
    """Independent address oracle rather than inverse reshape implementation."""
    rr = torch.arange(r, device=s.device)[:, None]
    cc = torch.arange(k // 32, device=s.device)[None, :]
    offsets = ((rr // 128) * (k // 128) + cc // 4) * 512 + rr % 32 * 16 + (rr % 128 // 32) * 4 + cc % 4
    return s[offsets]


def dequant(data, scale):
    r, k = data.shape
    natural = unswizzle(scale, r, k)
    return (data.view(torch.float8_e4m3fn).float().reshape(r, k // 32, 32)
            * torch.exp2(natural.float() - 127)[..., None]).reshape(r, k)


def toy(device="cpu"):
    model = nn.Module()
    with torch.device(device):
        model.endpoint = nn.Linear(37, 21, dtype=torch.bfloat16)
        model.transformer_blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            for path in SUPPORTED_ROLES.values():
                parent = block
                pieces = path.split(".")
                for piece in pieces[:-1]:
                    if piece not in parent._modules:
                        parent.add_module(piece, nn.Module())
                    parent = parent._modules[piece]
                parent.add_module(pieces[-1], nn.Linear(37, 21, dtype=torch.bfloat16))
            block.norm = nn.LayerNorm(37, dtype=torch.bfloat16)
            model.transformer_blocks.append(block)
    return model.eval().requires_grad_(False)


@pytest.mark.parametrize("shape", [(0, 1), (1, 1), (3, 31), (127, 32), (129, 129), (9, 256)])
def test_cpu_padding_scale_oracle_and_chunk_bound(shape):
    torch.manual_seed(13)
    x = torch.randn(shape, dtype=torch.bfloat16)
    before = x.clone()
    data, scales = quantize_cpu(x, chunk_rows=7)
    rp, kp = padded_shape(*shape)
    assert data.shape == (rp, kp) and data.dtype == scales.dtype == torch.uint8
    assert torch.equal(x, before)
    assert torch.equal(data, quantize_cpu(x, chunk_rows=1)[0])
    assert torch.equal(scales, quantize_cpu(x, chunk_rows=1)[1])
    restored = dequant(data, scales)
    assert torch.count_nonzero(restored[shape[0]:]) == 0
    assert torch.count_nonzero(restored[:, shape[1]:]) == 0
    if x.numel():
        torch.testing.assert_close(restored[:shape[0], :shape[1]], x.float(), rtol=.063, atol=.001)


def test_exact_scale_boundaries_zero_and_subnormal():
    # The author exponent rule bumps only above 448, not at 448.
    values = torch.tensor([0., 224., 448., 449., 896., 1e-38], dtype=torch.float32)
    x = values[:, None].expand(-1, 32)
    q, s = quantize_cpu(x)
    natural = unswizzle(s, *q.shape)
    assert natural[:5, 0].tolist() == [0, 126, 127, 128, 128]
    assert torch.isfinite(dequant(q, s)).all()
    for i in range(5):
        exponent = natural[i, 0].float() - 127
        expected = (x[i] * torch.exp2(-exponent)).to(torch.float8_e4m3fn).view(torch.uint8)
        assert torch.equal(q[i, :32], expected)


def test_swizzle_unique_address_oracle():
    natural = (torch.arange(256 * 12) % 251).to(torch.uint8).reshape(256, 12)
    assert torch.equal(unswizzle(swizzle_scales(natural), 256, 384), natural)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_cpu(value):
    with pytest.raises(ValueError, match="nonfinite"):
        quantize_cpu(torch.tensor([[value]]))


def test_default_off_meta_symmetry_and_root_precision():
    cpu, meta = toy(), toy("meta")
    root, norm = cpu.endpoint.weight, cpu.transformer_blocks[0].norm.weight
    before = root.clone()
    assert convert_linear_to_mxfp8(cpu) == []
    with patch("torch.cuda.init", side_effect=AssertionError("CPU conversion touched CUDA")):
        records = convert_linear_to_mxfp8(cpu, enabled=True, chunk_rows=3)
        assert records == convert_linear_to_mxfp8(meta, enabled=True, chunk_rows=3)
    assert len(records) == 14
    assert root is cpu.endpoint.weight and norm is cpu.transformer_blocks[0].norm.weight
    assert torch.equal(root, before)
    left, right = cpu.state_dict(), meta.state_dict()
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].shape == right[key].shape and left[key].dtype == right[key].dtype
    packed = cpu.transformer_blocks[0].attn.orig.to_q
    assert not hasattr(packed, "weight")
    assert projection_features(packed) == (37, 21)
    assert set(dict(packed.named_buffers())) == {"weight_data", "weight_scale"}
    with pytest.raises(ValueError, match="CUDA BF16"):
        packed(torch.zeros(2, 37, dtype=torch.bfloat16))


def test_reject_partial_conversion_and_ties():
    model = toy()
    model.transformer_blocks[1].attn.orig.to_k = nn.Identity()
    with pytest.raises(ValueError, match="merged"):
        convert_linear_to_mxfp8(model, enabled=True)
    assert isinstance(model.transformer_blocks[0].attn.orig.to_q, nn.Linear)
    model = toy()
    model.transformer_blocks[0].attn.orig.to_k.weight = model.transformer_blocks[0].attn.orig.to_q.weight
    with pytest.raises(ValueError, match="tied"):
        convert_linear_to_mxfp8(model, enabled=True)


def test_projection_helpers_preserve_normal_path():
    linear = nn.Linear(17, 9).eval()
    x = torch.randn(3, 17)
    torch.testing.assert_close(project_to_out(nn.ModuleList([linear, nn.Identity()]), x), linear(x))
    for y in project_qkv((linear, linear, linear), x):
        torch.testing.assert_close(y, linear(x))


def test_compressed_state_roundtrip_and_cross_root_tie(tmp_path):
    p = MXFP8Linear(nn.Linear(37, 21, dtype=torch.bfloat16))
    torch.save(p.state_dict(), tmp_path / "compressed.pt")
    other = MXFP8Linear(nn.Linear(37, 21, dtype=torch.bfloat16))
    other.load_state_dict(torch.load(tmp_path / "compressed.pt", weights_only=True))
    for name, t in p.state_dict().items():
        assert torch.equal(t, other.state_dict()[name])
    model = toy()
    model.endpoint.weight = model.transformer_blocks[0].attn.orig.to_q.weight
    with pytest.raises(ValueError, match="tied"):
        convert_linear_to_mxfp8(model, enabled=True)


def test_unsupported_device_has_no_fallback():
    from src.models.ops.mxfp8_linear import require_native_sm120
    with pytest.raises(RuntimeError, match="no fallback"):
        require_native_sm120("cpu")
    with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
        with pytest.raises(RuntimeError, match="no fallback"):
            require_native_sm120("cuda:0")


gpu = pytest.mark.skipif(os.getenv("VDN_TEST_MXFP8_GPU") != "1", reason="explicit SM120 GPU opt-in")


@gpu
@pytest.mark.parametrize("shape", [(0, 7, 3), (1, 1, 1), (17, 37, 31), (129, 133, 257), (128, 128, 256)])
@torch.no_grad()
def test_fused_quant_and_native_gemm_shape_oracle(shape):
    m, n, k = shape
    torch.manual_seed(42)
    # Non-contiguous rows exercise the fused kernel's explicit input stride.
    base = torch.randn(m * 2, k, dtype=torch.bfloat16)
    x = base[::2]
    linear = nn.Linear(k, n, bias=True, dtype=torch.bfloat16).eval()
    packed = MXFP8Linear(linear).cuda()
    xg = base.cuda()[::2]
    q, s = quantize_activation(xg)
    cq, cs = quantize_cpu(x)
    assert torch.equal(q.cpu(), cq) and torch.equal(s.cpu(), cs)
    y = packed(xg)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16 and y.isfinite().all()
    if m:
        oracle = (dequant(q, s) @ dequant(packed.weight_data, packed.weight_scale).T)[:m, :n]
        oracle = oracle.to(torch.bfloat16) + packed.bias
        torch.testing.assert_close(y, oracle, rtol=.01, atol=.016)
    assert torch.equal(packed.weight_data.cpu(), quantize_cpu(linear.weight)[0])


@gpu
@torch.no_grad()
def test_fused_matches_pinned_author_and_shared_qkv():
    import importlib.util
    from pathlib import Path
    # Load only the standalone author kernel module with its unused common helper stub.
    import sys
    import types
    common = types.ModuleType("sglang.kernels.ops.diffusion.common.numerics")
    common.round_bf16_to_fp32 = None
    path = Path("artifacts/rtx5090/paper-stack-20261005/evidence/sglang-merge/python/sglang/kernels/ops/diffusion/quantization/mxfp8_swizzled_triton.py")
    spec = importlib.util.spec_from_file_location("pinned_author_mxfp8", path)
    author = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {common.__name__: common}):
        spec.loader.exec_module(author)
    x = torch.randn(129, 256, dtype=torch.bfloat16, device="cuda")
    q, s = quantize_activation(x)
    aq, ass = author.mxfp8_quantize_swizzled(x)
    assert torch.equal(q[:129], aq.view(torch.uint8))
    assert torch.equal(s, ass)
    projections = [MXFP8Linear(nn.Linear(256, 17, bias=False, dtype=torch.bfloat16)).cuda() for _ in range(3)]
    for actual, p in zip(project_qkv(projections, x), projections):
        assert torch.equal(actual, p(x))


@gpu
@torch.no_grad()
def test_reject_nonfinite_gpu_inputs_and_results():
    p = MXFP8Linear(nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)).cuda()
    x = torch.ones(3, 32, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite MXFP8 activation"):
        p(x)
    x.fill_(1)
    p.weight_data.fill_(127)  # E4M3 NaN, simulates corrupted compressed input.
    with pytest.raises(ValueError, match="nonfinite MXFP8 result"):
        p(x)
