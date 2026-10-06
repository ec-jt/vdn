# SPDX-License-Identifier: Apache-2.0
"""Opt-in SM120 native MXFP8 linears. No fallback, dequantized weight or shadow.

The scaled GEMM recipe follows SGLang's mxfp8_online.py at merge
ff1ce11348673eea9b51165bd6b2da7238c958b4 (Apache-2.0). Unlike upstream,
unsupported execution raises instead of falling back to per-channel FP8.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .mxfp8_quant import padded_shape, quantize_activation, quantize_cpu, require_finite, scale_numel

SUPPORTED_ROLES = {
    "q": "attn.orig.to_q",
    "k": "attn.orig.to_k",
    "v": "attn.orig.to_v",
    "softmax_out": "attn.orig.to_out.0",
    "linear_out": "attn.to_out_linear",
    "ff_up": "ff.net.0.proj",
    "ff_down": "ff.net.2",
}


def require_native_sm120(device):
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("native MXFP8 requires validated SM120 CUDA execution; no fallback")
    if not all(hasattr(F, name) for name in ("scaled_mm", "ScalingType", "SwizzleType")):
        raise RuntimeError("installed PyTorch lacks native block-scaled functional GEMM")


class MXFP8Linear(nn.Module):
    """Inference-only effective weight, stored as two rebinding-safe uint8 buffers.

    There intentionally is no .weight property. Direct parameter consumers must
    use project(), projection_features(), or forward_quantized(), never a fake
    dequantized view. Construct after LoRA folding and before weight distribution.
    """
    def __init__(self, linear: nn.Linear, *, chunk_rows=128, check_finite=True):
        super().__init__()
        if type(check_finite) is not bool:
            raise ValueError("check_finite must be a bool")
        if type(linear) is not nn.Linear:
            raise TypeError("MXFP8 requires an ordinary merged nn.Linear")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.check_finite = check_finite
        data, scales = quantize_cpu(linear.weight, chunk_rows=chunk_rows)
        if linear.bias is not None and not linear.bias.is_meta:
            require_finite(linear.bias, "bias")
        self.register_buffer("weight_data", data)
        self.register_buffer("weight_scale", scales)
        self.register_parameter("bias", None if linear.bias is None else
                                nn.Parameter(linear.bias.detach(), requires_grad=False))
        self.eval()

    @torch.no_grad()
    def forward_quantized(self, data, scales, *, rows=None, out_dtype=torch.bfloat16):
        """Consume uint8 padded activation and swizzled scales, return cropped BF16."""
        if self.training or out_dtype != torch.bfloat16:
            raise ValueError("MXFP8 supports eval BF16 output only")
        require_native_sm120(data.device)
        tensors = (data, scales, self.weight_data, self.weight_scale)
        if any(t.dtype != torch.uint8 or not t.is_contiguous() or t.device != data.device for t in tensors):
            raise ValueError("MXFP8 data/scales must be contiguous uint8 on the same CUDA device")
        np, kp = padded_shape(self.out_features, self.in_features)
        if data.ndim != 2 or data.shape[1] != kp or data.shape[0] % 128:
            raise ValueError("invalid padded activation shape")
        if self.weight_data.shape != (np, kp) or self.weight_scale.shape != (scale_numel(np, kp),):
            raise ValueError("invalid compressed weight layout")
        if scales.shape != (scale_numel(data.shape[0], kp),):
            raise ValueError("invalid activation scale layout")
        if self.check_finite and (bool(((data & 127) == 127).any()) or bool((scales == 255).any())):
            raise ValueError("nonfinite MXFP8 prepared activation")
        rows = data.shape[0] if rows is None else rows
        if not 0 <= rows <= data.shape[0]:
            raise ValueError("invalid output row count")
        if rows == 0:
            return torch.empty((0, self.out_features), device=data.device, dtype=out_dtype)
        y = F.scaled_mm(data.view(torch.float8_e4m3fn), self.weight_data.view(torch.float8_e4m3fn).T,
                        scales.view(torch.float8_e8m0fnu), F.ScalingType.BlockWise1x32,
                        self.weight_scale.view(torch.float8_e8m0fnu), F.ScalingType.BlockWise1x32,
                        F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.SWIZZLE_32_4_4,
                        output_dtype=out_dtype)
        y = y[:rows, :self.out_features].contiguous()
        if self.bias is not None:
            y = y + self.bias.to(out_dtype)
        if self.check_finite:
            require_finite(y, "result")
        return y

    @torch.no_grad()
    def forward(self, x):
        if x.ndim < 2 or x.shape[-1] != self.in_features or x.requires_grad:
            raise ValueError("MXFP8 expects inference activation [...,in_features]")
        rows = math.prod(x.shape[:-1])
        data, scales = quantize_activation(x.reshape(rows, self.in_features).contiguous(),
                                           check_finite=self.check_finite)
        return self.forward_quantized(data, scales, rows=rows).reshape(*x.shape[:-1], self.out_features)


def projection_features(module):
    """Metadata replacement for shape reads through .weight in projection helpers."""
    if not isinstance(module, (nn.Linear, MXFP8Linear)):
        raise TypeError("unsupported projection module")
    return module.in_features, module.out_features


def project(module, x):
    """Explicit dispatch for helpers that formerly used functional.linear(weight)."""
    projection_features(module)
    return module(x)


def project_to_out(to_out, x):
    """Apply projection plus dropout without assuming projection has a parameter."""
    if isinstance(to_out, (nn.ModuleList, nn.Sequential, list, tuple)):
        if len(to_out) != 2:
            raise ValueError("expected projection/dropout pair")
        return to_out[1](project(to_out[0], x))
    return project(to_out, x)


def project_qkv(projections, x):
    """Share fused activation quantization for three native MXFP8 projections."""
    if len(projections) != 3:
        raise ValueError("expected Q, K, V")
    native = [isinstance(p, MXFP8Linear) for p in projections]
    if any(native) and not all(native):
        raise ValueError("mixed QKV precision is not supported")
    if not any(native):
        return tuple(project(p, x) for p in projections)
    if x.requires_grad or any(p.in_features != x.shape[-1] for p in projections):
        raise ValueError("invalid inference QKV input")
    rows = math.prod(x.shape[:-1])
    data, scales = quantize_activation(x.reshape(rows, x.shape[-1]).contiguous(),
                                       check_finite=any(p.check_finite for p in projections))
    return tuple(p.forward_quantized(data, scales, rows=rows).reshape(*x.shape[:-1], p.out_features)
                 for p in projections)


def convert_linear_to_mxfp8(model, *, enabled=False, chunk_rows=128, check_finite=True):
    """Default-off exact seven-role conversion, including first/last blocks.

    CPU source and meta follower models must call with identical structure. Root
    embeddings/endpoints, norms, gates and small projections stay untouched.
    Missing roles, unmerged adapters and tied selected weights fail explicitly.
    Returns role/module/shape records, empty when disabled. Never calls model.to.
    """
    if type(enabled) is not bool or type(check_finite) is not bool:
        raise ValueError("enabled and check_finite must be booleans")
    if not enabled:
        return []
    if model.training:
        raise ValueError("convert an eval model after effective-weight merging")
    targets, seen = [], set()
    parameter_uses = {}
    for _, parameter in model.named_parameters(remove_duplicate=False):
        parameter_uses[id(parameter)] = parameter_uses.get(id(parameter), 0) + 1
    for index, block in enumerate(model.transformer_blocks):
        for role, suffix in SUPPORTED_ROLES.items():
            child = block.get_submodule(suffix)
            if type(child) is not nn.Linear or child.weight.device.type not in ("cpu", "meta"):
                raise ValueError(f"{index}:{role} requires a merged CPU/meta nn.Linear")
            if id(child.weight) in seen or parameter_uses[id(child.weight)] != 1:
                raise ValueError("tied MXFP8 target weights are unsupported")
            seen.add(id(child.weight))
            parent_path, name = suffix.rsplit(".", 1)
            targets.append((index, role, block.get_submodule(parent_path), name, suffix))
    records = []
    for index, role, parent, name, suffix in targets:
        child = getattr(parent, name)
        records.append(dict(block=index, role=role, module=f"transformer_blocks.{index}.{suffix}",
                            in_features=child.in_features, out_features=child.out_features))
        setattr(parent, name, MXFP8Linear(child, chunk_rows=chunk_rows, check_finite=check_finite))
        del child
    return records
