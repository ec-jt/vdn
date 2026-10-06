# SPDX-License-Identifier: Apache-2.0
# Adapted tests from SGLang contributors' test_vdn_delta_factors.py at
# ff1ce11348673eea9b51165bd6b2da7238c958b4 (PR #37903), Apache-2.0:
# test/registered/kernel/diffusion/test_vdn_delta_factors.py.
# Modified for the standalone port: isolated imports, CPU FP64 solve oracle,
# strict validation, optional alpha, ownership, empty batches and stream tests.
"""Standalone factor contract tests, independent of the model runtime.

GPU availability/capability is checked only when a GPU fixture executes. Selecting
``-k cpu`` does not probe CUDA or build the extension. GPU tests intentionally do
not skip build/resource failures on supported hardware: those are failures, not
evidence of numerical correctness. No SGLang or local scan implementation is used.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/models/linear_attention/paper_factor.py"
HEAD_DIM = 128


def _load_factor_module():
    # Do not import src.models: its package initialization couples model backends.
    spec = importlib.util.spec_from_file_location("_standalone_paper_factor", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forbidden(*args, **kwargs):
    raise AssertionError("CPU validation/import must not probe CUDA or load an extension")


@pytest.fixture
def cpu_module(monkeypatch):
    # Replace the build module itself rather than import cpp_extension, whose
    # own initialization can inspect the CUDA installation on some torch builds.
    build_module = ModuleType("torch.utils.cpp_extension")
    build_module.load = _forbidden
    build_module.load_inline = _forbidden
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", build_module)
    monkeypatch.setattr(torch.utils, "cpp_extension", build_module, raising=False)
    for name in (
        "is_available", "device_count", "get_device_capability",
        "get_device_properties", "current_device", "init", "_lazy_init",
    ):
        monkeypatch.setattr(torch.cuda, name, _forbidden)
    module = _load_factor_module()
    monkeypatch.setattr(module, "_load_extension", _forbidden)
    return module


@pytest.fixture(scope="module")
def factor_module():
    return _load_factor_module()


@pytest.fixture(scope="module")
def cuda_device():
    # No CUDA calls in decorators, module globals, or collection hooks.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for standalone factor execution")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] < 8:
        pytest.skip("Standalone factor numerical tests require SM80+")
    return device


def _fp64_oracle(A, B, alpha=None):
    """Independent CPU right solves, with no inverse or project factor helpers."""
    a = A.detach().to(device="cpu", dtype=torch.float64)
    b = B.detach().to(device="cpu", dtype=torch.float64)
    eye = torch.eye(HEAD_DIM, dtype=torch.float64, device="cpu")
    system = eye + a
    rhs = eye.expand_as(a)
    if alpha is not None:
        rhs = torch.diag_embed(alpha.detach().to(device="cpu", dtype=torch.float64))
    # X M = R is solved as M^T X^T = R^T. In particular, solving M J = B
    # would put the inverse on the wrong side of asymmetric B.
    t = torch.linalg.solve(system.transpose(-1, -2), rhs.transpose(-1, -2))
    j = torch.linalg.solve(system.transpose(-1, -2), b.transpose(-1, -2))
    return t.transpose(-1, -2), j.transpose(-1, -2)


def _stats_inputs(frames, heads, tokens, beta_scale=1.0, seed=0):
    # Adapted from upstream _inputs; generate on CPU to avoid TF32/RNG policy
    # dependence when testing the factor operation rather than its prologue.
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (frames, heads, tokens, HEAD_DIM)
    k = F.normalize(F.silu(torch.randn(shape, generator=g, device="cpu")), dim=-1)
    v = F.silu(torch.randn(shape, generator=g, device="cpu"))
    beta = torch.sigmoid(torch.randn(shape[:-1], generator=g, device="cpu")) * beta_scale
    alpha = torch.rand(frames, heads, HEAD_DIM, generator=g, device="cpu") * 0.9 + 0.1
    alpha[0] = 1.0
    A = (k * beta.unsqueeze(-1)).transpose(-1, -2) @ k
    A = 0.5 * (A + A.transpose(-1, -2))
    B = (v * beta.unsqueeze(-1)).transpose(-1, -2) @ k
    return A.float().contiguous(), B.float().contiguous(), alpha.float().contiguous()


def _orientation_inputs(batch_shape=()):
    g = torch.Generator(device="cpu").manual_seed(701)
    u = torch.linspace(-0.8, 1.1, HEAD_DIM, device="cpu")
    A = torch.diag(torch.linspace(0.1, 2.0, HEAD_DIM, device="cpu"))
    A = A + 0.15 * u[:, None] * u[None, :]
    B = torch.randn(HEAD_DIM, HEAD_DIM, generator=g, device="cpu")
    alpha = torch.linspace(0.1, 0.95, HEAD_DIM, device="cpu")
    return (
        A.expand(*batch_shape, HEAD_DIM, HEAD_DIM).contiguous(),
        B.expand(*batch_shape, HEAD_DIM, HEAD_DIM).contiguous(),
        alpha.expand(*batch_shape, HEAD_DIM).contiguous(),
    )


def _to_device(inputs, device):
    return tuple(x.to(device) if x is not None else None for x in inputs)


def _assert_outputs(outputs, A, B):
    assert len(outputs) == 2
    for got, source in zip(outputs, (A, B)):
        assert got.shape == source.shape
        assert got.dtype == torch.float32
        assert got.device == source.device
        assert not got.requires_grad
        assert torch.isfinite(got).all()


def _assert_accuracy(outputs, refs):
    for got, ref, limit in zip(outputs, refs, (1e-5, 3e-5)):
        got = got.detach().to(device="cpu", dtype=torch.float64)
        relative_error = (got - ref).norm() / ref.norm()
        assert relative_error.item() < limit


def _assert_bits_equal(got, want):
    # Integer views also distinguish positive and negative zero.
    assert torch.equal(got.contiguous().view(torch.int32), want.contiguous().view(torch.int32))


def _storage_offset_copy(tensor, offset):
    # Adapted from upstream: contiguous does not imply float4 alignment.
    storage = torch.empty(tensor.numel() + offset, device=tensor.device, dtype=tensor.dtype)
    result = storage[offset:].view(tensor.shape)
    result.copy_(tensor)
    assert result.is_contiguous()
    assert result.storage_offset() == offset
    assert result.data_ptr() % 16 != 0
    return result


def test_cpu_import_is_lazy(cpu_module):
    assert callable(cpu_module.paper_delta_factors)


@pytest.mark.parametrize("batch_shape", [(), (2,), (0,), (2, 0, 3)])
@pytest.mark.parametrize("with_alpha", [False, True])
def test_cpu_rejected_before_extension_loading(cpu_module, batch_shape, with_alpha):
    A = torch.zeros(*batch_shape, HEAD_DIM, HEAD_DIM, device="cpu")
    B = torch.zeros_like(A)
    alpha = torch.ones(*batch_shape, HEAD_DIM, device="cpu") if with_alpha else None
    with pytest.raises(ValueError, match="CUDA"):
        cpu_module.paper_delta_factors(A, B, alpha)


@pytest.mark.parametrize("bad_A", [None, 1, "A", [1.0]])
def test_cpu_non_tensor_A_is_type_error(cpu_module, bad_A):
    B = torch.zeros(HEAD_DIM, HEAD_DIM, device="cpu")
    with pytest.raises(TypeError):
        cpu_module.paper_delta_factors(bad_A, B)


def test_cpu_oracle_right_solve_and_alpha_orientation(cpu_module):
    A, B, alpha = _orientation_inputs()
    t, j = _fp64_oracle(A, B, alpha)
    inverse, _ = _fp64_oracle(A, B)
    system = torch.eye(HEAD_DIM, dtype=torch.float64, device="cpu") + A.double()
    torch.testing.assert_close(t @ system, torch.diag(alpha.double()), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(j @ system, B.double(), rtol=1e-12, atol=1e-12)
    assert not torch.allclose(j, torch.linalg.solve(system, B.double()), rtol=1e-3, atol=1e-3)
    assert not torch.allclose(t, inverse * alpha.double().unsqueeze(-2), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("frames,heads,tokens", [(1, 1, 16), (5, 3, 64), (11, 7, 1008)])
@pytest.mark.parametrize("beta_scale", [1.0, 50.0])
@pytest.mark.parametrize("with_alpha", [False, True])
def test_cuda_normalized_silu_statistics(
    cuda_device, factor_module, frames, heads, tokens, beta_scale, with_alpha
):
    A, B, alpha = _stats_inputs(frames, heads, tokens, beta_scale)
    if not with_alpha:
        alpha = None
    refs = _fp64_oracle(A, B, alpha)
    A, B, alpha = _to_device((A, B, alpha), cuda_device)
    outputs = factor_module.paper_delta_factors(A, B, alpha)
    _assert_outputs(outputs, A, B)
    _assert_accuracy(outputs, refs)


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1, 3)])
@pytest.mark.parametrize("with_alpha", [False, True])
def test_cuda_asymmetric_B_nonuniform_alpha(cuda_device, factor_module, batch_shape, with_alpha):
    A, B, alpha = _orientation_inputs(batch_shape)
    if not with_alpha:
        alpha = None
    refs = _fp64_oracle(A, B, alpha)
    A, B, alpha = _to_device((A, B, alpha), cuda_device)
    # Exercise the omitted optional argument, not only an explicit None.
    outputs = (factor_module.paper_delta_factors(A, B, alpha) if with_alpha
               else factor_module.paper_delta_factors(A, B))
    _assert_outputs(outputs, A, B)
    _assert_accuracy(outputs, refs)
    for got, ref in zip(outputs, refs):
        torch.testing.assert_close(got.double().cpu(), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("which", ["A", "B", "alpha", "all"])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_cuda_storage_offsets_bitwise_match_aligned(cuda_device, factor_module, which, offset):
    inputs = dict(zip(("A", "B", "alpha"), _to_device(_stats_inputs(3, 2, 64), cuda_device)))
    assert all(x.data_ptr() % 16 == 0 for x in inputs.values())
    aligned = factor_module.paper_delta_factors(**inputs)
    for name in inputs:
        if which in (name, "all"):
            inputs[name] = _storage_offset_copy(inputs[name], offset)
    snapshots = {name: x.clone() for name, x in inputs.items()}
    outputs = factor_module.paper_delta_factors(**inputs)
    for got, ref in zip(outputs, aligned):
        _assert_bits_equal(got, ref)
    for name in inputs:
        _assert_bits_equal(inputs[name], snapshots[name])


@pytest.mark.parametrize("which", ["A", "B"])
def test_cuda_storage_offset_without_alpha(cuda_device, factor_module, which):
    A, B, _ = _to_device(_orientation_inputs(), cuda_device)
    aligned = factor_module.paper_delta_factors(A, B)
    inputs = {"A": A, "B": B}
    inputs[which] = _storage_offset_copy(inputs[which], 1)
    for got, ref in zip(factor_module.paper_delta_factors(**inputs), aligned):
        _assert_bits_equal(got, ref)


@pytest.mark.parametrize("which", ["A", "B", "alpha"])
@pytest.mark.parametrize("bad_value", [7, "not a tensor", [1.0]])
def test_cuda_non_tensor_is_type_error(cuda_device, factor_module, monkeypatch, which, bad_value):
    inputs = dict(zip(("A", "B", "alpha"), _to_device(_orientation_inputs(), cuda_device)))
    inputs[which] = bad_value
    monkeypatch.setattr(factor_module, "_load_extension", _forbidden)
    with pytest.raises(TypeError):
        factor_module.paper_delta_factors(**inputs)


@pytest.mark.parametrize("which", ["A", "B", "alpha"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64, torch.int32])
def test_cuda_wrong_dtype_is_type_error(cuda_device, factor_module, monkeypatch, which, dtype):
    inputs = dict(zip(("A", "B", "alpha"), _to_device(_orientation_inputs(), cuda_device)))
    inputs[which] = inputs[which].to(dtype)
    monkeypatch.setattr(factor_module, "_load_extension", _forbidden)
    with pytest.raises(TypeError):
        factor_module.paper_delta_factors(**inputs)


@pytest.mark.parametrize("which,shape", [
    ("A", ()), ("A", (128,)), ("A", (2, 64, 64)), ("A", (2, 128, 64)),
    ("B", (128, 128)), ("B", (1, 128, 128)), ("B", (2, 64, 128)),
    ("B", (1, 2, 128, 128)),
    ("alpha", (128,)), ("alpha", (1, 128)), ("alpha", (2, 127)),
    ("alpha", (2, 128, 1)),
])
def test_cuda_wrong_shape_is_value_error(cuda_device, factor_module, monkeypatch, which, shape):
    inputs = dict(zip(("A", "B", "alpha"), _to_device(_orientation_inputs((2,)), cuda_device)))
    inputs[which] = torch.zeros(shape, dtype=torch.float32, device=cuda_device)
    monkeypatch.setattr(factor_module, "_load_extension", _forbidden)
    with pytest.raises(ValueError):
        factor_module.paper_delta_factors(**inputs)


@pytest.mark.parametrize("which", ["A", "B", "alpha"])
def test_cuda_noncontiguous_is_value_error(cuda_device, factor_module, monkeypatch, which):
    inputs = dict(zip(("A", "B", "alpha"), _to_device(_orientation_inputs((2,)), cuda_device)))
    source = inputs[which]
    if which == "alpha":
        storage = torch.empty(2, HEAD_DIM * 2, device=cuda_device)
        inputs[which] = storage[:, ::2]
        inputs[which].copy_(source)
    else:
        inputs[which] = source.transpose(-1, -2)
    assert not inputs[which].is_contiguous()
    monkeypatch.setattr(factor_module, "_load_extension", _forbidden)
    with pytest.raises(ValueError):
        factor_module.paper_delta_factors(**inputs)


def test_cuda_expanded_alpha_is_value_error(cuda_device, factor_module, monkeypatch):
    A, B, alpha = _to_device(_orientation_inputs((2,)), cuda_device)
    alpha = alpha[..., :1].expand_as(alpha)
    assert not alpha.is_contiguous()
    monkeypatch.setattr(factor_module, "_load_extension", _forbidden)
    with pytest.raises(ValueError):
        factor_module.paper_delta_factors(A, B, alpha)


@pytest.mark.parametrize("which", ["A", "B", "alpha"])
def test_cuda_mixed_host_device_is_value_error(cuda_device, factor_module, monkeypatch, which):
    inputs = dict(zip(("A", "B", "alpha"), _to_device(_orientation_inputs(), cuda_device)))
    inputs[which] = inputs[which].cpu()
    monkeypatch.setattr(factor_module, "_load_extension", _forbidden)
    with pytest.raises(ValueError):
        factor_module.paper_delta_factors(**inputs)


@pytest.mark.parametrize("which", ["B", "alpha"])
def test_cuda_mixed_cuda_devices_is_value_error(cuda_device, factor_module, monkeypatch, which):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices are needed for the mixed-device contract")
    other = torch.device("cuda", (cuda_device.index + 1) % torch.cuda.device_count())
    inputs = dict(zip(("A", "B", "alpha"), _to_device(_orientation_inputs(), cuda_device)))
    inputs[which] = inputs[which].to(other)
    monkeypatch.setattr(factor_module, "_load_extension", _forbidden)
    with pytest.raises(ValueError):
        factor_module.paper_delta_factors(**inputs)


@pytest.mark.parametrize("which", ["A", "B", "alpha"])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("batch_shape", [(), (0,)])
def test_cuda_requires_grad_rejected_even_under_no_grad(
    cuda_device, factor_module, monkeypatch, which, grad_enabled, batch_shape
):
    inputs = dict(zip(("A", "B", "alpha"), _to_device(_orientation_inputs(batch_shape), cuda_device)))
    inputs[which].requires_grad_(True)
    monkeypatch.setattr(factor_module, "_load_extension", _forbidden)
    with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError):
        factor_module.paper_delta_factors(**inputs)


@pytest.mark.parametrize("with_alpha", [False, True])
def test_cuda_input_immutability_and_output_ownership(cuda_device, factor_module, with_alpha):
    A, B, alpha = _to_device(_orientation_inputs((2,)), cuda_device)
    inputs = (A, B, alpha) if with_alpha else (A, B)
    snapshots = tuple(x.clone() for x in inputs)
    t, j = factor_module.paper_delta_factors(*inputs)
    _assert_outputs((t, j), A, B)
    pointers = [x.untyped_storage().data_ptr() for x in (*inputs, t, j)]
    assert len(pointers) == len(set(pointers))
    for got, ref in zip(inputs, snapshots):
        _assert_bits_equal(got, ref)
    j_snapshot = j.clone()
    t.fill_(123.0)
    _assert_bits_equal(j, j_snapshot)
    j.fill_(-456.0)
    for got, ref in zip(inputs, snapshots):
        _assert_bits_equal(got, ref)


@pytest.mark.parametrize("batch_shape", [(0,), (2, 0, 3), (0, 2)])
@pytest.mark.parametrize("with_alpha", [False, True])
def test_cuda_empty_batches(cuda_device, factor_module, batch_shape, with_alpha):
    A = torch.empty(*batch_shape, HEAD_DIM, HEAD_DIM, device=cuda_device)
    B = torch.empty_like(A)
    alpha = torch.empty(*batch_shape, HEAD_DIM, device=cuda_device) if with_alpha else None
    outputs = factor_module.paper_delta_factors(A, B, alpha)
    _assert_outputs(outputs, A, B)
    assert all(x.numel() == 0 for x in outputs)
    # Empty allocations may legitimately share a null storage pointer.


def test_cuda_sm_below_80_is_runtime_error(cuda_device, factor_module, monkeypatch):
    inputs = _to_device(_orientation_inputs(), cuda_device)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (7, 5))
    monkeypatch.setattr(factor_module, "_load_extension", _forbidden)
    with pytest.raises(RuntimeError):
        factor_module.paper_delta_factors(*inputs)


@pytest.mark.parametrize("message", ["injected extension build failure", "injected shared-memory resource failure"])
def test_cuda_loader_failure_is_not_silently_fallback(
    cuda_device, factor_module, monkeypatch, message
):
    inputs = _to_device(_orientation_inputs(), cuda_device)

    def fail_load(*args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(factor_module, "_load_extension", fail_load)
    with pytest.raises(RuntimeError):
        factor_module.paper_delta_factors(*inputs)


def test_cuda_non_default_stream(cuda_device, factor_module):
    cpu_inputs = _orientation_inputs((2,))
    refs = _fp64_oracle(*cpu_inputs)
    inputs = _to_device(cpu_inputs, cuda_device)
    # Warm the extension before testing stream ordering, not during its build.
    factor_module.paper_delta_factors(*inputs)
    torch.cuda.synchronize(cuda_device)
    stream = torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(stream):
        # Both producers and consumers belong to this stream. A default-stream
        # launch can race either side; do not synchronize globally in this block.
        A, B, alpha = (x.clone() for x in inputs)
        outputs = factor_module.paper_delta_factors(A, B, alpha)
        consumed = tuple(x.clone() for x in outputs)
        ready = torch.cuda.Event()
        ready.record(stream)
    ready.synchronize()
    _assert_outputs(consumed, A, B)
    _assert_accuracy(consumed, refs)
