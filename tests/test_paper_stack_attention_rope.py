"""CPU contracts by default; CUDA execution requires explicit parent opt-in.

CPU: CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_paper_stack_attention_rope.py
GPU: PAPER_STACK_RUN_ROPE_CUDA=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q tests/test_paper_stack_attention_rope.py -k cuda
No CUDA availability/capability query occurs during test collection.
"""

import ast
import importlib.util
import inspect
import os
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
# Isolate this operator from unrelated attention-package import side effects.
SPEC = importlib.util.spec_from_file_location(
    "_paper_rope_test_module", ROOT / "src/models/softmax_attention/paper_qk_rope.py"
)
rope = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rope
SPEC.loader.exec_module(rope)
prepare = rope.paper_qk_rmsnorm_rope


def oracle(x, weight, eps, cos=None, sin=None, *, round_norm=True, round_products=True):
    """Independent FP32 oracle, with explicit low-precision boundaries.

    Deliberately use indexed partner channels instead of the implementation's
    chunk/concatenate rotary, and never invoke its private reference helpers.
    """
    acc = x.float()
    variance = torch.sum(acc * acc, dim=-1, keepdim=True) / x.shape[-1]
    norm = (acc * (variance + eps).rsqrt()) * weight.float()
    if round_norm:
        norm = norm.to(x.dtype).float()
    if cos is not None:
        r = cos.shape[-1]
        index = (torch.arange(r, device=x.device) + r // 2) % r
        sign = torch.ones(r, device=x.device)
        sign[:r // 2] = -1
        lhs = norm[..., :r] * cos.to(x.dtype).float()[:, None]
        rhs = norm.index_select(-1, index) * sign * sin.to(x.dtype).float()[:, None]
        if round_products:
            lhs, rhs = lhs.to(x.dtype).float(), rhs.to(x.dtype).float()
        norm = torch.cat((lhs + rhs, norm[..., r:]), dim=-1)
    return norm.to(x.dtype)


def inputs(dtype=torch.bfloat16, d=128, r=96, layout="qkv", t=13):
    g = torch.Generator(device="cpu").manual_seed(194)
    h = 7
    if layout == "qkv":
        storage = torch.randn(t, 3 * h * d, generator=g).to(dtype)
        q = storage[:, :h*d].view(t, h, d)
        k = storage[:, h*d:2*h*d].view(t, h, d)
    elif layout == "inner":
        storage = torch.randn(t, 3, h * 2, d * 2, generator=g).to(dtype)
        q, k = storage[:, 0, ::2, ::2], storage[:, 1, ::2, ::2]
    else:
        storage = torch.randn(2, t, h, d, generator=g).to(dtype)
        q, k = storage.unbind(0)
    qw = (torch.rand(d * 2, generator=g) + 0.5).to(dtype)[::2]
    kw = (torch.rand(d * 2, generator=g) + 0.5).to(dtype)[::2]
    # Intentionally nonduplicated halves: do not assume a compact half cache.
    angles = torch.randn(t, r * 2, generator=g)
    cos, sin = angles.cos()[:, ::2], angles.sin()[:, ::2]
    return storage, q, k, qw, kw, cos, sin


@pytest.fixture
def no_grad():
    with torch.no_grad():
        yield


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("d,r", [(64, 32), (128, 96), (128, 128), (256, 192)])
@pytest.mark.parametrize("layout", ["qkv", "inner"])
def test_cpu_oracle_geometry_dtype_and_immutability(no_grad, dtype, d, r, layout):
    storage, q, k, qw, kw, cos, sin = inputs(dtype, d, r, layout)
    tensors = (storage, qw, kw, cos, sin)
    before = [v.clone() for v in tensors]
    qo, ko = prepare(q, k, qw, kw, 1e-5, cos, sin)
    for output, raw, weight in ((qo, q, qw), (ko, k, kw)):
        torch.testing.assert_close(output, oracle(raw, weight, 1e-5, cos, sin), rtol=0, atol=0)
        assert output.dtype == dtype and output.shape == (13, 7, d)
        assert output.is_contiguous()
        assert output.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr()
        torch.testing.assert_close(output[..., r:], oracle(raw, weight, 1e-5)[..., r:], rtol=0, atol=0)
    assert qo.data_ptr() != ko.data_ptr()
    for original, copy in zip(tensors, before):
        assert torch.equal(original, copy)


def _installed_rotary():
    # Read the actually installed editable Diffusers function without importing
    # its transformers stack or triggering unrelated optional GPU backends.
    spec = importlib.util.find_spec("diffusers")
    path = Path(next(iter(spec.submodule_search_locations))) / "models/transformers/transformer_minimax_h3.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_rotary_emb")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["_apply_rotary_emb"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cpu_installed_diffusers_semantics(no_grad, dtype):
    _, q, k, qw, kw, c, s = inputs(dtype)
    rotate = _installed_rotary()
    qo, ko = prepare(q, k, qw, kw, 1e-6, c, s)
    for out, raw, weight in ((qo, q, qw), (ko, k, kw)):
        expected = rotate(F.rms_norm(raw, (128,), weight, 1e-6)[None], c, s)[0]
        torch.testing.assert_close(out, expected, atol=0 if dtype != torch.float32 else 2e-6,
                                   rtol=0 if dtype != torch.float32 else 2e-6)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_cpu_norm_only_fp32_weight_and_empty(no_grad, weight_dtype):
    _, q, k, qw, kw, _, _ = inputs()
    qw, kw = qw.to(weight_dtype), kw.to(weight_dtype)
    qo, ko = prepare(q, k, qw, kw)
    for out, raw, w in ((qo, q, qw), (ko, k, kw)):
        torch.testing.assert_close(out, oracle(raw, w, 1e-6), rtol=0, atol=0)
        assert out.data_ptr() != raw.data_ptr()
    for out in prepare(q[:0], k[:0], qw, kw):
        assert out.shape == (0, 7, 128) and out.dtype == q.dtype


def trap_inputs():
    g = torch.Generator(device="cpu").manual_seed(715)
    x = torch.ones(3, 7, 128, dtype=torch.bfloat16)
    w = (torch.rand(128, generator=g) * 1.7 + .1).bfloat16()
    angles = torch.randn(3, 96, generator=g)
    return x, w, angles.cos(), angles.sin()


def test_cpu_bf16_rounding_traps(no_grad):
    x, w, c, s = trap_inputs()
    exact = oracle(x, w, .25, c, s)
    assert torch.count_nonzero(exact != oracle(x, w, .25, c, s, round_norm=False)) > 100
    assert torch.count_nonzero(exact != oracle(x, w, .25, c, s, round_products=False)) > 100
    out, _ = prepare(x, x, w, w, .25, c, s)
    torch.testing.assert_close(out, exact, rtol=0, atol=0)
    # A separate trap for wrongly rounding the unweighted norm before weight.
    _, raw, _, weight, _, _, _ = inputs()
    unweighted = raw.float() * (raw.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    wrong = unweighted.bfloat16() * weight
    expected = oracle(raw, weight, 1e-6)
    assert torch.count_nonzero(wrong != expected) > 100


def test_cpu_default_off_no_dispatch_or_device_probe(no_grad, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default-off/CPU execution must not dispatch or probe CUDA")
    monkeypatch.setattr(rope, "_get_kernel", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setenv("VDN_PAPER_QK_ROPE", "1")
    _, q, k, qw, kw, c, s = inputs()
    assert inspect.signature(prepare).parameters["enabled"].default is False
    out, _ = prepare(q, k, qw, kw, cos=c, sin=s)
    assert not torch.equal(out, q)
    with pytest.raises(ValueError, match="requires CUDA"):
        prepare(q, k, qw, kw, enabled=True)


def test_cpu_lazy_kernel_definition_does_not_initialize_device(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("constructing the JIT definition must not probe CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    kernel = rope._get_kernel()
    assert kernel.fn.__name__ == "fused"
    # Do not compile, launch, or select a Triton driver in this CPU test.


@pytest.mark.parametrize("change,error,match", [
    ({"q": None}, TypeError, "q must be"),
    ({"q": torch.zeros(1, 128)}, ValueError, "identical"),
    ({"k": torch.zeros(13, 8, 128)}, ValueError, "identical"),
    ({"q": torch.zeros(13, 7, 128, dtype=torch.int32)}, TypeError, "dtype"),
    ({"k": torch.zeros(13, 7, 128)}, TypeError, "dtype"),
    ({"q_weight": torch.ones(7, 128)}, ValueError, "shape"),
    ({"k_weight": torch.ones(128, dtype=torch.float64)}, TypeError, "activation dtype"),
    ({"q_weight": torch.ones(128, device="meta")}, ValueError, "q/k device"),
    ({"sin": torch.ones(13, 96, device="meta")}, ValueError, "q/k device"),
    ({"cos": None}, ValueError, "both"),
    ({"cos": torch.ones(13, 1, 96)}, ValueError, "strided shape"),
    ({"cos": torch.ones(1, 96)}, ValueError, "identical"),
    ({"cos": torch.ones(13, 96, dtype=torch.int32)}, TypeError, "BF16"),
    ({"eps": 0}, ValueError, "positive"),
    ({"eps": -1}, ValueError, "positive"),
    ({"eps": float("nan")}, ValueError, "finite"),
    ({"eps": float("inf")}, ValueError, "finite"),
    ({"eps": True}, TypeError, "real scalar"),
    ({"eps": "1e-6"}, TypeError, "real scalar"),
    ({"enabled": 1}, TypeError, "bool"),
    ({"nvtx": "yes"}, TypeError, "bool"),
])
def test_cpu_invalid_arguments(no_grad, change, error, match):
    _, q, k, qw, kw, c, s = inputs()
    args = dict(q=q, k=k, q_weight=qw, k_weight=kw, cos=c, sin=s)
    args.update(change)
    with pytest.raises(error, match=match):
        prepare(**args)


@pytest.mark.parametrize("r", [0, 95, 130])
def test_cpu_invalid_rotary_geometry(no_grad, r):
    _, q, k, qw, kw, _, _ = inputs()
    with pytest.raises(ValueError, match="rotary width"):
        prepare(q, k, qw, kw, cos=torch.zeros(13, r), sin=torch.zeros(13, r))


@pytest.mark.parametrize("h,d", [(0, 128), (7, 127), (7, 512)])
def test_cpu_invalid_head_geometry(no_grad, h, d):
    x = torch.zeros(13, h, d, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="H must be positive"):
        prepare(x, x, torch.ones(d), torch.ones(d))


@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_inference_guard(enabled):
    _, q, k, qw, kw, _, _ = inputs()
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        prepare(q, k, qw, kw, enabled=enabled)
    # Parameters requiring grad are valid under an explicit inference scope.
    qw.requires_grad_(True)
    with torch.inference_mode():
        out, _ = prepare(q, k, qw, kw)
        assert not out.requires_grad


@pytest.fixture
def cuda_device():
    if os.environ.get("PAPER_STACK_RUN_ROPE_CUDA") != "1":
        pytest.skip("parent must opt in with PAPER_STACK_RUN_ROPE_CUDA=1")
    import subprocess
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv"], check=True)
    if not torch.cuda.is_available():
        pytest.fail("CUDA explicitly requested but unavailable")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.fail("CUDA operator validation requires native SM120")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["qkv", "inner"])
@pytest.mark.parametrize("r", [None, 96, 128])
def test_cuda_independent_oracle_strided_inputs(no_grad, cuda_device, dtype, layout, r):
    storage, q, k, qw, kw, c, s = inputs(dtype, r=r or 96, layout=layout, t=33)
    # Move backing storage, then reproduce views so transfer cannot compact them.
    backing = storage.to(cuda_device)
    qg = backing.as_strided(q.shape, q.stride(), q.storage_offset())
    kg = backing.as_strided(k.shape, k.stride(), k.storage_offset())
    qw, kw = qw.to(cuda_device), kw.to(cuda_device)
    cg, sg = (None, None) if r is None else (c.to(cuda_device), s.to(cuda_device))
    before = [v.clone() for v in (backing, qw, kw)]
    outputs = prepare(qg, kg, qw, kw, 1e-5, cg, sg, enabled=True)
    eager_outputs = prepare(qg, kg, qw, kw, 1e-5, cg, sg)
    for output, raw, weight in zip(outputs, (qg, kg), (qw, kw)):
        expected = oracle(raw, weight, 1e-5, cg, sg)
        error = (output.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
        assert error.item() < (2e-3 if dtype == torch.bfloat16 else 3e-4 if dtype == torch.float16 else 2e-6)
        torch.testing.assert_close(output, expected, rtol=0.02 if dtype == torch.bfloat16 else .003 if dtype == torch.float16 else 2e-5,
                                   atol=.032 if dtype == torch.bfloat16 else .004 if dtype == torch.float16 else 2e-6)
        assert output.dtype == dtype and output.shape == (33, 7, 128)
        assert output.is_contiguous() and output.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()
    for value, copy in zip((backing, qw, kw), before):
        assert torch.equal(value, copy)
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    for eager, raw, weight in zip(eager_outputs, (qg, kg), (qw, kw)):
        torch.testing.assert_close(eager, oracle(raw, weight, 1e-5, cg, sg), rtol=0, atol=0)


def test_cuda_bf16_intermediate_round_trap(no_grad, cuda_device):
    x, w, c, s = [v.to(cuda_device) for v in trap_inputs()]
    expected = oracle(x, w, .25, c, s)
    assert torch.count_nonzero(expected != oracle(x, w, .25, c, s, round_norm=False)) > 100
    out, _ = prepare(x, x, w, w, .25, c, s, enabled=True)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("d,r", [(64, 32), (256, 192)])
def test_cuda_other_widths_fp32_weights_strided_tables(no_grad, cuda_device, d, r):
    _, q, k, qw, kw, c, s = inputs(d=d, r=r)
    q, k, qw, kw = [v.to(cuda_device) for v in (q, k, qw.float(), kw.float())]
    wb = torch.empty(2, d * 2, device=cuda_device)
    wb[0, ::2], wb[1, ::2] = qw, kw
    qw, kw = wb[0, ::2], wb[1, ::2]
    cb, sb = torch.empty(13, r * 2, device=cuda_device), torch.empty(13, r * 2, device=cuda_device)
    cb[:, ::2], sb[:, ::2] = c.to(cuda_device), s.to(cuda_device)
    c, s = cb[:, ::2], sb[:, ::2]
    out, _ = prepare(q, k, qw, kw, cos=c, sin=s, enabled=True)
    torch.testing.assert_close(out, oracle(q, qw, 1e-6, c, s), atol=.032, rtol=.02)
    qo, ko = prepare(q[:0], k[:0], qw, kw, enabled=True)
    assert qo.shape == ko.shape == (0, 7, d)


def test_cuda_matches_installed_eager_diffusers(no_grad, cuda_device):
    _, q, k, qw, kw, c, s = inputs()
    q, k, qw, kw, c, s = [v.to(cuda_device) for v in (q, k, qw, kw, c, s)]
    rotate = _installed_rotary()
    out, _ = prepare(q, k, qw, kw, 1e-6, c, s, enabled=True)
    expected = rotate(F.rms_norm(q, (128,), qw, 1e-6)[None], c, s)[0]
    torch.testing.assert_close(out, expected, atol=.032, rtol=.02)
