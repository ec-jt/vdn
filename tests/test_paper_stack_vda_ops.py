"""Operator-only tests. Run serially with CUDA_VISIBLE_DEVICES=0, no full model."""
import pytest
import torch
import torch.nn.functional as F

from src.models.linear_attention.paper_stack import (
    paper_prepare, paper_statistics, paper_boundary_plan, paper_boundary_scan,
    paper_gather, vdn_linear_epilogue,
)
from src.models.linear_attention.paper_factor import paper_delta_factors
from src.models.linear_attention.branch import BidirectionalLinearBranch
from src.models.linear_attention.features import LinearAttentionSepConv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def precision():
    torch.manual_seed(711)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        yield
    torch.backends.cuda.matmul.allow_tf32 = old


def close_rel(a, b, limit):
    err = (a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-20)
    assert err.item() < limit, err.item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("frames,spatial", [(1, 1), (3, 7), (17, 12)])
@pytest.mark.parametrize("l2", [False, True])
@pytest.mark.parametrize("conv", [False, True])
def test_prep_fp64(dtype, frames, spatial, l2, conv):
    h, d = 2, 128
    raw = torch.randn(frames * spatial, h, d, device="cuda", dtype=dtype)
    module = LinearAttentionSepConv(h * d, ("k",)).cuda().to(dtype) if conv else None
    result = paper_prepare(raw, frames=frames, spatial=spatial, l2norm=l2,
                           conv=module, proj="k", frame_size=(1, spatial))
    if conv:
        x, w = module.spatial("k", raw, frames, (1, spatial))
        x = x.double()
        acc = torch.zeros_like(x)
        for t in range(frames):
            for j in range(5):
                if 0 <= t + j - 2 < frames:
                    acc[t] += x[t + j - 2] * w[:, j].double()
        x = acc.reshape(frames * spatial, h, d)
    else:
        x = raw.double()
    x = F.silu(x)
    if l2:
        x = x / x.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
    expected = x.reshape(frames, spatial, h, d).permute(0, 2, 1, 3)
    assert result.is_contiguous()
    if dtype == torch.bfloat16:
        torch.testing.assert_close(result, expected.to(dtype), atol=0., rtol=0.008)
    else:
        close_rel(result, expected, 5e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("s", [1, 33, 129, 1008])
def test_joint_statistics_oracle(dtype, s):
    k = F.normalize(torch.randn(3, 2, s, 128, device="cuda"), dim=-1).to(dtype)
    v = torch.randn_like(k)
    beta = torch.rand(3, 2, s, device="cuda", dtype=torch.float32)
    a, b = paper_statistics(k, v, beta)
    oracle_a = (k.double() * beta.double()[..., None]).transpose(-1, -2) @ k.double()
    if dtype == torch.bfloat16:
        vb = (v * beta.to(dtype)[..., None]).double()
        oracle_b = (vb.transpose(-1, -2) @ k.double()).to(dtype).float()
    else:
        oracle_b = (v.double() * beta.double()[..., None]).transpose(-1, -2) @ k.double()
    assert a.dtype == b.dtype == torch.float32
    close_rel(a, oracle_a, 5e-7)
    close_rel(b, oracle_b, 1e-4 if dtype == torch.bfloat16 else 8e-7)
    torch.testing.assert_close(a, a.transpose(-1, -2), atol=2e-6, rtol=1e-6)


def scan_oracle(t, inj, seed):
    t, inj = t.double(), inj.double()
    start = torch.zeros_like(inj[0]) if seed is None else seed.double()
    forward, reverse = [], [None] * len(t)
    state = start
    for f in range(len(t)):
        state = state @ t[f] + inj[f]
        forward.append(state)
    state = start
    for f in range(len(t) - 1, -1, -1):
        state = state @ t[f] + inj[f]
        reverse[f] = state
    return torch.stack(forward), torch.stack(reverse)


def gather_oracle(prefix, suffix, alpha, bounds, seed, bridge=True):
    start = torch.zeros_like(prefix[0]) if seed is None else seed.double()
    result = []
    fcount = len(prefix)
    for f, (lo, hi) in enumerate(bounds):
        left = prefix[lo - 1] if lo > 0 else start
        right = suffix[hi + 1] if hi + 1 < fcount else start
        if bridge:
            left = left * alpha[max(0, lo):f + 1].double().clamp_min(1e-12).prod(0)[..., None, :]
            right = right * alpha[f:min(fcount, hi + 1)].double().clamp_min(1e-12).prod(0)[..., None, :]
        result.append(left + right)
    return torch.stack(result)


@pytest.mark.parametrize("frames,chunk", [(1, 1), (2, 8), (11, 4), (17, 8), (31, 1)])
@pytest.mark.parametrize("text", [False, True])
@pytest.mark.parametrize("decay", [1., 0.7, 0.001])
def test_compact_both_scan_and_gather_fp64(frames, chunk, text, decay):
    h, d = 2, 128
    k = F.normalize(torch.randn(frames, h, 9, d, device="cuda"), dim=-1)
    a = (k.transpose(-1, -2) @ k).contiguous()
    b = torch.randn_like(a) * 0.1
    alpha = (torch.rand(frames, h, d, device="cuda") * 0.1 + 0.9) * decay
    t, inj = paper_delta_factors(a, b, alpha)
    seed = torch.randn_like(a[0]) * .3 if text else None
    # Bounds cross the original clip ends as anchor rebasing does.
    bounds = [(f // chunk * chunk - chunk, (f // chunk + 2) * chunk - 1)
              for f in range(frames)]
    plan = paper_boundary_plan(bounds, frames, a.device, chunk)
    bank = paper_boundary_scan(t, inj, plan, seed)
    prefix, suffix = scan_oracle(t, inj, seed)
    torch.testing.assert_close(bank[0].double(), prefix[torch.tensor(plan.cuts[1:]) - 1],
                               atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(bank[1].double(), suffix[list(plan.cuts[:-1])],
                               atol=3e-6, rtol=3e-5)
    result = paper_gather(bank, alpha, plan, seed, torch.float32)
    expected = gather_oracle(prefix, suffix, alpha, bounds, seed)
    torch.testing.assert_close(result.double(), expected, atol=4e-6, rtol=5e-5)
    assert bank.shape == (2, plan.chunks, h, d, d)
    assert bank.dtype == torch.float32
    if chunk > 1 and frames > 2:
        assert plan.chunks < frames


def test_irregular_bounds_zero_decay_and_no_bridge():
    frames, h, d = 9, 1, 32
    t = torch.eye(d, device="cuda").expand(frames, h, d, d).contiguous() * .9
    inj = torch.randn_like(t)
    alpha = torch.zeros(frames, h, d, device="cuda")
    bounds = [(max(0, f - f % 3), min(frames - 1, f + 2)) for f in range(frames)]
    plan = paper_boundary_plan(bounds, frames, t.device, 4)
    seed = torch.randn_like(t[0])
    bank = paper_boundary_scan(t, inj, plan, seed)
    prefix, suffix = scan_oracle(t, inj, seed)
    for bridge in (True, False):
        out = paper_gather(bank, alpha, plan, seed, torch.float32, bridge)
        ref = gather_oracle(prefix, suffix, alpha, bounds, seed, bridge)
        torch.testing.assert_close(out.double(), ref, atol=5e-6, rtol=1e-5)


def test_epilogue_bf16_residual():
    r = torch.randn(3, 2, 7, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(128, device="cuda")
    g = torch.rand(21, 2, 128, device="cuda", dtype=torch.bfloat16)
    out = vdn_linear_epilogue(r, w, g, 1e-6)
    ref = r.double() * torch.rsqrt(r.double().square().mean(-1, keepdim=True) + 1e-6) * w.double()
    ref = (ref.permute(0, 2, 1, 3).reshape_as(g) * g.double()).reshape_as(out)
    torch.testing.assert_close(out, ref.to(out.dtype), atol=0.0078125, rtol=0.008)
    residual = torch.randn_like(out)
    result = residual.clone().add_(out)
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()


@pytest.mark.parametrize("short", [(), ("k", "v"), ("q", "k", "v")])
@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("text", [False, True])
def test_branch_explicit_route_no_old_fallback(short, anchor, text, monkeypatch):
    f, s, h, d, hidden = 9, 6, 2, 128, 24
    branch = BidirectionalLinearBranch(hidden, h, d, delta_rule="vdn_solve", short_conv=short).cuda()
    assert not branch.paper_vda_enabled
    branch.paper_vda_enabled = True
    def forbidden(*args, **kwargs):
        raise AssertionError("old reference path was called")
    monkeypatch.setattr(branch, "_readout", forbidden)
    monkeypatch.setattr(branch, "_readout_inference", forbidden)
    monkeypatch.setattr(branch, "_text_state", forbidden)
    raw = tuple(torch.randn(f*s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    xv = torch.randn(f*s, hidden, device="cuda", dtype=torch.bfloat16)
    tx = torch.randn(5, hidden, device="cuda", dtype=torch.bfloat16) if text else None
    tq = tuple(torch.randn(5, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)) if text else None
    bounds = [(max(0, i // 3 * 3 - 3), min(f-1, i // 3 * 3 + 5)) for i in range(f)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = branch(xv, f, s, bounds, raw, frame_size=(2, 3), skip_ends=anchor,
                     text_x=tx, text_qkv_raw=tq)
    assert out.dtype == torch.bfloat16 and out.shape == (f*s, h*d)
    assert torch.isfinite(out).all()
    if anchor:
        assert not out[:s].count_nonzero() and not out[-s:].count_nonzero()
    with pytest.raises(ValueError, match="exclusive"):
        branch(xv, f, s, bounds, raw, inference=True)


def test_requested_route_rejects_cpu():
    b = BidirectionalLinearBranch(16, 1, 128, delta_rule="vdn_solve")
    b.paper_vda_enabled = True
    raw = (torch.zeros(1, 1, 128, dtype=torch.bfloat16),) * 3
    with pytest.raises(ValueError, match="CUDA BF16"):
        b(torch.zeros(1, 16), 1, 1, [(0, 0)], raw)


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("text", [False, True])
def test_virtual_prompt_branch_against_independent_fp64(anchor, text):
    f, s, h, d, hidden = 11, 7, 2, 128, 24
    branch = BidirectionalLinearBranch(hidden, h, d, delta_rule="vdn_solve").cuda()
    branch.paper_vda_enabled = True
    branch.paper_vda_chunk_size = 4
    raw = tuple(torch.randn(f*s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    beta = torch.rand(f*s, h, device="cuda", dtype=torch.bfloat16)
    gate = torch.rand(f*s, h, d, device="cuda", dtype=torch.bfloat16)
    mean = torch.randn(f, hidden, device="cuda")
    tq = tuple(torch.randn(5, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)) if text else None
    tb = torch.rand(5, h, device="cuda", dtype=torch.bfloat16) if text else None
    bounds = [(max(0, i // 4 * 4 - 4), min(f-1, i // 4 * 4 + 7)) for i in range(f)]
    out = branch(None, f, s, bounds, raw, skip_ends=anchor, beta=beta, gate=gate,
                 frame_mean=mean, text_qkv_raw=tq, text_beta=tb)
    if anchor:
        raw = tuple(x[s:-s] for x in raw)
        beta, gate, mean = beta[s:-s], gate[s:-s], mean[1:-1]
        bounds = [(lo-1, hi-1) for lo, hi in bounds[1:-1]]
        f -= 2
        out = out[s:-s]
    def prep(x, norm, frames, spatial):
        y = F.silu(x.double())
        if norm:
            y /= y.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        return y.bfloat16().double().reshape(frames, spatial, h, d).permute(0, 2, 1, 3)
    def stats(k, v, b):
        b = b.reshape(k.shape[0], k.shape[2], h).permute(0, 2, 1)
        a = (k * b.double()[..., None]).transpose(-1, -2) @ k
        vb = (v * b.double()[..., None]).bfloat16().double()
        b = (vb.transpose(-1, -2) @ k).bfloat16().double()
        inv = torch.linalg.inv(a + torch.eye(d, device="cuda", dtype=torch.float64))
        return inv, b @ inv
    q, k, v = [prep(x, i < 2, f, s) for i, x in enumerate(raw)]
    inv, inj = stats(k, v, beta)
    alpha = branch.alpha(mean).double()
    seed = None
    if text:
        _, text_inj = stats(prep(tq[1], True, 1, 5), prep(tq[2], False, 1, 5), tb)
        seed = text_inj[0] * 0.5
    prefix, suffix = scan_oracle(alpha[..., None] * inv, inj, seed)
    state = gather_oracle(prefix, suffix, alpha, bounds, seed).bfloat16().double()
    readout = (q @ state.transpose(-1, -2)).bfloat16().double()
    ref = readout * torch.rsqrt(readout.square().mean(-1, keepdim=True) + branch.norm.eps)
    ref *= branch.norm.weight.double()
    ref = (ref.permute(0, 2, 1, 3).reshape_as(gate) * gate.double()).reshape_as(out)
    # Rounded BF16 phase boundaries plus FP32-vs-FP64 sums can differ by one ulp.
    close_rel(out, ref.bfloat16(), 0.004)


@pytest.mark.parametrize("anchor", [False, True])
def test_head_shard_shortconv_prompt_nondefault_stream(anchor):
    f, s, h, d, hidden = 9, 6, 3, 128, 24
    branch = BidirectionalLinearBranch(hidden, h, d, delta_rule="vdn_solve", short_conv=("k", "v")).cuda()
    branch.paper_vda_enabled = True
    raw = tuple(torch.randn(f*s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    beta = torch.rand(f*s, h, device="cuda", dtype=torch.bfloat16)
    gate = torch.rand(f*s, h, d, device="cuda", dtype=torch.bfloat16)
    mean = torch.randn(f, hidden, device="cuda")
    tq = tuple(torch.randn(5, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    tb = torch.rand(5, h, device="cuda", dtype=torch.bfloat16)
    bounds = [(max(0, i//3*3-3), min(f-1, i//3*3+5)) for i in range(f)]
    kwargs = dict(frame_size=(2, 3), skip_ends=anchor, beta=beta, gate=gate,
                  frame_mean=mean, text_qkv_raw=tq, text_beta=tb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = branch(None, f, s, bounds, raw, **kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
        part = branch(None, f, s, bounds, tuple(x[:, 1:2] for x in raw),
                      frame_size=(2, 3), skip_ends=anchor, heads=slice(1, 2),
                      beta=beta[:, 1:2], gate=gate[:, 1:2], frame_mean=mean,
                      text_qkv_raw=tuple(x[:, 1:2] for x in tq), text_beta=tb[:, 1:2])
    torch.cuda.current_stream().wait_stream(stream)
    close_rel(part, full.view(f*s, h, d)[:, 1].reshape_as(part), 0.003)


@pytest.mark.parametrize("frames", [1, 2])
def test_anchor_only_clip_does_not_call_kernels(frames, monkeypatch):
    branch = BidirectionalLinearBranch(16, 1, 128, delta_rule="vdn_solve").cuda()
    branch.paper_vda_enabled = True
    from src.models.linear_attention import paper_stack
    def forbidden(*a, **kw):
        raise AssertionError("anchor-only clip should not read out")
    monkeypatch.setattr(paper_stack, "paper_readout", forbidden)
    raw = (torch.randn(frames, 1, 128, device="cuda", dtype=torch.bfloat16),) * 3
    out = branch(torch.zeros(frames, 16, device="cuda", dtype=torch.bfloat16), frames,
                 1, [(0, frames-1)] * frames, raw, skip_ends=True)
    assert not out.count_nonzero()


@pytest.mark.parametrize("frames,offset", [(35, 1), (37, 0), (102, 0)])
def test_actual_chunk_five_grid_and_long_recurrence(frames, offset):
    from src.models.softmax_attention.window import window_bounds
    bounds = [(lo-offset, hi-offset) for lo, hi in
              window_bounds(frames+2*offset, 1, 5)[offset:frames+offset]]
    plan = paper_boundary_plan(bounds, frames, torch.device("cuda:0"), 8)
    if frames == 35:
        assert plan.cuts == (0, 4, 9, 14, 19, 24, 29, 34, 35)
    h, d = 1, 128
    # Noncommuting transitions near identity exercise accumulation over many steps.
    k = torch.randn(frames, h, 7, d, device="cuda") * .025
    a = (k.transpose(-1, -2) @ k).contiguous()
    b = torch.randn_like(a) * .01
    alpha = torch.rand(frames, h, d, device="cuda") * .01 + .99
    t, inj = paper_delta_factors(a, b, alpha)
    seed = torch.randn_like(a[0])
    bank = paper_boundary_scan(t, inj, plan, seed)
    prefix, suffix = scan_oracle(t, inj, seed)
    torch.testing.assert_close(bank[0].double(), prefix[torch.tensor(plan.cuts[1:])-1],
                               atol=4e-6, rtol=1e-4)
    torch.testing.assert_close(bank[1].double(), suffix[list(plan.cuts[:-1])],
                               atol=4e-6, rtol=1e-4)
    close_rel(paper_gather(bank, alpha, plan, seed, torch.float32),
              gather_oracle(prefix, suffix, alpha, bounds, seed), 5e-6)


def test_correlated_joint_statistics_spd():
    s = 1008
    base = torch.randn(2, 3, 1, 128, device="cuda")
    k = F.normalize(base + torch.randn(2, 3, s, 128, device="cuda") * .01, dim=-1).bfloat16()
    v = torch.randn_like(k) * 10
    beta = torch.rand(2, 3, s, device="cuda", dtype=torch.bfloat16)
    a, b = paper_statistics(k, v, beta)
    ref = (k.double() * beta.double()[..., None]).transpose(-1, -2) @ k.double()
    close_rel(a, ref, 5e-7)
    assert torch.linalg.eigvalsh(a.double() + torch.eye(128, device="cuda")).min() > .999
    t, inj = paper_delta_factors(a, b)
    inv = torch.linalg.inv(ref + torch.eye(128, device="cuda"))
    close_rel(t, inv, 3e-5)
    close_rel(inj, b.double() @ inv, 3e-5)
