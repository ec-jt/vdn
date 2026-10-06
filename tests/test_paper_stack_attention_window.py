"""Independent dense FP32 oracles. CUDA execution requires explicit opt-in."""
import inspect
import os
import subprocess

import pytest
import torch

from src.models.sequence_layout import SequenceLayout
from src.models.softmax_attention.paper_window import prepare_paper_window_plan, paper_window_attention
from src.models.softmax_attention.window import window_bounds


def dense_mask(layout, bounds, anchors):
    # Intentionally independent from range union / planner helpers.
    s, vs, ve = layout.seq_len, layout.video_start, layout.video_end
    mask = torch.zeros(s, s, dtype=torch.bool)
    for q in range(s):
        for k in range(s):
            if q < vs or q >= ve or k < vs or k >= ve:
                mask[q, k] = True
            else:
                qf, kf = (q - vs) // layout.tokens_per_frame, (k - vs) // layout.tokens_per_frame
                mask[q, k] = (bounds[qf][0] <= kf <= bounds[qf][1]
                              or anchors in ("rows", "both") and qf in (0, layout.num_frames - 1)
                              or anchors in ("columns", "both") and kf in (0, layout.num_frames - 1))
    return mask


def expand(ranges):
    return [i for a, b in ranges for i in range(a, b)]


LAYOUTS = [(0, 0), (3, 0), (0, 4), (3, 4)]


@pytest.mark.parametrize("before,after", LAYOUTS)
@pytest.mark.parametrize("frames,chunk,radius", [(1, 0, 0), (2, 5, 1), (7, 3, 0), (7, 3, 1), (7, 0, 1), (7, 5, 1), (7, 0, 99)])
@pytest.mark.parametrize("anchor", ["none", "rows", "columns", "both"])
def test_plan_exact_union_counts_order_and_budget(before, after, frames, chunk, radius, anchor):
    layout = SequenceLayout(before + frames * 3 + after, before, frames, 3,
                            text_start=0 if before else frames * 3, text_len=before or after)
    bounds = window_bounds(frames, radius, chunk)
    plan = prepare_paper_window_plan(layout, bounds, anchor, max_gather_rows=2 * layout.seq_len, max_query_rows=5)
    expected = dense_mask(layout, bounds, anchor)
    actual = torch.zeros_like(expected)
    seen = []
    for p in plan.passes:
        assert p.query_rows.numel() <= 5
        assert p.dense or p.key_rows.numel() <= 2 * layout.seq_len
        qs, ks = [], []
        for g in p.groups:
            qi, ki = expand(g.query_ranges), expand(g.key_ranges)
            assert ki == sorted(set(ki))
            qs.extend(qi)
            ks.extend(ki)
            for q in qi:
                seen.append(q)
                actual[q, ki] = True
                assert len(ki) == expected[q].sum().item()
        assert qs == p.query_rows.tolist()
        if not p.dense:
            assert ks == p.key_rows.tolist()
        assert p.cu_q[-1] == len(qs) and p.cu_k[-1] == len(ks)
    assert sorted(seen) == list(range(layout.seq_len))
    assert torch.equal(actual, expected)


def test_clamped_visible_sets_grouped_not_only_raw_bounds():
    layout = SequenceLayout(8, 0, 4, 2)
    plan = prepare_paper_window_plan(layout, [(-3, 1), (-2, 1), (2, 9), (2, 8)])
    assert len(plan.passes) == 1 and len(plan.passes[0].groups) == 2


def test_hard_budget_empty_keys_and_global_only():
    layout = SequenceLayout(21, 0, 7, 3)
    with pytest.raises(ValueError, match="requires 9 gather rows"):
        prepare_paper_window_plan(layout, window_bounds(7, 1), max_gather_rows=8)
    with pytest.raises(ValueError, match="no visible keys"):
        prepare_paper_window_plan(SequenceLayout(3, 0, 1, 3), [(4, 5)])
    # Global-only layout and full cover do not gather K/V, even with tiny budget.
    plan = prepare_paper_window_plan(SequenceLayout(9, 4, 0, 3), [], max_gather_rows=1, max_query_rows=2)
    assert len(plan.passes) == 5 and plan.peak_gather_rows == 0
    plan = prepare_paper_window_plan(layout, window_bounds(7, 99), max_gather_rows=1)
    assert all(p.dense for p in plan.passes)


@pytest.mark.parametrize("kwargs", [{"anchor_frames": "bad"}, {"max_gather_rows": 0}, {"max_query_rows": -1}])
def test_invalid_plan_options(kwargs):
    with pytest.raises(ValueError):
        prepare_paper_window_plan(SequenceLayout(6, 0, 2, 3), [(0, 1)] * 2, **kwargs)


@pytest.mark.parametrize("bounds", [[], [(0, 1)], [(2, 1), (0, 1)], [(0., 1), (0, 1)]])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        prepare_paper_window_plan(SequenceLayout(6, 0, 2, 3), bounds)


def test_default_off_no_implicit_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled API must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    assert inspect.signature(paper_window_attention).parameters["enabled"].default is False
    with pytest.raises(RuntimeError, match="disabled"):
        paper_window_attention(None, None, None, None)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        paper_window_attention(None, None, None, None, enabled=True)


@pytest.fixture
def cuda_device():
    if os.environ.get("PAPER_STACK_RUN_ATTENTION_CUDA") != "1":
        pytest.skip("requires serialized parent opt-in PAPER_STACK_RUN_ATTENTION_CUDA=1")
    # Observe other services before EACH test. Never terminate any process.
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv"], check=True)
    assert torch.cuda.get_device_capability() == (12, 0)
    return torch.device("cuda", torch.cuda.current_device())


def oracle(q, k, v, mask, scale):
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    scores.masked_fill_(~mask.to(q.device)[None], -float("inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float()), scores.logsumexp(-1).T


@pytest.mark.parametrize("before,after", LAYOUTS)
@pytest.mark.parametrize("frames,chunk,radius", [(1, 0, 0), (2, 5, 1), (7, 3, 0), (7, 3, 1), (7, 0, 1), (7, 5, 1), (7, 0, 99)])
@pytest.mark.parametrize("anchor", ["none", "rows", "columns", "both"])
@torch.no_grad()
def test_cuda_dense_fp32_oracle(cuda_device, before, after, frames, chunk, radius, anchor):
    torch.manual_seed(819)
    layout = SequenceLayout(before + frames * 3 + after, before, frames, 3)
    bounds = window_bounds(frames, radius, chunk)
    # Fused QKV and rank head slices are noncontiguous along token and head.
    backing = torch.randn(layout.seq_len + 5, 3, 14, 128, device=cuda_device, dtype=torch.bfloat16)
    q, k, v = [backing[:, i, 3:10] for i in range(3)]
    old = backing.clone()
    plan = prepare_paper_window_plan(layout, bounds, anchor, device=cuda_device,
                                    max_query_rows=5, max_gather_rows=2 * layout.seq_len)
    out, lse = paper_window_attention(q, k, v, plan, enabled=True, return_lse=True)
    expected, el = oracle(q[:layout.seq_len], k[:layout.seq_len], v[:layout.seq_len],
                          dense_mask(layout, bounds, anchor), 128 ** -.5)
    torch.testing.assert_close(out[:layout.seq_len].float(), expected, atol=.009, rtol=.015)
    rel = (out[:layout.seq_len].float() - expected).norm() / expected.norm()
    assert rel < .004
    torch.testing.assert_close(lse[:layout.seq_len], el, atol=3e-5, rtol=3e-6)
    assert out[layout.seq_len:].count_nonzero() == 0
    assert torch.isneginf(lse[layout.seq_len:]).all()
    assert torch.equal(backing, old)
    assert out.is_contiguous() and out.data_ptr() != backing.data_ptr()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("d", [64, 128, 256])
@torch.no_grad()
def test_cuda_pass_policy_and_native_dtype_width(cuda_device, dtype, d):
    torch.manual_seed(273)
    layout = SequenceLayout(137, 5, 17, 7)
    bounds = window_bounds(17, 1, 3)
    q, k, v = [torch.randn(137, 7, d, device=cuda_device, dtype=dtype) for _ in range(3)]
    small = prepare_paper_window_plan(layout, bounds, "both", device=cuda_device, max_query_rows=17, max_gather_rows=137)
    large = prepare_paper_window_plan(layout, bounds, "both", device=cuda_device, max_query_rows=1000, max_gather_rows=10000)
    a = paper_window_attention(q, k, v, small, enabled=True, scale=.17)
    b = paper_window_attention(q, k, v, large, enabled=True, scale=.17)
    expected, _ = oracle(q, k, v, dense_mask(layout, bounds, "both"), .17)
    torch.testing.assert_close(a.float(), expected, atol=.016 if dtype == torch.bfloat16 else .002, rtol=.015)
    torch.testing.assert_close(a, b, atol=.008 if dtype == torch.bfloat16 else .001, rtol=.01)


@torch.no_grad()
def test_cuda_constant_values_anchor_union_mass(cuda_device):
    layout = SequenceLayout(45, 2, 13, 3)
    bounds = window_bounds(13, 1, 3)
    q = torch.zeros(45, 7, 128, device=cuda_device, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.arange(45, device=cuda_device, dtype=torch.float32)[:, None, None].expand_as(q).to(q.dtype)
    p = prepare_paper_window_plan(layout, bounds, "columns", device=cuda_device, max_gather_rows=70)
    out, lse = paper_window_attention(q, k, v, p, enabled=True, return_lse=True)
    mask = dense_mask(layout, bounds, "columns").to(cuda_device)
    expected = (mask.float() @ torch.arange(45, device=cuda_device).float()) / mask.sum(-1)
    torch.testing.assert_close(out[:, 0, 0], expected.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(lse[:, 0], mask.sum(-1).float().log(), rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_cuda_global_only_and_custom_stream(cuda_device):
    layout = SequenceLayout(13, 7, 0, 1)
    plan = prepare_paper_window_plan(layout, [], device=cuda_device, max_gather_rows=1, max_query_rows=4)
    q, k, v = [torch.randn(15, 7, 128, device=cuda_device, dtype=torch.bfloat16) for _ in range(3)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = paper_window_attention(q, k, v, plan, enabled=True)
    torch.cuda.current_stream().wait_stream(stream)
    expected, _ = oracle(q[:13], k[:13], v[:13], torch.ones(13, 13, dtype=torch.bool), 128 ** -.5)
    torch.testing.assert_close(out[:13].float(), expected, rtol=.015, atol=.009)
    assert out[13:].count_nonzero() == 0


@torch.no_grad()
def test_cuda_reject_bad_runtime_contract(cuda_device):
    p = prepare_paper_window_plan(SequenceLayout(7, 1, 2, 3), [(0, 1)] * 2, device=cuda_device)
    q = torch.randn(7, 7, 128, device=cuda_device, dtype=torch.bfloat16)
    for args in ((q[:6],) * 3, (q, q, q[:6]), (q.float(),) * 3, (q[..., ::2],) * 3):
        with pytest.raises(ValueError):
            paper_window_attention(*args, p, enabled=True)
    with pytest.raises(ValueError, match="finite"):
        paper_window_attention(q, q, q, p, scale=float("nan"), enabled=True)
    cp = prepare_paper_window_plan(p.layout, p.bounds)
    with pytest.raises(ValueError, match="same CUDA"):
        paper_window_attention(q, q, q, cp, enabled=True)
