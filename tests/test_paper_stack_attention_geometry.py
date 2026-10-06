"""Synthetic operator smoke at recorded 124-frame geometry, never a model run."""
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from src.models.sequence_layout import SequenceLayout
from src.models.softmax_attention.paper_window import prepare_paper_window_plan, paper_window_attention
from src.models.softmax_attention.paper_qk_rope import paper_qk_rmsnorm_rope
from src.models.softmax_attention.window import window_bounds


@pytest.mark.parametrize("frames", [37, 102])
@torch.inference_mode()
def test_cuda_actual_geometry_memory_smoke(frames):
    if os.environ.get("PAPER_STACK_RUN_ATTENTION_GEOMETRY") != "1":
        pytest.skip("requires explicit serialized geometry opt-in")
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"], text=True)
    print(smi)
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.manual_seed(20261005)
    # Recorded baseline: 800 prompt rows, stereo 207 audio latents, 37x24x42
    # video patches. 102 frames is additional stress, not another recorded run.
    globals_ = 800 + 2 * 207
    s = globals_ + frames * 1008
    padding = (-s) % 8
    layout = SequenceLayout(s, globals_, frames, 1008, 24, 42, 0, 800)
    bounds = window_bounds(frames, 1, 5)
    plan = prepare_paper_window_plan(layout, bounds, "both", device="cuda")
    backing = torch.randn(s + padding, 3, 7, 128, device="cuda", dtype=torch.bfloat16)
    q, k, v = backing.unbind(1)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(s + padding, 128, device="cuda")
    cos, sin = angles.cos(), angles.sin()
    del angles
    # Warm compilation and verify a small input without allocating a global mask.
    paper_qk_rmsnorm_rope(q[:3], k[:3], weight, weight, cos=cos[:3], sin=sin[:3], enabled=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    qr, kr = paper_qk_rmsnorm_rope(q, k, weight, weight, cos=cos, sin=sin, enabled=True)
    stop.record()
    torch.cuda.synchronize()
    rope_ms = start.elapsed_time(stop)
    # Warm native operator, then measure independently of JIT and planning.
    warm = paper_window_attention(qr, kr, v, plan, enabled=True)
    del warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    attention_resident = torch.cuda.memory_allocated()
    times = []
    for _ in range(3):
        start.record()
        out, lse = paper_window_attention(qr, kr, v, plan, enabled=True, return_lse=True)
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
        if len(times) != 3:
            del out, lse
    peak = torch.cuda.max_memory_allocated()
    # Output + current gathered pass + native linear workspace, not S squared.
    assert peak - attention_resident < 700 * 1024**2
    assert torch.isfinite(out).all() and torch.isfinite(lse[:s]).all()
    assert out[s:].count_nonzero() == 0 and torch.isneginf(lse[s:]).all()
    assert plan.peak_gather_rows <= 65536
    assert len([p for p in plan.passes if not p.dense]) > 1
    # Selected actual-length FP32 oracle rows, one row per distinct pass group.
    # Each reference uses only O(H*K) memory and independently derives visibility.
    rows = sorted({0, globals_ - 1, globals_, s - 1}
                  | {globals_ + f * 1008 for f in (1, 4, 5, frames // 2, frames - 2)})
    errors, lse_errors, max_abs = [], [], []
    all_keys = torch.arange(s, device="cuda")
    key_frames = (all_keys - globals_) // 1008
    for row in rows:
        f = (row - globals_) // 1008
        if row < globals_ or f in (0, frames - 1):
            visible = torch.ones(s, device="cuda", dtype=torch.bool)
        else:
            lo, hi = bounds[f]
            visible = ((all_keys < globals_) | ((key_frames >= lo) & (key_frames <= hi))
                       | (key_frames == 0) | (key_frames == frames - 1))
        keys, vals = kr[:s][visible].float(), v[:s][visible].float()
        scores = torch.einsum("hd,khd->hk", qr[row].float(), keys) * (128 ** -.5)
        expected = torch.einsum("hk,khd->hd", scores.softmax(-1), vals)
        delta = out[row].float() - expected
        errors.append((delta.norm() / expected.norm()).item())
        max_abs.append(delta.abs().max().item())
        lse_errors.append((lse[row] - scores.logsumexp(-1)).abs().max().item())
        torch.testing.assert_close(out[row].float(), expected, atol=.009, rtol=.015)
        assert errors[-1] < .004 and lse_errors[-1] < 5e-5
    result = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  frames=frames, recorded_geometry=frames == 37, text_rows=800, audio_rows=414,
                  tokens_per_frame=1008, used_rows=s, padding_rows=padding, heads=7, head_dim=128,
                  passes=len(plan.passes), peak_gather_rows=plan.peak_gather_rows,
                  resident_before_rope_bytes=resident, attention_resident_bytes=attention_resident,
                  peak_attention_bytes=peak, attention_increment_bytes=peak - attention_resident,
                  rope_ms=rope_ms, attention_ms=times, sampled_rows=rows,
                  max_sample_relative_l2=max(errors), max_sample_abs=max(max_abs), max_sample_lse_abs=max(lse_errors),
                  global_dense_mask_bytes_avoided=s*s, nvidia_smi_before=smi)
    print(json.dumps(result, indent=2))
    destination = os.environ.get("PAPER_STACK_ATTENTION_REPORT_DIR")
    if destination:
        Path(destination, f"geometry-{frames}.json").write_text(json.dumps(result, indent=2) + "\n")
