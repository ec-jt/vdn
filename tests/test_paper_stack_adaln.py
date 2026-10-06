"""Tiny installed-Diffusers operator oracles; never load model checkpoints.

GPU cases require PAPER_ADALN_GPU0_TESTS=1 and CUDA_VISIBLE_DEVICES=0.
Run CPU separately with CUDA_VISIBLE_DEVICES='' to avoid unrelated devices.
"""

import copy
import gc
import os
import pickle
import weakref
from contextlib import contextmanager

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3AdaLayerNormModulation,
)
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from src.inference.utils import paper_adaln as adaln


def toy(dtype=torch.bfloat16, blocks=3):
    torch.manual_seed(152)
    model = nn.Module()
    model.time_proj = Timesteps(16, flip_sin_to_cos=True, downscale_freq_shift=0)
    model.time_embedder = TimestepEmbedding(16, 32, out_dim=12)
    model.projections = nn.ModuleList([
        MiniMaxH3AdaLayerNormModulation(12, 8).to(dtype) for _ in range(blocks)])
    model.norm_out = MiniMaxH3AdaLayerNormOut(8, 12, 1e-6).to(dtype)
    return model.eval().requires_grad_(False)


def schedule(steps=4, video_shift=12, audio_shift=3, conditioned=True):
    video, audio = MiniMaxH3Scheduler(shift=video_shift), MiniMaxH3Scheduler(shift=audio_shift)
    video.set_timesteps(steps + 1, device="cpu")
    audio.set_timesteps(steps + 1, device="cpu")
    roles = ("video_text", "audio", "condition_video") if conditioned else ("video_text", "audio")
    return adaln.AdaLNSchedule.from_timesteps(video.timesteps, audio.timesteps, active_roles=roles)


def prepare(model, plan=None, **kwargs):
    return adaln.precompute_adaln(
        schedule() if plan is None else plan,
        time_proj=model.time_proj, time_embedder=model.time_embedder,
        block_projections=model.projections, norm_out=model.norm_out, **kwargs)


def assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_exact_installed_baseline_math_and_modality_order(dtype):
    model, plan = toy(dtype), schedule()
    with prepare(model, plan) as request, torch.no_grad():
        for index in range(len(plan)):
            times = plan.timestep(index)
            view = request.step(index, times)
            embedding = model.time_embedder(model.time_proj(times))
            assert_exact(view.embedding, embedding)
            for block_index, module in enumerate(model.projections):
                expected = module(embedding)
                actual = view.block(block_index)
                for a, e in zip(actual, expected):
                    assert a.dtype == dtype and not a.requires_grad
                    assert_exact(a, e)
                raw = F.linear(F.silu(embedding.float()).to(dtype),
                               module.linear.weight, module.linear.bias)
                for t in range(len(times)):
                    for modality in range(3):
                        for component in range(6):
                            start = (modality * 6 + component) * 8
                            assert_exact(actual[component][t * 3 + modality], raw[t, start:start + 8])
            indices = torch.arange(len(times)).repeat_interleave(3)
            hidden = torch.randn(1, len(indices), 8).to(dtype)
            assert_exact(view.apply_output(hidden, model.norm_out.norm, indices),
                         model.norm_out(hidden, embedding, indices))


@pytest.mark.parametrize("roles", [
    ("video_text", "audio"),
    ("video_text", "audio", "condition_video"),
    ("video_text", "audio", "condition_video", "condition_audio"),
    ("condition_video",),
    ("audio",),
])
def test_complete_schedule_matches_token_row_unique_including_collisions(roles):
    video = torch.tensor([0., .3, .999, 1.], dtype=torch.float32)
    audio = torch.tensor([0., .7, .999, 1.], dtype=torch.float32)
    plan = adaln.AdaLNSchedule.from_timesteps(video, audio, active_roles=roles)
    for index, (v, a) in enumerate(zip(video, audio)):
        row_values = {"video_text": float(v), "audio": float(a),
                      "condition_video": max(float(v), .999), "condition_audio": 1.}
        rows = torch.cat([torch.full((7,), row_values[r], dtype=torch.float32) for r in roles])
        times, timestep_indices = torch.unique(rows, sorted=True, return_inverse=True)
        assert_exact(plan.timestep(index), times)
        tags = torch.arange(len(rows)) % 3
        adaln_indices = timestep_indices * 3 + tags
        assert_exact(times[adaln_indices // 3], rows)


def test_scheduler_independent_audio_grid_and_zip_semantics():
    video = MiniMaxH3Scheduler(shift=12)
    audio = MiniMaxH3Scheduler(shift=3)
    video.set_timesteps(9, device="cpu")
    audio.set_timesteps(5, device="cpu")
    before = (video.step_index, audio.step_index)
    plan = adaln.AdaLNSchedule.from_timesteps(video.timesteps, audio.timesteps)
    assert len(plan) == 4
    assert len(plan.timestep(0)) == 1
    for i in range(len(plan)):
        assert_exact(plan.timestep(i), torch.unique(torch.stack((video.timesteps[i], audio.timesteps[i]))))
    assert (video.step_index, audio.step_index) == before


def test_repeated_and_changed_requests_never_reuse_tables():
    model = toy()
    plans = [schedule(2), schedule(4), schedule(4, audio_shift=7), schedule(4, conditioned=False)]
    for plan in plans:
        with prepare(model, plan) as first, prepare(model, plan) as second:
            for i in range(len(plan)):
                a, b = first.step(i, plan.timestep(i)), second.step(i, plan.timestep(i))
                assert_exact(a.embedding, b.embedding)
                for block in range(3):
                    for x, y in zip(a.block(block), b.block(block)):
                        assert x.data_ptr() != y.data_ptr()
                        assert_exact(x, y)
            if len(plan) == 4:
                changed = plan.timestep(1)
                changed[0] = changed[0] / 2
                with pytest.raises(ValueError, match="differs"):
                    first.step(1, changed)
    first_plan, changed_plan = plans[1:3]
    with prepare(model, first_plan) as first, prepare(model, changed_plan) as changed:
        assert not torch.equal(first.step(1, first_plan.timestep(1)).block(0)[0],
                               changed.step(1, changed_plan.timestep(1)).block(0)[0])


def test_final_merged_weights_used_and_changed_weights_invalidate_by_construction():
    model, plan = toy(), schedule()
    with prepare(model, plan) as old:
        old_value = old.step(1, plan.timestep(1)).block(0)[0].clone()
        linear = model.projections[0].linear
        # Same arithmetic/order as merge_lora_state, never unmerge/remerge here.
        a, b = torch.randn(2, 12), torch.randn(144, 2)
        delta = (b.float() @ a.float()) * .75
        linear.weight.add_(delta.to(linear.weight.dtype))
        with prepare(model, plan) as fresh:
            view = fresh.step(1, plan.timestep(1))
            assert_exact(view.block(0)[0], model.projections[0](view.embedding)[0])
            assert not torch.equal(view.block(0)[0], old_value)
        assert_exact(old.step(1, plan.timestep(1)).block(0)[0], old_value)


def test_fp32_silu_boundary_is_not_bfloat16_input_activation():
    model, plan = toy(), schedule()
    with prepare(model, plan) as request:
        view = request.step(1, plan.timestep(1))
        embedding = view.embedding
        weight = model.projections[0].linear.weight
        bias = model.projections[0].linear.bias
        correct = F.linear(F.silu(embedding.float()).to(weight.dtype), weight, bias)
        wrong = F.linear(F.silu(embedding.to(weight.dtype)), weight, bias)
        assert not torch.equal(correct, wrong)
        assert_exact(view.block(0)[0], correct.view(-1, 48).chunk(6, -1)[0])


def test_outer_autocast_does_not_change_tables_or_global_policy():
    model, plan = toy(torch.float32), schedule()
    policy = torch.get_float32_matmul_precision()
    with prepare(model, plan) as reference:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with prepare(model, plan) as actual:
                for i in range(len(plan)):
                    for a, b in zip(actual.step(i, plan.timestep(i)).block(0),
                                    reference.step(i, plan.timestep(i)).block(0)):
                        assert a.dtype == torch.float32
                        assert_exact(a, b)
    assert torch.get_float32_matmul_precision() == policy


def test_plan_snapshots_and_rejects_stale_same_shape_schedule():
    original = torch.tensor([0., .5])
    plan = adaln.AdaLNSchedule([original])
    original[1] = .6
    copy_of_times = plan.timestep(0)
    copy_of_times[1] = .7
    assert_exact(plan.timestep(0), torch.tensor([0., .5]))
    with prepare(toy(), plan) as request:
        with pytest.raises(ValueError, match="differs"):
            request.step(0, original)
        assert request.step(0, plan.timestep(0)).embedding.shape[0] == 2


@pytest.mark.parametrize("value", [torch.tensor([]), torch.tensor([.5, .5]),
    torch.tensor([.7, .2]), torch.tensor([float("nan")]), torch.tensor([1.1]),
    torch.tensor([[.1]]), torch.tensor([.1], dtype=torch.bfloat16)])
def test_invalid_schedule_fails_closed(value):
    with pytest.raises(ValueError):
        adaln.AdaLNSchedule([value])


def test_empty_schedule_and_invalid_roles_fail_closed():
    with pytest.raises(ValueError):
        adaln.AdaLNSchedule([])
    with pytest.raises(ValueError):
        adaln.AdaLNSchedule.from_timesteps(torch.tensor([0.]), torch.tensor([0.]), active_roles=("sigma",))


def test_bounded_staging_all_steps_per_projection_before_next(monkeypatch):
    model, plan = toy(blocks=7), schedule(5)
    baseline_state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    original_stage, original_linear = adaln._stage_linear, F.linear
    events, active = [], []

    @contextmanager
    def stage(linear, device):
        assert not active
        active.append(linear)
        events.append([id(linear), 0])
        try:
            with original_stage(linear, device) as tensors:
                yield tensors
        finally:
            active.pop()

    def linear(*args, **kwargs):
        if active:
            events[-1][1] += 1
        return original_linear(*args, **kwargs)

    monkeypatch.setattr(adaln, "_stage_linear", stage)
    monkeypatch.setattr(F, "linear", linear)
    with prepare(model, plan) as request:
        assert len(events) == 8 and not active
        assert all(calls == len(plan) for _, calls in events)
        assert request.stats.projection_calls == 8 * len(plan)
        assert request.stats.peak_staged_projection_bytes == (144 * 12 + 144) * 2
        before = copy.deepcopy(events)
        for i in range(len(plan)):
            view = request.step(i, plan.timestep(i))
            for j in range(7):
                view.block(j)
            view.output()
        assert events == before
        owned = sum(t.numel() * t.element_size() for t in request._schedule._timesteps)
        owned += sum(t.numel() * t.element_size() for t in request._embeddings)
        owned += sum(t.numel() * t.element_size() for block in request._blocks for t in block)
        owned += sum(t.numel() * t.element_size() for t in request._output)
        assert request.stats.table_bytes == owned
    for name, value in model.state_dict().items():
        assert_exact(value, baseline_state[name])
    assert_exact(torch.random.get_rng_state(), rng)


def test_request_does_not_retain_model_or_weights_and_close_invalidates_views():
    model, plan = toy(), schedule()
    model_ref, weight_ref = weakref.ref(model), weakref.ref(model.projections[0].linear.weight)
    request = prepare(model, plan)
    view = request.step(0, plan.timestep(0))
    table_ref = weakref.ref(request._blocks[0][0])
    del model
    gc.collect()
    assert model_ref() is None and weight_ref() is None
    request.close()
    request.close()
    gc.collect()
    assert table_ref() is None
    for access in (lambda: view.embedding, lambda: view.block(0), lambda: view.output(),
                   lambda: request.step(0, plan.timestep(0)), lambda: request.__enter__()):
        with pytest.raises(RuntimeError, match="closed"):
            access()


def test_context_exception_closes_request():
    request = prepare(toy())
    with pytest.raises(RuntimeError, match="injected"):
        with request:
            raise RuntimeError("injected")
    assert request._closed and not request._blocks


def test_budget_fails_before_projection_and_cleans_up_failure(monkeypatch):
    model, plan = toy(), schedule()
    with prepare(model, plan) as request:
        budget = request.stats.table_bytes
    with prepare(model, plan, max_table_bytes=budget):
        pass
    with pytest.raises(MemoryError):
        prepare(model, plan, max_table_bytes=budget - 1)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("must fail before projecting")
    monkeypatch.setattr(adaln, "_stage_linear", forbidden)
    with pytest.raises(MemoryError):
        prepare(model, plan, max_table_bytes=1)
    assert not calls


def test_projection_failure_releases_staging_and_partial_tables(monkeypatch):
    model, plan = toy(), schedule()
    original_stage = adaln._stage_linear
    original_project = adaln._project_schedule
    refs, active = [], []

    @contextmanager
    def stage(linear, device):
        assert not active
        active.append(linear)
        try:
            with original_stage(linear, device) as value:
                yield value
        finally:
            active.pop()

    def project(linear, *args):
        if refs:
            raise RuntimeError("injected projection failure")
        result = original_project(linear, *args)
        refs.extend(weakref.ref(t) for t in result)
        return result

    monkeypatch.setattr(adaln, "_stage_linear", stage)
    monkeypatch.setattr(adaln, "_project_schedule", project)
    with pytest.raises(RuntimeError, match="injected projection") as retained_exception:
        prepare(model, plan)
    gc.collect()
    assert retained_exception.value.__traceback__ is not None
    assert not active and all(ref() is None for ref in refs)


def test_embedding_budget_checked_before_embedding_evaluation(monkeypatch):
    model, plan = toy(), schedule()
    metadata_bytes = sum(plan.timestep(i).numel() * 4 for i in range(len(plan)))
    def forbidden(*args, **kwargs):
        raise AssertionError("embedding evaluation must not start")
    monkeypatch.setattr(adaln, "_embed_schedule", forbidden)
    with pytest.raises(MemoryError):
        prepare(model, plan, max_table_bytes=metadata_bytes + 1)


def test_inner_projection_failure_releases_tensors_with_retained_traceback(monkeypatch):
    model, plan = toy(), schedule()
    original_stage, original_linear = adaln._stage_linear, F.linear
    refs, active, calls = [], [], []

    @contextmanager
    def stage(linear, device):
        with original_stage(linear, device) as (weight, bias):
            # Independent staging simulates CPU-to-device ownership on CPU.
            weight, bias = weight.clone(), bias.clone()
            refs.extend((weakref.ref(weight), weakref.ref(bias)))
            active.append(1)
            try:
                yield weight, bias
            finally:
                active.pop()

    def linear(*args, **kwargs):
        if active:
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("inner failure")
        return original_linear(*args, **kwargs)

    monkeypatch.setattr(adaln, "_stage_linear", stage)
    monkeypatch.setattr(F, "linear", linear)
    with pytest.raises(RuntimeError, match="inner failure") as retained_exception:
        prepare(model, plan)
    gc.collect()
    assert retained_exception.value.__traceback__ is not None
    assert not active and all(ref() is None for ref in refs)


def test_reject_meta_unmerged_or_invalid_modules_and_indices():
    model = toy()
    model.projections[1].to("meta")
    with pytest.raises(ValueError, match="before shard distribution"):
        prepare(model)
    model = toy()
    model.time_embedder.to(torch.bfloat16)
    with pytest.raises(ValueError, match="remain FP32"):
        prepare(model)
    model = toy()
    model.projections[0].linear = nn.Sequential(model.projections[0].linear)
    with pytest.raises(TypeError, match="fully merged"):
        prepare(model)
    model, plan = toy(), schedule()
    with prepare(model, plan) as request:
        for index in (-1, len(plan), True):
            with pytest.raises(IndexError):
                request.step(index, plan.timestep(0))
        with pytest.raises(IndexError):
            request.step(0, plan.timestep(0)).block(-1)


def test_optional_output_and_iterator_api():
    model, plan = toy(), schedule()
    with adaln.precompute_adaln(plan, time_proj=model.time_proj, time_embedder=model.time_embedder,
                              block_projections=(p for p in model.projections)) as request:
        assert len(request.step(0, plan.timestep(0)).block(0)) == 6
        with pytest.raises(RuntimeError, match="not requested"):
            request.step(0, plan.timestep(0)).output()


def test_cpu_table_transport_roundtrip_and_explicit_move():
    model, plan = toy(), schedule()
    with prepare(model, plan) as original:
        # CPU-only payload contains request tables, not model/projection weights.
        with pickle.loads(pickle.dumps(original)) as peer:
            assert peer.to("cpu") is peer
            assert peer.stats == original.stats
            for i in range(len(plan)):
                for a, b in zip(peer.step(i, plan.timestep(i)).block(0),
                                original.step(i, plan.timestep(i)).block(0)):
                    assert_exact(a, b)
            peer.close()
            assert len(original.step(0, plan.timestep(0)).block(0)) == 6


GPU0 = (os.environ.get("PAPER_ADALN_GPU0_TESTS") == "1"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "0")


@pytest.mark.skipif(not GPU0, reason="requires explicit isolated GPU0 operator opt-in")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("table_device", ["cpu", "cuda:0"])
def test_gpu0_exact_baseline_and_bounded_allocator_lifecycle(dtype, table_device):
    assert torch.cuda.device_count() == 1
    model, plan = toy(dtype, blocks=4), schedule(5)
    baseline = copy.deepcopy(model).to("cuda:0")
    # Warm tiny libraries outside memory accounting; no transformer/model run.
    with prepare(model, plan, compute_device="cuda:0", table_device=table_device):
        pass
    torch.cuda.synchronize(0)
    start = torch.cuda.memory_allocated(0)
    torch.cuda.reset_peak_memory_stats(0)
    with prepare(model, plan, compute_device="cuda:0", table_device=table_device) as request:
        stats = request.stats
        request.to("cuda:0")
        for i in range(len(plan)):
            times = plan.timestep(i, device="cuda:0")
            view = request.step(i, times)
            embedding = baseline.time_embedder(baseline.time_proj(times))
            assert_exact(view.embedding.to("cuda:0"), embedding)
            for j, module in enumerate(baseline.projections):
                for actual, expected in zip(view.block(j), module(embedding)):
                    assert_exact(actual.to("cuda:0"), expected)
            shift, scale = view.output()
            expected = baseline.norm_out.linear(F.silu(embedding.float()).to(dtype)).chunk(2, -1)
            assert_exact(shift.to("cuda:0"), expected[0])
            assert_exact(scale.to("cuda:0"), expected[1])
        del times, view, embedding, actual, expected, shift, scale
    gc.collect()
    torch.cuda.synchronize(0)
    assert torch.cuda.memory_allocated(0) == start
    # Loose allocator/kernel-workspace allowance, not a production-size claim.
    assert torch.cuda.max_memory_allocated(0) - start < stats.table_bytes + stats.peak_staged_projection_bytes + 4 * 1024**2
    assert all(p.device.type == "cpu" for p in model.parameters())
