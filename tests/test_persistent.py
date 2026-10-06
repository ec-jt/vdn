"""Focused persistent ownership tests. No production model loads."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from src.config.inference import InferenceConfig
from src.inference.utils.persistent import (
    PersistentPaperRuntime, atomic_json, model_signature, validate_request,
)
from src.inference.utils.paper_runtime import PaperRuntime
from src.inference.utils.weight_sharding import ResidentWeightShards
from test_paper_stack_adaln import toy, schedule, prepare


def owner():
    cfg = OmegaConf.structured(InferenceConfig)
    cfg.render.num_frames, cfg.render.num_steps = 124, 8
    cfg.compute.paper_adaln = True
    rt = SimpleNamespace(device=torch.device("cpu"), rank=0, world_size=1, is_main=True)
    return cfg, PersistentPaperRuntime(cfg, rt)


def test_retained_sources_survive_meta_and_new_tables_match_fresh():
    cfg, paper = owner()
    model = toy()
    model.transformer_blocks = nn.ModuleList([nn.Module() for _ in model.projections])
    for block, projection in zip(model.transformer_blocks, model.projections):
        block.adaln_proj = projection
    fresh = copy.deepcopy(model)
    paper.retain_adaln_sources(model)
    model.to("meta")
    old = None
    for conditioned, seed in [(False, 42), (True, 17), (False, 42)]:
        cfg.render.seed = seed
        paper.request = paper.new_request(cfg)
        paper.schedule = schedule(steps=8, conditioned=conditioned)
        paper.prepare_adaln(paper.sources)
        view = paper.adaln.step(0, paper.schedule.timestep(0))
        expected = prepare(fresh, paper.schedule)
        for a, b in zip(view.block(0), expected.step(0, paper.schedule.timestep(0)).block(0)):
            assert torch.equal(a, b)
        assert paper.seed == seed and paper.step_index == 0
        if old is not None:
            with pytest.raises(RuntimeError, match="closed"):
                old.block(0)
        paper.close_tables()
        old = view
        expected.close()
    assert paper.sources.transformer_blocks[0].adaln_proj.linear.weight.device.type == "cpu"


def test_prompt_seed_and_model_admission(tmp_path):
    cfg, paper = owner()
    cfg.render.warmup_steps = 0
    for index in range(2):
        path = tmp_path / f"prompt{index}.pt"
        torch.save(dict(prompt_embeds=torch.full((2, 5120), float(index)),
                        text_token_tags=torch.zeros(2, dtype=torch.int64)), path)
    cfg.render.prompt_file = str(tmp_path / "prompt0.pt")
    first = validate_request(cfg, paper.signature)
    cfg.render.prompt_file = str(tmp_path / "prompt1.pt")
    cfg.render.seed = 19
    assert validate_request(cfg, paper.signature) != first
    cfg.render.num_frames = 345
    with pytest.raises(ValueError, match="124"):
        validate_request(cfg, paper.signature)
    cfg.render.num_frames = 124
    cfg.compute.paper_mxfp8 = True
    with pytest.raises(ValueError, match="signature"):
        validate_request(cfg, paper.signature)


def test_single_request_guard_remains_and_request_state_is_distinct():
    cfg, paper = owner()
    first = paper.request
    first.state = "finished"
    paper.request = paper.new_request(cfg)
    assert paper.request is not first and paper.guard is not first.guard
    assert not paper.calls and paper.step_index == 0
    assert first.state == "finished"
    legacy = PaperRuntime(cfg, paper.runtime)
    legacy.state = "finished"
    with pytest.raises(FloatingPointError, match="exactly one"):
        legacy.begin(None, 0, None, None, frames=124, steps=8, seed=42, shifts=(12., 3.))


def test_drain_does_not_release_and_transport_fault_poisons():
    _, paper = owner()
    events = []
    controller = SimpleNamespace(finish_request=lambda: events.append("drain"))
    for _ in range(20):
        paper.drain_request(controller, failed=True)
    assert events == ["drain"] * 20 and not paper.poisoned
    def fail():
        raise RuntimeError("transport")
    controller.finish_request = fail
    with pytest.raises(RuntimeError, match="transport"):
        paper.drain_request(controller, failed=True)
    assert paper.poisoned


def test_controller_finish_preserves_resources_and_rejects_active():
    c = object.__new__(ResidentWeightShards)
    c.released, c.initialized, c.active, c._in_forward, c._failed = False, True, None, False, False
    calls = []
    c.flush = lambda: calls.append("flush")
    c.weight_group = group = object()
    c.finish_request()
    assert c.weight_group is group and not c.released and c.initialized
    assert calls == ["flush"]
    c.active = 1
    with pytest.raises(RuntimeError, match="idle"):
        c.finish_request()


def test_atomic_outputs_never_overwrite(tmp_path):
    path = tmp_path / "metrics.json"
    atomic_json(path, {"first": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        atomic_json(path, {"first": False})
    assert path.read_bytes() == before


@torch.no_grad()
def test_changed_prompt_noise_schedule_matches_fresh_tiny_transformer(monkeypatch):
    from src.inference.utils import ulysses
    from src.inference.utils.paper_adaln import AdaLNSchedule
    from test_paper_stack_integration import adaln_model
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    torch.manual_seed(723)
    model = adaln_model()
    model.transformer_blocks = nn.ModuleList([
        MiniMaxH3TransformerBlock(8, 2, 4, 16, 12, 1e-6, 1e-6).bfloat16()])
    class Attention(nn.Module):
        num_heads = 2
        def forward(self, x, rotary, mask=None):
            return x * .125
    model.transformer_blocks[0].attn = Attention()
    for name in ("proj_in", "audio_proj_in", "context_embedder", "proj_out", "audio_proj_out"):
        setattr(model, name, nn.Linear(8, 8).bfloat16())
    model.token_refiner = nn.Identity()
    model.rope = lambda positions: (torch.ones(len(positions), 4), torch.zeros(len(positions), 4))
    model._ulysses_runtime = SimpleNamespace(configure=lambda *args: None, local_start=0, local_end=6,
        profile_start=lambda: None, profile_end=lambda *args: None, gather_sequence=lambda x: x)
    model.eval().requires_grad_(False)
    monkeypatch.setattr(ulysses, "iter_hybrids", lambda m: iter([m.transformer_blocks[0].attn]))
    cfg, paper = owner()
    paper.retain_adaln_sources(model)
    model._paper_runtime = paper
    previous = None
    for prompt_value, seed, times in [(0., 42, [0., .6]), (1., 18, [0., .8]), (0., 42, [0., .6])]:
        generator = torch.Generator().manual_seed(seed)
        inputs = dict(hidden_states=torch.randn(1, 2, 8, generator=generator).bfloat16(),
            audio_hidden_states=torch.randn(1, 2, 8, generator=generator).bfloat16(),
            encoder_hidden_states=torch.full((1, 2, 8), prompt_value).bfloat16(),
            timestep=torch.tensor(times), timestep_indices=torch.tensor([0,0,1,1,0,0]),
            token_tags=torch.tensor([1,1,2,2,0,0]), position_ids=torch.zeros(6,3),
            video_indices=torch.tensor([4,5]), audio_indices=torch.tensor([2,3]),
            text_indices=torch.tensor([0,1]), return_dict=False)
        fresh = copy.deepcopy(model)
        del fresh._paper_runtime
        expected = ulysses._ulysses_transformer_forward(fresh, **inputs)
        cfg.render.seed = seed
        paper.request = paper.new_request(cfg)
        paper.schedule = AdaLNSchedule([inputs["timestep"]])
        paper.steps = 1
        paper.prepare_adaln(paper.sources)
        paper.state = "active"
        actual = ulysses._ulysses_transformer_forward(model, **inputs)
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)
        if previous is not None:
            assert any(not torch.equal(a, b) for a, b in zip(actual, previous))
        previous = actual
        paper.finish()


def _admission_worker(rank, rendezvous, directory):
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    from src.inference.utils.persistent import consensus
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    runtime = SimpleNamespace(world_size=2, paper_runtime=SimpleNamespace(group=dist.group.WORLD))
    rejected = 0
    try:
        for i in range(6):
            try:
                consensus(runtime, "invalid shape", "345 frames rejected" if rank == i % 2 else None)
            except ValueError:
                rejected += 1
        consensus(runtime, "valid later request", signature=("prompt2", 17))
        Path(directory, str(rank)).write_text(str(rejected))
    finally:
        dist.destroy_process_group()


def test_asymmetric_invalid_admission_converges_and_group_remains_usable(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_admission_worker, args=(f"file://{tmp_path}/rendezvous", str(tmp_path)), nprocs=2)
    assert [(tmp_path / str(i)).read_text() for i in range(2)] == ["6", "6"]
