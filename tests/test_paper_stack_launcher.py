"""CPU-only launcher contract tests. Never launch a model or query real GPUs."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_stack_launcher", ROOT / "scripts/benchmark_rtx5090_paper_stack.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def plan(*extra):
    return launcher.build_plan(launcher.parser().parse_args(["cpu-test", "plain", *extra]))


def reserve(tmp_path, selected=None):
    selected = selected or plan()
    (tmp_path / launcher.INTEGRATION).mkdir(parents=True)
    return launcher.reserve(selected, tmp_path)


def test_preset_loads_with_real_schema_and_global_defaults_stay_off(tmp_path):
    from omegaconf import OmegaConf
    from src.config.inference import (InferenceConfig, PaperComputeConfig, validate_ablation,
                                      validate_kernels, validate_parallel)
    from src.config.loader import load_config
    selected = plan()
    config = tmp_path / "selected.yaml"
    config.write_text(yaml.safe_dump(selected["selected"]))
    cfg = load_config(InferenceConfig, ["--config", str(config)],
                      extra_validators=(validate_ablation, validate_kernels, validate_parallel))
    assert OmegaConf.to_container(cfg, resolve=True) == selected["selected"]
    for component in launcher.COMPONENTS:
        assert getattr(PaperComputeConfig(), "paper_" + component) is False
        assert cfg.compute["paper_" + component] is True
    assert cfg.render.num_frames == 124 and cfg.render.num_steps == 8
    assert cfg.render.seed == 42 and cfg.render.warmup_steps == 0
    assert cfg.render.prompt_file == "prompts/example_0.pt"
    assert cfg.memory.weight_prefetch and cfg.memory.shard_distribution == "broadcast"
    assert cfg.parallel.standard_exchange == "allgather" and cfg.parallel.softmax_ranks == 0
    assert not cfg.kernels.inference_kernels and not cfg.precision.fp8.enabled
    assert not cfg.kernels.vda_fused and not cfg.kernels.vda_overlap


@pytest.mark.parametrize("component", launcher.COMPONENTS)
def test_components_independent(component):
    cfg = plan("--disable", component)["selected"]
    for name in launcher.COMPONENTS:
        assert cfg["compute"]["paper_" + name] is (name != component)
    assert cfg["memory"]["weight_prefetch"] is True
    assert cfg["ablation"] == {"enabled": False, "overrides": {}}


def test_prefetch_independent_and_all_off_allowed():
    args = [part for component in launcher.COMPONENTS for part in ("--disable", component)]
    cfg = plan("--prefetch", "disabled", *args)["selected"]
    assert not cfg["memory"]["weight_prefetch"]
    assert not any(cfg["compute"]["paper_" + component] for component in launcher.COMPONENTS)
    assert all(plan()["selected"]["compute"]["paper_" + c] for c in launcher.COMPONENTS)


@pytest.mark.parametrize("extra", [
    ["render.num_frames=345"], ["--config", "evil.yaml"], ["--", "render.seed=1"],
    ["--disable", "unknown"], ["--prefetch", "true"], ["--timeout", "12"],
    ["--render-out", "/tmp/a.mp4"], ["compute.paper_vda=true"], ["--skip-gpu-check"],
    ["--min-free-gib", "0"], ["--export", "sqlite"],
])
def test_arbitrary_overrides_rejected(extra):
    with pytest.raises(SystemExit):
        plan(*extra)


@pytest.mark.parametrize("label", ["../escape", "/absolute", "a/b", "a.b", "a b", "-x", "a" * 62,
                                   "LAUNCHER_REQUIRED"])
def test_unsafe_labels_rejected(label):
    args = launcher.parser().parse_args(["cpu-test", "plain"])
    args.label = label
    with pytest.raises(ValueError):
        launcher.build_plan(args)


@pytest.mark.parametrize("seconds", ["0", "-1", "1801"])
def test_timeout_bounded(seconds):
    with pytest.raises(ValueError):
        plan("--timeout-seconds", seconds)


def test_mutated_preset_cannot_bypass_fixed_protocol(tmp_path):
    file = tmp_path / launcher.PRESET
    file.parent.mkdir(parents=True)
    cfg = launcher.expected_preset()
    cfg["render"]["num_frames"] = 345
    file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="fixed research protocol"):
        launcher.build_plan(launcher.parser().parse_args(["cpu-test", "plain"]), tmp_path)


def test_command_profile_and_plain():
    plain = plan()
    profile = launcher.build_plan(launcher.parser().parse_args(["cpu-test", "profile"]))
    alias = launcher.build_plan(launcher.parser().parse_args(["cpu-test", "nsys"]))
    assert profile == alias
    assert plain["command"][:5] == [".venv/bin/python", "scripts/supervise_rtx5090.py",
                                    "--seconds", "900", "--"]
    assert "--nproc_per_node=8" in plain["command"]
    assert "--max-restarts=0" in plain["command"]
    assert "--no-python" not in plain["command"]
    assert "scripts/nsys_rtx5090_rank.sh" in profile["command"]
    assert profile["environment"]["VDN_NSYS_CAPTURE"] == "1"
    assert plain["environment"]["VDN_NVTX"] == "0"
    assert plain["command"][-2:] == ["--config", plain["paths"]["selected_config"]]
    wrapper = (ROOT / "scripts/nsys_rtx5090_rank.sh").read_text()
    assert "--capture-range=cudaProfilerApi" in wrapper
    assert "--capture-range-end=repeat" in wrapper
    assert "--force-overwrite" not in wrapper and "nsys export" not in wrapper


def test_sanitized_environment_preserves_existing_caches():
    inherited = {"FA_CLC": "1", "PYTHONPATH": "evil", "NCCL_P2P_DISABLE": "1",
                 "VDN_NSYS_CAPTURE": "1", "HF_HOME": "/existing/cache", "PATH": "/bin"}
    env = launcher.child_environment(plan(), inherited)
    assert "FA_CLC" not in env and "PYTHONPATH" not in env and "NCCL_P2P_DISABLE" not in env
    assert env["HF_HOME"] == "/existing/cache"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["VDN_NSYS_CAPTURE"] == "0"
    assert env["NCCL_P2P_LEVEL"] == "SYS" and env["NCCL_SHM_USE_CUDA_MEMCPY"] == "0"


def gpu_csv():
    return "\n".join(f"{i}, NVIDIA GeForce RTX 5090, 1, 0" for i in range(8))


def test_gpu_idle_admission():
    launcher.check_gpu_inventory(gpu_csv(), "\n")


@pytest.mark.parametrize("gpu,apps", [
    (gpu_csv(), "1234, GPU-uuid"), (gpu_csv().replace(", 1, 0", ", 1000, 0"), ""),
    (gpu_csv().replace(", 1, 0", ", 1, 1"), ""), (gpu_csv().splitlines()[0], ""),
    (gpu_csv().replace("RTX 5090", "A100"), ""),
    (gpu_csv().replace("7, NVIDIA", "6, NVIDIA"), ""),
    (gpu_csv().replace(", 1, 0", ", N/A, 0"), ""),
])
def test_gpu_admission_fail_closed(gpu, apps):
    with pytest.raises(ValueError):
        launcher.check_gpu_inventory(gpu, apps)


def test_gpu_queries_bounded_and_read_only(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=gpu_csv() if len(calls) == 1 else "")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.gpu_preflight()
    assert len(calls) == 2
    assert all(c[0][0] == "nvidia-smi" and c[1]["timeout"] == 10 for c in calls)


def test_disk_preflight_current_22gib():
    launcher.check_space(plan(), 22 * launcher.GIB)
    profile = launcher.build_plan(launcher.parser().parse_args(["cpu-test", "profile"]))
    with pytest.raises(ValueError, match="72 GiB"):
        launcher.check_space(profile, 22 * launcher.GIB)
    with pytest.raises(ValueError):
        launcher.check_space(plan(), 16 * launcher.GIB - 1)


def test_execute_low_disk_refuses_before_gpu_query_or_reservation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    selected = plan()
    (tmp_path / launcher.INTEGRATION).mkdir(parents=True)
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(launcher, "gpu_preflight", lambda: pytest.fail("GPU query was not expected"))
    with pytest.raises(ValueError, match="insufficient disk"):
        launcher.execute(selected, tmp_path)
    assert not (tmp_path / selected["paths"]["run"]).exists()


def test_execute_lock_refuses_concurrent_launcher(tmp_path, monkeypatch):
    import fcntl
    parent = tmp_path / launcher.INTEGRATION
    parent.mkdir(parents=True)
    monkeypatch.setattr(launcher, "gpu_preflight", lambda: pytest.fail("unexpected GPU query"))
    with (parent / ".paper-stack-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            launcher.execute(plan(), tmp_path)


def test_reservation_does_not_create_renderer_targets_and_never_reuses(tmp_path):
    selected = plan()
    run = reserve(tmp_path, selected)
    assert run.stat().st_mode & 0o777 == 0o700
    for name in ("media", "partial", "metrics", "latents"):
        assert not (tmp_path / selected["paths"][name]).exists()
    sentinel = run / "existing.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        launcher.reserve(selected, tmp_path)
    assert sentinel.read_text() == "preserve"
    with pytest.raises(FileExistsError):
        launcher.write_new(sentinel, "overwrite")


def test_dangling_symlink_label_refused(tmp_path):
    selected = plan()
    parent = tmp_path / launcher.INTEGRATION
    parent.mkdir(parents=True)
    (tmp_path / selected["paths"]["run"]).symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        launcher.reserve(selected, tmp_path)


def test_latent_reservation_matches_real_renderer_resolver():
    from omegaconf import OmegaConf
    from src.inference.utils.assemble import latents_path
    selected = plan()
    assert latents_path(OmegaConf.create(selected["selected"])) == selected["paths"]["latents"]
    assert all(Path(value).is_relative_to(launcher.INTEGRATION) for value in selected["paths"].values())


def test_snapshot_exact_config_hashes_and_no_overwrite(tmp_path):
    selected = plan()
    # Snapshot fixture contains no checkpoints or large source tree.
    for relative in (launcher.PRESET, Path("tests/test_paper_stack_launcher.py"),
                     Path("pyproject.toml"), Path("prompts/example_0.pt")):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes() if relative == launcher.PRESET else b"fixture")
    run = reserve(tmp_path, selected)
    launcher.snapshot(selected, run, tmp_path)
    hashes = json.loads((run / "source-sha256.json").read_text())
    assert hashes[str(launcher.PRESET)] == selected["preset_sha256"]
    assert yaml.safe_load((tmp_path / selected["paths"]["selected_config"]).read_text()) == selected["selected"]
    assert not (run / "actual-runtime-manifest.json").exists()
    with pytest.raises(FileExistsError):
        launcher.snapshot(selected, run, tmp_path)


def evidence(tmp_path, selected, run):
    ranks = [dict(rank=i, selected=selected["selected"]["compute"], called={"actual_call": 1},
                  finite_failed=False, completed_forwards=8) for i in range(8)]
    record = dict(overlay=selected["selected"], timings=dict(
        ranks=[dict(rank=i, paper_stack=r) for i, r in enumerate(ranks)],
        decoder_metrics=dict(distributed_video=selected["decoder_expected"])))
    launcher.write_json(tmp_path / selected["paths"]["metrics"], record)
    for key in ("media", "latents"):
        (tmp_path / selected["paths"][key]).write_bytes(b"CPU fixture")
    return record


def result(code=0):
    return dict(exit_code=code, stop_reason=None, wall_seconds=1, output_bytes=0)


def test_selected_is_not_actual_on_failure(tmp_path):
    selected = plan()
    run = reserve(tmp_path, selected)
    (run / "launch.log").write_text('paper_stack_inventory={"rank": 2, "called": {"x": 1}}\n')
    actual = launcher.actual_manifest(selected, run, result(1), tmp_path)
    assert actual["ranks"] == [] and actual["missing_ranks"] == list(range(8))
    assert actual["terminal_inventories"] == [{"rank": 2, "called": {"x": 1}}]
    assert not actual["launcher_success"]


def test_actual_manifest_checks_decoder_and_all_ranks(tmp_path):
    selected = plan()
    run = reserve(tmp_path, selected)
    record = evidence(tmp_path, selected, run)
    actual = launcher.actual_manifest(selected, run, result(), tmp_path)
    assert actual["launcher_success"] and actual["missing_ranks"] == []
    assert actual["promotion"] is False
    # Corruption of this synthetic runtime fixture must invalidate evidence.
    record["timings"]["decoder_metrics"]["distributed_video"] = {"total_jobs": 999}
    (tmp_path / selected["paths"]["metrics"]).write_text(json.dumps(record))
    actual = launcher.actual_manifest(selected, run, result(), tmp_path)
    assert not actual["launcher_success"]
    assert "decoder mismatch: total_jobs" in actual["evidence_errors"]


def test_dry_run_no_cuda_import_or_side_effects():
    code = """
import importlib.abc, runpy, sys
class RejectCUDA(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'diffusers'):
            raise RuntimeError('dry-run imported CUDA stack: ' + fullname)
sys.meta_path.insert(0, RejectCUDA())
sys.argv = ['launcher', 'cpu-dry-run-never-reserved', 'profile', '--dry-run']
runpy.run_path('scripts/benchmark_rtx5090_paper_stack.py', run_name='__main__')
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                               text=True, timeout=15, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert completed.returncode == 0, completed.stderr
    selected = json.loads(completed.stdout)
    assert not (ROOT / selected["paths"]["run"]).exists()


def test_supervisor_timeout_owns_detached_descendants(tmp_path):
    # Only synthetic sleeping Python children, including a new-session grandchild.
    pidfile = tmp_path / "owned-pid.txt"
    child_code = ("import subprocess,sys,time; "
                  "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
                  "start_new_session=True); "
                  f"open({str(pidfile)!r},'w').write(str(p.pid)); time.sleep(60)")
    command = [sys.executable, str(ROOT / "scripts/supervise_rtx5090.py"), "--seconds", "1", "--",
               sys.executable, "-c", child_code]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=25)
    assert completed.returncode == 124, completed.stderr
    pid = int(pidfile.read_text())
    if Path(f"/proc/{pid}/stat").exists():
        assert Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_disk_guard_signals_only_owned_supervisor(tmp_path, monkeypatch):
    selected = plan()
    run = reserve(tmp_path, selected)
    sent = []

    class Child:
        pid = 123456789
        stopped = False

        def poll(self):
            return 143 if self.stopped else None

        def send_signal(self, sig):
            sent.append(sig)
            self.stopped = True

        def wait(self, timeout):
            assert timeout <= 1830
            return 143

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(launcher, "output_bytes", lambda run: selected["output_budget_bytes"])
    actual = launcher.supervise_with_disk_guard(selected, run, tmp_path)
    assert sent == [signal.SIGTERM]
    assert actual["stop_reason"] == "output budget exceeded"


def test_shell_syntax():
    subprocess.run(["bash", "-n", "scripts/benchmark_rtx5090_paper_stack.sh"], cwd=ROOT,
                   check=True, timeout=5)
