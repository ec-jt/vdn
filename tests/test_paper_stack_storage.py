"""CPU-only external-storage contracts. No real mount, model or GPU operations."""
import importlib.util
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_stack_storage_launcher", ROOT / "scripts/benchmark_rtx5090_paper_stack.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def plan(*extra, root=ROOT):
    return launcher.build_plan(launcher.parser().parse_args(["storage-cpu", "plain", *extra]), root)


@pytest.fixture
def external(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    for relative in (launcher.PRESET, Path("tests/test_paper_stack_launcher.py"),
                     Path("tests/test_paper_stack_storage.py"), Path("pyproject.toml"),
                     Path("prompts/example_0.pt")):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes() if relative == launcher.PRESET else b"source fixture")
    (source / launcher.INTEGRATION).mkdir(parents=True)
    storage = tmp_path / "external"
    storage.mkdir(mode=0o700)
    # CPU tests need no second disk. Only the filesystem admission seam is fake;
    # directory traversal, exclusive creation, receipts and watchdog are real.
    def validate(path, root=source):
        with launcher.trusted_directory(path, private=True) as fd:
            info = os.fstat(fd)
            return dict(root=str(path), device=info.st_dev, inode=info.st_ino,
                        root_receipt_reserve_bytes=launcher.ROOT_RECEIPT_RESERVE,
                        mount={"filesystem": "cpu-fixture"})
    monkeypatch.setattr(launcher, "validate_storage", validate)
    selected = plan("--storage-root", str(storage), root=source)
    return source, storage, selected


@pytest.mark.parametrize("mode,budget", [("plain", 8), ("profile", 64), ("nsys", 64)])
def test_default_thresholds_unchanged(mode, budget):
    selected = launcher.build_plan(launcher.parser().parse_args(["default", mode]))
    assert "storage" not in selected and "receipt" not in selected["paths"]
    assert selected["output_budget_bytes"] == budget * launcher.GIB
    assert selected["min_free_bytes"] == (budget + 8) * launcher.GIB
    assert not Path(selected["paths"]["run"]).is_absolute()
    launcher.check_space(selected, (budget + 8) * launcher.GIB)
    with pytest.raises(ValueError):
        launcher.check_space(selected, (budget + 8) * launcher.GIB - 1)


def test_external_plan_only_paths_and_environment_change(external):
    source, storage, selected = external
    original = plan(root=source)
    cfg = json.loads(json.dumps(selected["selected"]))
    cfg["render"]["out"] = original["selected"]["render"]["out"]
    cfg["render"]["latents_root"] = original["selected"]["render"]["latents_root"]
    assert cfg == original["selected"]
    assert selected["output_budget_bytes"] == 8 * launcher.GIB
    assert selected["min_free_bytes"] == 16 * launcher.GIB
    assert selected["storage"]["root_receipt_reserve_bytes"] == 256 * 1024 ** 2
    for key, value in selected["paths"].items():
        assert Path(value).is_absolute()
        assert Path(value).is_relative_to(source if key == "receipt" else storage)
    assert not Path(selected["paths"]["run"]).exists()
    assert not Path(selected["paths"]["receipt"]).exists()


def test_all_cache_and_tmp_destinations_override_inherited(external):
    _, _, selected = external
    routes = launcher.storage_environment(Path(selected["paths"]["run"]))
    inherited = {name: "/old/do-not-touch" for name in routes}
    inherited.update(PATH="/bin", PYTHONPATH="/evil", VDN_TRACE_PATH="/evil")
    env = launcher.child_environment(selected, inherited)
    assert len(routes) >= 25
    assert all(env[k] == v for k, v in routes.items())
    assert all(Path(v).is_relative_to(selected["paths"]["run"]) for v in routes.values())
    assert "PYTHONPATH" not in env and "VDN_TRACE_PATH" not in env
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("value", ["relative", "../escape", "/mnt/../escape"])
def test_relative_and_traversal_rejected(value):
    with pytest.raises(ValueError, match="absolute"):
        with launcher.trusted_directory(value, private=True):
            pytest.fail("unsafe path accepted")


@pytest.mark.parametrize("kind", ["leaf", "ancestor", "dangling", "missing", "file"])
def test_symlinks_and_non_directories_rejected(tmp_path, kind):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    path = tmp_path / "candidate"
    if kind == "file":
        path.write_text("preserve")
    elif kind != "missing":
        path.symlink_to(target if kind != "dangling" else tmp_path / "absent")
    if kind == "ancestor":
        (target / "child").mkdir(mode=0o700)
        path = path / "child"
    with pytest.raises(OSError):
        with launcher.trusted_directory(path, private=True):
            pytest.fail("unsafe path accepted")


@pytest.mark.parametrize("permissions", [0o755, 0o750, 0o770, 0o777, 0o500])
def test_private_writable_leaf_required(tmp_path, permissions):
    leaf = tmp_path / "leaf"
    leaf.mkdir(mode=permissions)
    with pytest.raises(ValueError):
        with launcher.trusted_directory(leaf, private=True):
            pytest.fail("unsafe permissions accepted")


def test_shared_writable_ancestor_rejected(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o700)
    parent.chmod(0o777)
    leaf = parent / "leaf"
    leaf.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="shared"):
        with launcher.trusted_directory(leaf, private=True):
            pytest.fail("shared ancestor accepted")


def test_root_owned_trusted_ancestor_but_not_leaf(tmp_path, monkeypatch):
    leaf = tmp_path / "leaf"
    leaf.mkdir(mode=0o700)
    original = launcher.os.fstat
    inode = leaf.stat().st_ino
    def fake(fd):
        info = original(fd)
        return SimpleNamespace(st_uid=os.geteuid() if info.st_ino == inode else 0,
                               st_mode=info.st_mode, st_gid=info.st_gid)
    monkeypatch.setattr(launcher.os, "fstat", fake)
    with launcher.trusted_directory(leaf, private=True):
        pass
    monkeypatch.setattr(launcher.os, "fstat", lambda fd: SimpleNamespace(st_uid=1234567, st_mode=0o40700))
    with pytest.raises(ValueError, match="owner"):
        with launcher.trusted_directory(leaf, private=True):
            pass


@pytest.mark.parametrize("members,primary_uids,expected", [([], [1000], True),
    (["caller"], [1000], True), (["other"], [1000], False), ([], [1000, 1001], False)])
def test_private_group_requires_only_caller(monkeypatch, members, primary_uids, expected):
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(launcher.pwd, "getpwuid", lambda uid: SimpleNamespace(
        pw_name="caller", pw_uid=uid, pw_gid=1000))
    monkeypatch.setattr(launcher.grp, "getgrgid", lambda gid: SimpleNamespace(gr_mem=members))
    monkeypatch.setattr(launcher.pwd, "getpwall", lambda: [SimpleNamespace(
        pw_uid=uid, pw_gid=1000) for uid in primary_uids])
    assert launcher.private_group(1000) is expected
    assert not launcher.private_group(999)


@pytest.mark.parametrize("filesystem", ["nfs", "nfs4", "cifs", "fuse.hf", "fuse.sshfs", "tmpfs", "overlay", "unknown"])
def test_nonlocal_or_nonpersistent_filesystem_rejected(tmp_path, monkeypatch, filesystem):
    leaf = tmp_path / "private"
    leaf.mkdir(mode=0o700)
    monkeypatch.setattr(launcher, "mount_record", lambda p: dict(filesystem=filesystem, options=["rw"]))
    with pytest.raises(ValueError, match="LOCAL"):
        launcher.validate_storage(leaf)


def test_mountinfo_longest_match_and_escapes(monkeypatch):
    text = ("1 0 8:1 / / rw - ext4 /dev/root rw\n"
            "2 1 8:2 / /mnt/local\\040disk rw - xfs /dev/nvme0n1 rw\n"
            "3 2 0:9 / /mnt/local\\040disk/remote rw - fuse.hf hf rw\n")
    monkeypatch.setattr(launcher.Path, "read_text", lambda self: text)
    assert launcher.mount_record(Path("/mnt/local disk/owned"))["filesystem"] == "xfs"
    assert launcher.mount_record(Path("/mnt/local disk/remote/owned"))["filesystem"] == "fuse.hf"


@pytest.mark.parametrize("failure", ["same_device", "readonly", "mount_race", "models"])
def test_storage_device_model_and_mount_guards(tmp_path, monkeypatch, failure):
    leaf = tmp_path / "private"
    leaf.mkdir(mode=0o700)
    info = leaf.stat()
    mount = dict(filesystem="ext4", options=["rw"], device=f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}")
    if failure == "readonly":
        mount["options"] = ["ro"]
    if failure == "mount_race":
        mount["device"] = "999:999"
    class Source:
        def stat(self):
            return SimpleNamespace(st_dev=info.st_dev if failure == "same_device" else info.st_dev + 1)
        def __truediv__(self, path):
            return leaf if failure == "models" else tmp_path / "immutable-models"
    monkeypatch.setattr(launcher, "mount_record", lambda p: mount)
    with pytest.raises(ValueError):
        launcher.validate_storage(leaf, Source())


def test_distinct_local_owned_directory_accepted(tmp_path, monkeypatch):
    leaf = tmp_path / "owned"
    leaf.mkdir(mode=0o700)
    info = leaf.stat()
    class Source:
        def stat(self):
            return SimpleNamespace(st_dev=info.st_dev + 1)
        def __truediv__(self, path):
            return tmp_path / "models"
    monkeypatch.setattr(launcher, "mount_record", lambda p: dict(
        filesystem="xfs", options=["rw"], device=f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}"))
    assert launcher.validate_storage(leaf, Source())["inode"] == info.st_ino


@pytest.mark.parametrize("key,kind", [(k, v) for k in ("run", "receipt") for v in ("directory", "file", "symlink")])
def test_both_namespaces_exclusive(external, key, kind):
    source, _, selected = external
    path = Path(selected["paths"][key])
    if kind == "directory":
        path.mkdir(mode=0o700)
    elif kind == "file":
        path.write_text("preserve")
    else:
        path.symlink_to(path.parent / "absent")
    with pytest.raises(FileExistsError):
        launcher.reserve(selected, source)
    assert not Path(selected["paths"]["receipt" if key == "run" else "run"]).exists()
    if kind == "file":
        assert path.read_text() == "preserve"


def test_reserve_snapshot_and_no_overwrite(external):
    source, storage, selected = external
    untouched = storage / "unrelated"
    untouched.write_text("leave alone")
    run = launcher.reserve(selected, source)
    assert run.stat().st_mode & 0o777 == 0o700
    receipt = Path(selected["paths"]["receipt"])
    assert not receipt.is_symlink()
    pointer = json.loads((receipt / "storage-pointer.json").read_text())
    assert pointer["run"] == str(run)
    for key in ("media", "partial", "metrics", "latents"):
        assert not Path(selected["paths"][key]).exists()
    launcher.snapshot(selected, run, source)
    assert (run / "snapshot/tests/test_paper_stack_storage.py").read_bytes() == b"source fixture"
    assert yaml.safe_load((run / "selected-config.yaml").read_text()) == selected["selected"]
    assert (receipt / "selected-config.yaml").read_bytes() == (run / "selected-config.yaml").read_bytes()
    assert (receipt / "command.txt").read_bytes() == (run / "command.txt").read_bytes()
    assert launcher.output_bytes(receipt) < 128 * 1024
    assert untouched.read_text() == "leave alone"
    with pytest.raises(FileExistsError):
        launcher.reserve(selected, source)
    with pytest.raises(FileExistsError):
        launcher.snapshot(selected, run, source)


def test_storage_replaced_after_plan_fails_closed(external):
    source, storage, selected = external
    storage.rename(storage.with_name("preserved-original"))
    storage.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="changed"):
        launcher.reserve(selected, source)
    assert not Path(selected["paths"]["receipt"]).exists()


def test_reservation_mkdir_race_never_overwrites(external, monkeypatch):
    source, _, selected = external
    original = launcher.os.mkdir
    fired = False
    def raced(path, mode=0o777, *, dir_fd=None):
        nonlocal fired
        if path == selected["label"] and not fired:
            fired = True
            original(path, mode, dir_fd=dir_fd)
        return original(path, mode, dir_fd=dir_fd)
    monkeypatch.setattr(launcher.os, "mkdir", raced)
    with pytest.raises(FileExistsError):
        launcher.reserve(selected, source)
    assert Path(selected["paths"]["receipt"]).is_dir()
    assert not Path(selected["paths"]["run"]).exists()


@pytest.mark.parametrize("root_free,external_free", [(256 * 1024 ** 2 - 1, 100 * launcher.GIB),
                                                       (3 * launcher.GIB, 16 * launcher.GIB - 1)])
def test_disk_admission_independent_before_gpu(external, monkeypatch, root_free, external_free):
    source, storage, selected = external
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda p: SimpleNamespace(
        free=external_free if Path(p) == storage else root_free))
    monkeypatch.setattr(launcher, "gpu_preflight", lambda: pytest.fail("no GPU queries allowed"))
    with pytest.raises(ValueError):
        launcher.execute(selected, source)
    assert not Path(selected["paths"]["run"]).exists()
    assert not Path(selected["paths"]["receipt"]).exists()


@pytest.mark.parametrize("cause", ["root", "external", "budget", "identity"])
def test_watchdog_independent_and_source_cwd(external, monkeypatch, cause):
    source, _, selected = external
    run = launcher.reserve(selected, source)
    sent = []
    class Child:
        pid = 123456789
        def poll(self):
            return None
        def send_signal(self, sig):
            sent.append(sig)
        def wait(self, timeout):
            return 143
    child = Child()
    limits = []
    monkeypatch.setattr(launcher.resource, "setrlimit", lambda resource, value: limits.append((resource, value)))
    def popen(command, **kwargs):
        assert kwargs["cwd"] == source
        assert command[-1] == selected["paths"]["selected_config"]
        kwargs["preexec_fn"]()
        return child
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda p: SimpleNamespace(free=(
        0 if (cause == "root" and Path(p) == source) or (cause == "external" and Path(p) == run)
        else 100 * launcher.GIB)))
    monkeypatch.setattr(launcher, "output_bytes", lambda p: selected["output_budget_bytes"] if cause == "budget" else 0)
    if cause == "identity":
        selected["storage_reservations"]["run"][1] += 1
    result = launcher.supervise_with_disk_guard(selected, run, source)
    assert (launcher.resource.RLIMIT_CORE, (0, 0)) in limits
    assert (launcher.resource.RLIMIT_FSIZE, (selected["file_limit_bytes"],) * 2) in limits
    assert sent and set(sent) == {signal.SIGTERM}
    expected = {"root": "256 MiB", "external": "8 GiB", "budget": "output budget", "identity": "safety check"}
    assert expected[cause] in result["stop_reason"]


def test_caches_in_output_budget(external):
    source, _, selected = external
    run = launcher.reserve(selected, source)
    before = launcher.output_bytes(run)
    for name in ("TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TMPDIR"):
        (Path(selected["environment"][name]) / "compiler-output").write_bytes(b"x" * 1024)
    assert launcher.output_bytes(run) == before + 4096


def test_external_execution_receipts_without_model_or_gpu(external, monkeypatch):
    source, _, selected = external
    lock = source / launcher.INTEGRATION / ".paper-stack-launch.lock"
    lock.write_text("persistent lock contents")
    lock.chmod(0o664)
    for key in ("checkpoint", "base_source", "vae_source"):
        (source / selected["selected"][key]).mkdir(parents=True, exist_ok=True)
    python = source / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("CPU fixture, never executed")
    python.chmod(0o700)
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda p: SimpleNamespace(
        free=3 * launcher.GIB if Path(p) == source else 100 * launcher.GIB))
    queries = []
    monkeypatch.setattr(launcher, "gpu_preflight", lambda: queries.append("mock") or {})
    def supervise(plan, run, root):
        assert root == source
        assert plan["environment"]["HOME"].startswith(str(run))
        return dict(exit_code=2, stop_reason="CPU synthetic failure", wall_seconds=0, output_bytes=0)
    monkeypatch.setattr(launcher, "supervise_with_disk_guard", supervise)
    assert launcher.execute(selected, source) == 2
    assert queries == ["mock", "mock"]
    receipt = Path(selected["paths"]["receipt"])
    actual = json.loads((receipt / "actual-runtime-manifest.json").read_text())
    assert not actual["launcher_success"] and actual["stop_reason"] == "CPU synthetic failure"
    assert Path(actual["actual_manifest"]).is_file()
    assert (receipt / "exit.txt").read_text() == "2\n"
    assert not list(receipt.rglob("*.mp4")) and not (receipt / "snapshot").exists()
    assert lock.read_text() == "persistent lock contents"


def test_symlink_lock_refused_without_modifying_target(external):
    source, storage, selected = external
    sentinel = storage / "sentinel"
    sentinel.write_text("preserve")
    (source / launcher.INTEGRATION / ".paper-stack-launch.lock").symlink_to(sentinel)
    with pytest.raises(OSError):
        launcher.execute(selected, source)
    assert sentinel.read_text() == "preserve"
