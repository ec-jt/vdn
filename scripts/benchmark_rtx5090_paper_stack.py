"""Fail-closed, research-only fixed paper-stack launcher; no Torch imports.

Dry-run is side-effect free and requires neither CUDA nor model access. Actual
runs use the existing Linux subreaper, with a separate disk watchdog that signals
only that owned supervisor. Nothing here deletes outputs or stops services.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import fcntl
import grp
import hashlib
import io
import json
import os
import pwd
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = Path("artifacts/rtx5090/paper-stack-20261005/integration")
PRESET = Path("configs/inference/8nfe_rtx5090_paper_stack.yaml")
COMPONENTS = ("mxfp8", "qk_rope", "window", "vda", "vda_overlap", "adaln", "fused_block")
GIB = 1024 ** 3
RESERVE = 8 * GIB
BUDGET = {"plain": 8 * GIB, "profile": 64 * GIB}
FILE_LIMIT = {"plain": 2 * GIB, "profile": 8 * GIB}
MODEL = "../models/OpenVDN/vdn-minimax-h3/"
ROOT_RECEIPT_RESERVE = 256 * 1024 ** 2
LOCAL_FILESYSTEMS = frozenset(("ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs"))


def private_group(gid):
    """Permit the caller's Unix private group, not a shared writable group."""
    try:
        user = pwd.getpwuid(os.geteuid())
        group = grp.getgrgid(gid)
        return (gid == user.pw_gid and set(group.gr_mem) <= {user.pw_name}
                and all(p.pw_uid == user.pw_uid for p in pwd.getpwall() if p.pw_gid == gid))
    except KeyError:
        return False


@contextmanager
def trusted_directory(path, private=False):
    """Walk from / using openat/O_NOFOLLOW, never a check-then-follow lookup.

    Root and this uid are trusted. No shared writable ancestor is admitted,
    including sticky /tmp. The final storage root must already be private.
    Privileged or same-uid hostile mutation is outside this process's sandbox.
    """
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("storage paths must be absolute without parent traversal")
    fds = []
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fds.append(fd)
        for part in (None, *path.parts[1:]):
            if part is not None:
                fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                fds.append(fd)
            info = os.fstat(fd)
            if (info.st_uid not in (0, os.geteuid()) or info.st_mode & 0o002
                    or (info.st_mode & 0o020 and not private_group(info.st_gid))):
                raise ValueError("untrusted owner or writable shared storage ancestor")
        if private and (info.st_uid != os.geteuid() or info.st_mode & 0o077
                        or info.st_mode & 0o700 != 0o700):
            raise ValueError("storage root must be caller-owned, private and writable (0700)")
        if not os.access(path, os.W_OK | os.X_OK, effective_ids=True):
            raise ValueError("directory is not writable/searchable")
        yield fd
    finally:
        for fd in reversed(fds):
            os.close(fd)


def mount_record(path):
    """Linux mountinfo, not a remote-path naming heuristic; unknown FS fails closed."""
    def unescape(value):
        return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)

    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        mount = Path(unescape(fields[4]))
        if path.is_relative_to(mount):
            matches.append((len(mount.parts), dict(mount_id=fields[0], device=fields[2],
                            mount=str(mount), filesystem=fs[0], source=unescape(fs[1]),
                            options=fields[5].split(",") + fs[2].split(","))))
    if not matches:
        raise ValueError("cannot identify storage mount")
    return max(matches, key=lambda item: item[0])[1]


def validate_storage(path, root=ROOT):
    path = Path(path)
    with trusted_directory(path, private=True) as fd:
        info = os.fstat(fd)
        mount = mount_record(path)
        if mount["filesystem"] not in LOCAL_FILESYSTEMS or "rw" not in mount["options"] or "ro" in mount["options"]:
            raise ValueError("storage requires a writable persistent LOCAL filesystem (no remote/FUSE/tmpfs)")
        if mount["device"] != f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}":
            raise ValueError("storage mount changed during validation")
        if info.st_dev == root.stat().st_dev:
            raise ValueError("storage must be on a separate filesystem from the source root")
        models = (root / MODEL).resolve()
        if path.is_relative_to(models) or models.is_relative_to(path):
            raise ValueError("storage must not contain or be inside the immutable model tree")
        return dict(root=str(path), device=info.st_dev, inode=info.st_ino, mount=mount,
                    root_receipt_reserve_bytes=ROOT_RECEIPT_RESERVE)


def check_storage_identity(plan, root=ROOT):
    if validate_storage(plan["storage"]["root"], root) != plan["storage"]:
        raise ValueError("storage identity or mount changed since planning")


def check_root_space(root):
    if shutil.disk_usage(root).free < ROOT_RECEIPT_RESERVE:
        raise ValueError("source root has less than 256 MiB receipt reserve")


def check_reservations(plan, root=ROOT):
    check_storage_identity(plan, root)
    for key, expected in plan["storage_reservations"].items():
        with trusted_directory(plan["paths"][key], private=True) as fd:
            info = os.fstat(fd)
            if [info.st_dev, info.st_ino] != expected:
                raise ValueError(f"reserved {key} directory changed")


def storage_environment(run):
    """Fresh caches only. Never copy, remove or reuse existing caches."""
    locations = {
        "HOME": "home", "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "XDG_CACHE_HOME": "cache/xdg", "XDG_CONFIG_HOME": "home/config",
        "XDG_DATA_HOME": "home/data", "XDG_STATE_HOME": "home/state",
        "XDG_RUNTIME_DIR": "runtime", "TORCH_HOME": "cache/torch",
        "TORCH_EXTENSIONS_DIR": "cache/extensions", "CUDA_CACHE_PATH": "cache/cuda",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor", "TRITON_CACHE_DIR": "cache/triton",
        "TRITON_HOME": "cache/triton-home", "TRITON_DUMP_DIR": "cache/triton-dump",
        "TRITON_OVERRIDE_DIR": "cache/triton-override", "HF_HOME": "cache/hf",
        "HF_HUB_CACHE": "cache/hf/hub", "HUGGINGFACE_HUB_CACHE": "cache/hf/hub",
        "HF_ASSETS_CACHE": "cache/hf/assets", "HF_XET_CACHE": "cache/hf/xet",
        "HF_DATASETS_CACHE": "cache/hf/datasets", "TRANSFORMERS_CACHE": "cache/hf/transformers",
        "NUMBA_CACHE_DIR": "cache/numba", "MPLCONFIGDIR": "cache/matplotlib",
    }
    return {name: str(run / relative) for name, relative in locations.items()}


def expected_preset():
    """Pin every public input, not just runtime admission's narrower checks."""
    return {
        "checkpoint": MODEL + "stage-dmd-step-250",
        "base_source": MODEL + "h3-base", "vae_source": MODEL + "h3-base",
        "external_loras": [],
        "memory": dict(policy="resident_weight_shards", local_files_only=True,
                       shard_world_size=8, scatter_chunk_mib=32,
                       shard_distribution="broadcast", weight_prefetch=True),
        "kernels": dict(inference_kernels=False, softmax_backend="flex",
                        vda_fused=False, vda_overlap=False),
        "precision": dict(dtype="bfloat16", fp8=dict(enabled=False, skip_end_blocks=4)),
        "compute": {**{"paper_" + c: True for c in COMPONENTS},
                    "paper_max_gather_rows": 65536, "paper_max_query_rows": 8192,
                    "paper_adaln_table_mib": 256},
        "parallel": dict(softmax_ranks=0, standard_exchange="allgather",
                         profile=False, timeout_seconds=600),
        "behavior": dict(teacher_mode=False),
        "ablation": dict(enabled=False, overrides={}),
        "render": dict(prompt_file="prompts/example_0.pt",
                       out=str(INTEGRATION / "LAUNCHER_REQUIRED/render.mp4"),
                       latents_root=str(INTEGRATION / "LAUNCHER_REQUIRED/latents"),
                       device="cuda:0", num_frames=124, num_steps=8, warmup_steps=0,
                       seed=42, video_shift=12.0, audio_shift=3.0,
                       save_latents=True, denoise_only=False, record=True),
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("label", help="fresh label, never reused, 1-61 alphanumeric/_/- characters")
    p.add_argument("mode", choices=("plain", "profile", "nsys"))
    p.add_argument("--disable", action="append", default=[], choices=COMPONENTS,
                   help="independently disable one component; repeatable")
    p.add_argument("--prefetch", choices=("next_block", "disabled"), default="next_block")
    p.add_argument("--timeout-seconds", type=int, default=900)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--storage-root", help="existing owned private directory on persistent local external storage")
    return p


def build_plan(args, root=ROOT):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,60}", args.label):
        raise ValueError("invalid label")
    if args.label == "LAUNCHER_REQUIRED":
        raise ValueError("reserved label")
    if not 1 <= args.timeout_seconds <= 1800:
        raise ValueError("timeout must be in [1,1800] seconds")
    mode = "profile" if args.mode == "nsys" else args.mode
    preset_bytes = (root / PRESET).read_bytes()
    cfg = yaml.safe_load(preset_bytes)
    # JSON comparison also distinguishes bool from int; unknown keys are refused.
    if json.dumps(cfg, sort_keys=True) != json.dumps(expected_preset(), sort_keys=True):
        raise ValueError("preset differs from the fixed research protocol; refusing launch")
    cfg = copy.deepcopy(cfg)
    for component in args.disable:
        cfg["compute"]["paper_" + component] = False
    cfg["memory"]["weight_prefetch"] = args.prefetch == "next_block"
    run = INTEGRATION / args.label
    storage = None
    if args.storage_root is not None:
        storage = validate_storage(args.storage_root, root)
        run = Path(storage["root"]) / args.label
    paths = {"run": str(run), "media": str(run / "render.mp4"),
             "partial": str(run / "render.partial.mp4"),
             "metrics": str(run / "render.mp4.inference.json"),
             "latents": str(run / "latents/render.mp4.latents.pt"),
             "selected_config": str(run / "selected-config.yaml")}
    cfg["render"]["out"] = paths["media"]
    cfg["render"]["latents_root"] = str(run / "latents")
    env = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "8", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_P2P_LEVEL": "SYS", "NCCL_SHM_USE_CUDA_MEMCPY": "0",
        "TORCHINDUCTOR_CACHE_DIR": str(root / "artifacts/rtx5090/8gpu-kernel-cache/inductor"),
        "TRITON_CACHE_DIR": str(root / "artifacts/rtx5090/8gpu-kernel-cache/triton"),
        "VDN_NVTX": str(int(mode == "profile")), "VDN_NSYS_CAPTURE": str(int(mode == "profile")),
        "TMPDIR": str(root / run / "tmp"),
    }
    worker = ["src/inference/infer_ulysses.py"]
    if mode == "profile":
        env["VDN_PROFILE_DIR"] = str(root / run / "profile")
        worker = ["--no-python", "bash", "scripts/nsys_rtx5090_rank.sh", *worker]
    if storage is not None:
        env.update(storage_environment(run))
    command = [".venv/bin/python", "scripts/supervise_rtx5090.py", "--seconds",
               str(args.timeout_seconds), "--", ".venv/bin/python", "-m", "torch.distributed.run",
               "--standalone", "--nnodes=1", "--nproc_per_node=8", "--max-restarts=0",
               *worker, "--config", paths["selected_config"]]
    plan = dict(research_only=True, promotion=False, label=args.label, mode=mode,
                paths=paths, selected=cfg, command=command, environment=env,
                preset_sha256=hashlib.sha256(preset_bytes).hexdigest(),
                timeout_seconds=args.timeout_seconds, output_budget_bytes=BUDGET[mode],
                min_free_bytes=BUDGET[mode] + RESERVE, file_limit_bytes=FILE_LIMIT[mode],
                decoder_expected=dict(world_size=8, clips=7, tiles_per_clip=28, total_jobs=196,
                                      rank_job_counts=[25, 25, 25, 25, 24, 24, 24, 24], output_frames=124))
    if storage is not None:
        plan["storage"] = storage
        plan["paths"]["receipt"] = str(root / INTEGRATION / args.label)
    return plan


def child_environment(plan, inherited=None):
    inherited = os.environ if inherited is None else inherited
    # Do not inherit renderer/transport/profiler/Python injection switches. Retain
    # local cache locations and native library paths; never clear existing caches.
    keep = ("HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH",
            "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME",
            "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH")
    env = {k: inherited[k] for k in keep if k in inherited}
    env.update(plan["environment"])
    return env


def check_gpu_inventory(gpu_csv, apps_csv):
    if apps_csv.strip():
        raise ValueError("GPUs have compute applications; no service will be killed")
    rows = list(csv.reader(io.StringIO(gpu_csv.strip())))
    if len(rows) != 8:
        raise ValueError("exactly eight GPUs required")
    seen = set()
    for row in rows:
        if len(row) != 4:
            raise ValueError("invalid GPU inventory")
        index, name, memory, utilization = [x.strip() for x in row]
        seen.add(int(index))
        if "RTX 5090" not in name or int(memory) > 16 or int(utilization) != 0:
            raise ValueError("all eight RTX 5090 GPUs must be idle (0% utilization, <=16 MiB)")
    if seen != set(range(8)):
        raise ValueError("GPU indices must be exactly 0 through 7")


def gpu_preflight():
    def query(fields, kind):
        return subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                               "--format=csv,noheader,nounits"], check=True,
                              capture_output=True, text=True, timeout=10).stdout
    gpu = query("index,name,memory.used,utilization.gpu", "gpu")
    apps = query("pid,gpu_uuid", "compute-apps")
    check_gpu_inventory(gpu, apps)
    return dict(gpus=gpu, compute_apps=apps)


def check_space(plan, free):
    if free < plan["min_free_bytes"]:
        raise ValueError(f"insufficient disk: {free / GIB:.2f} GiB free; "
                         f"{plan['min_free_bytes'] / GIB:.0f} GiB required for {plan['mode']}")


def write_new(path, text):
    with path.open("x") as stream:
        stream.write(text)


def write_json(path, value):
    write_new(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def reserve(plan, root=ROOT):
    if "storage" in plan:
        check_storage_identity(plan, root)
        receipt = Path(plan["paths"]["receipt"])
        run = Path(plan["paths"]["run"])
        with trusted_directory(receipt.parent) as parent_fd, trusted_directory(
                plan["storage"]["root"], private=True) as storage_fd:
            # Refuse either collision before reserving anything. mkdirat also
            # catches a concurrent creator. Partial reservations are retained.
            for fd in (parent_fd, storage_fd):
                try:
                    os.stat(plan["label"], dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise FileExistsError("label already reserved")
            os.mkdir(plan["label"], mode=0o700, dir_fd=parent_fd)
            os.mkdir(plan["label"], mode=0o700, dir_fd=storage_fd)
        write_json(receipt / "storage-pointer.json", dict(run=str(run), storage=plan["storage"]))
        directories = {run / d for d in ("latents", "profile", "tmp", "snapshot")}
        directories.update(Path(p) for p in storage_environment(run).values())
        for directory in sorted(directories):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        plan["storage_reservations"] = {}
        for key in ("run", "receipt"):
            with trusted_directory(plan["paths"][key], private=True) as fd:
                info = os.fstat(fd)
                plan["storage_reservations"][key] = [info.st_dev, info.st_ino]
        return run
    run = root / plan["paths"]["run"]
    # lexists includes dangling symlinks. Atomic mkdir reserves the entire private
    # namespace, not placeholder files the renderer would correctly reject.
    if os.path.lexists(run):
        raise FileExistsError(f"run label already reserved: {run}")
    run.mkdir(mode=0o700)
    for directory in ("latents", "profile", "tmp", "snapshot"):
        (run / directory).mkdir(mode=0o700)
    for key in ("media", "partial", "metrics", "latents", "selected_config"):
        if os.path.lexists(root / plan["paths"][key]):
            raise FileExistsError(plan["paths"][key])
    return run


def snapshot(plan, run, root=ROOT):
    hashes = {}
    total = 0
    files = [p for directory in ("src", "scripts", "configs")
             for p in (root / directory).rglob("*")
             if p.suffix in (".py", ".sh", ".yaml", ".cu", ".cuh", ".cpp", ".h")]
    files += [root / "tests/test_paper_stack_launcher.py", root / "pyproject.toml",
              root / "prompts/example_0.pt"]
    storage_test = root / "tests/test_paper_stack_storage.py"
    if "storage" in plan or storage_test.is_file():
        files.append(storage_test)
    for source in sorted(files):
        data = source.read_bytes()
        total += len(data)
        if len(data) > 16 * 1024 ** 2 or total > 64 * 1024 ** 2:
            raise ValueError("source/prompt snapshot unexpectedly large")
        rel = source.relative_to(root)
        hashes[str(rel)] = hashlib.sha256(data).hexdigest()
        target = run / "snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    if hashes[str(PRESET)] != plan["preset_sha256"]:
        raise ValueError("preset changed during launch")
    selected = yaml.safe_dump(plan["selected"], sort_keys=False)
    write_new(root / plan["paths"]["selected_config"], selected)
    hashes[plan["paths"]["selected_config"]] = hashlib.sha256(selected.encode()).hexdigest()
    write_json(run / "source-sha256.json", hashes)
    write_json(run / "selected-manifest.json", plan)
    # This is the exact supervised command, with the selected (sanitized) environment.
    write_new(run / "command.txt", "env -i " + shlex.join(
        [f"{k}={v}" for k, v in sorted(plan["environment"].items())] + plan["command"]) + "\n")
    if "storage" in plan:
        receipt = Path(plan["paths"]["receipt"])
        write_new(receipt / "selected-config.yaml", selected)
        write_json(receipt / "selected-manifest.json", plan)
        write_new(receipt / "command.txt", (run / "command.txt").read_text())


def output_bytes(run):
    total = 0
    for path in run.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # Profiler temporary-file renames.
    return total


def supervise_with_disk_guard(plan, run, root=ROOT):
    interrupted = []
    previous = {s: signal.signal(s, lambda signum, frame: interrupted.append(signum))
                for s in (signal.SIGINT, signal.SIGTERM)}
    child = None
    reason = None
    started = time.monotonic()

    def file_limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (plan["file_limit_bytes"], plan["file_limit_bytes"]))
        if "storage" in plan:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # Never dump a core into source cwd.

    try:
        with (run / "launch.log").open("x") as log:
            child = subprocess.Popen(plan["command"], cwd=root, env=plan["environment"],
                                     stdout=log, stderr=subprocess.STDOUT, preexec_fn=file_limit,
                                     start_new_session=True)
            write_json(run / "supervisor.json", dict(pid=child.pid, command=plan["command"]))
            while child.poll() is None:
                if "storage" in plan:
                    try:
                        check_reservations(plan, root)
                    except (ValueError, OSError) as exc:
                        reason = f"storage safety check failed: {exc}"
                if interrupted:
                    reason = f"launcher received signal {interrupted[0]}"
                elif output_bytes(run) >= plan["output_budget_bytes"]:
                    reason = "output budget exceeded"
                elif shutil.disk_usage(run).free < RESERVE:
                    reason = "free disk fell below 8 GiB reserve"
                elif "storage" in plan and shutil.disk_usage(root).free < ROOT_RECEIPT_RESERVE:
                    reason = "source root fell below 256 MiB receipt reserve"
                if reason:
                    child.send_signal(signal.SIGTERM)  # Only our owned subreaper.
                    break
                time.sleep(0.25)
            code = child.wait(timeout=plan["timeout_seconds"] + 30)
            return dict(exit_code=code, stop_reason=reason,
                        wall_seconds=time.monotonic() - started,
                        output_bytes=output_bytes(run))
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=plan["timeout_seconds"] + 30)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def actual_manifest(plan, run, result, root=ROOT):
    """Report observed inventories only, never turn selected flags into call claims."""
    actual = dict(result, research_only=True, promotion=False, ranks=[], terminal_inventories=[],
                  missing_ranks=list(range(8)), evidence_errors=[], metrics_present=False)
    metrics = root / plan["paths"]["metrics"]
    errors = actual["evidence_errors"]
    if metrics.is_file():
        try:
            record = json.loads(metrics.read_text())
            actual["metrics_present"] = True
            actual["metrics_sha256"] = hashlib.sha256(metrics.read_bytes()).hexdigest()
            actual["timings"] = record.get("timings", {})
            actual["ranks"] = [r.get("paper_stack") for r in actual["timings"].get("ranks", [])]
            actual["decoder_observed"] = actual["timings"].get("decoder_metrics", {}).get("distributed_video", {})
            for key, value in plan["decoder_expected"].items():
                if actual["decoder_observed"].get(key) != value:
                    errors.append(f"decoder mismatch: {key}")
            if record.get("overlay") != plan["selected"]:
                errors.append("runtime overlay differs from selected config")
            enabled = any(plan["selected"]["compute"]["paper_" + c] for c in COMPONENTS)
            actual["paper_runtime_expected"] = enabled
            if enabled:
                ranks = [r for r in actual["ranks"] if isinstance(r, dict)]
                ids = [r.get("rank") for r in ranks]
                actual["missing_ranks"] = sorted(set(range(8)) - set(ids))
                if sorted(ids) != list(range(8)):
                    errors.append("incomplete or duplicate all-rank paper inventory")
                for rank in ranks:
                    if rank.get("selected") != plan["selected"]["compute"]:
                        errors.append(f"rank {rank.get('rank')} selection mismatch")
                    if rank.get("finite_failed") or rank.get("completed_forwards") != 8:
                        errors.append(f"rank {rank.get('rank')} incomplete or nonfinite")
            else:
                actual["missing_ranks"] = []  # All-off ablation has no paper owner.
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(f"invalid runtime metrics: {exc}")
    else:
        errors.append("runtime metrics absent; selection is not execution evidence")
    log = run / "launch.log"
    if log.exists():
        with log.open(errors="replace") as stream:
            for line in stream:
                if "paper_stack_inventory=" in line:
                    try:
                        obj, _ = json.JSONDecoder().raw_decode(line.split("paper_stack_inventory=", 1)[1])
                        actual["terminal_inventories"].append(obj)
                    except ValueError:
                        errors.append("unparseable/interleaved terminal inventory")
    for key in ("media", "latents"):
        if not (root / plan["paths"][key]).is_file():
            errors.append(f"missing output: {key}")
    actual["launcher_success"] = result["exit_code"] == 0 and not result["stop_reason"] and not errors
    actual["numerical_acceptance"] = "not evaluated; prior rejections remain rejected"
    return actual


def execute(plan, root=ROOT):
    if "storage" in plan:
        return execute_storage(plan, root)
    parent = root / INTEGRATION
    if parent.resolve() != parent.absolute() or not parent.is_dir():
        raise ValueError("integration directory must already exist without symlink parents")
    # Persistent advisory lock: never unlink a live lock inode or delete artifacts.
    with (parent / ".paper-stack-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.path.lexists(root / plan["paths"]["run"]):
            raise FileExistsError("label already reserved")
        check_space(plan, shutil.disk_usage(parent).free)
        for name in ("checkpoint", "base_source", "vae_source"):
            if not (root / plan["selected"][name]).is_dir():
                raise ValueError(f"missing immutable local model directory: {name}")
        if not os.access(root / ".venv/bin/python", os.X_OK):
            raise ValueError("existing .venv Python required; no installation attempted")
        if plan["mode"] == "profile" and not os.access("/usr/local/cuda/bin/nsys", os.X_OK):
            raise ValueError("existing per-rank nsys executable required")
        inventory = gpu_preflight()
        run = reserve(plan, root)
        plan["environment"] = child_environment(plan)
        try:
            snapshot(plan, run, root)
            write_json(run / "gpu-preflight.json", inventory)
            # Snapshot work may take time: recheck immediately before launch.
            check_space(plan, shutil.disk_usage(parent).free)
            write_json(run / "gpu-prelaunch.json", gpu_preflight())
            result = supervise_with_disk_guard(plan, run, root)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            result = dict(exit_code=2, stop_reason=f"launcher exception: {exc}",
                          wall_seconds=None, output_bytes=output_bytes(run))
        actual = actual_manifest(plan, run, result, root)
        write_json(run / "actual-runtime-manifest.json", actual)
        code = result["exit_code"] or (0 if actual["launcher_success"] else 1)
        write_new(run / "exit.txt", str(code) + "\n")
        print(json.dumps(dict(run=str(run), exit_code=code, launcher_success=actual["launcher_success"])))
        return code


def execute_storage(plan, root=ROOT):
    """External output mode; no storage creation before disk/model admission."""
    parent = root / INTEGRATION
    check_storage_identity(plan, root)
    with trusted_directory(parent) as parent_fd, trusted_directory(
            plan["storage"]["root"], private=True):
        if os.fstat(parent_fd).st_dev != root.stat().st_dev:
            raise ValueError("integration receipts must remain on the source filesystem")
        lock_fd = os.open(".paper-stack-launch.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW,
                          0o600, dir_fd=parent_fd)
        with os.fdopen(lock_fd, "w") as lock:
            info = os.fstat(lock.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1
                    or info.st_mode & 0o002 or (info.st_mode & 0o020 and not private_group(info.st_gid))):
                raise ValueError("unsafe launcher lock")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for key in ("run", "receipt"):
                if os.path.lexists(plan["paths"][key]):
                    raise FileExistsError("label already reserved")
            check_root_space(root)
            check_space(plan, shutil.disk_usage(plan["storage"]["root"]).free)
            for name in ("checkpoint", "base_source", "vae_source"):
                if not (root / plan["selected"][name]).is_dir():
                    raise ValueError(f"missing immutable local model directory: {name}")
            if not os.access(root / ".venv/bin/python", os.X_OK):
                raise ValueError("existing .venv Python required; no installation attempted")
            if plan["mode"] == "profile" and not os.access("/usr/local/cuda/bin/nsys", os.X_OK):
                raise ValueError("existing per-rank nsys executable required")
            inventory = gpu_preflight()
            run = reserve(plan, root)
            receipt = Path(plan["paths"]["receipt"])
            plan["environment"] = child_environment(plan)
            try:
                snapshot(plan, run, root)
                write_json(run / "gpu-preflight.json", inventory)
                check_reservations(plan, root)
                check_root_space(root)
                check_space(plan, shutil.disk_usage(run).free)
                write_json(run / "gpu-prelaunch.json", gpu_preflight())
                result = supervise_with_disk_guard(plan, run, root)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                result = dict(exit_code=2, stop_reason=f"launcher exception: {exc}",
                              wall_seconds=None, output_bytes=output_bytes(run))
            actual = actual_manifest(plan, run, result, root)
            code = result["exit_code"] or (0 if actual["launcher_success"] else 1)
            # Root gets a bounded summary, never the potentially large inventories.
            write_json(receipt / "actual-runtime-manifest.json", dict(
                result, launcher_success=actual["launcher_success"], exit_code=code,
                actual_manifest=str(run / "actual-runtime-manifest.json"),
                evidence_error_count=len(actual["evidence_errors"]),
                numerical_acceptance=actual["numerical_acceptance"]))
            write_new(receipt / "exit.txt", str(code) + "\n")
            write_json(run / "actual-runtime-manifest.json", actual)
            write_new(run / "exit.txt", str(code) + "\n")
            print(json.dumps(dict(run=str(run), receipt=str(receipt), exit_code=code,
                                  launcher_success=actual["launcher_success"])))
            return code


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        return execute(plan)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"paper-stack launcher refused/failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
