#!/usr/bin/env python3
"""Transport only. --plan is CPU/stdlib only; GPU execution requires --run.

Run via probe_rtx5090_transport.sh, only after production has exited.
No model/runtime imports, weights, downloads, or persistent configuration changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import statistics
import time


WORLD = 8
ROWS = 38_003
HEADS = 56
WIDTH = 770
GATHER_BYTES = 1_376_730_080
GIB = 1024**3
# Explicit allowlist, never dump the general environment or arbitrary prefix values.
ENV_KEYS = frozenset("""
CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER CUDA_MODULE_LOADING CUDA_LAUNCH_BLOCKING
CUDA_DEVICE_MAX_CONNECTIONS NCCL_P2P_DISABLE NCCL_P2P_LEVEL NCCL_P2P_DIRECT_DISABLE
NCCL_SHM_DISABLE NCCL_SHM_USE_CUDA_MEMCPY NCCL_SHM_MEMCPY_MODE NCCL_SHM_LOCALITY
NCCL_CUMEM_ENABLE NCCL_CUMEM_HOST_ENABLE NCCL_IB_DISABLE NCCL_NET NCCL_NET_GDR_LEVEL
NCCL_SOCKET_IFNAME NCCL_SOCKET_FAMILY NCCL_SOCKET_NTHREADS NCCL_NSOCKS_PERTHREAD
NCCL_BUFFSIZE NCCL_NTHREADS NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
NCCL_MIN_CTAS NCCL_MAX_CTAS NCCL_ALGO NCCL_PROTO NCCL_DEBUG NCCL_DEBUG_SUBSYS
NCCL_DEBUG_FILE NCCL_TOPO_FILE NCCL_TOPO_DUMP_FILE NCCL_CONF_FILE
NCCL_IGNORE_CPU_AFFINITY NCCL_NVLS_ENABLE NCCL_MNNVL_ENABLE NCCL_PXN_DISABLE
TORCH_NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_BLOCKING_WAIT
TORCH_NCCL_ENABLE_MONITORING TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
""".split())


def transport_env(environ):
    return {key: environ[key] for key in sorted(ENV_KEYS) if key in environ}


def transport_configs(environ):
    paths = {Path("/etc/nccl.conf"), Path.home() / ".nccl.conf"}
    if environ.get("NCCL_CONF_FILE"):
        paths.add(Path(environ["NCCL_CONF_FILE"]))
    result = {}
    for path in sorted(paths):
        try:
            lines = path.read_text().splitlines()
        except FileNotFoundError:
            result[str(path)] = {"exists": False}
            continue
        values = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if sep and key in {"NCCL_DEBUG_FILE", "NCCL_TOPO_DUMP_FILE", "NCCL_GRAPH_DUMP_FILE"} and value:
                raise RuntimeError(f"Refusing NCCL output destination configured in {path}")
            if sep and key in ENV_KEYS:
                values[key] = value
        result[str(path)] = {"exists": True, "settings": values}
    return result


def supervise(command, seconds=180, grace=10):
    """Linux subreaper: contain detached torchrun sessions without CUDA imports."""
    import ctypes
    import signal
    import subprocess

    if not 0 < seconds <= 180 or not 0 < grace <= 10:
        raise ValueError("Supervisor deadline must be <=180s and grace <=10s")
    # Orphaned worker sessions reparent here, not to init if torchrun exits first.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")
    interrupted = []
    previous = {sig: signal.signal(sig, lambda signum, frame: interrupted.append(signum))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    child = None

    def descendants():
        parents = {}
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                stat = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
                parents[int(proc.name)] = (int(stat[1]), stat[0])
            except (FileNotFoundError, ProcessLookupError):
                continue
        found = {os.getpid()}
        while True:
            more = {pid for pid, (parent, _) in parents.items() if parent in found}
            if more <= found:
                break
            found |= more
        return {pid for pid in found - {os.getpid()} if parents[pid][1] != "Z"}

    def send_all(sig):
        for pid in descendants():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    try:
        child = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + seconds
        while child.poll() is None and not interrupted and time.monotonic() < deadline:
            time.sleep(0.05)
        if interrupted:
            status = 128 + interrupted[0]
        elif child.poll() is None:
            status = 124
        else:
            status = child.returncode if child.returncode >= 0 else 128 - child.returncode
            if descendants() and status == 0:
                status = 1
        return status
    finally:
        # Poll descendants through grace: catches newly forked and reparented workers.
        cleanup_deadline = time.monotonic() + grace
        while descendants() and time.monotonic() < cleanup_deadline:
            send_all(signal.SIGTERM)
            time.sleep(0.05)
        send_all(signal.SIGKILL)
        if child is not None:
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        # Reap exited adopted grandchildren without an unbounded wait.
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def sequence_splits(rows=ROWS, world=WORLD):
    chunk = (rows + world - 1) // world
    return [max(0, min(chunk, rows - rank * chunk)) for rank in range(world)]


def plan():
    splits = sequence_splits()
    local_bytes = max(splits) * HEADS * WIDTH * 2
    head_bytes = ROWS * (HEADS // WORLD) * WIDTH * 2
    return {
        "world_size": WORLD, "dtype": "bfloat16", "samples": 4,
        "sample_labels": ["cold_payload", "warm_1", "warm_2", "warm_3"],
        "gather_output_bytes": GATHER_BYTES,
        "gather_input_bytes_per_rank": GATHER_BYTES // WORLD,
        "rows": ROWS, "heads": HEADS, "width": WIDTH, "sequence_splits": splits,
        "forward_output_split_elements": [n * (HEADS // WORLD) * WIDTH for n in splits],
        "alltoall_max_local_bytes": local_bytes, "alltoall_head_bytes": head_bytes,
        "gather_live_tensor_bytes": GATHER_BYTES + GATHER_BYTES // WORLD,
        "alltoall_peak_live_tensor_bytes": 3 * local_bytes + head_bytes,
        "allocator_limit_bytes": 3 * GIB, "device_budget_bytes": 4 * GIB,
        "timing": "barrier + synchronize before timer; final synchronize included; validation excluded",
        "cold_definition": "first payload sample, not first communicator operation; alltoall follows gather",
    }


def production_pids(proc_root=Path("/proc")):
    found = []
    for proc in proc_root.glob("[0-9]*"):
        try:
            args = proc.joinpath("cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError("Cannot establish production-idle state from /proc") from exc
        if any(a.endswith(b"/infer_ulysses.py") or a in (b"infer_ulysses.py", b"src.inference.infer_ulysses") for a in args):
            found.append(int(proc.name))
    return sorted(found)


def exclusive_json(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def aggregate(records):
    if sorted(r["rank"] for r in records) != list(range(WORLD)):
        raise ValueError("Exactly eight unique rank reports are required")
    if any(r["status"] != "ok" for r in records):
        raise ValueError("Cannot summarize unsuccessful ranks")
    result = {}
    for name in ("allgather_seconds", "alltoall_roundtrip_seconds"):
        if any(len(r[name]) != 4 for r in records):
            raise ValueError("Expected cold plus three warm samples on every rank")
        maxima = [max(r[name][i] for r in records) for i in range(4)]
        result[name] = {"max_rank_per_sample": maxima,
                        "warm_mean_max_rank": statistics.mean(maxima[1:]),
                        "warm_median_max_rank": statistics.median(maxima[1:])}
    seconds = result["allgather_seconds"]["warm_mean_max_rank"]
    result["allgather_effective_remote_GB_per_s"] = GATHER_BYTES * 7 / 8 / seconds / 1e9
    result["bandwidth_note"] = "Logical remote payload/time, not measured PCIe or NCCL bus bandwidth"
    return result


def sequence_to_heads(x, splits, torch, dist):
    # Direct equivalent of UlyssesRuntime.sequence_to_heads, standard eight-way mode.
    local, heads, width = x.shape
    per_head = heads // WORLD
    send = x.reshape(local, WORLD, per_head, width).permute(1, 0, 2, 3).contiguous().reshape(-1)
    output_splits = [n * per_head * width for n in splits]
    recv = torch.empty(sum(output_splits), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(recv, send, output_split_sizes=output_splits,
                           input_split_sizes=[local * per_head * width] * WORLD)
    return recv.view(sum(splits), per_head, width)


def heads_to_sequence(x, splits, rank, torch, dist):
    per_head, width = x.shape[1:]
    local = splits[rank]
    recv = torch.empty(local * WORLD * per_head * width, dtype=x.dtype, device=x.device)
    dist.all_to_all_single(recv, x.contiguous().reshape(-1),
                           output_split_sizes=[local * per_head * width] * WORLD,
                           input_split_sizes=[n * per_head * width for n in splits])
    return recv.view(WORLD, local, per_head, width).permute(1, 0, 2, 3).contiguous().view(local, WORLD * per_head, width)


def check_filled(tensor, value, torch):
    flat = tensor.view(-1)
    # At most 1 MiB of boolean comparison workspace, no full CPU tensor copies.
    for start in range(0, flat.numel(), 1024**2):
        if not bool(torch.all(flat[start:start + 1024**2] == value).item()):
            raise AssertionError(f"Rank-filled correctness check failed: expected {value}")


def timed(operation, torch, dist):
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return time.perf_counter() - start, result


def memory_report(torch, device):
    free, total = torch.cuda.mem_get_info(device)
    result = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "device_used_bytes": total - free}
    if result["peak_reserved_bytes"] >= 3 * GIB or result["device_used_bytes"] >= 4 * GIB:
        raise RuntimeError(f"Transport memory budget exceeded: {result}")
    return result


def run(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD or local_rank != rank or not 0 <= rank < WORLD:
        raise RuntimeError("Requires one local node with exactly eight ranks")
    if production_pids():
        raise RuntimeError("Production inference is still active; refusing CUDA initialization")
    output = args.output_dir.resolve(strict=True)
    # Reserve before importing torch or initializing CUDA; never replace old artifacts.
    with (output / f"rank-{rank}.json").open("x", encoding="utf-8") as handle:
        record = {"rank": rank, "local_rank": local_rank, "pid": os.getpid(),
                  "utc": datetime.now(timezone.utc).isoformat(), "status": "started",
                  "transport_env": transport_env(os.environ), "plan": plan()}
        try:
            record["transport_configs"] = transport_configs(os.environ)
            import torch
            import torch.distributed as dist

            if torch.cuda.device_count() != WORLD:
                raise RuntimeError("Exactly eight visible CUDA devices required")
            torch.set_num_threads(1)
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(3 * GIB / total, device)
            record.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                          nccl_version=list(torch.cuda.nccl.version()),
                          device_name=torch.cuda.get_device_name(device))
            init_start = time.perf_counter()
            dist.init_process_group("nccl", timeout=timedelta(seconds=90), device_id=device)
            record["init_seconds"] = time.perf_counter() - init_start
            record["allgather_seconds"] = []
            record["alltoall_roundtrip_seconds"] = []
            record["memory"] = {}
            torch.cuda.reset_peak_memory_stats(device)
            shard_elements = GATHER_BYTES // (2 * WORLD)
            shard = torch.full((shard_elements,), rank, dtype=torch.bfloat16, device=device)
            gathered = torch.empty(shard_elements * WORLD, dtype=shard.dtype, device=device)
            for _ in range(4):
                elapsed, _ = timed(lambda: dist.all_gather_into_tensor(gathered, shard), torch, dist)
                record["allgather_seconds"].append(elapsed)
                for source in range(WORLD):
                    check_filled(gathered[source * shard_elements:(source + 1) * shard_elements], source, torch)
            record["memory"]["allgather"] = memory_report(torch, device)
            del gathered, shard
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            splits = sequence_splits()
            x = torch.full((splits[rank], HEADS, WIDTH), rank, dtype=torch.bfloat16, device=device)

            def roundtrip():
                heads = sequence_to_heads(x, splits, torch, dist)
                restored = heads_to_sequence(heads, splits, rank, torch, dist)
                return heads, restored

            for _ in range(4):
                elapsed, pair = timed(roundtrip, torch, dist)
                record["alltoall_roundtrip_seconds"].append(elapsed)
                heads, restored = pair
                offset = 0
                for source, rows in enumerate(splits):
                    check_filled(heads[offset:offset + rows], source, torch)
                    offset += rows
                check_filled(restored, rank, torch)
                del pair, heads, restored
            record["memory"]["alltoall"] = memory_report(torch, device)
            record["correctness"] = "all elements checked on every rank after all four samples"
            record["status"] = "ok"
            del x
        except BaseException as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            raise
        finally:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
    dist.barrier()
    if rank == 0:
        records = [json.loads((output / f"rank-{r}.json").read_text()) for r in range(WORLD)]
        summary = {"plan": plan(), "aggregate": aggregate(records), "ranks": records}
        exclusive_json(output / "summary.json", summary)
        print(json.dumps(summary["aggregate"], sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Print the fixed CPU-only payload/memory plan")
    mode.add_argument("--run", action="store_true", help="Explicitly authorize GPU work after production exits")
    mode.add_argument("--supervise", action="store_true", help="CPU-only descendant-aware command supervisor")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(plan(), indent=2, sort_keys=True))
    elif args.supervise:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("--supervise requires a command after --")
        raise SystemExit(supervise(command))
    elif args.output_dir is None:
        parser.error("--run requires an existing, fresh --output-dir")
    else:
        run(args)


if __name__ == "__main__":
    main()
