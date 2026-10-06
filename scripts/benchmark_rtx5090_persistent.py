"""Persistent opt-in launcher reusing the existing storage/supervision policy."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

import benchmark_rtx5090_paper_stack as base


def actual_manifest(plan, run, result, root=base.ROOT):
    errors = []
    batch = {}
    try:
        batch = json.loads((run / "batch.json").read_text())
        requests = batch["requests"]
        if len(requests) != int(plan["environment"]["VDN_PERSISTENT_REQUESTS"]):
            errors.append("request count mismatch")
        pids = None
        for request in requests:
            ranks = request["ranks"]
            current = [(r["rank"], r["pid"], r["process_group_identity"], r["weight_group_identity"]) for r in ranks]
            if [r["rank"] for r in ranks] != list(range(8)) or (pids is not None and current != pids):
                errors.append("workers/groups changed or ranks missing")
            pids = current
            for r in ranks:
                if r["paper_stack"]["finite_failed"] or r["paper_stack"]["completed_forwards"] != 8:
                    errors.append("incomplete/nonfinite request")
                if r["model_setup_count"] != 1 or r["distribution_count"] != 1:
                    errors.append("repeated setup")
                for key, value in plan["decoder_expected"].items():
                    if r["decoder"]["distributed_video"].get(key) != value:
                        errors.append("decode geometry mismatch: " + key)
    except (OSError, KeyError, ValueError) as exc:
        errors.append(str(exc))
    return dict(result, launcher_success=result["exit_code"] == 0 and not errors,
                evidence_errors=errors, batch=batch,
                numerical_acceptance="drift accepted for performance; repeatability/media verification separate")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--request-manifest")
    parser.add_argument("--tensor-parallel", action="store_true",
                        help="Permanent eight-way head/MLP weights; existing path remains default")
    parser.add_argument("--fused-producers", action="store_true")
    parser.add_argument("--fast-finite", action="store_true")
    parser.add_argument("--cache-adaln", action="store_true")
    extra, remaining = parser.parse_known_args(argv)
    if not 1 <= extra.requests <= 8:
        parser.error("--requests must be 1..8")
    if (extra.fused_producers or extra.fast_finite or extra.cache_adaln) and not extra.tensor_parallel:
        parser.error("producer/guard/cache options require --tensor-parallel")
    args = base.parser().parse_args(remaining)
    plan = base.build_plan(args)
    if args.disable:
        parser.error("persistent first deliverable requires the complete paper stack")
    command = plan["command"]
    command[command.index("src/inference/infer_ulysses.py")] = "src/inference/infer_persistent.py"
    plan["environment"]["VDN_PERSISTENT_REQUESTS"] = str(extra.requests)
    plan["environment"]["VDN_PERMANENT_TP"] = "1" if extra.tensor_parallel else "0"
    for key, enabled in (("FUSED_PRODUCERS", extra.fused_producers),
                         ("FAST_FINITE", extra.fast_finite), ("CACHE_ADALN", extra.cache_adaln)):
        plan["environment"]["VDN_TP_" + key] = "1" if enabled else "0"
    if extra.request_manifest:
        manifest = Path(extra.request_manifest).resolve(strict=True)
        plan["environment"]["VDN_PERSISTENT_MANIFEST"] = str(manifest)
    plan["paths"]["metrics"] = str(Path(plan["paths"]["run"]) / "batch.json")
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    base.actual_manifest = actual_manifest
    supervise = base.supervise_with_disk_guard
    def sampled(plan, run, root=base.ROOT):
        with (run / "device-memory.csv").open("x") as log:
            sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits", "-lms", "200"], stdout=log, stderr=subprocess.STDOUT)
            try:
                return supervise(plan, run, root)
            finally:
                sampler.terminate()
                sampler.wait(timeout=10)
    base.supervise_with_disk_guard = sampled
    return base.execute(plan)


if __name__ == "__main__":
    raise SystemExit(main())
