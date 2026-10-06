"""Summarize recorded attempts without importing Torch or touching the GPU."""

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/rtx5090"


def summarize(frames, attempt, events, log_path=None):
    log_path = log_path or OUT / f"render-{frames}-attempt{attempt}.log"
    log = log_path.read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time .*: ([\d:.]+)\n", log)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    status = re.search(r"Exit status: (\d+)", log)
    if not (elapsed and rss and status):
        raise ValueError(f"Attempt not finished: {log_path}")
    wall = 0.0
    for field in elapsed.group(1).split(":"):
        wall = wall * 60 + float(field)
    ends = [e for e in events if e["event"] == "phase_end"]
    phases = [{k: e[k] for k in (
        "phase", "seconds", "status", "cpu_rss_bytes", "cpu_peak_rss_bytes",
        "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "step_index"
    ) if k in e} for e in ends]
    return {
        "frames_requested": frames, "attempt": attempt,
        "run_id": events[0]["run_id"], "log": str(log_path.relative_to(ROOT)),
        "exit_status": int(status.group(1)), "process_seconds": wall,
        "process_peak_rss_bytes": int(rss.group(1)) * 1024,
        "peak_cuda_allocated_bytes": max(e["cuda_peak_allocated_bytes"] for e in ends),
        "peak_cuda_reserved_bytes": max(e["cuda_peak_reserved_bytes"] for e in ends),
        "phases": phases,
    }


def main():
    runs = []
    for frames, name in ((124, "8nfe_rtx5090"), (345, "8nfe_rtx5090_345")):
        path = ROOT / f"results/{name}.mp4.metrics.jsonl"
        grouped = {}
        for event in map(json.loads, path.read_text().splitlines()):
            grouped.setdefault(event["run_id"], []).append(event)
        for attempt, events in enumerate(grouped.values(), 1):
            result = summarize(frames, attempt, events)
            result["metrics_path"] = str(path.relative_to(ROOT))
            record_path = ROOT / f"results/{name}.mp4.inference.json"
            if result["exit_status"] == 0 and record_path.exists():
                record = json.loads(record_path.read_text())
                result["step_seconds_inference_record"] = record["timings"]["step_seconds"]
            runs.append(result)
    final_path = ROOT / "results/8nfe_rtx5090_124_final.mp4.metrics.jsonl"
    if final_path.exists():
        events = list(map(json.loads, final_path.read_text().splitlines()))
        if len({event["run_id"] for event in events}) != 1:
            raise ValueError("Final baseline journal must contain exactly one run")
        result = summarize(124, "final_code", events, OUT / "render-124-final.log")
        result["metrics_path"] = str(final_path.relative_to(ROOT))
        record = json.loads((ROOT / "results/8nfe_rtx5090_124_final.mp4.inference.json").read_text())
        result["step_seconds_inference_record"] = record["timings"]["step_seconds"]
        runs.append(result)
    summary = {
        "units": "seconds and bytes; GiB = bytes / 2**30",
        "notes": [
            "124 attempt 1 is the preserved pre-fix baseline; final_code is the post-fix rerun; 345 attempts retain distinct run IDs.",
            "CUDA values are Torch allocator peaks, not total device usage.",
            "RSS peaks are process-lifetime high-water marks, not phase-local peaks.",
            "Cold step includes transfers, compute, possible compilation; compile-only not isolated.",
            "Nested phases overlap. Cached-prompt rendering excludes new-prompt encoding.",
        ],
        "runs": runs,
    }
    path = OUT / "summary-metrics.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    for run in runs:
        print(f"{run['frames_requested']} attempt {run['attempt']}: exit={run['exit_status']} "
              f"process={run['process_seconds']:.3f}s RSS={run['process_peak_rss_bytes']/2**30:.3f}GiB")
        for phase in run["phases"]:
            print(f"  {phase['phase']}: {phase['seconds']:.3f}s "
                  f"allocated={phase['cuda_peak_allocated_bytes']/2**30:.3f}GiB "
                  f"reserved={phase['cuda_peak_reserved_bytes']/2**30:.3f}GiB "
                  f"RSS={phase['cpu_rss_bytes']/2**30:.3f}GiB {phase['status']}")


if __name__ == "__main__":
    main()
