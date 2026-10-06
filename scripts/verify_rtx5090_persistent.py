"""Complete CPU numerical/media verification and persistent latency summary."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from verify_rtx5090_paper124 import compare_tensors, compare_streams, decode_arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    batch = json.loads((args.run / "batch.json").read_text())
    reference = torch.load(args.reference / "latents/render.mp4.latents.pt", map_location="cpu", weights_only=True)
    control = torch.load(args.control / "latents/render.mp4.latents.pt", map_location="cpu", weights_only=True)
    samples = []
    for row in csv.reader((args.run / "device-memory.csv").open()):
        if len(row) != 4:
            continue
        try:
            stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            samples.append((stamp, int(row[1]), int(row[2])))
        except ValueError:
            continue
    result = dict(run=str(args.run), reference=str(args.reference), control=str(args.control),
        model_setup_seconds=batch["model_setup_seconds"], process_seconds=batch["process_seconds"],
        policy="Deterministic BF16/low-precision drift accepted for performance, not a quality-equivalence claim",
        requests=[])
    for request in batch["requests"]:
        latents = torch.load(request["latents"], map_location="cpu", weights_only=True)
        entry = dict(request=request["request"], latents={}, control_delta={}, media={})
        for name in ("video", "audio"):
            entry["latents"][name] = compare_tensors(reference[name], latents[name], 1 << 20)
            entry["control_delta"][name] = compare_tensors(control[name], latents[name], 1 << 20)
            infos = [{}, {}]
            entry["media"][name] = dict(delta=compare_streams(
                decode_arrays(args.reference / "render.mp4", name, infos[0]),
                decode_arrays(request["media"], name, infos[1]), 1 << 20),
                reference=infos[0], candidate=infos[1])
        ranks = request["ranks"]
        entry["workers"] = [dict(rank=r["rank"], pid=r["pid"], group=r["process_group_identity"],
                                 weight_group=r["weight_group_identity"]) for r in ranks]
        for key in ("request_seconds", "preparation_seconds", "denoise_loop_seconds", "latent_write_seconds",
                    "peak_allocated_bytes", "peak_reserved_bytes", "resident_allocated_bytes"):
            entry[key] = max(r[key] for r in ranks)
        lo = min(r["started_unix"] for r in ranks)
        hi = max(r["finished_unix"] for r in ranks)
        entry["device_peak_mib_by_rank"] = [max((m for t, rank, m in samples if rank == i and lo <= t <= hi), default=0)
                                                   for i in range(8)]
        entry["device_sampling_interval_ms"] = 200
        entry["decode_seconds"] = max(r["decoder"]["total_seconds"] for r in ranks)
        entry["video_decode_seconds"] = max(r["decoder"]["distributed_video"]["wall_seconds"] for r in ranks)
        entry["root_decoder"] = ranks[0]["decoder"]["root"]
        entry["gather_count_by_rank"] = [r["weights"]["gather_count"] for r in ranks]
        entry["pass"] = all(entry["latents"][n]["byte_exact"] and entry["latents"][n]["finite"]
            and entry["media"][n]["delta"]["byte_exact"] and entry["media"][n]["delta"]["finite"]
            for n in ("video", "audio"))
        result["requests"].append(entry)
        print(f"request {request['request']}: exact={entry['pass']} wall={entry['request_seconds']:.6f}s", flush=True)
    result["workers_stable"] = all(r["workers"] == result["requests"][0]["workers"] for r in result["requests"])
    result["pass"] = result["workers_stable"] and all(r["pass"] for r in result["requests"])
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
