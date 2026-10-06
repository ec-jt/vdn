"""Export 8 x 3 bounded persistent captures and inventory actual weight traffic."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import subprocess

from summarize_rtx5090_nsight import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = sorted(args.directory.glob("rank-*.nsys-rep"))
    if len(reports) != 24:
        raise ValueError(f"expected 8 ranks x 3 captures, got {len(reports)}")
    results = []
    for report in reports:
        database = report.with_suffix(".sqlite")
        if not database.exists():
            with report.with_suffix(".export.log").open("x") as log:
                subprocess.run(["/usr/local/cuda/bin/nsys", "export", "--type=sqlite",
                    "--output=" + str(database), str(report)], stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=180)
        item = summarize(database)
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as c:
            ranges = Counter(name for (name,) in c.execute(
                "select coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on n.textId=s.id"))
            copies = c.execute("select copyKind,count(*),sum(bytes),max(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind").fetchall()
            kernels = c.execute("select s.value,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id group by s.value").fetchall()
        item["memcpy_by_kind"] = [dict(kind=k, count=n, bytes=b, largest_bytes=m) for k,n,b,m in copies]
        item["persistent_ranges"] = {k:v for k,v in ranges.items() if k and k.startswith("persistent.")}
        item["forbidden_setup_ranges"] = {k:v for k,v in ranges.items() if k and (
            k == "persistent.model_setup_once" or "decoder.video_load_once" in k or "decoder.video_onload_once" in k
            or "decoder.audio_load_once" in k or k in ("paper.prepare.mxfp8_cpu_conversion", "paper.prepare.mxfp8_meta_conversion"))}
        item["all_kernel_counts"] = dict(kernels)
        results.append(item)
        print(report.name, item["kernel_count"], item["memcpy_by_kind"], item["persistent_ranges"], flush=True)
    with args.output.open("x") as handle:
        json.dump(dict(reports=results, count=len(results),
            note="Separate request-two steps2/3, resident decode, and request2->3 boundary. Preparation legitimately stages rank-zero CPU AdaLN source weights; no denoiser shard distribution or decoder weight reload is expected."), handle, indent=2)


if __name__ == "__main__":
    main()
