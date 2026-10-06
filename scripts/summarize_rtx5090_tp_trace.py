"""Measured all-rank TP transport inventory from existing Nsight SQLite exports."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from summarize_rtx5090_nsight import summarize, union_ns


def inspect(database):
    result = summarize(database)
    with sqlite3.connect(f"file:{database}?mode=ro",uri=True) as c:
        names = dict(c.execute("select id,value from StringIds"))
        ranges = list(c.execute("select n.start,n.end,n.globalTid,coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on n.textId=s.id where n.end is not null"))
        counts = Counter(name for _,_,_,name in ranges)
        result["tp_ranges"] = {k:v for k,v in counts.items() if k and k.startswith("tp.")}
        result["collective_ranges"] = {k:v for k,v in counts.items() if k and k.startswith("nccl")}
        result["weight_gather_ranges"] = {k:v for k,v in counts.items() if k and (k.startswith("prefetch.gather.") or k.startswith("weights.gather"))}
        result["attention_exchange_ranges"] = {k:v for k,v in counts.items() if k and ("allgather" in k.lower() or "inputexchange" in k or "outputexchange" in k)}
        kernel_rows = list(c.execute("select start,end,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL"))
        types = defaultdict(list)
        for lo,hi,stream,nameid in kernel_rows:
            name = names[nameid]
            if "nccl" in name.lower():
                types[name].append((lo,hi))
        result["nccl_kernels"] = {k:dict(count=len(v),union_seconds=union_ns(v)/1e9) for k,v in types.items()}
        copies = c.execute("select copyKind,count(*),sum(bytes),max(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind").fetchall()
        result["memcpy_by_kind"] = [dict(kind=k,count=n,bytes=b,largest=m) for k,n,b,m in copies]
        n = sum(v for k,v in result["tp_ranges"].items() if k.startswith("tp.all_reduce."))
        result["tp_payload_bytes_inferred_from_validated_geometry"] = n * 38510 * 5376 * 2
        result["payload_note"] = "414059520 bytes per validated BF16 residual SUM; bytes from workload shape, counts from NVTX and actual NCCL kernels. Not measured PCIe byte counters."
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    rows=[inspect(f) for f in sorted(args.directory.glob("rank-*.sqlite"))]
    if not rows:
        raise ValueError("no exported rank databases")
    with args.output.open("x") as f:
        json.dump(dict(reports=rows),f,indent=2)
    for r in rows:
        print(Path(r["sqlite"]).name,r["tp_ranges"],r["collective_ranges"],r["nccl_union_seconds"],flush=True)


if __name__ == "__main__":
    main()
