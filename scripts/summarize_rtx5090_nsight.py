"""Export bounded per-rank Nsight reports and summarize actual CUDA/NVTX evidence."""
import argparse
from collections import defaultdict
import heapq
import json
from pathlib import Path
import sqlite3
import subprocess


def union_ns(intervals):
    total = 0
    end = -1
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, end))
        end = max(end, hi)
    return total


def summarize(path):
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    names = dict(c.execute('select id,value from StringIds'))
    kernels = list(c.execute('select start,end,deviceId,globalPid,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'))
    ranges = list(c.execute('select start,end,globalTid,coalesce(text,s.value) from NVTX_EVENTS n left join StringIds s on n.textId=s.id where end is not null'))
    runtime = list(c.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'))
    host = defaultdict(lambda: [0, 0])
    by_tid = defaultdict(list)
    step_by_tid = defaultdict(list)
    for start, end, tid, name in ranges:
        name = name or '<unnamed>'
        host[name][0] += 1
        host[name][1] += (end - start) / 1e9
        # NCCL adds its own inner ranges. Keep application-level attribution.
        if not name.startswith('nccl'):
            by_tid[tid].append((start, end, name))
        if name.startswith('denoise.step.'):
            step_by_tid[tid].append((start, end, name))
    # Attribution uses launch API timestamp and innermost enclosing host NVTX range.
    # These are summed kernel durations, not disjoint critical-path time.
    correlation = {}
    steps = {}
    runtime_by_tid = defaultdict(list)
    for start, end, tid, corr in runtime:
        runtime_by_tid[tid].append((start, corr))
    for tid, apis in runtime_by_tid.items():
        values = sorted(by_tid[tid])
        active = []
        i = 0
        for start, corr in sorted(apis):
            while i < len(values) and values[i][0] <= start:
                lo, hi, name = values[i]
                heapq.heappush(active, (hi - lo, hi, name))
                i += 1
            while active and active[0][1] < start:
                heapq.heappop(active)
            correlation[(tid >> 24, corr)] = active[0][2] if active else '<outside NVTX>'
            steps[(tid >> 24, corr)] = next((name for lo, hi, name in step_by_tid[tid] if lo <= start <= hi), 'other')
    attributed = defaultdict(lambda: [0, 0])
    by_step = defaultdict(lambda: [0, 0])
    kernel_names = defaultdict(lambda: [0, 0])
    devices = defaultdict(lambda: [0, 0])
    nccl, other = [], []
    for start, end, device, pid, corr, name_id in kernels:
        seconds = (end - start) / 1e9
        name = names[name_id]
        key = correlation.get((pid >> 24, corr), '<uncorrelated>')
        attributed[key][0] += 1; attributed[key][1] += seconds
        step_key = steps.get((pid >> 24, corr), 'other') + '/' + key
        by_step[step_key][0] += 1; by_step[step_key][1] += seconds
        kernel_names[name][0] += 1; kernel_names[name][1] += seconds
        devices[f'pid={(pid >> 24) & 0xffffff},device={device}'][0] += 1
        devices[f'pid={(pid >> 24) & 0xffffff},device={device}'][1] += seconds
        (nccl if 'nccl' in name.lower() else other).append((start, end))
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    memcpy = []
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        memcpy = list(c.execute('select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'))
    def rows(data):
        return [{'name': k, 'count': v[0], 'sum_seconds': v[1]} for k, v in sorted(data.items(), key=lambda kv: -kv[1][1])]
    result = dict(sqlite=str(path), kernel_count=len(kernels), nvtx_count=len(ranges),
                  kernel_pid_device=dict(devices), nvtx_host_inclusive=rows(host),
                  kernel_attribution=rows(attributed), kernel_attribution_by_step=rows(by_step),
                  top_kernels=rows(kernel_names)[:30],
                  kernel_span_seconds=(max(k[1] for k in kernels)-min(k[0] for k in kernels))/1e9,
                  kernel_union_seconds=union_ns([(k[0],k[1]) for k in kernels])/1e9,
                  nccl_union_seconds=union_ns(nccl)/1e9,
                  non_nccl_union_seconds=union_ns(other)/1e9,
                  memcpy_count=len(memcpy), memcpy_bytes=sum(r[2] for r in memcpy),
                  memcpy_sum_seconds=sum(r[1]-r[0] for r in memcpy)/1e9)
    c.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-name', default='nsight-summary.json')
    args = parser.parse_args()
    results = []
    reports = sorted(args.directory.glob('rank-*.nsys-rep'))
    if len(reports) != 16:
        raise ValueError(f'Expected 8 ranks x 2 capture windows, got {len(reports)}')
    for report in reports:
        database = report.with_suffix('.sqlite')
        if not database.exists():
            with report.with_suffix('.export.log').open('x') as log:
                subprocess.run(['/usr/local/cuda/bin/nsys', 'export', '--type=sqlite',
                                '--output=' + str(database), str(report)],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        result = summarize(database)
        results.append(result)
        print(database.name, result['kernel_count'], result['nvtx_count'], result['kernel_pid_device'], flush=True)
    with (args.directory / args.output_name).open('x') as f:
        json.dump(dict(note='NVTX host inclusive; kernel sums can overlap. Union times are per report, not cross-rank critical paths.', reports=results), f, indent=2)


if __name__ == '__main__':
    main()
