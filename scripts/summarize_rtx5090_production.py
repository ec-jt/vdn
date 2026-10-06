"""CPU-only exact run/activity/latent comparison, exclusive report output."""
import argparse
import json
from pathlib import Path
import re


def activity(path):
    samples, current = [], None
    rss = []
    for line in Path(path).read_text().splitlines():
        if re.fullmatch(r'\d{4}-\d\d-\d\dT.*Z', line):
            current = {'utc': line, 'gpus': []}
            samples.append(current)
        elif re.match(r'^\d+,', line) and current is not None:
            index, util, memory, power = map(float, line.split(','))
            current['gpus'].append(dict(index=int(index), utilization=util, memory_mib=memory, power_w=power))
        elif line.startswith('aggregate_inference_rss_kib '):
            parts = line.split()
            if len(parts) == 2:
                rss.append(int(parts[1]))
    complete = [s for s in samples if len(s['gpus']) == 8]
    ranks = []
    for rank in range(8):
        rows = [g for s in complete for g in s['gpus'] if g['index'] == rank]
        ranks.append(dict(rank=rank, peak_memory_mib=max(g['memory_mib'] for g in rows),
                          peak_power_w=max(g['power_w'] for g in rows),
                          samples_nonzero_util=sum(g['utilization'] > 0 for g in rows),
                          mean_utilization=sum(g['utilization'] for g in rows)/len(rows)))
    return dict(samples=len(complete), first_utc=complete[0]['utc'], last_utc=complete[-1]['utc'],
                all_eight_nonzero_util_samples=sum(all(g['utilization'] > 0 for g in s['gpus']) for s in complete),
                aggregate_sampled_rss_peak_bytes=max(rss)*1024, ranks=ranks,
                caveat='Utilization includes NCCL wait kernels; compute proof requires rank profiles. RSS sums processes and may double count shared pages; sampled not instantaneous peak.')


def delta(actual, reference):
    import torch
    torch.set_num_threads(2)
    a = torch.load(actual, map_location='cpu', weights_only=True)
    b = torch.load(reference, map_location='cpu', weights_only=True)
    result = {}
    for key in ('video', 'audio'):
        assert a[key].shape == b[key].shape
        d = a[key].double() - b[key].double()
        result[key] = dict(shape=list(d.shape), finite=bool(torch.isfinite(a[key]).all()),
                           max_abs=d.abs().max().item(), mean_abs=d.abs().mean().item(),
                           rmse=d.square().mean().sqrt().item(),
                           relative_l2=(d.norm()/b[key].double().norm()).item(),
                           differing_values=torch.count_nonzero(d).item(), count=d.numel())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--media', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--activity', required=True)
    p.add_argument('--baseline-latents', required=True)
    p.add_argument('--baseline-denoise', type=float, required=True)
    p.add_argument('--baseline-whole', type=float, required=True)
    p.add_argument('--compare-latents')
    p.add_argument('--report', required=True)
    args = p.parse_args()
    record = json.loads(Path(args.media + '.inference.json').read_text())
    text = Path(args.log).read_text()
    elapsed = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', text).group(1)
    seconds = 0
    for part in elapsed.split(':'):
        seconds = seconds*60 + float(part)
    latents = str(Path('artifacts/latents')/(Path(args.media).name + '.latents.pt'))
    result = dict(media=args.media, log=args.log, activity_path=args.activity,
                  whole_process_seconds=seconds, timings=record['timings'],
                  activity=activity(args.activity), baseline_latents=args.baseline_latents,
                  baseline_delta=delta(latents, args.baseline_latents),
                  baseline_denoise_seconds=args.baseline_denoise, baseline_whole_seconds=args.baseline_whole,
                  denoise_speedup_vs_cpu_offload=args.baseline_denoise/record['timings']['denoise_seconds'],
                  whole_speedup_vs_cpu_offload=args.baseline_whole/seconds,
                  exit_status=int(re.search(r'Exit status: (\d+)', text).group(1)))
    if args.compare_latents:
        result['comparison_latents'] = args.compare_latents
        result['comparison_delta'] = delta(latents, args.compare_latents)
    with open(args.report, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('timings', 'activity')}, indent=2))


if __name__ == '__main__':
    main()
