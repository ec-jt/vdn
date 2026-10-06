"""Fail closed on both repeated full-geometry decoder parity results."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', type=int, choices=(124, 345))
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    frames = args.frames
    d = json.loads(args.report.read_text())
    latent_t, clips, jobs = {124: (37, 7, 196), 345: (102, 20, 560)}[frames]
    assert d['world_size'] == 8
    assert d['input_shape'] == [1, 24, latent_t, 48, 84]
    assert len(d['runs']) == 2
    for run in d['runs']:
        assert run['output_frames'] == frames
        assert run['clips'] == clips and run['total_jobs'] == jobs
        assert run['tiles_per_clip'] == 28
        assert run['rank_job_counts'] == [sum(j % 8 == rank for j in range(jobs)) for rank in range(8)]
        p = run['parity']
        assert p['raw_shape'] == [1, 3, frames, 768, 1344]
        assert p['raw_dtype'] == 'torch.float16' and p['finite'] is True
        for key in ('float_max_abs', 'float_mean_abs', 'float_rmse', 'float_differing_values',
                    'byte_max_abs', 'byte_mean_abs', 'byte_differing_values'):
            assert p[key] == 0, (frames, key, p[key])
        assert p['float_values'] == p['byte_values'] == 3 * frames * 768 * 1344
        assert p['reference_rgb_sha256'] == p['distributed_rgb_sha256']
    assert d['runs'][0]['parity']['distributed_rgb_sha256'] == d['runs'][1]['parity']['distributed_rgb_sha256']
    print(f'{frames}-frame decoder parity passed for both full-geometry calls')


if __name__ == '__main__':
    main()
