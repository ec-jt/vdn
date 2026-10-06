"""Summarize the fixed paper124 experiment without modifying prior evidence."""
import hashlib
import json
from pathlib import Path
import re
import statistics


ROOT = Path('artifacts/rtx5090/paper-bf16-20261005')
SUFFIX = '20261005T1237Z'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = load(ROOT / 'model-protocol.json')
    runs = {}
    first_sources = None
    for name in protocol['order']:
        label = f'paper124-{name}-{SUFFIX}'
        directory = Path('artifacts/rtx5090') / label
        metrics_path = Path('results') / f'{label}.mp4.inference.json'
        data = load(metrics_path)
        timings = data['timings']
        assert (directory / 'exit.txt').read_text().strip() == '0'
        sources = load(directory / 'source-sha256.json')
        model_sources = {k: v for k, v in sources.items() if k.startswith(('src/', 'configs/'))}
        if first_sources is None:
            first_sources = model_sources
        assert model_sources == first_sources, name
        sampled = {rank: 0 for rank in range(8)}
        counts = {rank: 0 for rank in range(8)}
        for line in (directory / 'activity.log').read_text().splitlines():
            fields = line.split(',')
            if len(fields) == 4 and fields[0].strip().isdigit():
                rank = int(fields[0])
                sampled[rank] = max(sampled[rank], int(fields[2]))
                counts[rank] += 1
        denoise_ranks = timings['ranks']
        decode = timings['decoder_metrics']['distributed_video']
        profile = (directory / 'selectedprofile.txt').read_text().strip()
        mode = 'nsys' if name.endswith('nsys') else 'plain'
        launch = (directory / 'launch.log').read_text()
        elapsed = re.findall(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)', launch)
        runs[name] = {
            'label': label, 'profile': profile, 'mode': mode, 'exit_code': 0,
            'command': f'bash scripts/benchmark_rtx5090_nsight.sh {label} {mode} allgather next_block {profile}',
            'denoise_seconds': timings['denoise_seconds'],
            'decode_and_encode_seconds': timings['decode_and_encode_seconds'],
            'denoise_plus_decode_encode_seconds': timings['denoise_seconds'] + timings['decode_and_encode_seconds'],
            'distributed_video_decode_wall_seconds': decode['wall_seconds'],
            'rank0_video_kernel_call_seconds': decode['ranks'][0]['decode_seconds'],
            'model_setup_seconds_excluded': timings['model_setup_seconds'],
            'inference_end_to_end_seconds_not_acceptance_metric': timings['end_to_end_seconds'],
            'supervised_process_elapsed_not_acceptance_metric': elapsed,
            'step_seconds': timings['step_seconds'],
            'memory': {
                'denoise_ranks': [{k: r[k] for k in ('rank', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'cpu_peak_rss_bytes', 'startup_cuda_peak')} for r in denoise_ranks],
                'decode_ranks': [{k: r[k] for k in ('rank', 'peak_allocated_bytes', 'peak_reserved_bytes', 'peak_host_rss_bytes')} for r in decode['ranks']],
                'sampled_whole_device_peak_mib_by_rank': sampled,
                'device_sample_counts_by_rank': counts,
                'sampling_note': 'Approximately 1 second; whole-device samples, not allocator peaks or a guaranteed instantaneous maximum',
                'max_denoise_allocated_gib': max(r['cuda_peak_allocated_bytes'] for r in denoise_ranks) / 2**30,
                'max_denoise_reserved_gib': max(r['cuda_peak_reserved_bytes'] for r in denoise_ranks) / 2**30,
            },
            'all_ranks_finite': all(r['finite'] for r in denoise_ranks),
            'weights_by_rank': [r['weights'] for r in denoise_ranks],
            'metrics_path': str(metrics_path), 'metrics_sha256': sha(metrics_path),
            'source_hashes_path': str(directory / 'source-sha256.json'),
            'source_hashes_sha256': sha(directory / 'source-sha256.json'),
            'model_config_sources_identical_to_control1': model_sources == first_sources,
        }
    arithmetic = {}
    for field in ('denoise_seconds', 'decode_and_encode_seconds', 'denoise_plus_decode_encode_seconds'):
        controls = [runs[n][field] for n in ('control1', 'control2')]
        fused = [runs[n][field] for n in ('fused1', 'fused2')]
        cm, fm = statistics.mean(controls), statistics.mean(fused)
        gate = max(.01, (max(controls) - min(controls)) / cm, (max(fused) - min(fused)) / fm)
        paired = [(c - f) / c for c, f in zip(controls, fused)]
        arithmetic[field] = {
            'control_samples': controls, 'fused_samples': fused,
            'control_mean': cm, 'fused_mean': fm,
            'fused_minus_control_seconds': fm - cm,
            'fused_reduction_fraction': (cm - fm) / cm,
            'control_range_fraction': (max(controls) - min(controls)) / cm,
            'fused_range_fraction': (max(fused) - min(fused)) / fm,
            'noise_gate_fraction': gate, 'paired_reduction_fractions': paired,
            'passes_fixed_performance_rule': (cm - fm) / cm > gate and all(x > 0 for x in paired),
            'overlap_only_sample': runs['overlap1'][field],
            'overlap_reduction_vs_control_mean_fraction': (cm - runs['overlap1'][field]) / cm,
            'formula': 'reduction=(control_mean-candidate_mean)/control_mean; gate=max(.01,control_range/control_mean,fused_range/fused_mean); both paired reductions must be positive',
        }
    controls = load(ROOT / 'model-controls-verification.json')
    candidates = load(ROOT / 'model-candidates-verification.json')
    numerical = []
    for report in (controls, candidates):
        for candidate in report['candidates']:
            numerical.append({
                'reference': report['reference_label'],
                **{k: candidate[k] for k in ('label', 'expectation', 'passed', 'latents', 'decoded_media')},
            })
    fused_entries = [c for c in candidates['candidates'] if c['expectation'] == 'fused']
    reproducible = all(
        len({c[group][kind]['candidate']['sha256'] for c in fused_entries}) == 1
        for group in ('latents', 'decoded_media') for kind in ('video', 'audio')
    )
    current_mismatch = [k for k, v in first_sources.items() if sha(k) != v]
    assert not current_mismatch
    unittest_log = (ROOT / 'model-unittest.log').read_text()
    pytest_log = (ROOT / 'model-pytest.log').read_text()
    assert 'Ran 201 tests' in unittest_log and '\nOK\n' in unittest_log
    assert '208 passed, 18 warnings, 482 subtests passed' in pytest_log
    assert ' skipped' not in pytest_log
    result = {
        'protocol': protocol, 'protocol_sha256': sha(ROOT / 'model-protocol.json'),
        'scope_note': 'Final tensors after the complete 8-NFE trajectory, not per-step saved tensors; decode/encode separated; no new345 runs',
        'runs': runs, 'arithmetic': arithmetic,
        'prior_denoise_mean_seconds': 115.720364,
        'fresh_control_denoise_mean_minus_prior_seconds': arithmetic['denoise_seconds']['control_mean'] - 115.720364,
        'numerical': numerical,
        'fused_all_three_runs_latent_and_decoded_hashes_identical': reproducible,
        'decision': {
            'fused': 'REJECT: video/audio final trajectory relativeL2 exceeds predeclared0.001 gate; no reliable unprofiled speed win. Research-only/default-off, no fast recommendation.',
            'overlap': 'EXACT parity but one unprofiled result is within noise. No demonstrated speed win; default-off retained. No combined run.',
            'recommended_profile': 'baseline allgather next_block, preserving corrected distributed reference; no CPU-reference or quality equivalence claim',
            'fused_numerical_pass': all(c['passed'] for c in fused_entries),
            'fused_performance_pass': arithmetic['denoise_plus_decode_encode_seconds']['passes_fixed_performance_rule'],
            'controls_exact_pass': controls['passed'],
            'overlap_exact_pass': next(c['passed'] for c in candidates['candidates'] if c['expectation'] == 'exact'),
        },
        'regression': {
            'prior172_retained': 172, 'missing_prior_ids': 0,
            'pre_verifier_suite': {'unittest_methods': 185, 'pytest_only_cases': 7, 'total': 192},
            'new_verifier_methods': 16, 'unittest_passed': 201, 'pytest_passed': 208,
            'pytest_subtests_passed': 482, 'skips': 0, 'failures': 0, 'pytest_warnings': 18,
            'unittest_seconds': 50.329, 'pytest_seconds': 68.52,
            'environment': {'CUDA_VISIBLE_DEVICES': '0', 'RUN_ULYSSES_REFERENCE_CUDA': '1', 'RUN_VDA_OVERLAP_CUDA': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            'commands': ['.venv/bin/python scripts/supervise_rtx5090.py --seconds 900 -- .venv/bin/python -B -m unittest discover -s tests -p "test_*.py" -v', '.venv/bin/python scripts/supervise_rtx5090.py --seconds 900 -- .venv/bin/python -B -m pytest -q -ra tests'],
            'inventory': str(ROOT / 'model-test-inventory.json'),
        },
        'failures_and_limitations': [
            'Candidate verification exit1 is the expected recorded gate rejection, not an inference process failure. All six inference exits, both regression exits, all16 exports and both Nsight analyses exit0.',
            'Large fused drift repeats exactly across two unprofiled runs and the profiled run; all controls/overlap exact. Root cause within full deployed trajectory not isolated; local tiny synthetic branch drift is not representative evidence.',
            'No model source edits, tolerance relaxation, corrective reruns or default promotion. Bounded evaluation stops at reproducible rejection.',
            'Nsight driver compatibility/potential event-loss warnings: structural step/block/job coverage passes, but lossless capture is not established.',
            'Only two controls and two fused plain runs, one overlap plain run. No statistical confidence claim. Startup/JIT remains outside optimization scope.',
            'Saved tensors are final trajectory states only. Original CPU path not used as this experiment reference.',
            'Initial combined handoff read used shell brace expansion unsupported by tool shell; explicit path read immediately succeeded without any writes.',
        ],
        'source_preservation': {'model_config_hashes_unchanged_all_runs_and_current': not current_mismatch, 'changed_model_config_paths': current_mismatch, 'initial_status': str(ROOT / 'model-initial-git-status.txt'), 'initial_diff': str(ROOT / 'model-initial-working.diff')},
        'nsight': {'before': 'pf124-nsys-20261005T1007Z', 'after': runs['fused-nsys']['label'], 'reports': 16, 'new_sqlite_exports': 16, 'performance_claim': False, 'analysis': str(ROOT / 'model-nsys-analysis.json'), 'host_wait_analysis': str(ROOT / 'model-nsys-waits.json'), 'handoff': str(ROOT / 'model-nsys-HANDOFF.md')},
        'verification_evidence': [str(ROOT / name) for name in ('model-controls-verification.json', 'model-candidates-verification.json', 'model-verifier-HANDOFF.md')],
    }
    with (ROOT / 'measured-summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'arithmetic': arithmetic, 'decision': result['decision'], 'memory': {n: {k: v for k, v in r['memory'].items() if k in ('max_denoise_allocated_gib', 'max_denoise_reserved_gib', 'sampled_whole_device_peak_mib_by_rank')} for n, r in runs.items()}}, indent=2))


if __name__ == '__main__':
    main()
