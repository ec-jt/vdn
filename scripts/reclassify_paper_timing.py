"""Reclassify retained measurements without modifying historical records."""
import json
from pathlib import Path

ROOT = Path('artifacts/rtx5090/paper-stack-20261005')
MODEL = {'mxfp8_cpu_conversion', 'mxfp8_meta_conversion', 'vda_factor_extension'}
REQUEST = {'prompt_layout_and_masks', 'schedule_and_window_plan',
           'adaln_precompute', 'adaln_cpu_transport_and_device_tables'}


def classify(record):
    t = record['timings']
    ranks = [r['paper_stack']['preparation_seconds'] for r in t['ranks']
             if r.get('paper_stack')]
    if any(set(r) - MODEL - REQUEST for r in ranks):
        raise ValueError('Unclassified preparation interval')
    rows = [dict(rank=i, model_setup=sum(r.get(k, 0) for k in MODEL),
                 request_prepare=sum(r.get(k, 0) for k in REQUEST), intervals=r)
            for i,r in enumerate(ranks)]
    # A rank sum is a lower bound on the distributed critical path. The sum of
    # phase maxima is an upper accounting bound, not an independently measured
    # wall interval. All phases execute sequentially in the current assembly.
    lower = max((r['request_prepare'] for r in rows), default=0)
    upper = sum(max((r.get(k, 0) for r in ranks), default=0) for k in REQUEST)
    loop = t['denoise_loop_seconds']
    decoder = t['decoder_metrics']
    decode = decoder['distributed_video']['wall_seconds'] + decoder['audio_decode_seconds']
    return dict(
        historical_model_setup_including_request_prepare_seconds=t['model_setup_seconds'],
        legacy_model_plus_request_preparation_charge_seconds=t['paper_preparation_added_seconds'],
        legacy_charged_denoise_seconds=t['denoise_seconds'],
        model_setup_excluding_measured_request_prepare_bounds_seconds=[t['model_setup_seconds']-upper,t['model_setup_seconds']-lower],
        request_prepare_bounds_seconds=[lower,upper],
        request_denoise_bounds_seconds=[loop+lower,loop+upper],
        request_denoise_plus_video_audio_decode_bounds_seconds=[loop+lower+decode,loop+upper+decode],
        denoise_loop_seconds=loop, video_decode_seconds=decoder['distributed_video']['wall_seconds'],
        audio_decode_seconds=decoder['audio_decode_seconds'], decode_encode_phase_seconds=t['decode_and_encode_seconds'],
        first_process_instrumented_end_to_end_seconds=t['end_to_end_seconds'],
        preparation_components_by_rank=rows,
        request_bounds_complete_for_instrumented_paper_phases=bool(ranks),
    )


def main():
    runs = []
    for p in sorted((ROOT/'integration').glob('paper-warm-*/render.mp4.inference.json')):
        runs.append(dict(label=p.parent.name, source=str(p), **classify(json.loads(p.read_text()))))
    result = dict(
        note='Historical source measurements remain unchanged. Disjoint cost classes, not summed cross-rank wall times. Request critical-path bounds use max rank sum and sum phase maxima. Baselines have no separately instrumented request preparation; their zero is not proof of zero cost. No repeated-request service was measured.',
        model_setup_components=sorted(MODEL), request_prepare_components=sorted(REQUEST), runs=runs)
    (ROOT/'timing-reclassified.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in runs:
        print(r['label'], 'request prepare',r['request_prepare_bounds_seconds'],
              'request denoise',r['request_denoise_bounds_seconds'],
              '+decode',r['request_denoise_plus_video_audio_decode_bounds_seconds'])


if __name__ == '__main__':
    main()
