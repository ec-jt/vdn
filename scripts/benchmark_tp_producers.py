"""Full-row producer/quantization and finite guard probe, no model weights."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.models.ops.fused_block import _compiled, _pre_ref, _swiglu_ref
from src.models.ops.mxfp8_kernels import launch_prenorm_quantize, launch_swiglu_quantize
from src.models.ops.mxfp8_quant import quantize_activation
from src.inference.utils.paper_runtime import CoordinatedFiniteGuard


def bench(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_bytes = torch.cuda.memory_allocated()
    samples = []
    for _ in range(15):
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b))
    return dict(median_ms=statistics.median(samples), samples_ms=samples,
                peak_extra_bytes=torch.cuda.max_memory_allocated()-start_bytes)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(429)
    rows,k = 38510,5376
    x = torch.randn(rows,k,device='cuda',dtype=torch.bfloat16)
    w = torch.randn(k,device='cuda',dtype=torch.bfloat16)
    tables = torch.randn(6,k*6,device='cuda',dtype=torch.bfloat16)
    scale,shift = tables.chunk(6,-1)[:2]
    indices = torch.arange(rows,device='cuda')%6
    oldpre = _compiled('pre',_pre_ref,deterministic=True)
    oldsw = _compiled('swiglu',_swiglu_ref,deterministic=True)
    report = dict(rows=rows,k=k,ffn_k=1792,warmups=2,repeats=15)
    report['prenorm_old'] = bench(lambda:quantize_activation(oldpre(x,w,1e-6,scale,shift,indices),check_finite=False))
    for dual in (True,False):
        report['prenorm_dual' if dual else 'prenorm_only'] = bench(lambda:launch_prenorm_quantize(x,w,1e-6,scale,shift,indices,write_bf16=dual))
    y,q,s,f = launch_prenorm_quantize(x,w,1e-6,scale,shift,indices,write_bf16=True)
    reference = oldpre(x,w,1e-6,scale,shift,indices)
    report['prenorm_vs_old_relative_l2'] = float((y.float()-reference.float()).norm()/reference.float().norm())
    qr,sr = quantize_activation(y,check_finite=False)
    report['prenorm_quant_bytes_exact'] = torch.equal(q,qr) and torch.equal(s,sr)
    report['prenorm_finite'] = bool(f.all())
    del y,q,s,f,qr,sr,reference
    h = torch.randn(rows,3584,device='cuda',dtype=torch.bfloat16)
    report['swiglu_old'] = bench(lambda:quantize_activation(oldsw(h),check_finite=False))
    report['swiglu_fused'] = bench(lambda:launch_swiglu_quantize(h))
    q,s,f = launch_swiglu_quantize(h)
    qr,sr = quantize_activation(oldsw(h),check_finite=False)
    report['swiglu_quant_bytes_exact'] = torch.equal(q,qr) and torch.equal(s,sr)
    report['swiglu_finite'] = bool(f.all())
    for fast in (False,True):
        guard = CoordinatedFiniteGuard('cuda',fast=fast)
        report['finite_fast' if fast else 'finite_strict'] = bench(lambda:guard.check('probe',x,h))
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
