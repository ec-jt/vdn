"""Controlled production-shape component repeats, not a renderer benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.models.linear_attention.paper_stack import (
    paper_prepare, paper_statistics, paper_boundary_plan, paper_boundary_scan,
    paper_gather,
)
from src.models.linear_attention.paper_factor import paper_delta_factors
from src.models.ops.mxfp8_kernels import launch_quantize


def raw(t):
    return t.detach().contiguous().view(torch.uint8).cpu()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(711)
    torch.backends.cuda.matmul.allow_tf32 = False
    f, h, s, d = 37, 7, 1008, 128
    inputs = [torch.randn(f*s, h, d, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    beta = torch.rand(f, h, s, device='cuda')
    alpha = torch.rand(f, h, d, device='cuda') * .1 + .9
    bounds = [(i//4*4, min(f-1, i//4*4+3)) for i in range(f)]
    plan = paper_boundary_plan(bounds, f, inputs[0].device)
    originals = [raw(t) for t in (*inputs, beta, alpha)]
    report = {'geometry': [f,h,s,d], 'input_sha256': [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in originals], 'stages': []}

    def check(name, fn, sources):
        expected = tuple(fn())
        expected_bytes = [raw(t) for t in expected]
        for mode in ('caller', 'side_delayed'):
            mismatches = []
            side = torch.cuda.Stream()
            for iteration in range(args.repeats):
                caller = torch.cuda.current_stream()
                stream = caller if mode == 'caller' else side
                stream.wait_stream(caller)
                with torch.cuda.stream(stream):
                    if mode != 'caller':
                        torch.cuda._sleep(1000000)
                    actual = tuple(fn())
                caller.wait_stream(stream)
                equal = [torch.equal(raw(t), b) for t,b in zip(actual,expected_bytes)]
                if not all(equal):
                    mismatches.append({'iteration': iteration, 'exact': equal,
                        'maxabs': [(a.float()-b.float()).abs().max().item() for a,b in zip(actual,expected)]})
                del actual
                churn = [torch.empty_like(t).fill_(iteration + 13) for t in sources]
                del churn
            row = {'stage':name,'mode':mode,'repeats':args.repeats,'mismatches':mismatches}
            report['stages'].append(row)
            print(json.dumps(row), flush=True)
            Path(args.output).write_text(json.dumps(report,indent=2))
        return expected

    q, = check('prepare_q', lambda: (paper_prepare(inputs[0],frames=f,spatial=s,l2norm=True),), inputs[:1])
    k, = check('prepare_k', lambda: (paper_prepare(inputs[1],frames=f,spatial=s,l2norm=True),), inputs[1:2])
    v, = check('prepare_v', lambda: (paper_prepare(inputs[2],frames=f,spatial=s,l2norm=False),), inputs[2:])
    a,b = check('statistics',lambda:paper_statistics(k,v,beta),[k,v,beta])
    t,j = check('factors',lambda:paper_delta_factors(a,b,alpha),[a,b,alpha])
    bank, = check('scan',lambda:(paper_boundary_scan(t,j,plan),),[t,j])
    state, = check('gather',lambda:(paper_gather(bank,alpha,plan),),[bank,alpha])
    check('readout',lambda:(q @ state.transpose(-1,-2),),[q,state])
    x = torch.randn(4736,5376,device='cuda',dtype=torch.bfloat16)
    check('mxfp8_quant',lambda:launch_quantize(x),[x])
    report['inputs_byte_exact_after'] = all(torch.equal(raw(t),b) for t,b in zip((*inputs,beta,alpha), originals))
    Path(args.output).write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
