"""Debug-only first real block replay; exits before the remaining trajectory."""
import hashlib
import json
import os
from pathlib import Path
import functools

import torch
import torch.distributed as dist

from src.inference import infer_ulysses
from src.models.ops import fused_block


class ReplayDone(BaseException):
    pass


def byte_cpu(t):
    return t.detach().contiguous().view(torch.uint8).cpu()


def replay(block, hidden, temb, indices, rotary, **kwargs):
    paper = block.attn._paper_runtime
    runtime = block.attn._ulysses_runtime
    shard = runtime.weight_shards
    rank = runtime.rank
    output = Path(os.environ['VDN_REPLAY_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    sources = (hidden, temb, indices, *rotary, *kwargs.get('tables', ()))
    originals = [byte_cpu(t) for t in sources]
    report = {'rank': rank, 'input_shapes': [list(t.shape) for t in sources],
              'input_sha256': [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in originals],
              'repeats': []}
    original_prefetch = shard.prefetch_next
    shard.prefetch_next = lambda: None
    tracing = os.environ.get('VDN_REPLAY_TRACE') == '1'
    trace_active = [tracing]
    trace_counter = [0]
    restores = []
    def snap(name, value):
        if not trace_active[0]:
            return
        def cpu(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu()
            if isinstance(v, (tuple,list)):
                return [cpu(x) for x in v]
            if isinstance(v, dict):
                return {k:cpu(x) for k,x in v.items()}
            return v if isinstance(v, (str, int, float, bool, type(None))) else None
        directory = output / f'trace-rank-{rank}'
        directory.mkdir(exist_ok=True)
        torch.save(cpu(value), directory / f'{trace_counter[0]:03d}-{name}.pt')
        trace_counter[0] += 1
    def wrap(owner, name):
        fn = getattr(owner,name)
        @functools.wraps(fn)
        def traced(*args,**kw):
            snap(name+'-input', (args,kw))
            result=fn(*args,**kw)
            snap(name+'-output',result)
            return result
        restores.append((owner,name,fn))
        setattr(owner,name,traced)
    if tracing:
        snap('weights',dict(block.named_parameters()) | dict(block.named_buffers()))
        from src.models.linear_attention import paper_stack
        from src.models.ops import mxfp8_linear
        from src.models.softmax_attention import paper_window, paper_qk_rope
        for name in ('paper_prepare','paper_statistics','paper_delta_factors','paper_boundary_scan','paper_gather','vdn_linear_epilogue'):
            wrap(paper_stack,name)
        for name in ('quantize_activation','project_qkv'):
            wrap(mxfp8_linear,name)
        wrap(paper_window,'paper_window_attention')
        wrap(paper_qk_rope,'paper_qk_rmsnorm_rope')
        for name,module in block.named_modules():
            if isinstance(module,mxfp8_linear.MXFP8Linear):
                wrap(module,'forward_quantized')
        old_compiled = fused_block._compiled
        def traced_compiled(name,body,**compile_options):
            fn=old_compiled(name,body,**compile_options)
            def call(*args,**kw):
                snap('compiled-'+name+'-input',(args,kw))
                result=fn(*args,**kw)
                snap('compiled-'+name+'-output',result)
                return result
            return call
        restores.append((fused_block,'_compiled',old_compiled))
        fused_block._compiled=traced_compiled
    expected = None
    try:
        for mode in ('overlap', 'serial', 'synchronized'):
            paper.options['paper_vda_overlap'] = mode != 'serial'
            for iteration in range(3):
                if mode == 'synchronized':
                    torch.cuda.synchronize()
                actual = original(block, hidden, temb, indices, rotary, **kwargs)
                trace_active[0] = False
                torch.cuda.synchronize()
                current = actual.cpu()
                if expected is None:
                    expected = current.clone()
                    torch.save({'inputs': [t.cpu() for t in sources], 'output': current},
                               output / f'first-rank-{rank}.pt')
                delta = current.double() - expected.double()
                row = {'mode': mode, 'iteration': iteration,
                       'byte_exact': torch.equal(byte_cpu(current), byte_cpu(expected)),
                       'maxabs': delta.abs().max().item(),
                       'relative_l2': (delta.norm()/expected.double().norm()).item()}
                report['repeats'].append(row)
                print('oneblock_replay=' + json.dumps(dict(rank=rank, **row)), flush=True)
                del actual
        report['inputs_byte_exact_after'] = all(torch.equal(byte_cpu(t), b) for t,b in zip(sources, originals))
        (output / f'rank-{rank}.json').write_text(json.dumps(report,indent=2))
    finally:
        shard.prefetch_next = original_prefetch
        for owner,name,fn in reversed(restores):
            setattr(owner,name,fn)
    raise ReplayDone()


original = fused_block.paper_block_forward
fused_block.paper_block_forward = replay
try:
    infer_ulysses.main()
except ReplayDone:
    print('oneblock_replay_completed', flush=True)
finally:
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.destroy_process_group()
