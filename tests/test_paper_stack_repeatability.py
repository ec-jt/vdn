"""Regression for paper-only deterministic reduction compilation."""
import os
import subprocess
import sys

import pytest
import torch

from src.models.ops import fused_block


def test_paper_compilation_policy_is_scoped_and_cache_is_separate(monkeypatch):
    calls = []
    def compile_fn(body, **options):
        calls.append(options)
        return object()
    monkeypatch.setattr(torch, 'compile', compile_fn)
    monkeypatch.setattr(fused_block, '_CACHE', {})
    legacy = fused_block._compiled('pre', fused_block._pre_ref)
    paper = fused_block._compiled('pre', fused_block._pre_ref, deterministic=True)
    assert legacy is not paper
    assert fused_block._compiled('pre', fused_block._pre_ref) is legacy
    assert fused_block._compiled('pre', fused_block._pre_ref, deterministic=True) is paper
    assert calls == [dict(dynamic=False, options=None),
                     dict(dynamic=False, options={'deterministic': True})]


def test_only_paper_block_requests_deterministic_compile(monkeypatch):
    seen = []
    def forward(*args, **kwargs):
        seen.append(kwargs)
        return 'result'
    monkeypatch.setattr(fused_block, 'fast_block_forward', forward)
    assert fused_block.paper_block_forward(None, None, None, None, None,
                                            tables='tables', fused=True) == 'result'
    assert seen == [dict(adaln_tables='tables', deterministic_compile=True)]


@pytest.mark.skipif(os.environ.get('RUN_PAPER_REPEAT_CUDA') != '1',
                    reason='explicit fresh-process production-shape CUDA test')
def test_fresh_process_real_shape_delayed_stream_and_allocator_churn(tmp_path):
    code = r'''
import sys, torch
from src.models.ops.fused_block import _compiled, _pre_ref
torch.manual_seed(711)
with torch.no_grad():
    h = torch.randn(1, 4814, 5376, dtype=torch.bfloat16, device='cuda')
    w = torch.randn(5376, dtype=torch.bfloat16, device='cuda')
    tables = torch.randn(3, 6, 5376, dtype=torch.bfloat16, device='cuda')
    scale, shift = tables[:,0], tables[:,1]
    indices = torch.arange(4814, device='cuda') % 3
    sources = [h,w,scale,shift,indices]
    saved = [x.clone() for x in sources]
    before = torch.are_deterministic_algorithms_enabled()
    fn = _compiled('pre', _pre_ref, deterministic=True)
    ref = fn(h,w,1e-5,scale,shift,indices)
    caller, side = torch.cuda.current_stream(), torch.cuda.Stream()
    for i in range(4):
        side.wait_stream(caller)
        try:
            with torch.cuda.stream(side):
                torch.cuda._sleep(1000000)
                result = fn(h,w,1e-5,scale,shift,indices)
                if i == 2:
                    raise RuntimeError('injected diagnostic exception')
        except RuntimeError as exc:
            assert str(exc) == 'injected diagnostic exception'
        finally:
            caller.wait_stream(side)
        assert torch.equal(result.view(torch.uint8),ref.view(torch.uint8))
        torch.cuda.synchronize()
        del result
        churn = torch.empty_like(h).fill_(i+17)
        del churn
    assert all(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)) for x,y in zip(sources,saved))
    assert torch.are_deterministic_algorithms_enabled() == before
    torch.save({'input':h.cpu(),'output':ref.cpu()},sys.argv[1])
'''
    paths = [tmp_path / f'process-{i}.pt' for i in range(2)]
    for path in paths:
        subprocess.run([sys.executable, '-c', code, str(path)], check=True, timeout=180)
    first, second = [torch.load(p, weights_only=True) for p in paths]
    for key in ('input','output'):
        assert torch.equal(first[key].view(torch.uint8), second[key].view(torch.uint8))
