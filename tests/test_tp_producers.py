"""Explicit GPU producer, predicate and bounded cache correctness tests."""
import os
import pytest
import torch
from torch import nn

from src.inference.utils.paper_adaln import AdaLNTableCache
from test_paper_stack_adaln import toy, prepare, schedule


def test_cache_views_budget_schedule_and_mutation():
    model = toy()
    table = prepare(model)
    cache = AdaLNTableCache(table.stats.table_bytes)
    assert cache.lookup('a') is None
    view = cache.store('a', table)
    step = view.step(0, schedule().timestep(0))
    ptr = step.embedding.data_ptr()
    with pytest.raises(RuntimeError, match='active'):
        cache.lookup('b')
    view.close()
    with pytest.raises(RuntimeError, match='closed'):
        _ = step.embedding
    view = cache.lookup('a')
    assert view.step(0, schedule().timestep(0)).embedding.data_ptr() == ptr
    view.close()
    assert cache.lookup('b') is None and cache.bytes == 0 and cache.evictions == 1
    table = prepare(model)
    view = cache.store('b', table)
    view.close()
    table._embeddings[0].add_(1)
    with pytest.raises(RuntimeError, match='mutated'):
        cache.lookup('b')
    cache.clear()
    with pytest.raises(MemoryError):
        AdaLNTableCache(1).store('a', prepare(model))


def test_persistent_cache_inputs_and_source_invalidation(monkeypatch):
    from test_persistent import owner
    cfg, paper = owner()
    model = toy()
    model.transformer_blocks = nn.ModuleList([nn.Module() for _ in model.projections])
    for b, p in zip(model.transformer_blocks, model.projections):
        b.adaln_proj = p
    paper.retain_adaln_sources(model)
    paper.schedule = schedule()
    paper.request_key = ('geometry', 0)
    key = paper.cached_adaln_key(cfg)
    cfg.render.seed = 19
    cfg.render.prompt_file = 'different.pt'
    assert paper.cached_adaln_key(cfg) == key
    paper.schedule = schedule(video_shift=8)
    assert paper.cached_adaln_key(cfg) != key
    paper.schedule = schedule()
    paper.request_key = ('geometry', 1)
    assert paper.cached_adaln_key(cfg) != key
    with torch.no_grad():
        paper.sources.transformer_blocks[0].adaln_proj.linear.weight.add_(1)
    with pytest.raises(RuntimeError, match='signature'):
        paper.cached_adaln_key(cfg)
    assert paper.poisoned


def test_request_cache_miss_hit_hit_and_eviction():
    from test_persistent import owner
    cfg, paper = owner()
    model = toy()
    model.transformer_blocks = nn.ModuleList([nn.Module() for _ in model.projections])
    for b, p in zip(model.transformer_blocks, model.projections):
        b.adaln_proj = p
    paper.retain_adaln_sources(model)
    previous = None
    pointer = None
    for request in range(4):
        cfg.render.seed += 1
        cfg.render.prompt_file = f'prompt{request}.pt'
        paper.request = paper.new_request(cfg)
        paper.schedule = schedule(conditioned=request == 3)
        paper.request_key = ('geometry', int(request == 3))
        paper.prepare_cached_adaln(cfg)
        hit = request in (1,2)
        assert paper.adaln_stats['cache_hit'] == hit
        assert bool(paper.calls['adaln_precompute_projection']) != hit
        assert bool(paper.calls['adaln_table_transport']) != hit
        assert paper.table_cache.references == 1
        view = paper.adaln.step(0, paper.schedule.timestep(0))
        if hit:
            assert view.embedding.data_ptr() == pointer
        pointer = view.embedding.data_ptr()
        if previous is not None:
            with pytest.raises(RuntimeError, match='closed'):
                previous.block(0)
        previous = view
        paper.close_tables()
        assert paper.table_cache.references == 0
    assert paper.table_cache.hits == 2 and paper.table_cache.misses == 2
    assert paper.table_cache.evictions == 1
    paper.close()
    assert paper.table_cache.bytes == 0


gpu = pytest.mark.skipif(os.getenv('VDN_TEST_MXFP8_GPU') != '1', reason='explicit GPU opt-in')


@gpu
@pytest.mark.parametrize('rows,k', [(1,128), (129,256), (131,1792), (129,5376)])
@torch.no_grad()
def test_producer_layout_rounding_and_finite(rows, k):
    from src.models.ops.mxfp8_kernels import launch_swiglu_quantize, launch_prenorm_quantize
    from src.models.ops.mxfp8_quant import quantize_activation
    from src.models.ops.fused_block import _compiled, _swiglu_ref
    torch.manual_seed(429)
    x = torch.randn(rows * 2, 2*k, device='cuda', dtype=torch.bfloat16)[::2]
    q,s,f = launch_swiglu_quantize(x)
    ref = _compiled('swiglu', _swiglu_ref, deterministic=True)(x)
    qr,sr = quantize_activation(ref)
    assert torch.equal(q,qr) and torch.equal(s,sr) and f.all()
    hidden = x[:, :k]
    w = torch.randn(k, device='cuda', dtype=torch.bfloat16)
    scale,shift = torch.randn(7, k*6, device='cuda', dtype=torch.bfloat16).chunk(6, -1)[:2]
    idx = torch.arange(rows, device='cuda') % 7
    y,q,s,f = launch_prenorm_quantize(hidden,w,1e-6,scale,shift,idx,write_bf16=True)
    oracle = (hidden.float()*torch.rsqrt(hidden.float().square().mean(-1,keepdim=True)+1e-6)*w.float())
    oracle = (oracle*(1+scale[idx].float())+shift[idx].float()).bfloat16()
    torch.testing.assert_close(y,oracle,atol=.03125,rtol=.008)
    qr,sr = quantize_activation(y)
    assert torch.equal(q,qr) and torch.equal(s,sr) and f.all()
    _,q2,s2,f2 = launch_prenorm_quantize(hidden,w,1e-6,scale,shift,idx,write_bf16=False)
    assert torch.equal(q,q2) and torch.equal(s,s2) and f2.all()
    x[0,0] = float('nan')
    assert not launch_swiglu_quantize(x)[2].all()
    assert not launch_prenorm_quantize(hidden,w,1e-6,scale,shift,idx,write_bf16=False)[3].all()


@gpu
def test_fast_guard_stream_join_and_encoded_failure():
    from src.inference.utils.paper_runtime import CoordinatedFiniteGuard
    from src.models.ops.mxfp8_kernels import finite_parts
    guard = CoordinatedFiniteGuard('cuda', fast=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.ones(100000, device='cuda')
        torch.cuda._sleep(1000000)
        x[-1] = float('inf')
        guard.predicate(finite_parts(x))
    with pytest.raises(FloatingPointError):
        guard.check('joined')
    for data,scale in [(127,0), (0,255)]:
        guard = CoordinatedFiniteGuard('cuda', fast=True)
        guard.encoded(torch.full((128,),data,device='cuda',dtype=torch.uint8),
                      torch.full((4,),scale,device='cuda',dtype=torch.uint8))
        with pytest.raises(FloatingPointError):
            guard.check('encoded')
