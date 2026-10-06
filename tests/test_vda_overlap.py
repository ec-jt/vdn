"""Stream lifetime tests are opt-in, CPU ordering/skip checks always run."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.inference.utils.vda_overlap import vda_on_side_stream
from src.inference.utils.ulysses import _ulysses_attention_forward
from src.models.sequence_layout import SequenceLayout


@pytest.mark.parametrize('full_cover,enabled', [(True, True), (False, False)])
def test_no_side_stream_for_inactive_vda(full_cover, enabled):
    layout = SequenceLayout(3, 0, 3, 1)
    runtime = SimpleNamespace(rank=0, heads_per_rank=2, local_start=0, local_end=3,
        profile_start=lambda: None, profile_end=lambda *a: None,
        video_frame_mean_async=lambda x, l: (x, SimpleNamespace(wait=lambda: None)),
        sequence_to_heads=lambda x: x, heads_to_sequence=lambda x: x)
    branch = SimpleNamespace(head_dim=2, beta_proj=lambda x: x[:, :2],
                             output_gate=lambda x: x.reshape(3, 2, 2))
    attn = SimpleNamespace(teacher_mode=False, _ulysses_runtime=runtime, layout=layout,
        enable_softmax_gate=False, linear_attention=branch, head_dim=2, num_heads=2,
        _bounds=lambda l: [(0, 2)]*3 if full_cover else [(0, 0), (1, 1), (2, 2)],
        inference_vda_overlap=True, inference_mode=False, linear_attention_enabled=enabled,
        softmax_impl='ref', anchor_frames='both',
        orig=SimpleNamespace(processor=object(), to_out=[torch.nn.Identity(), torch.nn.Identity()]))
    attn._qkv = lambda x, r: (x.reshape(3,2,2),)*3 + ((x.reshape(3,2,2),)*3,)
    with patch('src.inference.utils.ulysses.vda_on_side_stream', side_effect=AssertionError('side launch')), \
         patch('src.inference.utils.ulysses.dispatch_attention_fn', side_effect=lambda q,k,v,**kw: v), \
         patch('src.inference.utils.ulysses.window_softmax_reference', side_effect=lambda q,k,v,*a,**kw: v):
        x = torch.ones(1,3,4)
        assert torch.equal(_ulysses_attention_forward(attn, x), x)


cuda = pytest.mark.skipif(os.environ.get('RUN_VDA_OVERLAP_CUDA') != '1', reason='opt-in CUDA')


@cuda
@torch.no_grad()
def test_nondefault_repeated_allocator_lifetimes():
    owner = SimpleNamespace()
    main = torch.cuda.Stream()
    module = torch.nn.Linear(128,128,bias=False,device='cuda',dtype=torch.bfloat16)
    main.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(main):
        for i in range(12):
            packed = torch.full((4096,256), .25, device='cuda', dtype=torch.bfloat16)
            x = packed[:, 128:]
            mean = torch.full((128,), .5, device='cuda', dtype=torch.float32)
            expected = module(x) + mean.to(x.dtype)
            with vda_on_side_stream(owner, lambda: module(x) + mean.to(x.dtype),
                                    inputs=(packed, mean), module=module) as out:
                assert torch.cuda.current_stream() == main
                churn = [torch.empty_like(packed).fill_(i) for _ in range(8)]
            del packed, x, mean, churn
            more = [torch.empty((4096,256), device='cuda',dtype=torch.bfloat16).fill_(-10) for _ in range(8)]
            assert torch.equal(out, expected)
            assert out.dtype == torch.bfloat16
            stream = owner._vda_stream
            del more, expected, out
            assert owner._vda_stream is stream
    main.synchronize()


@cuda
@pytest.mark.parametrize('side_raises', [False, True])
@torch.no_grad()
def test_exception_cleanup(side_raises):
    owner, module = SimpleNamespace(), torch.nn.Identity()
    main = torch.cuda.Stream()
    with torch.cuda.stream(main):
        x = torch.zeros(1, device='cuda')
        done = torch.cuda.Event()
        def branch():
            torch.cuda._sleep(10_000_000)
            x.fill_(7)
            done.record()
            if side_raises:
                raise RuntimeError('side failure')
            return x.clone()
        with pytest.raises(RuntimeError, match='failure'):
            with vda_on_side_stream(owner, branch, inputs=(x,), module=module):
                raise RuntimeError('main failure')
        main.synchronize()
        assert done.query()
        assert x.item() == 7
        assert torch.cuda.current_stream() == main


@cuda
def test_no_grad_required():
    with pytest.raises(RuntimeError, match='no-grad'):
        with vda_on_side_stream(SimpleNamespace(), lambda: None,
                                inputs=(torch.ones(1,device='cuda'),), module=torch.nn.Identity()):
            pass


@cuda
@torch.no_grad()
def test_active_forward_joins_before_collectives_and_prefetch():
    main = torch.cuda.Stream()
    with torch.cuda.stream(main):
        rows, heads, dim = 3, 2, 128
        layout = SequenceLayout(rows,0,rows,1)
        events = []
        done = torch.cuda.Event()
        def exchange(x):
            assert torch.cuda.current_stream() == main
            events.append('exchange')
            return x
        def prefetch():
            assert torch.cuda.current_stream() == main
            events.append('prefetch')
        runtime = SimpleNamespace(rank=0, heads_per_rank=heads, local_start=0, local_end=rows,
            profile_start=lambda: None, profile_end=lambda *a: None,
            video_frame_mean_async=lambda x,l: (x.float(), SimpleNamespace(wait=lambda: None)),
            sequence_to_heads=exchange, heads_to_sequence=exchange,
            weight_shards=SimpleNamespace(prefetch=True,prefetch_next=prefetch))
        branch = torch.nn.Module()
        branch.head_dim=dim
        branch.beta_proj=lambda x: x[:,:heads]
        branch.output_gate=lambda x: x.reshape(rows,heads,dim)
        attn = SimpleNamespace(teacher_mode=False,_ulysses_runtime=runtime,layout=layout,
            enable_softmax_gate=False,linear_attention=branch,head_dim=dim,num_heads=heads,
            _bounds=lambda l: [(0,0),(1,1),(2,2)], inference_vda_overlap=True,
            inference_mode=False,linear_attention_enabled=True,softmax_impl='flex',
            anchor_frames='both', to_out_linear=torch.nn.Identity(),
            orig=SimpleNamespace(processor=object(),to_out=[torch.nn.Identity(),torch.nn.Identity()]))
        attn._qkv=lambda x,r: (x.reshape(rows,heads,dim),)*3 + ((x.reshape(rows,heads,dim),)*3,)
        def linear(*args):
            assert torch.cuda.current_stream() == runtime._vda_stream
            events.append('linear')
            torch.cuda._sleep(10_000_000)
            result = torch.full((rows,heads,dim),2,device='cuda',dtype=torch.bfloat16)
            done.record()
            return result
        def softmax(a,q,k,v,*args):
            assert torch.cuda.current_stream() == main
            events.append('softmax')
            return v
        with patch('src.inference.utils.ulysses._linear_branch_shard',side_effect=linear), \
             patch('src.inference.utils.ulysses._window_softmax_branch',side_effect=softmax):
            result = _ulysses_attention_forward(attn,torch.ones(1,rows,heads*dim,device='cuda',dtype=torch.bfloat16))
        assert torch.equal(result,torch.full_like(result,3))
        assert done.query()
        assert events == ['exchange','linear','softmax','exchange','prefetch']
    main.synchronize()
