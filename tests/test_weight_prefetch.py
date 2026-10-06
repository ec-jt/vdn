"""CPU state-machine tests with explicit fake CUDA events, never GPU work."""
from contextlib import nullcontext
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.inference.utils.weight_sharding import ResidentWeightShards


class Event:
    def __init__(self):
        self.recorded = False
        self.finished = False

    def record(self, stream):
        self.recorded = True

    def synchronize(self):
        assert self.recorded
        self.finished = True


class Stream:
    def synchronize(self):
        pass

    def wait_event(self, event):
        assert event.finished


class PrefetchStateTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.Linear(3, 3) for _ in range(3)])
        self.model.transformer_blocks[0].alias = self.model.transformer_blocks[0].weight
        for block in self.model.transformer_blocks:
            block.register_buffer('mixed', torch.arange(5, dtype=torch.int64))
        self.model.eval().requires_grad_(False)
        self.runtime = SimpleNamespace(device=torch.device('cuda', 0), rank=0,
                                       world_size=8, backend='nccl', branch_parallel=False)
        self.c = ResidentWeightShards(self.model, self.runtime, prefetch=True)
        for groups in self.c.groups:
            for group in groups:
                group.shard = torch.ones(group.shard_numel, dtype=group.dtype)
        for entry in self.c.entries:
            entry.dematerialize()
        self.c.initialized = True
        self.c.weight_group = object()
        self.c.prefetch_stream = Stream()
        self.consumer = Stream()
        empty = torch.empty
        self.calls = []

        def gather(out, shard, *, group, async_op):
            self.assertIs(group, self.c.weight_group)
            self.assertTrue(async_op)
            self.calls.append(shard.dtype)
            out.copy_(shard.repeat(8))
            return SimpleNamespace(wait=lambda: None)

        patches = [patch('torch.empty', side_effect=lambda *a, **kw:
                         empty(*a, **dict(kw, device='cpu'))),
                   patch('torch.cuda.Event', Event),
                   patch('torch.cuda.current_stream', return_value=self.consumer),
                   patch('torch.cuda.stream', side_effect=lambda s: nullcontext()),
                   patch('torch.distributed.all_gather_into_tensor', side_effect=gather),
                   patch('torch.distributed.destroy_process_group')]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.grad = torch.no_grad()
        self.grad.__enter__()
        self.addCleanup(self.grad.__exit__, None, None, None)

    def test_repeated_forward_dtype_order_padding_alias_and_bound(self):
        self.assertEqual(self.c.groups[0][0].numel, 12)
        self.assertEqual(self.c.groups[0][0].shard_numel, 2)
        for _ in range(3):
            with self.c.forward():
                for i, block in enumerate(self.model.transformer_blocks):
                    with self.c.block(i):
                        if i == 0:
                            self.assertIs(block.weight, block.alias)
                        self.c.prefetch_next()
                        if i < 2:
                            self.assertEqual(self.c.pending.index, i + 1)
                            self.assertTrue(self.model.transformer_blocks[i+1].weight.is_meta)
                        else:
                            self.assertIsNone(self.c.pending)
                    self.assertTrue(block.weight.is_meta)
            self.assertIsNone(self.c.pending)
            self.assertIsNone(self.c._active_gather)
        self.assertEqual(self.c.gather_count, 9)
        self.assertEqual(self.c.max_live_gathered, 2)
        self.assertEqual(self.calls, [torch.float32, torch.int64] * 9)

    def test_order_reentry_duplicate_tail_and_release_guards(self):
        with self.assertRaises(RuntimeError), self.c.block(0):
            pass
        with self.c.forward():
            with self.assertRaises(RuntimeError), self.c.block(1):
                pass
            with self.assertRaises(RuntimeError), self.c.forward():
                pass
            with self.c.block(0):
                with self.assertRaises(RuntimeError), self.c.block(1):
                    pass
                with self.assertRaises(RuntimeError):
                    self.c.release()
                self.c.prefetch_next()
                with self.assertRaises(RuntimeError):
                    self.c.prefetch_next()
                with self.assertRaises(RuntimeError):
                    self.c._enqueue(2)
        self.assertIsNone(self.c.pending)

    def test_missing_tail_no_fallback_collective(self):
        with self.c.forward():
            with self.c.block(0):
                pass
            count = len(self.calls)
            with self.assertRaisesRegex(RuntimeError, 'did not enqueue'), self.c.block(1):
                pass
            self.assertEqual(len(self.calls), count)

    def test_exception_drains_pending_and_consumer_then_repeated_call(self):
        with self.assertRaisesRegex(ValueError, 'injected'), self.c.forward():
            with self.c.block(0):
                self.c.prefetch_next()
                active, pending = self.c._active_gather, self.c.pending
                raise ValueError('injected')
        self.assertTrue(active.consumed.finished)
        self.assertTrue(pending.ready.finished)
        self.assertIsNone(self.c.active)
        self.assertIsNone(self.c.pending)
        with self.c.forward(), self.c.block(0):
            self.c.prefetch_next()
        self.c.release()
        self.c.release()
        self.assertTrue(self.c.released)
        self.assertIsNone(self.c.weight_group)
        self.assertTrue(all(e.tensor().is_meta for e in self.c.entries))
        self.assertTrue(all(g.shard is None for gs in self.c.groups for g in gs))
        with self.assertRaises(RuntimeError), self.c.forward():
            pass

    def test_partial_enqueue_failure_retains_and_drains_owned_work(self):
        original = torch.distributed.all_gather_into_tensor
        count = 0

        def failure(*a, **kw):
            nonlocal count
            count += 1
            if count == 2:
                raise ValueError('enqueue failure')
            return original(*a, **kw)

        with patch('torch.distributed.all_gather_into_tensor', side_effect=failure):
            with self.assertRaisesRegex(ValueError, 'enqueue failure'), self.c.forward():
                with self.c.block(0):
                    pass
        self.assertIsNone(self.c.pending)
        self.c.release()

    def test_failed_drain_is_terminal_and_does_not_drop_buffers(self):
        with self.c.forward(), self.c.block(0):
            pass
        self.c._enqueue(1)
        pending = self.c.pending
        with patch.object(pending.ready, 'synchronize', side_effect=RuntimeError('CUDA failure')):
            with self.assertRaisesRegex(RuntimeError, 'CUDA failure'):
                self.c.flush()
        self.assertIs(self.c.pending, pending)
        self.assertTrue(self.c._failed)
        with self.assertRaises(RuntimeError), self.c.forward():
            pass
        self.c.release()

    def test_ready_record_failure_uses_stream_drain_not_unrecorded_event(self):
        with patch.object(Event, 'record', side_effect=RuntimeError('record failed')), \
                patch.object(self.c.prefetch_stream, 'synchronize') as drain:
            with self.assertRaisesRegex(RuntimeError, 'record failed'), self.c.forward():
                with self.c.block(0):
                    pass
            drain.assert_called_once()
        self.assertIsNone(self.c.pending)
        self.c.release()

    def test_consumer_event_failure_dematerializes_and_drains(self):
        p = patch('torch.cuda.Event', side_effect=RuntimeError('event failed'))
        try:
            with self.assertRaisesRegex(RuntimeError, 'event failed'), self.c.forward():
                with self.c.block(0):
                    self.c.prefetch_next()
                    p.start()
        finally:
            p.stop()
        self.assertIsNone(self.c.active)
        self.assertIsNone(self.c.pending)
        self.assertIsNone(self.c._active_gather)
        self.assertTrue(self.c._failed)
        self.assertTrue(self.model.transformer_blocks[0].weight.is_meta)
        self.c.release()

    def test_disabled_does_not_touch_cuda_or_create_group(self):
        c = ResidentWeightShards(self.model, self.runtime)
        with patch('torch.cuda.Event', side_effect=AssertionError('CUDA')), \
                patch('torch.distributed.new_group', side_effect=AssertionError('group')):
            with c.forward():
                c.prefetch_next()
            c.flush()
            c.release()
        self.assertFalse(c.summary()['prefetch'])


class PrefetchIntegrationTests(unittest.TestCase):
    def test_enqueue_after_last_attention_exchange_before_projection(self):
        tree = ast.parse(Path('src/inference/utils/ulysses.py').read_text())
        attention = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == '_ulysses_attention_forward')
        calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(attention)
                 if isinstance(n, ast.Call)]
        enqueue = [line for line, name in calls if name == 'shards.prefetch_next']
        self.assertEqual(len(enqueue), 1)
        last_exchange = max(line for line, name in calls if name in
                            ('runtime.heads_to_sequence', 'runtime.sequence_to_heads',
                             'runtime.video_frame_mean_async'))
        first_projection = min(line for line, name in calls if name == 'self.orig.to_out[0]')
        self.assertLess(last_exchange, enqueue[0])
        self.assertLess(enqueue[0], first_projection)
        forward = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == '_ulysses_transformer_forward')
        scope = next(n for n in ast.walk(forward) if isinstance(n, ast.With)
                     and 'shards.forward()' in ast.unparse(n.items[0].context_expr))
        self.assertTrue(any(isinstance(n, ast.For) for n in scope.body))
        self.assertNotIn('gather_sequence', ast.unparse(scope))


if __name__ == '__main__':
    unittest.main()
