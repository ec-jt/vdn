"""Independent coordinate, layout, lifetime and validation gates for allgather."""
from datetime import timedelta
import multiprocessing
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from src.config.inference import InferenceConfig, validate_parallel
from src.inference.utils.ulysses_runtime import UlyssesRuntime


def runtime(rank=0, **kwargs):
    return UlyssesRuntime(rank, 3, rank, torch.device('cpu'), 'gloo',
                          standard_exchange='allgather', **kwargs)


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=3,
                            timeout=timedelta(seconds=25))
    try:
        for rows in (4, 8, 9):
            rt = runtime(rank)
            rt.configure(rows, 6)
            prior = []
            for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64):
                full = torch.arange(rows * 6 * 4).reshape(rows, 6, 4).to(dtype)
                reverse = (-torch.arange(rows * 6 * 4).reshape(rows, 6, 4) - 1).to(dtype)
                for strided in (False, True):
                    for method, source, expected in (
                        ('sequence_to_heads', full[rt.local_start:rt.local_end], full[:, rank * 2:(rank + 1) * 2]),
                        ('heads_to_sequence', reverse[:, rank * 2:(rank + 1) * 2], reverse[rt.local_start:rt.local_end]),
                    ):
                        if strided:
                            storage = torch.empty((*source.shape[:-1], source.shape[-1] * 2), dtype=dtype)
                            storage[..., ::2].copy_(source)
                            source = storage[..., ::2]
                        before = source.clone()
                        with patch.object(dist, 'broadcast', side_effect=AssertionError('unexpected broadcast')), \
                                patch.object(dist, 'all_to_all_single', side_effect=AssertionError('unexpected alltoall')):
                            actual = getattr(rt, method)(source)
                        assert torch.equal(actual, expected)
                        assert torch.equal(source, before)
                        assert actual.dtype == dtype and actual.is_contiguous()
                        prior.append((actual, expected.clone()))
                    for actual, expected in prior:
                        assert torch.equal(actual, expected)
    finally:
        dist.destroy_process_group()


class AllGatherTests(unittest.TestCase):
    def test_independent_routes_dtypes_strides_padding_repeated_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context('spawn')
            processes = [context.Process(target=worker, args=(rank, 'file://' + directory + '/pg'))
                         for rank in range(3)]
            try:
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
                    for process in processes:
                        process.start()
                deadline = time.monotonic() + 60
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                self.assertEqual([p.exitcode for p in processes], [0, 0, 0])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join()

    def test_config_opt_in_and_branch_rejection(self):
        cfg = OmegaConf.structured(InferenceConfig)
        self.assertEqual(cfg.parallel.standard_exchange, 'alltoall')
        cfg.parallel.standard_exchange = 'allgather'
        with self.assertRaises(ValueError):
            validate_parallel(cfg)
        cfg.parallel.softmax_ranks = 0
        validate_parallel(cfg)
        with self.assertRaises(ValueError):
            runtime().enable_branch_parallel(1)
        with self.assertRaises(ValueError):
            runtime(softmax_ranks=1).configure(8, 6)

    def test_invalid_geometry_and_shape_before_collective(self):
        rt = runtime()
        with self.assertRaises(ValueError):
            rt.configure(2, 6)
        with self.assertRaises(ValueError):
            rt.configure(8, 7)
        rt.configure(8, 6)
        with self.assertRaises(RuntimeError):
            rt.configure(9, 6)
        with self.assertRaises(ValueError):
            rt.sequence_to_heads(torch.empty(2, 6, 4))
        with self.assertRaises(ValueError):
            rt.sequence_to_heads(torch.empty(3, 5, 4))
        with self.assertRaises(ValueError):
            rt.heads_to_sequence(torch.empty(8, 3, 4))

    def test_launcher_default_and_allowed_routes(self):
        from pathlib import Path
        source = Path('scripts/benchmark_rtx5090_nsight.sh').read_text()
        self.assertIn('parallel.standard_exchange=${3:-broadcast}', source)
        self.assertIn('^(broadcast|allgather)$', source)
        self.assertIn('memory.shard_distribution=broadcast', source)


if __name__ == '__main__':
    unittest.main()
