"""Independent exact three-rank CPU/Gloo standard-exchange regressions."""
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

from src.config.inference import InferenceConfig, ParallelConfig, validate_parallel
from src.inference.utils.ulysses_runtime import UlyssesRuntime


def _runtime(rank=0, **kwargs):
    return UlyssesRuntime(rank=rank, world_size=3, local_rank=rank,
                          device=torch.device("cpu"), backend="gloo", **kwargs)


def _strided(tensor):
    storage = torch.full((*tensor.shape[:-1], tensor.shape[-1] * 2), -999.0)
    view = storage[..., ::2]
    view.copy_(tensor)
    assert not view.is_contiguous()
    return view


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
            patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA forbidden")):
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                                timeout=timedelta(seconds=20))
        try:
            broadcast = _runtime(rank, standard_exchange="broadcast")
            alltoall = _runtime(rank)
            for runtime in (broadcast, alltoall):
                runtime.configure(sequence_length=8, num_heads=6)
                assert runtime.splits == (3, 3, 2)
                assert (runtime.local_start, runtime.local_end) == ((0, 3), (3, 6), (6, 8))[rank]
            start, end = ((0, 3), (3, 6), (6, 8))[rank]
            previous = []
            for iteration in range(3):
                # Explicit global oracle: every row/head/channel has a distinct value.
                full = torch.arange(8 * 6 * 4, dtype=torch.float32).reshape(8, 6, 4)
                full = full + iteration * 10000
                sequence_input = full[start:end].clone()
                expected_heads = full[:, rank * 2:(rank + 1) * 2]
                # Reverse input is independent, not the result of the forward exchange.
                reverse_full = -full * 3 - 70000
                head_input = reverse_full[:, rank * 2:(rank + 1) * 2].contiguous()
                expected_sequence = reverse_full[start:end]
                if iteration % 2:
                    sequence_input, head_input = _strided(sequence_input), _strided(head_input)
                for method, source, expected in (
                    ("sequence_to_heads", sequence_input, expected_heads),
                    ("heads_to_sequence", head_input, expected_sequence),
                ):
                    before = source.clone()
                    with patch.object(dist, "all_to_all_single",
                                      side_effect=AssertionError("broadcast used alltoall")), \
                            patch.object(dist, "broadcast", wraps=dist.broadcast) as calls:
                        actual = getattr(broadcast, method)(source)
                        assert calls.call_count == 3
                    reference = getattr(alltoall, method)(source)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    torch.testing.assert_close(source, before, rtol=0, atol=0)
                    previous.append((actual, expected.clone()))
                # Later calls must neither reuse stale data nor mutate old outputs.
                for actual, expected in previous:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert not torch.cuda.is_initialized()
        finally:
            dist.destroy_process_group()


class UlyssesBroadcastTests(unittest.TestCase):
    def test_three_rank_independent_directions_uneven_repeated_and_strided(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=_worker,
                         args=(rank, "file://" + os.path.join(directory, "pg")))
                         for rank in range(3)]
            try:
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                    for process in processes:
                        process.start()
                deadline = time.monotonic() + 45
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                self.assertEqual([p.exitcode for p in processes], [0, 0, 0])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join()

    def test_defaults_and_valid_config_modes(self):
        self.assertEqual(_runtime().standard_exchange, "alltoall")
        self.assertEqual(ParallelConfig().standard_exchange, "alltoall")
        self.assertEqual(ParallelConfig().softmax_ranks, 6)
        cfg = OmegaConf.structured(InferenceConfig)
        validate_parallel(cfg)
        for mode in ("alltoall", "broadcast"):
            cfg.parallel.softmax_ranks = 0
            cfg.parallel.standard_exchange = mode
            validate_parallel(cfg)
            runtime = _runtime(standard_exchange=mode)
            runtime.configure(8, 6)
            self.assertEqual(runtime.standard_exchange, mode)

    def test_invalid_strings_rejected_in_runtime_and_config(self):
        for value in ("invalid", "Broadcast", "", "all_to_all"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "standard_exchange"):
                    _runtime(standard_exchange=value).configure(8, 6)
                cfg = OmegaConf.structured(InferenceConfig)
                cfg.parallel.standard_exchange = value
                with self.assertRaisesRegex(ValueError, "standard_exchange"):
                    validate_parallel(cfg)

    def test_branch_mix_rejected_in_runtime_and_config(self):
        runtime = _runtime(standard_exchange="broadcast")
        with self.assertRaisesRegex(ValueError, "branch"):
            runtime.enable_branch_parallel(1)
        with self.assertRaisesRegex(ValueError, "branch"):
            _runtime(standard_exchange="broadcast", softmax_ranks=1).configure(8, 6)
        cfg = OmegaConf.structured(InferenceConfig)
        cfg.parallel.standard_exchange = "broadcast"
        for softmax_ranks in (1, 6):
            cfg.parallel.softmax_ranks = softmax_ranks
            with self.subTest(softmax_ranks=softmax_ranks), \
                    self.assertRaisesRegex(ValueError, "softmax_ranks=0"):
                validate_parallel(cfg)
        default = _runtime()
        default.enable_branch_parallel(1)
        default.configure(8, 6)
        self.assertTrue(default.branch_parallel)


if __name__ == "__main__":
    unittest.main()
