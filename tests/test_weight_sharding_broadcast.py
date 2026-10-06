"""Tiny three-rank CPU/Gloo regressions; no checkpoints or GPU initialization."""
import multiprocessing
import os
from pathlib import Path
import shlex
import tempfile
import time
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from src.inference.utils.weight_sharding import ResidentWeightShards


class Toy(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        with torch.device(device):
            self.register_buffer("root", torch.arange(7, dtype=torch.float64))
            self.transformer_blocks = torch.nn.ModuleList()
            for index in range(2):
                block = torch.nn.Module()
                block.weight = torch.nn.Parameter(
                    torch.arange(13, dtype=torch.float32) + index * 20,
                    requires_grad=False,
                )
                block.alias = block.weight
                block.register_buffer("bf16", torch.arange(7, dtype=torch.bfloat16) + index)
                block.register_buffer("count", torch.arange(5, dtype=torch.int64), persistent=False)
                self.transformer_blocks.append(block)
            shared = torch.nn.Parameter(torch.arange(5, dtype=torch.float32), requires_grad=False)
            for block in self.transformer_blocks:
                block.shared = shared
        self.eval()


def _worker(rank, rendezvous):
    check = unittest.TestCase()
    runtime = SimpleNamespace(device=torch.device("cpu"), rank=rank, world_size=3,
                              profile_start=lambda: None, profile_end=lambda *args: None)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
            patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA forbidden")):
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                                timeout=timedelta(seconds=20))
        try:
            def make():
                return ResidentWeightShards(Toy("cpu" if rank == 0 else "meta"), runtime,
                                            scatter_chunk_bytes=8, distribution="broadcast")

            def released(control):
                check.assertTrue(control.released)
                check.assertFalse(control.initialized)
                check.assertIsNone(control.active)
                check.assertTrue(all(e.tensor().is_meta for e in control.entries))
                check.assertTrue(all(g.shard is None for gs in control.groups for g in gs))

            control = make()
            reference = ResidentWeightShards(Toy("cpu"), runtime, scatter_chunk_bytes=8)
            expected = {e.names[0]: e.tensor().clone() for e in reference.entries}
            with patch.object(dist, "scatter", side_effect=AssertionError("scatter forbidden")):
                control.distribute()
            check.assertEqual(control.summary()["distribution"], "broadcast")
            for entry in control.root:
                check.assertTrue(torch.equal(entry.tensor(), expected[entry.names[0]]))
            check.assertIs(control.model.transformer_blocks[0].shared,
                           control.model.transformer_blocks[1].shared)
            for groups in control.groups:
                for group in groups:
                    flat = torch.cat([expected[e.names[0]].reshape(-1) for e in group.entries])
                    padded = torch.zeros(group.shard_numel * 3, dtype=group.dtype)
                    padded[:flat.numel()] = flat
                    check.assertTrue(torch.equal(group.shard, padded.chunk(3)[rank]))
            for _ in range(3):
                for index in range(2):
                    with torch.no_grad(), control.block(index):
                        for entry in control.blocks[index]:
                            check.assertEqual(entry.tensor().dtype, expected[entry.names[0]].dtype)
                            check.assertTrue(torch.equal(entry.tensor(), expected[entry.names[0]]))
                        block = control.model.transformer_blocks[index]
                        check.assertIs(block.weight, block.alias)
                    check.assertTrue(all(e.tensor().is_meta for e in control.blocks[index]))
            check.assertEqual(control.gather_count, 6)
            with torch.no_grad(), check.assertRaisesRegex(RuntimeError, "body failure"):
                with control.block(0):
                    raise RuntimeError("body failure")
            check.assertIsNone(control.active)
            check.assertTrue(all(e.tensor().is_meta for e in control.blocks[0]))
            original_gather = dist.all_gather_into_tensor
            calls = 0

            def fail_gather(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("gather failure")
                return original_gather(*args, **kwargs)

            with patch.object(dist, "all_gather_into_tensor", side_effect=fail_gather):
                with torch.no_grad(), check.assertRaisesRegex(RuntimeError, "gather failure"):
                    with control.block(0):
                        check.fail("must not yield")
            check.assertIsNone(control.active)
            check.assertTrue(all(e.tensor().is_meta for e in control.blocks[0]))
            with torch.no_grad(), control.block(0):
                check.assertTrue(torch.equal(control.model.transformer_blocks[0].weight,
                                             expected["transformer_blocks.0.weight"]))
            control.release()
            control.release()
            released(control)
            with torch.no_grad(), check.assertRaises(RuntimeError), control.block(0):
                pass

            # Every rank injects at the same point, avoiding unmatched collectives.
            # Test both a partially broadcast root and a partially allocated shard.
            for fail_at in (2, 12):
                control = make()
                original_broadcast = dist.broadcast
                calls = 0

                def fail_broadcast(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == fail_at:
                        raise RuntimeError("startup failure")
                    return original_broadcast(*args, **kwargs)

                with patch.object(dist, "broadcast", side_effect=fail_broadcast):
                    with check.assertRaisesRegex(RuntimeError, "startup failure"):
                        control.distribute()
                released(control)
                control.release()
            check.assertFalse(torch.cuda.is_initialized())
        finally:
            dist.destroy_process_group()


class BroadcastTests(unittest.TestCase):
    def test_three_rank_exact_lifecycle_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=_worker,
                         args=(rank, "file://" + os.path.join(directory, "pg"))) for rank in range(3)]
            try:
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
                    process.join()

    def config(self, *overrides):
        from src.config.inference import InferenceConfig, validate_memory
        from src.config.loader import load_config
        return load_config(InferenceConfig,
                           argv=["--config", "configs/inference/8nfe_rtx5090_8gpu.yaml", *overrides],
                           extra_validators=[validate_memory])

    def test_config_scatter_default_and_broadcast(self):
        self.assertEqual(self.config().memory.shard_distribution, "scatter")
        self.assertEqual(self.config("memory.shard_distribution=broadcast").memory.shard_distribution,
                         "broadcast")

    def test_invalid_distribution_rejected(self):
        from src.config.loader import ConfigError
        for value in ("invalid", "Broadcast", "null", "123"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.config("memory.shard_distribution=" + value)
        runtime = SimpleNamespace(device=torch.device("cpu"), rank=0, world_size=3)
        with self.assertRaisesRegex(ValueError, "distribution"):
            ResidentWeightShards(Toy("cpu"), runtime, distribution="invalid")
        self.assertEqual(ResidentWeightShards(Toy("cpu"), runtime).distribution, "scatter")

    def test_launcher_injection_and_explicit_override_order(self):
        # Inspect shell argv, never execute a production launcher or GPU preflight.
        script = Path("scripts/inference/8nfe_rtx5090_8gpu_transport.sh").read_text()
        command = next(line for line in script.splitlines() if line.startswith("exec "))
        tokens = shlex.split(command)
        self.assertEqual(tokens[3:], ["memory.shard_distribution=broadcast",
                                     "parallel.standard_exchange=broadcast", "$@"])
        injected = tokens[3:-1]
        self.assertEqual(self.config(*injected).memory.shard_distribution, "broadcast")
        self.assertEqual(self.config(*injected, "memory.shard_distribution=scatter")
                         .memory.shard_distribution, "scatter")
        parent = Path(tokens[2]).read_text()
        self.assertIn('configs/inference/8nfe_rtx5090_8gpu.yaml "$@"', parent)


if __name__ == "__main__":
    unittest.main()
