"""CPU-only prefetch configuration and launcher contracts; no model assembly."""
import ast
import os
from pathlib import Path
import subprocess
import unittest

from omegaconf import OmegaConf

from src.config.inference import (
    InferenceConfig, MemoryConfig, validate_ablation, validate_kernels,
    validate_memory, validate_parallel, validate_single_process,
)
from src.config.loader import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/benchmark_rtx5090_nsight.sh"
VALIDATORS = (validate_ablation, validate_kernels, validate_parallel)


def sharded_config(*overrides):
    return load_config(InferenceConfig, [
        "--config", str(ROOT / "configs/inference/8nfe_rtx5090_8gpu.yaml"),
        *overrides,
    ], extra_validators=VALIDATORS)


class WeightPrefetchConfigTests(unittest.TestCase):
    def test_defaults_remain_disabled(self):
        self.assertIs(MemoryConfig().weight_prefetch, False)
        cfg = InferenceConfig()
        validate_memory(cfg)
        self.assertEqual(cfg.memory.policy, "legacy")
        self.assertIs(cfg.memory.weight_prefetch, False)
        self.assertIs(sharded_config().memory.weight_prefetch, False)

    def test_valid_opt_in_for_standard_ulysses_exchanges(self):
        for exchange in ("alltoall", "broadcast", "allgather"):
            for device in ("cuda", "cuda:0", "cuda:7"):
                with self.subTest(exchange=exchange, device=device):
                    cfg = sharded_config(
                        "memory.weight_prefetch=true",
                        f"parallel.standard_exchange={exchange}",
                        f"render.device={device}",
                    )
                    self.assertIs(cfg.memory.weight_prefetch, True)
                    self.assertEqual(cfg.parallel.softmax_ranks, 0)
                    self.assertEqual(cfg.precision.dtype, "bfloat16")
                    self.assertFalse(cfg.precision.fp8.enabled)

    def test_opt_in_rejects_other_policies(self):
        for policy in ("legacy", "cpu_block_offload"):
            with self.subTest(policy=policy), self.assertRaisesRegex(
                    ConfigError, "weight_prefetch requires resident_weight_shards"):
                sharded_config("memory.weight_prefetch=true", f"memory.policy={policy}")

    def test_opt_in_rejects_branch_parallel(self):
        for ranks in (-1, 1, 6):
            with self.subTest(ranks=ranks), self.assertRaisesRegex(
                    ConfigError, "weight_prefetch requires standard Ulysses"):
                sharded_config("memory.weight_prefetch=true", f"parallel.softmax_ranks={ranks}")

    def test_opt_in_rejects_non_cuda(self):
        for device in ("cpu", "meta", "mps", "xpu:0", "cudafoo"):
            with self.subTest(device=device), self.assertRaisesRegex(
                    ConfigError, "weight_prefetch requires a CUDA"):
                sharded_config("memory.weight_prefetch=true", f"render.device={device}")

    def test_direct_config_rejects_non_boolean_even_when_falsey(self):
        # Structured OmegaConf coerces some inputs. Direct dataclasses must not
        # treat strings, integer flags or falsey containers as booleans.
        for policy in ("legacy", "resident_weight_shards"):
            for value in (None, 0, 1, 0.0, "false", "true", "next_block", [], {}):
                cfg = OmegaConf.to_object(sharded_config())
                cfg.memory.policy = policy
                cfg.memory.weight_prefetch = value
                with self.subTest(policy=policy, value=value), self.assertRaisesRegex(
                        ValueError, "memory.weight_prefetch must be a bool"):
                    validate_memory(cfg)

    def test_cli_rejects_invalid_boolean_mode(self):
        with self.assertRaisesRegex(ConfigError, "weight_prefetch"):
            sharded_config("memory.weight_prefetch=next_block")

    def test_opt_in_preserves_existing_resident_restrictions(self):
        for override in (
            "precision.dtype=float32", "precision.fp8.enabled=true",
            "kernels.inference_kernels=true", "kernels.softmax_backend=auto",
            "behavior.teacher_mode=true", "memory.local_files_only=false",
            "checkpoint=null", "memory.shard_world_size=4",
            "memory.scatter_chunk_mib=0", "memory.scatter_chunk_mib=65",
            "memory.shard_distribution=invalid", "ablation.enabled=true",
            "parallel.timeout_seconds=0",
        ):
            with self.subTest(override=override), self.assertRaises(ConfigError):
                sharded_config("memory.weight_prefetch=true", override)
        with self.assertRaisesRegex(ValueError, "infer_ulysses"):
            validate_single_process(sharded_config("memory.weight_prefetch=true"))

    def test_disabled_preserves_existing_cpu_configuration_validation(self):
        cfg = sharded_config("memory.weight_prefetch=false", "render.device=cpu")
        self.assertIs(cfg.memory.weight_prefetch, False)

    def test_all_assembly_constructors_pass_through_prefetch(self):
        # Parse only: importing assembly would pull in model/runtime modules.
        tree = ast.parse((ROOT / "src/inference/utils/assemble.py").read_text())
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "ResidentWeightShards"]
        self.assertEqual(len(calls), 2, "header/meta and owner construction")
        for call in calls:
            keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
            self.assertEqual(keywords["prefetch"], "cfg.memory.weight_prefetch")
            self.assertEqual(keywords["distribution"], "cfg.memory.shard_distribution")


class WeightPrefetchLauncherTests(unittest.TestCase):
    def run_argument_prefix(self, *args):
        source = LAUNCHER.read_text()
        # Execute only the exact argument-validation/mapping prefix. No GPU
        # command, monitoring, artifact writes or inference launch is reachable.
        prefix, separator, _ = source.partition("apps=$(timeout")
        self.assertTrue(separator, "GPU-check boundary must remain identifiable")
        result = subprocess.run(
            ["bash", "-c", prefix + '\nprintf "%s|%s|%s\\n" "${3:-broadcast}" '
             '"${4:-disabled}" "$prefetch"', str(LAUNCHER), *args],
            cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True, text=True, timeout=10,
        )
        return result

    def test_launcher_defaults_and_explicit_modes(self):
        cases = (
            (("prefetch-test", "plain"), "broadcast|disabled|false"),
            (("prefetch-test", "nsys"), "broadcast|disabled|false"),
            (("prefetch-test", "plain", "allgather"), "allgather|disabled|false"),
            (("prefetch-test", "plain", "broadcast", "disabled"), "broadcast|disabled|false"),
            (("prefetch-test", "plain", "broadcast", "next_block"), "broadcast|next_block|true"),
            (("prefetch-test", "nsys", "allgather", "next_block"), "allgather|next_block|true"),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                result = self.run_argument_prefix(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_launcher_rejects_invalid_fourth_and_extra_arguments(self):
        for args in (
            ("prefetch-test", "plain", "broadcast", "true"),
            ("prefetch-test", "plain", "broadcast", "false"),
            ("prefetch-test", "plain", "broadcast", "invalid"),
            ("prefetch-test", "plain", "broadcast", "next_block", "extra"),
            ("prefetch-test", "plain", "next_block"),
            ("prefetch-test", "invalid"),
            (),
        ):
            with self.subTest(args=args):
                result = self.run_argument_prefix(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("[disabled|next_block]", result.stderr)

    def test_launcher_static_pass_through_and_fixed_benchmark(self):
        source = LAUNCHER.read_text()
        for expected in (
            '"memory.weight_prefetch=$prefetch"',
            'printf \'%s\\n\' "${4:-disabled}" > "$out/prefetch.txt"',
            'printf \'%s\\n\' "${3:-broadcast}" > "$out/route.txt"',
            '"parallel.standard_exchange=${3:-broadcast}"',
            "memory.shard_distribution=broadcast", "parallel.profile=false",
            "--config configs/inference/8nfe_rtx5090_8gpu.yaml",
            "--nproc_per_node=8", "render.num_frames=124", "render.num_steps=8",
            "render.seed=42", "render.prompt_file=prompts/example_0.pt",
            "precision.dtype=bfloat16", "precision.fp8.enabled=false",
            "kernels.softmax_backend=flex", "render.save_latents=true",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
