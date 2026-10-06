"""CPU-only independent VDA config and shell-profile integration contracts."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from src.config.inference import InferenceConfig, validate_kernels, validate_parallel
from src.config.loader import ConfigError, load_config
from src.inference.utils.assemble import apply_vda_candidates

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/benchmark_rtx5090_nsight.sh"


def candidate(*overrides):
    return load_config(InferenceConfig, [
        "--config", str(ROOT / "configs/inference/8nfe_rtx5090_8gpu.yaml"),
        *overrides,
    ], extra_validators=(validate_kernels, validate_parallel))


class VDAConfigTests(unittest.TestCase):
    def test_defaults_and_independent_combinations(self):
        defaults = InferenceConfig()
        self.assertFalse(defaults.kernels.vda_fused)
        self.assertFalse(defaults.kernels.vda_overlap)
        for fused in (False, True):
            for overlap in (False, True):
                cfg = candidate(f"kernels.vda_fused={fused}", f"kernels.vda_overlap={overlap}")
                self.assertEqual(cfg.kernels.vda_fused, fused)
                self.assertEqual(cfg.kernels.vda_overlap, overlap)
                self.assertFalse(cfg.kernels.inference_kernels)
                self.assertEqual(OmegaConf.to_container(cfg)["kernels"]["vda_fused"], fused)

    def test_candidate_constraints(self):
        invalid = (
            "memory.policy=legacy", "memory.policy=cpu_block_offload",
            "precision.dtype=float32", "precision.fp8.enabled=true",
            "kernels.inference_kernels=true", "parallel.softmax_ranks=6",
            "memory.shard_world_size=4", "behavior.teacher_mode=true",
            "ablation.enabled=true", "memory.local_files_only=false",
            "checkpoint=null", "kernels.softmax_backend=ref", "render.device=cpu",
        )
        for flag in ("vda_fused", "vda_overlap"):
            for invalid_override in invalid:
                with self.subTest(flag=flag, override=invalid_override):
                    with self.assertRaises(ConfigError):
                        candidate(f"kernels.{flag}=true", invalid_override)

    def test_flags_are_boolean(self):
        for flag in ("vda_fused", "vda_overlap"):
            cfg = InferenceConfig()
            setattr(cfg.kernels, flag, "false")
            with self.assertRaisesRegex(ValueError, "must be a bool"):
                validate_kernels(cfg)
            with self.assertRaises(ConfigError):
                candidate(f"kernels.{flag}=not-a-bool")

    def test_propagation_and_reset_on_all_blocks(self):
        for fused in (False, True):
            for overlap in (False, True):
                cfg = candidate(f"kernels.vda_fused={fused}", f"kernels.vda_overlap={overlap}")
                blocks = [SimpleNamespace(linear_attention=SimpleNamespace()) for _ in range(3)]
                with patch("src.inference.utils.assemble.iter_hybrids", return_value=iter(blocks)):
                    apply_vda_candidates(object(), cfg)
                for attn in blocks:
                    self.assertEqual(attn.inference_vda_overlap, overlap)
                    self.assertEqual(attn.linear_attention.inference_vda_kernels, fused)
                cfg.kernels.vda_fused = cfg.kernels.vda_overlap = False
                with patch("src.inference.utils.assemble.iter_hybrids", return_value=iter(blocks)):
                    apply_vda_candidates(object(), cfg)
                self.assertTrue(all(not a.inference_vda_overlap and not a.linear_attention.inference_vda_kernels for a in blocks))

    def test_owner_and_meta_assembly_both_wire_flags(self):
        tree = ast.parse((ROOT / "src/inference/utils/assemble.py").read_text())
        for function, target in (("build_inference_model", "transformer"), ("build_weight_sharded_model", "meta")):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "apply_vda_candidates"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(ast.unparse(calls[0]), f"apply_vda_candidates({target}, cfg)")


class VDAWrapperTests(unittest.TestCase):
    def run_selection(self, *args):
        # Execute only argument parsing. No GPU query, files, monitor or launcher.
        prefix = SCRIPT.read_text().split("apps=$(", 1)[0]
        prefix = prefix.replace('cd "$(dirname "$0")/.."', ':')
        return subprocess.run(["bash", "-c", prefix + '\nprintf "%s %s %s %s\\n" "$profile" "$vda_fused" "$vda_overlap" "$prefetch"', "wrapper", *args], text=True, capture_output=True)

    def test_default_and_all_profiles(self):
        result = self.run_selection("unit", "plain")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "baseline false false false")
        for profile, flags in (("baseline", "false false"), ("vda_fused", "true false"), ("vda_overlap", "false true"), ("vda_both", "true true")):
            for mode in ("plain", "nsys"):
                result = self.run_selection("unit", mode, "allgather", "next_block", profile)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), f"{profile} {flags} true")

    def test_invalid_profile_rejected_before_gpu_query(self):
        for args in (("unit", "plain", "broadcast", "disabled", "bad"), ("unit", "plain", "broadcast", "disabled", "baseline", "extra")):
            result = self.run_selection(*args)
            self.assertEqual(result.returncode, 2)
            self.assertIn("baseline|vda_fused|vda_overlap|vda_both", result.stderr)

    def test_launch_and_metadata_contract(self):
        text = SCRIPT.read_text()
        self.assertIn('"kernels.vda_fused=$vda_fused"', text)
        self.assertIn('"kernels.vda_overlap=$vda_overlap"', text)
        self.assertIn('kernels.inference_kernels=false', text)
        self.assertIn('"$profile" > "$out/selectedprofile.txt"', text)
        self.assertIn('> "$out/source-sha256.json"', text)
        code = text.split('> "$out/source-sha256.json"\n', 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "src/inference/utils/vda_overlap.py"
            helper.parent.mkdir(parents=True)
            helper.write_text("# synthetic CPU source-hash fixture\n")
            import sys
            result = subprocess.run([sys.executable, "-c", code], cwd=directory, capture_output=True, text=True, check=True)
            hashes = json.loads(result.stdout)
            self.assertIn("src/inference/utils/vda_overlap.py", hashes)
            self.assertEqual(len(hashes["src/inference/utils/vda_overlap.py"]), 64)


if __name__ == "__main__":
    unittest.main()
