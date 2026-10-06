"""Config-only baseline contracts, with no model loading or downloads."""
from pathlib import Path
import unittest

from omegaconf import OmegaConf

from src.config.loader import ConfigError, load_config
from src.config.inference import (
    InferenceConfig, validate_ablation, validate_kernels, validate_single_process,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATORS = (validate_ablation, validate_kernels, validate_single_process)


def baseline(name, *overrides):
    return load_config(InferenceConfig, [
        "--config", str(ROOT / "configs/inference" / name), *overrides,
    ], extra_validators=VALIDATORS)


class RTX5090ConfigTests(unittest.TestCase):
    def test_50nfe_preserves_shipped_sampler_and_adapter_selection(self):
        cfg = baseline("50nfe_rtx5090.yaml")
        shipped = baseline("50nfe.yaml", "render.prompt_file=prompts/example_2.pt",
                           "render.out=results/unused.mp4")
        for field in ("num_steps", "warmup_steps", "video_shift", "audio_shift", "seed"):
            self.assertEqual(cfg.render[field], shipped.render[field], field)
        self.assertEqual(cfg.render.num_steps, 50)
        self.assertTrue(cfg.checkpoint.endswith("/stage-b-step-2000"))
        self.assertEqual(cfg.render.prompt_file, "prompts/example_2.pt")
        self.assertEqual(list(cfg.external_loras), [])
        self.assertEqual(OmegaConf.to_container(cfg.behavior),
                         OmegaConf.to_container(shipped.behavior))
        self.assertFalse(cfg.ablation.enabled)
        self.assertEqual(dict(cfg.ablation.overrides), {})

    def test_both_baselines_are_local_bf16_flex_and_have_separate_outputs(self):
        fast, stage_b = [baseline(f"{n}nfe_rtx5090.yaml") for n in (8, 50)]
        self.assertNotEqual(fast.checkpoint, stage_b.checkpoint)
        self.assertNotEqual(fast.render.out, stage_b.render.out)
        self.assertEqual(fast.render.num_steps, 8)
        for cfg in (fast, stage_b):
            self.assertEqual(cfg.memory.policy, "cpu_block_offload")
            self.assertTrue(cfg.memory.local_files_only)
            self.assertEqual(cfg.precision.dtype, "bfloat16")
            self.assertFalse(cfg.precision.fp8.enabled)
            self.assertFalse(cfg.kernels.inference_kernels)
            self.assertEqual(cfg.kernels.softmax_backend, "flex")
            self.assertEqual(cfg.render.num_frames, 124)
            self.assertEqual(cfg.render.warmup_steps, 0)
            self.assertEqual(cfg.render.device, "cuda:0")
            self.assertTrue(cfg.render.record)
            self.assertEqual(cfg.base_source, cfg.vae_source)
            self.assertTrue(cfg.base_source.endswith("/h3-base"))

    def test_unsupported_combinations_fail_validation(self):
        for override in ("precision.fp8.enabled=true", "precision.dtype=float32",
                         "kernels.inference_kernels=true", "behavior.teacher_mode=true",
                         "parallel.softmax_ranks=0", "parallel.profile=true"):
            with self.subTest(override=override), self.assertRaises(ConfigError):
                baseline("50nfe_rtx5090.yaml", override)

    def test_cli_can_select_fresh_output_and_saved_latents(self):
        cfg = baseline("50nfe_rtx5090.yaml", "render.out=results/new-stage-b.mp4",
                       "render.save_latents=true")
        self.assertEqual(cfg.render.out, "results/new-stage-b.mp4")
        self.assertTrue(cfg.render.save_latents)
        self.assertEqual(cfg.render.num_steps, 50)


if __name__ == "__main__":
    unittest.main()
