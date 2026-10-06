"""Small real Gloo collectives and lifecycle checks, no production payload loads."""
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import save_file

from src.config.inference import InferenceConfig, ExternalLora, validate_memory, validate_single_process
from src.inference.utils.weight_sharding import ResidentWeightShards, header_tensors


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.root = torch.nn.Linear(3, 3).double()
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(2)])
        self.transformer_blocks[0].alias = self.transformer_blocks[0].weight
        self.register_buffer("computed", torch.arange(3.), persistent=False)
        self.transformer_blocks[0].register_buffer("count", torch.tensor(7))


class WeightShardingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        dist.init_process_group("gloo", init_method="file://" + os.path.join(cls.tmp.name, "pg"), rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.tmp.cleanup()

    def controller(self, model):
        runtime = SimpleNamespace(device=torch.device("cpu"), rank=0, world_size=1,
                                  profile_start=lambda: None, profile_end=lambda *args: None)
        return ResidentWeightShards(model, runtime, scatter_chunk_bytes=8)

    def test_roundtrip_mixed_dtype_buffers_alias_and_repeat(self):
        model = Toy().eval().requires_grad_(False)
        expected = {k: t.clone() for k, t in model.state_dict().items()}
        control = self.controller(model)
        control.distribute()
        self.assertFalse(model.root.weight.is_meta)
        for _ in range(3):
            for i, block in enumerate(model.transformer_blocks):
                self.assertTrue(block.weight.is_meta)
                with torch.no_grad(), control.block(i):
                    for key, tensor in block.state_dict().items():
                        torch.testing.assert_close(tensor, expected[f"transformer_blocks.{i}.{key}"])
                    if i == 0:
                        self.assertIs(block.weight, block.alias)
                self.assertTrue(block.weight.is_meta)
        self.assertEqual(control.gather_count, 6)
        control.release(); control.release()
        self.assertTrue(all(t.is_meta for t in list(model.parameters()) + list(model.buffers())))
        self.assertTrue(all(g.shard is None for groups in control.groups for g in groups))
        with torch.no_grad(), self.assertRaises(RuntimeError), control.block(0):
            pass

    def test_exception_cleanup_and_no_reentry(self):
        model = Toy().eval()
        control = self.controller(model); control.distribute()
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "injected"):
            with control.block(0):
                with self.assertRaises(RuntimeError), control.block(1):
                    pass
                raise ValueError("injected")
        self.assertIsNone(control.active)
        self.assertTrue(model.transformer_blocks[0].weight.is_meta)
        control.release()

    def test_broadcast_distribution_exact_and_no_scatter(self):
        model = Toy().eval().requires_grad_(False)
        expected = {k: t.clone() for k, t in model.state_dict().items()}
        runtime = self.controller(model).runtime
        control = ResidentWeightShards(model, runtime, scatter_chunk_bytes=8,
                                       distribution="broadcast")
        with patch.object(dist, "scatter", side_effect=AssertionError("scatter forbidden")):
            control.distribute()
        for index, block in enumerate(model.transformer_blocks):
            with torch.no_grad(), control.block(index):
                for key, tensor in block.state_dict().items():
                    torch.testing.assert_close(tensor, expected[f"transformer_blocks.{index}.{key}"], rtol=0, atol=0)
        self.assertEqual(control.summary()["distribution"], "broadcast")
        control.release()

    def test_cross_block_shared_is_replicated(self):
        model = Toy().eval()
        model.transformer_blocks[1].weight = model.transformer_blocks[0].weight
        control = self.controller(model); control.distribute()
        self.assertIs(model.transformer_blocks[0].weight, model.transformer_blocks[1].weight)
        self.assertFalse(model.transformer_blocks[1].weight.is_meta)
        control.release()

    def test_distinct_storage_alias_refused(self):
        model = Toy().eval()
        model.view = torch.nn.Parameter(model.root.weight.view(-1))
        with self.assertRaisesRegex(ValueError, "storage views"):
            self.controller(model)

    def test_grad_and_double_distribution_refused(self):
        model = Toy().eval()
        control = self.controller(model); control.distribute()
        with self.assertRaises(RuntimeError), control.block(0):
            pass
        with self.assertRaises(RuntimeError):
            control.distribute()
        control.release()

    def test_header_dtype_and_duplicate_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "weights.safetensors")
            save_file({"w": torch.zeros(2, 3, dtype=torch.bfloat16)}, path)
            state = header_tensors([path])
            self.assertTrue(state["w"].is_meta)
            self.assertEqual(state["w"].dtype, torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                header_tensors([path, path])
            save_file({"w": torch.zeros(2, 3, dtype=torch.float16)}, path)
            with self.assertRaisesRegex(ValueError, "dtype"):
                header_tensors([path])

    def test_policy_fail_closed(self):
        cfg = OmegaConf.merge(OmegaConf.structured(InferenceConfig),
                              OmegaConf.load("configs/inference/8nfe_rtx5090_8gpu.yaml"))
        validate_memory(cfg)
        for key, value in [("precision.fp8.enabled", True), ("kernels.inference_kernels", True),
                           ("memory.local_files_only", False), ("memory.shard_world_size", 4),
                           ("memory.scatter_chunk_mib", 100), ("parallel.softmax_ranks", 6),
                           ("kernels.softmax_backend", "auto"), ("ablation.enabled", True)]:
            changed = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{key}={str(value).lower()}"]))
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_memory(changed)
        with self.assertRaisesRegex(ValueError, "infer_ulysses"):
            validate_single_process(cfg)

    def test_real_meta_headers_match_baseline_owner_assembly(self):
        from diffusers import MiniMaxH3Transformer3DModel
        from src.models.hybrid_transform import apply_hybrid_attention_transform, hybrid_new_parameters
        from src.models.model_spec import ModelSpec, BaseSpec, TransformSpec, AdapterSpec
        from src.checkpoints.schema import CheckpointArtifact
        from src.checkpoints.export_dir import export_checkpoint_dir
        from src.inference.utils.assemble import build_inference_model, load_released_decoders
        from src.inference.utils.weight_sharding import build_meta_model
        from src.inference.utils.cpu_loading import load_cpu_model
        from src.inference.utils.lora import load_external_lora, merge_lora_state
        with tempfile.TemporaryDirectory() as root:
            base = os.path.join(root, "base")
            source = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
                hidden_size=32, num_layers=2, num_refiner_layers=1, ffn_dim=64,
                in_channels=4, audio_in_channels=4, patch_size=(1, 1, 1), text_dim=8,
                freq_dim=16, time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=2)
            source.to(torch.bfloat16)
            for name in source._keep_in_fp32_modules:
                getattr(source, name).float()
            source.save_pretrained(os.path.join(base, "transformer"))
            transform = dict(enable_softmax_gate=True, anchor_frames="both",
                softmax_attention=dict(radius=1, chunk=0), linear_attention=dict(delta_rule="sana_scaled",
                linear_head_dim=16, bridge="alpha", a_fp32=True, short_conv=dict(targets=["k", "v"]),
                enable_text_state=True))
            apply_hybrid_attention_transform(source, transform)
            weights = {name: p.detach().to(torch.bfloat16) for _, pairs in hybrid_new_parameters(source)
                       for name, p in pairs}
            target = "transformer_blocks.0.attn.orig.to_q"
            for name in ("default", "turbo"):
                weights[f"{target}.lora_A.{name}.weight"] = torch.randn(2, 32).bfloat16()
                weights[f"{target}.lora_B.{name}.weight"] = torch.randn(32, 2).bfloat16()
            spec = ModelSpec(2, BaseSpec("diffusers", "MiniMaxH3Transformer3DModel", base,
                "transformer", None, {"hidden_size": 32}), [TransformSpec("hybrid_attention", 2, transform)],
                [AdapterSpec("lora", 1, {"name": name, "rank": 2, "alpha": 2, "targets": [target]})
                 for name in ("default", "turbo")])
            checkpoint = os.path.join(root, "checkpoint")
            export_checkpoint_dir(CheckpointArtifact(kind="weights", model_spec=spec.to_dict(),
                                                      weights=weights), checkpoint)
            external = os.path.join(root, "external.safetensors")
            save_file({"blocks.0.attn.qkv_proj.lora_A.default.weight": torch.randn(2, 32).bfloat16(),
                       "blocks.0.attn.qkv_proj.lora_B.default.weight": torch.randn(96, 2).bfloat16(),
                       "blocks.0.mlp.fc1.lora_A.default.weight": torch.randn(2, 32).bfloat16(),
                       "blocks.0.mlp.fc1.lora_B.default.weight": torch.randn(128, 2).bfloat16()}, external)
            cfg = OmegaConf.merge(OmegaConf.structured(InferenceConfig),
                                  OmegaConf.load("configs/inference/8nfe_rtx5090_8gpu.yaml"))
            cfg.base_source, cfg.checkpoint = base, checkpoint
            cfg.external_loras = [ExternalLora(path=external, alpha=1.0)]
            meta, _ = build_meta_model(cfg)
            real_to = MiniMaxH3Transformer3DModel.to
            def no_cuda(model, *args, **kwargs):
                destination = kwargs.get("device", args[0] if args else None)
                if isinstance(destination, (str, torch.device)):
                    self.assertNotEqual(torch.device(destination).type, "cuda")
                return real_to(model, *args, **kwargs)
            with patch.object(MiniMaxH3Transformer3DModel, "to", no_cuda):
                owner = build_inference_model(cfg, "cpu", load_decoders=False, log=False, _shard_owner=True)
            self.assertEqual(self.controller(meta).manifest(), self.controller(owner.transformer).manifest())
            self.assertIsNone(owner.vae)
            torch.testing.assert_close(meta.rope.inv_freq, owner.transformer.rope.inv_freq, atol=0, rtol=0)
            reference = load_cpu_model(MiniMaxH3Transformer3DModel, base, subfolder="transformer",
                                       torch_dtype=torch.bfloat16, local_files_only=True)
            apply_hybrid_attention_transform(reference, transform)
            from src.models.factory import load_model_weights
            load_model_weights(reference, {k: v for k,v in weights.items() if "lora_" not in k})
            for name in ("default", "turbo"):
                merge_lora_state(reference, {k: v for k,v in weights.items() if f".{name}." in k})
            state, scale, _ = load_external_lora(external, 1.0)
            merge_lora_state(reference, state, scale)
            for block in reference.transformer_blocks:
                block.attn.to(torch.bfloat16)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(value, owner.transformer.state_dict()[key], atol=0, rtol=0)
            with self.assertRaisesRegex(RuntimeError, "released"):
                load_released_decoders(cfg, owner)


if __name__ == "__main__":
    unittest.main()
