"""Small synthetic checkpoints and actual H3 GPU forwards; no production weights."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from src.config.inference import ExternalLora, InferenceConfig, validate_memory, validate_parallel
from src.inference.utils.assemble import build_inference_model
from src.inference.utils.block_offload import BlockOffloadController
from src.inference.utils.cpu_loading import (
    assert_cpu_materialized, load_cpu_model, resolve_inference_weights,
    validate_branch_weights,
)
from src.models.hybrid_transform import apply_hybrid_attention_transform, set_layout, set_softmax_backend
from src.models.sequence_layout import SequenceLayout


class BufferedModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width=4):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        self.register_buffer("persistent", torch.arange(width).float())
        self.register_buffer("computed", torch.arange(width).float() + 3, persistent=False)


def tiny_h3():
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=16, hidden_size=32,
        num_layers=2, num_refiner_layers=1, ffn_dim=64, in_channels=4,
        audio_in_channels=4, patch_size=(1, 1, 1), text_dim=8, freq_dim=16,
        time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=2)


def hybrid_config():
    return dict(enable_softmax_gate=True, anchor_frames="none",
                softmax_attention=dict(radius=0, chunk=0),
                linear_attention=dict(delta_rule="vdn_scaled", linear_head_dim=16,
                                      bridge="alpha", a_fp32=True,
                                      short_conv=dict(targets=["k", "v"]),
                                      enable_text_state=False))


def inputs():
    return dict(hidden_states=torch.randn(1, 3, 4, device="cuda"),
                audio_hidden_states=torch.randn(1, 1, 4, device="cuda"),
                encoder_hidden_states=torch.randn(1, 2, 8, device="cuda"),
                timestep=torch.tensor([0.4], device="cuda"),
                timestep_indices=torch.zeros(6, device="cuda", dtype=torch.long),
                token_tags=torch.tensor([1, 1, 2, 0, 0, 0], device="cuda"),
                position_ids=torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                                           [0, 0, 0], [1, 0, 0], [2, 0, 0]], device="cuda"),
                video_indices=torch.tensor([3, 4, 5], device="cuda"),
                audio_indices=torch.tensor([2], device="cuda"),
                text_indices=torch.tensor([0, 1], device="cuda"), return_dict=False)


class CPUAssemblyTests(unittest.TestCase):
    def test_config_defaults_and_rejections(self):
        cfg = InferenceConfig()
        self.assertEqual(cfg.memory.policy, "legacy")
        validate_memory(cfg)
        cfg.memory.policy = "cpu_block_offload"
        with self.assertRaisesRegex(ValueError, "inference_kernels"):
            validate_memory(cfg)
        cfg.kernels.inference_kernels = False
        validate_memory(cfg)
        for section, field, value in [(cfg.precision.fp8, "enabled", True),
                                      (cfg.precision, "dtype", "float32"),
                                      (cfg.behavior, "teacher_mode", True),
                                      (cfg.parallel, "profile", True)]:
            old = getattr(section, field)
            setattr(section, field, value)
            with self.assertRaises(ValueError):
                validate_memory(cfg)
            setattr(section, field, old)
        with self.assertRaisesRegex(ValueError, "Ulysses"):
            validate_parallel(cfg)

    def test_strict_meta_loading_and_buffers(self):
        with tempfile.TemporaryDirectory() as root:
            original = BufferedModel()
            original.save_pretrained(Path(root) / "model")
            loaded = load_cpu_model(BufferedModel, root, subfolder="model",
                                    torch_dtype=torch.float32, local_files_only=True)
            self.assertIs(type(loaded), BufferedModel)
            assert_cpu_materialized(loaded)
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])
            torch.testing.assert_close(loaded.computed, original.computed)
            loaded.computed = torch.empty(4, device="meta")
            with self.assertRaisesRegex(RuntimeError, "computed"):
                assert_cpu_materialized(loaded)

    def test_missing_unexpected_wrong_shapes_and_persistent_buffers(self):
        for corruption in ("missing", "unexpected", "shape", "buffer"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as root:
                BufferedModel().save_pretrained(Path(root) / "model")
                path = Path(root) / "model/diffusion_pytorch_model.safetensors"
                state = load_file(path)
                if corruption == "missing":
                    del state["linear.weight"]
                elif corruption == "unexpected":
                    state["surprise"] = torch.zeros(1)
                elif corruption == "buffer":
                    del state["persistent"]
                else:
                    state["linear.weight"] = torch.zeros(5, 4)
                save_file(state, path)
                with self.assertRaises((RuntimeError, ValueError)):
                    load_cpu_model(BufferedModel, root, subfolder="model",
                                   torch_dtype=torch.float32, local_files_only=True)

    def test_offline_resolver_never_uses_online_fallback(self):
        with patch("src.inference.utils.cpu_loading.resolve_weights", side_effect=AssertionError), \
                patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError) as download:
            with self.assertRaises(FileNotFoundError):
                resolve_inference_weights("ckpts/definitely-missing", local_files_only=True)
            self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_stale_shard_index_rejected(self):
        for corruption in ("missing_buffer", "extra", "duplicate", "assignment"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as root:
                directory = Path(root) / "model"
                BufferedModel().save_pretrained(directory, max_shard_size=32)
                index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
                index = json.loads(index_path.read_text())
                mapping = index["weight_map"]
                buffer_file = directory / mapping["persistent"]
                state = load_file(buffer_file)
                if corruption == "missing_buffer":
                    del state["persistent"]
                elif corruption == "extra":
                    state["unindexed"] = torch.zeros(1)
                elif corruption == "duplicate":
                    other = next(key for key in mapping if mapping[key] != buffer_file.name)
                    state[other] = load_file(directory / mapping[other])[other]
                else:
                    mapping["persistent"] = next(value for value in mapping.values()
                                                  if value != buffer_file.name)
                    index_path.write_text(json.dumps(index))
                save_file(state, buffer_file)
                with self.assertRaises(RuntimeError):
                    load_cpu_model(BufferedModel, root, subfolder="model",
                                   torch_dtype=torch.float32, local_files_only=True)

    def test_embedded_quantization_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "model"
            BufferedModel().save_pretrained(directory)
            config_path = directory / "config.json"
            config = json.loads(config_path.read_text())
            config["quantization_config"] = {"quant_method": "bitsandbytes", "load_in_8bit": True}
            config_path.write_text(json.dumps(config))
            with patch.object(BufferedModel, "from_pretrained", side_effect=AssertionError):
                with self.assertRaisesRegex(ValueError, "quantization_config"):
                    load_cpu_model(BufferedModel, root, subfolder="model",
                                   torch_dtype=torch.float32, local_files_only=True)

    def test_actual_h3_low_memory_cpu_load(self):
        with tempfile.TemporaryDirectory() as root:
            model = tiny_h3()
            model.save_pretrained(Path(root) / "transformer", max_shard_size="40KB")
            loaded = load_cpu_model(type(model), root, subfolder="transformer",
                                    torch_dtype=torch.bfloat16, local_files_only=True)
            assert_cpu_materialized(loaded)
            self.assertEqual(loaded.transformer_blocks[0].attn.to_q.weight.dtype, torch.bfloat16)
            self.assertEqual(loaded.proj_in.weight.dtype, torch.float32)
            torch.testing.assert_close(loaded.rope.inv_freq, model.rope.inv_freq)

    def test_branch_completeness(self):
        from src.models.hybrid_transform import hybrid_new_parameters
        model = tiny_h3()
        apply_hybrid_attention_transform(model, hybrid_config())
        weights = {name: p.detach().clone() for _, pairs in hybrid_new_parameters(model) for name, p in pairs}
        validate_branch_weights(model, weights)
        key = next(iter(weights))
        value = weights.pop(key)
        with self.assertRaisesRegex(RuntimeError, "missing keys"):
            validate_branch_weights(model, weights)
        weights[key] = value.flatten()[:1]
        with self.assertRaisesRegex(RuntimeError, "wrong shapes"):
            validate_branch_weights(model, weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GPUOffloadTests(unittest.TestCase):
    def _parity(self, hybrid, backend="ref"):
        torch.manual_seed(72)
        with tempfile.TemporaryDirectory() as root:
            tiny_h3().save_pretrained(Path(root) / "transformer")
            model = load_cpu_model(MiniMaxH3Transformer3DModel, root, subfolder="transformer",
                                   torch_dtype=torch.bfloat16, local_files_only=True)
        if hybrid:
            apply_hybrid_attention_transform(model, hybrid_config())
            for block in model.transformer_blocks:
                block.attn.to(torch.bfloat16)
            set_softmax_backend(model, backend)
            set_layout(model, SequenceLayout(seq_len=6, video_start=3, num_frames=3,
                                             tokens_per_frame=1, frame_height=1, frame_width=1,
                                             text_start=0, text_len=2))
        model.eval().requires_grad_(False)
        resident = copy.deepcopy(model).to("cuda")
        original_ptrs = {name: tensor.data_ptr() for name, tensor in
                         list(model.named_parameters()) + list(model.named_buffers())}
        controller = BlockOffloadController(model, "cuda:0")
        self.assertEqual(len(controller.groups), 3)
        for group in controller.groups:
            self.assertIsNone(group.stream)
            self.assertFalse(group.non_blocking)
        # No leaf hooks, even on direct-read branch projections/convolutions.
        self.assertFalse(hasattr(model.transformer_blocks[0].attn, "_diffusers_hook"))
        def check_residency(active_block, args):
            for block in model.transformer_blocks:
                expected_device = "cuda" if block is active_block else "cpu"
                self.assertTrue(all(p.device.type == expected_device for p in block.parameters()))
            self.assertEqual(model.rope.inv_freq.device.type, "cuda")
            self.assertEqual(model.token_refiner.final_norm.weight.device.type, "cuda")
        # Attention's pre-hook runs inside the block's Diffusers onload wrapper.
        handles = [block.attn.register_forward_pre_hook(
            lambda module, args, block=block: check_residency(block, args))
            for block in model.transformer_blocks]
        batch = inputs()
        with torch.no_grad():
            reference = resident(**batch)
            for _ in range(3):
                real_to = torch.Tensor.to
                def forbid_d2h(tensor, *args, **kwargs):
                    destination = kwargs.get("device", args[0] if args else None)
                    if isinstance(destination, (str, torch.device)):
                        if tensor.device.type == "cuda" and torch.device(destination).type == "cpu":
                            raise AssertionError("unexpected device-to-host tensor transfer")
                    return real_to(tensor, *args, **kwargs)
                with patch.object(torch.Tensor, "to", forbid_d2h):
                    result = model(**batch)
                for actual, expected in zip(result, reference):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                controller.release()
                controller.release()
                assert_cpu_materialized(model)
                for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                    self.assertEqual(tensor.data_ptr(), original_ptrs[name])
            # A failure after root/block onload must leave a usable model on CPU.
            def fail(*args, **kwargs):
                raise RuntimeError("injected forward failure")
            with patch.object(model.transformer_blocks[0].attn, "forward", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    model(**batch)
            assert_cpu_materialized(model)
            for actual, expected in zip(model(**batch), reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "no_grad"):
            model(**batch)
        assert_cpu_materialized(model)
        for handle in handles:
            handle.remove()

    def test_actual_dense_h3_parity_repeat_release_exception(self):
        self._parity(False)

    def test_actual_hybrid_h3_direct_reads_parity_repeat_release_exception(self):
        self._parity(True)

    def test_actual_hybrid_h3_flex_parity_repeat_release_exception(self):
        self._parity(True, "flex")

    def test_build_never_moves_whole_transformer_to_cuda(self):
        with tempfile.TemporaryDirectory() as root:
            tiny_h3().save_pretrained(Path(root) / "transformer")
            cfg = OmegaConf.structured(InferenceConfig)
            cfg.base_source = root
            cfg.memory.policy = "cpu_block_offload"
            cfg.memory.local_files_only = True
            cfg.kernels.inference_kernels = False
            real_to = MiniMaxH3Transformer3DModel.to
            def guarded_to(model, *args, **kwargs):
                device = kwargs.get("device", args[0] if args else None)
                if device is not None and not isinstance(device, torch.dtype):
                    self.assertNotEqual(torch.device(device).type, "cuda")
                return real_to(model, *args, **kwargs)
            from src.inference.utils.metrics import PhaseMetrics
            metrics = PhaseMetrics(str(Path(root) / "assembly.metrics.jsonl"), "cuda:0")
            before = torch.cuda.memory_allocated()
            with patch.object(MiniMaxH3Transformer3DModel, "to", guarded_to), metrics.phase("cpu_assembly"):
                result = build_inference_model(cfg, "cuda:0", load_decoders=False, log=False)
            event = metrics.events[-1]
            self.assertEqual(event["cuda_peak_allocated_bytes"], before)
            print("tiny CPU assembly measurement: " + json.dumps(event, sort_keys=True), flush=True)
            self.assertIsNotNone(result.offload)
            assert_cpu_materialized(result.transformer)
            with torch.no_grad():
                result.transformer(**inputs())
            result.offload.release()

    def test_compile_and_training_rejected(self):
        for target in ("root", "block"):
            model = tiny_h3().eval().requires_grad_(False)
            compiled = model if target == "root" else model.transformer_blocks[0]
            compiled.compile(backend="eager")
            with self.assertRaisesRegex(ValueError, "compile"):
                BlockOffloadController(model, "cuda:0")
        with self.assertRaisesRegex(ValueError, "eval"):
            BlockOffloadController(tiny_h3(), "cuda:0")

    def test_hybrid_checkpoint_and_external_lora_cpu_merge_order(self):
        from src.inference.utils.lora import merge_lora_state
        from src.models.hybrid_transform import hybrid_new_parameters
        with tempfile.TemporaryDirectory() as root:
            tiny_h3().save_pretrained(Path(root) / "transformer")
            reference = load_cpu_model(MiniMaxH3Transformer3DModel, root, subfolder="transformer",
                                       torch_dtype=torch.bfloat16, local_files_only=True)
            apply_hybrid_attention_transform(reference, hybrid_config())
            weights = {name: p.detach().clone() for _, pairs in hybrid_new_parameters(reference)
                       for name, p in pairs}
            target = "transformer_blocks.0.attn.orig.to_q"
            own_lora = {target + ".lora_A.default.weight": torch.randn(2, 32) * 0.1,
                        target + ".lora_B.default.weight": torch.randn(32, 2) * 0.1,
                        target + ".lora_A.turbo.weight": torch.randn(2, 32) * 0.1,
                        target + ".lora_B.turbo.weight": torch.randn(32, 2) * 0.1}
            external_lora = {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(2, 32) * 0.1,
                             "transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(32, 2) * 0.1}
            merge_lora_state(reference, own_lora)
            merge_lora_state(reference, external_lora, 0.5)
            for block in reference.transformer_blocks:
                block.attn.to(torch.bfloat16)
            artifact = SimpleNamespace(metadata={}, weights={**weights, **own_lora},
                                       model_spec={"base": {"source": root},
                                                   "transforms": [{"config": hybrid_config()}]})
            cfg = OmegaConf.structured(InferenceConfig)
            cfg.checkpoint = root
            cfg.base_source = root
            cfg.external_loras = [ExternalLora(path="synthetic-external")]
            cfg.memory.policy = "cpu_block_offload"
            cfg.memory.local_files_only = True
            cfg.kernels.inference_kernels = False
            cfg.kernels.softmax_backend = "ref"
            events = []
            def checked_merge(model, state, scale=1.0):
                assert_cpu_materialized(model)
                events.append("checkpoint" if state.keys() == own_lora.keys() else "external")
                return merge_lora_state(model, state, scale)
            real_to = MiniMaxH3Transformer3DModel.to
            def guarded_to(model, *args, **kwargs):
                destination = kwargs.get("device", args[0] if args else None)
                if isinstance(destination, (str, torch.device)):
                    self.assertEqual(torch.device(destination).type, "cpu")
                return real_to(model, *args, **kwargs)
            with patch("src.inference.utils.assemble.load_checkpoint", return_value=artifact), \
                    patch("src.inference.utils.assemble.load_external_lora", return_value=(external_lora, 0.5, [2])), \
                    patch("src.inference.utils.assemble.merge_lora_state", side_effect=checked_merge), \
                    patch.object(MiniMaxH3Transformer3DModel, "to", guarded_to):
                assembled = build_inference_model(cfg, "cuda:0", load_decoders=False, log=False)
            self.assertEqual(events, ["checkpoint", "external"])
            self.assertEqual(assembled.merged_lora_pairs, 2)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(assembled.transformer.state_dict()[name], value, rtol=0, atol=0)
            assembled.offload.release()


if __name__ == "__main__":
    print(f"Python/Torch: {os.sys.version} / {torch.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    unittest.main(verbosity=2)
