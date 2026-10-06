"""CPU-only encoder contract checks. No released weights or GPU execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from src.inference import encode_prompt as ep
from src.inference.utils.prompt_cache import load_prompt


ROOT = Path(__file__).resolve().parents[1]


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.processor = self.root / "processor"
        self.model = self.root / "encoder"
        self.processor.mkdir()
        self.model.mkdir()
        (self.processor / "tokenizer_config.json").write_text("{}")
        (self.model / "config.json").write_text('{"model_type": "qwen3_vl"}')
        # A sentinel for mocked loader tests, never loaded by Transformers.
        (self.model / "model.safetensors").write_bytes(b"mock weights")

    def local_args(self):
        return ["--local_only", "--processor_root", str(self.processor),
                "--text_encoder_root", str(self.model), "--prompt", "verbatim prompt",
                "--out", str(self.root / "cache.pt")]

    def run_cli(self, *args, module=False):
        command = [sys.executable]
        command += ["-m", "src.inference.encode_prompt"] if module else [str(ROOT / "src/inference/encode_prompt.py")]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        env.pop("PYTHONPATH", None)
        return subprocess.run(command + list(args), cwd=ROOT if module else self.root,
                              env=env, text=True, capture_output=True, timeout=30)

    def test_direct_and_module_help_exit_without_gpu(self):
        for module in (False, True):
            result = self.run_cli("--help", module=module)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--cpu_offload", result.stdout)

    def test_missing_roots_cli_fails_before_work(self):
        result = self.run_cli("--cpu_offload", "--prompt", "test", "--out", str(self.root / "out.pt"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("explicit --processor_root", result.stderr)
        self.assertFalse((self.root / "out.pt").exists())

    def test_local_validation_missing_weights_and_shards(self):
        (self.model / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "no local safetensors"):
            ep.local_roots(self.processor, self.model)
        index = self.model / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "missing or out-of-root"):
            ep.local_roots(self.processor, self.model)
        (self.model / "shard.safetensors").write_bytes(b"mock")
        index.write_text(json.dumps({"weight_map": {"weight": "shard.safetensors"}}))
        self.assertEqual(ep.local_roots(self.processor, self.model), (str(self.processor), str(self.model)))
        index.write_text(json.dumps({"weight_map": {"weight": "../processor/tokenizer_config.json"}}))
        with self.assertRaisesRegex(ValueError, "out-of-root"):
            ep.local_roots(self.processor, self.model)

    def test_local_rejects_quantized_weights(self):
        (self.model / "config.json").write_text('{"quantization_config": {"quant_method": "fp8"}}')
        with self.assertRaisesRegex(ValueError, "unquantized"):
            ep.local_roots(self.processor, self.model)

    def test_no_overwrite_fails_before_any_load(self):
        (self.root / "cache.pt").write_bytes(b"do not change")
        with patch.object(ep, "upstream_snapshot", side_effect=AssertionError("network")), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ep.main(self.local_args() + ["--no_overwrite"])
        self.assertEqual((self.root / "cache.pt").read_bytes(), b"do not change")

    def test_local_load_options_and_no_whole_model_cuda_move(self):
        processor, encoder = Mock(), Mock()
        encoder.eval.return_value = encoder
        encoder.requires_grad_.return_value = encoder
        with patch("transformers.Qwen3VLProcessor.from_pretrained", return_value=processor) as pload, \
                patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=encoder) as mload, \
                patch.object(ep, "offload_encoder", return_value=encoder) as offload, \
                patch.object(ep, "encode") as encode, \
                patch.object(ep, "upstream_snapshot", side_effect=AssertionError("network")), \
                patch.dict(os.environ, {}, clear=False):
            ep.main(self.local_args() + ["--cpu_offload", "--device", "cuda:0"])
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        pload.assert_called_once_with(str(self.processor), local_files_only=True, trust_remote_code=False)
        self.assertEqual(mload.call_args.kwargs["device_map"], {"": "cpu"})
        self.assertTrue(mload.call_args.kwargs["local_files_only"])
        self.assertFalse(mload.call_args.kwargs["trust_remote_code"])
        self.assertTrue(mload.call_args.kwargs["use_safetensors"])
        offload.assert_called_once_with(encoder, "cuda:0")
        encoder.to.assert_not_called()
        self.assertEqual(encode.call_args.kwargs["max_tokens"], 1024)

    def test_legacy_defaults_and_original_batch_indices(self):
        encoder = Mock()
        encoder.eval.return_value = encoder
        encoder.requires_grad_.return_value = encoder
        encoder.to.return_value = encoder
        rows = self.root / "rows.jsonl"
        rows.write_text('\n'.join(json.dumps({"prompt": str(i)}) for i in range(3)))
        with patch("transformers.Qwen3VLProcessor.from_pretrained") as pload, \
                patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=encoder) as mload, \
                patch.object(ep, "upstream_snapshot", return_value="legacy-snapshot") as snapshot, \
                patch.object(ep, "encode") as encode:
            ep.main(["--jsonl", str(rows), "--rows", "2", "0", "--out_dir", str(self.root)])
        snapshot.assert_called_once_with("processor", "text_encoder")
        pload.assert_called_once_with("legacy-snapshot", subfolder="processor")
        mload.assert_called_once_with("legacy-snapshot", subfolder="text_encoder", dtype=torch.bfloat16)
        encoder.to.assert_called_once_with("cuda:0")
        self.assertEqual([c.args[2] for c in encode.call_args_list], ["2", "0"])
        self.assertEqual([Path(c.args[3]).name for c in encode.call_args_list], ["prompt_2.pt", "prompt_0.pt"])
        self.assertTrue(all(c.kwargs["max_tokens"] is None for c in encode.call_args_list))

    def test_cache_contract_and_token_bound(self):
        processor = Mock()
        processor.tokenizer.return_value = {"input_ids": [2, 3, 4]}
        processor.create_mm_token_type_ids.return_value = [[0, 0, 0]]
        encoder = Mock()
        states = tuple(torch.full((1, 3, 5120), float(i)) for i in range(52))
        encoder.model.return_value = SimpleNamespace(hidden_states=states)
        out = self.root / "cache.pt"
        ep.encode(processor, encoder, " exact text ", out, "cpu", max_tokens=3, no_overwrite=True)
        processor.tokenizer.assert_called_once_with(" exact text ", add_special_tokens=False)
        kwargs = encoder.model.call_args.kwargs
        self.assertFalse(kwargs["use_cache"])
        self.assertTrue(kwargs["output_hidden_states"])
        cache = torch.load(out, weights_only=True)
        self.assertEqual(set(cache), {"prompt", "prompt_embeds", "text_token_tags"})
        self.assertEqual(cache["prompt"], " exact text ")
        embeds, tags, conditions = load_prompt(str(out), "cpu")
        self.assertEqual(embeds.shape, (3, 5120))
        self.assertEqual(embeds.dtype, torch.bfloat16)
        self.assertTrue(torch.all(embeds == 50))
        self.assertEqual(tags.dtype, torch.long)
        self.assertTrue(torch.all(tags == 1))
        self.assertIsNone(conditions)
        with self.assertRaisesRegex(ValueError, "allowed range"):
            ep.encode(processor, encoder, "too long", self.root / "bad.pt", "cpu", max_tokens=2)
        self.assertFalse((self.root / "bad.pt").exists())
        with self.assertRaises(FileExistsError):
            ep.encode(processor, encoder, "replacement", out, "cpu", no_overwrite=True)
        self.assertEqual(torch.load(out, weights_only=True)["prompt"], " exact text ")

    def test_launcher_refuses_missing_roots_and_exits(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        for key in ("PROCESSOR_ROOT", "TEXT_ENCODER_ROOT", "ENCODER_GPU"):
            env.pop(key, None)
        result = subprocess.run(["bash", str(ROOT / "scripts/inference/encode_prompt_rtx5090.sh")],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PROCESSOR_ROOT", result.stderr)

    def test_mocked_local_encode_save_process_exits(self):
        code = '''
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from src.inference import encode_prompt as ep
processor, encoder = Mock(), Mock()
processor.tokenizer.return_value = {"input_ids": [2, 3]}
processor.create_mm_token_type_ids.return_value = [[0, 0]]
encoder.eval.return_value = encoder
encoder.requires_grad_.return_value = encoder
encoder.to.return_value = encoder
encoder.model.return_value = SimpleNamespace(hidden_states=(torch.ones(1, 2, 5120),) * 51)
with patch("transformers.Qwen3VLProcessor.from_pretrained", return_value=processor), \\
     patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=encoder), \\
     patch.object(ep, "upstream_snapshot", side_effect=AssertionError("network")):
    ep.main(sys.argv[1:])
assert not torch.cuda.is_initialized()
print("CPU mocked encoder phase finished")
'''
        result = subprocess.run([sys.executable, "-c", code, *self.local_args(), "--device", "cpu"],
                                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPU mocked encoder phase finished", result.stdout)
        cache = torch.load(self.root / "cache.pt", weights_only=True)
        self.assertEqual(cache["prompt_embeds"].device.type, "cpu")
        self.assertEqual(cache["prompt"], "verbatim prompt")


class AccelerateCPUHookTests(unittest.TestCase):
    def test_tiny_actual_qwen_whole_layers_offload_and_repeat(self):
        cuda_initialized = torch.cuda.is_initialized()
        cuda_allocated = torch.cuda.memory_allocated() if cuda_initialized else 0
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, head_dim=8,
                             rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                               num_heads=2, out_hidden_size=16, deepstack_visual_indexes=[]),
        )
        model = Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
        inputs = dict(input_ids=torch.tensor([[2, 3, 4]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                      mm_token_type_ids=torch.zeros(1, 3, dtype=torch.long),
                      use_cache=False, output_hidden_states=True)
        with torch.no_grad():
            expected = model.model(**inputs).hidden_states
        # A torch.device CPU target exercises real offload hooks on CPU: the
        # installed dispatch API distinguishes it from its string "cpu" sentinel.
        ep.offload_encoder(model, torch.device("cpu"))
        layers = model.model.language_model.layers
        self.assertTrue(all(layer._hf_hook.offload and layer._hf_hook.place_submodules for layer in layers))
        self.assertFalse(hasattr(layers[0].mlp.down_proj, "_hf_hook"))
        self.assertTrue(all(p.device.type == "meta" for p in model.parameters()))
        visits = []
        for index, layer in enumerate(layers):
            original = layer._old_forward

            def wrapped(*args, _index=index, _original=original, **kwargs):
                self.assertTrue(all(p.device.type == "cpu" for p in layers[_index].parameters()))
                self.assertTrue(all(p.device.type == "meta" for i, other in enumerate(layers)
                                    if i != _index for p in other.parameters()))
                self.assertEqual(model.model.language_model.embed_tokens.weight.device.type, "meta")
                self.assertTrue(all(p.device.type == "meta" for p in model.model.visual.parameters()))
                self.assertEqual(model.lm_head.weight.device.type, "meta")
                visits.append(_index)
                return _original(*args, **kwargs)

            layer._old_forward = wrapped
        for _ in range(2):
            with torch.no_grad():
                actual = model.model(**inputs).hidden_states
            for want, got in zip(expected, actual, strict=True):
                torch.testing.assert_close(want, got)
            self.assertTrue(all(p.device.type == "meta" for p in model.parameters()))
        self.assertEqual(visits, [0, 1, 0, 1])
        self.assertEqual(torch.cuda.is_initialized(), cuda_initialized)
        if cuda_initialized:
            self.assertEqual(torch.cuda.memory_allocated(), cuda_allocated)


if __name__ == "__main__":
    unittest.main()
