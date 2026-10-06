"""Fail-closed, low-memory CPU loading for the opt-in inference policy."""
import json
import os

import torch
from safetensors import safe_open

from src.paths import (HF_WEIGHTS_REPO, LEGACY_WEIGHTS_PREFIX, WEIGHTS_PREFIX,
                       resolve_repo_path, resolve_weights)


def resolve_inference_weights(path, *, local_files_only=False):
    """Keep the legacy resolver, but never let its Hub fallback bypass offline mode."""
    if not local_files_only:
        return resolve_weights(path)
    if path is not None and path.startswith(LEGACY_WEIGHTS_PREFIX):
        path = WEIGHTS_PREFIX + path[len(LEGACY_WEIGHTS_PREFIX):]
    local = resolve_repo_path(path)
    if local is None or os.path.exists(local):
        return local
    if path.startswith(WEIGHTS_PREFIX):
        from huggingface_hub import snapshot_download

        name = path[len(WEIGHTS_PREFIX):].strip("/")
        root = snapshot_download(HF_WEIGHTS_REPO, allow_patterns=[f"{name}/**"],
                                 local_files_only=True)
        local = os.path.join(root, name)
    if not os.path.exists(local):
        raise FileNotFoundError(f"local_files_only: weights not cached: {path}")
    return local


def assert_cpu_materialized(model):
    """State dict alone misses nonpersistent buffers such as H3 RoPE inv_freq."""
    bad = [f"{name} ({tensor.device})" for name, tensor in
           list(model.named_parameters()) + list(model.named_buffers())
           if tensor.device.type != "cpu"]
    if bad:
        raise RuntimeError(f"CPU assembly has unmaterialized/non-CPU tensors: {bad[:8]}")


def verify_checkpoint_tensors(model, files):
    """Check real safetensors headers, not merely the potentially stale shard index."""
    expected = model.state_dict()
    seen = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key in seen:
                    raise RuntimeError(f"Duplicate checkpoint tensor: {key}")
                seen[key] = os.path.basename(path)
                if key not in expected:
                    raise RuntimeError(f"Unexpected checkpoint tensor: {key}")
                tensor = shard.get_slice(key)
                if tuple(tensor.get_shape()) != tuple(expected[key].shape):
                    raise RuntimeError(f"Wrong checkpoint shape: {key}")
                if tensor.get_dtype().startswith(("F8", "F4")):
                    raise ValueError("CPU assembly does not support quantized checkpoint tensors")
    missing = expected.keys() - seen.keys()
    if missing:
        raise RuntimeError(f"Missing checkpoint tensors: {sorted(missing)[:8]}")
    if files:
        index_path = os.path.join(os.path.dirname(files[0]),
                                  "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as handle:
                index = json.load(handle)["weight_map"]
            if index != seen:
                raise RuntimeError("Checkpoint shard index does not match actual tensor assignments")


def load_cpu_model(model_cls, source, *, subfolder, torch_dtype,
                   local_files_only=False):
    """Use Diffusers' real meta loader, verifying its contract before and after load.

    Diffusers permits missing/unexpected keys by default. A local subclass adds strict
    pre-load key checks without monkeypatching global library state. Native meta loading
    checks every tensor shape (ignore_mismatched_sizes=False). Its empty-weights context
    leaves computed/nonpersistent buffers on CPU, with their initialized values intact.
    No device map dispatch or whole-model accelerator move is involved.
    """
    config = model_cls.load_config(source, subfolder=subfolder, local_files_only=local_files_only)
    if config.get("quantization_config"):
        raise ValueError("CPU assembly does not support embedded quantization_config")
    verified_meta = False

    class StrictCPUModel(model_cls):
        @classmethod
        def _load_pretrained_model(cls, model, state_dict, resolved_model_file,
                                   pretrained_model_name_or_path, loaded_keys, **kwargs):
            nonlocal verified_meta
            if kwargs.get("hf_quantizer") is not None or kwargs.get("device_map") is not None:
                raise ValueError("CPU assembly forbids quantization and device dispatch")
            if not kwargs.get("low_cpu_mem_usage"):
                raise RuntimeError("Diffusers did not enable low_cpu_mem_usage")
            if any(not p.is_meta for p in model.parameters()):
                raise RuntimeError("Diffusers did not construct all parameters on meta")
            if any(b.device.type != "cpu" for b in model.buffers()):
                raise RuntimeError("Diffusers must initialize all buffers on CPU")
            expected, loaded = set(model.state_dict()), set(loaded_keys)
            missing, unexpected = sorted(expected - loaded), sorted(loaded - expected)
            if missing or unexpected:
                raise RuntimeError(f"Incomplete checkpoint: missing keys={missing[:8]}, "
                                   f"unexpected keys={unexpected[:8]}")
            verify_checkpoint_tensors(model, resolved_model_file)
            verified_meta = True
            return super()._load_pretrained_model(
                model, state_dict, resolved_model_file, pretrained_model_name_or_path,
                loaded_keys, **kwargs)

    with torch.device("cpu"):
        model, info = StrictCPUModel.from_pretrained(
            source, subfolder=subfolder, torch_dtype=torch_dtype,
            local_files_only=local_files_only, low_cpu_mem_usage=True,
            device_map=None, use_safetensors=True, output_loading_info=True,
            ignore_mismatched_sizes=False)
    if not verified_meta:
        raise RuntimeError("Diffusers bypassed the verified meta loading path")
    failures = {k: info[k] for k in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(k)}
    if failures:
        raise RuntimeError(f"Incomplete CPU checkpoint load: {failures}")
    assert_cpu_materialized(model)
    if any(type(t) not in (torch.Tensor, torch.nn.Parameter)
           for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("CPU assembly does not support tensor subclasses")
    # Preserve the public class/config identity rather than exporting our local subclass.
    model.__class__ = model_cls
    return model


def validate_branch_weights(model, weights):
    from src.models.hybrid_transform import hybrid_new_parameters

    expected = dict(pair for _, pairs in hybrid_new_parameters(model) for pair in pairs)
    missing, unexpected = sorted(expected.keys() - weights.keys()), sorted(weights.keys() - expected.keys())
    wrong = [name for name in expected.keys() & weights.keys()
             if expected[name].shape != weights[name].shape]
    if missing or unexpected or wrong:
        raise RuntimeError(f"Incomplete branch checkpoint: missing keys={missing[:8]}, "
                           f"unexpected keys={unexpected[:8]}, wrong shapes={wrong[:8]}")
