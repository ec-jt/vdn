"""Synchronous whole-H3-block offload with retained immutable CPU backing.

Diffusers ModuleGroup only retains CPU backing when streams are enabled. Its default
no-stream path calls module.to(cpu) every forward, copying immutable weights back.
This specialization uses the same ModuleGroup/GroupOffloadingHook protocol but keeps
CPU storage without streams, pinning, prefetch, or device-to-host weight copies.
"""
import torch
from diffusers.hooks.group_offloading import (
    GroupOffloadingConfig, GroupOffloadingHook, GroupOffloadingType, ModuleGroup,
)
from diffusers.hooks.hooks import HookRegistry

from src.inference.utils.cpu_loading import assert_cpu_materialized


class RetainedCPUGroup(ModuleGroup):
    def _init_cpu_param_dict(self):
        tensors = list(self.parameters) + list(self.buffers)
        for module in self.modules:
            tensors.extend(module.parameters())
            tensors.extend(module.buffers())
        # detach shares the original CPU storage and survives tensor.data replacement.
        return {tensor: tensor.detach() for tensor in tensors}

    def _onload_from_memory(self):
        for tensor, backing in self.cpu_param_dict.items():
            tensor.data = backing.to(self.onload_device, non_blocking=False)

    def _offload_to_memory(self):
        for tensor, backing in self.cpu_param_dict.items():
            tensor.data = backing


class BlockOffloadController:
    """Inference-only, serial/reentrant-across-calls, not concurrent or trainable.

    Every block is one group, including children read directly without forward calls.
    The root group owns every other tensor: endpoints, whole text refiner, root and
    nonpersistent buffers. It lives for the duration of a transformer forward only.
    release() is idempotent, does not uninstall hooks, and works after exceptions and
    before decoder onload. Do not cast, mutate, compile, or move the model after install.
    """
    def __init__(self, model, execution_device):
        assert_cpu_materialized(model)
        device = torch.device(execution_device)
        if device.type != "cuda":
            raise ValueError("cpu_block_offload requires a CUDA execution device")
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("block offload requires eval and frozen parameters")
        if any(type(t) not in (torch.Tensor, torch.nn.Parameter)
               for t in list(model.parameters()) + list(model.buffers())):
            raise ValueError("block offload does not support quantized tensor subclasses")
        for module in model.modules():
            if (hasattr(module, "_hf_hook") or hasattr(module, "_orig_mod") or
                    getattr(module, "_compiled_call_impl", None) is not None or
                    getattr(module, "gradient_checkpointing", False) or
                    getattr(module, "_diffusers_hook", None)):
                raise ValueError("block offload cannot be combined with existing hooks, compile, or checkpointing")
        blocks = list(model.transformer_blocks)
        if not blocks:
            raise ValueError("block offload requires transformer_blocks")
        self.execution_device = device
        self.groups = []
        self.model = model
        config = GroupOffloadingConfig(
            onload_device=device, offload_device=torch.device("cpu"),
            offload_type=GroupOffloadingType.BLOCK_LEVEL, non_blocking=False,
            record_stream=False, low_cpu_mem_usage=False, num_blocks_per_group=1,
            stream=None)
        owned = set()
        for block in blocks:
            tensors = set(block.parameters()) | set(block.buffers())
            if owned & tensors:
                raise ValueError("shared tensors between H3 blocks are unsupported")
            owned.update(tensors)
            self.groups.append(RetainedCPUGroup(
                modules=[block], onload_device=device, offload_device=torch.device("cpu"),
                onload_leader=block, offload_leader=block))
        self.groups.append(RetainedCPUGroup(
            modules=[], onload_device=device, offload_device=torch.device("cpu"),
            onload_leader=model, offload_leader=model,
            parameters=[p for p in model.parameters() if p not in owned],
            buffers=[b for b in model.buffers() if b not in owned]))
        covered = [id(t) for group in self.groups for t in group.cpu_param_dict]
        expected = {id(t) for t in list(model.parameters()) + list(model.buffers())}
        if len(covered) != len(set(covered)) or set(covered) != expected:
            raise RuntimeError("offload groups must cover every parameter and buffer exactly once")
        for group in self.groups:
            registry = HookRegistry.check_if_exists_or_initialize(group.onload_leader)
            registry.register_hook(GroupOffloadingHook(group, config=config), "group_offloading")
        self._guard_handle = model.register_forward_pre_hook(self._guard)
        # Diffusers post_forward is not exception-safe. PyTorch's always_call cleanup is.
        self._release_handle = model.register_forward_hook(self._after_forward, always_call=True)

    def _guard(self, module, args):
        if module.training or torch.is_grad_enabled():
            raise RuntimeError("block offload requires eval and torch.no_grad()/inference_mode()")

    def _after_forward(self, module, args, output):
        self.release()

    def release(self):
        """Drop all accelerator weight references; preserve CPU storage and live hooks."""
        for group in self.groups:
            group.offload_()
