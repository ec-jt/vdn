"""Resident weight shards, independent of Ulysses sequence sharding.

Rank zero assembles the strict baseline on CPU (NOT block-bounded CPU assembly).
Other ranks construct only a meta graph. Startup scatters dtype-flat block weights
in bounded chunks. A forward gathers one block, binds views, computes, and restores
meta slots. Optional next-block prefetch overlaps only a communication-free tail.
No CPU backing, weight approximation, or optimizer is retained.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import glob
import hashlib
import json
import math
import os

import torch
import torch.distributed as dist
from safetensors import safe_open

from src.inference.utils.nsight import nvtx_annotate, nvtx_range


def header_tensors(files):
    """Read shapes/dtypes only, never safetensors payloads."""
    types = {"BF16": torch.bfloat16, "F32": torch.float32}
    result = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise ValueError(f"duplicate header tensor: {key}")
                item = handle.get_slice(key)
                if item.get_dtype() not in types:
                    raise ValueError(f"unsupported checkpoint dtype: {key}: {item.get_dtype()}")
                result[key] = torch.empty(item.get_shape(), device="meta",
                                          dtype=types[item.get_dtype()])
    return result


def build_meta_model(cfg):
    """Reproduce baseline graph/casts from local config and actual tensor headers.

    The base's storage dtypes must equal its declared BF16/FP32 load policy. Branch
    and adapter headers are checked, then the existing merger validates meta LoRA
    shapes/targets, including original fused QKV and SwiGLU external adapters.
    """
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
    from src.checkpoints.schema import CheckpointArtifact
    from src.models.model_spec import ModelSpec
    from src.models.factory import _verify_base_config
    from src.models.hybrid_transform import apply_hybrid_attention_transform, iter_hybrids, set_softmax_backend
    from src.inference.utils.cpu_loading import resolve_inference_weights, verify_checkpoint_tensors, validate_branch_weights
    from src.inference.utils.lora import merge_lora_state, _translate_original_layout

    root = resolve_inference_weights(cfg.checkpoint, local_files_only=True)
    with open(os.path.join(root, "model_spec.json")) as f:
        spec = json.load(f)
    with open(os.path.join(root, "metadata.json")) as f:
        metadata = json.load(f)
    if metadata.get("kind") != "weights" or metadata.get("checkpoint_format_version") != 2:
        raise ValueError("weight sharding requires an exploded v2 weights artifact")
    parsed = ModelSpec.from_dict(spec)
    if (parsed.base.library != "diffusers" or parsed.base.class_name != "MiniMaxH3Transformer3DModel"
            or parsed.base.subfolder != "transformer" or len(parsed.transforms) != 1
            or parsed.transforms[0].type != "hybrid_attention"):
        raise ValueError("resident sharding requires the H3 hybrid architecture")
    if metadata.get("metadata", {}).get("truncated_blocks"):
        raise ValueError("truncated checkpoint is not a production model")
    source = resolve_inference_weights(cfg.base_source or spec["base"]["source"], local_files_only=True)
    config = MiniMaxH3Transformer3DModel.load_config(source, subfolder="transformer", local_files_only=True)
    if config.get("quantization_config"):
        raise ValueError("embedded quantization is unsupported")
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    _verify_base_config(model, parsed.base)
    files = sorted(glob.glob(os.path.join(source, "transformer", "*.safetensors")))
    verify_checkpoint_tensors(model, files)
    headers = header_tensors(files)
    for name, param in model.named_parameters():
        dtype = (torch.float32 if any(k in name.split(".") for k in model._keep_in_fp32_modules)
                 else torch.bfloat16)
        if headers[name].dtype != dtype:
            raise ValueError(f"base storage/load dtype mismatch: {name}: {headers[name].dtype} != {dtype}")
        param.data = param.data.to(dtype=dtype)
    # Computed buffers are not in state_dict; never materialize uninitialized meta.
    with torch.device("cpu"):
        model.rope = MiniMaxH3RotaryPosEmbed(config["rope_freq_dim"], config["rope_theta"])
    with torch.device("meta"):
        apply_hybrid_attention_transform(model, spec["transforms"][0]["config"])
    branch = header_tensors([os.path.join(root, "linear_branch", "model.safetensors")])
    validate_branch_weights(model, branch)
    if any(v.dtype != torch.bfloat16 for v in branch.values()):
        raise ValueError("resident sharding currently requires BF16 branch storage")
    folders = sorted(glob.glob(os.path.join(root, "adapters", "*")))
    expected_adapters = {a.config.get("name", "default") for a in parsed.adapters}
    if {os.path.basename(p) for p in folders} != expected_adapters:
        raise ValueError("adapter directories do not match the ModelSpec")
    for folder in folders:
        state = header_tensors([os.path.join(folder, "adapter_model.safetensors")])
        if not state or any(".lora_" not in k for k in state):
            raise ValueError("adapter must contain only nonempty LoRA pairs")
        merge_lora_state(model, state)
    for adapter in cfg.external_loras:
        state = {k.replace(".default.", "."): v for k, v in header_tensors([adapter.path]).items()}
        merge_lora_state(model, _translate_original_layout(state))
    for attn in iter_hybrids(model):
        for param in attn.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
    set_softmax_backend(model, cfg.kernels.softmax_backend)
    model.eval().requires_grad_(False)
    art = CheckpointArtifact(kind="weights", model_spec=spec, weights={}, metadata=metadata.get("metadata", {}))
    return model, art


@dataclass
class TensorSlot:
    names: list[str]
    slots: list[tuple]  # (module, local name, is_parameter)
    shape: tuple
    dtype: torch.dtype
    parameter: bool
    placeholder: torch.Tensor

    @property
    def numel(self):
        return math.prod(self.shape)

    def tensor(self):
        module, name, parameter = self.slots[0]
        return (module._parameters if parameter else module._buffers)[name]

    def bind(self, tensor):
        value = torch.nn.Parameter(tensor, requires_grad=False) if self.parameter else tensor
        for module, name, parameter in self.slots:
            (module._parameters if parameter else module._buffers)[name] = value

    def dematerialize(self):
        for module, name, parameter in self.slots:
            (module._parameters if parameter else module._buffers)[name] = self.placeholder


def tensor_slots(model):
    """Deduplicate identity, including persistent/nonpersistent buffers and aliases."""
    found = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise ValueError("compiled modules are incompatible with weight rebinding")
        if (getattr(module, "_hf_hook", None) is not None
                or getattr(module, "_diffusers_hook", None) is not None):
            raise ValueError("weight sharding cannot coexist with offload/dispatch hooks")
        for parameter, mapping in ((True, module._parameters), (False, module._buffers)):
            for name, tensor in mapping.items():
                if tensor is None:
                    continue
                if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or not tensor.is_contiguous():
                    raise ValueError("weight sharding requires ordinary contiguous tensors")
                full = f"{path}.{name}" if path else name
                if id(tensor) not in found:
                    empty = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
                    placeholder = torch.nn.Parameter(empty, requires_grad=False) if parameter else empty
                    found[id(tensor)] = TensorSlot([], [], tuple(tensor.shape), tensor.dtype, parameter, placeholder)
                slot = found[id(tensor)]
                if slot.parameter != parameter:
                    raise ValueError("a tensor cannot alias both a parameter and buffer")
                slot.names.append(full)
                slot.slots.append((module, name, parameter))
    return list(found.values())


@dataclass
class FlatGroup:
    entries: list[TensorSlot]
    dtype: torch.dtype
    numel: int
    shard_numel: int
    shard: torch.Tensor | None = None


@dataclass
class GatheredBlock:
    index: int
    buffers: list
    works: list
    shards: list
    ready: object = None
    consumed: object = None
    consumer_stream: object = None


class ResidentWeightShards:
    """Own resident block shards and replicated root; never call model.to afterwards.

    ``block(index)`` is serial, no-grad and exception-safe. ``release()`` is terminal,
    idempotent and issues no recovery collectives; prefetch release drains owned work
    and destroys its dedicated group before the caller's decoder barrier.
    Object aliases inside a block remain tied; aliases across blocks live in root.
    Distinct Tensor views sharing a storage are rejected rather than silently untied.
    Compressed weights/scales must be ordinary contiguous registered buffers on
    both graphs before construction. Their dtype, shape and byte order are opaque:
    uint8 packed data and flat swizzled scales share a uint8 group, never a BF16
    conversion. Metadata attributes are not transported and must match on all ranks.
    Only distribution reads CPU payloads; block acquisition uses resident shards.
    """
    def __init__(self, model, runtime, *, scatter_chunk_bytes=32 << 20, distribution="scatter",
                 prefetch=False):
        if model.training:
            raise ValueError("resident sharding requires eval mode")
        if scatter_chunk_bytes <= 0:
            raise ValueError("scatter chunk must be positive")
        if distribution not in ("scatter", "broadcast"):
            raise ValueError("distribution must be scatter or broadcast")
        if type(prefetch) is not bool:
            raise ValueError("prefetch must be a bool")
        if prefetch and (runtime.device.type != "cuda" or runtime.backend != "nccl"
                         or getattr(runtime, "branch_parallel", False)):
            raise ValueError("prefetch requires CUDA/NCCL standard Ulysses")
        self.prefetch = prefetch
        self.weight_group = None
        self.prefetch_stream = None
        self.pending = None
        self._active_gather = None
        self._in_forward = False
        self._next_index = 0
        self._tail_enqueued = False
        self._failed = False
        self.max_live_gathered = 0
        self.distribution = distribution
        self.model, self.runtime = model, runtime
        self.device = runtime.device
        self.world_size, self.rank = runtime.world_size, runtime.rank
        self.scatter_chunk_bytes = scatter_chunk_bytes
        self.entries = tensor_slots(model)
        self.root = []
        self.blocks = [[] for _ in model.transformer_blocks]
        self.active = None
        self.released = False
        self.initialized = False
        self.gather_count = 0
        storages = {}
        for entry in self.entries:
            t = entry.tensor()
            if not t.is_meta and t.numel():
                key = (str(t.device), t.untyped_storage().data_ptr())
                if key in storages:
                    raise ValueError(f"distinct storage views unsupported: {entry.names} and {storages[key]}")
                storages[key] = entry.names
            owners = set()
            for name in entry.names:
                parts = name.split(".")
                owners.add(int(parts[1]) if parts[0] == "transformer_blocks" else -1)
            if len(owners) != 1 or -1 in owners:
                self.root.append(entry)
            else:
                self.blocks[next(iter(owners))].append(entry)
        self.groups = []
        for entries in self.blocks:
            groups = []
            for dtype in sorted({e.dtype for e in entries}, key=str):
                same = [e for e in entries if e.dtype == dtype]
                n = sum(e.numel for e in same)
                groups.append(FlatGroup(same, dtype, n, math.ceil(n / self.world_size)))
            self.groups.append(groups)

    def manifest(self):
        return [(e.names, e.shape, str(e.dtype), e.parameter) for e in self.entries]

    def summary(self):
        size = lambda e: e.numel * torch.empty((), dtype=e.dtype).element_size()
        return dict(root_bytes=sum(map(size, self.root)),
                    resident_shard_bytes=sum(g.shard_numel * torch.empty((), dtype=g.dtype).element_size()
                                             for groups in self.groups for g in groups),
                    max_gather_bytes=max((sum(g.shard_numel * self.world_size * torch.empty((), dtype=g.dtype).element_size()
                                              for g in groups) for groups in self.groups), default=0),
                    blocks=len(self.blocks), gather_count=self.gather_count,
                    assembly="rank0_strict_full_cpu_not_block_bounded", prefetch=self.prefetch,
                    max_live_gathered=self.max_live_gathered,
                    distribution=self.distribution)

    @torch.no_grad()
    def distribute(self):
        if self.initialized or self.released:
            raise RuntimeError("distribution is single-use")
        # These options change the collective schedule, not just local staging.
        # Reject disagreement before any rank starts a payload collective.
        protocol = (self.manifest(), self.prefetch, self.distribution, self.scatter_chunk_bytes)
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        manifests = [None] * self.world_size
        dist.all_gather_object(manifests, digest)
        if len(set(manifests)) != 1:
            raise ValueError("rank graph keys/shapes/dtypes/aliases or transport options differ")
        invalid = (any(e.tensor().device.type != "cpu" for e in self.entries) if self.rank == 0
                   else any(not e.tensor().is_meta for entries in self.blocks for e in entries))
        statuses = [None] * self.world_size
        dist.all_gather_object(statuses, invalid)
        if any(statuses):
            raise ValueError("owner must be CPU and nonowner blocks must be meta before distribution")
        try:
            for entry in self.root:
                value = torch.empty(entry.shape, dtype=entry.dtype, device=self.device)
                flat = value.view(-1)
                chunk = max(1, self.scatter_chunk_bytes // value.element_size())
                for start in range(0, entry.numel, chunk):
                    part = flat[start:start + chunk]
                    if self.rank == 0:
                        part.copy_(entry.tensor().view(-1)[start:start + chunk])
                    dist.broadcast(part, src=0)
                    if self.distribution == "broadcast" and self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                entry.bind(value)
            for groups in self.groups:
                for group in groups:
                    n = group.shard_numel
                    group.shard = torch.empty(n, dtype=group.dtype, device=self.device)
                    flat = None
                    if self.rank == 0:
                        flat = torch.zeros(n * self.world_size, dtype=group.dtype, device="cpu")
                        offset = 0
                        for entry in group.entries:
                            flat[offset:offset + entry.numel].copy_(entry.tensor().view(-1))
                            offset += entry.numel
                    chunk = max(1, self.scatter_chunk_bytes // group.shard.element_size())
                    for start in range(0, n, chunk):
                        count = min(chunk, n - start)
                        if self.distribution == "broadcast":
                            # NCCL 2.29.7 SHM CUDA-copy scatter stalls on this host.
                            # Broadcast one destination chunk at a time instead.
                            # Exact bytes, at most one chunk of send staging/rank;
                            # this adds startup traffic, never denoise traffic.
                            for target in range(self.world_size):
                                value = (flat[target*n + start:target*n + start + count].to(self.device)
                                         if self.rank == 0 else torch.empty(count, dtype=group.dtype, device=self.device))
                                dist.broadcast(value, src=0)
                                if self.rank == target:
                                    group.shard[start:start + count].copy_(value)
                                if self.device.type == "cuda":
                                    torch.cuda.synchronize(self.device)
                                del value
                            continue
                        sends = None
                        if self.rank == 0:
                            sends = [flat[r*n + start:r*n + start + count].to(self.device)
                                     for r in range(self.world_size)]
                        dist.scatter(group.shard[start:start + count], scatter_list=sends, src=0)
                        del sends
                    del flat
                    for entry in group.entries:
                        entry.dematerialize()
            if self.prefetch:
                # All ranks create this group once, after default-group startup is
                # drained. Never create groups opportunistically in the block loop.
                torch.cuda.synchronize(self.device)
                self.weight_group = dist.new_group(
                    ranks=list(range(self.world_size)), backend="nccl",
                    timeout=timedelta(seconds=getattr(self.runtime, "timeout_seconds", 120)))
                self.prefetch_stream = torch.cuda.Stream(device=self.device)
            self.initialized = True
        except BaseException:
            self.release()
            raise

    @contextmanager
    def forward(self):
        """One ordered block sequence; drain even on early return or exceptions."""
        if not self.prefetch:
            yield
            return
        if (not self.initialized or self.released or self._failed or self._in_forward
                or self.active is not None or self.pending is not None):
            raise RuntimeError("prefetch forward requires clean initialized state")
        self._in_forward = True
        self._next_index = 0
        try:
            yield
        finally:
            try:
                self.flush()
            finally:
                self._in_forward = False
                self._next_index = 0

    def _enqueue(self, index):
        if self._failed:
            raise RuntimeError("failed prefetch transport requires release")
        if self.pending is not None:
            raise RuntimeError("only one pending gathered block is permitted")
        bundle = GatheredBlock(index, [], [], [])
        # Own partially launched work before any allocation/collective can fail.
        self.pending = bundle
        self.max_live_gathered = max(self.max_live_gathered,
                                    1 + int(self._active_gather is not None))
        try:
            with nvtx_range(f"prefetch.enqueue.block.{index}"):
                with torch.cuda.stream(self.prefetch_stream):
                    for group in self.groups[index]:
                        flat = torch.empty(group.shard_numel * self.world_size,
                                           dtype=group.dtype, device=self.device)
                        bundle.buffers.append(flat)
                        bundle.shards.append(group.shard)
                        with nvtx_range(f"prefetch.gather.block.{index}.{group.dtype}"):
                            work = dist.all_gather_into_tensor(
                                flat, group.shard, group=self.weight_group, async_op=True)
                            bundle.works.append(work)
                            # Inserts NCCL completion dependency into our stream, not
                            # the consumer stream. Every dtype uses manifest order.
                            work.wait()
                    ready = torch.cuda.Event()
                    ready.record(self.prefetch_stream)
                    bundle.ready = ready
        except BaseException:
            # The bundle can be incomplete even when draining succeeds. Never let
            # a caught enqueue exception turn it into a consumable next block.
            self._failed = True
            raise
        self.gather_count += 1

    def prefetch_next(self):
        """Called exactly once after the active block's LAST attention exchange.

        No default-group collective may follow until next block acquisition (or
        forward flush). A host-visible attention fence plus the ready fence at
        acquisition avoids concurrent cross-group launches, including delayed ranks.
        """
        if not self.prefetch:
            return
        if not self._in_forward or self.active is None or self._tail_enqueued:
            raise RuntimeError("prefetch tail requires one active, unscheduled block")
        self._tail_enqueued = True
        if self.active + 1 == len(self.blocks):
            return
        with nvtx_range(f"prefetch.attention_fence.block.{self.active}"):
            attention_done = torch.cuda.Event()
            attention_done.record(torch.cuda.current_stream(self.device))
            attention_done.synchronize()
        self._enqueue(self.active + 1)

    def _drain(self, bundle):
        if bundle is None:
            return
        # A failed enqueue may not have recorded readiness. Preserve every input,
        # output and Work until the dedicated stream is actually drained.
        with torch.cuda.stream(self.prefetch_stream):
            for work in bundle.works:
                work.wait()
        if bundle.ready is not None:
            bundle.ready.synchronize()
        else:
            self.prefetch_stream.synchronize()
        if bundle.consumed is not None:
            bundle.consumed.synchronize()
        elif bundle.consumer_stream is not None:
            bundle.consumer_stream.synchronize()

    def flush(self):
        """No new collectives. Drain or fail terminally for supervisor cleanup."""
        if not self.prefetch:
            return
        if self.active is not None:
            raise RuntimeError("cannot flush during block execution")
        with nvtx_range("prefetch.flush"):
            try:
                self._drain(self._active_gather)
                self._drain(self.pending)
            except BaseException:
                self._failed = True
                raise
            self._active_gather = None
            self.pending = None

    @contextmanager
    def _prefetched_block(self, index):
        if not self._in_forward or self._failed or index != self._next_index:
            raise RuntimeError("prefetch blocks require ordered forward scope")
        if self.pending is None:
            if index != 0:
                raise RuntimeError("previous block did not enqueue at attention tail")
            # Root compute/startup on the consumer must finish before changing PG.
            torch.cuda.current_stream(self.device).synchronize()
            self._enqueue(index)
        if self.pending.index != index:
            raise RuntimeError("pending block does not match consumer")
        bundle = self.pending
        consumer = torch.cuda.current_stream(self.device)
        with nvtx_range(f"prefetch.wait.block.{index}"):
            # CPU completion also prevents launching the next attention communicator
            # before ALL dtype collectives have completed on the weight communicator.
            try:
                bundle.ready.synchronize()
                consumer.wait_event(bundle.ready)
            except BaseException:
                self._failed = True
                raise
        self.pending = None
        self._active_gather = bundle
        bundle.consumer_stream = consumer
        self.active = index
        self._tail_enqueued = False
        try:
            for group, flat in zip(self.groups[index], bundle.buffers):
                offset = 0
                for entry in group.entries:
                    entry.bind(flat[offset:offset + entry.numel].view(entry.shape))
                    offset += entry.numel
            yield
        finally:
            # Single consumer stream is the API contract. Retain ownership until
            # the last queued consumer actually finishes, including exceptional exit.
            completed = False
            try:
                consumed = torch.cuda.Event()
                consumed.record(consumer)
                bundle.consumed = consumed
                consumed.synchronize()
                completed = True
            except BaseException:
                self._failed = True
                # Event creation/record can fail before a valid fence exists.
                # A successful stream drain is the only permitted fallback.
                consumer.synchronize()
                completed = True
                raise
            finally:
                for entry in self.blocks[index]:
                    entry.dematerialize()
                self.active = None
                if completed:
                    self._active_gather = None
                self._next_index += 1

    @contextmanager
    def block(self, index):
        if torch.is_grad_enabled() or self.model.training:
            raise RuntimeError("resident sharding is eval/no-grad only")
        if not self.initialized or self.released or self.active is not None:
            raise RuntimeError("uninitialized, released or concurrent block gather")
        if not 0 <= index < len(self.blocks):
            raise IndexError(index)
        if self.prefetch:
            with self._prefetched_block(index):
                yield
            return
        self.active = index
        gathered = []
        try:
            with nvtx_range(f"weightgather.block.{index}"):
                event = self.runtime.profile_start()
                for group in self.groups[index]:
                    flat = torch.empty(group.shard_numel * self.world_size,
                                       dtype=group.dtype, device=self.device)
                    with nvtx_range("weightgather.transport.allgather"):
                        dist.all_gather_into_tensor(flat, group.shard)
                    gathered.append(flat)
                    with nvtx_range("weightgather.bind"):
                        offset = 0
                        for entry in group.entries:
                            entry.bind(flat[offset:offset + entry.numel].view(entry.shape))
                            offset += entry.numel
                self.runtime.profile_end("weight_allgather", event)
                self.gather_count += 1
            yield
        finally:
            with nvtx_range(f"release.block.{index}"):
                for entry in self.blocks[index]:
                    entry.dematerialize()
                gathered.clear()
                self.active = None

    def finish_request(self):
        """Drain transient work without destroying immutable resident resources."""
        if self.released or not self.initialized or self.active is not None or self._in_forward:
            raise RuntimeError("request finish requires an idle initialized controller")
        self.flush()
        if self._failed:
            raise RuntimeError("failed weight transport poisons the model owner")

    @nvtx_annotate('release.weights')
    def release(self):
        """Drop every root weight and shard before any decoder construction."""
        if self.active is not None:
            raise RuntimeError("cannot release during block execution")
        self.flush()
        if self.weight_group is not None:
            dist.destroy_process_group(self.weight_group)
            self.weight_group = None
            self.prefetch_stream = None
        for entry in self.entries:
            entry.dematerialize()
        for groups in self.groups:
            for group in groups:
                group.shard = None
        self.released = True
        self.initialized = False
