"""Explicit default-off permanent head/MLP ownership for persistent inference.

Residual rows are replicated. Attention and MLP each reduce one residual-width
partial result. Wide weights stay compact MXFP8, including repacked K scales.
No gathered backing storage or floating shadow is retained by the local block.
"""
from __future__ import annotations

from contextlib import nullcontext
import types

import torch
from torch import nn
import torch.distributed as dist

from src.inference.utils.nsight import nvtx_range
from src.models.ops.mxfp8_linear import MXFP8Linear, project_qkv
from src.models.ops.mxfp8_quant import padded_shape, swizzle_scales


def unswizzle_scales(scales, rows, k):
    """Inverse of canonical 128-row by four-scale-column packing."""
    rp, kp = padded_shape(rows, k)
    return (scales.reshape(-1, 32, 4, 4).transpose(1, 2)
            .reshape(rp // 128, kp // 128, 128, 4).permute(0, 2, 1, 3)
            .reshape(rp, kp // 32).contiguous())


def shard_linear(module, *, rows=None, columns=None, row_parallel=False):
    """Mutate a CPU/meta linear into independent compact storage, never views.

    Row parallel stores bias separately for addition ONCE after reduction. K
    cuts must align with block32 quantization, not with encoded scale offsets.
    Rows can select matched nonadjacent value/gate halves of SwiGLU.
    """
    if not isinstance(module, (nn.Linear, MXFP8Linear)):
        raise TypeError("TP requires ordinary or native MXFP8 linear")
    n, k = module.out_features, module.in_features
    rows = list(range(n)) if rows is None else list(rows)
    start, end = (0, k) if columns is None else columns
    if not rows or len(set(rows)) != len(rows) or min(rows) < 0 or max(rows) >= n:
        raise ValueError("invalid TP output rows")
    if not 0 <= start < end <= k:
        raise ValueError("invalid TP input columns")
    if isinstance(module, MXFP8Linear):
        if start % 32 or end % 32:
            raise ValueError("MXFP8 K cuts must be block32 aligned")
        source = module.weight_data
        idx = torch.tensor(rows, device=source.device, dtype=torch.long)
        rp, kp = padded_shape(len(rows), end-start)
        data = torch.zeros((rp, kp), device=source.device, dtype=torch.uint8)
        data[:len(rows), :end-start] = source.index_select(0, idx)[:, start:end]
        natural = unswizzle_scales(module.weight_scale, n, k)
        scales = torch.zeros((rp, kp//32), device=source.device, dtype=torch.uint8)
        scales[:len(rows), :(end-start)//32] = natural.index_select(0, idx)[:, start//32:end//32]
        module.weight_data = data
        module.weight_scale = swizzle_scales(scales)
    else:
        idx = torch.tensor(rows, device=module.weight.device, dtype=torch.long)
        module.weight = nn.Parameter(module.weight.index_select(0, idx)[:, start:end].clone().contiguous(),
                                     requires_grad=False)
    if module.bias is not None:
        idx = torch.tensor(rows, device=module.bias.device, dtype=torch.long)
        module.bias = nn.Parameter(module.bias.index_select(0, idx).clone(), requires_grad=False)
    if row_parallel:
        module.register_parameter("tp_bias", module.bias)
        module.bias = None
    module.out_features, module.in_features = len(rows), end-start
    return module


def shard_block(block, rank, world_size):
    """Convert one unwrapped full CPU/meta block into permanently owned fields.

    Call before runtime wrappers and before device distribution. AdaLN source is
    deliberately untouched here: owner removes it from GPU inventory and uses
    the existing per-request tables. Refuse aliases rather than silently untie.
    """
    if getattr(block, "_tp_sharded", False):
        raise ValueError("block already TP sharded")
    if not 0 <= rank < world_size:
        raise ValueError("invalid rank")
    tensors = list(block.named_parameters(remove_duplicate=False)) + list(block.named_buffers(remove_duplicate=False))
    seen = set()
    for name, t in tensors:
        if id(t) in seen:
            raise ValueError(f"TP selected block aliases unsupported: {name}")
        seen.add(id(t))
        if t.device.type not in ("cpu", "meta"):
            raise ValueError("TP slicing must precede GPU placement")
    attn, branch = block.attn, block.attn.linear_attention
    h, d = attn.num_heads, attn.head_dim
    ff = block.ff.net
    width = ff[2].in_features
    if h % world_size or width % world_size or ff[0].proj.out_features != width*2:
        raise ValueError("TP requires divisible heads and matched SwiGLU halves")
    local_h, local_ff = h//world_size, width//world_size
    hs, he = rank*local_h, (rank+1)*local_h
    cs, ce = hs*d, he*d
    fs, fe = rank*local_ff, (rank+1)*local_ff
    for name in ("to_q", "to_k", "to_v"):
        shard_linear(getattr(attn.orig, name), rows=range(cs, ce))
    shard_linear(attn.orig.to_out[0], columns=(cs, ce), row_parallel=True)
    shard_linear(attn.to_out_linear, columns=(cs, ce), row_parallel=True)
    shard_linear(ff[0].proj, rows=[*range(fs, fe), *range(width+fs, width+fe)])
    shard_linear(ff[2], columns=(fs, fe), row_parallel=True)
    shard_linear(attn.softmax_gate.up, rows=range(hs, he))
    attn.softmax_gate.num_heads = local_h
    shard_linear(branch.beta_proj, rows=range(hs, he))
    shard_linear(branch.output_gate.up, rows=range(cs, ce))
    branch.output_gate.num_heads = local_h
    shard_linear(branch.alpha.up, rows=range(cs, ce))
    branch.alpha.A_log = nn.Parameter(branch.alpha.A_log[hs:he].clone(), requires_grad=False)
    branch.alpha.dt_bias = nn.Parameter(branch.alpha.dt_bias[cs:ce].clone(), requires_grad=False)
    branch.alpha.num_heads = branch.num_heads = local_h
    if branch.short_conv is not None:
        for proj in branch.short_conv.projs:
            for suffix in ("sp", "tm"):
                conv = getattr(branch.short_conv, f"{proj}_{suffix}")
                conv.weight = nn.Parameter(conv.weight[cs:ce].clone(), requires_grad=False)
                if conv.bias is not None:
                    conv.bias = nn.Parameter(conv.bias[cs:ce].clone(), requires_grad=False)
                conv.in_channels = conv.out_channels = conv.groups = local_h*d
    block._tp_sharded = True
    attn._tp_heads = local_h
    return block


def reduce_partial(value, runtime, label, guard=None):
    if guard is not None:
        guard.check(label, value)
    with nvtx_range("tp.all_reduce." + label):
        if runtime.world_size > 1:
            dist.all_reduce(value, group=getattr(runtime, "tp_group", None))
    return value


def add_bias(value, module):
    bias = getattr(module, "tp_bias", None)
    return value if bias is None else value + bias.to(value.dtype)


def tp_ff_forward(self, hidden, *, prepared=None):
    from src.models.ops.fused_block import _compiled, _swiglu_ref
    up, _, down = self.net
    with nvtx_range("tp.ffn"):
        paper = getattr(self, "_paper_runtime", None)
        phase = paper.guard.local_phase("tp_ffn_compute") if paper is not None else nullcontext()
        with phase:
            if prepared is None:
                h = up.proj(hidden)
            else:
                data, scales, rows = prepared
                h = up.proj.forward_quantized(data, scales, rows=rows)
            if paper is not None and getattr(paper, "fused_producers", False):
                from src.models.ops.mxfp8_kernels import launch_swiglu_quantize
                with nvtx_range("tp.producer.swiglu_mxfp8"):
                    data, scales, finite = launch_swiglu_quantize(h.reshape(-1, h.shape[-1]))
                paper.guard.predicate(finite)
                out = down.forward_quantized(data, scales, rows=h.numel() // h.shape[-1])
                out = out.reshape(*h.shape[:-1], -1)
            else:
                h = _compiled("swiglu", _swiglu_ref, deterministic=True)(h)
                out = down(h)
        out = reduce_partial(out, self._tp_runtime, "ffn", None if paper is None else paper.guard)
        return add_bias(out, down)


def tp_attention_forward(self, hidden_states, rotary_emb=None, attention_mask=None, *, prepared=None):
    """Full sequence, local raw QKV and gates; one merged output reduction.

    Merging BF16 partial softmax/VDA projections before SUM changes rounding
    versus two separate full-K GEMMs. FP32 algebra is unchanged. Non-video rows
    never receive VDA output or its bias. Anchor rows retain branch zeroing.
    """
    from src.inference.utils.ulysses import _linear_branch_shard
    from src.inference.utils.vda_overlap import vda_on_side_stream
    from src.models.softmax_attention.paper_qk_rope import paper_qk_rmsnorm_rope
    from src.models.softmax_attention.paper_window import paper_window_attention
    from src.models.softmax_attention.kernels import apply_softmax_gate
    paper, runtime = self._paper_runtime, self._tp_runtime
    guard, layout = paper.guard, self.layout
    x = hidden_states[0]
    d, heads = self.head_dim, self._tp_heads
    with guard.local_phase("tp_projection"):
        if attention_mask is not None or hidden_states.shape[0] != 1 or self.teacher_mode:
            raise ValueError("TP attention requires batch1, no mask, teacher off")
        if x.shape[0] != layout.seq_len:
            raise ValueError("TP attention needs complete packed rows")
        projections = (self.orig.to_q, self.orig.to_k, self.orig.to_v)
        qkv = (project_qkv(projections, x) if prepared is None else
               tuple(p.forward_quantized(*prepared, rows=x.shape[0]) for p in projections))
        raw_q, raw_k, value = (t.unflatten(-1, (heads, d)) for t in qkv)
        gate = self.softmax_gate(x) if self.enable_softmax_gate else x.new_ones(x.shape[0], heads, 1)
        beta = torch.sigmoid(self.linear_attention.beta_proj(x))
        linear_gate = self.linear_attention.output_gate(x)
        video = slice(layout.video_start, layout.video_end)
        frame_mean = x[video].view(layout.num_frames, layout.tokens_per_frame, -1).float().mean(1)
        guard.observe(raw_q, raw_k, value, gate, beta, linear_gate, frame_mean)
    bounds = self._bounds(layout)
    active = self.linear_attention_enabled and not all(lo <= 0 and hi >= layout.num_frames-1 for lo, hi in bounds)
    overlap = active and paper.options["paper_vda_overlap"]
    def linear():
        return _linear_branch_shard(self, (raw_q, raw_k, value), beta, linear_gate,
                                    frame_mean, 0, heads)
    work = (vda_on_side_stream(runtime, linear,
            inputs=(raw_q, raw_k, value, beta, linear_gate, frame_mean),
            module=self.linear_attention, paper_mode=True) if overlap else nullcontext(None))
    with guard.local_phase("tp_attention_compute"):
        with work as far:
            cos, sin = paper.rotary if paper.rotary is not None else (None, None)
            query, key = paper_qk_rmsnorm_rope(raw_q, raw_k, self.orig.norm_q.weight,
                    self.orig.norm_k.weight, self.orig.norm_q.eps, cos, sin, enabled=True)
            soft = paper_window_attention(query.contiguous(), key.contiguous(), value.contiguous(),
                                         paper.window_plan(self), d**-0.5, enabled=True)
            soft = apply_softmax_gate(soft, gate, inference=False)
            if active and not overlap:
                far = linear()
        out = self.orig.to_out[0](soft.reshape(x.shape[0], -1).type_as(x))
        if active:
            out[video] += self.to_out_linear(far[video].reshape(layout.video_end-layout.video_start, -1).type_as(x))
        guard.observe(out)
    out = reduce_partial(out, runtime, "attention", guard)
    out = add_bias(out, self.orig.to_out[0])
    out = self.orig.to_out[1](out)
    if active and getattr(self.to_out_linear, "tp_bias", None) is not None:
        out[video] += self.to_out_linear.tp_bias.to(out.dtype)
    return out.unsqueeze(0)


def install_tp_block(block, runtime, paper):
    if not getattr(block, "_tp_sharded", False):
        raise ValueError("TP execution requires permanent compact ownership")
    block.attn._tp_runtime = runtime
    block.attn._paper_runtime = paper
    block.attn.forward = types.MethodType(tp_attention_forward, block.attn)
    block.ff._tp_runtime = runtime
    block.ff._paper_runtime = paper
    block.ff.forward = types.MethodType(tp_ff_forward, block.ff)


class PermanentTensorShards:
    """Single-owner CPU assembly, bounded startup transport, permanent GPU tensors.

    Implements the persistent controller protocol without a gather communicator.
    CPU block replicas are temporary, one at a time; rank zero alone assembles
    weights. Distinct storage aliases are rejected before any GPU allocation.
    """
    def __init__(self, model, runtime, *, scatter_chunk_bytes=32 << 20,
                 distribution="broadcast", prefetch=False):
        from src.inference.utils.weight_sharding import tensor_slots
        self.model, self.runtime = model, runtime
        self.entries = tensor_slots(model)
        self.device = runtime.device
        self.chunk_bytes = scatter_chunk_bytes
        self.prefetch = False
        self.weight_group = None
        self.initialized = self.released = self._failed = self._in_forward = False
        self.active = None
        self.gather_count = 0
        self.root_bytes = self.block_bytes = 0
        self.owned_inventory = []
        seen = set()
        for entry in self.entries:
            if len(entry.names) != 1:
                raise ValueError("TP aliases must be explicitly implemented, not silently untied")
            t = entry.tensor()
            if not t.is_meta and t.numel():
                key = (str(t.device), t.untyped_storage().data_ptr())
                if key in seen:
                    raise ValueError("TP distinct storage views unsupported")
                seen.add(key)

    def manifest(self):
        return [(e.names, e.shape, str(e.dtype), e.parameter) for e in self.entries]

    def summary(self):
        return dict(policy="permanent_tp8", root_bytes=self.root_bytes,
                    resident_shard_bytes=self.block_bytes, max_gather_bytes=0,
                    blocks=len(self.model.transformer_blocks), gather_count=0,
                    assembly="rank0_strict_full_cpu_then_compact_tensor_slices",
                    prefetch=False, max_live_gathered=0, distribution="startup_broadcast_compact",
                    owned_inventory=self.owned_inventory)

    @torch.no_grad()
    def distribute(self):
        import copy
        import hashlib
        import json
        from src.inference.utils.weight_sharding import tensor_slots
        if self.initialized or self.released:
            raise RuntimeError("TP distribution is single-use")
        if self.runtime.world_size != 8 or len(self.model.transformer_blocks) != 50:
            raise ValueError("permanent production TP requires eight ranks and fifty blocks")
        paper = self.runtime.paper_runtime
        if not getattr(paper, "persistent", False) or not all(paper.options[k] for k in
                ("paper_mxfp8", "paper_qk_rope", "paper_window", "paper_vda", "paper_adaln", "paper_fused_block")):
            raise ValueError("permanent TP requires the complete persistent paper stack")
        rank = self.runtime.rank
        digest = hashlib.sha256(json.dumps(self.manifest(),sort_keys=True).encode()).hexdigest()
        digests = [None]*8
        dist.all_gather_object(digests,digest,group=paper.group)
        if len(set(digests)) != 1:
            raise ValueError("TP full source manifests differ")
        # One reusable bounded buffer per dtype; never full GPU source matrices.
        buffers = {}
        def transport(entry, source, destination):
            value = torch.empty(entry.shape,dtype=entry.dtype,device=self.device) if destination in (None,rank) else None
            elements = max(1,self.chunk_bytes//torch.empty((),dtype=entry.dtype).element_size())
            buffer = buffers.get(entry.dtype)
            if buffer is None:
                buffer = torch.empty(elements,dtype=entry.dtype,device=self.device)
                buffers[entry.dtype] = buffer
            for start in range(0,entry.numel,elements):
                length = min(elements,entry.numel-start)
                part = buffer[:length]
                if rank == 0:
                    part.copy_(source.view(-1)[start:start+length])
                dist.broadcast(part,src=0)
                if value is not None:
                    value.view(-1)[start:start+length].copy_(part)
            if value is not None:
                entry.bind(value)
        with paper.guard.local_phase("tp_source_validation"):
            if rank == 0 and any(e.tensor().device.type != "cpu" for e in self.entries):
                raise ValueError("TP source must be CPU-only")
        for entry in self.entries:
            if not entry.names[0].startswith("transformer_blocks."):
                transport(entry,entry.tensor() if rank == 0 else None,None)
                self.root_bytes += entry.numel*entry.tensor().element_size()
        for index, original in enumerate(self.model.transformer_blocks):
            follower = None
            if rank != 0:
                follower = shard_block(original,rank,8)
            selected = None
            for destination in range(8):
                local = None
                with paper.guard.local_phase("tp_slice_block"):
                    if rank == 0:
                        # Avoid copying the unused 520MB AdaLN projection. A fresh
                        # meta copy keeps graph metadata while sources stay CPU.
                        memo = {id(t): nn.Parameter(torch.empty_like(t,device="meta"),requires_grad=False)
                                for t in original.adaln_proj.parameters()}
                        memo.update({id(t): torch.empty_like(t,device="meta")
                                     for t in original.adaln_proj.buffers()})
                        adaln = copy.deepcopy(original.adaln_proj,memo=memo)
                        local = copy.deepcopy(original,memo={id(original.adaln_proj):adaln})
                        shard_block(local,destination,8)
                    elif rank == destination:
                        local = follower
                # All ranks use identical compact metadata from a meta local block.
                if local is None:
                    # Peers reuse their own equal-shaped local metadata; its data
                    # is never read for other destinations.
                    local = follower
                entries = [e for e in tensor_slots(local) if not e.names[0].startswith("adaln_proj.")]
                for entry in entries:
                    transport(entry,entry.tensor() if rank == 0 else None,destination)
                if rank == destination:
                    selected = local
                if rank == 0 and destination != 0:
                    del local
            self.model.transformer_blocks[index] = selected
            # Record exact independent allocations, not gathered views.
            for name,t in list(selected.named_parameters())+list(selected.named_buffers()):
                if t.is_meta:
                    if not name.startswith("adaln_proj."):
                        raise RuntimeError("unexpected meta TP field")
                    continue
                size = t.numel()*t.element_size()
                if t.storage_offset() or t.untyped_storage().nbytes() != size:
                    raise RuntimeError("TP tensor is a view into larger storage")
                self.block_bytes += size
                self.owned_inventory.append(dict(name=f"transformer_blocks.{index}.{name}",
                    shape=list(t.shape),dtype=str(t.dtype),bytes=size,storage_bytes=t.untyped_storage().nbytes()))
        buffers.clear()
        self.entries = tensor_slots(self.model)
        self.initialized = True

    def prefetch_next(self):
        raise RuntimeError("permanent TP must never prefetch block weights")

    def forward(self):
        from contextlib import contextmanager
        @contextmanager
        def scope():
            if not self.initialized or self.released or self._failed or self._in_forward:
                raise RuntimeError("invalid TP forward lifecycle")
            self._in_forward = True
            try:
                yield
            except BaseException:
                self._failed = True
                raise
            finally:
                self._in_forward = False
        return scope()

    def block(self,index):
        from contextlib import contextmanager
        @contextmanager
        def scope():
            if not self._in_forward or self.active is not None or self._failed:
                raise RuntimeError("invalid TP block lifecycle")
            self.active = index
            try:
                yield
            finally:
                self.active = None
        return scope()

    def finish_request(self):
        if self.active is not None or self._in_forward or self.released or self._failed:
            raise RuntimeError("TP request did not drain cleanly")
        torch.cuda.synchronize(self.device)

    def release(self):
        if self.released:
            return
        if self.active is not None or self._in_forward:
            raise RuntimeError("cannot release active TP weights")
        torch.cuda.synchronize(self.device)
        from src.inference.utils.weight_sharding import tensor_slots
        for entry in tensor_slots(self.model):
            entry.dematerialize()
        self.entries.clear()
        self.released = True
