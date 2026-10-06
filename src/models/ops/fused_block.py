"""Inference-only kernels for the pointwise parts of a DiT block: the two pre-norms,
the two AdaLN affines, the two gated residual adds, and the feed-forward's SwiGLU
activation.

WHY THIS FILE EXISTS. Every one of those steps is pure elementwise work on the
[~105k, 5376] residual stream -- over 1 GiB per bf16 copy. Written the obvious way they
run as seven separate kernels that between them move several times more memory than
the arithmetic needs. Nothing here is clever: it is the same expressions, handed to
inductor in one piece so the intermediates stay in registers -- one read and one write
per stage.

The AdaLN affine is the worst of them: `scale.index_select(0, adaln_indices)` and
`shift.index_select(0, adaln_indices)` each materialise a full-size gathered copy of a
table with a handful of distinct rows, purely so a broadcast can consume it. Fused, the
gather happens per element in the kernel and never reaches memory.

INFERENCE ONLY, and installed by `hybrid_transform.set_inference_mode`. Two reasons it is
not simply the block's forward:

  * `dynamic=False`. The whole point is that inductor gets to specialise on the real
    shape; a training run varies caption length every step and would recompile.
  * NOT BITWISE. Inductor holds the RMSNorm reciprocal and the affine chain in fp32 and
    rounds once at the store, where eager rounds after every op. Same direction and the
    same size as the fused QK-norm+rope (see _qk_prep) -- an improvement in accuracy,
    but a difference, and a training step must not silently take it.

The eager bodies stay here next to the fused ones as the reference, so what is being
claimed equal is visible in one screen.
"""

import torch
import torch.nn.functional as F

from src.inference.utils.nsight import nvtx_annotate, nvtx_range
from src.models.ops.fp8_linear import Fp8Linear, swiglu_quantize_activation
from src.models.ops.mxfp8_linear import MXFP8Linear


# --------------------------------------------------------------------------------------
# The bodies. `_ref` is what diffusers' MiniMaxH3TransformerBlock.forward does inline.
# --------------------------------------------------------------------------------------

def _pre_ref(hidden, weight, eps, scale, shift, indices):
    """RMSNorm, then the AdaLN affine: x_hat * (1 + scale[i]) + shift[i]."""
    normed = F.rms_norm(hidden, (hidden.shape[-1],), weight, eps)
    return normed * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def _post_ref(residual, gate, indices, branch_out):
    """The gated residual add: residual + gate[i] * branch_out."""
    return residual + gate.index_select(0, indices) * branch_out


_CACHE = {}


def _compiled(name, body, *, deterministic=False):
    """One compiled callable per body, built on first use.

    Separate entries rather than one compile of the whole block: the two halves are
    called with different tensors around an attention that is itself several compiled
    regions, and keeping the graphs small keeps a recompile cheap and its cause obvious.
    """
    # Reduction autotuning can choose different floating-point trees in fresh
    # processes. MXFP8 amplifies even one changed BF16 rounding into trajectory
    # drift. Keep this policy local to paper callers and separate their cache
    # from the legacy fused path; never mutate global Inductor/Torch settings.
    key = ("paper_deterministic", name) if deterministic else name
    if key not in _CACHE:
        options = {"deterministic": True} if deterministic else None
        _CACHE[key] = torch.compile(body, dynamic=False, options=options)
    return _CACHE[key]


def _swiglu_ref(h):
    """diffusers SwiGLU.forward after its projection: a * silu(gate)."""
    a, gate = h.chunk(2, dim=-1)
    return a * F.silu(gate)


@nvtx_annotate('FFN.compute')
def fast_ff_forward(self, hidden_states):
    """Drop-in for diffusers' FeedForward.forward ([SwiGLU, Dropout, Linear]) under
    inference: the projection GEMM, the activation as ONE kernel, the down GEMM. Eager,
    `chunk -> silu -> mul` over the [T, 28672] projection output runs well below
    bandwidth; compiled it sits at the bandwidth floor. With the down projection in fp8
    the activation is fused with its quantisation instead, so the bf16 [T, 14336]
    intermediate is never written. Same non-bitwise caveat as everything else in this
    file."""
    swiglu, _, down = self.net
    shape = hidden_states.shape
    with nvtx_range("FFN.projection.up"):
        h = swiglu.proj(hidden_states.reshape(-1, shape[-1]))

    with nvtx_range("FFN.activation_projection.down"):
        if isinstance(down, Fp8Linear):
            out = down.forward_quantized(*swiglu_quantize_activation(h), out_dtype=h.dtype)
        elif isinstance(down, MXFP8Linear):
            # Native block scaling owns activation preparation, not the unrelated
            # per-channel Fp8Linear quantizer or a nonexistent floating weight.
            out = down(_compiled("swiglu", _swiglu_ref, deterministic=True)(h))
        else:
            out = down(_compiled("swiglu", _swiglu_ref)(h))

    return out.reshape(*shape[:-1], -1)


def fast_block_forward(self, hidden_states, temb, adaln_indices, rotary_emb,
                       attention_mask=None, *, adaln_tables=None,
                       deterministic_compile=False):
    """Drop-in for MiniMaxH3TransformerBlock.forward under inference.

    Line for line the same as the original; only the two pointwise stretches around each
    sub-layer are handed to one kernel instead of four.
    """
    with nvtx_range("projection.adaln"):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_proj(temb) if adaln_tables is None else adaln_tables)
    pre = _compiled("pre", _pre_ref, deterministic=deterministic_compile)
    post = _compiled("post", _post_ref, deterministic=deterministic_compile)

    normed = pre(hidden_states, self.norm1.weight, self.norm1.eps,
                 scale_msa, shift_msa, adaln_indices)
    hidden_states = post(hidden_states, gate_msa, adaln_indices,
                         self.attn(normed, rotary_emb, attention_mask))

    with nvtx_range("FFN"):
        normed = pre(hidden_states, self.norm2.weight, self.norm2.eps,
                     scale_mlp, shift_mlp, adaln_indices)
        hidden_states = post(hidden_states, gate_mlp, adaln_indices, self.ff(normed))
    return hidden_states


def paper_block_forward(block, hidden, temb, indices, rotary, *, tables=None, fused=False):
    """Explicit checked-table seam; no monkeypatched AdaLN or current-step cache."""
    paper = getattr(block.attn, "_paper_runtime", None)
    if paper is not None and getattr(paper, "fused_producers", False):
        from src.models.ops.mxfp8_kernels import launch_prenorm_quantize
        if not getattr(block, "_tp_sharded", False) or tables is None or not fused:
            raise ValueError("fused MXFP8 producers require TP with prepared AdaLN")
        a_shift, a_scale, a_gate, f_shift, f_scale, f_gate = tables
        post = _compiled("post", _post_ref, deterministic=True)
        with nvtx_range("tp.producer.prenorm_mxfp8.dual"):
            normed, data, scales, finite = launch_prenorm_quantize(
                hidden.reshape(-1, hidden.shape[-1]), block.norm1.weight, block.norm1.eps,
                a_scale, a_shift, indices, write_bf16=True)
        paper.guard.predicate(finite)
        hidden = post(hidden, a_gate, indices,
                      block.attn(normed.unsqueeze(0), rotary, None, prepared=(data, scales)))
        del normed, data, scales, finite
        with nvtx_range("tp.producer.prenorm_mxfp8.only"):
            _, data, scales, finite = launch_prenorm_quantize(
                hidden.reshape(-1, hidden.shape[-1]), block.norm2.weight, block.norm2.eps,
                f_scale, f_shift, indices, write_bf16=False)
        paper.guard.predicate(finite)
        out = block.ff(None, prepared=(data, scales, hidden.numel() // hidden.shape[-1]))
        return post(hidden, f_gate, indices, out.reshape_as(hidden))
    if fused:
        return fast_block_forward(block, hidden, temb, indices, rotary,
                                  adaln_tables=tables, deterministic_compile=True)
    with nvtx_range("paper.adaln.block_lookup"):
        a_shift, a_scale, a_gate, f_shift, f_scale, f_gate = (
            block.adaln_proj(temb) if tables is None else tables)
    normed = block.norm1(hidden)
    normed = normed * (1.0 + a_scale.index_select(0, indices)) + a_shift.index_select(0, indices)
    hidden = hidden + a_gate.index_select(0, indices) * block.attn(normed, rotary, None)
    normed = block.norm2(hidden)
    normed = normed * (1.0 + f_scale.index_select(0, indices)) + f_shift.index_select(0, indices)
    return hidden + f_gate.index_select(0, indices) * block.ff(normed)
