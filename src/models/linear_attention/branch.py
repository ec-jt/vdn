"""BidirectionalLinearBranch: everything the softmax window cannot see,
summarised per query token.
"""
import torch
import torch.nn.functional as F
from torch import nn

from src.models.attention_gates import OutputGate
from src.models.linear_attention.delta_rule import DELTA_BACKENDS
from src.checkpoints.key_mapping import SHORT_CONV_TARGETS
from src.models.linear_attention.features import (LinearAttentionSepConv,
                                                  _activate_body, _temporal_conv,
                                                  prepare_linear_features,
                                                  prepare_linear_features_inference)
from src.models.linear_attention.kernels import linear_epilogue, vda_bf16_epilogue
from src.models.linear_attention.layers import FrameKDAAlpha, RMSNorm
from src.models.linear_attention.scan import (BRIDGE_MODES, frame_statistics,
                                              gather_linear_state, _run_scans,
                                              _run_scans_inference)

class _HeadSliceSepConv:
    """A head range of the branch's separable conv, for either body under Ulysses:
    the same weights read through a channel slice, so `prepare_linear_features_inference`
    sees an object with the conv's interface (`projs`, `KERNEL`, `spatial`) and nothing
    else changes. The module itself is untouched."""

    def __init__(self, conv, heads, head_dim):
        self._conv = conv
        self.projs = conv.projs
        self.KERNEL = conv.KERNEL
        self.channels = slice(heads.start * head_dim, heads.stop * head_dim)

    def spatial(self, proj, tokens, num_frames, frame_size):
        n_heads, head_dim = tokens.shape[-2:]
        grid_h, grid_w = frame_size
        channels = n_heads * head_dim
        volume = (tokens.reshape(num_frames, grid_h, grid_w, channels)
                  .permute(0, 3, 1, 2))
        weight_spatial = getattr(self._conv, f"{proj}_sp").weight[self.channels]
        volume = F.conv2d(volume, weight_spatial, padding=self.KERNEL // 2, groups=channels)
        x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        weight_temporal = (getattr(self._conv, f"{proj}_tm").weight[self.channels]
                           .squeeze(1).to(x.dtype))
        return x, weight_temporal


class BidirectionalLinearBranch(nn.Module):
    """NoPE linear-attention branch. Consumes video tokens grouped per frame, produces
    the linear readout for every video token (globals get zeros)."""

    # Each directional scan starts from TEXT_STATE_SCALE * S_text (see forward's docstring
    # for why a half). Not a hyperparameter and not per-instance: it is baked into every
    # trained checkpoint, so changing it retrains, it does not reconfigure. Applied in
    # exactly one place, _text_state.
    TEXT_STATE_SCALE = 0.5

    def __init__(self, hidden_size, num_heads, head_dim, delta_rule="sana_scaled",
                 gate_bottleneck=None, bridge="alpha", a_fp32=True, short_conv=()):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        assert bridge in BRIDGE_MODES, bridge
        self.bridge = bridge                     # see gather_linear_state

        # False reproduces the pre-fix bf16 A; kept so a checkpoint can be evaluated at
        # the precision it was TRAINED at. New runs always train with True.
        self.a_fp32 = a_fp32

        # short_conv: the projections the depthwise (5x5 spatial x 5-tap temporal) conv
        # runs on -- a subset of ("q", "k", "v"), empty for none. None rather than a
        # disabled module when off, so conv-less checkpoints keep their exact parameter
        # set (loaders refuse extra/missing keys).
        targets = tuple(short_conv) if isinstance(short_conv, (list, tuple)) else None
        if (targets is None or any(t not in SHORT_CONV_TARGETS for t in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError(f"short_conv={short_conv!r}; expected a distinct subset of "
                             f"{SHORT_CONV_TARGETS} (the projections to convolve)")
        self.short_conv = (LinearAttentionSepConv(num_heads * head_dim, targets)
                           if targets else None)
        self.alpha = FrameKDAAlpha(hidden_size, num_heads, head_dim)
        # fla's b_proj: per-head beta, no bias, live weights -> beta centred on 0.5.
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # fla's g_proj shape: rank = head_dim, live init (see OutputGate).
        self.output_gate = OutputGate(hidden_size, num_heads, head_dim,
                               bottleneck=gate_bottleneck or head_dim, init="random")
        self.norm = RMSNorm(head_dim)         # fla-aligned weighted norm, see the class
        # `delta_rule` NAMES the rule (a DELTA_BACKENDS key, what the ModelSpec stores);
        # `backend` / `text_backend` are the rule's instances, built lazily because they
        # are sized by the chunk length (S for frames, L_text for the prompt).
        self.delta_rule = delta_rule
        self.backend = None                                               # built lazily (needs S)
        self.text_backend = None       # same class, scaled by L_text (see _text_state)

        # Recompute _features in backward instead of saving its activations. OFF by
        # default and OWNED BY THE TRAINER: Stage A2/B wrap each block in a gradient
        # checkpoint already (this region would be recomputed twice), but Stage A1 has
        # no block checkpointing, and with short_conv the saved conv+SiLU+L2Norm chain
        # adds a few GiB to every per-layer student re-run.
        self.checkpoint_features = False
        # Experimental, independent of the model-wide inference kernel switch.
        # Only boundary gather and BF16-rounded epilogue are fused. Statistics prep
        # was measured separately and rejected for production124 latency regression.
        # Features, FP32 alpha/inverse/scans and the text state stay on the reference.
        self.inference_vda_kernels = False
        # Independent full paper-stack experiment. Never inferred from a global
        # inference flag, and never silently falls back to the old branch body.
        self.paper_vda_enabled = False
        self.paper_vda_chunk_size = 8

    def _features(self, qkv_raw, num_frames=None, frame_size=None, inference=False,
                  query_fhsd=None, heads=None):
        """Linear-branch features: [ShortConv ->] SiLU + L2Norm(q,k) + NoPE.

        The branch always shares the softmax branch's QKV projections: it takes the raw
        pre-QK-norm, pre-RoPE q/k/v and applies its own post-processing, so it adds no
        projection cost — and under Stage-B LoRA it sees the adapted projections.

        `num_frames`/`frame_size` are consumed only when short_conv is on (the conv
        needs the (frames, height, width) volume); conv-less branches keep working with
        layouts that carry no spatial grid."""
        if self.short_conv is not None and frame_size is None:
            raise ValueError(
                "short_conv needs the spatial grid; pass frame_size=(H, W) "
                "(layouts built without frame_size cannot drive the 3D conv)"
            )
        outs = []
        for proj, tokens in zip(("q", "k", "v"), qkv_raw):
            if inference:
                outs.append(self._feature_one(
                    tokens, proj, num_frames, frame_size, inference=True,
                    fhsd=(query_fhsd if proj == "q" else None), heads=heads))
                continue
            if self.checkpoint_features and torch.is_grad_enabled():
                # One region PER PROJECTION, not one for all three: while a region's
                # backward runs, its whole recompute graph is alive, and a single one
                # holding q+k+v would keep all three projections' conv transposes and
                # their conv outputs live at once — ~3x the transient. Three regions
                # retire their transients one at a time.
                outs.append(torch.utils.checkpoint.checkpoint(
                    self._feature_one, tokens, proj, num_frames, frame_size,
                    use_reentrant=False))
            else:
                outs.append(self._feature_one(tokens, proj, num_frames, frame_size,
                                              heads=heads))
        return tuple(outs)

    def _feature_one(self, tokens, proj, num_frames, frame_size, use_conv=True,
                     inference=False, fhsd=None, heads=None):
        """One projection's linear-branch features: [ShortConv ->] SiLU [-> L2Norm for q/k].

        use_conv=False is the text chunk: the short conv is a (t, h, w) stencil over
        the video volume and the prompt has no such grid. Text keeps the rest of the
        pipeline — same SiLU, same L2Norm, same NoPE — so the delta rule it feeds is
        the one the video frames feed."""
        conv = self.short_conv if use_conv else None
        if conv is not None and heads is not None:
            conv = _HeadSliceSepConv(conv, heads, self.head_dim)
            if not inference and proj in conv.projs:
                # Head-sharded Ulysses must run the SAME short-conv kernel as the
                # single-rank reference. The reference reaches the conv through
                # prepare_linear_features -> _temporal_conv (inductor-compiled on
                # CUDA); an earlier version called eager _temporal_shift here, which
                # is numerically close but NOT identical (measured ~8.5e-3 rel L2
                # per branch call at bf16) and inflated trajectory drift. Both conv
                # weights stay head-sliced.
                x, weight = conv.spatial(proj, tokens, num_frames, frame_size)
                out = _temporal_conv(x, weight, conv.KERNEL, conv.KERNEL // 2)
                return _activate_body(out.reshape_as(tokens), l2norm=(proj != "v"))
        features = prepare_linear_features_inference if inference else prepare_linear_features
        return features(tokens, l2norm=(proj != "v"), conv=conv, proj=proj,
                        num_frames=num_frames, frame_size=frame_size, fhsd=fhsd)

    def _delta_backend(self, attr, length):
        """Lazily built, cached delta backend for a chunk of `length` rows.

        Keyed on (rule name, length), never on length alone: `delta_rule` can be
        re-pointed on a LIVE module (ablations, tools that switch checkpoints without
        rebuilding), and a length-only key would silently keep the previous rule's
        algorithm (sana_scaled's first-order truncation standing in for vdn_solve's
        exact inverse). Nothing would error and the render would look entirely normal.
        """
        cached = getattr(self, attr)
        key = (self.delta_rule, length)
        if cached is None or getattr(cached, "_key", None) != key:
            cached = DELTA_BACKENDS[self.delta_rule](length)
            cached._key = key
            cached._S = length
            setattr(self, attr, cached)
        return cached

    def _text_state(self, text_x, text_qkv_raw, heads=None, text_beta=None):
        """The state BOTH scans start from: TEXT_STATE_SCALE * S_text when the prompt
        rows were handed in, None (a zero start) when they were not. [H, d_v, d_k] fp32.
        `heads` / `text_beta` are the Ulysses inference case: this rank's head range,
        and the beta its sequence owner already computed (it has no `text_x`)."""
        if text_x is None and text_beta is None:
            return None
        return self.TEXT_STATE_SCALE * self._text_chunk_state(
            text_x, text_qkv_raw, heads=heads, text_beta=text_beta)

    def _text_chunk_state(self, text_x, text_qkv_raw, heads=None, text_beta=None):
        """S_text: the whole prompt written into a zero state as ONE delta-rule chunk.

            A_text = K^T diag(beta) K,  B_text = V^T diag(beta) K   over all L text rows
            S_text = 0 @ transition + injection(A_text, B_text)

        Same rule as a video frame, one difference: the backend's key scaling is
        c = 1/sqrt(L_text), not 1/sqrt(S). The scaling is what bounds trace(c^2 A) <= 1
        (L2-normed keys, sum over the chunk's rows), so it has to count the rows the
        chunk actually has — reusing the per-frame S would mis-scale a prompt that is
        much shorter than a frame and hand the scan a state of the wrong magnitude.

        No causal scan inside the text: a chunk update over all L rows at once is what
        the video path does per frame, and the text encoder + token refiner have already
        written word order into every text hidden state. alpha plays no part here (the
        old state is zero, so the transition multiplies nothing) — only the injection.

        Returns [H, d_v, d_k] fp32.
        """
        head_dim = self.head_dim
        n_heads = self.num_heads if heads is None else heads.stop - heads.start
        length = text_qkv_raw[1].shape[0]

        # No 3D conv on text (no grid), and beta reuses the video beta_proj: the same
        # projection reading the same hidden width, so the branch gains no parameters
        # and the text keys are weighted on the same scale as the video keys.
        key = self._feature_one(text_qkv_raw[1], "k", None, None, use_conv=False)
        value = self._feature_one(text_qkv_raw[2], "v", None, None, use_conv=False)
        key = key.view(1, length, n_heads, head_dim).permute(0, 2, 1, 3)   # [1,H,L,d]
        value = value.view(1, length, n_heads, head_dim).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(text_x)) if text_beta is None else text_beta
        beta = beta.view(1, length, n_heads).permute(0, 2, 1)              # [1,H,L]
        heads = n_heads
        A, B = frame_statistics(key, value, beta, a_fp32=self.a_fp32)      # [1,H,d,d]
        backend = self._delta_backend("text_backend", length)
        with torch.autocast(device_type=A.device.type, enabled=False):
            ones = torch.ones(1, heads, head_dim, device=A.device, dtype=A.dtype)
            _, injection = backend.factor_apply(ones, A, B)
        return injection[0]                                                # [H,d_v,d_k]

    def _validate_sharded(self, heads, beta, gate, frame_mean, text_beta, text_qkv_raw):
        """Owner-supplied inputs are a forward-only contract, not a training API."""
        if any(v is not None for v in (heads, beta, gate, frame_mean, text_beta)):
            if torch.is_grad_enabled():
                raise RuntimeError("head-sharded/precomputed branch inputs require "
                                   "torch.no_grad(); distributed training is not supported")
        if heads is not None:
            if (not isinstance(heads, slice) or heads.step not in (None, 1)
                    or not isinstance(heads.start, int) or not isinstance(heads.stop, int)
                    or not 0 <= heads.start < heads.stop <= self.num_heads):
                raise ValueError("heads must be a bounded contiguous slice within num_heads")
            if beta is None or gate is None or frame_mean is None:
                raise ValueError("head-sharded calls require beta, gate, and frame_mean")
            if text_qkv_raw is not None and text_beta is None:
                raise ValueError("head-sharded text_qkv_raw requires text_beta")

    def forward(self, xv, num_frames, tokens_per_frame, bounds, qkv_raw,
                frame_size=None, skip_ends=False, text_x=None, text_qkv_raw=None,
                inference=False, heads=None, beta=None, gate=None, frame_mean=None,
                text_beta=None):
        """Everything the softmax window CANNOT see, summarised for every video token.

            0. text chunk (optional)    the prompt -> S_text, the state both scans start
                                        from (see _text_state)
            1. per-token features       q, k, v      (SiLU, L2-normed q/k, no RoPE)
            2. per-frame summaries      A, B         S tokens -> two d x d matrices
            3. two scans over frames    forward/reverse state banks
            4. boundary gather          the state just outside the window, both sides,
                                        decayed in to the query frame
            5. read out with q, RMSNorm, gate

        xv: [F*S, hidden] -> linear readout [F*S, H*d_v] (pre-to_out_linear, gated+normed).
        qkv_raw: the softmax branch's raw q/k/v for these rows (see _features).
        bounds: per-frame inclusive softmax-window [lo, hi] (see window_bounds).
        frame_size: patched (height, width) of a frame — required iff short_conv is on.
        skip_ends: the partner of HybridAttention.anchor_frames == "both". The softmax side
            makes frames 0 and F-1 exact in both directions, so this branch drops them
            from its INPUT entirely -- not "scanned but ignored" -- and their readout
            rows are exactly zero (softmax and branch stay an exact partition). With
            the two frames gone the window bounds rebase by one: the complement of
            [lo, hi] inside frames 1..F-2 is [1..lo-1] u [hi+1..F-2], i.e.
            gather_linear_state's own arithmetic on (lo-1, hi-1); windows touching a
            clip end get an empty side, which has_before/has_after already handle.
        text_x / text_qkv_raw: the prompt rows. Supplying them turns the branch from
            "the video the window cannot see" into a text-conditioned recurrence: BOTH
            directional scans start from half the text state instead of from zero, so
            every frame's state is a prompt memory that video has written over, and the
            boundary rows read the prompt instead of reading nothing.

            Each direction starts from TEXT_STATE_SCALE (= 0.5) * S_text because
            both carry the SAME prompt and the gather adds them: half each keeps the sum
            at roughly one copy of the prompt, while the per-frame video injections —
            one direction each — stay at full weight.

            This is deliberately NOT the exact complement of the softmax window any
            more: the softmax already sees every text row densely, so the prompt is now
            read twice, once exactly and once as a recurrent state. That is the point of
            the change — the linear branch is being given a condition, not a missing input.

        NOTHING here is arithmetic: the anchor case is a slice, a rebase and a scatter,
        and both cases then run the SAME `_readout`. That split is deliberate — see the
        comment on _readout for why the boundary between the two lives exactly here.

        heads: explicit contiguous slice of the original heads. Both bodies accept
            head-local qkv_raw, beta [F*S,H_local], gate [F*S,H_local,d], and the
            full-hidden-width FP32 frame_mean [F,hidden]; xv may then be None.
            Text likewise uses head-local text_qkv_raw/text_beta. These supplied
            inputs require no_grad even for inference=False, which selects the
            untuned reference body, not a distributed training implementation.
        """

        # WHICH ALGORITHM BODY -- an explicit argument, not ambient state.
        #
        # Deliberately NOT `not torch.is_grad_enabled()`. The inference body differs by
        # more than where a tensor is written -- fused kernels, dropped intermediates,
        # slightly different numerics -- and "which kernel ran" must be something a
        # caller states, not something it infers from a global it set for another
        # reason. Otherwise every no_grad forward in a TRAINING step would silently
        # switch to the fused kernels, including a teacher whose whole job is to be
        # numerically comparable to the student.
        #
        # `inference` selects optimized kernels. A head-sharded reference call is
        # also forward-only, but deliberately keeps inference=False.
        # Grad-enabled is still the correctness precondition,
        # and the two disagreeing is a bug in the caller, not a case to resolve: the
        # inference body writes into preallocated banks (see _run_scans_inference) and
        # HybridAttention adds its readout in place, neither of which a graph survives.
        # So say so, rather than quietly picking one.
        if self.paper_vda_enabled:
            if torch.compiler.is_compiling():
                raise RuntimeError("paper VDA is an eager inference API; compile is unsupported")
            if inference or self.inference_vda_kernels:
                raise ValueError("paper_vda_enabled is exclusive with inference=True "
                                 "and inference_vda_kernels")
            if torch.is_grad_enabled():
                raise RuntimeError("paper_vda_enabled requires torch.no_grad()")
            if (self.delta_rule != "vdn_solve" or not self.a_fp32
                    or self.head_dim != 128 or self.bridge not in ("alpha", "none")):
                raise ValueError("paper VDA requires vdn_solve, a_fp32, head_dim=128")
            if (not isinstance(self.paper_vda_chunk_size, int)
                    or isinstance(self.paper_vda_chunk_size, bool)
                    or self.paper_vda_chunk_size < 1):
                raise ValueError("paper_vda_chunk_size must be a positive integer")
            if any(t.device.type != "cuda" or t.dtype != torch.bfloat16 for t in qkv_raw):
                raise ValueError("paper VDA branch requires CUDA BF16 Q/K/V")
        if self.inference_vda_kernels:
            if inference:
                raise ValueError("inference_vda_kernels requires inference=False; "
                                 "do not combine with global inference_kernels")
            if torch.is_grad_enabled():
                raise RuntimeError("inference_vda_kernels requires torch.no_grad()")
            if (self.delta_rule != "vdn_solve" or not self.a_fp32
                    or self.head_dim != 128 or self.bridge != "alpha"):
                raise ValueError("inference_vda_kernels requires vdn_solve, a_fp32, "
                                 "head_dim=128 and bridge=alpha")
            if any(t.device.type != "cuda" or t.dtype != torch.bfloat16
                   for t in qkv_raw):
                raise ValueError("inference_vda_kernels requires CUDA BF16 Q/K/V")
        if inference and torch.is_grad_enabled():
            raise RuntimeError(
                "inference=True inside an enabled-grad region. The inference body is "
                "not autograd-safe (preallocated state banks, in-place readout add). "
                "Wrap the forward in torch.no_grad(), or leave inference=False -- the "
                "training body is always correct, only heavier. Model-level switch: "
                "hybrid_transform.set_inference_mode(model, True).")
        # The Ulysses inference case hands in what this rank cannot compute itself
        # (its head range; beta, the output gate and the frame mean, which the sequence
        # owner computed from `xv` before dispatch). Either algorithm body can use
        # these, but neither exposes distributed autograd through this interface.
        sharded = dict(heads=heads, beta=beta, gate=gate, frame_mean=frame_mean,
                       text_beta=text_beta)
        self._validate_sharded(heads, beta, gate, frame_mean, text_beta, text_qkv_raw)
        readout_body = self._readout_inference if inference else self._readout
        if self.paper_vda_enabled:
            from src.models.linear_attention.paper_stack import paper_readout
            readout_body = lambda *a, **kw: paper_readout(self, *a, **kw)
        run_readout = lambda *a: readout_body(*a, **sharded)
        n_heads = self.num_heads if heads is None else heads.stop - heads.start
        ref = xv if xv is not None else gate      # dtype/device for the empty rows

        if not skip_ends:
            out = run_readout(xv, num_frames, tokens_per_frame, bounds, qkv_raw,
                              frame_size, text_x, text_qkv_raw)
        elif num_frames <= 2:                    # the anchors ARE the clip
            out = ref.new_zeros(num_frames * tokens_per_frame, n_heads * self.head_dim)
        else:
            inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
            sharded.update(
                beta=None if beta is None else beta[inner],
                gate=None if gate is None else gate[inner],
                frame_mean=None if frame_mean is None else frame_mean[1:-1])
            readout = run_readout(
                None if xv is None else xv[inner], num_frames - 2, tokens_per_frame,
                [(lo - 1, hi - 1) for lo, hi in bounds[1:num_frames - 1]],   # rebased
                tuple(t[inner] for t in qkv_raw),
                frame_size, text_x, text_qkv_raw)
            # zero only the two anchor rows rather than the whole ~1.5 GiB tensor
            out = readout.new_empty(num_frames * tokens_per_frame, readout.shape[-1])
            out[:tokens_per_frame].zero_()
            out[(num_frames - 1) * tokens_per_frame:].zero_()
            out[inner] = readout

        return out

    def _readout(self, xv, num_frames, tokens_per_frame, bounds, qkv_raw,
                 frame_size=None, text_x=None, text_qkv_raw=None, heads=None,
                 beta=None, gate=None, frame_mean=None, text_beta=None):
        """The branch algorithm itself, over exactly the frames it owns.

        NO skip_ends parameter, on purpose: this function has no notion of anchors, and
        the two frames it must never touch are absent from `xv` rather than masked out
        of it. `forward` does the slicing. The rule for what goes where is "would the
        training and inference paths write this line differently?" — the anchor prune
        and scatter are pure tensor motion and would not, so they sit above the split;
        everything below (state banks, autograd-shaped choices) would, so it sits here.
        Keeping the prune and the scatter next to each other in `forward` also means an
        early return added to the algorithm cannot skip the scatter and leave the two
        anchor frames holding garbage instead of zero.
        """
        self._validate_sharded(heads, beta, gate, frame_mean, text_beta, text_qkv_raw)
        num_tokens = num_frames * tokens_per_frame
        head_slice = heads
        heads = self.num_heads if heads is None else heads.stop - heads.start
        head_dim = self.head_dim
        ref = xv if xv is not None else gate
        backend = self._delta_backend("backend", tokens_per_frame)
        shape_per_frame = (num_frames, tokens_per_frame, heads, head_dim)

        # --- 1. features ------------------------------------------------------------
        query, key, value = self._features(qkv_raw, num_frames, frame_size,
                                           heads=head_slice)              # [F*S, H, d]
        query_by_frame = query.view(shape_per_frame)                     # [F, S, H, d]
        key_by_frame = key.view(shape_per_frame).permute(0, 2, 1, 3)     # [F, H, S, d]
        value_by_frame = value.view(shape_per_frame).permute(0, 2, 1, 3)
        if beta is None:
            beta = torch.sigmoid(self.beta_proj(xv))                     # [F*S, H]
        beta = beta.view(num_frames, tokens_per_frame, heads).permute(0, 2, 1)  # [F,H,S]

        # --- 2. collapse each frame's S tokens into two d x d matrices ---------------
        A, B = frame_statistics(key_by_frame, value_by_frame, beta,
                                a_fp32=self.a_fp32)                      # [F,H,d,d]
        # No later operation reads these features. In no-grad evaluation retaining
        # them through normalization needlessly holds several GiB for long clips;
        # autograd retains any tensors its backward needs independently.
        del key, value, key_by_frame, value_by_frame, beta

        # dtype=fp32 on the mean, not just inside alpha: xv is bf16, so without it the
        # frame mean is rounded to bf16 BEFORE reaching the fp32 island, and `.float()`
        # in there cannot recover what the mean already threw away.
        if frame_mean is None:
            frame_mean = xv.view(num_frames, tokens_per_frame, -1).mean(
                dim=1, dtype=torch.float32)
        alpha = self.alpha(frame_mean, heads=head_slice)

        # --- 0/3. scans: prefix_states[t] = frames 0..t, suffix_states[t] = frames t..F-1
        # Seeded with half the text state when the prompt was handed in, so both
        # directions start from the prompt rather than from nothing.
        text_state = self._text_state(text_x, text_qkv_raw,
                                      heads=head_slice, text_beta=text_beta)
        prefix_states, suffix_states = _run_scans(backend, alpha, A, B,
                                                  text_state=text_state)

        # --- 4. boundary gather: the complement of the softmax window, brought to t
        linear_state = gather_linear_state(prefix_states, suffix_states, alpha, bounds,
                                     bridge=self.bridge, text_state=text_state,
                                     inference=self.inference_vda_kernels,
                                     out_dtype=(ref.dtype if self.inference_vda_kernels
                                                else None)).to(ref.dtype)
        del prefix_states, suffix_states, alpha, A, B, text_state

        # --- 5. read out with the query, then normalise and gate ---------------------
        readout = torch.einsum("fhvk,fshk->fshv", linear_state, query_by_frame)
        del linear_state, query_by_frame, query

        readout = readout.reshape(num_tokens, heads, head_dim)
        if self.inference_vda_kernels:
            if gate is None:
                gate = self.output_gate(xv)
            return vda_bf16_epilogue(readout, self.norm.weight, gate, self.norm.eps)
        readout = self.norm(readout)
        if gate is None:
            gate = self.output_gate(xv)
        return (readout * gate).reshape(num_tokens, heads * head_dim)

    def _readout_inference(self, xv, num_frames, tokens_per_frame, bounds, qkv_raw,
                           frame_size=None, text_x=None, text_qkv_raw=None, heads=None,
                           beta=None, gate=None, frame_mean=None, text_beta=None):
        """`_readout` for a forward-only call. Same math, same numbers, less live memory.

        Selected by `forward(..., inference=True)`; never by ambient grad state.

        TWO differences from `_readout`, and nothing else:

          1. the scan writes into preallocated banks (`_run_scans_inference`) -- same
             ops in the same order, BITWISE equal, half the live memory;
          2. the norm and the output gate go through one compiled kernel
             (`linear_epilogue`) -- several passes of pointwise work on a ~1.4 GiB
             tensor collapsed to one, and NOT bitwise, because inductor rounds once
             where eager rounds after every op (~1e-3 relative, in the accurate
             direction).

        Everything else is a verbatim copy on purpose rather than one body taking an
        `inference` flag. The flag chooses BETWEEN the two bodies, at the top; a flag
        threaded THROUGH one body is the thing being avoided, because then the two paths
        share every line and drift only in the branches nobody reads. A further
        difference goes here -- with a line saying why, and with its own error budget.

        Both bodies can run on a HEAD RANGE. Branch-parallel Ulysses
        gives each linear rank all rows of `heads` = slice(a, b) heads, plus `beta`,
        `gate` and `frame_mean` computed by the sequence owner (this rank has no `xv`),
        and `text_beta` for the prompt rows. Every parameter read below then takes the
        head slice (conv channels, alpha's up/dt_bias/A_log; the norm weight is per
        head_dim and shared by all heads, so it stays whole). The math is independent
        across heads, but shape-dependent kernel selection and reduction order can
        change rounding, including BF16 stores. The reference body retains eager
        feature, scan and epilogue operations;
        head slicing is not a bitwise parity guarantee for either body.
        """
        num_tokens = num_frames * tokens_per_frame
        head_dim = self.head_dim
        n_heads = self.num_heads if heads is None else heads.stop - heads.start
        channels = (slice(None) if heads is None
                    else slice(heads.start * head_dim, heads.stop * head_dim))
        backend = self._delta_backend("backend", tokens_per_frame)
        shape_per_frame = (num_frames, tokens_per_frame, n_heads, head_dim)

        # q comes back frame-major, [F, H, S, d]: the readout below is then a plain
        # batched matmul instead of an einsum that permutes q in and the result out.
        query_by_frame, key, value = self._features(
            qkv_raw, num_frames, frame_size, inference=True,
            query_fhsd=(num_frames, tokens_per_frame), heads=heads)
        key_by_frame = key.view(shape_per_frame).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape_per_frame).permute(0, 2, 1, 3)
        if beta is None:
            beta = torch.sigmoid(self.beta_proj(xv))
        beta = beta.view(num_frames, tokens_per_frame, n_heads).permute(0, 2, 1)

        A, B = frame_statistics(key_by_frame, value_by_frame, beta, a_fp32=self.a_fp32,
                                inference=True)
        del key, value, key_by_frame, value_by_frame     # 2.4 GiB, summarised into A and B
        if frame_mean is None:
            frame_mean = xv.view(num_frames, tokens_per_frame, -1).mean(
                dim=1, dtype=torch.float32)
        alpha = self.alpha(frame_mean, heads=heads)

        # _text_chunk_state deliberately stays on the eager prologue: the prompt is a
        # few hundred rows against the video's ~100k, a negligible share of the branch,
        # and a second static shape in the same compiled function would cost a
        # recompile per caption length.
        text_state = self._text_state(text_x, text_qkv_raw, heads=heads, text_beta=text_beta)
        prefix_states, suffix_states = _run_scans_inference(   # <-- the one difference
            backend, alpha, A, B, text_state=text_state)

        if gate is None:
            gate = self.output_gate(xv)
        linear_state = gather_linear_state(prefix_states, suffix_states, alpha, bounds,
                                     bridge=self.bridge, text_state=text_state, inference=True,
                                     out_dtype=gate.dtype)
        del prefix_states, suffix_states       # ~0.7 GiB, dead before the einsum runs

        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))  # [F,H,S,dv]
        # The norm is RMSNorm(head_dim), one weight vector shared by every head: no slice.
        return linear_epilogue(readout, self.norm.weight, gate, self.norm.eps,
                             inference=True, fhsd=True)
