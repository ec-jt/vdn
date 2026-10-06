"""Opt-in, env-gated per-block activation tracer for drift investigation.

Enabled only when ``VDN_TRACE_PATH`` is set. Registers forward hooks on every
transformer block plus its attention and feed-forward children, and records
compact per-(step, block) scalars for the first ``VDN_TRACE_STEPS`` denoising
forwards. The first ``VDN_TRACE_ROWS`` rows of each captured tensor are reduced,
so a single-GPU (full-sequence) run and rank zero of a row-sharded run produce
comparable numbers for the same sequence rows.

Read-only: this never changes arithmetic or tensor shapes. It writes a JSON
file on process exit. Only rank zero writes, so torchrun peers cannot race on
the output path.
"""
from __future__ import annotations

import atexit
import json
import os

import torch


def _snap(out, trace_rows):
    if isinstance(out, (tuple, list)):
        out = out[0] if out else None
    if out is None or not torch.is_tensor(out) or out.dim() < 2:
        return None
    v = out.detach()
    # Blocks run on [1, S, H] (3D) or [S, H] (2D); the sequence axis is 1 or 0.
    axis = 1 if v.dim() >= 3 else 0
    seq = int(v.shape[axis])
    if seq > trace_rows:
        v = v.narrow(axis, 0, trace_rows)
    f = v.float()
    return {
        "norm": float(f.norm().item()),
        "mean": float(f.mean().item()),
        "amax": float(f.abs().max().item()),
        "checksum": float(f.double().sum().item()),
        "shape": list(v.shape),
        "seq": int(v.shape[axis]),
    }


def install_block_trace(model, *, rank=0, trace_rows=4814):
    """Install hooks on ``model.transformer_blocks`` when VDN_TRACE_PATH is set.

    Returns the mutable state dict (for tests) or None when disabled / off rank.
    """
    path = os.environ.get("VDN_TRACE_PATH")
    if not path or rank != 0:
        return None
    steps = int(os.environ.get("VDN_TRACE_STEPS", "8"))

    blocks = list(getattr(model, "transformer_blocks", []))
    if not blocks:
        return None
    n = len(blocks)
    state = {"call": 0, "cur": None, "records": []}

    def _current():
        c = state["call"]
        step, b = divmod(c, n)
        return step, b

    for blk in blocks:
        def pre(mod, args, state=state):
            c = state["call"]
            state["call"] = c + 1
            step, b = divmod(c, n)
            if step >= steps:
                state["cur"] = None
                return
            state["cur"] = (step, b)
            s = _snap(args[0] if args else None, trace_rows)
            if s is not None:
                s.update(step=step, block=b, kind="block_in")
                state["records"].append(s)

        def post(mod, args, out, state=state):
            cur = state["cur"]
            if cur is None:
                return
            s = _snap(out, trace_rows)
            if s is not None:
                s.update(step=cur[0], block=cur[1], kind="block_out")
                state["records"].append(s)

        blk.register_forward_pre_hook(pre)
        blk.register_forward_hook(post)

        attn = getattr(blk, "attn", None)
        if attn is not None:
            def apost(mod, args, out, state=state):
                cur = state["cur"]
                if cur is None:
                    return
                s = _snap(out, trace_rows)
                if s is not None:
                    s.update(step=cur[0], block=cur[1], kind="attn_out")
                    state["records"].append(s)

            attn.register_forward_hook(apost)

        ff = getattr(blk, "ff", None)
        if ff is not None:
            def fpost(mod, args, out, state=state):
                cur = state["cur"]
                if cur is None:
                    return
                s = _snap(out, trace_rows)
                if s is not None:
                    s.update(step=cur[0], block=cur[1], kind="ff_out")
                    state["records"].append(s)

            ff.register_forward_hook(fpost)

    def dump():
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump({"rank": rank, "steps": steps, "nblocks": n,
                           "trace_rows": trace_rows, "records": state["records"]}, f)
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"block trace dump failed: {exc}", flush=True)

    atexit.register(dump)
    return state
