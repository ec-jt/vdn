"""Bounded rank-local VDA overlap. No collectives belong in the submitted callable."""
from contextlib import contextmanager

import torch


@contextmanager
def vda_on_side_stream(owner, fn, *, inputs, module, paper_mode=False):
    """Submit one branch and join on the caller stream even if either branch raises.

    The caller must have established input/weight readiness on its current stream.
    Recording storage use protects packed views, frame means and resident weight
    views against allocator reuse. The reverse wait bounds the lifetime to this
    attention call, before output exchange or weight-shard eviction/prefetch.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("VDA overlap requires no-grad inference")
    if paper_mode and getattr(module, "inference_vda_kernels", False):
        raise ValueError("explicit paper overlap cannot use legacy fused VDA")
    device = inputs[0].device
    if device.type != "cuda":
        raise ValueError("VDA overlap requires CUDA inputs")
    current = torch.cuda.current_stream(device)
    side = getattr(owner, "_vda_stream", None)
    if side is None:
        side = torch.cuda.Stream(device=device)
        owner._vda_stream = side
    if side.device != device:
        raise ValueError("VDA overlap stream device changed")
    side.wait_stream(current)
    output = None
    try:
        with torch.cuda.stream(side):
            # Keep references until the join is enqueued, and record every source
            # storage, including views into the weight-sharding materialization.
            for tensor in (*inputs, *module.parameters(), *module.buffers()):
                if tensor.is_cuda:
                    tensor.record_stream(side)
            output = fn()
        yield output
    finally:
        current.wait_stream(side)
        if output is not None:
            output.record_stream(current)
