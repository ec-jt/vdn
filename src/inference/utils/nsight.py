"""Opt-in host NVTX ranges and per-rank, bounded Nsight Systems capture.

Set VDN_NVTX=1 and/or VDN_NSYS_CAPTURE=1 before starting Python. Flags are
cached at import; neither importing this module nor disabled instrumentation
imports torch or calls CUDA. The event profiler is intentionally independent.
Use nsys --capture-range=cudaProfilerApi --capture-range-end=repeat for the
separate measured denoise (zero-based steps 2 and 3) and decode windows.
"""

from contextlib import contextmanager, nullcontext
from functools import wraps
import os


NVTX_ENABLED = os.environ.get("VDN_NVTX") == "1"
CAPTURE_ENABLED = os.environ.get("VDN_NSYS_CAPTURE") == "1"
_NULL_RANGE = nullcontext()


def _cuda():
    # Lazy access is important: disabled helpers must not even probe CUDA.
    import torch
    return torch.cuda


@contextmanager
def _nvtx_range(name):
    nvtx = _cuda().nvtx
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def nvtx_range(name):
    """Exception-balanced host range; no events, synchronization or GPU probe."""
    return _nvtx_range(name) if NVTX_ENABLED else _NULL_RANGE


def nvtx_annotate(name):
    """Return the original function when disabled (no wrapper-call overhead)."""
    def decorate(function):
        if not NVTX_ENABLED:
            return function

        @wraps(function)
        def wrapped(*args, **kwargs):
            with nvtx_range(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def annotate_forward(module, name):
    """Wrap an inference instance only when enabled; safe to install repeatedly.

    This also covers unfused external FFNs without changing their arithmetic or
    registering hooks on the normal path. It is not intended for compiled graphs.
    """
    if NVTX_ENABLED and not getattr(module.forward, "_vdn_nvtx", False):
        forward = nvtx_annotate(name)(module.forward)
        forward._vdn_nvtx = True
        module.forward = forward


class NsightCapture:
    """One controller per rank, constructed after runtime device initialization.

    Call step only for measured sampling, not warmup. All ranks enter decode,
    including workers waiting for rank-zero output. The caller must place the
    collective decode completion inside decode(). No new barriers are added.
    """

    def __init__(self, device):
        self.device = device
        self.active = False

    def start(self):
        if CAPTURE_ENABLED and not self.active:
            _cuda().profiler.start()
            self.active = True

    def stop(self):
        if not self.active:
            return
        # The only synchronization added by instrumentation: drain asynchronous
        # work before cudaProfilerStop, never at start or individual NVTX regions.
        try:
            _cuda().synchronize(self.device)
        finally:
            try:
                _cuda().profiler.stop()
            finally:
                self.active = False

    @contextmanager
    def step(self, index, *, last=False):
        if index == 2:
            self.start()
        try:
            with nvtx_range(f"denoise.step.{index}"):
                yield
        except BaseException:
            self.stop()
            raise
        finally:
            if index == 3 or last:
                self.stop()

    @contextmanager
    def decode(self):
        self.start()
        try:
            with nvtx_range("decode.all_ranks"):
                yield
        finally:
            self.stop()
