# SPDX-License-Identifier: Apache-2.0
# Modified 2026-10-05 from SGLang PR #37903's vdn_delta_factors_jit.py:
# PyTorch extension instead of TVM/SGLang JIT; strict standalone inference API,
# optional alpha, lazy CUDA toolchain selection. See licenses/SGLang-NOTICE.txt.
"""Author-associated blocked FP32 Gauss-Jordan factors, without a fallback.

Import does not probe CUDA or compile. The first nonempty supported call builds
the extension with a local CUDA toolkit (CUDA_HOME, default /usr/local/cuda-13.1).
The caller must provide statistics for which I + A is symmetric positive definite.
No symmetry, finiteness, condition-number or pivot checks are performed on device.
This is an eager inference API, not an autograd or torch.compile custom operator.
"""

from __future__ import annotations

import os
import hashlib
import importlib.util
import subprocess
import sysconfig
import fcntl
from pathlib import Path
from threading import Lock

import torch

__all__ = ["paper_delta_factors"]

_HEAD_DIM = 128
_BUILD_LOCK = Lock()
_EXTENSIONS: dict[tuple[str, int, int], object] = {}


def _load_extension(capability: tuple[int, int]):
    """Load once per local toolchain/SM, on demand; errors propagate unchanged."""
    toolkit = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.1")).resolve()
    nvcc = toolkit / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"paper_delta_factors needs a local CUDA toolkit: missing {nvcc}")
    # Do not silently let PyTorch select a compiler from another toolkit.
    override = os.environ.get("PYTORCH_NVCC")
    if override and Path(override).resolve() != nvcc:
        raise RuntimeError("PYTORCH_NVCC must name CUDA_HOME/bin/nvcc for paper_delta_factors")
    major, minor = capability
    key = (str(toolkit), major, minor)
    with _BUILD_LOCK:
        if key not in _EXTENSIONS:
            from torch.utils import cpp_extension

            if torch.version.cuda is None or cpp_extension.IS_HIP_EXTENSION:
                raise RuntimeError("paper_delta_factors requires a CUDA-enabled PyTorch build")
            source = Path(__file__).resolve().parents[1] / "ops" / "csrc"
            # cpp_extension caches CUDA_HOME at import. Override it only for this
            # build and restore it even on failure; importing this module changes
            # neither the environment nor PyTorch's global build configuration.
            old_home = cpp_extension.CUDA_HOME
            try:
                cpp_extension.CUDA_HOME = str(toolkit)
                module = _build_direct(cpp_extension, source, toolkit, major, minor)
            finally:
                cpp_extension.CUDA_HOME = old_home
            _EXTENSIONS[key] = module
        return _EXTENSIONS[key]


def _build_direct(ext, source, toolkit, major, minor):
    """Build the PyTorch extension directly: this environment has no Ninja.

    No dependency installation or alternate device algorithm. A content-addressed
    build directory and interprocess lock permit eight ranks to reuse one binary.
    """
    name = f"vdn_paper_delta_factors_sm{major}{minor}"
    digest = hashlib.sha256()
    for suffix in ("cpp", "cu", "cuh"):
        digest.update((source / f"vdn_delta_factors.{suffix}").read_bytes())
    digest.update((Path(__file__).read_text() + torch.__version__ + str(toolkit)).encode())
    build = Path(ext.get_default_build_root()) / (name + "_" + digest.hexdigest()[:16])
    build.mkdir(parents=True, exist_ok=True)
    target = build / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    with (build / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            inc = ["-I" + p for p in ext.include_paths(device_type="cuda")]
            inc += ["-I" + sysconfig.get_paths()["include"]]
            common = ["-O3", "-std=c++17", "-DTORCH_EXTENSION_NAME=" + name,
                      "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
            cxx = os.environ.get("CXX", "c++")
            commands = [
                [cxx, *common, *inc, "-fPIC", "-c", str(source / "vdn_delta_factors.cpp"),
                 "-o", str(build / "binding.o")],
                [str(toolkit / "bin/nvcc"), *common, *inc, "-Xcompiler=-fPIC",
                 f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
                 "-c", str(source / "vdn_delta_factors.cu"), "-o", str(build / "cuda.o")],
                [cxx, "-shared", str(build / "binding.o"), str(build / "cuda.o"),
                 *["-L" + p for p in ext.library_paths(device_type="cuda")],
                 "-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10_cuda",
                 "-ltorch_cuda", "-lcudart", "-o", str(target)],
            ]
            with (build / "build.log").open("w") as log:
                for command in commands:
                    log.write(" ".join(command) + "\n")
                    log.flush()
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f"VDN CUDA extension build failed; see {build / 'build.log'}")
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_tensor(t: torch.Tensor, name: str) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if t.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype float32, got {t.dtype}")
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.layout != torch.strided or not t.is_contiguous():
        raise ValueError(f"{name} must have contiguous strided storage")
    if t.requires_grad:
        raise ValueError(f"{name} requires_grad is unsupported by this inference-only kernel")
    if t.is_conj() or t.is_neg():
        raise ValueError(f"{name} must not have unresolved conjugate/negative view bits")


def _aligned(t: torch.Tensor) -> torch.Tensor:
    # Upstream float4 loads require 16-byte alignment. contiguous() is insufficient
    # because a contiguous view may retain a misaligned storage offset.
    return t if t.data_ptr() % 16 == 0 else t.clone()


def paper_delta_factors(
    A: torch.Tensor, B: torch.Tensor, alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(diag(alpha) @ inv(I + A), B @ inv(I + A))``.

    A and B must be CUDA contiguous FP32 tensors of the exact same shape
    ``[..., 128, 128]`` and device. Optional alpha has exact shape ``[..., 128]``
    with the same dtype/device/layout; None means all-one row scales. Alpha does
    not scale the injection. Outputs retain the input shapes and are independent
    FP32 allocations. No input is mutated and no broadcasting/conversion occurs.

    Requires SM80+ and at least 64 KiB dynamic plus 4 KiB static shared memory per
    block. Contiguous misaligned inputs are cloned to satisfy float4 alignment.
    Empty CUDA batches return empty outputs without compiling/launching. Invalid
    metadata raises TypeError/ValueError; unsupported hardware, builds and launches
    raise RuntimeError. There is no CPU, Cholesky, solve or alternate kernel path.

    Nonpivoting inversion assumes I + A is SPD (normally A is a PSD statistic).
    Singular/indefinite/nonfinite inputs violate that numerical precondition and
    may yield NaNs or inaccurate results rather than a diagnostic. Gradient inputs
    are rejected, including under no_grad. Warm up before any CUDA graph capture;
    this standalone port makes no graph-capture or compiler-integration guarantee.
    """
    _check_tensor(A, "A")
    _check_tensor(B, "B")
    if A.ndim < 2 or A.shape[-2:] != (_HEAD_DIM, _HEAD_DIM):
        raise ValueError("A must have shape [..., 128, 128]")
    if B.shape != A.shape:
        raise ValueError("B shape must exactly equal A shape")
    if B.device != A.device:
        raise ValueError("A and B must be on the same CUDA device")
    if alpha is not None:
        _check_tensor(alpha, "alpha")
        if alpha.shape != A.shape[:-1]:
            raise ValueError("alpha must have exact shape [..., 128] matching A")
        if alpha.device != A.device:
            raise ValueError("alpha must be on the same CUDA device as A")
    capability = torch.cuda.get_device_capability(A.device)
    if capability[0] < 8:
        raise RuntimeError("paper_delta_factors requires SM80 or newer")
    if A.numel() == 0:
        return torch.empty_like(A), torch.empty_like(B)
    with torch.cuda.device(A.device):
        module = _load_extension(capability)
        A_aligned, B_aligned = _aligned(A), _aligned(B)
        alpha_aligned = (torch.ones(A.shape[:-1], device=A.device, dtype=torch.float32)
                         if alpha is None else _aligned(alpha))
        n = A.numel() // (_HEAD_DIM * _HEAD_DIM)
        transition, injection = module.delta_factors(
            A_aligned.view(n, _HEAD_DIM, _HEAD_DIM),
            B_aligned.view(n, _HEAD_DIM, _HEAD_DIM),
            alpha_aligned.view(n, _HEAD_DIM),
        )
        return transition.view(A.shape), injection.view(B.shape)
