"""Inference overlay config. Four families, four sections: runtime optimization
(kernels), precision modification (fp8/dtype), multi-GPU layout (parallel, honoured by
infer_ulysses.py only; infer.py refuses non-default values), semantic ablation -- and
the last is gated: any semantic override with ablation.enabled false is a load-time
error, not a warning.

Every knob is a field here (YAML + dotlist); nothing on the inference path reads an
environment variable (FA_CLC is flash-attn's own). Arch-dependent choices resolve at
the setter, not the config: `softmax_backend: auto` is the decomposed window kernel on
any CUDA device, fp8 scale granularity follows the capability -- one YAML for both
architectures."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, OmegaConf

from src.config.common import KernelsConfig
from src.models.softmax_attention.decomposed import SOFTMAX_BACKENDS


@dataclass
class InferenceKernels(KernelsConfig):
    # softmax_backend (inherited): auto | flex | decomposed | ref -- WHICH window-softmax
    # kernel. auto = decomposed on every CUDA device (hybrid_transform.set_softmax_backend).
    #
    # inference_kernels: set_inference_mode(model, True) -- the forward-only kernel set,
    # one switch, same arithmetic:
    #   attention   fused QK-norm + RoPE in one pass, q/k/v consumed as views of the
    #               packed [T,H,d] buffer; the window softmax on the static
    #               FLASH/CuteDSL flex variant instead of the dynamic Triton one; the
    #               softmax gate fused with the to_out repack
    #   far branch  one-kernel frame-statistics prologue; Triton 5-tap temporal conv;
    #               compiled RMSNorm + output gate epilogue; one triangular solve for
    #               the vdn inverse; readout in the bmm-native layout; cached gather
    #               indices; no anchor zero-fill; baddbmm scan updates; TF32 on the one
    #               fp32 GEMM
    #   block       RMSNorm + AdaLN affine and gate + residual as two compiled kernels;
    #               SwiGLU as one compiled kernel
    # False = the released arithmetic spelled eagerly (rung 0 of the yaml ladder).
    inference_kernels: bool = True
    # Experimental reference-body candidates, independent of the global switch.
    vda_fused: bool = False
    vda_overlap: bool = False


@dataclass
class Fp8Config:
    enabled: bool = False                # NEVER default-on: fp8 changes the sample
    skip_end_blocks: int = 4             # scale granularity is per arch, not a knob


@dataclass
class PrecisionConfig:
    dtype: str = "bfloat16"
    fp8: Fp8Config = field(default_factory=Fp8Config)


@dataclass
class PaperComputeConfig:
    """Independent research switches. None changes the original kernel defaults."""
    paper_mxfp8: bool = False
    paper_qk_rope: bool = False
    paper_window: bool = False
    paper_vda: bool = False
    paper_vda_overlap: bool = False
    paper_adaln: bool = False
    paper_fused_block: bool = False
    paper_max_gather_rows: int = 65536
    paper_max_query_rows: int = 8192
    paper_adaln_table_mib: int = 256


def paper_enabled(cfg):
    compute = getattr(cfg, "compute", None)
    return compute is not None and any(getattr(compute, name) for name in (
        "paper_mxfp8", "paper_qk_rope", "paper_window", "paper_vda",
        "paper_vda_overlap", "paper_adaln", "paper_fused_block"))


@dataclass
class ParallelConfig:
    """torchrun layout for infer_ulysses.py."""
    softmax_ranks: int = 6               # 0 = standard Ulysses; n = n softmax + (world-n) linear
    profile: bool = False                # per-section CUDA events: profiling only
    timeout_seconds: int = 600           # bounded collectives/startup/decoder wait
    standard_exchange: str = "alltoall"  # alltoall | broadcast | allgather; standard only


@dataclass
class BehaviorConfig:
    teacher_mode: bool = False


@dataclass
class AblationConfig:
    enabled: bool = False
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    # Execution device remains render.device (or build_inference_model's device).
    policy: str = "legacy"             # legacy | cpu_block_offload | resident_weight_shards
    local_files_only: bool = False     # includes repository checkpoint resolution
    shard_world_size: int = 8
    scatter_chunk_mib: int = 32        # per destination; owner stages at most world * chunk
    shard_distribution: str = "scatter"  # scatter | broadcast (SHM-copy scatter workaround)
    weight_prefetch: bool = False     # next-block overlap; resident standard Ulysses only


@dataclass
class RenderConfig:
    prompt_file: str = MISSING
    out: str = MISSING
    num_frames: int = 345
    num_steps: int = 50
    warmup_steps: int = 0                # NFE run and discarded first: 2 for any timing
                                         # (compile + the 2nd-timestep re-specialisation)
    seed: int = 42
    device: str = "cuda:0"
    video_shift: float = 12.0            # scheduler shifts; a distilled adapter is only
    audio_shift: float = 3.0             # valid at the shifts it was trained for
    save_latents: bool = False           # optional parity/debug tensors live outside
    denoise_only: bool = False           # release denoiser and record; no decoder weights loaded
    latents_root: str = "artifacts/latents"  # results/: JSON/JSONL + MP4 only
    record: bool = False                 # also write <out>.inference.json (checkpoint
                                         # identity, resolved config, actual kernel
                                         # state, per-NFE timings) -- a debug aid


@dataclass
class ExternalLora:
    """A community adapter (safetensors, peft names against the DENSE model), merged
    after the checkpoint's own LoRA. alpha=None reads the file's metadata (missing
    there too -> alpha=rank, scale 1.0)."""
    path: str = MISSING
    alpha: Optional[float] = None


@dataclass
class InferenceConfig:
    checkpoint: Optional[str] = None     # None = render the DENSE base model
    base_source: Optional[str] = None    # base weights path override; identity stays with the spec
    vae_source: Optional[str] = None     # the decoders' root; None = the release copy (paths.H3_BASE)
    external_loras: List[ExternalLora] = field(default_factory=list)
    render: RenderConfig = field(default_factory=RenderConfig)
    kernels: InferenceKernels = field(default_factory=InferenceKernels)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    compute: PaperComputeConfig = field(default_factory=PaperComputeConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def validate_ablation(cfg) -> None:
    if cfg.ablation.overrides and not cfg.ablation.enabled:
        raise ValueError(
            f"semantic overrides {sorted(cfg.ablation.overrides)} require "
            f"ablation.enabled=true -- they change what the model computes"
        )


def validate_kernels(cfg) -> None:
    validate_memory(cfg)
    if cfg.kernels.softmax_backend not in SOFTMAX_BACKENDS:
        raise ValueError(f"kernels.softmax_backend {cfg.kernels.softmax_backend!r} "
                         f"not in {SOFTMAX_BACKENDS}")
    if cfg.render.warmup_steps < 0:
        raise ValueError("render.warmup_steps must be >= 0")
    if cfg.precision.fp8.skip_end_blocks < 0:
        raise ValueError("precision.fp8.skip_end_blocks must be >= 0")


def validate_parallel(cfg) -> None:
    """infer_ulysses.py: the layout is checked before the process group exists; the
    world-size bound (softmax_ranks < WORLD_SIZE) is checked at install."""
    if cfg.memory.policy == "cpu_block_offload":
        raise ValueError("cpu_block_offload is single-process only, not Ulysses")
    if cfg.parallel.softmax_ranks < 0:
        raise ValueError("parallel.softmax_ranks must be >= 0 (0 = standard Ulysses)")
    if cfg.parallel.timeout_seconds <= 0:
        raise ValueError("parallel.timeout_seconds must be positive")
    if cfg.parallel.standard_exchange not in ("alltoall", "broadcast", "allgather"):
        raise ValueError("parallel.standard_exchange must be alltoall, broadcast or allgather")
    if cfg.parallel.standard_exchange != "alltoall" and cfg.parallel.softmax_ranks:
        raise ValueError("parallel.standard_exchange=broadcast requires softmax_ranks=0")


def validate_single_process(cfg) -> None:
    """infer.py: a parallel.* override would silently do nothing on the single-GPU
    entrypoint, so it is refused."""
    if cfg.memory.policy == "resident_weight_shards" or cfg.render.denoise_only:
        raise ValueError("resident_weight_shards/denoise_only require infer_ulysses.py")
    if OmegaConf.to_container(cfg.parallel) != asdict(ParallelConfig()):
        raise ValueError("parallel.* only applies to src/inference/infer_ulysses.py "
                         "(torchrun); infer.py is single-process")


def validate_memory(cfg) -> None:
    if paper_enabled(cfg):
        for name, default in asdict(PaperComputeConfig()).items():
            value = getattr(cfg.compute, name)
            if isinstance(default, bool):
                if type(value) is not bool:
                    raise ValueError(f"compute.{name} must be a bool")
            elif type(value) is not int or value <= 0:
                raise ValueError(f"compute.{name} must be a positive integer")
        if (cfg.memory.policy != "resident_weight_shards"
                or cfg.parallel.softmax_ranks != 0
                or str(cfg.render.device).split(":", 1)[0] != "cuda"):
            raise ValueError("paper stack requires CUDA resident standard eight-way Ulysses")
        if cfg.kernels.inference_kernels or cfg.kernels.vda_fused or cfg.kernels.vda_overlap:
            raise ValueError("paper stack requires global/legacy experimental kernels off")
        if cfg.render.warmup_steps:
            raise ValueError("paper stack rejects warmup: exactly one preplanned request is supported")
        if (cfg.render.num_frames != 124 or cfg.render.num_steps != 8
                or cfg.render.seed != 42
                or str(cfg.render.prompt_file) != "prompts/example_0.pt"):
            raise ValueError("paper stack is restricted to 124 frames, 8 NFE, seed42, cached prompt0")
    for name in ("vda_fused", "vda_overlap"):
        if not isinstance(getattr(cfg.kernels, name), bool):
            raise ValueError(f"kernels.{name} must be a bool")
    if cfg.kernels.vda_fused or cfg.kernels.vda_overlap:
        if cfg.memory.policy != "resident_weight_shards":
            raise ValueError("VDA candidates require resident_weight_shards (standard eight-way Ulysses)")
        if str(cfg.render.device).split(":", 1)[0] != "cuda":
            raise ValueError("VDA candidates require a CUDA execution device")
    if cfg.memory.policy not in ("legacy", "cpu_block_offload", "resident_weight_shards"):
        raise ValueError(f"unknown memory.policy {cfg.memory.policy!r}")
    if not isinstance(cfg.memory.weight_prefetch, bool):
        raise ValueError("memory.weight_prefetch must be a bool")
    if cfg.memory.weight_prefetch:
        if cfg.memory.policy != "resident_weight_shards":
            raise ValueError("memory.weight_prefetch requires resident_weight_shards")
        if cfg.parallel.softmax_ranks != 0:
            raise ValueError("memory.weight_prefetch requires standard Ulysses (softmax_ranks=0)")
        if str(cfg.render.device).split(":", 1)[0] != "cuda":
            raise ValueError("memory.weight_prefetch requires a CUDA execution device")
    if cfg.memory.policy == "legacy":
        return
    if cfg.memory.policy == "resident_weight_shards":
        if (cfg.precision.dtype != "bfloat16" or cfg.precision.fp8.enabled
                or cfg.kernels.inference_kernels or cfg.kernels.softmax_backend != "flex"
                or cfg.behavior.teacher_mode or cfg.parallel.softmax_ranks != 0):
            raise ValueError("resident_weight_shards requires untuned BF16, FP8 off, Flex, teacher off, softmax_ranks=0")
        if not cfg.memory.local_files_only or not cfg.checkpoint:
            raise ValueError("resident_weight_shards requires local-only hybrid checkpoint")
        if cfg.memory.shard_world_size != 8:
            raise ValueError("resident_weight_shards production policy requires eight ranks")
        if not 1 <= cfg.memory.scatter_chunk_mib <= 64:
            raise ValueError("memory.scatter_chunk_mib must be in [1,64]")
        if cfg.memory.shard_distribution not in ("scatter", "broadcast"):
            raise ValueError("memory.shard_distribution must be scatter or broadcast")
        if cfg.ablation.enabled or cfg.ablation.overrides:
            raise ValueError("resident_weight_shards does not support ablation")
        if cfg.parallel.timeout_seconds <= 0:
            raise ValueError("parallel.timeout_seconds must be positive")
        return
    if cfg.precision.dtype != "bfloat16":
        raise ValueError("cpu_block_offload requires precision.dtype=bfloat16")
    if cfg.precision.fp8.enabled:
        raise ValueError("cpu_block_offload does not support FP8")
    if cfg.kernels.inference_kernels:
        raise ValueError("cpu_block_offload requires kernels.inference_kernels=false")
    if cfg.behavior.teacher_mode:
        raise ValueError("cpu_block_offload does not support the dense teacher attention path")
    parallel = cfg.parallel
    if (parallel.softmax_ranks != ParallelConfig().softmax_ranks or parallel.profile
            or parallel.standard_exchange != "alltoall"):
        raise ValueError("cpu_block_offload does not support parallel overrides")
