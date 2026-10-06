"""One model assembly for every inference entrypoint.

    model = build_inference_model(cfg, device)          # infer.py
    model = build_inference_model(cfg, device, load_decoders=runtime.is_main,
                                  log=runtime.is_main)  # infer_ulysses.py

Order of operations, the same for both: read the artifact's spec -> load the base ->
apply the transform -> load the branch weights -> fold the artifact's LoRAs -> fold the
external LoRAs -> eval/no-grad on device -> behaviour -> the inference kernel set
(set_inference_mode) -> window-softmax backend -> ablation -> fp8. Then `render_record`
says what ACTUALLY ran (fp8 layer count, the flex latch, the resolved backend), so a
silent downgrade is never silent in the JSON next to the mp4.

Every knob is a config field (src/config/inference.py); nothing here reads an
environment variable. `checkpoint: null` is the released DENSE model: no transform, no
branch, and the hybrid-only overlays are skipped.
"""
import os
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from diffusers import (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio,
                       MiniMaxH3Transformer3DModel)

from src.checkpoints import checkpoint_head_sha256, load_checkpoint
from src.config import resolved_dict
from src.config.inference import validate_memory, paper_enabled
from src.inference.utils.block_offload import BlockOffloadController
from src.inference.utils.cpu_loading import (assert_cpu_materialized, load_cpu_model,
                                           resolve_inference_weights, validate_branch_weights)
from src.inference.utils.lora import load_external_lora, merge_lora_state
from src.paths import H3_BASE
from src.models.factory import load_model_weights
from src.models.hybrid_transform import (apply_hybrid_attention_transform, iter_hybrids,
                                         set_inference_mode, set_softmax_backend)
from src.models.ops.fp8_linear import convert_linear_to_fp8
from src.models.softmax_attention.flex_attention import _FLEX_CACHE

# Semantic knobs an ablation may touch, and where they live. Anything else is refused.
ABLATABLE = ("radius", "chunk", "anchor_frames", "bridge", "enable_text_state",
             "linear_attention_enabled", "enable_softmax_gate")


@dataclass
class InferenceModel:
    transformer: Any
    vae: Any                                  # None on ranks that do not decode
    audio_vae: Any
    artifact: Any                             # CheckpointArtifact, or None for dense
    is_hybrid: bool
    merged_lora_pairs: int = 0
    external_loras: List[Dict[str, Any]] = field(default_factory=list)
    ablated: List[str] = field(default_factory=list)
    fp8_linears: int = 0
    softmax_backend: Optional[str] = None     # resolved (auto -> flex|decomposed)
    offload: Optional[BlockOffloadController] = None
    weight_shards: Any = None
    startup_metrics: Dict[str, Any] = field(default_factory=dict)
    paper_runtime: Any = None


def apply_ablation(model, overrides):
    for key in overrides:
        if key not in ABLATABLE:
            raise ValueError(f"{key!r} is not an ablatable knob ({ABLATABLE})")

    for attn in iter_hybrids(model):
        for key, value in overrides.items():
            setattr(attn, key, value)

    return sorted(overrides)


def flex_latch_state():
    return {"compiled_variants": [k for k in ("infer", "train") if k in _FLEX_CACHE],
            "flash_available": _FLEX_CACHE.get("flash_available")}


def apply_vda_candidates(model, cfg):
    """Set independent switches on owner and header/meta graphs alike."""
    for attn in iter_hybrids(model):
        attn.inference_vda_overlap = cfg.kernels.vda_overlap
        attn.linear_attention.inference_vda_kernels = cfg.kernels.vda_fused


def load_models(model_root: str, device: str, vae_source: str = None,
                load_decoders: bool = True, *, memory_policy: str = "legacy",
                local_files_only: bool = False):
    """`model_root` holds the transformer; both decoders come from `vae_source`, default
    the release copy (H3_BASE: transformer/ vae/ audio_vae/, which the dense
    `checkpoint=null` render loads from too). Ranks that never decode pass
    `load_decoders=False` and get (transformer, None, None).

    The decoders stay on the CPU: `decode_and_save` brings them to the GPU when
    denoising is over. Their 10 GB next to the bf16 transformer, which is what the card
    holds until the fp8 conversion, is what an 80 GB card does not have."""
    model_root = resolve_inference_weights(model_root, local_files_only=local_files_only)
    if memory_policy in ("cpu_block_offload", "resident_weight_shards"):
        transformer = load_cpu_model(
            MiniMaxH3Transformer3DModel, model_root, subfolder="transformer",
            torch_dtype=torch.bfloat16, local_files_only=local_files_only)
    else:
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            model_root, subfolder="transformer", torch_dtype=torch.bfloat16,
            local_files_only=local_files_only
        ).to(device)
    transformer.eval().requires_grad_(False)

    if not load_decoders:
        return transformer, None, None

    vae_source = resolve_inference_weights(vae_source or H3_BASE, local_files_only=local_files_only)
    if memory_policy == "cpu_block_offload":
        vae = load_cpu_model(AutoencoderKLMiniMaxH3, vae_source, subfolder="vae",
                             torch_dtype=torch.float32, local_files_only=local_files_only)
        audio_vae = load_cpu_model(AutoencoderKLMiniMaxH3Audio, vae_source, subfolder="audio_vae",
                                   torch_dtype=torch.float32, local_files_only=local_files_only)
    else:
        vae = AutoencoderKLMiniMaxH3.from_pretrained(
            vae_source, subfolder="vae", local_files_only=local_files_only)
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            vae_source, subfolder="audio_vae", local_files_only=local_files_only)
    vae.eval()
    audio_vae.eval()
    return transformer, vae, audio_vae


def build_inference_model(cfg, device, *, load_decoders: bool = True,
                          log: bool = True, runtime=None, _shard_owner=False) -> InferenceModel:
    say = print if log else (lambda *a, **k: None)
    validate_memory(cfg)
    sharded = cfg.memory.policy == "resident_weight_shards"
    if sharded and not _shard_owner:
        if runtime is None:
            raise ValueError("resident_weight_shards requires an initialized Ulysses runtime")
        if load_decoders:
            raise ValueError("resident_weight_shards must defer decoder loading until release")
        return build_weight_sharded_model(cfg, runtime, log=log)
    cpu_offload = cfg.memory.policy == "cpu_block_offload"
    cpu_assembly = cpu_offload or sharded
    local_only = cfg.memory.local_files_only
    if cpu_offload:
        if torch.device(device).type != "cuda":
            raise ValueError("cpu_block_offload requires a CUDA execution device")
        if torch.distributed.is_initialized():
            raise ValueError("cpu_block_offload does not support distributed execution")
    assembly_device = "cpu" if cpu_assembly else device

    art = None
    if cfg.checkpoint:
        art = load_checkpoint(resolve_inference_weights(cfg.checkpoint, local_files_only=local_only))
        if art.metadata.get("truncated_blocks"):
            raise RuntimeError(f"{cfg.checkpoint} is a truncated smoke-test artifact")

    base_source = cfg.base_source or (
        art.model_spec["base"]["source"] if art else H3_BASE)
    transformer, vae, audio_vae = load_models(
        base_source, assembly_device, vae_source=cfg.vae_source, load_decoders=load_decoders,
        memory_policy=cfg.memory.policy, local_files_only=local_only)

    merged = 0
    if art:
        with torch.device("cpu") if cpu_assembly else nullcontext():
            apply_hybrid_attention_transform(
                transformer, art.model_spec["transforms"][0]["config"])

        branch_weights = {k: v for k, v in art.weights.items() if "lora_" not in k}
        lora_weights = {k: v for k, v in art.weights.items() if "lora_" in k}
        if cpu_assembly:
            assert_cpu_materialized(transformer)
            validate_branch_weights(transformer, branch_weights)
        loaded = load_model_weights(transformer, branch_weights)
        merged = merge_lora_state(transformer, lora_weights) if lora_weights else 0

        say(f"built from spec: {loaded} branch tensors, {merged} LoRA pairs merged",
            flush=True)
    else:
        say("DENSE BASE render: no checkpoint -- no transform, no branch", flush=True)

    transformer.to(assembly_device).eval().requires_grad_(False)

    external_records = []
    for xl in cfg.external_loras:
        state, scale, ranks = load_external_lora(xl.path, xl.alpha)
        pairs = merge_lora_state(transformer, state, scale)

        say(f"external LoRA {xl.path}: {pairs} pairs merged (ranks {ranks}, "
            f"scale {scale:g})", flush=True)
        external_records.append({"path": os.path.abspath(xl.path), "pairs": pairs,
                                 "ranks": ranks, "scale": scale})

    # ---- overlays, in order ----
    is_hybrid = next(iter_hybrids(transformer), None) is not None
    apply_vda_candidates(transformer, cfg)

    for attn in iter_hybrids(transformer):
        attn.teacher_mode = cfg.behavior.teacher_mode
        for prm in attn.parameters():
            if prm.dtype == torch.float32:
                prm.data = prm.data.to(torch.bfloat16)

    softmax_backend = None
    if is_hybrid:
        if cfg.kernels.inference_kernels:
            set_inference_mode(transformer, True)

        softmax_backend = set_softmax_backend(transformer, cfg.kernels.softmax_backend)
        say(f"window softmax: {softmax_backend}", flush=True)

    ablated = (apply_ablation(transformer, dict(cfg.ablation.overrides))
               if cfg.ablation.enabled and cfg.ablation.overrides else [])
    if ablated:
        say(f"ABLATION ACTIVE: {ablated} -- this output is a study, not a sample",
            flush=True)

    fp8_linears = 0
    if cfg.precision.fp8.enabled:
        fp8_linears = convert_linear_to_fp8(
            transformer, skip_end_blocks=cfg.precision.fp8.skip_end_blocks)
        say(f"fp8: {fp8_linears} Linears quantised", flush=True)
        # The bf16 weights the conversion freed, about 20 GiB, stay in PyTorch's cache,
        # where only PyTorch can reuse them. NCCL allocates its buffers outside that cache
        # at the first collective of each communicator, so on an 80 GB card the eight-GPU
        # render ran out of memory in its first all-to-all. Hand them back to the driver.
        torch.cuda.empty_cache()

    offload = None
    if cpu_offload:
        assert_cpu_materialized(transformer)
        offload = BlockOffloadController(transformer, device)

    return InferenceModel(transformer=transformer, vae=vae, audio_vae=audio_vae,
                          artifact=art, is_hybrid=is_hybrid, merged_lora_pairs=merged,
                          external_loras=external_records, ablated=ablated,
                          fp8_linears=fp8_linears, softmax_backend=softmax_backend,
                          offload=offload)


def build_weight_sharded_model(cfg, runtime, *, log=True):
    """Rank zero full CPU baseline assembly, other ranks header/meta only.

    This deliberately does NOT claim strict block-bounded host assembly. All ranks
    validate actual headers before the owner reads payloads. Root assembly errors
    are broadcast before any peer enters weight collectives.
    """
    import gc
    import resource
    import time
    import torch.distributed as dist
    from src.inference.utils.weight_sharding import build_meta_model, ResidentWeightShards
    from src.inference.utils.paper_runtime import PaperRuntime, convert_paper_projections

    controller_class = getattr(runtime, "weight_controller_factory", ResidentWeightShards)
    if runtime.world_size != cfg.memory.shard_world_size:
        raise ValueError(f"expected {cfg.memory.shard_world_size} ranks, got {runtime.world_size}")
    paper_class = getattr(runtime, "paper_factory", PaperRuntime)
    paper = paper_class.create(cfg, runtime) if paper_enabled(cfg) else None
    runtime.paper_runtime = paper
    started = time.perf_counter()
    error = None
    try:
        meta, art = build_meta_model(cfg)
        apply_vda_candidates(meta, cfg)
        if paper is not None:
            with paper.timed("mxfp8_meta_conversion"):
                paper.conversion = convert_paper_projections(meta, cfg)
        expected = controller_class(meta, runtime, scatter_chunk_bytes=cfg.memory.scatter_chunk_mib << 20,
                                        distribution=cfg.memory.shard_distribution,
                                        prefetch=cfg.memory.weight_prefetch)
    except Exception as exc:
        error = f"rank {runtime.rank}: {type(exc).__name__}: {exc}"
    errors = [None] * runtime.world_size
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"metadata validation failed: {errors}")
    metadata_seconds = time.perf_counter() - started
    # Mask build can have quadratic temporary storage; do it before persistent
    # root/shards, not during the first gathered-block attention call.
    from src.inference.utils.ulysses import prebuild_prompt_masks
    mask_started = time.perf_counter()
    with paper.guard.local_phase("request_planning") if paper is not None else nullcontext():
        geometry = (None if getattr(paper, "persistent", False) else
                    prebuild_prompt_masks(cfg, meta, runtime.device, paper=paper))
    mask_seconds = time.perf_counter() - mask_started
    if paper is not None:
        # Include prompt/layout construction and any original-mask preparation,
        # not just the inner packed-plan/schedule helper interval.
        paper.preparation["prompt_layout_and_masks"] += max(
            0.0, mask_seconds - paper.preparation["schedule_and_window_plan"])
    expected_manifest = expected.manifest()
    model = None
    status = [None]
    if runtime.is_main:
        assembly_started = time.perf_counter()
        assembly_cuda_before = torch.cuda.memory_allocated(runtime.device)
        try:
            model = build_inference_model(cfg, "cpu", load_decoders=False, log=log, _shard_owner=True)
            if paper is not None:
                if getattr(paper, "persistent", False):
                    paper.retain_adaln_sources(model.transformer)
                else:
                    paper.prepare_adaln(model.transformer)
                with paper.timed("mxfp8_cpu_conversion"):
                    records = convert_paper_projections(model.transformer, cfg)
                if records != paper.conversion:
                    raise ValueError("CPU/meta paper conversion records differ")
            controller = controller_class(model.transformer, runtime,
                                               scatter_chunk_bytes=cfg.memory.scatter_chunk_mib << 20,
                                               distribution=cfg.memory.shard_distribution,
                                               prefetch=cfg.memory.weight_prefetch)
            if controller.manifest() != expected_manifest:
                raise ValueError("assembled CPU graph keys/shapes/dtypes/aliases differ from header/meta graph")
            # Checkpoint payload is not the resident backing; no decoder was loaded.
            model.artifact.weights.clear()
            del meta, expected
            gc.collect()
            status[0] = dict(ok=True, assembly_seconds=time.perf_counter() - assembly_started,
                             assembly_cuda_allocated_delta=torch.cuda.memory_allocated(runtime.device) - assembly_cuda_before,
                             root_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             merged_lora_pairs=model.merged_lora_pairs,
                             external_loras=model.external_loras)
        except Exception as exc:
            status[0] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
    dist.broadcast_object_list(status, src=0)
    if not status[0]["ok"]:
        raise RuntimeError(f"rank0 CPU assembly failed: {status[0]['error']}")
    if not runtime.is_main:
        model = InferenceModel(meta, None, None, art, True,
                               merged_lora_pairs=status[0]["merged_lora_pairs"],
                               external_loras=status[0]["external_loras"], softmax_backend="flex")
        controller = expected
    model.startup_metrics = status[0]
    model.paper_runtime = paper
    if paper is not None and not getattr(paper, "persistent", False):
        paper.transport_adaln()
    distribute_started = time.perf_counter()
    model.weight_shards = controller
    runtime.weight_shards = controller
    controller.distribute()
    model.transformer._weight_shards = controller
    torch.cuda.synchronize(runtime.device)
    model.startup_metrics = dict(status[0], distribution_seconds=time.perf_counter() - distribute_started,
                                total_seconds=time.perf_counter() - started,
                                metadata_seconds=metadata_seconds,
                                mask_prepare_seconds=mask_seconds, mask_geometry=geometry)
    gc.collect()
    torch.cuda.empty_cache()
    return model


def release_sharded_denoiser(model, runtime):
    """Collective staging boundary for decoder integrations, called on every rank.

    After return, all model parameters/buffers are meta, root/shard storages are
    gone, and ``load_released_decoders`` or a parent distributed decoder may run.
    """
    import gc
    if model.weight_shards is None:
        raise ValueError("not a resident weight-sharded model")
    if model.paper_runtime is not None:
        model.paper_runtime.close_tables()
    model.weight_shards.release()
    torch.cuda.synchronize(runtime.device)
    runtime.barrier()
    gc.collect()
    torch.cuda.empty_cache()


def load_released_decoders(cfg, model):
    """Rank-zero sequential-decode fallback; call only after collective release."""
    if model.weight_shards is None or not model.weight_shards.released:
        raise RuntimeError("decoder loading requires completely released denoiser")
    source = resolve_inference_weights(cfg.vae_source or H3_BASE, local_files_only=True)
    model.vae = load_cpu_model(AutoencoderKLMiniMaxH3, source, subfolder="vae",
                               torch_dtype=torch.float32, local_files_only=True).eval()
    model.audio_vae = load_cpu_model(AutoencoderKLMiniMaxH3Audio, source, subfolder="audio_vae",
                                     torch_dtype=torch.float32, local_files_only=True).eval()
    return model.vae, model.audio_vae


def render_record(cfg, model: InferenceModel) -> Dict[str, Any]:
    """The record next to the mp4: checkpoint identity, the overlay, the git commit and
    the ACTUAL kernel state. Entrypoints add their own sections (timings, parallel)."""
    art = model.artifact
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()

    return {
        "checkpoint": resolve_inference_weights(cfg.checkpoint, local_files_only=cfg.memory.local_files_only) if cfg.checkpoint else None,
        "checkpoint_head_sha256": (checkpoint_head_sha256(resolve_inference_weights(cfg.checkpoint, local_files_only=cfg.memory.local_files_only))
                                   if cfg.checkpoint else None),
        "model_spec_base_hash": art.model_spec["base"]["config_hash"] if art else None,
        "git_commit": commit,
        "overlay": resolved_dict(cfg),
        "ablation_active": model.ablated,
        "external_loras": model.external_loras,
        "fp8_linears": model.fp8_linears,
        "flex_backend": flex_latch_state(),
        "softmax_backend": {"resolved": model.softmax_backend},
        "merged_lora_pairs": model.merged_lora_pairs,
        "paper_stack": model.paper_runtime.inventory() if model.paper_runtime is not None else None,
    }


def latents_path(cfg) -> str:
    """Where `render.save_latents` puts the pre-VAE tensors: mirrors the mp4's path
    under results/ into `render.latents_root`."""
    out_abs = os.path.abspath(cfg.render.out)
    rel_out = os.path.relpath(out_abs, os.path.abspath("results"))

    if rel_out == os.pardir or rel_out.startswith(os.pardir + os.sep):
        rel_out = os.path.basename(out_abs)

    return os.path.join(cfg.render.latents_root, rel_out + ".latents.pt")


def write_json(record: Dict[str, Any], path: str) -> None:
    import json

    with open(path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
