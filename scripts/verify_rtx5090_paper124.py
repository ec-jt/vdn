"""CPU-only final-latent and FULL decoded-stream comparison, run after timed jobs.

No model imports, GPU calls, contact sheets, frame subsampling, or media writes.
SHA256 covers logical contiguous tensor bytes (not the torch archive), RGB24
in frame/row/column/channel order, and little-endian float32 audio in
sample/channel order. AAC decoder padding is retained, never trimmed.
"""

import argparse
from contextlib import closing
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import sys


def limit_cpu_threads():
    # Called before numeric-library imports in the CLI. No CUDA probing.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"


class DeltaAccumulator:
    """Incremental, float64 arithmetic with exact source-byte hashes.

    Numeric mismatches count elements, not bytes. Signed zeros are numerically
    equal but byte-distinct. Nonfinite inputs invalidate all numeric metrics.
    A zero reference norm gives relL2=0 for zero error, otherwise JSON null
    with relative_l2_defined=false (never a spurious passing epsilon gate).
    """

    def __init__(self):
        self.hashes = [hashlib.sha256(), hashlib.sha256()]
        self.counts = [0, 0]
        self.byte_counts = [0, 0]
        self.finite = [True, True]
        self.squared_error = 0.0
        self.reference_squared = 0.0
        self.maxabs = 0.0
        self.mismatched = 0
        self.aligned = True

    def update(self, reference, candidate, raw_reference=None, raw_candidate=None):
        import numpy as np

        arrays = [None if x is None else np.asarray(x).reshape(-1)
                  for x in (reference, candidate)]
        for i, (array, raw) in enumerate(zip(arrays, (raw_reference, raw_candidate))):
            if array is None:
                continue
            data = array.tobytes(order="C") if raw is None else raw
            self.hashes[i].update(data)
            self.byte_counts[i] += len(data)
            self.counts[i] += array.size
            self.finite[i] = self.finite[i] and bool(np.isfinite(array).all())
        if any(x is None for x in arrays):
            self.aligned = False
            return
        a, b = arrays
        if a.size != b.size:
            # Both complete hashes are still collected; deltas are undefined.
            self.aligned = False
            return
        self.mismatched += int(np.count_nonzero(a != b))
        if not all(self.finite):
            return
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            difference = b - a
            self.squared_error += float(np.sum(difference * difference))
            self.reference_squared += float(np.sum(a * a))
            if a.size:
                self.maxabs = max(self.maxabs, float(np.max(np.abs(difference))))

    def result(self):
        same_count = self.counts[0] == self.counts[1]
        valid = (same_count and self.aligned and self.counts[0] > 0 and all(self.finite)
                 and math.isfinite(self.squared_error)
                 and math.isfinite(self.reference_squared)
                 and math.isfinite(self.maxabs))
        relative = None
        if valid:
            if self.reference_squared > 0:
                relative = math.sqrt(self.squared_error) / math.sqrt(self.reference_squared)
                if not math.isfinite(relative):
                    relative = None
            elif self.squared_error == 0:
                relative = 0.0
        return {
            "reference": {"sha256": self.hashes[0].hexdigest(),
                          "elements": self.counts[0], "bytes": self.byte_counts[0],
                          "finite": self.finite[0]},
            "candidate": {"sha256": self.hashes[1].hexdigest(),
                          "elements": self.counts[1], "bytes": self.byte_counts[1],
                          "finite": self.finite[1]},
            "same_element_count": same_count,
            "byte_exact": (same_count and self.byte_counts[0] == self.byte_counts[1]
                           and self.hashes[0].digest() == self.hashes[1].digest()),
            "finite": all(self.finite), "numeric_metrics_valid": valid,
            "relative_l2": relative, "relative_l2_defined": relative is not None,
            "rmse": math.sqrt(self.squared_error / self.counts[0]) if valid else None,
            "maxabs": self.maxabs if valid else None,
            "mismatched_count": self.mismatched if same_count and self.aligned else None,
        }


def fixed_chunks(arrays, chunk_elements):
    """Reblock arbitrary decoder boundaries using bounded storage."""
    import numpy as np

    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    pending = None
    used = 0
    for array in arrays:
        flat = np.asarray(array).reshape(-1)
        if pending is None:
            pending = np.empty(chunk_elements, dtype=flat.dtype)
        if flat.dtype != pending.dtype:
            raise ValueError("stream dtype changed")
        for start in range(0, flat.size, chunk_elements):
            part = flat[start:start + chunk_elements]
            offset = 0
            while offset < part.size:
                count = min(chunk_elements - used, part.size - offset)
                pending[used:used + count] = part[offset:offset + count]
                used += count
                offset += count
                if used == chunk_elements:
                    yield pending.copy()
                    used = 0
    if used:
        yield pending[:used].copy()


def compare_streams(reference, candidate, chunk_elements):
    delta = DeltaAccumulator()
    for a, b in zip_longest(fixed_chunks(reference, chunk_elements),
                           fixed_chunks(candidate, chunk_elements)):
        delta.update(a, b)
    return delta.result()


def compare_tensors(reference, candidate, chunk_elements):
    import torch

    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    for tensor in (reference, candidate):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError("expected a CPU tensor")
        if not tensor.is_floating_point() or tensor.numel() == 0:
            raise ValueError("expected a nonempty floating-point latent")
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("latent shape/dtype mismatch")
    a = reference.detach().contiguous().reshape(-1)
    b = candidate.detach().contiguous().reshape(-1)
    delta = DeltaAccumulator()
    for start in range(0, a.numel(), chunk_elements):
        left, right = a[start:start + chunk_elements], b[start:start + chunk_elements]
        # Preserve BF16 bytes for hashing; float64 conversion is chunk bounded.
        delta.update(left.double().numpy(), right.double().numpy(),
                     left.view(torch.uint8).numpy().tobytes(),
                     right.view(torch.uint8).numpy().tobytes())
    return {"shape": list(reference.shape), "dtype": str(reference.dtype),
            **delta.result()}


def decode_arrays(path, kind, info):
    import av
    import numpy as np

    with av.open(str(path), options={"threads": "1"}) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise ValueError(f"{path}: expected one video and one audio stream")
        stream = (container.streams.video if kind == "video" else container.streams.audio)[0]
        stream.codec_context.thread_count = 1
        context = stream.codec_context
        info.update(codec=context.name, decoded_frames=0)
        if kind == "video":
            info.update(width=context.width, height=context.height,
                        fps=str(stream.average_rate), array_format="rgb24")
            if (context.width, context.height, stream.average_rate) != (1344, 768, 24):
                raise ValueError(f"{path}: expected 1344x768 at 24 fps")
            if context.name != "h264":
                raise ValueError(f"{path}: expected h264")
        else:
            info.update(sample_rate=context.sample_rate, layout=context.layout.name,
                        channels=context.channels, decoded_samples_per_channel=0,
                        array_format="interleaved float32 little-endian")
            if (context.sample_rate, context.layout.name, context.channels) != (32000, "stereo", 2):
                raise ValueError(f"{path}: expected stereo 32000 Hz")
            if context.name != "aac":
                raise ValueError(f"{path}: expected aac")
        for frame in container.decode(stream):
            info["decoded_frames"] += 1
            if kind == "video":
                if (frame.width, frame.height) != (1344, 768):
                    raise ValueError("decoded video geometry changed")
                yield frame.to_ndarray(format="rgb24")
            else:
                if (frame.sample_rate != 32000 or frame.layout.name != "stereo"
                        or frame.format.name not in ("flt", "fltp")):
                    raise ValueError("expected native float32 stereo AAC decode")
                array = frame.to_ndarray()
                if frame.format.is_planar:
                    array = array.T
                array = np.ascontiguousarray(array.reshape(-1, 2), dtype="<f4")
                if array.shape[0] != frame.samples:
                    raise ValueError("audio sample count mismatch")
                info["decoded_samples_per_channel"] += frame.samples
                yield array
        if kind == "video" and info["decoded_frames"] != 124:
            raise ValueError(f"{path}: decoded {info['decoded_frames']} frames, expected 124")
        if kind == "audio":
            samples = info["decoded_samples_per_channel"]
            info["decoded_duration_seconds"] = samples / 32000
            # AAC tail padding is included. Do not truncate to video duration.
            if samples == 0 or abs(samples / 32000 - 124 / 24) > 1024 / 32000:
                raise ValueError(f"{path}: audio duration outside one AAC frame of video")
        info["complete_decode"] = True


def load_metrics(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    render = data["overlay"]["render"]
    if render["num_frames"] != 124 or render["num_steps"] != 8:
        raise ValueError(f"{path}: expected 124 frames / 8 NFE")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "render": render, "checkpoint_head_sha256": data.get("checkpoint_head_sha256"),
            "model_spec_base_hash": data.get("model_spec_base_hash"),
            "parallel": data.get("parallel"), "timings": data.get("timings")}


def candidate_passes(latents, media, expectation, threshold=1e-3):
    comparisons = list(latents.values()) + list(media.values())
    if not all(item["finite"] and item["numeric_metrics_valid"] for item in comparisons):
        return False
    if expectation == "exact":
        return all(item["byte_exact"] for item in comparisons)
    if expectation != "fused":
        raise ValueError("expectation must be exact or fused")
    return all(item["relative_l2"] is not None and item["relative_l2"] <= threshold
               for item in latents.values())


def verify_candidate(args, label, expectation, reference_latents, reference_metrics):
    import torch

    latent_path = args.latents_root / f"{label}.mp4.latents.pt"
    candidate = torch.load(latent_path, map_location="cpu", weights_only=True)
    if not isinstance(candidate, dict) or set(candidate) != {"video", "audio"}:
        raise ValueError(f"{latent_path}: expected exactly video/audio tensor keys")
    metrics = load_metrics(args.results_root / f"{label}.mp4.inference.json")
    for field in ("seed", "prompt_file", "video_shift", "audio_shift", "num_steps", "num_frames"):
        if metrics["render"][field] != reference_metrics["render"][field]:
            raise ValueError(f"render identity mismatch: {field}")
    for field in ("checkpoint_head_sha256", "model_spec_base_hash"):
        if not metrics[field] or metrics[field] != reference_metrics[field]:
            raise ValueError(f"model identity mismatch: {field}")
    latents = {key: compare_tensors(reference_latents[key], candidate[key], args.chunk_elements)
               for key in ("video", "audio")}
    del candidate
    media = {}
    for kind in ("video", "audio"):
        ref_info, candidate_info = {}, {}
        with closing(decode_arrays(args.results_root / f"{args.reference}.mp4", kind, ref_info)) as a:
            with closing(decode_arrays(args.results_root / f"{label}.mp4", kind, candidate_info)) as b:
                delta = compare_streams(a, b, args.chunk_elements)
        delta.update(reference_stream=ref_info, candidate_stream=candidate_info)
        media[kind] = delta
    return {"label": label, "expectation": expectation, "latents_path": str(latent_path),
            "metrics": metrics, "latents": latents, "decoded_media": media,
            "passed": candidate_passes(latents, media, expectation)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", nargs="+", required=True, help="one or more candidate labels")
    parser.add_argument("--expect", action="append", required=True, metavar="LABEL=exact|fused",
                        help="repeat once per candidate; exact covers latent AND decoded bytes")
    parser.add_argument("--output", type=Path, required=True, help="exclusive-create JSON report")
    parser.add_argument("--latents-root", type=Path, default=Path("artifacts/latents"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("artifacts/rtx5090/paper-bf16-20261005/model-protocol.json"))
    parser.add_argument("--chunk-elements", type=int, default=262144)
    args = parser.parse_args(argv)
    if args.chunk_elements <= 0:
        parser.error("--chunk-elements must be positive")
    labels = [args.reference] + args.candidate
    if any(not label or label in (".", "..") or "/" in label or "\\" in label for label in labels):
        parser.error("labels must be plain filenames, not paths")
    if len(set(args.candidate)) != len(args.candidate):
        parser.error("duplicate candidate labels")
    expectations = {}
    for value in args.expect:
        label, sep, expectation = value.partition("=")
        if not sep or label in expectations or expectation not in ("exact", "fused"):
            parser.error("each --expect must be a unique LABEL=exact or LABEL=fused")
        expectations[label] = expectation
    if set(expectations) != set(args.candidate):
        parser.error("provide exactly one --expect for every candidate")
    args.expectations = expectations
    return args


def main(argv=None):
    args = parse_args(argv)
    limit_cpu_threads()
    # Fail on existing output before loading tensors or decoding media. Never mkdir
    # or overwrite an artifact. An interrupted run leaves an unmistakable empty file.
    with args.output.open("x") as output:
        report = {"schema_version": 1, "reference_label": args.reference,
                  "fused_latent_relative_l2_max": 1e-3, "candidates": [], "passed": False,
                  "scope": "final trajectory tensors and complete decoded arrays; not per-step latents",
                  "hash_encoding": {"latents": "native dtype C-order bytes, no metadata/header",
                                    "video": "uint8 RGB24 T,H,W,C",
                                    "audio": "little-endian float32 samples,channels; no trimming"},
                  "relative_l2_definition": "norm(candidate-reference)/norm(reference)",
                  "chunk_elements": args.chunk_elements, "cpu_threads": 1,
                  "performance_gate_evaluated": False, "quality_guarantee": False}
        try:
            import av
            import numpy as np
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            report["versions"] = {"torch": str(torch.__version__), "numpy": np.__version__,
                                  "pyav": av.__version__, "ffmpeg_libraries": av.library_versions,
                                  "byteorder": sys.byteorder}
            protocol_raw = args.protocol.read_bytes()
            protocol = json.loads(protocol_raw)
            expected_media = dict(frames=124, width=1344, height=768, fps=24,
                                  audio_channels=2, audio_rate=32000)
            if protocol["media"] != expected_media:
                raise ValueError("fixed protocol media specification mismatch")
            if protocol["fused_numerical_gate"]["finite_all"] is not True:
                raise ValueError("fixed protocol must require finite tensors")
            for kind in ("video", "audio"):
                if protocol["fused_numerical_gate"][f"{kind}_latent_relative_l2_max"] != 1e-3:
                    raise ValueError("fixed protocol numerical gate mismatch")
            report["protocol"] = {"path": str(args.protocol),
                                  "sha256": hashlib.sha256(protocol_raw).hexdigest(),
                                  "contents": protocol}
            path = args.latents_root / f"{args.reference}.mp4.latents.pt"
            reference = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(reference, dict) or set(reference) != {"video", "audio"}:
                raise ValueError("reference must have exactly video/audio tensor keys")
            report["reference_latents_path"] = str(path)
            metrics = load_metrics(args.results_root / f"{args.reference}.mp4.inference.json")
            report["reference_metrics"] = metrics
            for label in args.candidate:
                expectation = args.expectations[label]
                try:
                    item = verify_candidate(args, label, expectation, reference, metrics)
                except Exception as exc:
                    item = {"label": label, "expectation": expectation, "passed": False,
                            "error": f"{type(exc).__name__}: {exc}"}
                report["candidates"].append(item)
            report["passed"] = all(item["passed"] for item in report["candidates"])
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
