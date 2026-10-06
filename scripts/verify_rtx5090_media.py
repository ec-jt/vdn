"""CPU-only decode, contact sheet, and optional saved-latent verification."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--latents", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--contact", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = {"path": str(args.media), "streams": []}
    indices = sorted(set(np.linspace(0, args.frames - 1, 8, dtype=int).tolist()))
    sheet = Image.new("RGB", (4 * 336, 2 * 216), "#181818")
    draw = ImageDraw.Draw(sheet)
    with av.open(str(args.media)) as container:
        for stream in container.streams:
            context = stream.codec_context
            report["streams"].append({
                "type": stream.type, "codec": context.name,
                "duration": float(stream.duration * stream.time_base)
                if stream.duration is not None else None,
                "frames": stream.frames,
                "width": getattr(context, "width", None),
                "height": getattr(context, "height", None),
                "sample_rate": getattr(context, "sample_rate", None),
                "rate": str(stream.average_rate) if stream.type == "video" else None,
                "layout": context.layout.name if stream.type == "audio" else None,
            })
        assert len(container.streams.video) == len(container.streams.audio) == 1
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (1344, 768)
        assert stream.average_rate == 24
        assert stream.codec_context.name == "h264"
        count = 0
        for index, frame in enumerate(container.decode(video=0)):
            count += 1
            if index in indices:
                cell = indices.index(index)
                x, y = (cell % 4) * 336, (cell // 4) * 216
                sheet.paste(frame.to_image().resize((336, 192)), (x, y))
                draw.text((x + 8, y + 197), f"frame {index} / {index / 24:.3f}s", fill="white")
        report["decoded_video_frames"] = count
        assert count == args.frames, (count, args.frames)
    with av.open(str(args.media)) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "aac"
        assert stream.codec_context.sample_rate == 32000
        assert stream.codec_context.layout.name == "stereo"
        audio = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
        report["audio_samples"] = list(audio.shape)
        report["audio_finite"] = bool(np.isfinite(audio).all())
        report["audio_rms"] = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        assert report["audio_finite"] and report["audio_rms"] > 0
    if args.latents:
        tensors = torch.load(args.latents, map_location="cpu", weights_only=True)
        assert set(tensors) == {"video", "audio"}
        report["latents_path"] = str(args.latents)
        report["latents"] = {}
        for name, tensor in tensors.items():
            info = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                    "finite": bool(torch.isfinite(tensor).all()),
                    "min": tensor.min().item(), "max": tensor.max().item()}
            report["latents"][name] = info
            assert info["finite"], name
    args.contact.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.contact)
    report["contact_sheet"] = str(args.contact)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
