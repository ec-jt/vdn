"""Encode t2va prompts with the Qwen3-VL conditioner and cache the results.

Replicates `MiniMaxH3TextEncoderStep` exactly: the prompt verbatim, no chat template, no
special tokens, hidden state after decoder layer 50, tags all text. Run once on one GPU;
inference then reuses the cached file and never loads the 62 GB conditioner again.

    python src/inference/encode_prompt.py --prompt "..." --out prompts/my_prompt.pt
    python src/inference/encode_prompt.py --jsonl prompts.jsonl --take 4 \
        --out_dir prompts --name prompt

The batch form exists because loading the conditioner costs far more than encoding: four
separate --prompt runs would pay the 62 GB load four times. It also keeps a prompt set
reproducible -- the file it reads is the dataset as published, not text pasted onto a
command line.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

# Also support the documented direct-script invocation without PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import upstream_snapshot
TEXT_ENCODER_LAYER = 50
TEXT_TAG = 1


def local_roots(processor_root, text_encoder_root):
    """Fail before model loading, never interpret a missing directory as a Hub ID."""
    if not processor_root or not text_encoder_root:
        raise ValueError("local mode requires explicit --processor_root and --text_encoder_root")
    processor, model = Path(processor_root).resolve(), Path(text_encoder_root).resolve()
    for root in (processor, model):
        if not root.is_dir():
            raise ValueError(f"existing local directory required: {root}")
    if not (processor / "tokenizer_config.json").is_file():
        raise ValueError(f"missing local tokenizer_config.json in {processor}")
    config = model / "config.json"
    if not config.is_file():
        raise ValueError(f"missing local config.json in {model}")
    if json.loads(config.read_text()).get("quantization_config"):
        raise ValueError("local encoder mode requires unquantized BF16-compatible weights")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        if not shards:
            raise ValueError(f"empty local weight index: {index}")
        for shard in shards:
            path = (model / shard).resolve()
            if not path.is_relative_to(model) or not path.is_file():
                raise ValueError(f"missing or out-of-root local weight shard: {shard}")
    elif not (model / "model.safetensors").is_file():
        raise ValueError(f"no local safetensors encoder weights in {model}; downloads are disabled")
    return str(processor), str(model)


def offload_encoder(text_encoder, device):
    """Accelerate CPU storage with whole text layers, not whole-model CUDA loading.

    No automatic map, disk offload, prefetch stream, or distributed execution.
    Embeddings and other leaves offload individually; unused vision/head weights
    stay CPU-backed. Nonpersistent rotary buffers are small and may stay on CUDA.
    """
    from accelerate import dispatch_model
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    layers = text_encoder.model.language_model.layers
    if not layers or not all(isinstance(layer, Qwen3VLTextDecoderLayer) for layer in layers):
        raise ValueError("unsupported encoder layout: expected Qwen3VLTextDecoderLayer blocks")
    if any(p.device.type != "cpu" for p in text_encoder.parameters()):
        raise ValueError("offload setup requires a CPU-loaded encoder")
    return dispatch_model(
        text_encoder,
        device_map={"": "cpu"},
        main_device=device,
        # Accelerate 1.14's implicit CPU extraction does not match the empty
        # root prefix. Supply the complete CPU tensor mapping explicitly.
        state_dict=text_encoder.state_dict(),
        force_hooks=True,
        offload_buffers=True,
        preload_module_classes=["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"],
    )


def encode(processor, text_encoder, prompt, out, device, *, max_tokens=None,
           no_overwrite=False):
    token_ids = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if max_tokens is not None and not 0 < len(token_ids) <= max_tokens:
        raise ValueError(f"prompt has {len(token_ids)} tokens; allowed range is 1..{max_tokens}")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
    )
    with torch.no_grad():
        outputs = text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    prompt_embeds = outputs.hidden_states[TEXT_ENCODER_LAYER][0].to(torch.bfloat16).cpu()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "xb" if no_overwrite else "wb") as f:
        torch.save({
            "prompt": prompt,
            "prompt_embeds": prompt_embeds,  # (L, 5120) bf16
            "text_token_tags": torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long),
        }, f)
    print(f"wrote {out}: {len(token_ids)} tokens, embeds {tuple(prompt_embeds.shape)}",
          flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str)
    p.add_argument("--out", type=str)
    p.add_argument("--jsonl", type=str,
                   help="one JSON object per line with a 'prompt' field; batch mode")
    p.add_argument("--take", type=int, default=4, help="jsonl: encode the first N rows")
    p.add_argument("--rows", type=int, nargs="*", default=None,
                   help="jsonl: encode exactly these row indices instead of the first "
                        "--take; output names keep the ORIGINAL index "
                        "(<name>_<row>.pt), so a prompt's identity is stable no matter "
                        "which subset a batch encodes")
    p.add_argument("--out_dir", type=str, default="prompts")
    p.add_argument("--name", type=str, default="prompt",
                   help="jsonl: outputs are <out_dir>/<name>_<i>.pt")
    p.add_argument("--model_root", type=str, default=None,
                   help="a MiniMax-H3 snapshot with processor/ and text_encoder/; "
                        "default: downloaded from the Hub on first use")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--local_only", action="store_true",
                   help="offline only; requires explicit processor and encoder directories")
    p.add_argument("--processor_root", help="existing local processor directory (not its parent)")
    p.add_argument("--text_encoder_root", help="existing local encoder directory (not its parent)")
    p.add_argument("--cpu_offload", action="store_true",
                   help="local-only whole-decoder-layer Accelerate CPU offload on one CUDA device")
    p.add_argument("--max_tokens", type=int, default=1024,
                   help="CPU-offload prompt limit; reject rather than truncate (default: 1024)")
    p.add_argument("--no_overwrite", action="store_true", help="refuse existing output caches")
    args = p.parse_args(argv)
    if not args.jsonl and not (args.prompt and args.out):
        p.error("pass --jsonl, or both --prompt and --out")

    local = args.local_only or args.cpu_offload
    if local:
        # Set before importing Transformers or calling any Hub-aware loader.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if args.model_root:
            p.error("local mode uses --processor_root and --text_encoder_root, not --model_root")
        try:
            processor_root, encoder_root = local_roots(args.processor_root, args.text_encoder_root)
        except (ValueError, OSError, TypeError) as exc:
            p.error(str(exc))
    elif args.processor_root or args.text_encoder_root:
        p.error("explicit processor/encoder roots require --local_only or --cpu_offload")
    if args.cpu_offload:
        try:
            device = torch.device(args.device)
        except (RuntimeError, ValueError) as exc:
            p.error(str(exc))
        if device.type != "cuda" or device.index is None or args.max_tokens < 1:
            p.error("CPU offload requires an explicit cuda:N device and positive --max_tokens")

    if args.jsonl:
        with open(args.jsonl) as f:
            all_rows = [json.loads(l) for l in f]
        picked = (list(enumerate(all_rows))[:args.take] if args.rows is None
                  else [(i, all_rows[i]) for i in args.rows])
        jobs = [(r["prompt"], os.path.join(args.out_dir, f"{args.name}_{i}.pt"))
                for i, r in picked]
    else:
        jobs = [(args.prompt, args.out)]
    if not jobs:
        p.error("no prompts selected")
    for prompt, out in jobs:
        if not isinstance(prompt, str) or not prompt:
            p.error("each prompt must be a nonempty string")
        if args.no_overwrite and os.path.lexists(out):
            p.error(f"refusing to overwrite existing cache: {out}")

    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    if local:
        processor = Qwen3VLProcessor.from_pretrained(
            processor_root, local_files_only=True, trust_remote_code=False
        )
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            encoder_root, dtype=torch.bfloat16, device_map={"": "cpu"},
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
        )
    else:
        if args.model_root is None:
            args.model_root = upstream_snapshot("processor", "text_encoder")
        processor = Qwen3VLProcessor.from_pretrained(args.model_root, subfolder="processor")
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_root, subfolder="text_encoder", dtype=torch.bfloat16
        )
    text_encoder.eval().requires_grad_(False)
    if args.cpu_offload:
        text_encoder = offload_encoder(text_encoder, args.device)
    else:
        text_encoder = text_encoder.to(args.device)

    for prompt, out in jobs:
        encode(processor, text_encoder, prompt, out, args.device,
               max_tokens=args.max_tokens if args.cpu_offload else None,
               no_overwrite=args.no_overwrite)
    # This is a one-shot process, not an encoder service. Process exit releases
    # hooks, CPU storage and the CUDA context before a render is scheduled.


if __name__ == "__main__":
    main()
