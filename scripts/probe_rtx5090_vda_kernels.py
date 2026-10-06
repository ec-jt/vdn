"""Local-only, GPU-0 bounded VDA operator and real-weight branch probe.

Synthetic correlated hidden states, NOT captured trajectory activations. Loads actual
branch and base QKV weights; adapters are not merged. No full denoising claim.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from safetensors import safe_open

from src.models.linear_attention.branch import BidirectionalLinearBranch, _HeadSliceSepConv
from src.models.linear_attention.features import _temporal_shift
from src.models.linear_attention.kernels import linear_epilogue, vda_bf16_epilogue
from src.models.linear_attention.scan import frame_statistics, gather_linear_state, _run_scans
from src.models.softmax_attention.window import window_bounds


def errors(actual, expected):
    a, b = actual.double(), expected.double()
    d = a - b
    return dict(rel_l2=(d.norm() / b.norm().clamp_min(1e-30)).item(),
                rmse=d.square().mean().sqrt().item(), max_abs=d.abs().max().item(),
                exact=torch.equal(actual, expected))


def timing(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    wall, cuda = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t) * 1000)
        cuda.append(start.elapsed_time(end))
    return dict(wall_ms=wall, cuda_ms=cuda, wall_median_ms=statistics.median(wall),
                cuda_median_ms=statistics.median(cuda))


def alternating_timing(ref, candidate, repeats):
    """Alternate A/B and B/A sample order after warming both implementations."""
    for _ in range(3):
        ref()
        candidate()
    samples = {name: dict(wall_ms=[], cuda_ms=[]) for name in ('baseline','candidate')}
    for i in range(repeats):
        order = [('baseline', ref), ('candidate', candidate)]
        for name, fn in order[::1 if i % 2 == 0 else -1]:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples[name]['wall_ms'].append((time.perf_counter()-t)*1000)
            samples[name]['cuda_ms'].append(start.elapsed_time(end))
    for values in samples.values():
        values['wall_median_ms'] = statistics.median(values['wall_ms'])
        values['cuda_median_ms'] = statistics.median(values['cuda_ms'])
    return samples


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='../models/OpenVDN/vdn-minimax-h3/stage-dmd-step-250')
    p.add_argument('--base', default='../models/OpenVDN/vdn-minimax-h3/h3-base/transformer')
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--output', default='artifacts/rtx5090/paper-bf16-20261005/kernel-results.json')
    args = p.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    branch = BidirectionalLinearBranch(5376, 56, 128, delta_rule='vdn_solve',
                                       short_conv=('k', 'v')).eval()
    prefix = f'transformer_blocks.{args.block}.attn.linear_attention.'
    with safe_open(str(Path(args.checkpoint) / 'linear_branch/model.safetensors'), framework='pt') as f:
        weights = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    branch.to(device='cuda', dtype=torch.bfloat16)
    branch.alpha.float()
    # Copy after choosing destination dtypes: alpha master weights never pass through BF16.
    branch.load_state_dict(weights, strict=True)
    del weights
    frames, spatial, heads, dim = 37, 1008, 7, 128
    rows = frames * spatial
    # Correlated spatial patches exercise the conditioning random independent keys miss.
    x = (torch.randn(rows, 5376, device='cuda', dtype=torch.bfloat16) * .4
         + torch.randn(frames, 1, 5376, device='cuda', dtype=torch.bfloat16)
         .expand(frames, spatial, 5376).reshape(rows, 5376))
    text_rows = 800  # prompts/example_0.pt prompt_embeds.shape[0]
    tx = torch.randn(text_rows, 5376, device='cuda', dtype=torch.bfloat16)
    index = json.loads((Path(args.base) / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    qkv, tqkv = [], []
    for proj in ('q', 'k', 'v'):
        key = f'transformer_blocks.{args.block}.attn.to_{proj}.weight'
        with safe_open(str(Path(args.base) / index[key]), framework='pt') as f:
            w = f.get_slice(key)[:heads * dim].to(device='cuda', dtype=torch.bfloat16)
        qkv.append(F.linear(x, w).reshape(rows, heads, dim))
        tqkv.append(F.linear(tx, w).reshape(-1, heads, dim))
    beta = F.linear(x, branch.beta_proj.weight[:heads]).sigmoid()
    tbeta = F.linear(tx, branch.beta_proj.weight[:heads]).sigmoid()
    g = branch.output_gate
    gate = F.linear(g.down(x), g.up.weight[:heads * dim], g.up.bias[:heads * dim]).sigmoid().reshape(rows, heads, dim)
    mean = x.reshape(frames, spatial, -1).mean(1, dtype=torch.float32)
    del x, tx, w
    # Standard eight-way Ulysses pack: q,k,v,raw_q,raw_k,softmax_gate,beta,gate.
    # Keep the actual 770-wide head stride, including views with nonzero offsets.
    def pack(raw, b, g):
        n = raw[0].shape[0]
        storage = torch.zeros(n, heads, 770, device='cuda', dtype=torch.bfloat16)
        for t, offset in zip(raw, (384, 512, 256)):
            storage[..., offset:offset+128].copy_(t)
        storage[..., 641].copy_(b)
        storage[..., 642:].copy_(g)
        return tuple(storage[..., o:o+128] for o in (384,512,256)), storage[...,641], storage[...,642:]
    qkv, beta, gate = pack(qkv, beta, gate)
    tqkv, tbeta, _ = pack(tqkv, tbeta, torch.zeros_like(tqkv[0]))
    bounds = window_bounds(frames, 1, 5)
    kwargs = dict(xv=None, num_frames=frames, tokens_per_frame=spatial, bounds=bounds,
                  qkv_raw=tuple(qkv), frame_size=(24, 42), skip_ends=True,
                  heads=slice(0, heads), beta=beta, gate=gate, frame_mean=mean,
                  text_qkv_raw=tuple(tqkv), text_beta=tbeta, inference=False)
    result = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), block=args.block,
                  checkpoint=args.checkpoint, base=args.base, seed=42,
                  layout=dict(output_frames=124, latent_frames=frames, interior_frames=35,
                              tokens_per_frame=spatial, heads=heads, dim=dim, text_rows=text_rows),
                  inputs='synthetic correlated hidden, real base QKV and checkpoint branch weights; no LoRA merge',
                  retained='gather + BF16-rounded epilogue only; stats prep measured but not enabled',
                  alpha_loading='FP32 checkpoint copied directly into FP32 destination after branch BF16 conversion',
                  strides=dict(qkv=list(qkv[0].stride()), beta=list(beta.stride()), gate=list(gate.stride())),
                  timings={}, errors={})
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    def save():
        outpath.write_text(json.dumps(result, indent=2) + '\n')
    def pair(name, ref, candidate, oracle=None):
        a, b = ref(), candidate()
        if isinstance(a, tuple):
            result['errors'][name] = [errors(y, z) for y, z in zip(b, a)]
        else:
            result['errors'][name] = errors(b, a)
        if oracle is not None:
            o = oracle()
            if isinstance(a, tuple):
                result['errors'][name + '_oracle'] = dict(
                    baseline=[errors(y, z) for y, z in zip(a, o)],
                    candidate=[errors(y, z) for y, z in zip(b, o)])
            else:
                result['errors'][name + '_oracle'] = dict(baseline=errors(a, o), candidate=errors(b, o))
        result['timings'][name] = dict(baseline=timing(ref, args.repeats),
                                       candidate=timing(candidate, args.repeats))
        save()
        print(name, json.dumps(result['errors'][name]), flush=True)
    inner = slice(spatial, (frames - 1) * spatial)
    inner_qkv = tuple(t[inner] for t in qkv)
    q, k, v = branch._features(inner_qkv, 35, (24, 42), heads=slice(0, heads))
    kf = k.reshape(35, spatial, heads, dim).permute(0, 2, 1, 3)
    vf = v.reshape(35, spatial, heads, dim).permute(0, 2, 1, 3)
    bf = beta[inner].reshape(35, spatial, heads).permute(0, 2, 1)
    # FP64 oracle at the same prepared BF16 operands, with no intermediate BF16 rounding.
    def stats_oracle():
        kd, vd, bd = kf.double(), vf.double(), bf.double().unsqueeze(-1)
        return (kd * bd).transpose(-1, -2) @ kd, (vd * bd).transpose(-1, -2) @ kd
    pair('stats', lambda: frame_statistics(kf, vf, bf),
         lambda: frame_statistics(kf, vf, bf, fused_prep=True), stats_oracle)
    A, B = frame_statistics(kf, vf, bf)
    alpha = branch.alpha(mean[1:-1], heads=slice(0, heads))
    text = branch._text_state(None, tuple(tqkv), heads=slice(0, heads), text_beta=tbeta)
    prefix_states, suffix_states = _run_scans(branch._delta_backend('backend', spatial), alpha, A, B, text)
    ibounds = [(lo-1, hi-1) for lo, hi in bounds[1:-1]]
    ga = (prefix_states, suffix_states, alpha, ibounds)
    pair('gather', lambda: gather_linear_state(*ga, text_state=text).bfloat16(),
         lambda: gather_linear_state(*ga, text_state=text, inference=True, out_dtype=torch.bfloat16),
         lambda: gather_linear_state(prefix_states.double(), suffix_states.double(), alpha.double(),
                                     ibounds, text_state=text.double()))
    state = gather_linear_state(*ga, text_state=text).bfloat16()
    readout = torch.einsum('fhvk,fshk->fshv', state, q.reshape(35, spatial, heads, dim)).reshape(-1, heads, dim)
    eg = gate[inner]
    pair('epilogue', lambda: (branch.norm(readout) * eg).reshape(-1, heads*dim),
         lambda: vda_bf16_epilogue(readout, branch.norm.weight, eg, branch.norm.eps),
         lambda: (readout.double() * torch.rsqrt(readout.double().square().mean(-1, keepdim=True)
                     + branch.norm.eps) * branch.norm.weight.double() * eg.double()).reshape(-1, heads*dim))
    pair('existing_epilogue_rejected', lambda: (branch.norm(readout) * eg).reshape(-1, heads*dim),
         lambda: linear_epilogue(readout, branch.norm.weight, eg, branch.norm.eps, inference=True),
         lambda: (readout.double() * torch.rsqrt(readout.double().square().mean(-1, keepdim=True)
                     + branch.norm.eps) * branch.norm.weight.double() * eg.double()).reshape(-1, heads*dim))
    # Existing prep is measured, not enabled by the targeted switch.
    conv = _HeadSliceSepConv(branch.short_conv, slice(0,7), 128)
    spatial_x, temporal_w = conv.spatial('k', inner_qkv[1], 35, (24,42))
    def prep_oracle():
        y = _temporal_shift(spatial_x.double(), temporal_w.double(), 5, 2).reshape(-1,7,128)
        return F.normalize(F.silu(y), dim=-1, eps=1e-6)
    pair('existing_prep_k_rejected', lambda: branch._feature_one(inner_qkv[1], 'k', 35, (24,42), heads=slice(0,7)),
         lambda: branch._feature_one(inner_qkv[1], 'k', 35, (24,42), heads=slice(0,7), inference=True), prep_oracle)
    def run(enabled):
        branch.inference_vda_kernels = enabled
        return branch(**kwargs)
    pair('branch', lambda: run(False), lambda: run(True))
    result['timings']['branch_alternating'] = alternating_timing(
        lambda: run(False), lambda: run(True), args.repeats)
    output = run(True)
    result['anchors_exact_zero'] = bool(torch.count_nonzero(output[:spatial]) == 0 and
                                        torch.count_nonzero(output[-spatial:]) == 0)
    result['fp32'] = dict(alpha=str(alpha.dtype), A=str(A.dtype), B=str(B.dtype),
                          prefix=str(prefix_states.dtype), suffix=str(suffix_states.dtype), text=str(text.dtype))
    result['tf32_after'] = torch.backends.cuda.matmul.allow_tf32
    result['acceptance'] = dict(branch_rel_l2_limit=1e-4,
                                branch_pass=result['errors']['branch']['rel_l2'] < 1e-4,
                                stats_exact=all(e['exact'] for e in result['errors']['stats']))
    save()
    print(json.dumps(result['timings']['branch']), flush=True)


if __name__ == '__main__':
    main()
