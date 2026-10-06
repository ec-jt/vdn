"""GPU-0 real-weight isolated VDA/window-softmax ordering and latency probe.
Synthetic activations, no LoRA merge, no trajectory or distributed speed claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from safetensors import safe_open
from scripts.probe_rtx5090_vda_kernels import errors, alternating_timing
from src.models.linear_attention.branch import BidirectionalLinearBranch
from src.models.sequence_layout import SequenceLayout
from src.models.softmax_attention.window import window_bounds
from src.models.softmax_attention.kernels import apply_softmax_gate
from src.inference.utils.ulysses import _linear_branch_shard, _window_softmax_branch
from src.inference.utils.vda_overlap import vda_on_side_stream
from src.inference.render import audio_latent_num_frames

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='../models/OpenVDN/vdn-minimax-h3/stage-dmd-step-250')
    p.add_argument('--base', default='../models/OpenVDN/vdn-minimax-h3/h3-base/transformer')
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--repeats', type=int, default=15)
    p.add_argument('--output', default='artifacts/rtx5090/paper-bf16-20261005/overlap-results.json')
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
    audio_rows = audio_latent_num_frames(124)
    global_rows = text_rows + audio_rows
    seq_len = global_rows + rows
    packed = torch.zeros(seq_len, heads, 770, device='cuda', dtype=torch.bfloat16)
    for v, t, offset in zip(qkv, tqkv, (384, 512, 256)):
        packed[global_rows:, :, offset:offset+128].copy_(v)
        packed[:text_rows, :, offset:offset+128].copy_(t)
        packed[text_rows:global_rows, :, offset:offset+128].normal_()
    # Actual window-softmax kernel/layout, normalized synthetic Q/K. Not a full
    # deployed QK-norm/RoPE projection comparison; both orderings read identical data.
    packed[..., :128].copy_(F.rms_norm(packed[...,384:512], (128,)))
    packed[...,128:256].copy_(F.rms_norm(packed[...,512:640], (128,)))
    packed[...,640].fill_(.5)
    packed[global_rows:,:,641].copy_(beta)
    packed[:text_rows,:,641].copy_(tbeta)
    packed[global_rows:,:,642:].copy_(gate)
    del qkv, tqkv, beta, tbeta, gate
    q,k,v,rq,rk,sg,b,g = packed.split((128,128,128,128,128,1,1,128),dim=-1)
    b = b.squeeze(-1)
    layout = SequenceLayout(seq_len, global_rows, frames, spatial, 24,42,0,text_rows)
    bounds = window_bounds(frames,1,5)
    attn = SimpleNamespace(layout=layout, linear_attention=branch, enable_text_state=True,
        _bounds=lambda l: bounds, inference_mode=False, anchor_frames='both',
        _window_kernel=lambda inference,cuda: 'flex')
    owner = SimpleNamespace()
    def linear():
        return _linear_branch_shard(attn,(rq,rk,v),b,g,mean,0,heads)
    def softmax():
        return apply_softmax_gate(_window_softmax_branch(attn,q,k,v,layout,bounds,128**-.5),
                                  sg,inference=False).view(seq_len,heads,dim)
    def sequential():
        sm = softmax()
        return torch.cat((sm, linear()),dim=-1)
    def overlapped():
        with vda_on_side_stream(owner,linear,inputs=(packed,mean),module=branch) as out:
            sm = softmax()
        return torch.cat((sm,out),dim=-1)
    result = dict(block=args.block, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        inputs='real branch/base QKV weights; synthetic correlated hidden; no LoRA merge; normalized synthetic softmax Q/K',
        layout=dict(frames=frames,spatial=spatial,text_rows=text_rows,audio_rows=audio_rows,seq_len=seq_len,heads=heads,dim=dim),
        strides=list(rq.stride()), scope='isolated branch+actual window softmax, no collectives/offload/trajectory',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path('src/inference/utils/vda_overlap.py'),Path('src/inference/utils/ulysses.py'),Path(__file__).relative_to(Path.cwd())]})
    path = Path(args.output)
    def save():
        path.write_text(json.dumps(result,indent=2)+'\n')
    # Deliberately first branch invocation on the side stream. Includes lazy
    # cuBLAS/Triton setup, no preparatory eager branch and no dtype workaround.
    t = time.perf_counter()
    try:
        cold = overlapped()
        torch.cuda.synchronize()
    except Exception as exc:
        result['cold_error'] = repr(exc)
        save()
        raise
    result['cold_wall_ms'] = (time.perf_counter()-t)*1000
    ref = sequential()
    result['ordering_error'] = errors(cold,ref)
    assert torch.equal(cold,ref), result['ordering_error']
    del cold, ref
    print('cold',result['cold_wall_ms'],result['ordering_error'],flush=True)
    result['timings'] = alternating_timing(sequential,overlapped,args.repeats)
    save()
    # Repeat on a nondefault caller stream, with allocator churn between calls.
    main = torch.cuda.Stream()
    main.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(main):
        ref = sequential()
        for _ in range(3):
            out = overlapped()
            churn = torch.empty_like(packed).fill_(3)
            assert torch.equal(out,ref)
            del out,churn
    main.synchronize()
    result['nondefault_repeated_exact'] = True
    branch.inference_vda_kernels = True
    fused_ref = sequential()
    both = overlapped()
    result['both_vs_fused'] = errors(both,fused_ref)
    assert torch.equal(both,fused_ref), result['both_vs_fused']
    del both,fused_ref
    result['fused_timings'] = alternating_timing(sequential,overlapped,args.repeats)
    save()
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__':
    main()
