"""Offline, import-only dependency checks and tiny GPU0 tests. No model loading."""
import importlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/rtx5090'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ.pop('FA_CLC', None)
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(OUT / 'kernel-cache/inductor')
os.environ['TRITON_CACHE_DIR'] = str(OUT / 'kernel-cache/triton')
report = {'python': platform.python_version(), 'executable': sys.executable,
          'packages': {}, 'imports': {}, 'tests': {}, 'failures': []}


def check(name, fn):
    try:
        result = fn()
        report['tests'][name] = result
        print(f'PASS {name}: {json.dumps(result, default=str)}', flush=True)
    except Exception:
        error = traceback.format_exc()
        report['failures'].append(name)
        report['tests'][name] = {'error': error}
        print(f'FAIL {name}\n{error}', flush=True)


for dist in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
    report['packages'][dist.metadata['Name']] = dist.version
print('VERSIONS ' + json.dumps(report['packages'], sort_keys=True), flush=True)


def pins():
    from packaging.requirements import Requirement
    deps = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['dependencies']
    checked = {}
    for dep in deps:
        req = Requirement(dep)
        if req.name == 'flash-attn-4':
            continue
        version = metadata.version(req.name)
        assert req.specifier.contains(version), (dep, version)
        checked[req.name] = version
    assert metadata.version('torch') == '2.13.0+cu130'
    for module in ('flash_attn', 'fla'):
        assert importlib.util.find_spec(module) is None, module
    return checked


check('project_pins_and_omissions', pins)
modules = [
    'torch', 'triton', 'torchvision', 'transformers', 'accelerate', 'torchao',
    'peft', 'safetensors', 'huggingface_hub', 'omegaconf', 'yaml', 'einops',
    'numpy', 'PIL', 'av', 'tqdm', 'diffusers',
    'src.config.inference', 'src.checkpoints.loader', 'src.models.factory',
    'src.models.hybrid_attention', 'src.models.hybrid_transform',
    'src.models.softmax_attention.decomposed', 'src.inference.utils.assemble',
    'src.inference.render', 'src.inference.infer_diffusers',
    'src.inference.infer', 'src.inference.infer_ulysses',
    'src.inference.encode_prompt', 'src.inference.encode_keyframes',
    'src.diffusers_export.modeling',
]
for module in modules:
    try:
        imported = importlib.import_module(module)
        report['imports'][module] = {'status': 'PASS', 'file': imported.__file__}
        print(f'PASS import {module}', flush=True)
    except Exception:
        error = traceback.format_exc()
        report['imports'][module] = {'status': 'FAIL', 'error': error}
        report['failures'].append(f'import {module}')
        print(f'FAIL import {module}\n{error}', flush=True)


def symbols_and_apis():
    from diffusers import (MiniMaxH3Transformer3DModel, AutoencoderKLMiniMaxH3,
                           AutoencoderKLMiniMaxH3Audio, ModularPipeline)
    from diffusers.hooks import apply_group_offloading
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils.export_utils import encode_video
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from torch.nn.attention.varlen import varlen_attn
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig,
                                      PerRow, PerTensor)
    from src.models.softmax_attention.window import has_fa4_kernels
    symbols = [MiniMaxH3Transformer3DModel, AutoencoderKLMiniMaxH3,
               AutoencoderKLMiniMaxH3Audio, ModularPipeline, apply_group_offloading,
               ModelMixin.enable_group_offload, ModelMixin.from_pretrained,
               encode_video, Qwen3VLForConditionalGeneration, Qwen3VLProcessor,
               varlen_attn, Float8MMConfig, Float8DynamicActivationFloat8WeightConfig,
               PerRow, PerTensor]
    result = {}
    for obj in symbols:
        source_obj = inspect.unwrap(obj)
        result[obj.__qualname__] = {
            'signature': str(inspect.signature(obj)),
            'file': inspect.getsourcefile(source_obj),
            'line': inspect.getsourcelines(source_obj)[1],
        }
    result['has_fa4_kernels_gpu0'] = has_fa4_kernels()
    return result


check('symbols_and_loader_offload_apis', symbols_and_apis)


def cuda_bf16():
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability(0) == (12, 0)
    assert torch.cuda.is_bf16_supported()
    torch.manual_seed(5090)
    a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
    c = a @ b
    reference = (a.double() @ b.double()).float()
    relative_error = ((c.float() - reference).norm() / reference.norm()).item()
    assert torch.isfinite(c).all()
    assert relative_error < 0.01, relative_error
    torch.cuda.synchronize()
    return {'device': torch.cuda.get_device_name(0), 'capability': [12, 0],
            'torch_cuda': torch.version.cuda, 'compiled_arches': torch.cuda.get_arch_list(),
            'shape': list(c.shape), 'dtype': str(c.dtype),
            'relative_l2_error': relative_error, 'finite': True}


check('cuda_bf16_matmul', cuda_bf16)


def compiled_flex():
    import torch
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    from torch._dynamo.utils import counters
    from torch._inductor import metrics
    torch.manual_seed(5090)
    q, k, v = [torch.randn(1, 2, 128, 64, device='cuda', dtype=torch.bfloat16)
               for _ in range(3)]

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    mask = create_block_mask(causal, B=1, H=2, Q_LEN=128, KV_LEN=128, device='cuda')
    compiled = torch.compile(flex_attention, backend='inductor', fullgraph=True)
    start = time.monotonic()
    with torch.inference_mode():
        actual = compiled(q, k, v, block_mask=mask)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        repeated = compiled(q, k, v, block_mask=mask)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=True)
    delta = actual.float() - reference
    relative_error = (delta.norm() / reference.norm()).item()
    assert torch.isfinite(actual).all()
    assert relative_error < 0.02, relative_error
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    graphs = counters['stats']['unique_graphs']
    assert graphs >= 1, 'No compiled graph was recorded'
    return {'shape': list(actual.shape), 'dtype': str(actual.dtype),
            'causal_block_mask': True, 'fullgraph': True, 'backend': 'inductor',
            'first_compile_and_run_seconds': elapsed,
            'relative_l2_error_vs_fp32_sdpa': relative_error,
            'max_abs_error': delta.abs().max().item(), 'repeat_exact': True,
            'unique_graphs': graphs, 'generated_kernel_count': metrics.generated_kernel_count,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}


check('compiled_flex_attention_bf16', compiled_flex)
report['diffusers_head'] = subprocess.check_output(
    ['git', '-C', str(ROOT / 'diffusers'), 'rev-parse', 'HEAD'], text=True).strip()
(OUT / 'checks.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print('SUMMARY ' + json.dumps({'imports': len(report['imports']),
                               'failures': report['failures']}), flush=True)
sys.exit(bool(report['failures']))
