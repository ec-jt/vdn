"""Targeted VDA contract and parity checks; no global inference switch."""
import unittest
from unittest.mock import patch
import torch

from src.models.linear_attention.branch import BidirectionalLinearBranch
from src.models.linear_attention.kernels import vda_bf16_epilogue
from src.models.linear_attention.scan import frame_statistics, gather_linear_state


class VdaPaperKernelsTests(unittest.TestCase):
    def test_default_and_grad_guard(self):
        b = BidirectionalLinearBranch(16, 1, 128, delta_rule='vdn_solve')
        self.assertFalse(b.inference_vda_kernels)
        b.inference_vda_kernels = True
        qkv = (torch.zeros(3, 1, 128),) * 3
        with self.assertRaisesRegex(RuntimeError, 'no_grad'):
            b(None, 3, 1, [(0, 2)] * 3, qkv)
        with torch.no_grad():
            with self.assertRaisesRegex(ValueError, 'inference=False'):
                b(None, 3, 1, [(0, 2)] * 3, qkv, inference=True)
            with self.assertRaisesRegex(ValueError, 'CUDA BF16'):
                b(None, 3, 1, [(0, 2)] * 3, qkv)
            b.a_fp32 = False
            with self.assertRaisesRegex(ValueError, 'a_fp32'):
                b(None, 3, 1, [(0, 2)] * 3, qkv)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    @torch.no_grad()
    def test_stats_strided_and_no_tf32(self):
        torch.manual_seed(42)
        k = torch.randn(35, 19, 7, 128, device='cuda', dtype=torch.bfloat16).permute(0,2,1,3)
        v = torch.randn_like(k)
        beta = torch.rand(35, 19, 7, device='cuda', dtype=torch.bfloat16).permute(0,2,1)
        ref = frame_statistics(k, v, beta)
        with patch('src.models.linear_attention.scan._tf32_matmul', side_effect=AssertionError('TF32 forbidden')):
            actual = frame_statistics(k, v, beta, fused_prep=True)
        for a, b in zip(actual, ref):
            self.assertEqual(a.dtype, torch.float32)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    @torch.no_grad()
    def test_epilogue_packed_gate_and_errors(self):
        torch.manual_seed(123)
        x = torch.randn(37, 7, 128, device='cuda', dtype=torch.bfloat16)
        gate = torch.rand(37, 7, 770, device='cuda', dtype=torch.bfloat16)[..., 256:384]
        w = torch.randn(128, device='cuda', dtype=torch.bfloat16)
        ms = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).square()/128
        expected = (x * torch.rsqrt(ms + 1e-6).bfloat16() * w * gate).reshape(37,-1)
        got = vda_bf16_epilogue(x, w, gate, 1e-6)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertTrue(got.is_contiguous())
        with self.assertRaises(ValueError):
            vda_bf16_epilogue(x.float(), w, gate, 1e-6)
        with self.assertRaises(ValueError):
            vda_bf16_epilogue(x[..., :64], w, gate, 1e-6)
        with torch.enable_grad(), self.assertRaises(RuntimeError):
            vda_bf16_epilogue(x, w, gate, 1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    @torch.no_grad()
    def test_gather_text_edges_and_decay(self):
        torch.manual_seed(3)
        shape = (35, 7, 128, 128)
        p, s = (torch.randn(shape, device='cuda') for _ in range(2))
        alpha = torch.rand(35, 7, 128, device='cuda') * .1 + .89
        bounds = [((t+1)//5*5-6, (t+1)//5*5+8) for t in range(35)]
        for text in (None, torch.randn(7,128,128,device='cuda')):
            expected = gather_linear_state(p,s,alpha,bounds,text_state=text).bfloat16()
            actual = gather_linear_state(p,s,alpha,bounds,text_state=text,inference=True,out_dtype=torch.bfloat16)
            # FP32 log/exp fusion may cross a BF16 midpoint. Fixed tight aggregate budget.
            self.assertLess(((actual.float()-expected.float()).norm()/expected.float().norm()).item(), 1e-4)
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    @torch.no_grad()
    def test_branch_anchors_text_and_unchanged_scans(self):
        torch.manual_seed(5)
        b = BidirectionalLinearBranch(16, 2, 128, delta_rule='vdn_solve',
                                       short_conv=('k','v')).cuda().bfloat16().eval()
        b.alpha.float()
        f, s, h = 17, 6, 1
        packed = torch.randn(f*s, h, 770, device='cuda', dtype=torch.bfloat16)
        qkv = tuple(packed[..., o:o+128] for o in (384,512,256))
        beta = packed[...,641].sigmoid()
        gate = packed[...,642:].sigmoid()
        mean = torch.randn(f,16,device='cuda')
        bounds = [(t-2,t+2) for t in range(f)]
        text = tuple(torch.randn(5,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(3))
        for use_text in (False, True):
            kw = dict(frame_size=(2,3), skip_ends=True, heads=slice(1,2), beta=beta,
                      gate=gate, frame_mean=mean, text_qkv_raw=text if use_text else None,
                      text_beta=torch.full((5,h),.5,device='cuda',dtype=torch.bfloat16) if use_text else None)
            b.inference_vda_kernels = False
            expected = b(None,f,s,bounds,qkv,**kw)
            b.inference_vda_kernels = True
            with patch('src.models.linear_attention.branch._run_scans_inference',
                       side_effect=AssertionError('scan unchanged')):
                actual = b(None,f,s,bounds,qkv,**kw)
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertEqual(tuple(actual.shape), (f*s,128))
            self.assertEqual(torch.count_nonzero(actual[:s]).item(), 0)
            self.assertEqual(torch.count_nonzero(actual[-s:]).item(), 0)
            self.assertLess(((actual.float()-expected.float()).norm()/expected.float().norm()).item(), 1e-4)


if __name__ == '__main__':
    unittest.main()
