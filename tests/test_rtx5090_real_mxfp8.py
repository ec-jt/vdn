"""Focused standalone-harness tests; existing production tests are untouched."""

import io
import unittest

import torch

from scripts.benchmark_rtx5090_real_mxfp8 import (
    cpu_master, error_metrics, pack_prepared, speed_gate, unpack_prepared,
)
from scripts.probe_rtx5090_mxfp8 import prepare_torch_mxfp8


class TestRealMXFP8Harness(unittest.TestCase):
    def test_cpu_format_roundtrip_preserves_float_master(self):
        master = torch.arange(3 * 33, dtype=torch.float32).reshape(3, 33).to(torch.bfloat16)
        original = master.clone()
        q, s = prepare_torch_mxfp8(master)
        stream = io.BytesIO()
        torch.save(pack_prepared(q, s), stream)
        stream.seek(0)
        rq, rs = unpack_prepared(torch.load(stream, weights_only=True))
        self.assertEqual(rq.dtype, torch.float8_e4m3fn)
        self.assertEqual(rs.dtype, torch.float8_e8m0fnu)
        torch.testing.assert_close(q.view(torch.uint8), rq.view(torch.uint8))
        torch.testing.assert_close(s.view(torch.uint8), rs.view(torch.uint8))
        torch.testing.assert_close(master, original, rtol=0, atol=0)
        self.assertEqual(master.device.type, "cpu")

    def test_cpu_master_independent(self):
        source = torch.ones(3, 5, dtype=torch.bfloat16)
        saved = cpu_master(source)
        source.zero_()
        self.assertTrue(bool((saved == 1).all()))
        self.assertEqual(saved.dtype, torch.bfloat16)

    def test_placeholder_rejected(self):
        with self.assertRaisesRegex(ValueError, "materialized"):
            cpu_master(torch.empty(2, 3, device="meta"))

    def test_format_rejection(self):
        q, s = prepare_torch_mxfp8(torch.zeros(2, 3))
        record = pack_prepared(q, s)
        record["format"] = "per_tensor_fp8"
        with self.assertRaises(ValueError):
            unpack_prepared(record)

    def test_zero_norm_exact_absolute_check(self):
        ref = torch.zeros(5, dtype=torch.bfloat16)
        self.assertTrue(error_metrics(ref, ref)["pass"])
        result = error_metrics(ref, ref + .001)
        self.assertFalse(result["pass"])
        self.assertIsNone(result["relative_l2"])

    def test_fixed_numeric_gates(self):
        ref = torch.ones(100, dtype=torch.bfloat16)
        self.assertTrue(error_metrics(ref, ref + .03125)["pass"])
        self.assertFalse(error_metrics(ref, ref + .0625)["pass"])
        out = ref.clone()
        out[0] += 1
        self.assertFalse(error_metrics(ref, out)["pass"])
        out[0] = float("nan")
        self.assertFalse(error_metrics(ref, out)["pass"])

    def test_speed_gates(self):
        def case(b, c):
            return {"timing": {"bf16": {"median_ms": b}, "candidate": {"median_ms": c}}}
        self.assertTrue(speed_gate([case(1, .94)])["pass"])
        self.assertFalse(speed_gate([case(1, .96)])["pass"])
        result = speed_gate([case(10, 8), case(1, 1.06)])
        self.assertTrue(result["aggregate_pass"])
        self.assertFalse(result["individual_pass"])


if __name__ == "__main__":
    unittest.main()
