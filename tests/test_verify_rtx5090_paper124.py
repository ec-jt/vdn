"""Small CPU-only synthetic verifier tests; never decode production media."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.verify_rtx5090_paper124 import (
    DeltaAccumulator, candidate_passes, compare_streams, compare_tensors,
    fixed_chunks, main, parse_args,
)


class Paper124VerifierTests(unittest.TestCase):
    def test_exact_hash_covers_all_bytes(self):
        a = np.arange(35, dtype=np.uint8).reshape(5, 7)
        result = compare_streams([a], [a.copy()], 4)
        self.assertEqual(result["reference"]["sha256"], hashlib.sha256(a.tobytes()).hexdigest())
        self.assertTrue(result["byte_exact"])
        self.assertEqual(result["mismatched_count"], 0)
        self.assertEqual(result["relative_l2"], 0)
        self.assertEqual(result["reference"]["elements"], 35)

    def test_unsigned_subtraction_does_not_wrap(self):
        a = np.array([0, 255], dtype=np.uint8)
        b = np.array([255, 0], dtype=np.uint8)
        result = compare_streams([a], [b], 1)
        self.assertEqual(result["maxabs"], 255)
        self.assertEqual(result["rmse"], 255)
        self.assertAlmostEqual(result["relative_l2"], 2 ** 0.5)
        self.assertEqual(result["mismatched_count"], 2)

    def test_metrics_match_direct_float64(self):
        a = np.arange(1, 48, dtype=np.float32)
        b = a.copy()
        b[::3] += 0.25
        d = b.astype(np.float64) - a
        result = compare_streams([a], [b], 7)
        self.assertAlmostEqual(result["relative_l2"], np.linalg.norm(d) / np.linalg.norm(a.astype(np.float64)))
        self.assertAlmostEqual(result["rmse"], np.sqrt(np.mean(d * d)))
        self.assertEqual(result["maxabs"], 0.25)
        self.assertEqual(result["mismatched_count"], 16)

    def test_stream_frame_boundaries_do_not_change_hash(self):
        a = np.arange(57, dtype="<f4")
        result = compare_streams([a[:1], a[1:17], a[17:]], [a[:25], a[25:]], 8)
        self.assertTrue(result["byte_exact"])
        self.assertEqual(result["candidate"]["sha256"], hashlib.sha256(a.tobytes()).hexdigest())

    def test_signed_zero_numeric_equal_but_not_byte_exact(self):
        result = compare_streams([np.array([0.0], dtype="<f4")],
                                 [np.array([-0.0], dtype="<f4")], 1)
        self.assertFalse(result["byte_exact"])
        self.assertEqual(result["mismatched_count"], 0)
        self.assertEqual(result["relative_l2"], 0)

    def test_zero_reference_nonzero_candidate_is_undefined(self):
        result = compare_streams([np.zeros(3)], [np.ones(3)], 2)
        self.assertIsNone(result["relative_l2"])
        self.assertFalse(result["relative_l2_defined"])
        self.assertEqual(result["rmse"], 1)

    def test_nonfinite_is_reported_and_json_safe(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                result = compare_streams([np.array([1., value])], [np.ones(2)], 1)
                self.assertFalse(result["finite"])
                self.assertFalse(result["numeric_metrics_valid"])
                self.assertIsNone(result["rmse"])
                json.dumps(result, allow_nan=False)

    def test_different_lengths_drain_both_streams(self):
        a, b = np.arange(2), np.arange(17)
        result = compare_streams([a], [b], 3)
        self.assertFalse(result["same_element_count"])
        self.assertFalse(result["numeric_metrics_valid"])
        self.assertIsNone(result["mismatched_count"])
        self.assertEqual(result["candidate"]["sha256"], hashlib.sha256(b.tobytes()).hexdigest())

    def test_empty_stream_cannot_pass(self):
        result = compare_streams([], [], 3)
        self.assertFalse(result["numeric_metrics_valid"])
        self.assertFalse(candidate_passes({"video": result, "audio": result},
                                          {"video": result, "audio": result}, "exact"))

    def test_chunk_bounds_and_noncontiguous_order(self):
        a = np.arange(30, dtype=np.float32).reshape(5, 6).T
        chunks = list(fixed_chunks([a], 4))
        self.assertTrue(all(x.size <= 4 for x in chunks))
        np.testing.assert_array_equal(np.concatenate(chunks), a.reshape(-1))
        with self.assertRaises(ValueError):
            list(fixed_chunks([a], 0))

    def test_bfloat16_hash_preserves_native_bytes(self):
        a = torch.tensor([1., -0., 0.5, 4., 3.], dtype=torch.bfloat16)
        result = compare_tensors(a, a.clone(), 2)
        self.assertEqual(result["reference"]["sha256"],
                         hashlib.sha256(a.view(torch.uint8).numpy().tobytes()).hexdigest())
        self.assertEqual(result["reference"]["bytes"], 10)
        self.assertEqual(result["dtype"], "torch.bfloat16")
        self.assertTrue(result["byte_exact"])

    def test_tensor_schema_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            compare_tensors(torch.ones(2), torch.ones(3), 2)
        with self.assertRaises(ValueError):
            compare_tensors(torch.ones(2), torch.ones(2, dtype=torch.float64), 2)

    def test_fused_gate_checks_each_latent_and_media_finiteness(self):
        exact = compare_streams([np.ones(2)], [np.ones(2)], 1)
        near = {**exact, "relative_l2": 1e-3, "byte_exact": False}
        far = {**near, "relative_l2": 1.001e-3}
        media = {"video": exact, "audio": exact}
        self.assertTrue(candidate_passes({"video": near, "audio": near}, media, "fused"))
        self.assertFalse(candidate_passes({"video": near, "audio": far}, media, "fused"))
        self.assertFalse(candidate_passes({"video": near, "audio": near}, media, "exact"))
        self.assertFalse(candidate_passes({"video": exact, "audio": exact},
                                          {"video": exact, "audio": {**exact, "finite": False}}, "fused"))

    def test_cli_requires_explicit_candidate_expectations(self):
        args = parse_args(["--reference", "ref", "--candidate", "a", "b", "--expect", "a=exact",
                           "--expect", "b=fused", "--output", "new.json"])
        self.assertEqual(args.expectations, {"a": "exact", "b": "fused"})
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--reference", "ref", "--candidate", "a", "b", "--expect", "a=exact",
                        "--output", "new.json"])

    def test_output_no_overwrite_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("preserve")
            with patch("scripts.verify_rtx5090_paper124.limit_cpu_threads"), \
                    patch("torch.load") as load, self.assertRaises(FileExistsError):
                main(["--reference", "ref", "--candidate", "a", "--expect", "a=exact",
                      "--output", str(output)])
            load.assert_not_called()
            self.assertEqual(output.read_text(), "preserve")

    def test_numeric_overflow_fails_closed(self):
        result = compare_streams([np.array([1e308])], [np.array([-1e308])], 1)
        self.assertFalse(result["numeric_metrics_valid"])
        self.assertIsNone(result["relative_l2"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
