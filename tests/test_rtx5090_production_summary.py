"""CPU-only activity parsing and latent-delta statistics."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


spec = importlib.util.spec_from_file_location('production_summary', Path(__file__).resolve().parents[1] / 'scripts/summarize_rtx5090_production.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class ProductionSummaryTests(unittest.TestCase):
    def test_activity_ignores_partial_sample_and_keeps_rss_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'activity.log'
            lines = ['2026-10-04T00:00:00Z']
            lines += [f'{r}, 100, {1000+r}, 200.5' for r in range(8)]
            lines += ['aggregate_inference_rss_kib 12345', '2026-10-04T00:00:01Z', '0, 0, 0, 20']
            path.write_text('\n'.join(lines))
            result = summary.activity(path)
            self.assertEqual(result['samples'], 1)
            self.assertEqual(result['all_eight_nonzero_util_samples'], 1)
            self.assertEqual(result['aggregate_sampled_rss_peak_bytes'], 12345*1024)
            self.assertEqual(result['ranks'][7]['peak_memory_mib'], 1007)

    def test_delta_exact_and_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.pt', Path(tmp)/'b.pt'
            torch.save(dict(video=torch.tensor([1., 3.]), audio=torch.tensor([2.])), a)
            torch.save(dict(video=torch.tensor([1., 2.]), audio=torch.tensor([2.])), b)
            result = summary.delta(a, b)
            self.assertEqual(result['video']['max_abs'], 1.)
            self.assertEqual(result['video']['mean_abs'], .5)
            self.assertEqual(result['video']['differing_values'], 1)
            self.assertEqual(result['audio']['relative_l2'], 0.)
            self.assertTrue(result['audio']['finite'])
