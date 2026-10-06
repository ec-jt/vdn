"""CPU-only checks for bounded baseline tools."""
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BaselineToolsTests(unittest.TestCase):
    def test_interval_union(self):
        union = runpy.run_path(str(ROOT / 'scripts/summarize_rtx5090_nsight.py'))['union_ns']
        self.assertEqual(union([]), 0)
        self.assertEqual(union([(4, 8), (1, 5), (2, 3), (10, 12)]), 9)

    def test_supervisor_success(self):
        p = subprocess.run([sys.executable, str(ROOT / 'scripts/supervise_rtx5090.py'),
                            '--seconds', '5', '--', sys.executable, '-c', 'pass'], timeout=10)
        self.assertEqual(p.returncode, 0)

    def test_supervisor_detached_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / 'pid'
            child = 'import os,time,pathlib; pathlib.Path(' + repr(str(pidfile)) + ').write_text(str(os.getpid())); time.sleep(30)'
            code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",' + repr(child) + '],start_new_session=True); time.sleep(30)'
            p = subprocess.run([sys.executable, str(ROOT / 'scripts/supervise_rtx5090.py'),
                                '--seconds', '1', '--', sys.executable, '-c', code], timeout=22)
            self.assertEqual(p.returncode, 124)
            pid = int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_launcher_fixed_geometry(self):
        source = (ROOT / 'scripts/benchmark_rtx5090_nsight.sh').read_text()
        for flag in ['render.num_frames=124', 'render.num_steps=8', 'render.seed=42',
                     'parallel.profile=false', 'NCCL_SHM_USE_CUDA_MEMCPY=0',
                     'precision.fp8.enabled=false', 'scripts/supervise_rtx5090.py']:
            self.assertIn(flag, source)


if __name__ == '__main__':
    unittest.main()
