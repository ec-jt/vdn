"""Stdlib-only tests: never import torch or initialize CUDA."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import unittest


PROBE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/probe_rtx5090_transport.py"))


class TransportTests(unittest.TestCase):
    def test_import_is_stdlib_only(self):
        script = """
import runpy, sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Probe attempted to import torch')
sys.meta_path.insert(0, BlockTorch())
runpy.run_path('scripts/probe_rtx5090_transport.py')
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-B", "-c", script], check=True,
                       cwd=Path(__file__).resolve().parents[1])

    def test_config_output_destination_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nccl.conf"
            path.write_text("NCCL_DEBUG_FILE=/tmp/never-write.log\n")
            with self.assertRaises(RuntimeError):
                PROBE["transport_configs"]({"NCCL_CONF_FILE": str(path)})

    def test_supervisor_kills_detached_child(self):
        # CPU-only subprocess, isolated so subreaper state never affects the test runner.
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "child.pid"
            worker = ("import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                      f"open({str(pidfile)!r},'x').write(str(os.getpid())); time.sleep(30)")
            launcher = ("import subprocess,sys,time; "
                        f"subprocess.Popen([sys.executable,'-B','-c',{worker!r}],start_new_session=True); "
                        "time.sleep(30)")
            supervisor = ("import runpy,sys; p=runpy.run_path('scripts/probe_rtx5090_transport.py'); "
                          f"sys.exit(p['supervise']([sys.executable,'-B','-c',{launcher!r}],seconds=0.5,grace=0.2))")
            result = subprocess.run([sys.executable, "-B", "-c", supervisor], timeout=5,
                                    cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(result.returncode, 124)
            pid = int(pidfile.read_text())
            deadline = time.monotonic() + 2
            while Path(f"/proc/{pid}/stat").exists() and time.monotonic() < deadline:
                stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if stat[0] == "Z":
                    break
                time.sleep(0.01)
            if Path(f"/proc/{pid}/stat").exists():
                self.assertEqual(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0], "Z")

    def test_exact_payload_and_bound(self):
        p = PROBE["plan"]()
        self.assertEqual(p["sequence_splits"], [4751] * 7 + [4746])
        self.assertEqual(p["forward_output_split_elements"], [25607890] * 7 + [25580940])
        self.assertEqual(p["gather_input_bytes_per_rank"] * 8, 1376730080)
        self.assertEqual(p["gather_live_tensor_bytes"], 1548821340)
        self.assertEqual(p["alltoall_peak_live_tensor_bytes"], 1638851060)
        self.assertLess(p["alltoall_peak_live_tensor_bytes"], 2 * 1024**3)

    def test_env_is_allowlisted(self):
        env = {"NCCL_P2P_LEVEL": "NODE", "CUDA_VISIBLE_DEVICES": "",
               "HF_TOKEN": "secret", "NCCL_PRIVATE_TOKEN": "secret", "AWS_SECRET_ACCESS_KEY": "secret"}
        self.assertEqual(PROBE["transport_env"](env), {"NCCL_P2P_LEVEL": "NODE", "CUDA_VISIBLE_DEVICES": ""})

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank.json"
            PROBE["exclusive_json"](path, {"original": True})
            with self.assertRaises(FileExistsError):
                PROBE["exclusive_json"](path, {"replacement": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_aggregation_max_per_sample(self):
        records = [{"rank": r, "status": "ok", "allgather_seconds": [r + 1, 2, 3, 4],
                    "alltoall_roundtrip_seconds": [1, r + 1, 3, 4]} for r in range(8)]
        summary = PROBE["aggregate"](records)
        self.assertEqual(summary["allgather_seconds"]["max_rank_per_sample"], [8, 2, 3, 4])
        self.assertEqual(summary["allgather_seconds"]["warm_mean_max_rank"], 3)
        self.assertEqual(summary["alltoall_roundtrip_seconds"]["warm_mean_max_rank"], 5)
        with self.assertRaises(ValueError):
            PROBE["aggregate"](records[:-1])

    def test_production_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "123"
            proc.mkdir()
            (proc / "cmdline").write_bytes(b"python\0src/inference/infer_ulysses.py\0")
            self.assertEqual(PROBE["production_pids"](Path(tmp)), [123])
            (proc / "cmdline").write_bytes(b"python\0other.py\0")
            self.assertEqual(PROBE["production_pids"](Path(tmp)), [])


if __name__ == "__main__":
    unittest.main()
