"""CPU-only shell regressions: no torch import, real GPU query, or renderer."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "scripts/inference/8nfe_rtx5090_8gpu.sh"
TRANSPORT = "scripts/inference/8nfe_rtx5090_8gpu_transport.sh"
BENCHMARK = "scripts/benchmark_rtx5090_production.sh"


class ProductionShellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rtx-shell-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original_bytes = (ROOT / ORIGINAL).read_bytes()
        for name in (ORIGINAL, TRANSPORT, BENCHMARK):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # An allowlisted PATH cannot fall through to installed GPU/ML commands.
        for name in ("bash", "dirname", "basename", "mkdir", "tail", "awk",
                     "sleep", "date", "timeout"):
            (self.bin / name).symlink_to(shutil.which(name))
        self.write_command("bin/nvidia-smi", '''
printf '%s\n' "$*" >> "$SANDBOX/queries"
if [[ "$*" == *query-compute-apps* ]]; then
    printf '%s' "${FAKE_APPS:-}"
fi
exit "${FAKE_PROBE_STATUS:-0}"
''')
        self.write_command("bin/ps", "exit 0\n")
        self.write_command(".venv/bin/python", '''
printf '%s\n' "${NCCL_SHM_USE_CUDA_MEMCPY-unset}" > "$SANDBOX/setting"
printf '%s\n' "${NCCL_P2P_LEVEL-unset}" > "$SANDBOX/p2p"
printf '%s\n' "$@" > "$SANDBOX/args"
exit "${FAKE_RENDER_STATUS:-0}"
''')
        self.env = {
            "PATH": str(self.bin), "HOME": str(self.root),
            "SANDBOX": str(self.root), "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        }

    def tearDown(self):
        self.assertEqual((ROOT / ORIGINAL).read_bytes(), self.original_bytes)

    def write_command(self, name, body):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/bin/bash\nset -eu\n" + body)
        target.chmod(0o700)

    def run_script(self, script, *args, **env):
        return subprocess.run(
            ["/bin/bash", str(self.root / script), *args],
            cwd=self.root, env={**self.env, **env},
            capture_output=True, text=True, timeout=10,
        )

    def test_original_has_no_transport_default(self):
        result = self.run_script(ORIGINAL, "render.num_frames=124")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "setting").read_text(), "unset\n")
        self.assertNotIn("NCCL_SHM_USE_CUDA_MEMCPY", self.original_bytes.decode())

    def test_transport_default_and_explicit_overrides_preserve_arguments(self):
        for value in (None, "0", "1"):
            with self.subTest(value=value):
                env = {} if value is None else {"NCCL_SHM_USE_CUDA_MEMCPY": value}
                result = self.run_script(TRANSPORT, "render.out=a b.mp4", **env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((self.root / "setting").read_text().strip(),
                                 "0" if value is None else value)
                self.assertEqual((self.root / "p2p").read_text().strip(), "SYS")
                args = (self.root / "args").read_text().splitlines()
                self.assertIn("--nproc_per_node=8", args)
                self.assertIn("parallel.standard_exchange=broadcast", args)
                self.assertEqual(args[-1], "render.out=a b.mp4")

    def test_malformed_benchmark_inputs_stop_before_probe(self):
        for args in ((), ("label",), ("label", "124", "extra"),
                     ("../escape", "124"), ("-bad", "124"),
                     ("a" * 65, "124"), ("a b", "124"),
                     ("label", "125"), ("label", "124;false")):
            with self.subTest(args=args):
                result = self.run_script(BENCHMARK, *args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((self.root / "queries").exists())
                self.assertFalse((self.root / "args").exists())

    def test_each_existing_output_is_preserved_before_probe(self):
        outputs = (
            "artifacts/rtx5090/8gpu-safe.log",
            "artifacts/rtx5090/8gpu-safe-activity.log",
            "artifacts/rtx5090/8gpu-safe-exit.txt",
            "results/8nfe_rtx5090_8gpu_safe.mp4",
            "results/8nfe_rtx5090_8gpu_safe.mp4.inference.json",
            "artifacts/latents/8nfe_rtx5090_8gpu_safe.mp4.latents.pt",
        )
        for name in outputs:
            with self.subTest(name=name):
                target = self.root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"keep-existing-output")
                result = self.run_script(BENCHMARK, "safe", "124")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("Refusing existing artifact", result.stderr)
                self.assertEqual(target.read_bytes(), b"keep-existing-output")
                self.assertFalse((self.root / "queries").exists())
                self.assertFalse((self.root / "args").exists())
                target.unlink()

    def test_busy_gpu_guards_never_reach_fake_renderer(self):
        for script, args in ((TRANSPORT, ()), (BENCHMARK, ("busy", "124"))):
            with self.subTest(script=script):
                result = self.run_script(script, *args, FAKE_APPS="12345\n")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertFalse((self.root / "args").exists())
        self.assertFalse((self.root / "artifacts").exists())

    def test_benchmark_probe_failure_is_fail_closed(self):
        result = self.run_script(BENCHMARK, "probe", "124", FAKE_PROBE_STATUS="9")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertFalse((self.root / "args").exists())
        self.assertFalse((self.root / "artifacts").exists())

    def test_benchmark_success_failure_and_no_clobber_rerun(self):
        for frames, status in (("124", "0"), ("345", "7")):
            with self.subTest(frames=frames, status=status):
                label = "run_" + frames
                result = self.run_script(BENCHMARK, label, frames,
                                         FAKE_RENDER_STATUS=status)
                self.assertEqual(result.returncode, int(status), result.stderr)
                self.assertEqual((self.root / "setting").read_text(), "0\n")
                args = (self.root / "args").read_text().splitlines()
                self.assertEqual(args[-3:], ["render.num_frames=" + frames,
                    "render.out=results/8nfe_rtx5090_8gpu_" + label + ".mp4",
                    "render.save_latents=true"])
                stem = self.root / ("artifacts/rtx5090/8gpu-" + label)
                self.assertEqual(Path(str(stem) + "-exit.txt").read_text(), status + "\n")
                log = Path(str(stem) + ".log")
                before = log.read_bytes()
                rerun = self.run_script(BENCHMARK, label, frames)
                self.assertEqual(rerun.returncode, 1)
                self.assertEqual(log.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
