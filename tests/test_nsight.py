"""CPU-only Nsight tests: torch is replaced with a fake, never imported here."""

import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_helper(nvtx="0", capture="0"):
    spec = importlib.util.spec_from_file_location(
        "_test_nsight_helper", ROOT / "src/inference/utils/nsight.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("os.environ", VDN_NVTX=nvtx, VDN_NSYS_CAPTURE=capture):
        spec.loader.exec_module(module)
    return module


class NsightTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.cuda = SimpleNamespace(
            nvtx=SimpleNamespace(
                range_push=lambda name: self.events.append(("push", name)),
                range_pop=lambda: self.events.append(("pop",))),
            profiler=SimpleNamespace(
                start=lambda: self.events.append(("start",)),
                stop=lambda: self.events.append(("stop",))),
            synchronize=lambda device: self.events.append(("sync", device)),
        )

    @contextmanager
    def fake_torch(self):
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=self.cuda)):
            yield

    def test_disabled_does_not_import_torch_or_access_cuda(self):
        # A None entry makes any attempted import fail, even on a CUDA host.
        with patch.dict(sys.modules, torch=None):
            helper = load_helper()
            function = lambda: 17
            self.assertIs(helper.nvtx_annotate("disabled")(function), function)
            with helper.nvtx_range("outer"), helper.nvtx_range("inner"):
                self.assertEqual(function(), 17)
            capture = helper.NsightCapture("unused")
            capture.start()
            for step in range(8):
                with capture.step(step, last=step == 7):
                    pass
            with capture.decode():
                pass
            capture.stop()
            self.assertFalse(capture.active)

    def test_only_literal_one_enables_flags(self):
        for value in ("", "true", "yes", "01"):
            helper = load_helper(value, value)
            self.assertFalse(helper.NVTX_ENABLED)
            self.assertFalse(helper.CAPTURE_ENABLED)

    def test_instance_forward_wrapper_disabled_identity_and_enabled_balance(self):
        module = SimpleNamespace(forward=lambda value: value + 1)
        original = module.forward
        helper = load_helper()
        helper.annotate_forward(module, "FFN")
        self.assertIs(module.forward, original)
        helper = load_helper(nvtx="1")
        helper.annotate_forward(module, "FFN")
        installed = module.forward
        helper.annotate_forward(module, "FFN")
        self.assertIs(module.forward, installed)
        with self.fake_torch():
            self.assertEqual(module.forward(2), 3)
        self.assertEqual(self.events, [("push", "FFN"), ("pop",)])

    def test_nested_ranges_balance_on_exception_without_sync(self):
        helper = load_helper(nvtx="1")

        @helper.nvtx_annotate("function")
        def fail():
            with helper.nvtx_range("nested"):
                raise ValueError("injected")

        with self.fake_torch(), self.assertRaisesRegex(ValueError, "injected"):
            fail()
        self.assertEqual(self.events, [
            ("push", "function"), ("push", "nested"), ("pop",), ("pop",)])

    def test_capture_two_windows_exact_order_for_all_eight_ranks(self):
        helper = load_helper(capture="1")
        with self.fake_torch():
            for rank in range(8):
                self.events.clear()
                capture = helper.NsightCapture(f"cuda:{rank}")
                self.assertEqual(self.events, [])
                for step in range(8):
                    with capture.step(step, last=step == 7):
                        self.events.append(("step", step, capture.active))
                with capture.decode():
                    self.events.append(("decode", rank))
                    self.events.append(("collective_complete",))
                capture.stop()  # Cleanup is idempotent.
                self.assertEqual(self.events, [
                    ("step", 0, False), ("step", 1, False), ("start",),
                    ("step", 2, True), ("step", 3, True),
                    ("sync", f"cuda:{rank}"), ("stop",),
                    *[("step", step, False) for step in range(4, 8)],
                    ("start",), ("decode", rank), ("collective_complete",),
                    ("sync", f"cuda:{rank}"), ("stop",),
                ])

    def test_short_sampling_and_denoise_only_close_active_capture(self):
        helper = load_helper(capture="1")
        with self.fake_torch():
            for count in (1, 2, 3, 4, 8):
                self.events.clear()
                capture = helper.NsightCapture("device")
                for index in range(count):
                    with capture.step(index, last=index == count - 1):
                        pass
                self.assertFalse(capture.active)
                self.assertEqual(self.events, [] if count < 3 else [
                    ("start",), ("sync", "device"), ("stop",)])

    def test_failure_closes_ranges_then_active_capture(self):
        helper = load_helper(nvtx="1", capture="1")
        with self.fake_torch():
            capture = helper.NsightCapture("device")
            for context in (capture.step(2), capture.decode()):
                self.events.clear()
                with self.assertRaisesRegex(RuntimeError, "injected"), context:
                    raise RuntimeError("injected")
                self.assertEqual(self.events[0], ("start",))
                self.assertEqual(self.events[-3:], [("pop",), ("sync", "device"), ("stop",)])
                self.assertFalse(capture.active)

    def test_nvtx_only_never_starts_profiler_or_synchronizes(self):
        helper = load_helper(nvtx="1")
        with self.fake_torch():
            capture = helper.NsightCapture("unused")
            with capture.step(2), capture.decode():
                pass
        self.assertEqual([e[0] for e in self.events], ["push", "push", "pop", "pop"])

    def test_failed_synchronize_still_stops_once(self):
        helper = load_helper(capture="1")
        self.cuda.synchronize = Mock(side_effect=RuntimeError("sync failed"))
        with self.fake_torch():
            capture = helper.NsightCapture("device")
            capture.start()
            capture.start()
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                capture.stop()
            capture.stop()
        self.assertFalse(capture.active)
        self.assertEqual(self.events, [("start",), ("stop",)])

    def test_production_sampler_wires_step_before_transformer(self):
        tree = ast.parse((ROOT / "src/inference/render.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "generate_latents")
        loop = next(n for n in function.body if isinstance(n, ast.For)
                    and "step_index" in ast.unparse(n.target))
        text = ast.unparse(loop)
        self.assertLess(text.index("nsight_capture.step"), text.index("noise_pred, audio_noise_pred = transformer"))
        self.assertIn("last=step_index == num_steps - 1", text)


if __name__ == "__main__":
    unittest.main()
