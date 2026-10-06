"""Synthetic eager head-shard regressions, without weights or distributed workers.

CPU FP32 is the default. CUDA BF16 requires RUN_ULYSSES_REFERENCE_CUDA=1;
even CUDA availability is not queried unless that opt-in is set. These are
tolerance comparisons, not bitwise shard parity or multi-GPU validation.

The full-width comparison patches features._temporal_conv to _temporal_shift so
the whole comparison runs eager on CPU. The shard path now calls the SAME
_temporal_conv helper as the full-width reference: an earlier version called
eager _temporal_shift directly, which diverged from the reference's
inductor-compiled temporal conv by ~8.5e-3 relative L2 at bf16 and inflated
trajectory drift. On CPU the helper is eager either way, so the comparison stays
bitwise-friendly. A separate test preserves coverage of the original full-width
temporal dispatch.
"""

from contextlib import ExitStack, contextmanager
from itertools import combinations, product
import os
import unittest
from unittest.mock import patch

import torch

from src.models.linear_attention import branch as branch_module
from src.models.linear_attention import features, kernels, scan
from src.models.linear_attention.branch import BidirectionalLinearBranch
from src.models.linear_attention.delta_rule import DELTA_BACKENDS
from src.models.ops import temporal_conv


FP32_RTOL, FP32_ATOL = 2e-5, 2e-6
BF16_RTOL, BF16_ATOL = 3e-2, 2e-2
SHORT_CONVS = tuple(
    subset for size in range(4) for subset in combinations(("q", "k", "v"), size)
)
HEAD_SLICES = (slice(0, 1), slice(1, 3), slice(3, 5))


@contextmanager
def eager_only(*, shard):
    """Fail on fused entry points, including already-cached compiled dispatch."""
    def forbidden(*args, **kwargs):
        raise AssertionError("reference shard entered a fused/compiled path")

    original_statistics = branch_module.frame_statistics
    original_gather = branch_module.gather_linear_state

    def statistics(*args, **kwargs):
        if kwargs.get("inference", False):
            forbidden()
        return original_statistics(*args, **kwargs)

    def gather(*args, **kwargs):
        if kwargs.get("inference", False):
            forbidden()
        return original_gather(*args, **kwargs)

    with ExitStack() as stack:
        for module, name in (
            (torch, "compile"),
            (BidirectionalLinearBranch, "_readout_inference"),
            (branch_module, "prepare_linear_features_inference"),
            (branch_module, "_run_scans_inference"),
            (branch_module, "linear_epilogue"),
            (features, "prepare_linear_features_inference"),
            (features, "_compiled"),
            (features, "temporal_conv_activate"),
            (temporal_conv, "temporal_conv_activate"),
            (kernels, "linear_epilogue"),
            (scan, "_run_scans_inference"),
            (scan, "_tf32_matmul"),
        ):
            stack.enter_context(patch.object(module, name, side_effect=forbidden))
        stack.enter_context(patch.object(branch_module, "frame_statistics", statistics))
        stack.enter_context(patch.object(branch_module, "gather_linear_state", gather))
        # Empty these temporarily so pre-existing caches cannot hide compilation.
        for cache in (features._FEATURES_CACHE, scan._STATS_PREP_CACHE, scan._GATHER_CACHE):
            stack.enter_context(patch.dict(cache, {}, clear=True))
        # Both the full-width oracle and the shard now reach the temporal half
        # through _temporal_conv, so force it to the eager helper (no compilation
        # on CPU) and latch that helper rather than forbidding the entry point.
        stack.enter_context(patch.object(features, "_temporal_conv",
                                         features._temporal_shift))
        stack.enter_context(patch.dict(features._TCONV,
                                       {"fn": features._temporal_shift}, clear=True))
        yield


class ReferenceFixtures:
    """Helpers shared by CPU tests and explicitly opted-in CUDA tests."""

    def make_case(self, *, rule="vdn_solve", bridge="alpha", a_fp32=True,
                  short_conv=("k", "v"), frames=7, text=True,
                  device="cpu", dtype=torch.float32, text_length=5):
        heads, dim, hidden, per_frame = 5, 4, 19, 6
        # Preserve the CPU RNG, and never seed or initialize CUDA in default tests.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(7301)
            module = BidirectionalLinearBranch(
                hidden, heads, dim, delta_rule=rule, bridge=bridge,
                a_fp32=a_fp32, short_conv=short_conv,
            )
            with torch.no_grad():
                module.norm.weight.copy_(torch.linspace(0.7, 1.3, dim))
                if module.short_conv is not None:
                    # Nonidentity, channel-dependent taps detect wrong head offsets
                    # and temporal mixing across anchor frames before pruning.
                    channels = torch.arange(heads * dim).float().view(-1, 1, 1)
                    taps = torch.tensor([0.19, -0.13, 0.67, 0.23, -0.11])
                    for index, proj in enumerate(short_conv):
                        weight = getattr(module.short_conv, proj + "_tm").weight
                        weight.copy_(taps.view(1, 1, -1) * (1 + channels / 31 + index / 7))
            generator = torch.Generator(device="cpu").manual_seed(1907)

            def randn(*shape):
                return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

            case = {
                "xv": randn(frames * per_frame, hidden),
                "num_frames": frames,
                "tokens_per_frame": per_frame,
                "bounds": [(max(0, t - 1), min(frames - 1, t + 1))
                           for t in range(frames)],
                "qkv_raw": tuple(randn(frames * per_frame, heads, dim) for _ in range(3)),
                "frame_size": (2, 3),
                "text_x": randn(text_length, hidden) if text else None,
                "text_qkv_raw": (tuple(randn(text_length, heads, dim) for _ in range(3))
                                 if text else None),
            }
        return module.to(device=device, dtype=dtype).eval(), case

    def shard_arguments(self, module, case, heads):
        # Exactly the sequence-owner data: mean BEFORE head sharding, in FP32.
        with torch.no_grad():
            beta = torch.sigmoid(module.beta_proj(case["xv"]))
            gate = module.output_gate(case["xv"])
            mean = case["xv"].reshape(case["num_frames"], case["tokens_per_frame"], -1)
            mean = mean.mean(dim=1, dtype=torch.float32)
            text_beta = (None if case["text_x"] is None else
                         torch.sigmoid(module.beta_proj(case["text_x"])))
        arguments = dict(case, xv=None, heads=heads, beta=beta[:, heads],
                         gate=gate[:, heads], frame_mean=mean, text_x=None)
        arguments["qkv_raw"] = tuple(t[:, heads] for t in case["qkv_raw"])
        if case["text_qkv_raw"] is not None:
            arguments["text_qkv_raw"] = tuple(t[:, heads] for t in case["text_qkv_raw"])
            arguments["text_beta"] = text_beta[:, heads]
        return arguments

    def full_output(self, module, case, *, skip_ends=False):
        with torch.no_grad(), eager_only(shard=False):
            return module(**case, inference=False, skip_ends=skip_ends)

    def shard_output(self, module, case, *, skip_ends=False, direct=False):
        pieces = []
        for heads in HEAD_SLICES:
            arguments = self.shard_arguments(module, case, heads)
            self.assertIsNone(arguments["xv"])
            self.assertEqual(arguments["frame_mean"].dtype, torch.float32)
            self.assertEqual(arguments["frame_mean"].shape,
                             (case["num_frames"], case["xv"].shape[-1]))
            for tensor in (*arguments["qkv_raw"], arguments["beta"], arguments["gate"]):
                self.assertFalse(tensor.is_contiguous())
            if arguments["text_qkv_raw"] is not None:
                for tensor in (*arguments["text_qkv_raw"], arguments["text_beta"]):
                    self.assertFalse(tensor.is_contiguous())
            with torch.no_grad(), eager_only(shard=True):
                if direct:
                    piece = module._readout(**arguments)
                else:
                    piece = module(**arguments, inference=False, skip_ends=skip_ends)
            self.assertEqual(piece.shape, (case["xv"].shape[0],
                                          (heads.stop - heads.start) * module.head_dim))
            self.assertEqual(piece.dtype, case["xv"].dtype)
            self.assertTrue(torch.isfinite(piece).all())
            pieces.append(piece)
        return torch.cat(pieces, dim=-1)

    def compare(self, module, case, *, skip_ends=False, direct=False):
        expected = self.full_output(module, case, skip_ends=skip_ends)
        actual = self.shard_output(module, case, skip_ends=skip_ends, direct=direct)
        rtol, atol = ((BF16_RTOL, BF16_ATOL) if expected.dtype == torch.bfloat16
                      else (FP32_RTOL, FP32_ATOL))
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        if skip_ends:
            per_frame = case["tokens_per_frame"]
            self.assertEqual(torch.count_nonzero(actual[:per_frame]).item(), 0)
            self.assertEqual(torch.count_nonzero(actual[-per_frame:]).item(), 0)
        return actual


class UlyssesReferenceTests(ReferenceFixtures, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny depthwise convolutions/Cholesky batches otherwise oversubscribe CPUs.
        previous = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_cpu_fp32_delta_bridge_precision_and_convolution_matrix(self):
        for rule, bridge, a_fp32, short_conv in product(
            DELTA_BACKENDS, scan.BRIDGE_MODES, (False, True), SHORT_CONVS
        ):
            with self.subTest(rule=rule, bridge=bridge, a_fp32=a_fp32,
                              short_conv=short_conv):
                module, case = self.make_case(rule=rule, bridge=bridge,
                                             a_fp32=a_fp32, short_conv=short_conv)
                actual = self.compare(module, case)
                self.assertGreater(actual.abs().max().item(), 1e-4)

    def test_anchor_pruning_with_and_without_text(self):
        for rule, bridge, text in product(DELTA_BACKENDS, scan.BRIDGE_MODES, (False, True)):
            with self.subTest(rule=rule, bridge=bridge, text=text):
                module, case = self.make_case(rule=rule, bridge=bridge, text=text,
                                             short_conv=("q", "k", "v"))
                expected = self.compare(module, case, skip_ends=True)
                changed = dict(case, xv=case["xv"].clone(),
                               qkv_raw=tuple(t.clone() for t in case["qkv_raw"]))
                for tensor in (changed["xv"], *changed["qkv_raw"]):
                    tensor[:case["tokens_per_frame"]].add_(71)
                    tensor[-case["tokens_per_frame"]:].mul_(-39)
                # Neither convolution nor statistics may see excluded anchors.
                actual = self.compare(module, changed, skip_ends=True)
                torch.testing.assert_close(actual, expected, rtol=FP32_RTOL, atol=FP32_ATOL)
                self.assertGreater(actual[6:-6].abs().max().item(), 1e-4)

    def test_text_state_changes_readout_even_when_window_covers_all_video(self):
        for rule, bridge in product(DELTA_BACKENDS, scan.BRIDGE_MODES):
            with self.subTest(rule=rule, bridge=bridge):
                module, case = self.make_case(rule=rule, bridge=bridge)
                case["bounds"] = [(0, case["num_frames"] - 1)] * case["num_frames"]
                baseline = self.compare(module, case)
                changed = dict(case, text_x=case["text_x"] + 2,
                               text_qkv_raw=(case["text_qkv_raw"][0],
                                             case["text_qkv_raw"][1] * -2,
                                             case["text_qkv_raw"][2] + 3))
                actual = self.compare(module, changed)
                self.assertGreater((actual - baseline).abs().max().item(), 1e-3)
                empty = self.compare(module, dict(case, text_x=None, text_qkv_raw=None))
                self.assertEqual(torch.count_nonzero(empty).item(), 0)

    def test_short_clips_and_anchor_only_early_return(self):
        for frames, text, skip_ends in product((1, 2, 3), (False, True), (False, True)):
            with self.subTest(frames=frames, text=text, skip_ends=skip_ends):
                module, case = self.make_case(frames=frames, text=text,
                                             short_conv=("q", "k", "v"))
                if frames <= 2 and skip_ends:
                    with patch.object(module, "_readout", side_effect=AssertionError(
                        "anchor-only clip must not enter readout"
                    )):
                        output = self.compare(module, case, skip_ends=True)
                    self.assertEqual(torch.count_nonzero(output).item(), 0)
                    self.assertIsNone(module.backend)
                    self.assertIsNone(module.text_backend)
                else:
                    self.compare(module, case, skip_ends=skip_ends)

    def test_direct_eager_readout_accepts_precomputed_shards(self):
        for rule in DELTA_BACKENDS:
            with self.subTest(rule=rule):
                module, case = self.make_case(rule=rule, short_conv=("q", "k", "v"))
                self.compare(module, case, direct=True)

    def test_reference_statistics_cross_eager_chunk_boundary(self):
        # Preserve the bounded no-grad eager statistics baseline documented for
        # RTX 5090; anchor pruning still leaves more than one statistics chunk.
        module, case = self.make_case(frames=scan.STATS_CHUNK_FRAMES + 5)
        self.compare(module, case, skip_ends=True)

    def test_shard_uses_reference_temporal_conv_helper(self):
        # The head-sharded path must run the SAME short-conv helper as the
        # single-rank reference (_temporal_conv), not the eager _temporal_shift.
        module, case = self.make_case(short_conv=("q", "k", "v"), text=False)
        with ExitStack() as stack:
            spies = []
            for name in ("_temporal_conv", "_activate_body"):
                original = getattr(features, name)
                spy = stack.enter_context(patch.object(features, name, wraps=original))
                # Accept either a module-qualified call or a directly imported
                # alias; lookup style is not part of the reference API contract.
                for alias, value in list(vars(branch_module).items()):
                    if value is original:
                        stack.enter_context(patch.object(branch_module, alias, spy))
                spies.append(spy)
            self.shard_output(module, case)
        temporal, activate = spies
        self.assertEqual(temporal.call_count, 3 * len(HEAD_SLICES))
        self.assertEqual(activate.call_count, 3 * len(HEAD_SLICES))

    def test_full_eager_keeps_existing_temporal_helper(self):
        module, case = self.make_case(short_conv=("q", "k", "v"), text=False)
        with torch.no_grad(), patch.dict(features._TCONV, {"fn": features._temporal_shift}):
            with patch.object(features, "_temporal_conv", wraps=features._temporal_conv) as temporal:
                with patch.object(torch, "compile", side_effect=AssertionError("CPU compile")):
                    actual = module(**case, inference=False)
        self.assertEqual(temporal.call_count, 3)
        torch.testing.assert_close(actual, self.full_output(module, case),
                                   rtol=FP32_RTOL, atol=FP32_ATOL)

    def test_every_precomputed_argument_rejects_enabled_grad_even_in_eval(self):
        module, case = self.make_case()
        module.requires_grad_(False)
        full = self.shard_arguments(module, case, slice(0, module.num_heads))
        for name in ("heads", "beta", "gate", "frame_mean", "text_beta"):
            for direct in (False, True):
                with self.subTest(argument=name, direct=direct):
                    arguments = dict(case, **{name: full[name]})
                    self.assertFalse(module.training)
                    self.assertFalse(any(p.requires_grad for p in module.parameters()))
                    self.assertFalse(case["xv"].requires_grad)
                    with torch.enable_grad(), self.assertRaises(RuntimeError):
                        if direct:
                            module._readout(**arguments)
                        else:
                            module(**arguments, inference=False)
        # Validation must precede the zero-output shortcut as well.
        module, case = self.make_case(frames=2)
        arguments = self.shard_arguments(module, case, slice(1, 3))
        with torch.enable_grad(), self.assertRaises(RuntimeError):
            module(**arguments, skip_ends=True, inference=False)

    def test_explicit_contiguous_bounded_head_slice_required(self):
        module, case = self.make_case()
        arguments = self.shard_arguments(module, case, slice(1, 3))
        invalid = (slice(None), slice(None, 3), slice(1, None), slice(-1, 2),
                   slice(0, 6), slice(2, 2), slice(3, 1), slice(0, 4, 2),
                   slice(3, 0, -1), [1, 2], 1)
        for heads in invalid:
            with self.subTest(heads=heads), torch.no_grad():
                with self.assertRaises((ValueError, TypeError)):
                    module(**dict(arguments, heads=heads), inference=False)

    def test_shards_require_owner_statistics_and_text_beta(self):
        module, case = self.make_case()
        arguments = self.shard_arguments(module, case, slice(1, 3))
        for missing in ("beta", "gate", "frame_mean", "text_beta"):
            with self.subTest(missing=missing), torch.no_grad():
                with self.assertRaises(ValueError):
                    module(**dict(arguments, **{missing: None}), inference=False)

    def test_shards_preserve_parameters_metadata_and_backend_cache_keys(self):
        module, case = self.make_case(short_conv=("q", "k", "v"))
        state = {name: tensor.clone() for name, tensor in module.state_dict().items()}
        parameter_ids = {name: id(value) for name, value in module.named_parameters()}
        metadata = (module.num_heads, module.head_dim, module.bridge, module.a_fp32,
                    module.delta_rule, module.checkpoint_features, module.training,
                    module.short_conv.projs, module.alpha.num_heads,
                    module.output_gate.num_heads)
        self.full_output(module, case)
        backend, text_backend = module.backend, module.text_backend
        self.compare(module, case)
        self.assertIs(module.backend, backend)
        self.assertIs(module.text_backend, text_backend)
        self.assertEqual(module.backend._key, (module.delta_rule, 6))
        self.assertEqual(module.text_backend._key, (module.delta_rule, 5))
        self.assertEqual(metadata, (module.num_heads, module.head_dim, module.bridge,
                                   module.a_fp32, module.delta_rule, module.checkpoint_features,
                                   module.training, module.short_conv.projs,
                                   module.alpha.num_heads, module.output_gate.num_heads))
        self.assertEqual(parameter_ids, {name: id(value) for name, value in module.named_parameters()})
        self.assertEqual(set(state), set(module.state_dict()))
        for name, tensor in module.state_dict().items():
            torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
        # A live rule switch must rebuild both caches, regardless of head width.
        module.delta_rule = "sana_scaled"
        self.shard_output(module, case)
        self.assertIsNot(module.backend, backend)
        self.assertIsNot(module.text_backend, text_backend)
        self.assertEqual(module.backend._key, ("sana_scaled", 6))
        self.assertEqual(module.text_backend._key, ("sana_scaled", 5))
        changed = dict(case, text_x=case["text_x"][:3],
                       text_qkv_raw=tuple(t[:3] for t in case["text_qkv_raw"]))
        self.compare(module, changed)
        self.assertEqual(module.text_backend._key, ("sana_scaled", 3))

    def test_full_eager_backward_still_reaches_inputs_and_parameters(self):
        for rule, skip_ends in product(DELTA_BACKENDS, (False, True)):
            with self.subTest(rule=rule, skip_ends=skip_ends):
                module, case = self.make_case(rule=rule, short_conv=("q", "k", "v"))
                module.train()
                tensors = (case["xv"], *case["qkv_raw"], case["text_x"],
                           *case["text_qkv_raw"][1:])
                for tensor in tensors:
                    tensor.requires_grad_()
                expected = self.full_output(module, case, skip_ends=skip_ends)
                with torch.enable_grad(), eager_only(shard=False):
                    actual = module(**case, inference=False, skip_ends=skip_ends)
                    actual.square().mean().backward()
                torch.testing.assert_close(actual.detach(), expected,
                                           rtol=FP32_RTOL, atol=FP32_ATOL)
                for tensor in (*tensors, *module.parameters()):
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(tensor.grad.abs().sum().item(), 0)
                if skip_ends:
                    for tensor in (case["xv"], *case["qkv_raw"]):
                        self.assertEqual(torch.count_nonzero(tensor.grad[:6]).item(), 0)
                        self.assertEqual(torch.count_nonzero(tensor.grad[-6:]).item(), 0)


@unittest.skipUnless(os.environ.get("RUN_ULYSSES_REFERENCE_CUDA") == "1",
                     "CUDA BF16 requires RUN_ULYSSES_REFERENCE_CUDA=1")
class UlyssesReferenceCudaTests(ReferenceFixtures, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("CUDA BF16 unavailable")

    def test_cuda_bf16_shards_against_entirely_eager_full_width(self):
        for rule, bridge, a_fp32, short_conv, skip_ends in product(
            DELTA_BACKENDS, scan.BRIDGE_MODES, (False, True),
            ((), ("k", "v"), ("q", "k", "v")), (False, True)
        ):
            with self.subTest(rule=rule, bridge=bridge, a_fp32=a_fp32,
                              short_conv=short_conv, skip_ends=skip_ends):
                module, case = self.make_case(rule=rule, bridge=bridge, a_fp32=a_fp32,
                                             short_conv=short_conv, device="cuda",
                                             dtype=torch.bfloat16)
                self.compare(module, case, skip_ends=skip_ends)

    def test_cuda_full_width_retains_temporal_compiler_dispatch(self):
        module, case = self.make_case(short_conv=("q", "k", "v"), text=False,
                                     device="cuda", dtype=torch.bfloat16)
        # Test the dispatch contract without actually compiling a GPU kernel.
        with torch.no_grad(), patch.dict(features._TCONV, {"fn": None}, clear=True):
            with patch.object(torch, "compile", return_value=features._temporal_shift) as compiler:
                actual = module(**case, inference=False)
        compiler.assert_called_once_with(features._temporal_shift)
        torch.testing.assert_close(actual, self.full_output(module, case),
                                   rtol=BF16_RTOL, atol=BF16_ATOL)


if __name__ == "__main__":
    unittest.main()
