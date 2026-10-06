"""Opaque MXFP8 buffer transport, not quantization or GEMM numerical validation.

CPU tests never initialize CUDA. The NCCL test is gated by
VDN_TEST_MXFP8_SHARDING_NCCL=1 and must run under a two-rank torchrun.
The file can also be passed directly to torchrun (it invokes pytest).
"""
from contextlib import contextmanager, ExitStack, nullcontext
from datetime import timedelta
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.utils.weight_sharding import ResidentWeightShards, tensor_slots


class PackedLinear(torch.nn.Module):
    """Parent wrapper's storage contract, without importing its GPU kernels."""

    def __init__(self, index, device, *, bias):
        super().__init__()
        self.in_features, self.out_features = 61, 7
        with torch.device(device):
            # Real E4M3 encodings, including negative values, carried as uint8.
            data = ((torch.arange(128 * 128, dtype=torch.float32) + index) % 31 - 15)
            self.register_buffer("weight_data", data.to(torch.float8_e4m3fn).view(torch.uint8)
                                 .reshape(128, 128))
            # Nonuniform E8M0 exponent bytes in a flat, permuted storage order.
            scales = ((torch.arange(512, dtype=torch.int32) + index) % 16 + 120).to(torch.uint8)
            self.register_buffer("weight_scale", scales.reshape(4, 32, 4)
                                 .permute(0, 2, 1).contiguous().flatten())
            self.register_parameter("bias", torch.nn.Parameter(
                torch.arange(7, dtype=torch.bfloat16) + index, requires_grad=False) if bias else None)


class PackedModel(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        with torch.device(device):
            self.root_weight = torch.nn.Parameter(torch.arange(7, dtype=torch.bfloat16),
                                                  requires_grad=False)
            self.register_buffer("root_scale", torch.tensor([0.25, 2.0], dtype=torch.float32))
            self.transformer_blocks = torch.nn.ModuleList()
            shared = torch.tensor([0, 127, 128, 254, 255], dtype=torch.uint8)
            for index in range(3):
                block = torch.nn.Module()
                block.packed = PackedLinear(index, device, bias=index != 1)
                block.remaining = torch.nn.Parameter(
                    (torch.arange(21, dtype=torch.bfloat16) + index).reshape(7, 3),
                    requires_grad=False)
                block.register_buffer("fp32", torch.tensor([index + 0.25], dtype=torch.float32))
                block.register_buffer("counter", torch.arange(3, dtype=torch.int64), persistent=False)
                block.register_buffer("shared", shared)
                self.transformer_blocks.append(block)
            first = self.transformer_blocks[0].packed
            first.register_buffer("data_alias", first.weight_data)
        self.eval().requires_grad_(False)


def _runtime(device, rank=0, world_size=2):
    return SimpleNamespace(device=torch.device(device), rank=rank, world_size=world_size,
                           backend="nccl" if torch.device(device).type == "cuda" else "gloo",
                           branch_parallel=False, timeout_seconds=60,
                           profile_start=lambda: None, profile_end=lambda *args: None)


def _expected(device="cpu"):
    return {e.names[0]: e.tensor().detach().clone().to(device) for e in tensor_slots(PackedModel())}


def _raw_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    assert torch.equal(actual.detach().reshape(-1).view(torch.uint8),
                       expected.detach().reshape(-1).view(torch.uint8))


def _flat(group, expected, world_size):
    values = torch.cat([expected[e.names[0]].reshape(-1) for e in group.entries])
    padded = values.new_zeros(group.shard_numel * world_size)
    padded[:values.numel()].copy_(values)
    return padded


def _released(control):
    assert control.released and not control.initialized
    assert control.active is control.pending is control._active_gather is None
    assert control.weight_group is None
    assert all(e.tensor().is_meta for e in control.entries)
    assert all(g.shard is None for gs in control.groups for g in gs)


@contextmanager
def _no_h2d():
    """Reject common host weight upload paths after startup, without a profiler."""
    original_to, original_copy = torch.Tensor.to, torch.Tensor.copy_

    def to(tensor, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if isinstance(target, torch.Tensor):
            target = target.device
        if isinstance(target, int):
            target = torch.device("cuda", target)
        if isinstance(target, (str, torch.device)):
            assert not (tensor.device.type == "cpu" and torch.device(target).type == "cuda"), "H2D to"
        return original_to(tensor, *args, **kwargs)

    def copy(tensor, source, *args, **kwargs):
        assert not (source.device.type == "cpu" and tensor.device.type == "cuda"), "H2D copy"
        return original_copy(tensor, source, *args, **kwargs)

    with patch.object(torch.Tensor, "to", to), patch.object(torch.Tensor, "copy_", copy), \
            patch.object(torch.Tensor, "cuda", side_effect=AssertionError("H2D cuda")):
        yield


def _check_root(control, expected, pointers):
    for entry in control.root:
        value = entry.tensor()
        _raw_equal(value, expected[entry.names[0]])
        assert value.data_ptr() == pointers[entry.names[0]]
    blocks = control.model.transformer_blocks
    assert blocks[0].shared is blocks[1].shared is blocks[2].shared


def _rounds(control, expected, rounds=4):
    roots = {e.names[0]: e.tensor().data_ptr() for e in control.root}
    shards = [g.shard.data_ptr() for gs in control.groups for g in gs]
    expected_results = []
    for index in range(3):
        prefix = f"transformer_blocks.{index}."
        expected_results.append(expected[prefix + "packed.weight_data"].float().sum()
                                + expected[prefix + "packed.weight_scale"].float().sum()
                                + expected[prefix + "remaining"].float().square().sum())
    start = control.gather_count
    with torch.no_grad(), _no_h2d():
        for _ in range(rounds):
            with control.forward():
                for index, block in enumerate(control.model.transformer_blocks):
                    assert all(e.tensor().is_meta for e in control.blocks[index])
                    with control.block(index):
                        for entry in control.blocks[index]:
                            _raw_equal(entry.tensor(), expected[entry.names[0]])
                        packed = block.packed
                        assert (packed.in_features, packed.out_features) == (61, 7)
                        assert not hasattr(packed, "weight")
                        assert "weight_data" in packed.state_dict()
                        assert "weight_scale" in packed.state_dict()
                        assert "weight_data" not in dict(packed.named_parameters())
                        if index == 1:
                            assert packed.bias is None
                        else:
                            assert packed.bias.dtype == torch.bfloat16
                        if index == 0:
                            assert packed.weight_data is packed.data_alias
                        control.prefetch_next()
                        # Queue a consumer while the next block may be gathering.
                        # This is a lifetime check, deliberately not an MXFP8 GEMM.
                        result = (packed.weight_data.float().sum() + packed.weight_scale.float().sum()
                                  + block.remaining.float().square().sum())
                        if control.prefetch:
                            live = int(control._active_gather is not None) + int(control.pending is not None)
                            assert live == (2 if index < 2 else 1)
                            if index < 2:
                                assert all(e.tensor().is_meta for e in control.blocks[index + 1])
                                for group, flat in zip(control.groups[index + 1], control.pending.buffers):
                                    # Wait solely for inspection; production acquisition owns this fence.
                                    control.pending.ready.synchronize()
                                    _raw_equal(flat, _flat(group, expected, control.world_size))
                    assert all(e.tensor().is_meta for e in control.blocks[index])
                    _raw_equal(result, expected_results[index])
                    _check_root(control, expected, roots)
            assert control.active is control.pending is control._active_gather is None
            assert [g.shard.data_ptr() for gs in control.groups for g in gs] == shards
    assert control.gather_count == start + rounds * 3
    if control.prefetch:
        assert control.max_live_gathered == 2


def _body_failure(control, expected):
    with torch.no_grad(), pytest.raises(RuntimeError, match="injected body"), control.forward():
        with control.block(0):
            control.prefetch_next()
            raise RuntimeError("injected body")
    assert control.active is control.pending is control._active_gather is None
    assert all(e.tensor().is_meta for es in control.blocks for e in es)
    assert all(not e.tensor().is_meta for e in control.root)
    _rounds(control, expected, rounds=1)


def _collective_failure(control):
    gather = dist.all_gather_into_tensor
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Every rank fails before launching the same second dtype collective.
        if calls == 2:
            raise RuntimeError("injected gather")
        return gather(*args, **kwargs)

    with patch.object(dist, "all_gather_into_tensor", side_effect=fail):
        with torch.no_grad(), pytest.raises(RuntimeError, match="injected gather"), control.forward():
            with control.block(0):
                pytest.fail("partial gather must not be consumable")
    assert control.active is control.pending is control._active_gather is None
    assert all(e.tensor().is_meta for es in control.blocks for e in es)
    if control.prefetch:
        assert control._failed
        with pytest.raises(RuntimeError), control.forward():
            pass


def _exercise(runtime, distribution, prefetch):
    control = ResidentWeightShards(PackedModel("cpu" if runtime.rank == 0 else "meta"), runtime,
                                   distribution=distribution, scatter_chunk_bytes=4093, prefetch=prefetch)
    expected = _expected(runtime.device)
    try:
        control.distribute()
        for groups in control.groups:
            assert {g.dtype for g in groups} == {torch.uint8, torch.bfloat16, torch.float32, torch.int64}
            for group in groups:
                full = _flat(group, expected, runtime.world_size)
                _raw_equal(group.shard, full.chunk(runtime.world_size)[runtime.rank])
        _rounds(control, expected)
        _body_failure(control, expected)
        _collective_failure(control)
    finally:
        control.release()
    control.release()
    _released(control)


def _cpu_worker(rank, rendezvous, distribution):
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                                timeout=timedelta(seconds=40))
        try:
            _exercise(_runtime("cpu", rank), distribution, False)
            # Graphs agree but collective schedules do not: fail before payload.
            for options in ({"distribution": "scatter" if rank == 0 else "broadcast"},
                            {"scatter_chunk_bytes": 13 + rank}):
                control = ResidentWeightShards(PackedModel("cpu" if rank == 0 else "meta"),
                                               _runtime("cpu", rank), **options)
                with patch.object(dist, "broadcast", side_effect=AssertionError("payload forbidden")), \
                        patch.object(dist, "scatter", side_effect=AssertionError("payload forbidden")):
                    with pytest.raises(ValueError, match="transport options"):
                        control.distribute()
                control.release()
                _released(control)
            # Fail after some root and BF16 startup payloads have completed,
            # specifically when distribution reaches a packed uint8 shard.
            control = ResidentWeightShards(PackedModel("cpu" if rank == 0 else "meta"),
                                           _runtime("cpu", rank), distribution="broadcast")
            broadcast = dist.broadcast
            uint8_calls = 0

            def fail_uint8(tensor, *args, **kwargs):
                nonlocal uint8_calls
                if tensor.dtype == torch.uint8:
                    uint8_calls += 1
                    if uint8_calls == 2:
                        raise RuntimeError("packed startup failure")
                return broadcast(tensor, *args, **kwargs)

            with patch.object(dist, "broadcast", side_effect=fail_uint8):
                with pytest.raises(RuntimeError, match="packed startup failure"):
                    control.distribute()
            _released(control)
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("distribution", ["scatter", "broadcast"])
def test_cpu_two_rank_exact_buffers_and_mixed_parameters(tmp_path, distribution):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_cpu_worker,
                 args=(rank, "file://" + str(tmp_path / "pg"), distribution)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 90
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [p.exitcode for p in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()


class _Event:
    def __init__(self):
        self.recorded = self.finished = False

    def record(self, stream):
        self.recorded = True

    def synchronize(self):
        assert self.recorded
        self.finished = True


class _Stream:
    def synchronize(self):
        pass

    def wait_event(self, event):
        assert event.finished


@pytest.fixture
def fake_prefetch():
    """Real byte payloads; fake only CUDA scheduling and distributed transport."""
    control = ResidentWeightShards(PackedModel(), _runtime("cuda:0"), prefetch=True)
    expected = _expected()
    fulls = {}
    for groups in control.groups:
        for group in groups:
            full = _flat(group, expected, 2)
            group.shard = full[:group.shard_numel].clone()
            fulls[id(group.shard)] = full
    for entries in control.blocks:
        for entry in entries:
            entry.dematerialize()
    control.initialized = True
    control.weight_group = object()
    control.prefetch_stream = _Stream()
    empty = torch.empty

    def gather(out, shard, *, group, async_op):
        assert group is control.weight_group and async_op
        out.copy_(fulls[id(shard)])
        return SimpleNamespace(wait=lambda: None)

    def cpu_empty(*args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            kwargs["device"] = "cpu"
        return empty(*args, **kwargs)

    with ExitStack() as stack:
        for target, replacement in [
            ("torch.empty", cpu_empty), ("torch.cuda.Event", _Event),
            ("torch.cuda.current_stream", lambda *args: _Stream()),
            ("torch.cuda.stream", lambda stream: nullcontext()),
            ("torch.distributed.all_gather_into_tensor", gather),
            ("torch.distributed.destroy_process_group", lambda group: None),
        ]:
            stack.enter_context(patch(target, replacement))
        stack.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")))
        try:
            yield control, expected
        finally:
            control.release()


def test_cpu_prefetch_exact_repeated_buffers_root_and_body_failure(fake_prefetch):
    control, expected = fake_prefetch
    _rounds(control, expected)
    _body_failure(control, expected)
    _collective_failure(control)
    control.release()
    _released(control)


def test_cpu_caught_partial_enqueue_cannot_be_consumed(fake_prefetch):
    control, _ = fake_prefetch
    with torch.no_grad(), control.forward():
        with control.block(0):
            with patch.object(dist, "all_gather_into_tensor", side_effect=RuntimeError("tail failed")):
                with pytest.raises(RuntimeError, match="tail failed"):
                    control.prefetch_next()
            assert control._failed and control.pending.ready is None
        with pytest.raises(RuntimeError, match="ordered forward"), control.block(1):
            pass
    assert control.pending is None
    with pytest.raises(RuntimeError), control.forward():
        pass


def test_cpu_failed_drain_retains_byte_storage_until_safe_release(fake_prefetch):
    control, _ = fake_prefetch
    control._enqueue(0)
    pending = control.pending
    buffers = list(pending.buffers)
    with patch.object(pending.ready, "synchronize", side_effect=RuntimeError("drain failed")):
        with pytest.raises(RuntimeError, match="drain failed"):
            control.release()
        assert control._failed and not control.released
        assert control.pending is pending
        assert all(a is b for a, b in zip(pending.buffers, buffers))
        assert any(t.dtype == torch.uint8 for t in pending.buffers)
        assert all(g.shard is not None for gs in control.groups for g in gs)
    control.release()
    _released(control)


def test_cpu_readiness_failure_is_terminal_even_if_retry_drain_succeeds(fake_prefetch):
    control, _ = fake_prefetch
    with torch.no_grad(), control.forward():
        control._enqueue(0)
        with patch.object(control.pending.ready, "synchronize", side_effect=RuntimeError("ready failed")):
            with pytest.raises(RuntimeError, match="ready failed"), control.block(0):
                pass
        assert control._failed
    assert control.pending is None
    with pytest.raises(RuntimeError), control.forward():
        pass


def test_cpu_partial_buffer_binding_failure_dematerializes_and_can_retry(fake_prefetch):
    control, expected = fake_prefetch
    scale = next(e for e in control.blocks[0] if e.names[0].endswith("weight_scale"))
    with patch.object(scale, "bind", side_effect=RuntimeError("scale bind failed")):
        with torch.no_grad(), pytest.raises(RuntimeError, match="scale bind failed"), control.forward():
            with control.block(0):
                pytest.fail("partial buffer binding must not yield")
    assert all(e.tensor().is_meta for e in control.blocks[0])
    assert control.active is control.pending is control._active_gather is None
    _rounds(control, expected, rounds=1)


def test_cpu_manifest_matches_meta_graph_and_rejects_noncontiguous_buffers():
    owner = ResidentWeightShards(PackedModel(), _runtime("cpu"))
    peer = ResidentWeightShards(PackedModel("meta"), _runtime("cpu", rank=1))
    assert owner.manifest() == peer.manifest()
    uint8 = [e for es in owner.blocks for e in es if e.dtype == torch.uint8]
    assert len(uint8) == 6  # data and scale per block, identity aliases deduplicated
    assert all(not e.parameter and e.placeholder.is_meta for e in uint8)
    model = PackedModel()
    model.transformer_blocks[0].packed.weight_data = model.transformer_blocks[0].packed.weight_data.t()
    with pytest.raises(ValueError, match="ordinary contiguous"):
        ResidentWeightShards(model, _runtime("cpu"))


@pytest.fixture(scope="module")
def nccl_session():
    assert int(os.environ.get("WORLD_SIZE", "0")) == 2, "run under torchrun --nproc-per-node=2"
    assert not dist.is_initialized(), "tests own their default process group"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        yield _runtime(f"cuda:{local_rank}", dist.get_rank())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("VDN_TEST_MXFP8_SHARDING_NCCL") != "1",
                    reason="parent-owned opt-in two-rank NCCL test")
def test_nccl_two_rank_exact_mxfp8_transport(nccl_session):
    # Do not even query CUDA availability until explicitly opted in.
    # Broadcast avoids the documented startup scatter stall on this host.
    distribution = os.environ.get("VDN_TEST_MXFP8_DISTRIBUTION", "broadcast")
    for prefetch in (False, True):
        _exercise(nccl_session, distribution, prefetch)
        dist.barrier()


@pytest.mark.skipif(os.environ.get("VDN_TEST_MXFP8_SHARDING_NCCL") != "1",
                    reason="parent-owned opt-in two-rank native wrapper test")
@torch.no_grad()
def test_nccl_native_wrappers_exact_bytes_and_repeated_gemm(nccl_session):
    from src.models.ops.mxfp8_linear import MXFP8Linear

    def model(device):
        torch.manual_seed(1701)
        result = torch.nn.Module()
        with torch.device(device):
            result.root = torch.nn.Linear(37, 37, dtype=torch.bfloat16)
            result.transformer_blocks = torch.nn.ModuleList([
                MXFP8Linear(torch.nn.Linear(37, 37, dtype=torch.bfloat16)) for _ in range(3)])
        return result.eval().requires_grad_(False)

    control = None
    try:
        runtime = nccl_session
        reference = model("cpu").to(runtime.device)
        expected = {e.names[0]: e.tensor().clone() for e in tensor_slots(reference)}
        x = torch.randn(129, 37, dtype=torch.bfloat16, device=runtime.device)
        expected_y = [p(x) for p in reference.transformer_blocks]
        candidate = model("cpu" if runtime.rank == 0 else "meta")
        control = ResidentWeightShards(candidate, runtime, distribution="broadcast", prefetch=True)
        control.distribute()
        shard_pointers = [g.shard.data_ptr() for groups in control.groups for g in groups]
        # NCCL lazily initializes the dedicated weight communicator on first use.
        # Warm it before the steady-state trace, including its host metadata copies.
        with control.forward():
            for index, block in enumerate(candidate.transformer_blocks):
                with control.block(index):
                    control.prefetch_next()
                    assert torch.equal(block(x), expected_y[index])
        start_gathers = control.gather_count
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA])
        with profiler, _no_h2d():
            for _ in range(4):
                outputs = []
                with control.forward():
                    for index, block in enumerate(candidate.transformer_blocks):
                        with control.block(index):
                            for entry in control.blocks[index]:
                                _raw_equal(entry.tensor(), expected[entry.names[0]])
                            control.prefetch_next()
                            outputs.append(block(x))
                        assert block.weight_data.is_meta and block.weight_scale.is_meta
                for result, oracle in zip(outputs, expected_y):
                    assert torch.equal(result, oracle)
                assert control.pending is control.active is control._active_gather is None
                assert [g.shard.data_ptr() for groups in control.groups for g in groups] == shard_pointers
                for entry in control.root:
                    _raw_equal(entry.tensor(), expected[entry.names[0]])
        assert control.max_live_gathered == 2 and control.gather_count == start_gathers + 12
        report = os.environ.get("MXFP8_TRANSPORT_REPORT_DIR")
        if report:
            destination = Path(report)
            destination.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(destination / f"native-rank-{runtime.rank}.trace.json"))
        # Body failure drains the pending next block without invalidating shards.
        with pytest.raises(RuntimeError, match="native body"), control.forward():
            with control.block(0):
                control.prefetch_next()
                candidate.transformer_blocks[0](x)
                raise RuntimeError("native body")
        assert control.pending is control.active is control._active_gather is None
        control.release()
        control.release()
        _released(control)
        dist.barrier()
    finally:
        if control is not None:
            control.release()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
