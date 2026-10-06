"""Bounded eight-rank synthetic weight-prefetch safety probe, never a model run."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from src.inference.utils.weight_sharding import ResidentWeightShards
from src.inference.utils.ulysses_runtime import UlyssesRuntime


CHUNK = 1 << 20


def pattern(start, stop, block, group, device):
    # Position-dependent storage-byte oracle; no floating-point comparison/cast.
    ids = torch.arange(start, stop, dtype=torch.int64, device=device)
    return ((ids * 37 + ids // 251 + block * 43 + group * 71) % 256).to(torch.uint8)


def fill_bytes(tensor, offset, block, group):
    raw = tensor.view(torch.uint8).view(-1)
    for start in range(0, raw.numel(), CHUNK):
        stop = min(start + CHUNK, raw.numel())
        raw[start:stop].copy_(pattern(offset + start, offset + stop, block, group, raw.device))


def verify_bytes(tensor, offset, block, group):
    raw = tensor.view(torch.uint8).view(-1)
    for start in range(0, raw.numel(), CHUNK):
        stop = min(start + CHUNK, raw.numel())
        assert torch.equal(raw[start:stop], pattern(offset + start, offset + stop,
                                                   block, group, raw.device))


def make_model(mode, rank):
    sizes = ([(torch.bfloat16, 688_364_998), (torch.float32, 21)]
             if mode == 'representative' else
             [(torch.bfloat16, 1003), (torch.float32, 13), (torch.int64, 5), (torch.uint8, 19)])
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList()
    device = 'meta' if mode == 'representative' or rank else 'cpu'
    for index in range(3):
        block = torch.nn.Module()
        for group, (dtype, numel) in enumerate(sorted(sizes, key=lambda v: str(v[0]))):
            value = torch.empty(numel, dtype=dtype, device=device)
            if device == 'cpu':
                fill_bytes(value, 0, index, group)
            if dtype.is_floating_point:
                block.register_parameter(f'w{group}', torch.nn.Parameter(value, requires_grad=False))
            else:
                block.register_buffer(f'w{group}', value)
        block.alias = block.w0
        model.transformer_blocks.append(block)
    if mode == 'toy':
        # Cross-block aliases are root-owned and must stay tied.
        shared = torch.ones(3, device='cpu') if rank == 0 else torch.empty(3, device='meta')
        model.register_buffer('root_shared', shared)
        for block in model.transformer_blocks:
            block.register_buffer('root_alias', shared)
    return model.eval().requires_grad_(False)


def initialize(c, mode):
    if mode == 'toy':
        c.distribute()
        assert all(b.root_alias is c.model.root_shared for b in c.model.transformer_blocks)
        assert torch.equal(c.model.root_shared, torch.ones(3, device=c.device))
        return
    # Representative mode deliberately avoids CPU assembly and startup traffic:
    # same production controller gathers deterministic resident synthetic shards.
    for index, groups in enumerate(c.groups):
        for group_index, group in enumerate(groups):
            group.shard = torch.empty(group.shard_numel, dtype=group.dtype, device=c.device)
            fill_bytes(group.shard, c.rank * group.shard_numel * group.shard.element_size(),
                       index, group_index)
    torch.cuda.synchronize()
    if c.prefetch:
        c.weight_group = dist.new_group(list(range(8)), backend='nccl', timeout=timedelta(seconds=45))
        c.prefetch_stream = torch.cuda.Stream(device=c.device)
    c.initialized = True


def check_weights(c, index):
    for group_index, group in enumerate(c.groups[index]):
        offset = 0
        for entry in group.entries:
            verify_bytes(entry.tensor(), offset, index, group_index)
            offset += entry.numel * entry.tensor().element_size()
    assert c.model.transformer_blocks[index].alias is c.model.transformer_blocks[index].w0
    if c.prefetch:
        for gi, (group, flat) in enumerate(zip(c.groups[index], c._active_gather.buffers)):
            raw = flat.view(torch.uint8)
            logical = group.numel * flat.element_size()
            if hasattr(c.model, 'root_shared'):
                assert torch.count_nonzero(raw[logical:]).item() == 0
            else:
                verify_bytes(raw[logical:], logical, index, gi)


def run_case(mode, enabled, rank, args):
    model = make_model(mode, rank)
    device = torch.device('cuda', rank)
    runtime = SimpleNamespace(device=device, rank=rank, world_size=8, backend='nccl',
                              branch_parallel=False, timeout_seconds=45,
                              profile_start=lambda: None, profile_end=lambda *a: None)
    c = ResidentWeightShards(model, runtime, distribution='broadcast', prefetch=enabled,
                             scatter_chunk_bytes=32 << 20)
    try:
        initialize(c, mode)
        return exercise(c, mode, enabled, rank, args)
    finally:
        if not c.released:
            c.release()


def exercise(c, mode, enabled, rank, args):
    model, device = c.model, c.device
    logical = [sum(g.numel * g.shard.element_size() for g in groups) for groups in c.groups]
    if mode == 'representative':
        assert logical == [1_376_730_080] * 3
    padded = c.summary()['max_gather_bytes']
    assert padded - logical[0] == (16 if mode == 'representative' else 51)
    routes = ['broadcast', 'allgather'] if mode == 'toy' else ['allgather']
    matrix = torch.eye(1024 if mode == 'toy' else 2048, dtype=torch.float32, device=device)
    result = torch.empty_like(matrix)
    consumer = torch.cuda.Stream(device=device)
    consumer.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    outputs, overlap, timings = [], [], []
    sequence = []
    real_gather = dist.all_gather_into_tensor

    def tracked(out, inp, *a, **kw):
        if kw.get('group') is c.weight_group and enabled:
            sequence.append(str(inp.dtype))
        return real_gather(out, inp, *a, **kw)

    dist.all_gather_into_tensor = tracked
    try:
        for route in routes:
            exchange = UlyssesRuntime(rank, 8, rank, device, 'nccl', standard_exchange=route)
            exchange.configure(19, 8)
            if enabled:
                # Uninstrumented transition stress: no host equality/drain between
                # attention, enqueue, consumer clone and next block acquisition.
                observed = []
                with torch.cuda.stream(consumer), c.forward():
                    for index in range(3):
                        with c.block(index):
                            if rank == 7 and index == 1:
                                time.sleep(.15)
                            attention = exchange.heads_to_sequence(
                                torch.ones(19, 1, 1, device=device) * rank)
                            c.prefetch_next()
                            torch.cuda._sleep(2_000_000)
                            observed.append((index, model.transformer_blocks[index].w0[:1003]
                                             .view(torch.uint8).clone(), attention))
                        churn = torch.empty(c.groups[index][0].shard_numel * 8,
                                            dtype=c.groups[index][0].dtype, device=device)
                        churn.zero_()
                        del churn
                for index, value, attention in observed:
                    verify_bytes(value, 0, index, 0)
                    assert torch.equal(attention, torch.arange(8, device=device)
                                       .view(1, 8, 1).expand_as(attention).float())
            for repeat in range(3 if mode == 'toy' else 2):
                retained = []
                with torch.cuda.stream(consumer), c.forward():
                    for index in range(3):
                        with c.block(index):
                            check_weights(c, index)
                            # Real attention-group traffic before the enqueue point.
                            x = torch.arange(19, device=device).view(19, 1, 1).float() + rank * 100
                            attention = exchange.heads_to_sequence(x)
                            expected = torch.arange(exchange.local_start, exchange.local_end,
                                                    device=device).view(-1, 1, 1).float()
                            expected = expected + torch.arange(8, device=device).view(1, 8, 1) * 100
                            assert torch.equal(attention, expected)
                            if rank == 7 and repeat == 1 and index == 0:
                                time.sleep(.15)
                            origin = torch.cuda.Event(enable_timing=True)
                            gather_end = torch.cuda.Event(enable_timing=True)
                            tail_start = torch.cuda.Event(enable_timing=True)
                            tail_end = torch.cuda.Event(enable_timing=True)
                            origin.record(consumer)
                            if enabled:
                                c.prefetch_next()
                                gather_end.record(c.prefetch_stream)
                                if index < 2:
                                    assert c.pending.index == index + 1
                                    assert all(e.tensor().is_meta for e in c.blocks[index+1])
                            tail_start.record(consumer)
                            # Delayed consumer and GEMM tail run AFTER enqueue. These
                            # finite synthetic GEMMs are not model performance evidence.
                            if mode == 'toy':
                                torch.cuda._sleep(2_000_000)
                            for _ in range(8):
                                torch.mm(matrix, matrix, out=result)
                            snapshot = model.transformer_blocks[index].w0[:1003].view(torch.uint8).clone()
                            retained.append((index, snapshot))
                            tail_end.record(consumer)
                        # Consumer completion must be real at scope exit, not just
                        # queued. Drop/reallocate similarly sized buffers aggressively.
                        if not enabled:
                            tail_end.synchronize()
                        assert tail_end.query()
                        if enabled:
                            assert c._active_gather is None
                            if index < 2:
                                timings.append((route, repeat, index, origin, gather_end, tail_start, tail_end))
                        churn = [torch.empty(1 << 20, dtype=torch.uint8, device=device) for _ in range(4)]
                        for t in churn:
                            t.fill_(127)
                        del churn
                    assert torch.equal(result, matrix)
                assert c.pending is None and c.active is None and c._active_gather is None
                for index, value in retained:
                    verify_bytes(value, 0, index, 0)
                    outputs.append(value.cpu().tolist())
        if enabled:
            # Symmetric application exception with pending work, then a clean call.
            try:
                with torch.cuda.stream(consumer), c.forward(), c.block(0):
                    c.prefetch_next()
                    torch.mm(matrix, matrix, out=result)
                    if args.failure == 'rank' and rank == 3:
                        raise RuntimeError('rank-local injected failure')
                    if args.failure == 'rank':
                        # Peer dies without executing the next collective. The PG
                        # timeout and descendant-aware supervisor must clean workers.
                        raise_next = False
                    else:
                        raise_next = True
                    if raise_next:
                        raise ValueError('symmetric injected failure')
            except ValueError:
                pass
            assert c.pending is None and c.active is None
            if args.failure == 'rank':
                dist.barrier()
            with torch.cuda.stream(consumer), c.forward(), c.block(0):
                check_weights(c, 0)
                c.prefetch_next()
            assert c.pending is None
        summary = c.summary()
        for route, repeat, index, origin, gather_end, tail_start, tail_end in timings:
            gather_end.synchronize()
            overlap.append(dict(route=route, repeat=repeat, block=index,
                gather_window_ms=origin.elapsed_time(gather_end),
                tail_start_ms=origin.elapsed_time(tail_start),
                tail_end_ms=origin.elapsed_time(tail_end),
                overlap_ms=max(0., min(origin.elapsed_time(gather_end),
                    origin.elapsed_time(tail_end)) - origin.elapsed_time(tail_start))))
        alloc = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        assert alloc < 32 * 2**30 and reserved < 32 * 2**30
        assert not enabled or c.max_live_gathered == 2
        c.release()
        c.release()
        assert c.weight_group is None and c.prefetch_stream is None
        assert all(e.tensor().is_meta for e in c.entries)
        assert all(g.shard is None for groups in c.groups for g in groups)
        return dict(rank=rank, mode=mode, prefetch=enabled, logical_block_bytes=logical[0],
                    padded_block_bytes=padded, summary=summary, peak_allocated_bytes=alloc,
                    peak_reserved_bytes=reserved, storage_checks='all logical bytes in validation calls; padding in candidate; sampled clones in transition stress',
                    dtype_sequence=sequence, overlap=overlap, outputs=outputs,
                    release_clean=True)
    finally:
        dist.all_gather_into_tensor = real_gather
        if not c.released:
            c.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['toy', 'representative'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--failure', choices=['none', 'rank'], default='none')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 8
    assert os.environ.get('NCCL_SHM_USE_CUDA_MEMCPY') == '0'
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=timedelta(seconds=45))
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        baseline = run_case(args.mode, False, rank, args)
        torch.cuda.empty_cache()
        dist.barrier()
        candidate = run_case(args.mode, True, rank, args)
        assert baseline.pop('outputs') == candidate.pop('outputs')
        record = dict(baseline=baseline, candidate=candidate, exact_output_bytes=True)
        (args.output_dir / f'rank-{rank}.json').write_text(json.dumps(record, indent=2))
        records = [None] * 8
        dist.all_gather_object(records, record)
        assert len({tuple(r['candidate']['dtype_sequence']) for r in records}) == 1
        if rank == 0:
            (args.output_dir / 'summary.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(dict(mode=args.mode, ranks=8, exact_output_bytes=True,
                logical_block_bytes=candidate['logical_block_bytes'],
                padded_block_bytes=candidate['padded_block_bytes'],
                max_allocated=max(r['candidate']['peak_allocated_bytes'] for r in records),
                max_reserved=max(r['candidate']['peak_reserved_bytes'] for r in records),
                max_live_gathered=max(r['candidate']['summary']['max_live_gathered'] for r in records),
                release_clean=True)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
