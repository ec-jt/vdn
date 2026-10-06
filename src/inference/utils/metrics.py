"""Synchronized phase measurements, durably appended before the next phase runs.

Cold-step time includes transfer, kernels, and any compilation/cache work. It is not
a measurement of pure compilation. CPU peak RSS is process lifetime, not phase-local.
"""
import json
import os
import resource
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import torch


class PhaseMetrics:
    def __init__(self, path, device):
        self.path = path
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.denoise_steps = 0
        self.events = []
        self._active = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.run_id = f"{time.time_ns()}-{os.getpid()}"
        self.emit({"event": "run_start", "device": str(self.device),
                   "timing_note": "Synchronized wall time includes transfers and possible compilation; not pure compile time.",
                   "rss_note": "cpu_peak_rss_bytes is the process-lifetime high-water mark."})

    def emit(self, event):
        event = dict(event, run_id=self.run_id,
                     timestamp=datetime.now(timezone.utc).isoformat())
        self.events.append(event)
        with open(self.path, "a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _snapshot(self):
        with open("/proc/self/statm") as stream:
            rss = int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        result = {"cpu_rss_bytes": rss,
                  "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        if self.cuda:
            torch.cuda.synchronize(self.device)
            result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                          cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        return result

    def _accumulate(self, snapshot):
        # Child phases reset CUDA counters. Preserve their maxima for enclosing phases.
        for active in self._active:
            for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                if key in snapshot:
                    active[key] = max(active.get(key, 0), snapshot[key])

    @contextmanager
    def phase(self, name, **metadata):
        before = self._snapshot()
        self._accumulate(before)
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        peaks = {}
        self._active.append(peaks)
        self.emit({"event": "phase_start", "phase": name, **metadata, **before})
        started = time.perf_counter()
        error = None
        try:
            yield
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            after = self._snapshot()
            elapsed = time.perf_counter() - started
            self._accumulate(after)
            self._active.pop()
            self.emit({"event": "phase_end", "phase": name, **metadata, **after, **peaks,
                       "seconds": elapsed, "status": "error" if error else "ok", "error": error})

    @contextmanager
    def denoise_step(self, step_index):
        call_index = self.denoise_steps
        self.denoise_steps += 1
        label = ("first_cold_call_may_include_compilation" if call_index == 0
                 else "later_call_may_recompile")
        with self.phase("denoise_step", step_index=step_index, call_index=call_index,
                        execution_label=label):
            yield
