"""Bounded Linux subreaper for owned jobs, including detached torchrun workers."""
import argparse
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import time


def supervise(command, seconds, grace=15):
    if not command or not 0 < seconds <= 1800:
        raise ValueError('Command required; deadline must be at most 1800 seconds')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))

    def descendants():
        parents = {}
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                parents[int(proc.name)] = (int(fields[1]), fields[0])
            except (FileNotFoundError, ProcessLookupError):
                pass
        found = {os.getpid()}
        while True:
            more = {pid for pid, (parent, _) in parents.items() if parent in found}
            if more <= found:
                break
            found |= more
        return {pid for pid in found - {os.getpid()} if parents[pid][1] != 'Z'}

    def terminate(sig):
        for pid in descendants():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    child = subprocess.Popen(command, start_new_session=True)
    try:
        deadline = time.monotonic() + seconds
        while child.poll() is None and not interrupted and time.monotonic() < deadline:
            time.sleep(.1)
        if interrupted:
            return 128 + interrupted[0]
        if child.poll() is None:
            return 124
        return child.returncode or (1 if descendants() else 0)
    finally:
        deadline = time.monotonic() + grace
        while descendants() and time.monotonic() < deadline:
            terminate(signal.SIGTERM)
            time.sleep(.1)
        terminate(signal.SIGKILL)
        child.wait(timeout=5)
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    raise SystemExit(supervise(command, args.seconds))
