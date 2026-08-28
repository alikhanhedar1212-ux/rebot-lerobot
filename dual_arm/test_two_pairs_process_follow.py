"""Supervise two tuned arm pairs in isolated child processes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


CHILD = Path(__file__).with_name("test_single_pair_tuned_follow.py")


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)


def main() -> None:
    children: list[subprocess.Popen] = []
    try:
        for pair_number in (1, 2):
            children.append(
                subprocess.Popen(
                    [sys.executable, str(CHILD), "--pair", str(pair_number)],
                    start_new_session=True,
                )
            )
        print("Process-isolated dual-pair follow starting: 1->0 and 3->2", flush=True)
        while True:
            for index, child in enumerate(children, start=1):
                code = child.poll()
                if code is not None:
                    raise RuntimeError(f"pair {index} child exited with status {code}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping process-isolated dual-pair follow...", flush=True)
    finally:
        for child in children:
            stop_child(child)
        deadline = time.monotonic() + 5.0
        for child in children:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        print("All pair processes stopped.", flush=True)


if __name__ == "__main__":
    main()
