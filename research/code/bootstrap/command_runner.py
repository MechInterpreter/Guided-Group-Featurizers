"""Stream a notebook command and retain its output in a timestamped log."""

from collections import deque
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys


def run_command(command, log_dir, label="stage"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log_path = log_dir / f"{label}_{stamp}.log"
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    tail = deque(maxlen=60)
    with log_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(
            list(map(str, command)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        ) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    tail.append(line)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    print(f"Log: {log_path}", flush=True)
    if returncode:
        raise RuntimeError(
            f"{label} exited with status {returncode}. Log: {log_path}\n"
            + "".join(tail)
        )
    return subprocess.CompletedProcess(command, returncode)
