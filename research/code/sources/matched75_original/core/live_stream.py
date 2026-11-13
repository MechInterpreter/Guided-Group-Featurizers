"""Parent-side cell heartbeat, independent of the numerical worker's Python GIL."""
import json
import os
from pathlib import Path
import queue
import threading
import time


def cpu_seconds(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def heartbeat(pid, status_path, previous_cpu=None):
    now = time.monotonic()
    data = dict(kind='worker_liveness_not_proof_of_model_progress', worker_pid=pid,
                time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        status = json.loads(Path(status_path).read_text())
        if status.get('worker_pid') == pid:
            data['worker_status'] = status
            data['operation_elapsed_seconds'] = max(0., now - status['operation_started_monotonic'])
    except (OSError, ValueError, KeyError):
        pass
    cpu = cpu_seconds(pid)
    data.update(cpu_seconds=cpu, cpu_seconds_since_last_heartbeat=
                None if cpu is None or previous_cpu is None else max(0., cpu - previous_cpu))
    status = data.get('worker_status', {})
    message = '[heartbeat] Worker alive, PID {} | CPU total {}s, +{}s | {} | operation elapsed {:.0f}s'.format(
        pid, '?' if cpu is None else round(cpu, 1),
        '?' if data['cpu_seconds_since_last_heartbeat'] is None else round(data['cpu_seconds_since_last_heartbeat'], 1),
        status.get('operation', 'waiting for first worker status'), data.get('operation_elapsed_seconds', 0.))
    if 'images_completed' in status:
        message += ' | images {}/{}'.format(status['images_completed'], status['images_total'])
    if 'fallback_calls_completed' in status:
        message += ' | CPU fallbacks completed {}'.format(status['fallback_calls_completed'])
    return message, data, cpu


def stream_worker(process, status_path, receipt_path=None, interval=30., emit=None):
    """Forward stdout, add liveness on schedule, never interrupt or change a fit."""
    emit = emit or (lambda value: print(value, flush=True))
    # Re-enter after handled SIGINT using the SAME reader/queue, so the save
    # handler's output cannot be stolen by two simultaneous stdout readers.
    items = getattr(process, '_ggf_output_queue', None)
    if items is None:
        items = queue.Queue()
        process._ggf_output_queue = items
        def reader():
            try:
                for line in process.stdout:
                    items.put(line)
            finally:
                items.put(None)
        threading.Thread(target=reader, daemon=True).start()
    last = time.monotonic()
    previous_cpu = cpu_seconds(process.pid)
    writer = None
    def save_drive(data):
        # Separate thread: a slow Drive write cannot silence the cell heartbeat.
        if receipt_path is None:
            return
        try:
            receipt = json.loads(Path(receipt_path).read_text())
            if receipt.get('run_id') != 'd8a0a7c7bd2e6ffb880cee729c8fdfba2adea8ca3709f887e5c58e5ae77fb0fd':
                return
            folder = Path(receipt['results'])
            if folder.is_dir():
                (folder / 'logging_heartbeat.json').write_text(json.dumps(data, indent=2, allow_nan=False))
        except (OSError, ValueError, KeyError):
            pass
    while True:
        remaining = max(.001, interval - (time.monotonic() - last))
        try:
            line = items.get(timeout=remaining)
            if line is None:
                break
            emit(line.rstrip('\n'))
        except queue.Empty:
            pass
        if time.monotonic() - last >= interval and process.poll() is None:
            message, data, previous_cpu = heartbeat(process.pid, status_path, previous_cpu)
            emit(message)
            last = time.monotonic()
            if writer is None or not writer.is_alive():
                writer = threading.Thread(target=save_drive, args=(data,), daemon=True)
                writer.start()
    return process.wait()
