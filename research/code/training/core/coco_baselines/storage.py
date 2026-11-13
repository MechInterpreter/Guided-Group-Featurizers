"""Local-first, checksum-verified two-slot checkpoints; one writer per run."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def serial(v):
    if isinstance(v, dict):
        return {str(k): serial(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [serial(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    return v


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def identity(v):
    return hashlib.sha256(
        json.dumps(serial(v), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(
        str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()
    ).hexdigest()


class Store:
    def __init__(self, output, config, local, require_drive=True):
        self.output, self.local = Path(output), Path(local)
        self.require_drive = require_drive
        self.config, self.run_id = serial(config), identity(config)
        self.root = self.output / "checkpoints" / self.run_id[:20]
        self.rescue = self.local / self.run_id[:20]
        self.check_mount()
        self.root.mkdir(parents=True, exist_ok=True)
        self.rescue.mkdir(parents=True, exist_ok=True)
        self.results = self.output / "results" / self.run_id[:20]
        self.write("manifest.json", dict(run_id=self.run_id, config=self.config))

    def check_mount(self):
        if self.require_drive and (
            not os.path.ismount("/content/drive")
            or not Path("/content/drive/MyDrive").is_dir()
        ):
            raise OSError(
                "Drive is disconnected. Remount Drive and rerun unchanged; local rescue checkpoints remain in /content. No replacement Drive directory was created."
            )

    def _key(self, key):
        if not key or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in key
        ):
            raise ValueError("Use a simple checkpoint identifier")

    def _records(self, base, key):
        out = []
        for meta in (base / key).glob("slot*.json"):
            try:
                r = json.loads(meta.read_text())
                p = meta.with_suffix(".pt")
                if (
                    r["run_id"] == self.run_id
                    and p.exists()
                    and sha_file(p) == r["sha256"]
                ):
                    out.append((r["generation"], meta, p))
            except (OSError, ValueError, KeyError):
                pass
        return sorted(out, reverse=True)

    def load(self, key):
        self._key(key)
        found = self._records(self.rescue, key) + self._records(self.root, key)
        if found:
            for _, _, p in sorted(found, reverse=True):
                try:
                    return torch.load(p, map_location="cpu", weights_only=False)
                except (RuntimeError, EOFError, OSError):
                    pass
        if any(list((b / key).glob("slot*.json")) for b in (self.rescue, self.root)):
            raise OSError(
                f"No valid generation of {key} remains. Restore the archive rather than silently retraining."
            )
        return None

    def _save_slot(self, base, key, staging):
        recs = self._records(base, key)
        slot = "slot1" if recs and recs[0][1].stem == "slot0" else "slot0"
        folder = base / key
        folder.mkdir(parents=True, exist_ok=True)
        payload = folder / (slot + ".pt")
        with (
            open(staging, "rb") as src,
            open(payload, "r+b" if payload.exists() else "w+b") as dst,
        ):
            shutil.copyfileobj(src, dst, 4 * 1024**2)
            dst.truncate()
            dst.flush()
            os.fsync(dst.fileno())
        digest = sha_file(staging)
        if sha_file(payload) != digest:
            raise OSError("Checkpoint read-back failed; previous generation retained")
        meta = dict(run_id=self.run_id, sha256=digest, generation=time.time_ns())
        with open(folder / (slot + ".json"), "w") as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())

    def save(self, key, value, drive=True):
        self._key(key)
        staging = self.rescue / "_pending.pt"
        torch.save(value, staging)
        self._save_slot(self.rescue, key, staging)
        if drive:
            self.check_mount()
            self._save_slot(self.root, key, staging)
        staging.unlink(missing_ok=True)
        return value

    def write(self, name, value):
        raw = json.dumps(serial(value), indent=2, allow_nan=False)
        local = self.rescue / "results" / name
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(raw, encoding="utf-8")
        self.check_mount()
        p = self.results / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(raw, encoding="utf-8")
        return p

    def event(self, stage, **kw):
        row = dict(
            time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            stage=stage,
            **serial(kw),
        )
        p = self.rescue / "events.jsonl"
        with p.open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        self.write("progress.json", row)
        self.check_mount()
        self.results.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, self.results / "events.jsonl")
        print(stage, kw, flush=True)
