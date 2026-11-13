"""Checksummed local/Drive artifacts retaining two committed generations.

Drive FUSE is not transactional: a lost sync can cost a checkpoint. Publishing
metadata last and validating bytes prevents accepting partially synced payloads.
Only load trusted artifacts written by this run; torch payloads are Python data.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import tempfile
import shutil
from pathlib import Path
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactStore:
    def __init__(self, root, config):
        self.identity = digest(config)
        self.root = Path(root) / self.identity[:20]
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        manifest = self.root / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != config:
                raise ValueError("Run manifest mismatch")
        else:
            tmp = manifest.with_suffix(".tmp")
            tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
            os.replace(tmp, manifest)

    def _folder(self, name):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Artifact names must be simple identifiers")
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def load(self, name):
        for meta, info in self._records(name):
            try:
                path = meta.with_suffix(".pt")
                if info["run"] != self.identity or file_digest(path) != info["sha256"]:
                    continue
                return torch.load(path, weights_only=False, map_location="cpu")
            except (OSError, ValueError, KeyError, RuntimeError, EOFError):
                continue
        return None

    @classmethod
    def existing(cls, path):
        """Open an explicitly selected trusted archive without changing its identity."""
        path = Path(path)
        config = json.loads((path / "manifest.json").read_text())
        if path.name != digest(config)[:20]:
            raise ValueError("Archive directory does not match its manifest")
        return cls(path.parent, config)

    def _records(self, name):
        records = []
        for meta in self._folder(name).glob("*.json"):
            try:
                info = json.loads(meta.read_text())
                generation = int(info["generation"]) if "generation" in info else int(meta.stem.split("-")[0])
                records.append((generation, meta, info))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return [(p, v) for _, p, v in sorted(records, reverse=True)]

    def peek(self,name):
        """Lightweight shard summary; full payload hash is checked on consumption.

        Allows resume to skip terabytes of completed extraction. A bad payload
        found by load() invalidates its header, so rerunning harvest repairs it.
        """
        for meta, info in self._records(name):
            try:
                if info["run"]==self.identity and meta.with_suffix(".pt").stat().st_size==info["bytes"]:
                    return info.get("summary")
            except (OSError,ValueError,KeyError):
                continue
        return None

    def save(self, name, payload, summary=None):
        """Reuse two files IN PLACE: never unlink/replace a Drive checkpoint.

        Single writer only. Stage serialization on local temporary storage, then
        overwrite the older slot and publish its checksum last. The newest valid
        slot remains untouched during either write. This avoids Drive Trash churn;
        FUSE/cloud sync is still not transactional. Legacy files are read but never
        deleted automatically. Existing Trash must be reviewed separately.
        """
        folder = self._folder(name)
        protected = None
        for meta, info in self._records(name):
            try:
                if info["run"] == self.identity and file_digest(meta.with_suffix(".pt")) == info["sha256"]:
                    protected = meta.stem
                    break
            except (OSError, KeyError):
                pass
        stem = "00000000000000000001-slot" if protected == "00000000000000000000-slot" else "00000000000000000000-slot"
        path = folder / (stem + ".pt")
        with tempfile.TemporaryFile() as f:
            torch.save(payload, f)
            f.flush()
            f.seek(0)
            expected = hashlib.file_digest(f, "sha256").hexdigest()
            f.seek(0)
            with open(path, "r+b" if path.exists() else "w+b") as out:
                shutil.copyfileobj(f, out, length=8 * 1024**2)
                out.truncate(); out.flush(); os.fsync(out.fileno())
        if file_digest(path) != expected:
            raise OSError("Checkpoint read-back checksum failed; previous slot retained")
        info = {"run": self.identity, "sha256": expected, "generation": time.time_ns(),
                "bytes":path.stat().st_size,"summary":summary}
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as out:
            json.dump(info, out); out.flush(); os.fsync(out.fileno())
        return payload

    def result(self, name, fn):
        cached = self.load(name)
        if cached is not None:
            print(f"{name}: restored")
            return cached
        print(f"{name}: running", flush=True)
        return self.save(name, fn())

    def figure(self, fig, name):
        folder = self.root / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight", dpi=150)
