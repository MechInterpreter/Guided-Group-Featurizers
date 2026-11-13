"""Disk-backed feature arrays with the recorded normalization arithmetic."""

from hashlib import sha256

import numpy as np


def array_sha(array, batch=20):
    """Hash the original shape/dtype/C-order encoding without a full byte copy."""
    digest = sha256(str(array.shape).encode() + str(array.dtype).encode())
    for start in range(0, len(array), batch):
        digest.update(np.ascontiguousarray(array[start : start + batch]).tobytes())
    return digest.hexdigest()


def normalization(train, batch=20):
    """Keep the original reductions and float32 centering, bounding temporaries."""
    mean = np.asarray(train).mean(0, dtype=np.float64).astype(np.float32)
    energies = np.empty(train.shape[:-1], dtype=np.float64)
    for start in range(0, len(train), batch):
        centered = train[start : start + batch].astype(np.float32) - mean
        squared = centered.astype(np.float64)
        np.square(squared, out=squared)
        energies[start : start + batch] = np.sum(squared, axis=-1)
    scale = float(np.sqrt(train.shape[-1] / np.mean(energies)))
    return dict(
        mean=mean,
        scale=scale,
        train_sha=array_sha(train, batch),
        kind="training-only per-position mean and one global RMS scale",
    )


def normalize_to_file(raw, norm, path, batch=20):
    """Write the original float32 normalized values into a NumPy file by shard."""
    result = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=raw.shape
    )
    for start in range(0, len(raw), batch):
        result[start : start + batch] = (
            raw[start : start + batch].astype(np.float32) - norm["mean"]
        ) * norm["scale"]
    result.flush()
    del result
