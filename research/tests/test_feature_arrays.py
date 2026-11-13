"""Verify bounded-memory arithmetic against the frozen scientific functions."""

from pathlib import Path
import sys
import tempfile
import unittest
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code/bootstrap"), str(ROOT / "code/training/core")]
import feature_arrays as bounded
import fresh_matched75 as fresh
from coco_baselines import data, storage


class FeatureArrayTests(unittest.TestCase):
    def test_verified_old_shards_open_read_only_with_new_source_identity(self):
        from hashlib import sha256

        code = fresh.imports(ROOT / "code")
        from cd_common import make_store

        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "old", Path(tmp) / "new"
            _, manifest = fresh.prepare(target, code)
            previous = dict(
                manifest,
                implementation_sha256=next(
                    iter(fresh.EXTRACTION_CACHE_IMPLEMENTATIONS)
                ),
            )
            source.mkdir()
            (source / "FRESH_RUN.json").write_text(json.dumps(previous))
            journal = make_store(
                source / "features/journal",
                source / "scratch/features/journal",
                previous,
                drive=False,
            )
            shard = dict(ids=[123], x=np.zeros((1, 196, 768), dtype="float16"))
            journal.save("features_train_000000", shard)
            before = {
                p.relative_to(source): sha256(p.read_bytes()).hexdigest()
                for p in source.rglob("*")
                if p.is_file()
            }
            reader, provenance = fresh.previous_feature_store(source, target, manifest)
            loaded = reader.load("features_train_000000")
            self.assertEqual(loaded["ids"], [123])
            self.assertEqual(loaded["x"].tobytes(), shard["x"].tobytes())
            self.assertEqual(
                provenance["source_implementation_sha256"],
                previous["implementation_sha256"],
            )
            after = {
                p.relative_to(source): sha256(p.read_bytes()).hexdigest()
                for p in source.rglob("*")
                if p.is_file()
            }
            self.assertEqual(before, after)
            changed = dict(manifest, specification_sha256="changed")
            with self.assertRaisesRegex(
                ValueError, "specification or scientific source differs"
            ):
                fresh.previous_feature_store(source, target, changed)
            with self.assertRaisesRegex(ValueError, "separate directories"):
                fresh.previous_feature_store(source, source, manifest)

    def test_normalization_and_file_bytes_match_frozen_reference(self):
        rng = np.random.default_rng(74)
        for n in (2, 19, 20, 21, 61):
            with self.subTest(images=n), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                original = (
                    rng.normal(size=(n, 196, 23)).astype("float16").astype("float32")
                )
                np.save(folder / "raw.npy", original)
                raw = np.load(folder / "raw.npy", mmap_mode="r")
                expected = data.normalization(original)
                actual = bounded.normalization(raw)
                self.assertEqual(actual["mean"].tobytes(), expected["mean"].tobytes())
                self.assertEqual(actual["scale"].hex(), expected["scale"].hex())
                self.assertEqual(storage.identity(actual), storage.identity(expected))
                bounded.normalize_to_file(raw, actual, folder / "actual.npy")
                np.save(folder / "expected.npy", data.normalize(original, expected))
                self.assertEqual(
                    (folder / "actual.npy").read_bytes(),
                    (folder / "expected.npy").read_bytes(),
                )
                raw._mmap.close()

    def test_hash_matches_reference_for_strided_and_empty_arrays(self):
        base = np.arange(63 * 7 * 12, dtype="float32").reshape(63, 7, 12)
        for array in (base, base[::2, :, ::2], base[:0], base.astype("float16")):
            with self.subTest(shape=array.shape, dtype=str(array.dtype)):
                self.assertEqual(bounded.array_sha(array), storage.array_sha(array))

    def test_saved_normalizer_applies_identically_to_a_new_split(self):
        rng = np.random.default_rng(75)
        train = rng.normal(size=(41, 196, 32)).astype("float32")
        test = rng.normal(size=(27, 196, 32)).astype("float32")
        original = data.normalization(train)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "confirm.npy"
            bounded.normalize_to_file(test, original, path)
            actual = np.load(path, mmap_mode="r")
            expected = data.normalize(test, original)
            self.assertEqual(bounded.array_sha(actual), storage.array_sha(expected))
            self.assertEqual(actual.tobytes(), expected.tobytes())
            actual._mmap.close()


if __name__ == "__main__":
    unittest.main()
