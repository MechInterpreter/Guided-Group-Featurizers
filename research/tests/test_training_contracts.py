"""Actual CPU training/resume and solver contracts; no downloads or API calls.

Install the training dependencies to run these optional numerical checks.
Standard-library bundle audits remain available without them.
"""

import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

try:
    import numpy as np
    import torch
    import scipy
    import einops
except ImportError as error:
    raise unittest.SkipTest(
        "Optional CPU training contracts need torch, numpy, scipy and einops: "
        + str(error)
    ) from error

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE / "bootstrap"))
import fresh_matched75 as fresh

fresh.imports(CODE)

import all75_run
import cd_common
from coco_baselines import training
from coco_ggf.width_balance import settings
from eval_common import endpoint_state
from scale_models import model
from scale_plan import settings as rules
import scale_fit
import supervised_fit


def temporary():
    return tempfile.TemporaryDirectory(prefix="ggf-training-contract-")


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        original_spec = copy.deepcopy(cd_common.SPEC)
        original_rng = torch.random.get_rng_state()
        original_threads = torch.get_num_threads()

        def restore():
            cd_common.SPEC.clear()
            cd_common.SPEC.update(original_spec)
            torch.random.set_rng_state(original_rng)
            torch.set_num_threads(original_threads)

        self.addCleanup(restore)

    def test_exact_allocation_and_capacity(self):
        spec = fresh.specification()
        self.assertEqual(
            [len(spec["splits"][s]) for s in ["train", "select", "confirm", "audit"]],
            [2000, 400, 400, 1500],
        )
        self.assertEqual([len(f["guide"]) for f in spec["folds"]], [37, 38])
        self.assertEqual(len(fresh.required_banks("full_category")), 28)
        self.assertEqual(len(fresh.required_banks("category_split")), 35)
        self.assertEqual(
            spec["backbone"]["revision"], "5931719e67bbdb9737e363e781fb0c67687896bc"
        )

    def test_resume_manifest_and_path_guards(self):
        with temporary() as folder:
            root = Path(folder) / "new"
            first = fresh.prepare(root, CODE)
            self.assertEqual(first, fresh.prepare(root, CODE))
            manifest = fresh.read(root / "FRESH_RUN.json")
            manifest["seeds"] = [21]
            (root / "FRESH_RUN.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                fresh.prepare(root, CODE)
            with self.assertRaises(ValueError):
                fresh.validate_root(CODE / "new_run", CODE)
            old = Path(folder) / "old"
            old.mkdir()
            (old / "data").write_text("keep")
            with self.assertRaises(ValueError):
                fresh.validate_root(old, CODE)

    def test_audit_is_closed_and_rejects_partial_receipts(self):
        with temporary() as folder:
            root = Path(folder)
            with self.assertRaises(RuntimeError):
                fresh.require_frozen(root)
            for regime in fresh.REGIMES:
                path = root / "training" / regime / "BANKS.json"
                path.parent.mkdir(parents=True)
                rows = [
                    dict(
                        seed=s,
                        fold=f,
                        kind=k,
                        regime=regime,
                        groups=75,
                        width=12,
                        top_k=None if k == "slics" else 32,
                    )
                    for s, f, k in sorted(fresh.required_banks(regime))
                ]
                path.write_text(json.dumps(dict(complete=True, banks=rows)))
            self.assertEqual(
                set(fresh.require_frozen(root, verify_weights=False)),
                set(fresh.REGIMES),
            )
            path = root / "training/category_split/BANKS.json"
            data = fresh.read(path)
            data["banks"].pop()
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                fresh.require_frozen(root, verify_weights=False)

    def test_fresh_stores_load_with_original_endpoint_reader(self):
        with temporary() as folder:
            root = Path(folder)
            for kind in ["ggf", "slics"]:
                store = cd_common.make_store(
                    root / kind,
                    root / "local" / kind,
                    dict(guide=list(range(75)), excluded=[]),
                    drive=False,
                )
                if kind == "ggf":
                    weights = dict(
                        W_dec=torch.zeros(900, 768),
                        W_enc=torch.zeros(768, 900),
                        b_enc=torch.zeros(900),
                    )
                    state = dict(weights=weights, updates=3000, groups=75, active=32)
                    store.save("continuation", state)
                else:
                    state = dict(D=torch.zeros(900, 768), updates=10640)
                row = cd_common.finish_model(store, state, kind, 0, 21)
                row.update(groups=75, width=12, top_k=None if kind == "slics" else 32)
                loaded, _ = endpoint_state(row, root / "reader")
                self.assertTrue(loaded["training_complete"])

    def test_label_free_context_drops_masks(self):
        all75_run.configure("full_category")
        pack = dict(
            x=np.zeros((8, 2, 4), np.float32),
            ids=list(range(8)),
            masks=["sentinel"] * 8,
            identity="test",
        )
        ctx = dict(
            packs=dict(train=pack, select=pack),
            cfg={},
            norm={},
            provenance=dict(categories=[]),
            device="cpu",
            options=dict(local="scratch", regime="full_category"),
        )
        sub = all75_run.training_context(ctx, 0, 21, "vanilla", object())
        self.assertTrue(all("masks" not in p for p in sub["packs"].values()))

    def test_actual_fit_resume_and_implementation_parity(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(1)
        values = rng.normal(size=(8, 4, 6)).astype(np.float32)
        packs = {
            split: dict(x=values, ids=list(range(8)), identity="synthetic-fixture")
            for split in ["train", "select"]
        }
        rule = dict(
            rules(),
            width=2,
            images_per_update=8,
            evaluate_every=2,
            local_save_every=1,
            snr=0.0,
            sampling="epochs",
        )
        torch.manual_seed(21)
        initial = training.weights(model("released_vanilla", 3, 2, 2, 6))
        with temporary() as folder:
            outputs = []
            for name, module, staged in [
                ("scale", scale_fit, False),
                ("supervised", supervised_fit, False),
                ("resume", supervised_fit, True),
            ]:
                root = Path(folder) / name
                store = cd_common.make_store(
                    root / "out", root / "local", dict(test=name), drive=False
                )
                ctx = dict(
                    store=store,
                    packs=packs,
                    cfg=dict(batch=512),
                    rule=rule,
                    device="cpu",
                    balance_recipe=copy.deepcopy(settings()["balance"]),
                )
                if staged:
                    module.fit(ctx, "fit", "released_vanilla", 3, 2, 21, initial, 2, 1)
                result = module.fit(
                    ctx, "fit", "released_vanilla", 3, 2, 21, initial, 2, 2
                )
                outputs.append(result)
            self.assertEqual(
                len({training.weight_sha(s["weights"]) for s in outputs}), 1
            )
            self.assertEqual(len({s["sampled_batches_sha"] for s in outputs}), 1)
            self.assertEqual(len({s["sampled_noise_sha"] for s in outputs}), 1)

    def test_z_precision_overlay_installs_and_certifies_actual_solver(self):
        import seeds24_27_run
        import supervised_study
        import object_schedule
        from coco_ggf.slics_repair import Solver

        with ExitStack() as restore, temporary() as folder:
            for obj, name in [
                (Solver, "solve"),
                (supervised_study, "partial_nnls"),
                (supervised_study, "native_fit"),
                (supervised_study.NativeSLiCS, "forward"),
                (supervised_study, "evaluate_model"),
                (object_schedule, "final_reconstruction"),
                (supervised_study, "final_reconstruction"),
            ]:
                restore.enter_context(patch.object(obj, name, getattr(obj, name)))
            restore.enter_context(
                patch.object(
                    supervised_study, "_ggf_precision_atom_sweep", None, create=True
                )
            )
            root = Path(folder)
            store = cd_common.make_store(
                root / "output", root / "local", dict(test="precision"), drive=False
            )
            seeds24_27_run.install_legacy_precision(store)
            dictionary = torch.eye(2)
            inputs = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
            cfg = supervised_study.native_solver_config()
            values, info = Solver(dictionary, cfg).solve(inputs)
            self.assertLessEqual(info["max_relative_kkt"], 1e-5)
            self.assertTrue(torch.allclose(values, inputs.clamp_min(0)))
            partial, info = supervised_study.partial_nnls(inputs, dictionary, cfg, 2)
            self.assertLessEqual(info["max_relative_kkt"], 1e-5)
            self.assertTrue(torch.allclose(partial[:, 0], values))


if __name__ == "__main__":
    unittest.main()
