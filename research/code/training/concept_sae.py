"""Pinned released Concept-SAE architecture/objectives on shared DINO/mask data.

This is a documented task adaptation, not a CLIP/ImageNet replication or a
capacity/Top-k-matched BSF. Released defaults: 10 epochs per stage, effective
batch64; Adam LR1e-4/1e-3/1e-3; 36 free tokens. No paper-only sigmoid is added.
"""

import copy, gc, hashlib, importlib.util, json, os, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from coco_baselines import metrics, training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from analysis import objects, curve

COMMIT = "956d19b32baccd9c355b5ada9b281898ca998a16"
SHA = "8323fe5aebc108d3c5929f106446998d14a7b0e0129aa51f00d8d61b24ad6593"
URL = f"https://raw.githubusercontent.com/RafaDD/Concept-SAE/{COMMIT}/step1-3/replace_model.py"


def source(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=90) as r:
            raw = r.read()
        if hashlib.sha256(raw).hexdigest() != SHA:
            raise ValueError("Concept-SAE upstream hash mismatch")
        path.write_bytes(raw)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA:
        raise ValueError("Cached Concept-SAE source changed")
    spec = importlib.util.spec_from_file_location(
        "pinned_concept_sae_" + SHA[:12], path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(mod, C, p=196, d=768, seg=14, free=36):
    return mod.FunctionalConvBlock_with_pretrained_clip(
        in_channels=p,
        out_channels=p,
        input_size=d,
        output_size=d,
        seg_size=seg,
        num_tokens=C + free,
        concept_tokens=C,
    )


def configure(m, stage):
    for p in m.parameters():
        p.requires_grad_(False)
        p.grad = None
    part = (
        m.concept_model.tokenizer
        if stage == 1
        else m.concept_model
        if stage == 2
        else m.free_model
    )
    for p in part.parameters():
        p.requires_grad_(True)
    if stage == 2:
        for p in m.concept_model.tokenizer.parameters():
            p.requires_grad_(False)
    return part


def loss(m, x, masks, exists, known, stage, regularize=True):
    if stage == 1:
        raw, score = m.concept_model.tokenizer(x)
        weight = known[:, None, :]
        # Unknown pixels omitted, with a common per-batch denominator. With all
        # pixels known this is exactly the released mean squared mask loss.
        seg = ((raw - masks).square() * weight).sum() / (
            weight.sum() * raw.shape[1]
        ).clamp_min(1)
        cls = F.mse_loss(score, exists)
        sparse = (
            0.01
            * sum(
                layer[0].weight.abs().sum()
                for layer in m.concept_model.tokenizer.token_merge
            )
            / raw.shape[1]
        )
        return seg + cls + (sparse if regularize else 0)
    if stage == 2:
        pred, _, A = m.concept_model(x)
        rec = F.mse_loss(pred, x)
        merge = torch.cat(
            [
                layer[0].weight.detach()
                for layer in m.concept_model.tokenizer.token_merge
            ],
            0,
        )
        logp = F.log_softmax(merge.abs() / 0.02, dim=1)
        logq = F.log_softmax(A.T.abs() / 0.02, dim=1)
        divergence = (logp.exp() * (logp - logq)).sum(1).mean()
        return rec + (0.01 * divergence + A.abs().mean() if regularize else 0)
    pred, _, _ = m(x)
    return F.mse_loss(pred, x) + (
        m.free_model.token_aggregator.weight.abs().mean() if regularize else 0
    )


def batch(pack, idx, cats, device):
    x = torch.as_tensor(np.array(pack["x"][idx], copy=True), device=device)
    y = torch.as_tensor(
        np.stack([pack["masks"][i]["fraction"][:, cats].T for i in idx]) / 255,
        device=device,
        dtype=torch.float32,
    )
    known = torch.as_tensor(
        np.stack([pack["masks"][i]["crowd"] <= 0.1 for i in idx]),
        device=device,
        dtype=torch.float32,
    )
    # Presence concerns the aligned observed crop; ignore void in presence evidence.
    exists = ((y * known[:, None, :]).amax(-1, keepdim=True) > 0).float()
    return x, y, exists, known


def optimizer_step(m, part, opt, pack, idx, cats, device, stage, microbatch=8):
    opt.zero_grad(set_to_none=True)
    total = 0.0
    # Correct accumulation for mask losses with varying unknown-pixel counts.
    # A single full effective batch is evaluated in microbatches; regularizer
    # applied once. Stage1 mask and presence have different normalizers.
    if stage == 1:
        known_total = sum(
            float((np.asarray(pack["masks"][i]["crowd"]) <= 0.1).sum()) for i in idx
        )
    for lo in range(0, len(idx), microbatch):
        chosen = idx[lo : lo + microbatch]
        x, y, e, k = batch(pack, chosen, cats, device)
        ratio = len(chosen) / len(idx)
        if stage == 1:
            raw, score = m.concept_model.tokenizer(x)
            value = ((raw - y).square() * k[:, None, :]).sum() / max(
                known_total * len(cats), 1
            ) + F.mse_loss(score, e) * ratio
        else:
            value = loss(m, x, y, e, k, stage, regularize=False) * ratio
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite Concept-SAE loss")
        value.backward()
        total += float(value.detach())
    # Subtracting the no-regularization objective evaluates identical formulas,
    # but would unnecessarily keep a second graph. Compute the simple penalty.
    if stage == 1:
        reg = (
            0.01
            * sum(
                layer[0].weight.abs().sum()
                for layer in m.concept_model.tokenizer.token_merge
            )
            / len(cats)
        )
    elif stage == 2:
        A = m.concept_model.token_aggregator.weight
        merge = torch.cat(
            [q[0].weight.detach() for q in m.concept_model.tokenizer.token_merge], 0
        )
        logp = F.log_softmax(merge.abs() / 0.02, 1)
        logq = F.log_softmax(A.T.abs() / 0.02, 1)
        reg = 0.01 * (logp.exp() * (logp - logq)).sum(1).mean() + A.abs().mean()
    else:
        reg = m.free_model.token_aggregator.weight.abs().mean()
    if not torch.isfinite(reg):
        raise FloatingPointError("Nonfinite Concept-SAE regularizer")
    reg.backward()
    opt.step()
    return total + float(reg.detach())


@torch.no_grad()
def validation(m, pack, cats, device, stage):
    total = 0.0
    n = 0
    m.eval()
    for lo in range(0, len(pack["ids"]), 8):
        ids = list(range(lo, min(lo + 8, len(pack["ids"]))))
        x, y, e, k = batch(pack, ids, cats, device)
        total += float(loss(m, x, y, e, k, stage, regularize=False)) * len(ids)
        n += len(ids)
    return total / n


def fit(ctx, seed):
    s = ctx["store"]
    key = f"concept_sae_final_s{seed}"
    old = s.load(key)
    mod = source(Path(ctx["options"]["concept_source"]))
    cats = ctx["cfg"]["category_indices"]
    torch.manual_seed(seed)
    m = build(mod, len(cats)).to(ctx["device"])
    protocol = dict(
        commit=COMMIT,
        source_sha=SHA,
        train=ctx["packs"]["train"]["identity"],
        development=ctx["packs"]["select"]["identity"],
        seeds=[21, 22, 23],
        epochs=[10, 10, 10],
        learning_rates=[1e-4, 1e-3, 1e-3],
        effective_batch=64,
        microbatch=8,
        free_tokens=36,
        concept_tokens=len(cats),
        dimension=768,
        spatial_patches=196,
        mask_grid=14,
        released_vs_paper="Released mask/score outputs are linear (no added sigmoid); stage1 L1 patch merge, stage2 KL + L1 aggregation, stage3 L1 free aggregation weights. Use released 10-epoch defaults, not longer paper schedule.",
        task_adaptation="Frozen normalized DINO final patch sequence replaces CLIP hook activations; human object masks/presence replace source proxy labels; mask grid14 not20; void ignored. Native architecture remains image-sequence based.",
        comparison="Native stages/optimizer and development-best selection, not matched BSF group width, sparsity, update budget or parameter count; all actual costs reported. Not a native CLIP benchmark replication or tuned SOTA claim.",
    )
    sig = identity(protocol)
    s.write("concept_sae/protocol.json", protocol)
    if old is not None:
        if old["signature"] != sig:
            raise ValueError("Concept-SAE final protocol changed")
        m.load_state_dict(old["weights"])
        return m, old
    for stage in [1, 2, 3]:
        part = configure(m, stage)
        lr = [1e-4, 1e-3, 1e-3][stage - 1]
        opt = torch.optim.Adam([p for p in part.parameters() if p.requires_grad], lr=lr)
        schedule = torch.optim.lr_scheduler.StepLR(
            opt, step_size=10 if stage == 1 else 80, gamma=0.5 if stage == 1 else 0.1
        )
        rng = np.random.default_rng(730000 + seed + stage * 10000)
        nb = (len(ctx["packs"]["train"]["ids"]) + 63) // 64
        target = nb * 10
        stagekey = f"concept_sae_s{seed}_stage{stage}"
        st = s.load(stagekey)
        if st is not None:
            if st["signature"] != sig:
                raise ValueError("Concept-SAE resume protocol changed")
            part.load_state_dict(st["weights"])
            opt.load_state_dict(st["optimizer"])
            schedule.load_state_dict(st["scheduler"])
            rng.bit_generator.state = st["sample_rng"]
        else:
            st = dict(
                signature=sig,
                updates=0,
                target=target,
                epoch=0,
                permutation=None,
                best=None,
                best_loss=None,
                history=[],
                seconds=0.0,
            )
        ctx["active_key"] = stagekey

        def save():
            st.update(
                weights=training.weights(part),
                optimizer=copy.deepcopy(opt.state_dict()),
                scheduler=copy.deepcopy(schedule.state_dict()),
                sample_rng=copy.deepcopy(rng.bit_generator.state),
            )
            s._width_drive_recovery_original_save(stagekey, st, drive=True)

        # Save/restore only fully completed effective-batch Adam updates.
        committed = True
        try:
            while st["updates"] < target:
                committed = False
                u = st["updates"]
                epoch, bi = divmod(u, nb)
                if bi == 0:
                    st["permutation"] = rng.permutation(
                        len(ctx["packs"]["train"]["ids"])
                    )
                ids = st["permutation"][bi * 64 : (bi + 1) * 64].tolist()
                started = time.monotonic()
                m.train()
                value = optimizer_step(
                    m, part, opt, ctx["packs"]["train"], ids, cats, ctx["device"], stage
                )
                if str(ctx["device"]).startswith("cuda"):
                    torch.cuda.synchronize()
                st.update(
                    updates=u + 1, seconds=st["seconds"] + time.monotonic() - started
                )
                if (u + 1) % nb == 0:
                    schedule.step()
                    v = validation(
                        m, ctx["packs"]["select"], cats, ctx["device"], stage
                    )
                    if st["best_loss"] is None or v < st["best_loss"]:
                        st.update(
                            best=training.weights(part),
                            best_loss=v,
                            best_epoch=epoch + 1,
                        )
                    st["history"].append(
                        dict(
                            epoch=epoch + 1,
                            updates=u + 1,
                            development_objective=v,
                            training_loss=value,
                        )
                    )
                    save()
                    s.write(
                        "concept_sae/" + stagekey + ".json",
                        {
                            k: v
                            for k, v in st.items()
                            if k
                            in [
                                "history",
                                "updates",
                                "target",
                                "best_epoch",
                                "best_loss",
                                "seconds",
                            ]
                        },
                    )
                    committed = True
                    s.event(
                        "Concept_SAE_epoch",
                        seed=seed,
                        native_stage=stage,
                        epoch=epoch + 1,
                        total_epochs=10,
                        updates=u + 1,
                        total=target,
                        development=v,
                    )
                else:
                    committed = True
                    if (u + 1) % 10 == 0:
                        s.event(
                            "Concept_SAE_progress",
                            seed=seed,
                            native_stage=stage,
                            update=u + 1,
                            total=target,
                            seconds_per_update=st["seconds"] / (u + 1),
                        )
            if st["best"] is None:
                raise ValueError("Missing Concept-SAE development-best state")
            part.load_state_dict(st["best"])
        except BaseException:
            if committed:
                save()
            raise
        finally:
            del opt, schedule
            gc.collect()
            torch.cuda.empty_cache()
    out = dict(
        signature=sig,
        weights=training.weights(m),
        weights_sha=training.weight_sha(training.weights(m)),
        seed=seed,
        parameters=sum(p.numel() for p in m.parameters()),
        protocol=protocol,
    )
    s.save(key, out)
    ctx["active_key"] = None
    return m, out


@torch.no_grad()
def parts(m, x):
    tokens, score = m.concept_model.tokenizer(x)
    gated = tokens * score
    cp = torch.stack(
        [
            project(gated[:, j])
            for j, project in enumerate(m.concept_model.output_projectors)
        ],
        1,
    )
    free = m.free_model.tokenizer(x)
    fp = torch.stack(
        [
            project(free[:, j])
            for j, project in enumerate(m.free_model.output_projectors)
        ],
        1,
    )
    ca = m.concept_model.token_aggregator.weight
    fa = m.free_model.token_aggregator.weight
    prediction = torch.einsum("bgd,pg->bpd", cp, ca) + torch.einsum(
        "bgd,pg->bpd", fp, fa
    )
    return prediction, cp, ca, tokens * score


@torch.no_grad()
def evaluate(ctx, seed, m, meta, pack, cohort):
    s = ctx["store"]
    key = f"concept_sae_{cohort}_s{seed}"
    sig = identity(
        dict(
            weights=meta["weights_sha"],
            pack=pack["identity"],
            protocol=meta["signature"],
        )
    )
    old = s.load(key)
    if old is not None:
        if old["signature"] != sig:
            raise ValueError("Concept-SAE evaluation changed")
        return old
    cats = ctx["cfg"]["category_indices"]
    index = {c: j for j, c in enumerate(cats)}
    obs = objects(pack, ctx["cfg"])
    by = {}
    for o in obs:
        by.setdefault(o["image_index"], []).append(o)
    den = np.zeros((len(cats), len(pack["ids"])))
    err = np.zeros_like(den)
    native_scores = []
    contribution_scores = []
    frac = []
    crowd = []
    sse = 0.0
    summ = np.zeros(768)
    sq = 0.0
    patches = 0
    m.eval()
    for i in range(len(pack["ids"])):
        x = torch.as_tensor(
            np.array(pack["x"][i : i + 1], copy=True), device=ctx["device"]
        )
        pred, cp, A, mask = parts(m, x)
        for o in by.get(i, []):
            j = index[o["category"]]
            w = torch.as_tensor(o["weights"], device=ctx["device"])
            target = w @ x[0]
            contribution = (w @ A[:, j]) * cp[0, j]
            den[j, i] += float(target.double().square().sum())
            err[j, i] += float((target - contribution).double().square().sum())
        pick = np.random.default_rng(
            ctx["cfg"]["patch_sample_seed"] + int(pack["ids"][i])
        ).choice(196, 32, replace=False)
        native_scores.append(mask[0, :, pick].T.cpu().numpy())
        contribution_scores.append(
            (A[pick].abs() * cp[0].norm(dim=-1)[None, :]).cpu().numpy()
        )
        frac.append(pack["masks"][i]["fraction"][pick])
        crowd.append(pack["masks"][i]["crowd"][pick])
        sse += float((pred - x).double().square().sum())
        arr = x[0].double().cpu().numpy()
        summ += arr.sum(0)
        sq += float(np.square(arr).sum())
        patches += len(arr)
        if (i + 1) % 100 == 0:
            s.event(
                "Concept_SAE_evaluation",
                seed=seed,
                cohort=cohort,
                done=i + 1,
                total=len(pack["ids"]),
            )
    f = np.concatenate(frac)
    cr = np.concatenate(crowd)
    scores = np.concatenate(native_scores)
    contrib = np.concatenate(contribution_scores)
    rows = []
    for j, c in enumerate(cats):
        y = f[:, c] / 255 >= ctx["cfg"]["positive_fraction"]
        valid = (
            (f[:, c] / 255 >= ctx["cfg"]["positive_fraction"])
            | (f[:, c] / 255 <= ctx["cfg"]["negative_fraction"])
        ) & (cr <= ctx["cfg"]["negative_fraction"])
        ap = metrics.ap(scores[:, j], y, valid)
        cap = 1 - err[j].sum() / den[j].sum() if den[j].sum() > 1e-12 else None
        if ap is None or cap is None:
            raise ValueError("Concept-SAE fixed category denominator undefined")
        rows.append(
            dict(
                category_index=c,
                capture=float(cap),
                ap=float(ap),
                native_contribution_norm_AP=metrics.ap(contrib[:, j], y, valid),
                concept_token=j,
            )
        )
    report = dict(
        seed=seed,
        kind="concept_sae",
        macro_capture=float(np.mean([r["capture"] for r in rows])),
        macro_ap=float(np.mean([r["ap"] for r in rows])),
        reconstruction=dict(
            mse=sse / (patches * 768),
            ve=1 - sse / max(sq - float(summ @ summ) / patches, 1e-12),
        ),
        categories=rows,
        model_parameters=meta["parameters"],
        scope="Released Concept-SAE adapted to shared DINO/mask task; category-assigned native additive contributions for capture; native mask-times-presence for primary AP; no refitted BSF code or decoder; not capacity/sparsity/update matched",
    )
    result = dict(signature=sig, report=report)
    s.save(key, result)
    s.save(
        "concept_sae_compact_" + cohort + f"_s{seed}",
        dict(den=den, errors=err, image_ids=pack["ids"], category_indices=cats),
    )
    s.write("concept_sae/" + key + ".json", result)
    print(
        "CONCEPT-SAE RESULT",
        cohort,
        seed,
        report["macro_capture"],
        report["macro_ap"],
        report["reconstruction"]["ve"],
        flush=True,
    )
    return result


def run(ctx, pack, cohort):
    for seed in [21, 22, 23]:
        m = None
        try:
            m, meta = fit(ctx, seed)
            evaluate(ctx, seed, m, meta, pack, cohort)
        finally:
            del m
            gc.collect()
            torch.cuda.empty_cache()
