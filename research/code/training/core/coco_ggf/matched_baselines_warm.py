import copy, gc, hashlib, time
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.training import make_model, weights, weight_sha, measure
from coco_baselines.storage import identity
from coco_adaptation import core
from .study import component_gradients, sync


def fit_warm(
    store,
    key,
    method,
    x,
    selection,
    cfg,
    seed,
    lr,
    device,
    max_updates=None,
    initial=None,
    drive_every=50,
):
    """No masks or labels enter this function. Full floor batches like bsf.train."""
    torch.manual_seed(seed)
    model = make_model(method, cfg, x.shape[1]).to(device)
    initial_sha = None if initial is None else weight_sha(initial)
    if initial is not None:
        model.load_state_dict(initial)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    nb = len(x) // cfg["batch"]
    target = nb * cfg["epochs"]
    if nb < 1 or target < 4:
        raise ValueError("Need at least one full batch and four training updates")
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=target,
        pct_start=cfg["warmup"],
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )
    state = store.load(key)
    if state is None:
        state = dict(
            updates=0,
            weights=weights(model),
            best=None,
            fallback=None,
            best_score=None,
            fallback_score=None,
            best_update=None,
            history=[],
            seconds=0.0,
            permutation=None,
            opt=opt.state_dict(),
            scheduler=schedule.state_dict(),
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if device.startswith("cuda")
            else None,
            method=method,
            seed=seed,
            lr=lr,
            target=target,
            parameters=sum(p.numel() for p in model.parameters()),
            class_name=type(model).__module__ + "." + type(model).__name__,
        )
    else:
        if state["target"] != target or state.get("initial_sha") != initial_sha:
            raise ValueError("Warm resume horizon/initialization changed")
        model.load_state_dict(state["weights"])
        opt.load_state_dict(state["opt"])
        schedule.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        if device.startswith("cuda") and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    state.setdefault("initial_sha", initial_sha)
    state.setdefault("sampled_batches_sha", hashlib.sha256().hexdigest())
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    stop = target if max_updates is None else min(target, max_updates)
    print(key, "resume", state["updates"], "/", target, flush=True)
    if state["updates"] >= stop:
        return state

    def checkpoint(drive=True):
        state.update(
            weights=weights(model),
            opt=copy.deepcopy(opt.state_dict()),
            scheduler=copy.deepcopy(schedule.state_dict()),
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if device.startswith("cuda")
            else None,
        )
        store.save(key, state, drive)

    checkpoint()
    committed = True
    try:
        while state["updates"] < stop:
            committed = False
            u = state["updates"]
            epoch, bi = divmod(u, nb)
            if bi == 0:
                state["permutation"] = torch.randperm(len(tx))
            idx = state["permutation"][bi * cfg["batch"] : (bi + 1) * cfg["batch"]]
            state["sampled_batches_sha"] = hashlib.sha256(
                state["sampled_batches_sha"].encode() + idx.numpy().tobytes()
            ).hexdigest()
            xb = tx[idx.to(device)]
            model.train()
            started = time.monotonic()
            noisy = (
                xb
                if cfg["snr"] <= 0
                else xb + cfg["snr"] * xb.std() * torch.randn_like(xb)
            )
            loss, info = model.loss(noisy, target=xb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            opt.zero_grad()
            loss.backward()
            if not all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            ):
                raise FloatingPointError("Nonfinite native warm-up gradient")
            opt.step()
            schedule.step()
            model.normalize_decoder()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            state["seconds"] += time.monotonic() - started
            state["updates"] += 1
            committed = True
            due = (
                state["updates"] % cfg["save_updates"] == 0
                or state["updates"] == target
            )
            if due:
                val = training.measure(model, selection, device, cfg["batch"])
                eligible = (
                    method != "group_lasso"
                    or abs(val["active_groups"] - cfg["active_groups"])
                    <= cfg["sparsity_tolerance"] * cfg["active_groups"]
                )
                row = dict(
                    update=state["updates"],
                    epoch=state["updates"] / nb,
                    training_loss=float(loss.detach()),
                    lr=float(opt.param_groups[0]["lr"]),
                    selection=val,
                    sparsity_eligible=eligible,
                    loss_info={k: float(v) for k, v in info.items()},
                    seconds_per_update=state["seconds"] / state["updates"],
                )
                state["history"].append(row)
                if (
                    state["fallback_score"] is None
                    or val["mse"] < state["fallback_score"]
                ):
                    state.update(fallback=weights(model), fallback_score=val["mse"])
                if eligible and (
                    state["best_score"] is None or val["mse"] < state["best_score"]
                ):
                    state.update(
                        best=weights(model),
                        best_score=val["mse"],
                        best_update=state["updates"],
                    )
                checkpoint()
                store.write(f"histories/{key}.json", state["history"])
                store.event(
                    "training_checkpoint",
                    key=key,
                    updates=state["updates"],
                    total=target,
                    epoch=row["epoch"],
                    selection_mse=val["mse"],
                    selection_ve=val["ve"],
                    active_groups=val["active_groups"],
                    seconds_per_update=row["seconds_per_update"],
                )
            elif state["updates"] % cfg["local_save_updates"] == 0:
                checkpoint(drive=state["updates"] % drive_every == 0)
                store.event(
                    "training_heartbeat",
                    key=key,
                    updates=state["updates"],
                    total=target,
                    compute_seconds=state["seconds"],
                )
        checkpoint()
        del tx, model, opt, schedule
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except BaseException as exc:
        if committed:
            checkpoint(drive=False)
            try:
                checkpoint()
            except BaseException:
                pass
        try:
            store.event(
                "training_interrupted",
                key=key,
                updates=state["updates"],
                replay_last_complete_slot=not committed,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except BaseException:
            pass
        raise
    return state
