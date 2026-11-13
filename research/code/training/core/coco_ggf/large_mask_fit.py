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

    try:
        while state["updates"] < stop:
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
            opt.step()
            schedule.step()
            model.normalize_decoder()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            state["seconds"] += time.monotonic() - started
            state["updates"] += 1
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
        checkpoint(drive=False)
        try:
            checkpoint()
            store.event(
                "training_interrupted",
                key=key,
                updates=state["updates"],
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except BaseException:
            pass
        raise
    return state


def fit_guided(
    ctx,
    key,
    initial,
    targets,
    supervised,
    seed,
    max_updates=None,
    reference_mse=None,
    observer=None,
):
    """Identical batches/noise/horizon for paired arms; uninterrupted optimizer."""
    s, cfg, base = ctx["store"], ctx["cfg"], ctx["base"]
    device = cfg["device"]
    target = cfg["updates"]
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    m = core.new_model("vanilla", base, initial, device)
    head = torch.nn.Parameter(torch.zeros(len(targets), 2, device=device))
    opt = torch.optim.Adam(
        list(m.parameters()) + ([head] if supervised else []), lr=cfg["lr"]
    )
    rng = np.random.default_rng(seed)
    st = s.load(key)
    if st is None:
        v = core.validation(m, ctx["packs"]["select"], targets, cfg, device)
        first_ok = v["reconstruction"]["mse"] <= (
            v["reconstruction"]["mse"] if reference_mse is None else reference_mse
        ) * (1 + cfg["fit_reconstruction_tolerance"])
        st = dict(
            updates=0,
            target=target,
            initial_sha=training.weight_sha(initial),
            supervised=supervised,
            target_sha=identity(targets),
            weights=training.weights(m),
            head=head.detach().cpu(),
            best=training.weights(m),
            best_score=v["score"] if first_ok else -1e100,
            best_update=0,
            best_mse=training.weights(m),
            best_mse_score=v["reconstruction"]["mse"],
            best_mse_update=0,
            initial_mse=v["reconstruction"]["mse"]
            if reference_mse is None
            else reference_mse,
            reference_mse=reference_mse,
            own_start_mse=v["reconstruction"]["mse"],
            history=[
                dict(
                    update=0,
                    validation=v,
                    object_checkpoint_eligible=bool(first_ok),
                    rejection_reasons=[]
                    if first_ok
                    else ["common_reconstruction_guard"],
                )
            ],
            seconds=0.0,
            sampled_batches_sha=hashlib.sha256().hexdigest(),
        )
    else:
        if (st["target"], st["initial_sha"], st["supervised"], st["target_sha"]) != (
            target,
            training.weight_sha(initial),
            supervised,
            identity(targets),
        ):
            raise ValueError("Branch identity changed")
        if st.get("reference_mse") != reference_mse:
            raise ValueError("Common reconstruction floor changed")
        m.load_state_dict(st["weights"])
        head.data.copy_(st["head"].to(device))
        opt.load_state_dict(st["optimizer"])
        rng.bit_generator.state = st["sample_rng"]
        torch.set_rng_state(st["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(st["cuda_rng"])
    limit = target if max_updates is None else min(target, max_updates)
    print(key, "resume", st["updates"], "/", target, flush=True)

    def save(drive=True):
        st.update(
            weights=training.weights(m),
            head=head.detach().cpu().clone(),
            optimizer=copy.deepcopy(opt.state_dict()),
            sample_rng=copy.deepcopy(rng.bit_generator.state),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if device.startswith("cuda")
            else None,
        )
        s.save(key, st, drive)

    try:
        while st["updates"] < limit:
            idx = rng.integers(
                len(ctx["packs"]["train"]["ids"]), size=cfg["images_per_update"]
            )
            st["sampled_batches_sha"] = hashlib.sha256(
                st["sampled_batches_sha"].encode() + idx.tobytes()
            ).hexdigest()
            x = torch.as_tensor(
                np.array(ctx["packs"]["train"]["x"][idx], copy=True), device=device
            )
            m.train()
            began = time.monotonic()
            xx = x.reshape(-1, m.d)
            noisy = (
                xx + cfg["snr"] * xx.std() * torch.randn_like(xx) if cfg["snr"] else xx
            )
            native, info = m.loss(noisy, target=xx)
            cap = det = native * 0
            objects = 0
            if supervised:
                cap, det, objects = core.supervised_loss(
                    m,
                    x,
                    [ctx["packs"]["train"]["masks"][i] for i in idx],
                    targets,
                    head,
                    cfg,
                )
            terms = dict(
                native=native,
                capture=cfg["lambda_capture"] * cap,
                detection=cfg["lambda_detection"] * det,
            )
            u = st["updates"] + 1
            diagnostics = None
            if supervised and u % cfg["diagnostic_grad_every"] == 0:
                diagnostics = component_gradients(m, terms)
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite GGF/native loss")
            opt.zero_grad()
            loss.backward()
            norms = {
                name: float(p.grad.norm())
                for name, p in m.named_parameters()
                if p.grad is not None
            }
            grad = float(
                torch.nn.utils.clip_grad_norm_(
                    list(m.parameters()) + ([head] if supervised else []),
                    cfg["grad_clip"],
                )
            )
            if not np.isfinite(grad):
                raise FloatingPointError("Nonfinite gradient")
            opt.step()
            m.normalize_decoder()
            sync(device)
            st["seconds"] += time.monotonic() - began
            st["updates"] = u
            if u % cfg["evaluate_every"] == 0 or u == target:
                val = core.validation(m, ctx["packs"]["select"], targets, cfg, device)
                eligible = val["reconstruction"]["mse"] <= st["initial_mse"] * (
                    1 + cfg["fit_reconstruction_tolerance"]
                )
                if (
                    eligible
                    and val["score"] is not None
                    and val["score"] > st["best_score"]
                ):
                    st.update(
                        best=training.weights(m), best_score=val["score"], best_update=u
                    )
                if val["reconstruction"]["mse"] < st["best_mse_score"]:
                    st.update(
                        best_mse=training.weights(m),
                        best_mse_score=val["reconstruction"]["mse"],
                        best_mse_update=u,
                    )
                st["history"].append(
                    dict(
                        update=u,
                        validation=val,
                        object_checkpoint_eligible=bool(eligible),
                        rejection_reasons=[]
                        if eligible
                        else ["reconstruction_exceeded_predeclared_guard"],
                        native_loss=float(native.detach()),
                        capture_loss=float(cap.detach()),
                        detection_loss=float(det.detach()),
                        total_loss=float(loss.detach()),
                        native_loss_info=info,
                        gradient_norm=grad,
                        parameter_gradient_norms=norms,
                        component_gradients=diagnostics,
                        objects_in_batch=objects,
                        seconds_per_update=st["seconds"] / u,
                    )
                )
                if observer is not None:
                    observer(m, u)
                save()
                s.write("histories/" + key + ".json", st["history"])
                s.event(
                    "branch_checkpoint",
                    key=key,
                    update=u,
                    total=target,
                    best_object_update=st["best_update"],
                    best_mse_update=st["best_mse_update"],
                    capture=val["macro_capture"],
                    AP=val["macro_ap"],
                    VE=val["reconstruction"]["ve"],
                )
            elif u % cfg["local_save_every"] == 0:
                save(True)
                print(key, u, "/", target, flush=True)
        save()
    except BaseException as exc:
        save(False)
        try:
            save()
            s.event(
                "fit_error",
                key=key,
                update=st["updates"],
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except BaseException:
            pass
        raise
    del m, head, opt
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return st
