import copy, gc, hashlib, time
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.training import make_model, weights, weight_sha, measure
from coco_baselines.storage import identity
from coco_adaptation import core
from .study import component_gradients, sync


def fit_recovery(
    ctx,
    key,
    initial,
    targets,
    supervised,
    seed,
    max_updates=None,
    reference_mse=None,
    observer=None,
    initial_head=None,
    reconstruction_weight=1.0,
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
    if (
        initial_head is None
        or tuple(initial_head.shape) != tuple(head.shape)
        or not torch.isfinite(initial_head).all()
    ):
        raise ValueError("Finite trained endpoint head is required")
    head.data.copy_(initial_head.to(device))
    if reconstruction_weight not in (1.0, 4.0):
        raise ValueError("Predeclared reconstruction weights are1 and4")
    fit_identity = dict(
        initial_head_sha=training.weight_sha({"head": initial_head}),
        reconstruction_weight=reconstruction_weight,
    )
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
            recovery_identity=fit_identity,
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
        if st.get("recovery_identity") != fit_identity:
            raise ValueError("Recovery head/objective changed")
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
                native=reconstruction_weight * native,
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
                        reconstruction_weight=reconstruction_weight,
                        weighted_native_loss=float(
                            (reconstruction_weight * native).detach()
                        ),
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
