"""Disposable update/inference benchmarks with explicit FLOP coverage.

Registered FLOPs are never labelled full FLOPs. CPU NNLS and eigensolvers
are retained in runtime measurements and listed as coverage gaps.
"""

import collections, copy, gc, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode, flop_registry
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class Operations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[str(func)] += 1
        return func(*args, **(kwargs or {}))


def summarize_flops(counts, calls):
    registered = {str(k): int(v) for k, v in counts.items()}
    packets = {str(k) for k in flop_registry}
    unregistered = {
        k: v for k, v in calls.items() if k.rsplit(".", 1)[0] not in packets
    }
    # This is a coverage inventory, not a count of missing floating operations.
    return dict(
        registered_operator_FLOPs=sum(registered.values()),
        operator_FLOPs=registered,
        operators_without_registered_formula=unregistered,
        full_FLOPs_available=False,
        accounting="FMA=2; executed PyTorch registered formulas, including backward when in scope; no fabricated total for unsupported work",
    )


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(make_call, images, repeats=3, flop_repeats=1):
    """Factory resets the model/optimizer for each independent measured repeat."""
    times = []
    peaks = []
    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        call, cleanup = make_call()
        sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            call()
            sync()
        finally:
            times.append(time.perf_counter() - start)
            if torch.cuda.is_available():
                peaks.append(
                    dict(
                        allocated=torch.cuda.max_memory_allocated(),
                        reserved=torch.cuda.max_memory_reserved(),
                    )
                )
            cleanup()
    probes = []
    for _ in range(flop_repeats):
        call, cleanup = make_call()
        ops = Operations()
        try:
            with ops, FlopCounterMode(display=False) as counter:
                call()
            sync()
            probes.append(
                summarize_flops(counter.get_flop_counts()["Global"], ops.calls)
            )
        finally:
            cleanup()
    return dict(
        seconds=times,
        seconds_mean=float(np.mean(times)),
        seconds_SD=float(np.std(times, ddof=1)),
        images=images,
        seconds_per_image_mean=float(np.mean(times)) / images,
        peak_GPU_allocated_bytes=max((p["allocated"] for p in peaks), default=None),
        peak_GPU_reserved_bytes=max((p["reserved"] for p in peaks), default=None),
        uncertainty="Sample SD over repeated disposable execution, not training-seed uncertainty",
        cold_first_update=True,
        warmup_excluded=False,
        flop_probes=probes,
        scope="Input batch preparation through update/inference return; excludes checkpoint I/O and validation",
    )


def bsf_factory(ctx, row, state, ids, training_mode=False, adapt=False):
    from scale_models import model, encoder_parameters
    from coco_adaptation import core
    from coco_ggf.capacity_fit import Balancer

    def factory():
        m = model("released_vanilla", row["groups"], 12, row["active_groups"], 768).to(
            ctx["device"]
        )
        m.load_state_dict(state["weights"])
        m.train(training_mode)
        supervised = training_mode and state.get("head") is not None
        head = (
            torch.nn.Parameter(state["head"].to(ctx["device"]).clone())
            if supervised
            else None
        )
        opt = (
            torch.optim.Adam(
                list(m.parameters()) + ([head] if supervised else []),
                lr=row.get("lr", 0.0003),
            )
            if training_mode
            else None
        )
        if opt is not None and state.get("optimizer"):
            opt.load_state_dict(copy.deepcopy(state["optimizer"]))
        controller = None
        if supervised and state.get("balance"):
            recipe = copy.deepcopy(ctx["balance_recipe"])
            recipe["bounds"]["capture"][1] = row.get("cap", 0.4)
            recipe["bounds"]["detection"][1] = row.get("cap", 0.4)
            controller = Balancer(recipe, ctx["device"])
            controller.load_state_dict(copy.deepcopy(state["balance"]))

        def call():
            p = ctx["packs"]["train"]
            x = torch.as_tensor(np.array(p["x"][ids], copy=True), device=ctx["device"])
            xx = x.reshape(-1, 768)
            if not training_mode:
                with torch.no_grad():
                    return m(xx)
            # Exact common loss/controller formulas; no checkpoint or source write.
            rx = xx
            if row["groups"] == 19:
                known = torch.as_tensor(
                    np.stack([p["masks"][i]["crowd"] for i in ids]).reshape(-1)
                    <= ctx["cfg"]["negative_fraction"],
                    device=ctx["device"],
                )
                rx = xx[known]
            snr = 0.0 if row["groups"] == 19 else ctx["rule"]["snr"]
            noise = torch.randn(
                rx.shape,
                device=ctx["device"],
                generator=torch.Generator(device=ctx["device"]).manual_seed(319),
            )
            rec, _ = m.loss(rx + snr * rx.std() * noise, target=rx)
            terms = [rec, rec * 0, rec * 0]
            weights = [1.0, 0.0, 0.0]
            if supervised:
                terms[1], terms[2], _ = core.supervised_loss(
                    m, x, [p["masks"][i] for i in ids], row["targets"], head, ctx["cfg"]
                )
                if controller is not None:
                    controller.observe(terms)
                    weights = controller.weights().detach().cpu().tolist()
                    if adapt and controller.reference is not None:
                        norms = []
                        for term in terms:
                            gradients = torch.autograd.grad(
                                term,
                                encoder_parameters(m),
                                retain_graph=True,
                                allow_unused=True,
                            )
                            norms.append(
                                torch.sqrt(
                                    sum(
                                        (
                                            g.detach().square().sum()
                                            for g in gradients
                                            if g is not None
                                        ),
                                        rec.detach() * 0,
                                    )
                                )
                            )
                        controller.adapt(terms, torch.stack(norms))
                else:
                    weights = ctx["rule"]["fixed_weights"]
            loss = sum(w * t for w, t in zip(weights, terms))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(m.parameters()) + ([head] if supervised else []),
                ctx["rule"]["clip"],
            )
            opt.step()
            m.normalize_decoder()
            return loss.detach()

        def cleanup():
            nonlocal m, opt, controller, head
            m = opt = controller = head = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return call, cleanup

    return factory


def native_factory(ctx, D, ids, mode):
    from voc_pair import native_step, NativeSLiCS, SOLVER

    def factory():
        dictionary = D.to(ctx["device"]).clone()
        m = None
        if mode == "joint":
            from gpu_joint_nnls import GPUJointModel

            m = GPUJointModel(dictionary, SOLVER, device=ctx["device"])
        elif mode == "partial":
            m = NativeSLiCS(dictionary, SOLVER, "partial")

        def call():
            p = ctx["packs"]["train"]
            x = torch.as_tensor(
                np.array(p["x"][ids], copy=True), device=ctx["device"]
            ).reshape(-1, 768)
            if mode == "training":
                return native_step(ctx, dictionary, ids)[0]
            return (
                m.encode(x) if mode == "partial" else m.solve(x) @ dictionary.double()
            )

        def cleanup():
            nonlocal dictionary, m
            dictionary = m = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return call, cleanup

    return factory


def concept_factory(ctx, state, ids, stage=None):
    import concept_sae as native

    def factory():
        m = native.build(
            native.source(ctx["options"]["concept_source"]),
            len(ctx["cfg"]["category_indices"]),
        ).to(ctx["device"])
        m.load_state_dict(state["weights"])
        m.train(stage is not None)
        part = native.configure(m, stage) if stage else None
        opt = (
            torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.001)
            if stage
            else None
        )

        def call():
            if stage:
                return native.optimizer_step(
                    m,
                    part,
                    opt,
                    ctx["packs"]["train"],
                    ids,
                    ctx["cfg"]["category_indices"],
                    ctx["device"],
                    stage,
                    8,
                )
            x = torch.as_tensor(
                np.array(ctx["packs"]["train"]["x"][ids], copy=True),
                device=ctx["device"],
            )
            with torch.no_grad():
                return m(x)

        def cleanup():
            nonlocal m, part, opt
            m = part = opt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return call, cleanup

    return factory
