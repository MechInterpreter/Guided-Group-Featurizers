"""Variable-width supervised owners plus exclusive scalar reconstruction atoms."""

import copy
import torch
from torch import nn
from torch.nn import functional as F


def checked_layout(layout, atoms, max_width=16):
    flat = [a for group in layout for a in group]
    if not layout or any(not 1 <= len(g) <= max_width for g in layout):
        raise ValueError("Owner width outside declared bounds")
    if len(flat) != len(set(flat)) or any(
        type(a) is not int or not 0 <= a < atoms for a in flat
    ):
        raise ValueError("Duplicated or invalid atom ownership")
    return copy.deepcopy(layout)


class BankedAGF(nn.Module):
    mode = "owner_router"  # Existing supervised loss adapter; inference has no masks.

    def __init__(
        self,
        weights,
        layout,
        targets,
        head,
        total_budget=48,
        owner_budget=24,
        max_width=16,
    ):
        super().__init__()
        self.d = weights["W_dec"].shape[1]
        self.total_atoms = weights["W_dec"].shape[0]
        self.layout = checked_layout(layout, self.total_atoms, max_width)
        if not 0 < owner_budget <= total_budget <= self.total_atoms:
            raise ValueError("Invalid coordinate budget")
        self.total_budget, self.owner_budget = total_budget, owner_budget
        self.n_groups, self.group_size = len(layout), max_width
        self.targets = copy.deepcopy(targets)
        if len(targets) != len(layout) or any(
            t["contrast_group"] != i or t["detector_group"] != i
            for i, t in enumerate(targets)
        ):
            raise ValueError("Owner readouts must be distinct and persistent")
        for key in ("W_enc", "W_dec", "b_enc"):
            setattr(self, key, nn.Parameter(weights[key].detach().clone()))
        self.head = nn.Parameter(head.detach().clone())
        self.register_buffer(
            "heat_mean", torch.tensor([t["heat_mean"] for t in targets])
        )
        self.register_buffer(
            "heat_scale", torch.tensor([t["heat_scale"] for t in targets])
        )
        ids = torch.zeros(self.n_groups, max_width, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        atom_owner = torch.full((self.total_atoms,), -1, dtype=torch.long)
        for c, g in enumerate(layout):
            ids[c, : len(g)] = torch.tensor(g)
            valid[c, : len(g)] = True
            atom_owner[g] = c
        self.register_buffer("atom_ids", ids)
        self.register_buffer("valid_atoms", valid)
        self.register_buffer("atom_owner", atom_owner)
        self.register_buffer("widths", valid.sum(1))
        self.register_buffer("scalar_ids", torch.nonzero(atom_owner < 0).flatten())
        if len(self.scalar_ids) < total_budget:
            raise ValueError(
                "Scalar bank must be able to fill the complete patch budget"
            )

    def route(self, x, *, total_budget=None, owner_budget=None, oracle_requests=None):
        total = self.total_budget if total_budget is None else total_budget
        allowance = self.owner_budget if owner_budget is None else owner_budget
        if not 0 < allowance <= total <= len(self.scalar_ids):
            raise ValueError("Invalid diagnostic budget")
        raw = x @ self.W_enc + self.b_enc
        owner_raw = raw[:, self.atom_ids] * self.valid_atoms
        logits = (owner_raw.norm(dim=-1) - self.heat_mean) / self.heat_scale
        logits = logits * F.softplus(self.head[:, 0]) + self.head[:, 1]
        p = logits.sigmoid()
        requested = p >= 0.5
        # The explicit oracle is used ONLY in training diagnostics, never forward.
        if oracle_requests is not None:
            requested = oracle_requests.bool()
        order = torch.argsort(p, dim=1, descending=True, stable=True)
        admitted = torch.zeros_like(requested)
        remaining = torch.full((len(x),), allowance, device=x.device, dtype=torch.long)
        rows = torch.arange(len(x), device=x.device)
        for j in range(self.n_groups):
            c = order[:, j]
            w = self.widths[c]
            yes = requested[rows, c] & (w <= remaining)
            admitted[rows, c] = yes
            remaining = remaining - w * yes
        used = (admitted * self.widths).sum(1)
        # Forward is hard; backward uses sigmoid surrogate, not an exact derivative.
        gate = admitted.to(p.dtype)
        if self.training and oracle_requests is None:
            gate = gate + (p - p.detach())
        z_owner = owner_raw * gate[:, :, None]
        scalar_order = torch.argsort(
            raw[:, self.scalar_ids].abs(), dim=1, descending=True, stable=True
        )[:, :total]
        chosen = self.scalar_ids[scalar_order]
        scalar_keep = (
            torch.arange(total, device=x.device)[None] < (total - used)[:, None]
        )
        scalar_z = raw.new_zeros(raw.shape).scatter(
            1, chosen, raw.gather(1, chosen) * scalar_keep
        )
        owner_atom_gate = gate[:, self.atom_owner.clamp_min(0)] * (self.atom_owner >= 0)
        codes = raw * owner_atom_gate + scalar_z
        return (
            codes,
            z_owner,
            logits,
            dict(
                requested=requested,
                admitted=admitted,
                owner_coordinates=used,
                scalar_coordinates=scalar_keep.sum(1),
                scalar_ids=chosen,
                scalar_keep=scalar_keep,
                raw=raw,
                scalar_codes=scalar_z,
            ),
        )

    def routed(self, x):
        _, z, logits, _ = self.route(x)
        return z, logits

    def encode(self, x):
        return self.routed(x)[0]

    def decoder_atoms(self):
        return (self.W_dec[self.atom_ids] * self.valid_atoms[:, :, None]).reshape(
            -1, self.d
        )

    def atoms(self):
        return (
            self.decoder_atoms()
            .detach()
            .reshape(self.n_groups, self.group_size, self.d)
        )

    def forward(self, x):
        codes, z, _, _ = self.route(x)
        return codes @ self.W_dec, z

    def loss(self, x, target=None):
        pred, _ = self(x)
        value = (pred - (x if target is None else target)).square().mean()
        return value, dict(recon=float(value.detach()))

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))


def edited_layout(layout, proposal, atoms, max_width=16):
    out = checked_layout(layout, atoms, max_width)
    c, j = proposal["owner"], proposal.get("atom")
    owned = {a for g in out for a in g}
    op = proposal["operation"]
    if op in ("grow", "exchange") and j in owned:
        raise ValueError("Incoming atom must be scalar")
    if op == "grow":
        out[c].append(j)
    elif op == "exchange":
        out[c].remove(proposal["remove"])
        out[c].append(j)
    elif op == "shrink":
        out[c].remove(proposal["remove"])
    else:
        raise ValueError("This first diagnostic implements scalar transfers only")
    return checked_layout(out, atoms, max_width)
