"""Pinned released BSFs and explicitly separate paper-specification adapters.

Paper adapters reproduce equations, not the authors' unpublished checkpoints.
All fits in this study use a common COCO budget, not the paper's ImageNet schedule.
"""
import torch
from torch import nn
from bsf.base import BSF, group_topk
from coco_baselines.training import make_model as released


class PaperVanilla(BSF):
    def __init__(self, d, groups, width, active):
        super().__init__(d, groups, width)
        self.l0 = active
        w = torch.randn(groups, width, d)
        w /= w.norm(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        self.W_dec = nn.Parameter(w.reshape(-1, d))
        self.W_enc = nn.Parameter(self.W_dec.detach().T.clone())
        self.b_enc = nn.Parameter(torch.zeros(groups * width))

    def preact(self, x):
        return (x @ self.W_enc + self.b_enc).reshape(-1, self.n_groups, self.group_size)

    def encode(self, x):
        a = self.preact(x)
        return a * group_topk(a.norm(dim=-1), self.l0)[..., None]

    def auxiliary(self, x, target, z):
        a = self.preact(x)
        selected = z.ne(0).any(-1)
        norms = a.norm(dim=-1).masked_fill(selected, -torch.inf)
        mask = group_topk(norms, min(self.l0, self.n_groups - self.l0)) * (~selected)
        runner_up = self.decode(a * mask[..., None])
        residual = (target - self.decode(z)).detach()
        return (residual - runner_up).square().mean() / self.l0

    def loss(self, x, target=None):
        target = x if target is None else target
        z = self.encode(x)
        rec = (target - self.decode(z)).square().mean()
        aux = self.auxiliary(x, target, z)
        return rec + aux, dict(recon=float(rec.detach()), aux=float(aux.detach()))

    @torch.no_grad()
    def normalize_decoder(self):
        w = self.W_dec.reshape(self.n_groups, self.group_size, self.d)
        w.div_(w.norm(dim=(1, 2), keepdim=True).clamp_min(1.0))


class PaperGrassmannian(PaperVanilla):
    def __init__(self, d, groups, width, active):
        BSF.__init__(self, d, groups, width)
        self.l0 = active
        q = torch.linalg.qr(torch.randn(groups, d, width), mode='reduced').Q
        self.W_dec = nn.Parameter(q.transpose(1, 2).reshape(-1, d))
        self.log_gamma = nn.Parameter(torch.zeros(()))
        self.register_buffer('projection_steps', torch.zeros((), dtype=torch.long))

    def preact(self, x):
        return (x @ self.W_dec.T).reshape(-1, self.n_groups, self.group_size) * self.log_gamma.exp()

    @torch.no_grad()
    def normalize_decoder(self):
        self.projection_steps.add_(1)
        if int(self.projection_steps) % 20 == 0:
            w = self.W_dec.reshape(self.n_groups, self.group_size, self.d)
            w.copy_(torch.linalg.qr(w.transpose(1, 2), mode='reduced').Q.transpose(1, 2))


class PaperGroupLasso(PaperVanilla):
    """Block shrinkage, learned positive thresholds and excess-activity L2,1.

The paper does not publish every threshold/penalty calibration detail. This
adapter makes those choices explicit and tunes the penalty on development only.
"""
    def __init__(self, d, groups, width, active, coefficient=.001):
        super().__init__(d, groups, width, active)
        self.raw_theta = nn.Parameter(torch.full((groups,), -2.0))
        self.register_buffer('threshold_initialized', torch.tensor(False))
        self.register_buffer('coefficient', torch.tensor(float(coefficient)))

    def encode(self, x):
        a = self.preact(x)
        norms = a.norm(dim=-1)
        if self.training and not bool(self.threshold_initialized):
            with torch.no_grad():
                t = torch.quantile(norms.flatten(), 1-self.l0/self.n_groups).clamp_min(1e-4)
                self.raw_theta.copy_(torch.log(torch.expm1(t)).expand_as(self.raw_theta))
                self.threshold_initialized.fill_(True)
        theta = torch.nn.functional.softplus(self.raw_theta)
        factor = (1-theta/norms.clamp_min(1e-12)).clamp_min(0)
        return a * factor[..., None]

    def loss(self, x, target=None):
        target = x if target is None else target
        z = self.encode(x)
        rec = (target-self.decode(z)).square().mean()
        activity = (z.norm(dim=-1)>0).float().sum(-1).mean()
        penalty = self.coefficient * z.norm(dim=-1).sum(-1).mean() if float(activity.detach()) > self.l0 else rec*0
        # AuxK follows the paper's next-block rule using PRE-gate block norms.
        aux = self.auxiliary(x, target, z)
        return rec+penalty+aux, dict(recon=float(rec.detach()), aux=float(aux.detach()),
                                  l21=float(penalty.detach()), active=float(activity.detach()))


def model(method, groups, width, active, d=768, coefficient=.001):
    if active < 1 or active >= groups or width > d:
        raise ValueError('Invalid block configuration')
    if method.startswith('released_'):
        cfg=dict(groups=groups,block_rank=width,active_groups=active)
        return released(method.removeprefix('released_'), cfg, d)
    cls={'paper_vanilla':PaperVanilla,'paper_grassmannian':PaperGrassmannian,
         'paper_group_lasso':PaperGroupLasso}[method]
    return cls(d,groups,width,active,coefficient) if method=='paper_group_lasso' else cls(d,groups,width,active)


def encoder_parameters(m):
    if not hasattr(m,'W_enc'):
        raise ValueError('AGF must use the released free encoder')
    return [m.W_enc, m.b_enc]


BASELINE_AUDIT = {
    'released_vanilla':'Pinned released unit-atom normalization; reconstruction only; no AuxK.',
    'released_grassmannian':'Pinned released QR on forward and one gamma per group; no AuxK.',
    'released_group_lasso':'Pinned released hard BlockJumpReLU, dual L0 penalty and tracked bandwidth; no shrinkage/AuxK.',
    'paper_vanilla':'Free maps, Frobenius unit-ball decoder blocks, Top-k and AuxK coefficient 1/k.',
    'paper_grassmannian':'Tied decoder, one positive shared gamma, QR projection every 20 updates and AuxK.',
    'paper_group_lasso':'Soft block shrinkage, learned thresholds, excess-activity L2,1 and AuxK; unspecified penalty details tuned on development.',
}
