"""Read-only rank diagnostics of coefficients conditional on block activity.

All accumulation is float64. Covariance is centered, not a second moment.
Covariance stable rank and coefficient-matrix stable rank are both reported
with different names; they must not be confused with participation ratio.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def guard_path(path):
    if Path(path).as_posix().startswith('/content/drive/') and (not os.path.ismount('/content/drive') or not Path('/content/drive/MyDrive').is_dir()):
        raise OSError('Drive disconnected. Remount and resume; no replacement output directory was created.')


def atomic_json(path, value):
    path = Path(path); guard_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.pending')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


class Moments:
    def __init__(self, groups, width):
        self.n = np.zeros(groups, np.int64)
        self.mean = np.zeros((groups, width), np.float64)
        self.m2 = np.zeros((groups, width, width), np.float64)
        self.patches = 0

    def add_selected(self, group_ids, values, patches):
        """Chan merge of per-chunk centered moments for selected nonzero codes."""
        values = np.asarray(values, np.float64)
        group_ids = np.asarray(group_ids, np.int64)
        G, b = self.mean.shape
        if values.shape != (len(group_ids), b) or not np.isfinite(values).all():
            raise ValueError('Invalid selected coefficient array')
        if len(group_ids) and (group_ids.min() < 0 or group_ids.max() >= G):
            raise ValueError('Invalid group ID')
        count = np.bincount(group_ids, minlength=G)
        sums = np.stack([np.bincount(group_ids, weights=values[:, j], minlength=G) for j in range(b)], axis=1)
        cm = np.divide(sums, count[:, None], out=np.zeros_like(sums), where=count[:, None] > 0)
        # Center before forming products: avoids subtracting large second moments.
        centered = values - cm[group_ids]
        chunk_m2 = np.zeros_like(self.m2)
        for j in range(b):
            for k in range(j, b):
                product = np.bincount(group_ids, weights=centered[:, j] * centered[:, k], minlength=G)
                chunk_m2[:, j, k] = product; chunk_m2[:, k, j] = product
        total = self.n + count; delta = cm - self.mean
        factor = np.divide(self.n * count.astype(np.float64), total, out=np.zeros(G), where=total > 0)
        self.m2 += chunk_m2 + factor[:, None, None] * delta[:, :, None] * delta[:, None, :]
        self.mean += delta * np.divide(count, total, out=np.zeros(G), where=total > 0)[:, None]
        self.n = total; self.patches += int(patches)

    def add_dense(self, z):
        z = np.asarray(z)
        if z.ndim != 3 or z.shape[1:] != self.mean.shape:
            raise ValueError('Wrong coefficient shape')
        active = np.any(z != 0, axis=-1)
        group_ids = np.nonzero(active)[1]
        self.add_selected(group_ids, z[active], len(z))


def rank_statistics(eigenvalues):
    ev = np.asarray(eigenvalues, np.float64)
    if ev.ndim != 1 or not np.isfinite(ev).all():
        raise ValueError('Invalid covariance spectrum')
    scale = max(float(np.max(np.abs(ev))), np.finfo(np.float64).tiny)
    if ev.min(initial=0) < -1e-9 * scale:
        raise FloatingPointError('Covariance has a materially negative eigenvalue')
    ev = np.maximum(ev, 0); trace = float(ev.sum())
    if trace <= np.finfo(np.float64).tiny:
        return dict(effective_rank=None, participation_ratio=None, covariance_stable_rank=None,
                    matrix_stable_rank=None, trace=trace)
    p = ev / trace; positive = p > 0
    return dict(effective_rank=float(np.exp(-np.sum(p[positive] * np.log(p[positive])))),
                participation_ratio=float(trace * trace / np.square(ev).sum()),
                covariance_stable_rank=float(np.square(ev).sum() / ev.max() ** 2),
                matrix_stable_rank=float(trace / ev.max()), trace=trace)


def block_statistics(moments, decoder, concept_count=75, reliable_min=32):
    G, b = moments.mean.shape
    decoder = np.asarray(decoder, np.float64)
    if decoder.ndim != 3 or decoder.shape[:2] != (G, b):
        raise ValueError('Wrong decoder shape')
    spectra = np.zeros((G, b), np.float64)
    contribution_spectra = np.zeros_like(spectra); records = []
    cov = np.zeros_like(moments.m2)
    for g in range(G):
        n = int(moments.n[g]); status = 'inactive' if n == 0 else 'too_few_samples' if n < 2 else 'valid'
        if n >= 2:
            cov[g] = (moments.m2[g] + moments.m2[g].T) / (2 * (n - 1))
            spectra[g] = np.linalg.eigvalsh(cov[g])
            code = rank_statistics(spectra[g])
            gram = decoder[g] @ decoder[g].T
            v, q = np.linalg.eigh((gram + gram.T) / 2)
            root = (q * np.sqrt(np.maximum(v, 0))) @ q.T
            small = root @ cov[g] @ root
            contribution_spectra[g] = np.linalg.eigvalsh((small + small.T) / 2)
            geometry = rank_statistics(contribution_spectra[g])
            if code['effective_rank'] is None: status = 'zero_variation'
        else:
            code = rank_statistics(np.zeros(b)); geometry = code.copy()
        raw_energy = np.diag(cov[g]) * max(n - 1, 0) + n * np.square(moments.mean[g])
        total_energy = float(raw_energy.sum())
        row = dict(group=g, slot_family='initialized_slots' if g < concept_count else 'reserve_slots',
                   allocated_width=b, active_patches=n, activity_fraction=n / max(moments.patches, 1),
                   status=status, reliable_samples=bool(n >= reliable_min and status == 'valid'),
                   added_coordinate_energy_fraction=None if b <= 5 or total_energy == 0 else float(raw_energy[5:].sum() / total_energy))
        row.update({'code_' + k: v for k, v in code.items()})
        row.update({'contribution_' + k: v for k, v in geometry.items()})
        row['code_utilization'] = None if code['effective_rank'] is None else code['effective_rank'] / b
        row['contribution_utilization'] = None if geometry['effective_rank'] is None else geometry['effective_rank'] / b
        records.append(row)
    return records, dict(counts=moments.n, means=moments.mean, covariance=cov,
                         code_eigenvalues=spectra, contribution_eigenvalues=contribution_spectra)


def summarize(records):
    result = []
    metrics = ['code_effective_rank', 'code_participation_ratio', 'code_covariance_stable_rank',
               'code_matrix_stable_rank', 'code_utilization', 'contribution_effective_rank',
               'contribution_utilization', 'added_coordinate_energy_fraction', 'activity_fraction']
    for family in ['all', 'initialized_slots', 'reserve_slots']:
        rows = [r for r in records if family == 'all' or r['slot_family'] == family]
        for subset in ['all_estimable', 'at_least_32_active_patches']:
            valid = [r for r in rows if r['status'] == 'valid' and (subset == 'all_estimable' or r['reliable_samples'])]
            out = dict(slot_family=family, sample_subset=subset, total_groups=len(rows), valid_groups=len(valid),
                       inactive_groups=sum(r['status'] == 'inactive' for r in rows),
                       insufficient_groups=sum(r['status'] == 'too_few_samples' for r in rows),
                       zero_variation_groups=sum(r['status'] == 'zero_variation' for r in rows),
                       median_active_patches=float(np.median([r['active_patches'] for r in rows])) if rows else None)
            for metric in metrics:
                rr = [r for r in valid if r[metric] is not None]
                out[metric + '_mean'] = float(np.mean([r[metric] for r in rr])) if rr else None
                out[metric + '_median'] = float(np.median([r[metric] for r in rr])) if rr else None
                out[metric + '_activity_weighted'] = float(np.average([r[metric] for r in rr], weights=[r['active_patches'] for r in rr])) if rr else None
            result.append(out)
    return result


def save_moments(folder, signature, moments, generation):
    folder = Path(folder); guard_path(folder); folder.mkdir(parents=True, exist_ok=True)
    slot = generation % 2; path = folder / f'slot{slot}.npz'; temp = path.with_suffix('.pending.npz')
    with temp.open('wb') as f:
        np.savez_compressed(f, n=moments.n, mean=moments.mean, m2=moments.m2, patches=moments.patches)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    checksum = sha(path)
    if sha(path) != checksum: raise OSError('Analysis checkpoint readback mismatch')
    atomic_json(path.with_suffix('.json'), dict(signature=signature, generation=generation, sha256=checksum, patches=moments.patches))


def load_moments(folder, signature, groups, width):
    folder = Path(folder); candidates = []
    for meta in folder.glob('slot*.json'):
        record = json.loads(meta.read_text())
        if record['signature'] != signature: raise ValueError('Analysis signature changed; use a new output')
        path = meta.with_suffix('.npz')
        if path.exists() and sha(path) == record['sha256']:
            candidates.append((record['generation'], path, record))
    if not candidates:
        if list(folder.glob('slot*.json')): raise OSError('Both analysis checkpoint generations are corrupt')
        return Moments(groups, width), 0
    generation, path, record = max(candidates)
    with np.load(path, allow_pickle=False) as a:
        moments = Moments(groups, width)
        moments.n = a['n']; moments.mean = a['mean']; moments.m2 = a['m2']; moments.patches = int(a['patches'])
    if moments.mean.shape != (groups, width) or moments.m2.shape != (groups, width, width) or moments.n.shape != (groups,) or moments.patches != record['patches']:
        raise ValueError('Invalid analysis checkpoint dimensions')
    if not np.isfinite(moments.mean).all() or not np.isfinite(moments.m2).all():
        raise FloatingPointError('Nonfinite saved analysis state')
    return moments, generation
