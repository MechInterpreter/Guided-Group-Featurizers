"""Annotation-only quota allocation with a constrained covering fallback."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack


def allocate(ids, evidence, counts, minimums, seed, min_categories, greedy, event=None):
    emit = event or (lambda *args, **kwargs: None)
    try:
        return greedy(ids, evidence, counts, minimums, seed, min_categories)
    except ValueError as exc:
        if list(counts) != ['test'] or 'Cannot meet test' not in str(exc):
            raise
        emit('holdout_greedy_capacity_failure', error=str(exc))
    ids = np.asarray(ids)
    usable = evidence.all(2).astype(np.int32)
    quota, size = int(minimums['test']), int(counts['test'])
    eligible = np.flatnonzero(usable.sum(0) >= int(np.ceil(quota * 1.2)))
    rng = np.random.default_rng(seed)
    priority = np.empty(len(ids), int)
    priority[rng.permutation(len(ids))] = np.arange(len(ids))
    # Identical category-coverage patterns need at most quota representatives
    # for the quota core. Retain the full pool for prevalence-based filling.
    patterns = {}
    for i in np.argsort(priority):
        bits = usable[i, eligible]
        if not bits.any():
            continue
        bucket = patterns.setdefault(bits.tobytes(), [])
        if len(bucket) < quota:
            bucket.append(int(i))
    kernel = np.array([i for bucket in patterns.values() for i in bucket], int)
    matrix = vstack([csc_matrix(usable[kernel][:, eligible].T.astype(float)),
                     csc_matrix(np.ones((1,len(kernel))))], format='csc')
    emit('holdout_constrained_selection_start', candidates=len(ids), kernel_candidates=len(kernel),
         categories=len(eligible), quota=quota, images=size, solver_seconds_limit=60)
    result = milp(c=np.ones(len(kernel)), integrality=np.ones(len(kernel)),
        bounds=Bounds(0., 1.), constraints=LinearConstraint(matrix,
            np.r_[np.full(len(eligible),quota),0.],np.r_[np.full(len(eligible),np.inf),size]),
        options=dict(time_limit=60., mip_rel_gap=.02))
    emit('holdout_constrained_selection_result', status=int(result.status),
         message=str(result.message), objective=None if result.fun is None else float(result.fun))
    if result.x is None:
        if result.status == 2:
            raise ValueError('The annotation pool cannot meet all quotas within the declared test size. '
                             'Expand the annotation-only pool or revise the test size before training.')
        raise ValueError('No feasible quota allocation found within the solver limit. '
                         'This does not establish infeasibility; inspect the logged solver result.')
    chosen = kernel[result.x > .5].tolist()
    have = usable[chosen].sum(0)
    if (have[eligible] < quota).any():
        raise ValueError('Solver incumbent failed the exact category quota check')
    if len(chosen) > size:
        raise ValueError(f'Quota core uses {len(chosen)} images, above the declared {size}. '
                         'Expand the annotation-only pool or revise the test size before training. '
                         'No quotas or categories were dropped.')
    free = np.ones(len(ids), bool)
    free[chosen] = False
    target = usable.mean(0)
    target[eligible] = np.maximum(target[eligible], quota / size)
    while len(chosen) < size:
        pool = np.flatnonzero(free)
        pool = pool[np.argsort(priority[pool])[:1024]]
        deficit = target[eligible] * size - have[eligible]
        bits = usable[pool][:, eligible]
        cost = np.sum(((deficit-bits)**2-deficit**2) / np.maximum(target[eligible]*size, 1), axis=1)
        i = int(pool[np.lexsort((priority[pool], cost))[0]])
        chosen.append(i)
        free[i] = False
        have += usable[i]
    assert len(chosen) == size and len(set(chosen)) == size
    assert (have[eligible] >= quota).all()
    report = dict(category_indices=eligible.tolist(), candidate_usable_counts=usable.sum(0).tolist(),
        excluded_categories=[dict(category_index=int(c), usable_images=int(usable[:, c].sum()),
            required_with_reserve=int(np.ceil(quota*1.2))) for c in range(usable.shape[1]) if c not in eligible],
        split_counts=dict(test=have.tolist()), minimums=minimums, sizes=counts,
        shared_target_prevalence=target.tolist(), allocator='constrained_cover_then_prevalence_fill',
        solver_status=int(result.status), solver_message=str(result.message),
        quota_core_images=int(np.sum(result.x > .5)),
        note='Annotation-only stratification. Exact quotas checked; no model scores or historical images.')
    return dict(test=ids[rng.permutation(chosen)].tolist()), report
