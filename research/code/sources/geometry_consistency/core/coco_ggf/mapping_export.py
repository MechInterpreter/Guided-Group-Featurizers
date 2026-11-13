"""JSON-only recovery for undefined AP cells outside the frozen cohort.

Never changes cached numeric mappings, trained weights, calibration, or scoring.
Kept outside the embedded source ZIP to preserve the existing run identity.
"""
import json
from pathlib import Path
import numpy as np
from coco_baselines.storage import serial


def mapping_json_view(mapping, cohort):
    matrix = np.asarray(mapping['calibration_group_ap'], dtype=float)
    cohort = list(cohort)
    if matrix.ndim != 2 or matrix.shape[0] != mapping['groups']:
        raise ValueError('Calibration AP matrix shape mismatch')
    if len(set(cohort)) != len(cohort) or not cohort or min(cohort) < 0 or max(cohort) >= matrix.shape[1]:
        raise ValueError('Invalid frozen calibration category cohort')
    if not np.isfinite(matrix[:, cohort]).all():
        raise ValueError('Nonfinite calibration AP in the evaluated cohort; inspect data/model, do not hide it')
    if np.isinf(matrix).any():
        raise ValueError('Infinite calibration AP is not a missing-category placeholder')
    missing = np.isnan(matrix)
    view = serial(mapping)
    view['calibration_group_ap'] = np.where(missing, None, matrix).tolist()
    # Other fields must remain finite. No general NaN-to-zero/null suppression.
    json.dumps(view, allow_nan=False)
    index = {m['category_index']: m for m in mapping['mappings']}
    excluded = []
    for c in np.flatnonzero(missing.any(axis=0)):
        m = index.get(int(c), {})
        excluded.append(dict(category_index=int(c), category=m.get('category'),
                             undefined_group_AP_cells=int(missing[:, c].sum()),
                             calibration_positive_images=m.get('calibration_positive_images'),
                             calibration_negative_images=m.get('calibration_negative_images'),
                             outside_evaluated_cohort=True))
    audit = dict(undefined_cells=int(missing.sum()), categories=excluded,
                 evaluated_categories=len(cohort), evaluated_AP_cells_finite=True,
                 export_only=True, numeric_checkpoint_unchanged=True,
                 rule='Undefined AP cells outside the frozen cohort export as null, never zero. All evaluated columns and every other mapping field remain strictly finite.')
    return view, audit


def install_mapping_export_fix(ctx):
    store = ctx['store']
    frozen = store.load('all_models_frozen')
    if frozen is None or not frozen['complete']:
        raise RuntimeError('Models must be frozen before evaluation export recovery')
    original = getattr(store, '_mapping_original_write', store.write)
    store._mapping_original_write = original

    def write(name, value):
        if str(name).startswith('mappings/') and str(name).endswith('.json'):
            view, audit = mapping_json_view(value, ctx['cfg']['category_indices'])
            original('json_export_diagnostics/' + Path(name).name, audit)
            return original(name, view)
        return original(name, value)

    store.write = write
    repaired = []
    # The failing cell saved its numeric mapping before its JSON write failed.
    # Restore that text export directly from the verified checkpoint.
    for row in frozen['models']:
        tag = f"s{row['seed']}_{row['arm']}"
        mapping = store.load('mapping_' + tag)
        if mapping is not None:
            store.write('mappings/' + tag + '.json', mapping)
            repaired.append(tag)
    original('mapping_export_fix.json', dict(version=1, run_id=store.run_id,
        repaired_cached_mappings=repaired, retraining=False,
        calibration_recomputed=False, acceptance_changed=False,
        note='Only JSON representations are repaired. Original numeric mappings and frozen model checkpoints are retained.'))
    return dict(run_id=store.run_id, repaired_cached_mappings=repaired,
                training_repeated=False, scoring_changed=False)
