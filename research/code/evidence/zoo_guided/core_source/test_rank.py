import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from rank_core import Moments, block_statistics, load_moments, rank_statistics, save_moments, summarize
from rank_worker import analyze_model, catalog_rows, across_seeds


def test_known_spectra_and_distinct_stable_rank_definitions():
    a = rank_statistics([1, 1, 1, 1])
    assert a['effective_rank'] == pytest.approx(4)
    assert a['participation_ratio'] == pytest.approx(4)
    assert a['covariance_stable_rank'] == pytest.approx(4)
    b = rank_statistics([4, 1])
    assert b['covariance_stable_rank'] == pytest.approx(17 / 16)
    assert b['matrix_stable_rank'] == pytest.approx(5 / 4)
    assert b['participation_ratio'] == pytest.approx(25 / 17)
    assert rank_statistics([0, 0])['effective_rank'] is None
    assert rank_statistics([0, 0, 8])['effective_rank'] == pytest.approx(1)
    with pytest.raises(FloatingPointError): rank_statistics([-1, 2])


def test_conditional_covariance_excludes_inactive_patches():
    z = np.zeros((8, 3, 2))
    z[:4, 0] = [[1, 0], [-1, 0], [2, 0], [-2, 0]]
    z[0, 1] = [3, 1]
    m = Moments(3, 2); m.add_dense(z)
    rows, arrays = block_statistics(m, np.broadcast_to(np.eye(2), (3, 2, 2)), concept_count=1)
    assert m.n.tolist() == [4, 1, 0]
    assert arrays['covariance'][0, 0, 0] == pytest.approx(10 / 3)
    assert rows[0]['code_effective_rank'] == pytest.approx(1)
    assert rows[0]['activity_fraction'] == .5
    assert rows[1]['status'] == 'too_few_samples'
    assert rows[1]['code_effective_rank'] is None
    assert rows[2]['status'] == 'inactive'
    assert rows[2]['code_effective_rank'] is None


def test_isotropic_rank_constant_and_reliability_counts():
    z = np.zeros((40, 3, 3))
    z[:, 0] = np.tile(np.vstack([np.eye(3), -np.eye(3)]), (7, 1))[:40]
    z[:36, 0] = np.tile(np.vstack([np.eye(3), -np.eye(3)]), (6, 1)); z[36:, 0] = 0
    z[:, 1] = [1, 2, 3]
    m = Moments(3, 3); m.add_dense(z)
    records, _ = block_statistics(m, np.broadcast_to(np.eye(3), (3, 3, 3)), concept_count=2)
    assert records[0]['code_effective_rank'] == pytest.approx(3)
    assert records[0]['code_utilization'] == pytest.approx(1)
    assert records[1]['status'] == 'zero_variation'
    assert records[1]['code_effective_rank'] is None
    summaries = summarize(records)
    assert next(s for s in summaries if s['slot_family'] == 'all' and s['sample_subset'] == 'at_least_32_active_patches')['valid_groups'] == 1


def test_streaming_covariance_matches_direct_centering_with_large_offset():
    rng = np.random.default_rng(9)
    z = rng.normal(size=(97, 5, 4)) + 1e7
    z[rng.random((97, 5)) < .7] = 0
    m = Moments(5, 4)
    for part in np.array_split(z, 13): m.add_dense(part)
    rows, arrays = block_statistics(m, np.broadcast_to(np.eye(4), (5, 4, 4)), concept_count=2)
    for g in range(5):
        selected = z[:, g][np.any(z[:, g] != 0, axis=-1)]
        np.testing.assert_allclose(arrays['covariance'][g], np.cov(selected, rowvar=False), rtol=2e-8, atol=2e-8)
        assert rows[g]['active_patches'] == len(selected)


def test_contribution_spectrum_matches_full_activation_covariance():
    rng = np.random.default_rng(14)
    z = rng.normal(size=(100, 2, 3)); decoder = rng.normal(size=(2, 3, 11))
    m = Moments(2, 3); m.add_dense(z)
    rows, arrays = block_statistics(m, decoder, concept_count=1)
    for g in range(2):
        full = np.linalg.eigvalsh(np.cov(z[:, g] @ decoder[g], rowvar=False))[-3:]
        np.testing.assert_allclose(arrays['contribution_eigenvalues'][g], full, atol=1e-11)
        assert rows[g]['contribution_effective_rank'] == pytest.approx(rank_statistics(full)['effective_rank'])


def test_orthogonal_rotation_preserves_ranks():
    rng = np.random.default_rng(1); z = rng.normal(size=(91, 2, 4)); D = rng.normal(size=(2, 4, 7))
    q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    a = Moments(2, 4); a.add_dense(z)
    b = Moments(2, 4); b.add_dense(z @ q)
    ra, _ = block_statistics(a, D, concept_count=1)
    rb, _ = block_statistics(b, np.einsum('ij,gjd->gid', q.T, D), concept_count=1)
    for x, y in zip(ra, rb):
        assert x['code_effective_rank'] == pytest.approx(y['code_effective_rank'])
        assert x['contribution_effective_rank'] == pytest.approx(y['contribution_effective_rank'])


def test_two_slot_resume_falls_back_from_corrupt_latest(tmp_path):
    m = Moments(2, 3); m.add_dense(np.ones((3, 2, 3)))
    save_moments(tmp_path, 'same', m, 1)
    m.add_dense(np.ones((4, 2, 3))); save_moments(tmp_path, 'same', m, 2)
    (tmp_path / 'slot0.npz').write_bytes(b'partial interrupted write')
    restored, generation = load_moments(tmp_path, 'same', 2, 3)
    assert generation == 1 and restored.patches == 3
    with pytest.raises(ValueError): load_moments(tmp_path, 'different', 2, 3)


def test_native_gating_inference_and_interrupted_resume_without_weight_changes(tmp_path, monkeypatch):
    from bsf import VanillaBSF
    torch.manual_seed(7)
    model = VanillaBSF(d=6, n_groups=5, group_size=3, l0=2)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    x = np.random.default_rng(5).normal(size=(29, 6)).astype('float32')
    direct = analyze_model(model, x, tmp_path / 'direct', 'identity', 'cpu', batch=4, checkpoint_batches=2)
    original_add = Moments.add_selected; calls = []
    def interrupted(self, *args):
        original_add(self, *args); calls.append(1)
        if len(calls) == 3: raise KeyboardInterrupt()
    monkeypatch.setattr(Moments, 'add_selected', interrupted)
    with pytest.raises(KeyboardInterrupt): analyze_model(model, x, tmp_path / 'resumed', 'identity', 'cpu', batch=4, checkpoint_batches=2)
    monkeypatch.setattr(Moments, 'add_selected', original_add)
    resumed = analyze_model(model, x, tmp_path / 'resumed', 'identity', 'cpu', batch=4, checkpoint_batches=2)
    np.testing.assert_array_equal(direct.n, resumed.n)
    np.testing.assert_array_equal(direct.mean, resumed.mean)
    np.testing.assert_array_equal(direct.m2, resumed.m2)
    for k, v in model.state_dict().items(): assert torch.equal(v, original[k])
    with torch.inference_mode(): gated = model.encode(torch.as_tensor(x)).numpy()
    expected = Moments(5, 3); expected.add_dense(gated)
    np.testing.assert_array_equal(direct.n, expected.n)
    np.testing.assert_allclose(direct.mean, expected.mean, atol=2e-7)
    np.testing.assert_allclose(direct.m2, expected.m2, atol=2e-6)


def catalogs():
    rows = []
    for seed in [21, 22, 23]:
        for cfg, r, k in [('5d16', 5, 16), ('6d16', 6, 16), ('8d16', 8, 16), ('5d24', 5, 24)]:
            for kind in ['agf', 'vanilla']:
                rows.append(dict(seed=seed, arm=kind+'_'+cfg, rank=r, active_groups=k, checkpoint_updates=3000, total_training_updates=16640))
    old = [dict(seed=s, arm=a, rank=5, checkpoint_updates=3000, total_training_updates=13640) for s in [21, 22, 23] for a in ['guided5_continue', 'vanilla5_continue']]
    return dict(complete=True, models=rows), dict(complete=True, models=old)


def test_catalog_requires_all_30_frozen_configuration_identities():
    a, b = catalogs(); rows = catalog_rows(a, b)
    assert len(rows) == 30 and len({(r['seed'], r['arm']) for r in rows}) == 30
    a['models'][0]['active_groups'] = 32
    with pytest.raises(ValueError): catalog_rows(a, b)
    a, b = catalogs(); a['models'][-1] = a['models'][0]
    with pytest.raises(ValueError): catalog_rows(a, b)
    a, b = catalogs(); b['complete'] = False
    with pytest.raises(ValueError): catalog_rows(a, b)


def test_three_seed_means_keep_model_and_sample_subset_separate():
    summaries = []
    for seed in [21, 22, 23]:
        m = Moments(4, 3)
        m.add_dense(np.random.default_rng(seed).normal(size=(40, 4, 3)))
        rows, _ = block_statistics(m, np.broadcast_to(np.eye(3), (4, 3, 3)), concept_count=2)
        summaries.extend(dict(s, arm='example', seed=seed, rank=3, active_groups=2) for s in summarize(rows))
    result = across_seeds(summaries)
    assert len(result) == 6 and all(r['seeds'] == 3 for r in result)
    assert all(r['code_effective_rank_mean'] <= 3 for r in result)
    with pytest.raises(ValueError): across_seeds(summaries[:-1])


def test_saved_tables_and_figures_include_all_arms_and_both_slot_families(tmp_path):
    from rank_worker import csv_write, plots
    m = Moments(4, 6)
    m.add_dense(np.random.default_rng(11).normal(size=(40, 4, 6)))
    records, _ = block_statistics(m, np.broadcast_to(np.eye(6), (4, 6, 6)), concept_count=2)
    arms = ['guided5_continue', 'agf_5d16', 'agf_6d16', 'agf_8d16', 'agf_5d24',
            'vanilla5_continue', 'vanilla_5d16', 'vanilla_6d16', 'vanilla_8d16', 'vanilla_5d24']
    summaries = [dict(s, arm=arm, seed=seed, rank=6, active_groups=16) for arm in arms for seed in [21,22,23] for s in summarize(records)]
    means = across_seeds(summaries)
    csv_write(tmp_path/'means.csv', means)
    plots(tmp_path/'figures', summaries)
    assert len(means) == 60
    assert all((tmp_path/'figures'/name).stat().st_size > 1000 for name in ['initialized_slots.png', 'initialized_slots.pdf', 'reserve_slots.png', 'reserve_slots.pdf'])
    assert 'code_matrix_stable_rank_mean' in (tmp_path/'means.csv').read_text()


def test_disconnected_drive_guard_does_not_create_output(monkeypatch):
    from rank_core import atomic_json
    monkeypatch.setattr('rank_core.os.path.ismount', lambda p: False)
    with pytest.raises(OSError, match='Drive disconnected'):
        atomic_json('LOCAL_INPUTS/rank_test/progress.json', {'complete':False})
