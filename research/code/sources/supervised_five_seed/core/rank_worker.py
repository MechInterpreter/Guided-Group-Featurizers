"""Analyze 30 completed capacity/reference models without training or test inference."""
import argparse
import csv
import gc
import json
import os
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from rank_core import Moments, atomic_json, block_statistics, fingerprint, guard_path, load_moments, save_moments, sha, summarize

CAPACITY_RUN = '2921b057f824362ab83e76e7fb35645f94c376fadf8b86f4f22d1a2311105b9e'
CAPACITY_SOURCE = '6f9138f8885fdbb2d5971f30ab0c5d744d4c8fa914e1c191491c895d5ca8cd79'
PARENT_RUN = '371d2d1d81ea6c9ce15fa17a6cdecc716f42304e385ac0cf9d2f3280572d6a0f'
ARMS = {'agf_5d16', 'vanilla_5d16', 'agf_6d16', 'vanilla_6d16', 'agf_8d16', 'vanilla_8d16', 'agf_5d24', 'vanilla_5d24'}
SOURCE_ARMS = {'guided5_continue', 'vanilla5_continue'}


def catalog_rows(capacity, parent):
    if not capacity['complete'] or len(capacity['models']) != 24:
        raise ValueError('All 24 capacity endpoints must have completed')
    if {(r['seed'], r['arm']) for r in capacity['models']} != {(s, a) for s in [21, 22, 23] for a in ARMS}:
        raise ValueError('Incomplete or duplicate capacity catalog')
    roots = [r for r in parent['models'] if r['arm'] in SOURCE_ARMS]
    if not parent['complete'] or len(roots) != 6 or {(r['seed'], r['arm']) for r in roots} != {(s, a) for s in [21, 22, 23] for a in SOURCE_ARMS}:
        raise ValueError('Incomplete or duplicate source catalog')
    rows = []
    for row in capacity['models']:
        r = dict(row, input_store='capacity')
        expected = {'6d16': (6, 16), '8d16': (8, 16), '5d24': (5, 24), '5d16': (5, 16)}[r['arm'].split('_', 1)[1]]
        if (r['rank'], r['active_groups']) != expected or r['checkpoint_updates'] != 3000 or r['total_training_updates'] != 16640:
            raise ValueError('Unexpected capacity model configuration')
        rows.append(r)
    for row in roots:
        if row['rank'] != 5 or row['checkpoint_updates'] != 3000 or row['total_training_updates'] != 13640:
            raise ValueError('Unexpected saved reference configuration')
        rows.append(dict(row, input_store='parent', active_groups=16))
    return rows


def csv_write(path, rows):
    path = Path(path); guard_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    temp = path.with_suffix('.pending.csv')
    with temp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    os.replace(temp, path)


def analyze_model(model, x, output, signature, device, batch=2048, checkpoint_batches=32):
    """No decoder prediction, gradients, noise or optimizer operations."""
    groups, width = model.n_groups, model.group_size
    moments, generation = load_moments(output, signature, groups, width)
    if moments.patches > len(x) or moments.patches % batch and moments.patches != len(x):
        raise ValueError('Saved analysis offset is not a complete batch')
    model.eval(); started = time.monotonic(); steps = 0
    print('Rank inference resume:', moments.patches, '/', len(x), flush=True)
    try:
        with torch.inference_mode():
            for lo in range(moments.patches, len(x), batch):
                xb = torch.as_tensor(np.array(x[lo:lo + batch], copy=True), dtype=torch.float32, device=device)
                z = model.encode(xb)
                if not torch.isfinite(z).all(): raise FloatingPointError('Nonfinite coefficients')
                on = z.ne(0).any(-1)
                if int(on.sum(-1).max()) > model.l0: raise ValueError('Gated code exceeds Top-k')
                selected = on.nonzero(as_tuple=False)
                # Transfer only selected coefficients, not the large dense tensor.
                group_ids = selected[:, 1].cpu().numpy()
                values = z[on].cpu().numpy()
                # Only complete accumulation batches may enter a saved checkpoint.
                before = (moments.n.copy(), moments.mean.copy(), moments.m2.copy(), moments.patches)
                try:
                    moments.add_selected(group_ids, values, len(xb))
                except BaseException:
                    moments.n, moments.mean, moments.m2, moments.patches = before
                    raise
                del before
                steps += 1
                if steps % checkpoint_batches == 0 or moments.patches == len(x):
                    generation += 1; save_moments(output, signature, moments, generation)
                    print('Rank inference:', moments.patches, '/', len(x), 'elapsed_seconds', round(time.monotonic() - started, 1), flush=True)
    except BaseException:
        generation += 1; save_moments(output, signature, moments, generation)
        raise
    return moments


def across_seeds(rows):
    result = []
    for arm in sorted({r['arm'] for r in rows}):
        for family in ['all', 'initialized_slots', 'reserve_slots']:
            for subset in ['all_estimable', 'at_least_32_active_patches']:
                rr = [r for r in rows if r['arm'] == arm and r['slot_family'] == family and r['sample_subset'] == subset]
                if len(rr) != 3: raise ValueError('Need all three seed summaries')
                out = dict(arm=arm, slot_family=family, sample_subset=subset, seeds=3,
                           allocated_width=rr[0]['rank'], active_groups=rr[0]['active_groups'])
                for key in ['code_effective_rank_mean', 'code_utilization_mean', 'contribution_effective_rank_mean',
                            'contribution_utilization_mean', 'code_participation_ratio_mean',
                            'code_covariance_stable_rank_mean', 'code_matrix_stable_rank_mean',
                            'added_coordinate_energy_fraction_mean', 'valid_groups', 'inactive_groups', 'median_active_patches']:
                    values = [r[key] for r in rr if r[key] is not None]
                    out[key] = float(np.mean(values)) if values else None
                    out[key + '_seed_min'] = float(np.min(values)) if values else None
                    out[key + '_seed_max'] = float(np.max(values)) if values else None
                result.append(out)
    return result


def plots(folder, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = ['guided5_continue', 'agf_5d16', 'agf_6d16', 'agf_8d16', 'agf_5d24']
    controls = ['vanilla5_continue', 'vanilla_5d16', 'vanilla_6d16', 'vanilla_8d16', 'vanilla_5d24']
    labels = ['5D saved', '5D continued', '6D', '8D', '5D / top-24']
    for family in ['initialized_slots', 'reserve_slots']:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
        for ax, metric, title in zip(axes, ['code_effective_rank_mean', 'contribution_effective_rank_mean', 'code_utilization_mean'],
                                     ['Coefficient effective rank', 'Contribution effective rank', 'Coefficient utilization']):
            for arms, color, name, offset in [(order, '#007f83', 'AGF', -.12), (controls, '#6b5794', 'Vanilla BSF', .12)]:
                for i, arm in enumerate(arms):
                    rr = [r for r in summaries if r['arm'] == arm and r['slot_family'] == family and r['sample_subset'] == 'at_least_32_active_patches']
                    values = [r[metric] for r in rr if r[metric] is not None]
                    if values:
                        ax.scatter(np.full(len(values), i + offset), values, color=color, s=25, alpha=.65)
                        ax.scatter(i + offset, np.mean(values), color=color, s=65, marker='_', linewidths=2.4, label=name if i == 0 else None)
            ax.set_xticks(range(5), labels, rotation=25, ha='right'); ax.set_title(title, fontsize=10)
            ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.15); ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(('First 75 slots (semantic initialization applies only to AGF)' if family == 'initialized_slots' else '949 reserve slots') + '\nTRAIN-only; active-block conditional covariance; dots are seeds', fontsize=10)
        folder.mkdir(exist_ok=True)
        fig.savefig(folder / (family + '.png'), dpi=220); fig.savefig(folder / (family + '.pdf')); plt.close(fig)


def run(options):
    from coco_ggf import matched_baselines as matched
    from coco_coordinated import run as memory
    from coco_baselines import training
    local = Path(options['rank_local']); local.mkdir(parents=True, exist_ok=True)
    if matched.versions() != options['expected_versions']: raise RuntimeError('Numerical versions differ')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if options.get('require_t4', True) and (device != 'cuda' or 'T4' not in torch.cuda.get_device_name()):
        raise RuntimeError('Select a Tesla T4 runtime')
    torch.set_num_threads(2)
    from threadpoolctl import threadpool_limits
    threadpool_limits(2); torch.use_deterministic_algorithms(True, warn_only=True)
    capacity = matched.discover(options['capacity_output'], local / 'read_capacity', run=CAPACITY_RUN)
    if capacity.config['source_sha'] != CAPACITY_SOURCE or capacity.config['parent_run'] != PARENT_RUN:
        raise ValueError('Capacity source identity differs')
    completed = memory.required(capacity, 'capacity_result')
    if not completed['complete'] or len(completed['models']) != 30: raise ValueError('Finish test evaluation first')
    del completed
    parent = matched.discover(options['paths']['continued5'], local / 'read_parent', run=PARENT_RUN)
    if parent.config['source_sha'] != capacity.config['parent_source_sha']: raise ValueError('Parent source differs')
    original = matched.discover(options['paths']['original'], local / 'read_original', run=capacity.config['activation_run'])
    frozen = memory.required(capacity, 'capacity_frozen'); old_frozen = memory.required(parent, 'frozen_continuation5')
    rows = catalog_rows(frozen, old_frozen)
    provenance = json.loads((original.results / 'provenance.json').read_text())
    norm = memory.required(original, 'normalization')
    from coco_baselines.storage import identity
    if identity(norm) != capacity.config['normalization_sha']: raise ValueError('Normalization identity differs')
    if len(provenance['splits']['train']) != 2000 or len(set(provenance['splits']['train'])) != 2000:
        raise ValueError('Expected the same 2,000-image TRAIN cohort')
    if len(provenance['categories']) < 75: raise ValueError('Category provenance is missing')
    protocol = dict(version=1, task='conditional_block_effective_rank', analysis_source_sha=options['rank_source_sha'],
                    capacity_run=CAPACITY_RUN, parent_run=PARENT_RUN, numerical_versions=matched.versions(),
                    models=[dict(seed=r['seed'], arm=r['arm'], weights_sha=r['weights_sha'], rank=r['rank'], active_groups=r['active_groups']) for r in rows],
                    train_ids=provenance['splits']['train'], normalization_sha=identity(norm), patch_batch=2048,
                    covariance='Centered unbiased coefficient covariance, conditional on selected block having nonzero gated code',
                    stable_rank='Covariance: sum(lambda^2)/max(lambda)^2; centered code matrix: sum(lambda)/max(lambda)',
                    effective_rank='exp(-sum(p log(p))), p=lambda/sum(lambda)',
                    participation_ratio='sum(lambda)^2/sum(lambda^2)', reliable_sample_threshold=32,
                    contribution_geometry='Eigenvalues of sqrt(D_g D_g.T) Cov(z_g) sqrt(D_g D_g.T)',
                    new_gradient_updates=0, test_features_opened=False, new_readouts_fitted=False,
                    note='Empirical dimensions of active-code variation, not latent-factor ground truth or MDL. Activity-conditioned populations can differ across models. Vanilla first-75 summaries are slot-based.')
    run_id = fingerprint(protocol); output = Path(options['rank_output']) / 'results' / run_id[:20]
    # Output cannot be inside any input or contain an input root.
    target = Path(options['rank_output']).resolve()
    for source in [capacity.output, parent.output, original.output]:
        p = source.resolve()
        if target == p or target in p.parents or p in target.parents: raise ValueError('Rank output must be separate from inputs')
    guard_path(output); output.mkdir(parents=True, exist_ok=True); atomic_json(output / 'protocol.json', protocol)
    snapshots = memory.source_snapshot([capacity, parent, original])
    pack = memory.normalized_pack(original, 'train', provenance, norm, Path(options['cache_local']), original.config['config']['extract_chunk'])
    x = pack['x'].reshape(-1, 768)
    if tuple(pack['x'].shape) != (2000, 196, 768) or pack['ids'] != provenance['splits']['train']:
        raise ValueError('TRAIN shape or image order differs')
    # No selection, confirmation or test activation shard is loaded here.
    atomic_json(output / 'cohort.json', dict(split='train', images=2000, patches=len(x), ids=pack['ids'], identity=pack['identity'], raw_float32_sha=pack['raw_float32_sha']))
    all_blocks = []; all_summaries = []; finished = []
    stores = {'capacity': capacity, 'parent': parent}
    for i, row in enumerate(rows):
        name = row['arm'] + '_s' + str(row['seed']); folder = output / 'models' / name
        guard_path(folder); folder.mkdir(parents=True, exist_ok=True)
        model_signature = fingerprint(dict(protocol=run_id, train=pack['identity'], model=row['weights_sha'], rank=row['rank'], k=row['active_groups']))
        print(f'Rank model {i+1}/30:', name, flush=True)
        # Loading is through FrozenStore only. No source .save/.write is possible.
        state = memory.required(stores[row['input_store']], row['key']); matched.complete(state, 3000)
        if training.weight_sha(state['weights']) != row['weights_sha']: raise ValueError('Checkpoint weight hash differs')
        cfg = dict(groups=1024, block_rank=row['rank'], active_groups=row['active_groups'])
        matched.valid_weights(state['weights'], 'vanilla', cfg)
        decoder = state['weights']['W_dec'].reshape(1024, row['rank'], 768).double().numpy().copy()
        model = training.make_model('vanilla', cfg).to(device)
        model.load_state_dict(state['weights'], strict=True); del state; gc.collect()
        meta = dict(arm=row['arm'], seed=row['seed'], rank=row['rank'], active_groups=row['active_groups'],
                    weights_sha=row['weights_sha'], scheduled_updates=row['total_training_updates'],
                    semantic_initialization=row['arm'].startswith('agf_') or row['arm'] == 'guided5_continue')
        done_path = folder / 'complete.json'
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if done['signature'] != model_signature: raise ValueError('Completed analysis fingerprint changed')
            for file, checksum in done['file_hashes'].items():
                if sha(folder / file) != checksum: raise OSError('Completed rank file checksum differs')
            records = json.loads((folder / 'blocks.json').read_text()); summaries = json.loads((folder / 'summary.json').read_text())
            print('Reused completed rank analysis', name, flush=True)
        else:
            moments = analyze_model(model, x, output / 'accumulators' / name, model_signature, device)
            records, arrays = block_statistics(moments, decoder)
            if len(row['targets']) != 75: raise ValueError('Expected 75 initialized category slots')
            for record in records:
                record.update(meta)
                record['initialized_category'] = row['targets'][record['group']]['category'] if meta['semantic_initialization'] and record['group'] < 75 else None
            summaries = [dict(s, **meta) for s in summarize(records)]
            atomic_json(folder / 'blocks.json', records); csv_write(folder / 'blocks.csv', records)
            atomic_json(folder / 'summary.json', summaries)
            with (folder / 'spectra.npz').open('wb') as f: np.savez_compressed(f, **arrays)
            hashes = {p.name: sha(p) for p in folder.iterdir() if p.name != 'complete.json'}
            atomic_json(done_path, dict(signature=model_signature, complete=True, file_hashes=hashes, patches=len(x)))
        all_blocks.extend(records); all_summaries.extend(summaries); finished.append(meta)
        csv_write(output / 'model_summaries.csv', all_summaries)
        atomic_json(output / 'progress.json', dict(complete=False, completed_models=len(finished), total_models=30, latest=name))
        del model, decoder; gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
    csv_write(output / 'block_statistics.csv', all_blocks)
    means = across_seeds(all_summaries); csv_write(output / 'three_seed_means.csv', means)
    plots(output / 'figures', all_summaries)
    memory.assert_immutable(snapshots)
    atomic_json(output / 'summary.json', dict(complete=True, run_id=run_id, models=finished, three_seed_means=means,
                new_gradient_updates=0, test_features_opened=False, limitations=protocol['note']))
    atomic_json(output / 'progress.json', dict(complete=True, completed_models=30, total_models=30))
    archive = Path(options['rank_output']) / ('agf_effective_rank_' + run_id[:12] + '_results.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and 'accumulators' not in p.relative_to(output).parts:
                z.write(p, 'results/' + p.relative_to(output).as_posix())
        for name in ['rank_core.py', 'rank_worker.py', 'test_rank.py']:
            z.write(Path(__file__).parent / name, 'source/' + name)
    receipt = dict(complete=True, models=30, results=str(output), result_zip=str(archive), zip_sha256=sha(archive), new_gradient_updates=0)
    atomic_json(Path(options['rank_receipt']), receipt)
    print('EFFECTIVE RANK ANALYSIS COMPLETE:', archive, flush=True)
    print('First 75 slots, at least 32 active patches, TRAIN ONLY:', flush=True)
    for r in means:
        if r['slot_family'] == 'initialized_slots' and r['sample_subset'] == 'at_least_32_active_patches':
            print(json.dumps({k: r[k] for k in ['arm', 'allocated_width', 'code_effective_rank_mean', 'contribution_effective_rank_mean', 'code_utilization_mean', 'valid_groups']}, allow_nan=False), flush=True)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); args = p.parse_args()
    run(json.loads(Path(args.config).read_text()))
