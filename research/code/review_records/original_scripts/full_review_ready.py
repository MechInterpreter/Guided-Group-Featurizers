"""Separate, fail-closed handoff for the frozen three-seed all75 collection.

No annotation generation, label transfer, fitting, or protocol changes. Commands
use the original frozen scorer; only method-blinded public cards reach packets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import review_ready as pilot
from review_ready import read, sha, write, relative, copy_exact, SOURCE, CONTEXTS, CODES

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
SCOPE = 'three_seed_comparison'
LOCAL = 'local_full_review'
SEMANTIC = 'Specific shared object, identifiable part/action/relation or named material/region at BOTH marked patches. Generic color/edge/orientation or broad animal/furniture umbrella is not enough. State the shared meaning.'
VISUAL = 'Separately judge shared specific texture/form/color/material/geometric structure. Parts and cross-category geometry are valid visual concepts; do not use category labels as unit names.'
PUBLIC_HEADER = dict(schema='all75-blinded-pair-review-v1', semantic_rubric=SEMANTIC,
    visual_rubric=VISUAL, codes=['yes', 'no', 'unclear'],
    unavailable='Missing or zero-positive pairs must be unclear',
    annotation='Three fresh contexts, one independent file each; every case and three pairs retained. Use public cards only. No method/native-unit/category identities in this pack.')


def key(*xs):
    return hashlib.sha256(('all75-pair-v1|' + '|'.join(map(str, xs))).encode()).hexdigest()


def exact_hash(path, digest, message):
    if sha(path) != digest:
        raise ValueError(message)


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def source_pins():
    pins = read(HERE / 'FULL_REVIEW_SOURCE_PINS.json')['files']
    for name, digest in pins.items():
        exact_hash(relative(SOURCE, name), digest, 'Original frozen source identity changed: ' + name)
    return pins


def expected_matrix():
    expected = {(s, 'full_category', k, 0, 75, True) for s in (21, 22, 23)
                for k in ('ggf', 'slics', 'vanilla', 'supervised_vanilla')}
    expected |= {(s, 'category_split', k, f, 75, True) for s in (21, 22, 23)
                 for k in ('ggf', 'supervised_vanilla') for f in (0, 1)}
    expected |= {(s, 'category_split', 'vanilla', 0, 75, True) for s in (21, 22, 23)}
    expected |= {(21, 'category_split', 'slics', f, n, False) for f, n in ((0, 37), (1, 38))}
    return expected


def destination(root):
    root = Path(root).resolve()
    # A full-study import must never absorb or overwrite a pilot/semantic root.
    for protected in (SOURCE.parent, WORKSPACE / 'output/ggf_all75_semantic_review_20261027'):
        if root == protected.resolve() or root.is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(root):
            raise ValueError('Use a separate full-study root; preserve seed21 artifacts')
    if (root / 'COLLECTION_COMPLETE.json').exists() and read(root / 'COLLECTION_COMPLETE.json').get('scope') != SCOPE:
        raise ValueError('Use a separate full-study root; existing collection is not full-study')
    return root


def verify_review(root):
    root = destination(root)
    pins = source_pins()
    if read(root / 'PROTOCOL.json') != read(SOURCE / 'PROTOCOL.json'):
        raise ValueError('Different frozen metric protocol')
    collection = read(root / 'COLLECTION_COMPLETE.json')
    if (collection.get('complete') is not True or collection.get('scope') != SCOPE
            or collection.get('training_seeds') != [21, 22, 23]
            or collection.get('full_three_seed_study_complete') is not True):
        raise ValueError('Complete three-seed collection required')
    if collection.get('collector_sha256') != pins['collect_evaluation.py'] or collection.get('protocol_sha256') != pins['PROTOCOL.json']:
        raise ValueError('Collector/protocol source identity differs')
    exact_hash(root / 'INPUTS.json', collection['inputs_sha256'], 'Input manifest changed')
    inputs = read(root / 'INPUTS.json')
    if inputs['definitions_sha256'] != pins['TARGET_DEFINITIONS.json']:
        raise ValueError('Target definition identity changed')
    exact_hash(root / 'TARGET_DEFINITIONS.json', inputs['definitions_sha256'], 'Target definitions changed')
    if inputs.get('categories') != [r['category_index'] for r in read(root / 'TARGET_DEFINITIONS.json')['categories']]:
        raise ValueError('Fixed 75-category input cohort differs')
    banks = {b['bank_id']: b for b in collection['banks']}
    actual = [(b['seed'], b['regime'], b['kind'], b['fold'], b['groups'], b['capacity_matched']) for b in collection['banks']]
    if (len(banks) != 29 or len(actual) != 29 or len(set(actual)) != 29
            or set(actual) != expected_matrix()
            or any(type(b['capacity_matched']) is not bool or any(type(b[k]) is not int for k in ('seed', 'fold', 'groups')) for b in banks.values())):
        raise ValueError('Missing, duplicate, or capacity-mismatched full native bank matrix')
    endpoint = read(root / 'ENDPOINT_LOCK.json')
    if endpoint['scope'] != SCOPE or endpoint['protocol_sha256'] != pins['PROTOCOL.json'] or endpoint['normalization_identity'] != inputs['normalization_identity']:
        raise ValueError('Endpoint protocol/normalization identity differs')
    endpoints = {b['bank_id']: b for b in endpoint['banks']}
    if len(endpoints) != len(endpoint['banks']) or set(endpoints) != set(banks):
        raise ValueError('Endpoint bank identities differ')
    if set(endpoint['training_receipt_sha256']) != {'full_category', 'category_split'} or any(not is_sha256(v) for v in endpoint['training_receipt_sha256'].values()):
        raise ValueError('Both frozen training receipt identities required')
    portable = read(root / 'PORTABLE_LOCK.json')
    if portable.get('complete') is not True or portable['collection_sha256'] != sha(root / 'COLLECTION_COMPLETE.json'):
        raise ValueError('Portable collection source identity differs')
    views_by_bank = {}
    refs = {r['bank_id']: r for r in portable['banks']}
    if len(refs) != len(portable['banks']) or set(refs) != set(banks):
        raise ValueError('Portable bank identities differ')
    for bank_id, bank in banks.items():
        row = endpoints[bank_id]
        # The frozen collector constructs bank=dict(row, ..., observed_peak_GPU_bytes=...).
        # Endpoint telemetry describes training/source loading; completed-bank telemetry
        # describes response collection. It is deliberately replaced, not checkpoint identity.
        if any(bank.get(k) != v for k, v in row.items() if k != 'observed_peak_GPU_bytes'):
            raise ValueError('Completed bank differs from frozen endpoint source')
        for record in (row, bank):
            peak = record.get('observed_peak_GPU_bytes')
            if peak is not None and (type(peak) is not int or peak < 0):
                raise ValueError('Invalid phase-specific GPU peak telemetry')
        if bank_id != f"{bank['regime']}_{bank['kind']}_s{bank['seed']}_f{bank['fold']}":
            raise ValueError('Native bank source identity differs')
        folds = [0, 1] if bank['regime'] == 'category_split' and bank['kind'] == 'vanilla' else [bank['fold']]
        if bank.get('evaluation_folds') != folds or bank.get('complete') is not True or not is_sha256(bank.get('signature')) or not is_sha256(bank.get('weights_sha')):
            raise ValueError('Complete checkpoint/fold identity required')
        p = root / 'portable' / bank_id / 'VIEWS.json'
        exact_hash(p, refs[bank_id]['views_sha256'], 'Portable view coordinates changed')
        pooled = p.with_name('POOLED.npz')
        exact_hash(pooled, refs[bank_id]['pooled_sha256'], 'Portable response source changed')
        pooled_meta = read(pooled.with_suffix('.json'))
        if pooled_meta['sha256'] != refs[bank_id]['pooled_sha256'] or pooled_meta['signature'] != bank['signature']:
            raise ValueError('Portable response signature differs')
        view = read(p)
        if view['signature'] != bank['signature'] or view['bank_id'] != bank_id:
            raise ValueError('Portable view source identity differs')
        if set(view['roles']) != {'calibration', 'audit'}:
            raise ValueError('Fixed source view roles required')
        for role, units in view['roles'].items():
            if set(units) != {str(u) for u in range(bank['groups'])}:
                raise ValueError('Full source unit coverage required')
            for selected in units.values():
                if len(selected) > 6 or len({v['image_id'] for v in selected}) != len(selected):
                    raise ValueError('Invalid fixed source view allocation')
                for v in selected:
                    if v['image_id'] not in inputs['ids'][role] or len(v['patches']) > 4 or len(set(v['patches'])) != len(v['patches']) or any(type(p) is not int or not 0 <= p < 196 for p in v['patches']):
                        raise ValueError('Source image/patch geometry differs')
        views_by_bank[bank_id] = view
    image_assets = read(root / 'IMAGE_ASSETS.json')
    if image_assets['source_inputs_sha256'] != collection['inputs_sha256']:
        raise ValueError('Crop assets belong to different input source')
    assets = {r['image_id']: r for r in image_assets['images']}
    all_ids = inputs['ids']['calibration'] + inputs['ids']['audit']
    if len(set(all_ids)) != len(all_ids) or set(assets) != set(all_ids) or len(assets) != len(image_assets['images']):
        raise ValueError('Missing or duplicate source image identities')
    for i, asset in assets.items():
        original = inputs['records'][str(i)]
        if not is_sha256(asset.get('original_sha256')) or any(asset.get(k) != original[k] for k in ('image_id', 'role', 'geometry', 'original_sha256')):
            raise ValueError('Source crop geometry/hash identity differs')
        if asset['status'] == 'ready':
            exact_hash(relative(root, asset['crop_path']), asset['crop_sha256'], 'Source crop changed')
        elif asset['status'] != 'missing':
            raise ValueError('Unknown crop availability status')
    folder = root / 'reviews/coherence'
    lock = read(folder / 'REVIEW_LOCK.json')
    for name, field in (('PUBLIC.json', 'public_sha256'), ('PRIVATE.json', 'private_sha256')):
        exact_hash(folder / name, lock[field], 'Frozen public/private mapping changed')
    public, private = read(folder / 'PUBLIC.json'), read(folder / 'PRIVATE.json')
    if set(public) != set(PUBLIC_HEADER) | {'cases'} or any(public[k] != v for k, v in PUBLIC_HEADER.items()):
        raise ValueError('Public rubric/schema differs or contains private fields')
    if private['collection_sha256'] != sha(root / 'COLLECTION_COMPLETE.json'):
        raise ValueError('Review collection identity differs')
    pids = [r['case_id'] for r in public['cases']]
    qids = [r['case_id'] for r in private['cases']]
    if pids != [f'G{j:04d}' for j in range(1, 2101)] or qids != pids:
        raise ValueError('Full fixed 2100-case coverage/order required')
    expected_cases = []
    for bank in banks.values():
        for unit in range(bank['groups']):
            opaque = key('group', bank['bank_id'], bank['weights_sha'], unit)
            expected_cases.append((opaque, bank['bank_id'], unit))
    expected_cases.sort(key=lambda r: key('display', r[0]))
    card_names = set()
    strata = {}
    for item, case, (opaque, bank_id, unit) in zip(public['cases'], private['cases'], expected_cases):
        if set(item) != {'case_id', 'card', 'pair_ids', 'invalid_pairs'}:
            raise ValueError('Public case contains private or unexpected fields')
        if case['bank_id'] != bank_id or type(case['unit']) is not int or case['unit'] != unit:
            raise ValueError('Duplicate, missing, or remapped native unit identity')
        slots = sorted(range(6), key=lambda j: key(opaque, j))
        selected = views_by_bank[bank_id]['roles']['audit'][str(unit)]
        views = [selected[j] if j < len(selected) else {} for j in slots]
        image_ids = [v.get('image_id') for v in views]
        bad = [not (assets.get(v.get('image_id'), {}).get('status') == 'ready' and v.get('patches')) for v in views]
        invalid = [any(bad[2*k:2*k+2]) for k in range(3)]
        if case['source_slots'] != slots or case['image_ids'] != image_ids or case['invalid_pairs'] != invalid:
            raise ValueError('Fixed pair allocation/source geometry differs')
        if item['pair_ids'] != [0, 1, 2] or item['invalid_pairs'] != invalid or any(type(v) is not bool for v in item['invalid_pairs']):
            raise ValueError('Frozen pair coverage differs')
        name = f"cards/{case['case_id']}.png"
        if item['card'] != name or name not in lock['cards']:
            raise ValueError('Public card identity differs')
        card_names.add(name)
        exact_hash(relative(folder, name), lock['cards'][name], 'Review card changed')
        bank = banks[bank_id]
        strata.setdefault((bank['regime'], bank['kind'], bank['capacity_matched']), []).append(case)
    if set(lock['cards']) != card_names:
        raise ValueError('Unexpected or missing locked cards')
    human = read(folder / 'HUMAN_VALIDATION_SAMPLE.json')
    selected = []
    for stratum, cases in sorted(strata.items()):
        selected.extend(c['case_id'] for c in sorted(cases, key=lambda c: key('human-validation', c['case_id']))[:4 if stratum[2] else 2])
    if human['case_ids'] != selected or len(selected) > 30 or len(set(selected)) != len(selected):
        raise ValueError('Human sample differs from frozen pre-outcome selection')
    return collection, public, private, lock, human


def import_archives(root, archives):
    root = destination(root)
    # Validate returned artifacts before committing anything to the real root.
    scratch = WORKSPACE / 'tmp'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='all75_full_import_', dir=scratch) as staging:
        stage = Path(staging)
        count = pilot.import_archives(stage, archives)
        verify_review(stage)
        files = [p for p in stage.rglob('*') if p.is_file() and 'local_review' not in p.relative_to(stage).parts]
        for path in files:
            target = relative(root, path.relative_to(stage).as_posix())
            if target.exists() and sha(target) != sha(path):
                raise ValueError('Existing full-study artifact differs')
        for path in files:
            copy_exact(path, relative(root, path.relative_to(stage).as_posix()))
        write(root / LOCAL / 'RETURN_BUNDLES.json', read(stage / 'local_review/RETURN_BUNDLES.json'))
    return dict(files=count, scope=SCOPE, validated=True)


def validate_batch(data, packet, public, lock):
    if data.get('public_sha256') != lock['public_sha256'] or data.get('method_blind') is not True or not isinstance(data.get('reviewer'), str) or not data['reviewer'].strip():
        raise ValueError('Actual reviewer ID and locked public source required')
    cases = {c['case_id']: c for c in data['cases']}
    if len(cases) != len(data['cases']) or set(cases) != {c['case_id'] for c in packet['cases']}:
        raise ValueError('Batch must exactly cover its fixed cases')
    by_id = {c['case_id']: c for c in public['cases']}
    for cid, case in cases.items():
        if sorted(p['pair'] for p in case['pairs']) != [0, 1, 2] or any(type(p['pair']) is not int for p in case['pairs']):
            raise ValueError('All three original pairs required')
        for pair in case['pairs']:
            for endpoint in ('semantic', 'visual'):
                if pair.get(endpoint) not in CODES:
                    raise ValueError('Unfilled or invalid research judgment')
                if by_id[cid]['invalid_pairs'][pair['pair']] and pair[endpoint] != 'unclear':
                    raise ValueError('Missing evidence must remain unclear')
                if pair[endpoint] == 'yes' and not pair.get('shared_meaning', '').strip():
                    raise ValueError('Positive judgments require concrete meaning')


def packet_for(public, start):
    return dict(cases=public['cases'][start:start+12])


def status(root):
    _, public, _, lock, human = verify_review(root)
    root = Path(root)
    out = dict(scope=SCOPE, label_status='pending', planned_groups=2100, planned_pairs=6300,
               human_groups=len(human['case_ids']), context_cases={})
    prior_reviewers = set()
    raw_root = root / 'reviews/coherence/batch_labels'
    if raw_root.exists() and any(p.is_dir() and p.name not in CONTEXTS for p in raw_root.iterdir()):
        raise ValueError('Unknown raw review stream')
    for context in CONTEXTS:
        ids, reviewers = [], set()
        expected_batches = {f'B{j//12:03d}': j for j in range(0, 2100, 12)}
        for p in sorted((root / 'reviews/coherence/batch_labels' / context).glob('*.json')):
            if p.stem not in expected_batches:
                raise ValueError('Unknown raw review batch')
            data = read(p)
            validate_batch(data, packet_for(public, expected_batches[p.stem]), public, lock)
            ids.extend(c['case_id'] for c in data['cases'])
            reviewers.add(data['reviewer'])
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate raw review cases')
        if reviewers & prior_reviewers:
            raise ValueError('Fresh reviewer contexts cannot be reused across streams')
        prior_reviewers.update(reviewers)
        out['context_cases'][context] = len(ids)
    out['complete_raw_coverage'] = all(n == 2100 for n in out['context_cases'].values())
    return out


def accept_batch(root, context, batch, labels):
    if context not in CONTEXTS or batch not in {f'B{j:03d}' for j in range(175)}:
        raise ValueError('Unknown review context or batch')
    _, public, _, lock, _ = verify_review(root)
    data = read(labels)
    validate_batch(data, packet_for(public, int(batch[1:])*12), public, lock)
    status(root)  # Revalidate all accepted raw files, including reviewer identities.
    for other in CONTEXTS:
        if other != context:
            for p in (Path(root) / 'reviews/coherence/batch_labels' / other).glob('*.json'):
                if read(p)['reviewer'] == data['reviewer']:
                    raise ValueError('Fresh reviewer contexts cannot be reused across streams')
    target = Path(root) / 'reviews/coherence/batch_labels' / context / f'{batch}.json'
    copy_exact(labels, target)
    return dict(context=context, batch=batch, cases=len(data['cases']), source_sha256=sha(target))


def score_coherence(root):
    root = destination(root)
    if not status(root)['complete_raw_coverage']:
        raise ValueError('All three complete blinded streams required')
    # Execute only the locally pinned original code, never returned archive code.
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SOURCE), str(SOURCE / 'core'), os.environ.get('PYTHONPATH', '')]))
    for context in CONTEXTS:
        subprocess.run([sys.executable, str(SOURCE / 'prepare_evaluation_reviews.py'), '--root', str(root), '--phase', 'assemble-' + context], env=environment, check=True)
    subprocess.run([sys.executable, str(SOURCE / 'score_evaluation.py'), '--root', str(root), '--phase', 'coherence'], env=environment, check=True)
    subprocess.run([sys.executable, str(SOURCE / 'report_evaluation.py'), '--root', str(root)], env=environment, check=True)
    result = read(root / 'COHERENCE_COMPLETE.json')
    exact_hash(root / 'reviews/coherence/PUBLIC.json', result['public_sha256'], 'Scored public source changed')
    if result.get('complete') is not True or set(result['bank_results']) != {b['bank_id'] for b in read(root / 'COLLECTION_COMPLETE.json')['banks']}:
        raise ValueError('Complete fixed bank readout required')
    receipt = dict(scope=SCOPE, complete_coherence=True, all_other_metrics_complete=read(root / 'COMPARISON_RESULTS.json')['complete'],
                   groups=2100, pairs=6300, raw_pair_assessments=18900,
                   coherence_sha256=sha(root / 'COHERENCE_COMPLETE.json'), scorer_sha256=sha(SOURCE / 'score_evaluation.py'))
    write(root / LOCAL / 'COHERENCE_SCORED.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--phase', required=True, choices=['import', 'packets', 'status', 'accept', 'score-coherence'])
    p.add_argument('--archives', nargs='+')
    p.add_argument('--context', choices=CONTEXTS)
    p.add_argument('--batch')
    p.add_argument('--labels')
    a = p.parse_args()
    if a.phase == 'import':
        result = import_archives(a.root, a.archives or [])
    elif a.phase == 'packets':
        result = make_packets(a.root)
    elif a.phase == 'status':
        result = status(a.root)
    elif a.phase == 'accept':
        result = accept_batch(a.root, a.context, a.batch, a.labels)
    else:
        result = score_coherence(a.root)
    print(json.dumps(result, indent=2))



def make_packets(root):
    root = Path(root)
    collection, public, private, lock, human = verify_review(root)
    # The reviewer directory has no registry, PRIVATE mapping, AP scores, prior
    # interpretations, category names, or other contexts' annotations.
    out = root / 'local_full_review/reviewer_public'
    source = root / 'reviews/coherence'
    allowed = {'PUBLIC.json', 'QUEUE.json', 'HUMAN_CHAT_QUEUE.json'}
    allowed |= {case['card'] for case in public['cases']}
    allowed |= {f'packets/{context}/B{j//12:03d}.json' for context in CONTEXTS for j in range(0, len(public['cases']), 12)}
    if out.exists() and any(p.is_file() and p.relative_to(out).as_posix() not in allowed for p in out.rglob('*')):
        raise ValueError('Unexpected file in public reviewer directory; preserve the public-only boundary')
    copy_exact(source / 'PUBLIC.json', out / 'PUBLIC.json')
    for case in public['cases']:
        copy_exact(relative(source, case['card']), relative(out, case['card']))
    instructions = (
        'Read only this public packet and its PNG cards in a fresh review context. '
        'Do not access the study registry, private mapping, previous interpretations, '
        'method identities, target labels, numerical scores, or another context\'s judgments. '
        'Inspect every numbered image at sufficient resolution, including the zoom strips. '
        'Only the marked patches are evidence; an object elsewhere in a crop is insufficient. '
        'For each fixed pair (views 1â€“2, 3â€“4, 5â€“6), return two separate codes: yes/no/unclear. '
        'SEMANTIC: ' + public['semantic_rubric'] + ' VISUAL: ' + public['visual_rubric'] + ' '
        'An interpretable fold, texture, contour, or cross-category geometric motif can be '
        'visual=yes even if semantic=no or unclear. Do not dismiss a part because the whole '
        'object has another category. Do not stretch a vague theme to accommodate exceptions. '
        'Missing planned images or zero-positive pairs must remain unclear for both endpoints. '
        'Write the concrete shared meaning and any exceptions for each pair. Positive codes '
        'require a stated meaning. Do not fill any judgment until you inspect its card. '
        'Return the exact JSON template with your actual fresh-context reviewer ID. '
        'AI contexts are review replicates, not independent human raters.'
    )
    queue = []
    for context in CONTEXTS:
        for start in range(0, len(public['cases']), 12):
            batch = f'B{start // 12:03d}'
            cases = public['cases'][start:start + 12]
            template = dict(public_sha256=lock['public_sha256'], method_blind=True, reviewer='', cases=[
                dict(case_id=r['case_id'], pairs=[dict(pair=k, semantic='', visual='', shared_meaning='', exceptions='') for k in range(3)]) for r in cases])
            packet = dict(context=context, batch=batch, instructions=instructions, cases=cases, output_template=template,
                          card_root='../../', public_sha256=lock['public_sha256'])
            write(out / 'packets' / context / f'{batch}.json', packet)
            queue.append(dict(context=context, batch=batch, packet=f'packets/{context}/{batch}.json', cases=len(cases)))
    write(out / 'QUEUE.json', dict(batches=queue, instruction='One fresh context per packet; retain actual reviewer IDs and all raw outcomes.'))
    write(out / 'HUMAN_CHAT_QUEUE.json', dict(cases=[dict(chat_id=f'H{j:02d}', **next(r for r in public['cases'] if r['case_id'] == cid)) for j, cid in enumerate(human['case_ids'], 1)],
        instruction='Fixed pre-outcome validation sample, at most30 groups; use the same two pair rubrics. Show these cards in chat. Human labels remain separate; no change to AI majority.'))
    counts = dict(total_groups=len(public['cases']), total_pairs=3 * len(public['cases']),
                  main_matched_groups=sum(b['groups'] for b in collection['banks'] if b['capacity_matched']),
                  reference_groups=sum(b['groups'] for b in collection['banks'] if not b['capacity_matched']),
                  ai_contexts=3, batches=len(queue), human_groups=len(human['case_ids']))
    write(root / 'local_full_review/PACKETS_READY.json', dict(complete=True, counts=counts, public_sha256=lock['public_sha256'],
        review_lock_sha256=sha(source / 'REVIEW_LOCK.json'), label_status='pending_actual_blinded_review', training_unchanged=True))
    return counts


if __name__ == "__main__":
    main()
