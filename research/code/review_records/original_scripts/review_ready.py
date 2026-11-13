"""Local handoff for the frozen all75 semantic review; never edits training code.

Only public cards enter reviewer packets. Research labels must come from actual
image review; this module does not synthesize or infer annotations.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
SOURCE = WORKSPACE / 'output/ggf_all75_evaluation_20261027/source'
CONTEXTS = ('context1', 'context2', 'context3')
CODES = ('yes', 'no', 'unclear')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf8')
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Existing frozen artifact differs: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def safe_relative(name):
    # Archives are exported on Linux and may be returned via Windows tools.
    normalized = name.replace('\\', '/')
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(p in ('..', '') or ':' in p for p in path.parts):
        raise ValueError('Unsafe artifact path: ' + name)
    return path


def relative(root, name):
    path = Path(root).joinpath(*safe_relative(name).parts)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact escapes its root')
    return path


def copy_exact(source, destination):
    destination = Path(destination)
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError('Existing returned artifact differs: ' + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha(source) != sha(destination):
        raise ValueError('Copy verification failed')


def import_archives(root, archives):
    """Validate all entries/conflicts first; never overwrite a different artifact."""
    root = Path(root)
    if not archives:
        raise ValueError('Supply the actual returned research archives')
    entries = {}
    receipts = []
    for archive in map(Path, archives):
        receipts.append(dict(name=archive.name, sha256=sha(archive)))
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                path = safe_relative(item.filename)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError('Archive links are not permitted')
                if item.is_dir():
                    continue
                if path.suffix.lower() not in ('.json', '.png', '.npz', '.txt', '.html', '.csv'):
                    raise ValueError('Unexpected research artifact type')
                raw = z.read(item)
                digest = hashlib.sha256(raw).hexdigest()
                key = '/'.join(path.parts).casefold()
                if key in entries and (entries[key][2] != digest or entries[key][1] != item.filename):
                    raise ValueError('Conflicting or case-colliding archive entries')
                target = relative(root, str(path))
                if target.exists() and sha(target) != digest:
                    raise ValueError('Local artifact differs from returned archive')
                entries[key] = (archive, item.filename, digest, target)
    for archive, name, digest, target in entries.values():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            raw = z.read(name)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Archive changed during import')
        with target.open('xb') as stream:
            stream.write(raw)
    write(root / 'local_review/RETURN_BUNDLES.json', dict(archives=receipts, files=len(entries)))
    return len(entries)


def verify_review(root):
    root = Path(root)
    # The collector serializes the frozen object via eval_common.save, so the
    # returned JSON has expanded indentation rather than the source formatting.
    # Compare the complete parsed protocol; raw card/mapping hashes stay exact.
    if read(root / 'PROTOCOL.json') != read(SOURCE / 'PROTOCOL.json'):
        raise ValueError('The returned review uses a different frozen metric protocol')
    collection = read(root / 'COLLECTION_COMPLETE.json')
    if not collection.get('complete') or collection.get('scope') != 'seed21_pilot':
        raise ValueError('This local handoff requires the complete seed21 pilot collection')
    banks = {b['bank_id']: b for b in collection['banks']}
    expected = {('full_category', k, 0, 75, True) for k in ('ggf', 'slics', 'vanilla', 'supervised_vanilla')}
    expected |= {('category_split', k, f, 75, True) for k in ('ggf', 'supervised_vanilla') for f in (0, 1)}
    expected |= {('category_split', 'vanilla', 0, 75, True)}
    expected |= {('category_split', 'slics', f, n, False) for f, n in ((0, 37), (1, 38))}
    actual = {(b['regime'], b['kind'], b['fold'], b['groups'], b['capacity_matched']) for b in banks.values()}
    if len(banks) != len(collection['banks']) or len(banks) != 11 or actual != expected or any(b['seed'] != 21 for b in banks.values()):
        raise ValueError('Missing, duplicate, or capacity-mismatched native bank')
    folder = root / 'reviews/coherence'
    lock = read(folder / 'REVIEW_LOCK.json')
    for name, key in (('PUBLIC.json', 'public_sha256'), ('PRIVATE.json', 'private_sha256')):
        if sha(folder / name) != lock[key]:
            raise ValueError('Frozen review mapping changed')
    for name, digest in lock['cards'].items():
        if sha(relative(folder, name)) != digest:
            raise ValueError('Review card changed')
    public, private = read(folder / 'PUBLIC.json'), read(folder / 'PRIVATE.json')
    if private['collection_sha256'] != sha(root / 'COLLECTION_COMPLETE.json'):
        raise ValueError('Cards do not belong to the returned collection')
    pids = [r['case_id'] for r in public['cases']]
    qids = [r['case_id'] for r in private['cases']]
    if len(pids) != len(set(pids)) or len(qids) != len(set(qids)) or set(pids) != set(qids):
        raise ValueError('Unknown, repeated, or missing blinded cases')
    for bank_id, bank in banks.items():
        units = [r['unit'] for r in private['cases'] if r['bank_id'] == bank_id]
        if sorted(units) != list(range(bank['groups'])):
            raise ValueError('Every native group must be retained exactly once')
    if any(r['bank_id'] not in banks for r in private['cases']):
        raise ValueError('Unknown native bank in review')
    if sum(b['groups'] for b in banks.values()) != len(pids):
        raise ValueError('Incorrect whole-bank denominator')
    by_id = {r['case_id']: r for r in public['cases']}
    for case in private['cases']:
        item = by_id[case['case_id']]
        if item['pair_ids'] != [0, 1, 2] or len(item['invalid_pairs']) != 3 or item['invalid_pairs'] != case['invalid_pairs']:
            raise ValueError('Frozen pair structure differs')
        if item['card'] not in lock['cards']:
            raise ValueError('Public card is not hash locked')
    human = read(folder / 'HUMAN_VALIDATION_SAMPLE.json')
    if len(human['case_ids']) > 30 or len(set(human['case_ids'])) != len(human['case_ids']) or not set(human['case_ids']) <= set(pids):
        raise ValueError('Invalid fixed human validation subset')
    def frozen_key(*xs):
        return hashlib.sha256(('all75-pair-v1|'+'|'.join(map(str, xs))).encode()).hexdigest()
    strata = {}
    for case in private['cases']:
        bank = banks[case['bank_id']]
        strata.setdefault((bank['regime'], bank['kind'], bank['capacity_matched']), []).append(case)
    expected_human = []
    for stratum, cases in sorted(strata.items()):
        expected_human.extend(r['case_id'] for r in sorted(cases, key=lambda r:frozen_key('human-validation', r['case_id']))[:(4 if stratum[2] else 2)])
    if human['case_ids'] != expected_human:
        raise ValueError('Human validation sample differs from the frozen pre-outcome selection')
    return collection, public, private, lock, human


def make_packets(root):
    root = Path(root)
    collection, public, private, lock, human = verify_review(root)
    # The reviewer directory has no registry, PRIVATE mapping, AP scores, prior
    # interpretations, category names, or other contexts' annotations.
    out = root / 'local_review/reviewer_public'
    source = root / 'reviews/coherence'
    copy_exact(source / 'PUBLIC.json', out / 'PUBLIC.json')
    for case in public['cases']:
        copy_exact(relative(source, case['card']), relative(out, case['card']))
    instructions = (
        'Read only this public packet and its PNG cards in a fresh review context. '
        'Do not access the study registry, private mapping, previous interpretations, '
        'method identities, target labels, numerical scores, or another context\'s judgments. '
        'Inspect every numbered image at sufficient resolution, including the zoom strips. '
        'Only the marked patches are evidence; an object elsewhere in a crop is insufficient. '
        'For each fixed pair (views 1–2, 3–4, 5–6), return two separate codes: yes/no/unclear. '
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
    write(root / 'local_review/PACKETS_READY.json', dict(complete=True, counts=counts, public_sha256=lock['public_sha256'],
        review_lock_sha256=sha(source / 'REVIEW_LOCK.json'), label_status='pending_actual_blinded_review', training_unchanged=True))
    return counts


def accept_batch(root, context, batch, labels):
    if context not in CONTEXTS or not batch.startswith('B') or not batch[1:].isdigit():
        raise ValueError('Unknown review context or batch')
    root = Path(root)
    _, public, _, lock, _ = verify_review(root)
    packet = read(root / 'local_review/reviewer_public/packets' / context / f'{batch}.json')
    data = read(labels)
    if data.get('public_sha256') != lock['public_sha256'] or data.get('method_blind') is not True or not data.get('reviewer', '').strip():
        raise ValueError('Actual reviewer ID and locked public source are required')
    for other in CONTEXTS:
        if other != context:
            for prior in (root / 'reviews/coherence/batch_labels' / other).glob('*.json'):
                if read(prior).get('reviewer') == data['reviewer']:
                    raise ValueError('Each stream requires fresh reviewer contexts, not reused identities')
    cases = {r['case_id']: r for r in data['cases']}
    if len(cases) != len(data['cases']) or set(cases) != {r['case_id'] for r in packet['cases']}:
        raise ValueError('Batch does not exactly cover its frozen cases')
    by_id = {r['case_id']: r for r in public['cases']}
    for cid, case in cases.items():
        if sorted(p['pair'] for p in case['pairs']) != [0, 1, 2]:
            raise ValueError('All three original pairs are required')
        for pair in case['pairs']:
            for endpoint in ('semantic', 'visual'):
                if pair[endpoint] not in CODES:
                    raise ValueError('Unfilled or invalid research judgment')
                if by_id[cid]['invalid_pairs'][pair['pair']] and pair[endpoint] != 'unclear':
                    raise ValueError('Missing evidence must remain unclear')
                if pair[endpoint] == 'yes' and not pair.get('shared_meaning', '').strip():
                    raise ValueError('Positive judgments require a concrete interpretation')
    target = root / 'reviews/coherence/batch_labels' / context / f'{batch}.json'
    copy_exact(labels, target)
    return dict(context=context, batch=batch, cases=len(cases), source_sha256=sha(target))


def status(root):
    collection, public, _, _, human = verify_review(root)
    out = dict(label_status='pending', planned_groups=len(public['cases']), planned_pairs=3 * len(public['cases']),
               human_groups=len(human['case_ids']), context_cases={})
    previous_reviewers = set()
    for context in CONTEXTS:
        files = list((Path(root) / 'reviews/coherence/batch_labels' / context).glob('*.json'))
        data = [read(p) for p in files]
        reviewers = {d.get('reviewer') for d in data}
        if reviewers & previous_reviewers:
            raise ValueError('Fresh reviewer contexts cannot be reused across streams')
        previous_reviewers.update(reviewers)
        ids = [r['case_id'] for d in data for r in d['cases']]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate raw review cases')
        out['context_cases'][context] = len(ids)
    out['complete_raw_coverage'] = all(n == len(public['cases']) for n in out['context_cases'].values())
    return out


def score_coherence(root):
    """Use the frozen scorer directly; concentration/presence need not be ready."""
    verify_review(root)
    if not status(root)['complete_raw_coverage']:
        raise ValueError('All three complete blinded streams are required')
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SOURCE), str(SOURCE / 'core'), os.environ.get('PYTHONPATH', '')]))
    for context in CONTEXTS:
        subprocess.run([sys.executable, str(SOURCE / 'prepare_evaluation_reviews.py'), '--root', str(root), '--phase', 'assemble-' + context], env=environment, check=True)
    subprocess.run([sys.executable, str(SOURCE / 'score_evaluation.py'), '--root', str(root), '--phase', 'coherence'], env=environment, check=True)
    subprocess.run([sys.executable, str(SOURCE / 'report_evaluation.py'), '--root', str(root)], env=environment, check=True)
    return comparative_readout(root)


def comparative_readout(root):
    root = Path(root)
    collection, public, _, _, _ = verify_review(root)
    result = read(root / 'COHERENCE_COMPLETE.json')
    if not result.get('complete') or result['public_sha256'] != sha(root / 'reviews/coherence/PUBLIC.json'):
        raise ValueError('Complete frozen coherence result required')
    for source in result['sources']:
        if sha(relative(root, source['path'])) != source['sha256']:
            raise ValueError('Coherence annotation source changed')
    if set(result['bank_results']) != {b['bank_id'] for b in collection['banks']}:
        raise ValueError('A native bank is missing from the readout')
    grouped = {}
    for bank in collection['banks']:
        grouped.setdefault((bank['regime'], bank['kind'], bank['capacity_matched']), []).append(bank)
    rows = []
    for (regime, method, matched), banks in sorted(grouped.items()):
        row = dict(regime=regime, method=method, capacity_matched=matched, groups=sum(b['groups'] for b in banks), native_banks=len(banks), endpoints={})
        for endpoint in ('semantic', 'visual'):
            pieces = [result['bank_results'][b['bank_id']][endpoint] for b in banks]
            if any(p['denominator'] != b['groups'] * 3 for p, b in zip(pieces, banks)):
                raise ValueError('Incorrect planned pair denominator')
            counts = {k:sum(p['counts'][k] for p in pieces) for k in ('yes', 'no', 'unclear', 'missing')}
            n = sum(p['denominator'] for p in pieces)
            if sum(counts.values()) != n:
                raise ValueError('Coherence counts do not cover every planned pair')
            row['endpoints'][endpoint] = dict(counts=counts, denominator=n, confirmed=counts['yes']/n,
                unresolved_upper=(counts['yes']+counts['unclear']+counts['missing'])/n)
        rows.append(row)
    effects = []
    for regime in ('full_category', 'category_split'):
        matched = {r['method']:r for r in rows if r['regime'] == regime and r['capacity_matched']}
        for baseline, other in matched.items():
            if baseline == 'ggf':
                continue
            for endpoint in ('semantic', 'visual'):
                effects.append(dict(regime=regime, baseline=baseline, endpoint=endpoint,
                    GGF_minus_baseline=matched['ggf']['endpoints'][endpoint]['confirmed']-other['endpoints'][endpoint]['confirmed']))
    output = dict(scope='seed21_pilot', complete_coherence=True, all_other_metrics_complete=False, methods=rows, descriptive_effects=effects,
        source_sha256=sha(root / 'COHERENCE_COMPLETE.json'),
        limits=['Whole-bank frozen pair consistency, not the broader historical C/P/U group rubric',
            'Texture and geometry count in the separate visual endpoint; no category-label anchoring',
            'Category folds use distinct GGF/SV fits but share one Vanilla fit; shared Vanilla is counted once',
            'Native SLiCS37/38 remains a separate unmatched reference; full-category SLiCS75 is matched',
            'AI majority is not human ground truth; fixed human validation is reported separately',
            'One training seed; no training-variability confidence interval or universal superiority claim',
            'Unresolved upper bounds are not statistical confidence bounds; no outcome-dependent composite'])
    write(root / 'local_review/SEMANTIC_COMPARISON.json', output)
    table = ''.join('<tr><td>'+html.escape(r['regime'])+'</td><td>'+html.escape(r['method'])+('</td><td>matched75' if r['capacity_matched'] else '</td><td>native reference')+'</td><td>'+str(r['groups'])+'</td>'+''.join('<td>'+str(r['endpoints'][e]['counts']['yes'])+'/'+str(r['endpoints'][e]['denominator'])+' ('+format(r['endpoints'][e]['confirmed'],'.1%')+')</td>' for e in ('semantic','visual'))+'</tr>' for r in rows)
    rendered = '<!doctype html><meta charset="utf-8"><title>Seed21 semantic comparison</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:35px auto;padding:20px}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}table{border-collapse:collapse}</style><h1>Seed21 semantic comparison</h1><p>Complete blinded coherence endpoint. Other dashboard metrics may still be pending.</p><table><tr><th>Experiment</th><th>Method</th><th>Capacity</th><th>Groups</th><th>Specific semantic</th><th>Visual structure</th></tr>'+table+'</table><p>Both scores use all planned pairs. Unknowns remain in the denominator.</p><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in output['limits'])+'</ul><p><a href="SEMANTIC_COMPARISON.json">Exact counts, descriptive differences and source hash</a></p>'
    path = root / 'local_review/SEMANTIC_COMPARISON.html'
    path.write_text(rendered, encoding='utf8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--phase', required=True, choices=['import', 'packets', 'status', 'accept', 'score-coherence'])
    parser.add_argument('--archives', nargs='+')
    parser.add_argument('--context', choices=CONTEXTS)
    parser.add_argument('--batch')
    parser.add_argument('--labels')
    args = parser.parse_args()
    if args.phase == 'import':
        value = import_archives(args.root, args.archives or [])
    elif args.phase == 'packets':
        value = make_packets(args.root)
    elif args.phase == 'status':
        value = status(args.root)
    elif args.phase == 'accept':
        value = accept_batch(args.root, args.context, args.batch, args.labels)
    else:
        value = score_coherence(args.root)
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
