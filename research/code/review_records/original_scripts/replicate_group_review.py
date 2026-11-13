"""Whole-group replication operator. Never generates research judgments.

Only explicit observed-FINAL completion receipts authorize acceptance. A raw JSON
appearing on disk does not authorize import. Original seed21 judgments are reused
with their original provenance, never counted as fresh independent review.
"""
import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import group_review as old
import review_ready as original

WORK = Path(__file__).resolve().parents[2]
STUDY = WORK / 'output/ggf_all75_presence_review_20261028/full_collection_audit'
ROOT = WORK / 'output/ggf_all75_coherence_replication_20261028/whole'
PUBLIC = ROOT / 'reviewer_public'
PILOT = old.STUDY
PRIOR = old.ROOT
STREAMS, CODES, SPEC = old.STREAMS, old.CODES, old.SPEC
INSTRUCTIONS = old.INSTRUCTIONS
read, sha, freeze, now = old.read, original.sha, old.freeze, old.now


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_label(n, fits):
    if n < 4:
        return 'insufficient_evidence'
    return 'clear' if fits >= math.ceil(5*n/6) else 'partial' if fits >= math.ceil(n/2) else 'no_clear_theme'


def validate_case(v, c):
    fit = v.get('fit_views', [])
    require(isinstance(fit, list) and all(type(x) is int for x in fit), 'Integer fit views required')
    require(len(set(fit)) == len(fit) and set(fit) <= set(c['valid_high_views']) and
            type(v.get('high_fit_count')) is int and v['high_fit_count'] == len(fit), 'Invalid fit count')
    exceptions = []
    for ex in v.get('exceptions', []):
        if type(ex) is int:
            exceptions.append(ex)
        elif isinstance(ex, dict) and type(ex.get('view')) is int and str(ex.get('reason', '')).strip():
            exceptions.append(ex['view'])
        elif isinstance(ex, str) and (m := re.match(r'^(?:View\s+([1-6])\b|([1-6])\s*:\s*\S)', ex)):
            # Both explicit forms identify the same numbered view. Preserve the
            # actual reviewer bytes/reason; only parse its unambiguous prefix.
            exceptions.append(int(m.group(1) or m.group(2)))
        else:
            raise ValueError('Each exception must identify its numbered view')
    require(sorted(exceptions) == sorted(set(c['valid_high_views']) - set(fit)), 'Every nonmatching valid view must be an exception')
    require(v.get('coherence') in CODES and v.get('specificity') in SPEC and
            v.get('confidence') in ('high', 'medium', 'low'), 'Invalid label')
    require(v.get('high_low_contrast') == 'not_assessed', 'No LOW evidence supplied')
    require(isinstance(v.get('theme'), str) and len(v['theme'].split()) <= 30 and
            isinstance(v.get('observations'), str) and v['observations'].strip(), 'Theme/rationale missing or too long')
    if v['coherence'] != 'insufficient_evidence':
        require(v['coherence'] == expected_label(c['valid_high_count'], len(fit)), 'Five-of-six rubric mismatch')
        require(c['valid_high_count'] >= 4, 'Fewer than four valid HIGH views')
        if len(fit) >= math.ceil(c['valid_high_count']/2):
            require(v['theme'].strip(), 'Positive motif must be stated')


def verify_full():
    collection = read(STUDY/'COLLECTION_COMPLETE.json')
    require(collection.get('complete') and collection['training_seeds'] == [21, 22, 23], 'Complete three-seed collection required')
    banks = {b['bank_id']: b for b in collection['banks']}
    require(len(banks) == len(collection['banks']), 'Duplicate banks')
    matched = {k: b for k, b in banks.items() if b['capacity_matched']}
    require(len(matched) == 27 and all(b['groups'] == 75 for b in matched.values()), '27 matched75 banks required')
    folder = STUDY/'reviews/coherence'
    lock = read(folder/'REVIEW_LOCK.json')
    for name, k in [('PUBLIC.json', 'public_sha256'), ('PRIVATE.json', 'private_sha256')]:
        require(sha(folder/name) == lock[k], 'Full public/private mapping changed')
    private = read(folder/'PRIVATE.json')
    require(private['collection_sha256'] == sha(STUDY/'COLLECTION_COMPLETE.json'), 'Collection mapping mismatch')
    pub = read(folder/'PUBLIC.json')['cases']
    require([c['case_id'] for c in pub] == [c['case_id'] for c in private['cases']], 'Full public/private order mismatch')
    for bid, b in banks.items():
        require(sorted(c['unit'] for c in private['cases'] if c['bank_id'] == bid) == list(range(b['groups'])), 'Native units missing or duplicated')
    require(len(private['cases']) == sum(b['groups'] for b in banks.values()), 'Full group denominator mismatch')
    for card, digest in lock['cards'].items():
        require(sha(folder/card) == digest, 'Actual full card changed: '+card)
    return banks, private['cases'], lock


def seed21_reuse(banks, cases):
    """Recheck both historical audits against actual current sources and mapping."""
    from PIL import Image
    render_path = STUDY/'SEED21_RENDERED_EVIDENCE_REUSE_AUDIT.json'
    model_path = STUDY/'SEED21_CHECKPOINT_AND_VIEW_REUSE_AUDIT.json'
    render, model = read(render_path), read(model_path)
    require(render['complete'] and model['complete'] and model['all750_rendered_evidence_bodies_byte_identical'], 'Both seed21 reuse audits must pass')
    prior_private = read(PRIOR/'PRIVATE.json')
    prior_banks = {b['bank_id']: b for b in read(PILOT/'COLLECTION_COMPLETE.json')['banks']}
    mapped = {(c['bank_id'], c['unit']): c for c in cases}
    audited_cases = {c['original_case']: c for c in render['cases']}
    audited_banks = {b['bank_id']: b for b in model['banks']}
    anchors = {}
    def pin(p):
        anchors[str(p.relative_to(WORK)).replace('\\', '/')] = sha(p)
    for p in [render_path, model_path, PRIOR/'PROTOCOL.json', PRIOR/'REVIEW_LOCK.json',
              PRIOR/'RESULTS.json', PRIOR/'PRIVATE.json', PRIOR/'reviewer_public/PUBLIC.json',
              PILOT/'COLLECTION_COMPLETE.json', PILOT/'PORTABLE_LOCK.json',
              STUDY/'COLLECTION_COMPLETE.json', STUDY/'PORTABLE_LOCK.json',
              STUDY/'reviews/coherence/REVIEW_LOCK.json', STUDY/'reviews/coherence/PRIVATE.json',
              STUDY/'reviews/coherence/PUBLIC.json']:
        pin(p)
    require(read(PRIOR/'PROTOCOL.json')['instructions'] == INSTRUCTIONS, 'Original group rubric differs')
    for bid in {c['bank_id'] for c in prior_private['cases']}:
        a, b, audit = prior_banks[bid], banks[bid], audited_banks[bid]
        fields = set(audit['model_identity_fields']) | {'normalization_identity'}
        require(all(a.get(k) == b.get(k) for k in fields), 'Checkpoint/model identity changed')
        require(a['weights_sha'] == audit['weights_sha'], 'Checkpoint digest mismatch')
        for base, field in [(PILOT, 'original_views_sha256'), (STUDY, 'full_views_sha256')]:
            vp, pp = base/'portable'/bid/'VIEWS.json', base/'portable'/bid/'POOLED.npz'
            require(sha(vp) == audit[field] and sha(pp) == audit['pooled_sha256'], 'Portable evidence changed')
            pin(vp); pin(pp)
        va, vb = read(PILOT/'portable'/bid/'VIEWS.json'), read(STUDY/'portable'/bid/'VIEWS.json')
        va.pop('signature', None); vb.pop('signature', None)
        require(va == vb, 'View identity/order/patch data changed')
    mapping = []
    old_public = {c['case_id']: c for c in read(PRIOR/'reviewer_public/PUBLIC.json')['cases']}
    for c in prior_private['cases']:
        target = mapped[(c['bank_id'], c['unit'])]
        audit = audited_cases[c['case_id']]
        require(target['case_id'] == audit['full_case'] and all(c[k] == target[k] for k in ('bank_id', 'unit', 'source_slots', 'image_ids', 'invalid_pairs')), 'Seed21 mapping changed')
        op = PILOT/'reviews/coherence/cards'/f"{c['case_id']}.png"
        fp = STUDY/'reviews/coherence/cards'/f"{target['case_id']}.png"
        pp = PRIOR/'reviewer_public/cards'/op.name
        require(sha(op) == sha(pp) == audit['original_card_sha256'] and sha(fp) == audit['full_card_sha256'], 'Seed21 card hash changed')
        with Image.open(op) as x, Image.open(fp) as y:
            require(x.size == y.size and x.mode == y.mode, 'Rendered evidence dimensions changed')
            bx, by = x.crop((0, 28, x.width, 364)).tobytes(), y.crop((0, 28, y.width, 364)).tobytes()
            require(bx == by and hashlib.sha256(bx).hexdigest() == audit['body_pixel_sha256'], 'Rendered evidence body changed')
        require(old_public[c['case_id']]['valid_high_count'] == 6, 'Old HIGH validity changed')
        mapping.append(dict(original_case_id=c['case_id'], mapped_case_id=target['case_id'],
                            original_card_sha256=sha(op), full_card_sha256=sha(fp),
                            body_pixel_sha256=audit['body_pixel_sha256'], original_source_case=c, full_source_case=target))
    require(len(mapping) == 675, 'Exactly675 original group cases required')
    by_old = {m['original_case_id']: m for m in mapping}
    rows, files, reviewers = [], [], set()
    for ctx in STREAMS:
        covered = set()
        paths = sorted((PRIOR/'accepted'/ctx).glob('B*.json'))
        require(len(paths) == 57, 'Prior group stream incomplete')
        for p in paths:
            d = old.validate_labels(p, ctx, p.stem)
            require(d['reviewer'] not in reviewers, 'Prior reviewer provenance duplicated')
            reviewers.add(d['reviewer']); pin(p)
            returned = PRIOR/'returned_reviews'/ctx/p.name
            require(sha(returned) == sha(p), 'Original accepted/raw judgment mismatch'); pin(returned)
            files.append(dict(context=ctx, batch=p.stem, accepted_file=str(p.relative_to(WORK)).replace('\\', '/'),
                              raw_file=str(returned.relative_to(WORK)).replace('\\', '/'), sha256=sha(p),
                              original_public_sha256=d['public_sha256'], reviewer=d['reviewer']))
            for v in d['cases']:
                require(v['case_id'] not in covered, 'Duplicate reused group')
                covered.add(v['case_id'])
                m = by_old[v['case_id']]
                rows.append(dict(context=ctx, case_id=m['mapped_case_id'], original_case_id=v['case_id'],
                                 reviewer=d['reviewer'], original_raw_file=str(returned.relative_to(WORK)).replace('\\', '/'),
                                 original_raw_sha256=sha(p), original_public_sha256=d['public_sha256'], original_label=v))
        require(covered == set(by_old), 'Reused stream coverage differs')
    return dict(schema='whole-group-seed21-provenance-reuse-v1', original_groups=675, reused_group_ratings=2025,
                new_reviews=0, independent_new_review_claim=False, adaptation='Case IDs map to the full study; original label objects and original raw/public hashes remain unchanged. PNG title G IDs alone may differ; rows28through363 are identical.',
                mapping=mapping, source_files=files, ratings=rows, source_anchors=anchors)


def prepare():
    banks, cases, full_lock = verify_full()
    reuse = seed21_reuse(banks, cases)
    selected = [c for c in cases if banks[c['bank_id']]['capacity_matched'] and banks[c['bank_id']]['seed'] in (22, 23)]
    require(len(selected) == 1350, 'Exactly1350 new matched groups required')
    images = {r['image_id']: r for r in read(STUDY/'IMAGE_ASSETS.json')['images']}
    locks = {b['bank_id']: b for b in read(STUDY/'PORTABLE_LOCK.json')['banks']}
    viewmaps = {}
    for bid in {c['bank_id'] for c in selected}:
        p = STUDY/'portable'/bid/'VIEWS.json'
        require(sha(p) == locks[bid]['views_sha256'], 'New seed view cache changed')
        viewmaps[bid] = read(p)
        reuse['source_anchors'][str(p.relative_to(WORK)).replace('\\', '/')] = sha(p)
    pub, hashes = [], {}
    for c in selected:
        cid = c['case_id']; card = f'cards/{cid}.png'
        source = STUDY/'reviews/coherence'/card
        original.copy_exact(source, PUBLIC/card); hashes[card] = full_lock['cards'][card]
        views = viewmaps[c['bank_id']]['roles']['audit'][str(c['unit'])]
        valid = [j+1 for j, slot in enumerate(c['source_slots']) if slot < len(views) and
                 views[slot].get('patches') and views[slot].get('pooled', 0) > 0 and
                 images.get(views[slot].get('image_id'), {}).get('status') == 'ready']
        require(len(valid) == 6, 'Expected six original nonzero HIGH views')
        pub.append(dict(case_id=cid, card=card, valid_high_views=valid, valid_high_count=len(valid)))
    protocol = dict(schema='all75-whole-group-replication-v1', scope='matched75_seed22_seed23_whole_group_replication',
                    new_groups=1350, groups_per_stream=1350, streams=3, new_group_ratings=4050,
                    seed21_reused_groups=675, seed21_reused_group_ratings=2025, combined_groups=2025,
                    order='Original full study case order, filtered solely by capacity_matched and seed22/23, frozen before new ratings; no outcome filtering.',
                    packet_size=12, packets_per_stream=113, final_packet_size=6,
                    instructions=INSTRUCTIONS, original_protocol=read(PRIOR/'PROTOCOL.json'),
                    original_protocol_sha256=sha(PRIOR/'PROTOCOL.json'),
                    primary='At least two of three clear votes / all native groups; unknowns and unresolved votes remain in denominator.',
                    secondary='Majority clear AND at least two specific or broad_but_discriminating specificity ratings / all native groups.',
                    completion_gate='Explicit start assignment, actual agent FINAL observed by root, finish receipt with finalized raw hash and FINAL text hash, then explicit accept. File presence alone never imports labels.',
                    concurrency='At most3 local active assignments; root coordinates total active agents across whole and pair work.',
                    reporting='Per seed/bank/fold and seed-method equal-fold means; aggregate equal-seed means and full distributions; shared category-split Vanilla counted once. Seed21 reuse is not fresh review. AI replicates are not independent human validation. No LOW contrast.')
    freeze(ROOT/'PROTOCOL.json', protocol)
    freeze(PUBLIC/'PUBLIC.json', dict(schema=protocol['schema'], instructions=INSTRUCTIONS, cases=pub))
    freeze(ROOT/'PRIVATE.json', dict(cases=[c for c in cases if banks[c['bank_id']]['capacity_matched']],
                                      banks=[b for b in banks.values() if b['capacity_matched']], new_case_ids=[c['case_id'] for c in selected]))
    freeze(ROOT/'SEED21_REUSE.json', reuse)
    queue, packets = [], {}
    public_sha = sha(PUBLIC/'PUBLIC.json')
    for ctx in STREAMS:
        for start in range(0, 1350, 12):
            batch = f'B{start//12:03d}'; cc = pub[start:start+12]
            template = dict(public_sha256=public_sha, method_blind=True, reviewer='', cases=[dict(case_id=c['case_id'], theme='', fit_views=[], high_fit_count=None, exceptions=[], coherence='', specificity='', confidence='', high_low_contrast='not_assessed', observations='') for c in cc])
            packet = dict(context=ctx, batch=batch, instructions=INSTRUCTIONS, cases=cc, card_root='../../', output_template=template)
            rel = f'packets/{ctx}/{batch}.json'; freeze(PUBLIC/rel, packet); packets[rel] = sha(PUBLIC/rel)
            queue.append(dict(context=ctx, batch=batch, cases=len(cc)))
    freeze(PUBLIC/'QUEUE.json', dict(batches=queue, total_groups=1350, streams=3, groups_per_stream=1350))
    freeze(ROOT/'REVIEW_LOCK.json', dict(protocol_sha256=sha(ROOT/'PROTOCOL.json'), public_sha256=public_sha,
                                        private_sha256=sha(ROOT/'PRIVATE.json'), reuse_sha256=sha(ROOT/'SEED21_REUSE.json'),
                                        queue_sha256=sha(PUBLIC/'QUEUE.json'), cards=hashes, packets=packets))
    return status()


def verify_lock():
    lock = read(ROOT/'REVIEW_LOCK.json')
    for p, k in [(ROOT/'PROTOCOL.json', 'protocol_sha256'), (PUBLIC/'PUBLIC.json', 'public_sha256'),
                 (ROOT/'PRIVATE.json', 'private_sha256'), (ROOT/'SEED21_REUSE.json', 'reuse_sha256'), (PUBLIC/'QUEUE.json', 'queue_sha256')]:
        require(sha(p) == lock[k], 'Locked source changed: '+str(p))
    return lock


def packet_path(ctx, batch):
    require(ctx in STREAMS and re.fullmatch(r'B\d{3}', batch) and int(batch[1:]) < 113, 'Unknown packet')
    return PUBLIC/'packets'/ctx/(batch+'.json')


def verified_packet(ctx, batch):
    lock = verify_lock(); p = packet_path(ctx, batch)
    require(sha(p) == lock['packets'][f'packets/{ctx}/{batch}.json'], 'Packet changed')
    return read(p)


def validate_labels(path, ctx, batch):
    lock = verify_lock()
    packet, d = verified_packet(ctx, batch), read(path)
    require(d.get('public_sha256') == lock['public_sha256'] and d.get('method_blind') is True and
            isinstance(d.get('reviewer'), str) and d['reviewer'].startswith('agent:/root/'), 'Actual reviewer provenance required')
    require([v['case_id'] for v in d.get('cases', [])] == [c['case_id'] for c in packet['cases']], 'Fixed case coverage/order differs')
    for v, c in zip(d['cases'], packet['cases']):
        require(sha(PUBLIC/c['card']) == lock['cards'][c['card']], 'Card changed')
        validate_case(v, c)
    return d


def state_read():
    p = ROOT/'WORK_STATE.json'
    return read(p) if p.exists() else dict(active=[], finished=[], accepted=[], created_at_utc=now())


def state_write(s):
    s['updated_at_utc'] = now()
    p = ROOT/'WORK_STATE.json'; p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(s, indent=2), encoding='utf8'); tmp.replace(p)


def status():
    verify_lock(); s = state_read()
    for r in s['accepted']:
        p = ROOT/'accepted'/r['context']/(r['batch']+'.json')
        require(sha(p) == r['sha256'], 'Accepted artifact changed')
        require(sha(ROOT/'returned_reviews'/r['context']/p.name) == r['sha256'], 'Accepted raw file changed')
        require(read(ROOT/'completion_receipts'/r['context']/p.name) == r and
                sha(ROOT/'observed_finals'/r['context']/(r['batch']+'.txt')) == r['final_text_sha256'], 'Accepted completion provenance changed')
    done = {(r['context'], r['batch']) for r in s['accepted']}
    busy = {(r['context'], r['batch']) for r in s['active'] + s['finished']}
    return dict(accepted_batches=len(done), accepted_new_group_ratings=sum(r['cases'] for r in s['accepted']),
                required_batches=339, required_new_group_ratings=4050, seed21_reused_group_ratings=2025,
                active=s['active'], finalized_pending_accept=s['finished'], complete=len(done) == 339,
                next_batch_by_context={ctx: next((f'B{i:03d}' for i in range(113) if (ctx, f'B{i:03d}') not in done|busy), None) for ctx in STREAMS})


def sync(start=None, finish=None, final_text=None, observed_by=None):
    status(); s = state_read()
    require(not (start and finish), 'Start and finish are separate operations')
    if start:
        ctx, batch, agent = start; verified_packet(ctx, batch)
        allrows = s['active'] + s['finished'] + s['accepted']
        require(len(s['active']) < 3 and not any((r['context'], r['batch']) == (ctx, batch) for r in allrows), 'Duplicate assignment or more than3 active')
        require(agent.startswith('agent:/root/') and not any(r.get('agent', r.get('reviewer')) == agent for r in allrows), 'Fresh agent required per packet')
        require(agent not in {r['reviewer'] for r in read(ROOT/'SEED21_REUSE.json').get('source_files', [])}, 'Reused seed21 reviewers cannot count as fresh reviewers')
        s['active'].append(dict(context=ctx, batch=batch, agent=agent, started_at_utc=now()))
    if finish:
        ctx, batch, agent = finish
        assignment = next((r for r in s['active'] if (r['context'], r['batch'], r['agent']) == (ctx, batch, agent)), None)
        require(assignment is not None, 'Active assignment required before finish')
        require(observed_by == 'agent:/root' and final_text and Path(final_text).is_file() and Path(final_text).read_text(encoding='utf-8-sig').strip(), 'Root-observed actual FINAL text file required')
        raw = ROOT/'returned_reviews'/ctx/(batch+'.json'); d = validate_labels(raw, ctx, batch)
        require(d['reviewer'] == agent, 'Assigned reviewer mismatch')
        # This is an operator attestation after the root sees actual FINAL; the
        # script cannot independently observe the collaboration transport.
        receipt = dict(context=ctx, batch=batch, agent=agent, reviewer=d['reviewer'], cases=len(d['cases']),
                       sha256=sha(raw), final_text_sha256=sha(final_text), observed_by=observed_by,
                       observation='Actual agent FINAL observed by root before this finish operation', finished_at_utc=now())
        target = ROOT/'completion_receipts'/ctx/(batch+'.json')
        original.copy_exact(final_text, ROOT/'observed_finals'/ctx/(batch+'.txt'))
        freeze(target, receipt); s['active'].remove(assignment); s['finished'].append(receipt)
    if start or finish:
        state_write(s)
    return status()


def accept(ctx, batch):
    status(); s = state_read()
    receipt_path = ROOT/'completion_receipts'/ctx/(batch+'.json')
    receipt = next((r for r in s['finished'] if (r['context'], r['batch']) == (ctx, batch)), None)
    require(receipt is not None and read(receipt_path) == receipt, 'Explicit hash-finalized completion receipt required')
    raw = ROOT/'returned_reviews'/ctx/(batch+'.json')
    require(sha(raw) == receipt['sha256'] and sha(ROOT/'observed_finals'/ctx/(batch+'.txt')) == receipt['final_text_sha256'], 'Finalized artifacts changed')
    d = validate_labels(raw, ctx, batch)
    require(d['reviewer'] == receipt['agent'], 'Finalized reviewer differs')
    original.copy_exact(raw, ROOT/'accepted'/ctx/raw.name)
    s['finished'].remove(receipt); s['accepted'].append(receipt); state_write(s)
    return status()


def summarize(outcomes):
    n = len(outcomes); require(n > 0, 'Empty denominator')
    counts = collections.Counter(o['majority'] for o in outcomes)
    sc = collections.Counter(o['specificity_majority'] for o in outcomes)
    raw_co = collections.Counter(r['coherence'] for o in outcomes for r in o['raw'])
    raw_sp = collections.Counter(r['specificity'] for o in outcomes for r in o['raw'])
    joint = sum(o['clear_and_discriminating'] for o in outcomes)
    return dict(groups=n, counts={k: counts[k] for k in (*CODES, 'unresolved_vote')},
                specificity_counts={k: sc[k] for k in (*SPEC, 'unresolved_vote')},
                raw_coherence_counts={k: raw_co[k] for k in CODES}, raw_specificity_counts={k: raw_sp[k] for k in SPEC},
                clear_rate=counts['clear']/n, clear_and_discriminating=joint, clear_and_discriminating_rate=joint/n,
                disagreements=sum(len({r['coherence'] for r in o['raw']}) > 1 for o in outcomes),
                specificity_disagreements=sum(len({r['specificity'] for r in o['raw']}) > 1 for o in outcomes),
                unknown_group_count=sum(any(r['coherence'] == 'insufficient_evidence' for r in o['raw']) for o in outcomes),
                unknown_raw_ratings=raw_co['insufficient_evidence'])


def majority(values, choices):
    return next((k for k in choices if values.count(k) >= 2), 'unresolved_vote')


def score():
    require(status()['complete'], 'All339 explicitly completed and accepted actual packets required')
    lock = verify_lock(); reuse = read(ROOT/'SEED21_REUSE.json')
    for p, h in reuse['source_anchors'].items():
        require(sha(WORK/p) == h, 'Original evidence or judgment changed: '+p)
    private = read(ROOT/'PRIVATE.json'); banks = {b['bank_id']: b for b in private['banks']}
    streams = {ctx: {} for ctx in STREAMS}
    for r in reuse['ratings']:
        streams[r['context']][r['case_id']] = r['original_label']
    for ctx in STREAMS:
        for i in range(113):
            p = ROOT/'accepted'/ctx/f'B{i:03d}.json'; d = validate_labels(p, ctx, p.stem)
            for v in d['cases']:
                require(v['case_id'] not in streams[ctx], 'Duplicate group rating')
                streams[ctx][v['case_id']] = v
        require(set(streams[ctx]) == {c['case_id'] for c in private['cases']}, 'All2025 group denominator required per stream')
    outcomes = []
    for c in private['cases']:
        raw = [streams[ctx][c['case_id']] for ctx in STREAMS]
        winner = majority([v['coherence'] for v in raw], CODES)
        spec = majority([v['specificity'] for v in raw], SPEC)
        b = banks[c['bank_id']]
        outcomes.append(dict(case_id=c['case_id'], bank_id=c['bank_id'], unit=c['unit'], seed=b['seed'], regime=b['regime'], method=b['kind'], fold=b['fold'],
                             majority=winner, specificity_majority=spec, clear_and_discriminating=winner == 'clear' and sum(v['specificity'] in ('specific', 'broad_but_discriminating') for v in raw) >= 2,
                             evidence_origin='reused_seed21' if b['seed'] == 21 else 'fresh_seed22_seed23', raw=raw))
    bank_results = []
    for bid, b in banks.items():
        oo = [o for o in outcomes if o['bank_id'] == bid]; require(len(oo) == 75, 'Bank denominator must be75')
        bank_results.append(dict(bank_id=bid, seed=b['seed'], regime=b['regime'], method=b['kind'], fold=b['fold'], **summarize(oo)))
    seed_methods = []
    for seed, regime, method in sorted({(o['seed'], o['regime'], o['method']) for o in outcomes}):
        oo = [o for o in outcomes if (o['seed'], o['regime'], o['method']) == (seed, regime, method)]
        rr = [r for r in bank_results if (r['seed'], r['regime'], r['method']) == (seed, regime, method)]
        seed_methods.append(dict(seed=seed, regime=regime, method=method, native_banks=len(rr), equal_fold_clear_rate=sum(r['clear_rate'] for r in rr)/len(rr),
                                 equal_fold_clear_and_discriminating_rate=sum(r['clear_and_discriminating_rate'] for r in rr)/len(rr), **summarize(oo)))
    aggregate = []
    for regime, method in sorted({(o['regime'], o['method']) for o in outcomes}):
        oo = [o for o in outcomes if (o['regime'], o['method']) == (regime, method)]
        rr = [r for r in seed_methods if (r['regime'], r['method']) == (regime, method)]
        require(len(rr) == 3, 'Three seed summaries required')
        aggregate.append(dict(regime=regime, method=method, seeds=[21, 22, 23], equal_seed_equal_fold_clear_rate=sum(r['equal_fold_clear_rate'] for r in rr)/3,
                              equal_seed_equal_fold_clear_and_discriminating_rate=sum(r['equal_fold_clear_and_discriminating_rate'] for r in rr)/3, **summarize(oo)))
    result = dict(complete=True, protocol_sha256=lock['protocol_sha256'], new_groups=1350, reused_seed21_groups=675,
                  new_group_ratings=4050, reused_seed21_group_ratings=2025, aggregate=summarize(outcomes),
                  seeds=[dict(seed=s, **summarize([o for o in outcomes if o['seed'] == s])) for s in (21, 22, 23)],
                  banks=bank_results, seed_methods=seed_methods, methods=aggregate, group_outcomes=outcomes,
                  accepted_sources=state_read()['accepted'], reuse_manifest_sha256=lock['reuse_sha256'], limits=read(ROOT/'PROTOCOL.json')['reporting'])
    freeze(ROOT/'RESULTS.json', result)
    return {k: result[k] for k in ('complete', 'new_groups', 'reused_seed21_groups', 'methods')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('prepare', 'status', 'sync', 'accept', 'score'), required=True)
    p.add_argument('--start', nargs=3, metavar=('CONTEXT', 'BATCH', 'AGENT'))
    p.add_argument('--finish', nargs=3, metavar=('CONTEXT', 'BATCH', 'AGENT'))
    p.add_argument('--final-text', type=Path)
    p.add_argument('--observed-by')
    p.add_argument('--context', choices=STREAMS); p.add_argument('--batch')
    a = p.parse_args()
    if a.phase == 'prepare': result = prepare()
    elif a.phase == 'status': result = status()
    elif a.phase == 'sync': result = sync(a.start, a.finish, a.final_text, a.observed_by)
    elif a.phase == 'accept': result = accept(a.context, a.batch)
    else: result = score()
    print(json.dumps(result, indent=2))
