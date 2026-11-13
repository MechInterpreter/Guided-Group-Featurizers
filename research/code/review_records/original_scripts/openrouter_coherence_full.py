"""Run and recount the frozen seven-order, matched-75 cross-model review."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, stdev
import time
from urllib.request import Request, urlopen

import openrouter_coherence_pilot as pilot
import replicate_group_review as old
from openrouter_coherence_pilot import encode, require, save, sha


OUT = pilot.WORK / 'output/ggf_all75_openrouter_full'
MANIFEST = OUT / 'PRODUCTION_MANIFEST.json'
GATE = OUT / 'PILOT_GATE.json'
SEEDS = list(range(21, 28))
CONTEXTS = ('context1', 'context2', 'context3')
KEY_URL = 'https://openrouter.ai/api/v1/key'
RESERVE = 0.25
OLD_FULL = old.STUDY / 'reviews/coherence'
REPLICATION = old.ROOT.parent
LATE = pilot.SOURCE


def read(path, digest=None):
    raw = path.read_bytes()
    if digest is not None:
        require(sha(raw) == digest, f'Frozen input changed: {path}')
    return json.loads(raw)


def index(rows, key='case_id'):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f'Duplicate {key}')
    return result


def rel(path):
    return path.relative_to(pilot.WORK).as_posix()


def public(path, lock):
    doc = read(path, lock['public_sha256'])
    return index(doc['cases']), doc.get('instructions')


def build_catalog():
    full_lock = read(OLD_FULL / 'REVIEW_LOCK.json')
    pair_early, _ = public(OLD_FULL / 'PUBLIC.json', full_lock)
    whole_lock = read(old.ROOT / 'REVIEW_LOCK.json')
    early = read(old.ROOT / 'PRIVATE.json', whole_lock['private_sha256'])
    reused = read(old.ROOT / 'SEED21_REUSE.json', whole_lock['reuse_sha256'])
    earlier = {r['mapped_case_id']: r for r in reused['mapping']}
    original_lock = read(old.PRIOR / 'REVIEW_LOCK.json')
    whole_21, _ = public(old.PRIOR / 'reviewer_public/PUBLIC.json', original_lock)
    whole_22_23, whole_rubric = public(old.PUBLIC / 'PUBLIC.json', whole_lock)
    pair_lock = read(REPLICATION / 'pair/REVIEW_LOCK.json')
    pair_public, pair_rubric = public(REPLICATION / 'pair/reviewer_public/PUBLIC.json', pair_lock)
    pair_private = index(read(REPLICATION / 'pair/PRIVATE.json', pair_lock['private_sha256'])['cases'])
    full_private = index(read(OLD_FULL / 'PRIVATE.json', full_lock['private_sha256'])['cases'])
    late_lock = read(LATE / 'REVIEW_LOCK.json')
    late = read(LATE / 'PRIVATE.json', late_lock['private_sha256'])
    late_public = {}
    for family in pilot.FAMILIES:
        # The late lock pins each packet and the public file hash in its template.
        packet_name = f'{family}/context1/B000.json'
        first = read(pilot.PUBLIC / packet_name, late_lock['packets'][packet_name])
        digest = first['output_template']['public_sha256']
        doc = read(pilot.PUBLIC / family / 'PUBLIC.json', digest)
        require(doc['instructions'] == (pair_rubric if family == 'pair' else whole_rubric),
                'Late rubric differs')
        late_public[family] = index(doc['cases'])

    banks = index(early['banks'] + late['banks'], 'bank_id')
    require(len(banks) == 63 and all(b['capacity_matched'] and b['groups'] == 75
                                    for b in banks.values()), 'Unexpected matched bank census')
    groups = []
    for era, study in [('early', early), ('late', late)]:
        for row in study['cases']:
            cid, bank = row['case_id'], banks[row['bank_id']]
            group = dict(era=era, case_id=cid, bank_id=bank['bank_id'], unit=row['unit'],
                         seed=bank['seed'], regime=bank['regime'], method=bank['kind'], fold=bank['fold'])
            if era == 'late':
                sources = {family: (late_public[family][cid], pilot.PUBLIC, late_lock) for family in pilot.FAMILIES}
            else:
                require(full_private[cid] == row, 'Early pair and whole group identities differ')
                source = pair_early[cid]
                if bank['seed'] == 21:
                    item = earlier[cid]
                    source_id = item['original_case_id']
                    old_case = whole_21[source_id]
                    require(item['full_card_sha256'] == full_lock['cards'][source['card']]
                            and item['original_card_sha256'] == original_lock['cards'][old_case['card']],
                            'Seed 21 original evidence differs')
                    whole = (old_case, old.PRIOR / 'reviewer_public', original_lock)
                else:
                    require(pair_private[cid] == row and pair_public[cid] == source,
                            'Early pair evidence identity differs')
                    require(pair_lock['cards'][source['card']] == full_lock['cards'][source['card']],
                            'Early pair card differs from reviewed PNG')
                    whole = (whole_22_23[cid], old.PUBLIC, whole_lock)
                sources = {'pair': (source, OLD_FULL, full_lock), 'whole': whole}
            group['evidence'] = {}
            for family, (case, root, lock) in sources.items():
                fields = ('pair_ids', 'invalid_pairs') if family == 'pair' else ('valid_high_views', 'valid_high_count')
                card = root / case['card']
                require(card.is_file(), f'Frozen card missing: {card}')
                group['evidence'][family] = dict(case={k: case[k] for k in ('case_id', *fields)},
                                                path=rel(card), sha256=lock['cards'][case['card']])
            groups.append(group)
    require(len(groups) == 4725 and len({(r['bank_id'], r['unit']) for r in groups}) == 4725,
            'Unexpected group census or duplicate native units')
    require(Counter(r['seed'] for r in groups) == {s: 675 for s in SEEDS}, 'Unexpected seed census')
    require(all(sum(1 for b in banks.values() if b['seed'] == s) == 9 for s in SEEDS),
            'Unexpected bank count per seed')
    random.Random('ggf-openrouter-full-v1').shuffle(groups)
    return dict(groups=groups, rubrics={'pair': pair_rubric, 'whole': whole_rubric},
                model=pilot.STRONGER_MODEL, provider=pilot.STRONGER_PROVIDER,
                concise={'pair': True, 'whole': False},
                payload_source_sha256=sha(Path(pilot.__file__).read_bytes()))


def catalog():
    current = build_catalog()
    raw = encode(current)
    if MANIFEST.exists():
        require(MANIFEST.read_bytes() == raw, 'Full evidence or request-builder manifest changed')
    else:
        save(MANIFEST, raw)
    return current


def inputs(group, family, rubrics):
    e = group['evidence'][family]
    case = e['case']
    card = (pilot.WORK / e['path']).read_bytes()
    require(sha(card) == e['sha256'], f'Frozen card changed: {e["path"]}')
    if family == 'pair':
        template = dict(case_id=case['case_id'], pairs=[
            dict(pair=i, semantic='', visual='', shared_meaning='', exceptions='') for i in range(3)])
    else:
        template = dict(case_id=case['case_id'], theme='', fit_views=[], high_fit_count=None,
                        exceptions=[], coherence='', specificity='', confidence='',
                        high_low_contrast='not_assessed', observations='')
    body = pilot.request_body(case, template, rubrics[family], card, family,
                              paid=True, stronger=True, concise=family == 'pair')
    return case, template, body


def response_path(group, family):
    return OUT / 'responses' / group['era'] / family / f"{group['case_id']}.json"


def short_pilot():
    plan_path = pilot.OUT / 'SHORT_PILOT_PLAN.json'
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    require(plan['model'] == pilot.STRONGER_MODEL and plan['provider'] == pilot.STRONGER_PROVIDER,
            'Unexpected pilot model or provider')
    require([(j['case_id'], j['family']) for j in plan['jobs']]
            == [(cid, f) for cid in pilot.SHORT_CARDS for f in pilot.FAMILIES], 'Unexpected pilot sample')
    lock = read(pilot.SOURCE / 'REVIEW_LOCK.json', plan['source_lock_sha256'])
    costs, hashes, valid = defaultdict(list), {}, Counter()
    for j in plan['jobs']:
        case, template, body = pilot.job_inputs(j, lock)
        require(j.get('concise') and sha(body) == j['request_sha256'], 'Pilot payload changed')
        path = pilot.raw_path(j)
        reply = read(path)
        raw_response = json.loads(reply['raw_response'])
        label, error = pilot.checked(j, case, template)
        hashes[rel(path)] = sha(path.read_bytes())
        if label is not None:
            valid[j['family']] += 1
        cost = raw_response.get('usage', {}).get('cost')
        require(isinstance(cost, (int, float)) and math.isfinite(cost) and cost > 0
                and raw_response.get('service_tier') == 'flex'
                and raw_response.get('provider') == 'Google AI Studio', 'Pilot billing or route missing')
        costs[j['family']].append(cost)
    return dict(plan_sha256=sha(raw), response_hashes=hashes, valid=dict(valid),
                cost_per_call={f: mean(costs[f]) for f in pilot.FAMILIES},
                cost=sum(map(sum, costs.values())))


def production_pilot():
    concise = short_pilot()
    require(concise['valid'] == {'pair': 8, 'whole': 8}, 'Short pilot replies are incomplete')
    path = pilot.OUT / 'STRONGER_FLEX_PLAN.json'
    raw = path.read_bytes()
    plan = json.loads(raw)
    require(plan['model'] == pilot.STRONGER_MODEL and plan['provider'] == pilot.STRONGER_PROVIDER,
            'Earlier pilot route differs')
    lock = read(pilot.SOURCE / 'REVIEW_LOCK.json', plan['source_lock_sha256'])
    jobs = [j for j in plan['jobs'] if j['family'] == 'whole']
    require(len(jobs) == 24 and {j['case_id'] for j in jobs} == set(pilot.select_cards(lock)),
            'Earlier whole pilot census differs')
    costs, hashes = [], {}
    for job in jobs:
        case, template, body = pilot.job_inputs(job, lock)
        require(not job.get('concise') and sha(body) == job['request_sha256'], 'Whole pilot request changed')
        response = pilot.raw_path(job)
        label, error = pilot.checked(job, case, template)
        require(label is not None, f'Whole pilot validation failed: {error}')
        reply = json.loads(read(response)['raw_response'])
        require(reply.get('service_tier') == 'flex' and reply.get('provider') == 'Google AI Studio',
                'Whole pilot route differs')
        cost = reply.get('usage', {}).get('cost')
        require(isinstance(cost, (int, float)) and math.isfinite(cost) and cost > 0, 'Whole pilot cost missing')
        hashes[rel(response)] = sha(response.read_bytes())
        costs.append(cost)
    rates = {'pair': concise['cost_per_call']['pair'], 'whole': mean(costs)}
    return dict(concise=concise, whole_plan_sha256=sha(raw), whole_response_hashes=hashes,
                cost_per_call=rates)


def remaining_credit(key):
    req = Request(KEY_URL, headers={'Authorization': f'Bearer {key}'})
    with urlopen(req, timeout=25) as response:
        data = json.load(response)['data']
    limit = data.get('limit')
    remaining = data.get('limit_remaining')
    require(isinstance(limit, (int, float)) and 0 < limit <= 10
            and isinstance(remaining, (int, float)) and 0 <= remaining <= limit
            and data.get('limit_reset') is None, 'Use a dedicated API key capped at $10 total with no reset')
    return remaining


def outcome(response, case, template, family):
    error = response.get('error')
    if isinstance(error, dict):
        metadata = error.get('metadata', {})
        if isinstance(metadata, dict) and metadata.get('error_type') == 'content_policy_violation':
            return 'provider_blocked', None
        raise ValueError(f'API reply error {error.get("code", "unknown")}')
    require(response.get('service_tier') == 'flex' and response.get('provider') == 'Google AI Studio',
            'Unexpected production route')
    cost = response.get('usage', {}).get('cost')
    require(isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0, 'Missing billed cost')
    try:
        return 'valid', pilot.validate(response, case, template, family, paid=True, stronger=True)
    except (ValueError, KeyError, TypeError, AttributeError):
        return 'invalid_rubric', None


def call(group, family, rubrics, key):
    case, template, body = inputs(group, family, rubrics)
    req = Request(pilot.URL, data=body, headers={'Authorization': f'Bearer {key}',
                                                'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urlopen(req, timeout=120) as response:
        raw = response.read().decode('utf-8')
    reply = json.loads(raw)
    path = response_path(group, family)
    save(path, encode(dict(group=group['case_id'], era=group['era'], family=family,
                           request_sha256=sha(body), evidence_sha256=group['evidence'][family]['sha256'],
                           timestamp=datetime.now(timezone.utc).isoformat(),
                           elapsed_seconds=time.perf_counter()-start, raw_response=raw)))
    status, _ = outcome(reply, case, template, family)
    return status, reply.get('usage', {}).get('cost', 0)


def pending_jobs(data):
    return [(g, f) for g in data['groups'] for f in pilot.FAMILIES if not response_path(g, f).exists()]


def estimate(pending, rates):
    return sum(rates[f] for _, f in pending)


def run(data, limit):
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    require(key, 'Set OPENROUTER_API_KEY in this PowerShell window')
    gate = read(GATE)
    pilot_stats = production_pilot()
    require(gate.get('approved') is True and gate.get('pilot') == pilot_stats
            and gate.get('manifest_sha256') == sha(MANIFEST.read_bytes())
            and gate.get('quality_audit_sha256') == sha((OUT / 'PILOT_AUDIT.json').read_bytes()),
            'Awaiting verified pilot approval')
    pending = pending_jobs(data)
    rates = dict(pilot_stats['cost_per_call'])
    projected = estimate(pending, rates)
    if not pending:
        print('All 9,450 jobs have saved responses. Run score.')
        return
    active = OUT / 'RUNNING'
    try:
        active.touch(exist_ok=False)
    except FileExistsError:
        raise ValueError(f'Another run may be active. Check {active}') from None
    try:
        balance = remaining_credit(key)
        require(projected + RESERVE <= balance,
                f'Budget gate: projected ${projected:.3f} plus ${RESERVE:.2f} reserve; key has ${balance:.3f}')
        print(f'{len(pending)} pending, estimated ${projected:.3f}; key has ${balance:.3f}.', flush=True)
        selected = pending[:limit] if limit else pending
        recent_costs = defaultdict(list)
        counts = Counter()
        spent = 0.0
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=6) as pool:
            for offset in range(0, len(selected), 6):
                wave = selected[offset:offset + 6]
                if offset % 60 == 0:
                    balance = min(balance - spent, remaining_credit(key))
                    spent = 0.0
                    projected = estimate(pending[offset:], rates)
                    require(projected + RESERVE <= balance,
                            f'Paused for budget: projected ${projected:.3f}; key has ${balance:.3f}')
                futures = {pool.submit(call, g, f, data['rubrics'], key): (g, f) for g, f in wave}
                errors = []
                for future in as_completed(futures):
                    g, f = futures[future]
                    try:
                        status, cost = future.result()
                    except (OSError, ValueError) as error:
                        # HTTPError's string has the status but no authorization or prompt.
                        errors.append(f'{g["era"]}/{g["case_id"]}/{f}: {error}')
                        continue
                    spent += cost
                    recent_costs[f].append(cost)
                    counts[status] += 1
                    if len(recent_costs[f]) >= 12:
                        rates[f] = max(pilot_stats['cost_per_call'][f], mean(recent_costs[f][-120:]))
                if errors:
                    raise ValueError('Paused after request errors. Saved replies are resumable.\n' + '\n'.join(errors))
                done = sum(counts.values())
                if done >= 48:
                    require(counts['invalid_rubric'] / done <= .05,
                            f'Paused: {counts["invalid_rubric"]}/{done} rubric-invalid responses')
                if offset % 60 == 0 or offset + len(wave) == len(selected):
                    elapsed = time.perf_counter() - started
                    print(f'{done}/{len(selected)} saved ({counts["valid"]} valid, '
                          f'{counts["provider_blocked"]} blocked, {counts["invalid_rubric"]} rubric-invalid), '
                          f'{done * 60 / elapsed:.1f}/min', flush=True)
    finally:
        active.unlink()


def majority(values, default):
    label, n = Counter(values).most_common(1)[0]
    return label if n >= 2 else default


def gpt_reference():
    verify = read(REPLICATION / 'INDEPENDENT_VERIFICATION.json')
    require(verify['complete'] and verify['all_raw_sources_and_actual_reviewer_provenance_verified'],
            'Early GPT verification missing')
    pair = {ctx: {} for ctx in CONTEXTS}
    for item in verify['pair']['raw_sources']:
        rows = read(Path(item['path']), item['sha256'])['cases']
        by_id = index(rows)
        dest = pair[item['context']]
        require(not (dest.keys() & by_id.keys()), 'Duplicate early GPT pair case')
        dest.update(by_id)
    whole = index(read(old.ROOT / 'RESULTS.json', verify['whole']['result_sha256'])['group_outcomes'])
    late = defaultdict(dict)
    state = read(LATE / 'WORK_STATE.json')
    require(len(state['accepted']) == 1350 and not state['active'], 'Late GPT review incomplete')
    for receipt in state['accepted']:
        family, ctx, batch = (receipt[k] for k in ('family', 'context', 'batch'))
        rows = read(LATE / 'accepted' / family / ctx / f'{batch}.json', receipt['sha256'])['cases']
        dest = late[family, ctx]
        by_id = index(rows)
        require(not (dest.keys() & by_id.keys()), 'Duplicate late GPT case')
        dest.update(by_id)
    require(all(len(v) == 2100 for v in pair.values()) and len(whole) == 2025
            and len(late) == 6 and all(len(v) == 2700 for v in late.values()), 'GPT reference coverage differs')
    return pair, whole, late


def pair_codes(value):
    return {p['pair']: p for p in value['pairs']}


def audit_pilot():
    pilot_stats = short_pilot()
    _, _, late = gpt_reference()
    old_plan = read(pilot.OUT / 'STRONGER_FLEX_PLAN.json')
    old_jobs = {(j['case_id'], j['family']): j for j in old_plan['jobs']}
    current = read(pilot.OUT / 'SHORT_PILOT_PLAN.json')
    lock = read(LATE / 'REVIEW_LOCK.json', current['source_lock_sha256'])
    counts = defaultdict(Counter)
    for j in current['jobs']:
        cid, family = j['case_id'], j['family']
        case, template, _ = pilot.job_inputs(j, lock)
        previous, _ = pilot.checked(old_jobs[cid, family], case, template)
        new, error = pilot.checked(j, case, template)
        if previous is None or new is None:
            print(f'{cid}/{family}: comparison unavailable ({error})')
            continue
        votes = [late[family, ctx][cid] for ctx in CONTEXTS]
        if family == 'whole':
            for endpoint in ('coherence', 'specificity'):
                ref = majority([v[endpoint] for v in votes], 'unresolved_vote')
                counts[endpoint][ref, previous[endpoint], new[endpoint]] += 1
        else:
            for k in range(3):
                for endpoint in ('semantic', 'visual'):
                    ref = majority([pair_codes(v)[k][endpoint] for v in votes], 'unclear')
                    counts[endpoint][ref, pair_codes(previous)[k][endpoint], pair_codes(new)[k][endpoint]] += 1
    summary = {}
    for endpoint, values in counts.items():
        total = sum(values.values())
        before = sum(n for (g, a, b), n in values.items() if g == a)
        after = sum(n for (g, a, b), n in values.items() if g == b)
        summary[endpoint] = dict(compared=total, prior_matches=before, concise_matches=after,
                                 joint={f'{g}/{a}/{b}': n for (g, a, b), n in sorted(values.items())})
        print(f'{endpoint}: {before}/{total} prior vs {after}/{total} concise agreement')
    data = dict(pilot=pilot_stats, comparison=summary)
    path = OUT / 'PILOT_AUDIT.json'
    if path.exists():
        require(path.read_bytes() == encode(data), 'Existing pilot audit differs')
    else:
        save(path, encode(data))
    return pilot_stats


def score(data):
    require(not pending_jobs(data), 'All 9,450 responses are required before scoring')
    pair, whole, late = gpt_reference()
    bank = defaultdict(Counter)
    agreement = defaultdict(Counter)
    statuses = Counter()
    generation_ids = set()
    response_hashes = {}
    cost = 0.0
    for g in data['groups']:
        cid, bid = g['case_id'], g['bank_id']
        for family in pilot.FAMILIES:
            path = response_path(g, family)
            raw = path.read_bytes()
            row = json.loads(raw)
            response_hashes[rel(path)] = sha(raw)
            case, template, body = inputs(g, family, data['rubrics'])
            require((row['group'], row['era'], row['family'], row['request_sha256'], row['evidence_sha256'])
                    == (cid, g['era'], family, sha(body), g['evidence'][family]['sha256']),
                    'Stored response identity differs')
            reply = json.loads(row['raw_response'])
            status, value = outcome(reply, case, template, family)
            if status == 'valid':
                require(reply['id'] not in generation_ids, 'Duplicate generation ID')
                generation_ids.add(reply['id'])
            statuses[family, status] += 1
            cost += reply.get('usage', {}).get('cost', 0)
            tally = bank[bid]
            if family == 'pair':
                votes = [(pair[ctx] if g['era'] == 'early' else late['pair', ctx])[cid] for ctx in CONTEXTS]
                for endpoint in ('semantic', 'visual'):
                    for j in range(3):
                        gt = majority([next(p for p in v['pairs'] if p['pair'] == j)[endpoint] for v in votes], 'unclear')
                        actual = next(p for p in value['pairs'] if p['pair'] == j)[endpoint] if value else status
                        tally[endpoint, actual] += 1
                        agreement[endpoint][gt, actual] += 1
            else:
                if g['era'] == 'early':
                    gt, gs = (whole[cid][k] for k in ('majority', 'specificity_majority'))
                else:
                    votes = [late['whole', ctx][cid] for ctx in CONTEXTS]
                    gt, gs = [majority([v[k] for v in votes], 'unresolved_vote') for k in ('coherence', 'specificity')]
                for endpoint, reference in [('coherence', gt), ('specificity', gs)]:
                    actual = value[endpoint] if value else status
                    tally[endpoint, actual] += 1
                    agreement[endpoint][reference, actual] += 1
                if value and value['coherence'] == 'clear' and value['specificity'] in ('specific', 'broad_but_discriminating'):
                    tally['coherence', 'clear_and_discriminating'] += 1
    identity = {g['bank_id']: {k: g[k] for k in ('bank_id', 'seed', 'regime', 'method', 'fold')}
                for g in data['groups']}
    rows, by_seed = [], defaultdict(list)
    for bid, tally in sorted(bank.items()):
        for endpoint, planned in [('semantic', 225), ('visual', 225), ('coherence', 75), ('specificity', 75)]:
            require(sum(n for (e, code), n in tally.items() if e == endpoint and code != 'clear_and_discriminating')
                    == planned, 'Scored bank denominator differs')
        rates = dict(semantic=tally['semantic', 'yes']/225, visual=tally['visual', 'yes']/225,
                     whole_clear=tally['coherence', 'clear']/75,
                     whole_clear_and_discriminating=tally['coherence', 'clear_and_discriminating']/75)
        b = identity[bid]
        by_seed[b['regime'], b['method'], b['seed']].append(rates)
        rows.append(dict(b, rates=rates, counts={f'{e}/{c}': n for (e, c), n in sorted(tally.items())}))
    grouped = defaultdict(dict)
    for (regime, method, seed), rr in by_seed.items():
        grouped[regime, method][seed] = {m: mean(r[m] for r in rr) for m in rr[0]}
    methods = []
    for (regime, method), per_seed in sorted(grouped.items()):
        require(sorted(per_seed) == SEEDS, 'Missing method seeds')
        aggregate = {}
        for m in per_seed[21]:
            values = [per_seed[s][m] for s in SEEDS]
            aggregate[m] = dict(mean=mean(values), sample_sd=stdev(values))
        methods.append(dict(regime=regime, method=method, per_seed=per_seed, seven_order=aggregate))
    result = dict(model=pilot.STRONGER_MODEL, provider=pilot.STRONGER_PROVIDER,
                  manifest_sha256=sha(MANIFEST.read_bytes()), groups=4725, planned_calls=9450,
                  costs=cost, banks=rows, methods=methods,
                  response_hashes=response_hashes,
                  response_status={f'{f}/{s}': n for (f, s), n in sorted(statuses.items())},
                  agreement={k: {f'{a}/{b}': n for (a, b), n in sorted(v.items())}
                             for k, v in agreement.items()})
    path = OUT / 'RESULTS.json'
    raw = encode(result)
    if path.exists():
        require(path.read_bytes() == raw, 'Existing score differs')
    else:
        save(path, raw)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'status', 'budget', 'pilot-audit', 'run', 'score'))
    parser.add_argument('--limit', type=int, help='Maximum new requests this invocation')
    args = parser.parse_args()
    require(args.limit is None or 1 <= args.limit <= 9450, 'Invalid request limit')
    data = catalog()
    pending = pending_jobs(data)
    if args.action == 'prepare':
        for seed in SEEDS:
            group = next(g for g in data['groups'] if g['seed'] == seed)
            for family in pilot.FAMILIES:
                inputs(group, family, data['rubrics'])
        files = {e['path']: e['sha256'] for g in data['groups'] for e in g['evidence'].values()}
        for filename, digest in files.items():
            require(sha((pilot.WORK / filename).read_bytes()) == digest, f'Frozen card changed: {filename}')
        print(f'Frozen 4,725 groups, 63 banks and 9,450 jobs. Manifest: {MANIFEST}')
    elif args.action == 'status':
        print(f'{9450 - len(pending)}/9450 saved. Pilot approval: {GATE.exists()}.')
    elif args.action in ('budget', 'pilot-audit'):
        if args.action == 'pilot-audit':
            s = audit_pilot()
            print(f'Short pilot: {s["valid"]}, cost ${s["cost"]:.5f}.')
        s = production_pilot()
        print(f'Projected production inference: ${estimate(pending, s["cost_per_call"]):.3f} '
              f'plus ${RESERVE:.2f} reserve.')
        key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        if key and args.action == 'budget':
            print(f'OpenRouter key has ${remaining_credit(key):.3f} remaining.')
    elif args.action == 'run':
        run(data, args.limit)
        if not pending_jobs(data):
            score(data)
    else:
        score(data)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
