"""Review a fixed, blinded 24-card pilot through the free Gemma endpoint."""

import argparse
import base64
import hashlib
import html
import io
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from full_review_ready import validate_batch
from group_review import CODES as WHOLE_CODES, SPEC as WHOLE_SPEC
from replicate_group_review import validate_case


WORK = Path(__file__).resolve().parents[2]
SOURCE = WORK / 'output/ggf_all75_seeds24_27/coherence_review'
PUBLIC = SOURCE / 'reviewer_public'
OUT = WORK / 'output/ggf_all75_openrouter_pilot'
PLAN = OUT / 'PLAN.json'
MODEL = 'google/gemma-4-26b-a4b-it:free'
PROVIDER = 'google-ai-studio'
PAID_MODEL = 'google/gemini-2.5-flash-lite'
STRONGER_MODEL = 'google/gemini-3.5-flash-lite'
PAID_PROVIDER = 'google-ai-studio'
STRONGER_PROVIDER = 'google-ai-studio/flex'
URL = 'https://openrouter.ai/api/v1/chat/completions'
FAMILIES = ('pair', 'whole')
SALT = 'ggf-openrouter-pilot-v1'
SHORT_CARDS = ('G1837', 'G0611', 'G2321', 'G1272', 'G0768', 'G1091', 'G2437', 'G1980')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')


def save(path, raw, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(raw)
    try:
        if replace:
            os.replace(tmp, path)
        elif os.name == 'nt':
            os.rename(tmp, path)
        else:
            os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def select_cards(lock):
    """Choose before any outcomes are read. Only opaque IDs leave this function."""
    raw = (SOURCE / 'PRIVATE.json').read_bytes()
    require(sha(raw) == lock['private_sha256'], 'Frozen sampling map changed')
    study = json.loads(raw)
    by_bank = defaultdict(list)
    for case in study['cases']:
        by_bank[case['bank_id']].append(case['case_id'])
    banks = {row['bank_id']: row for row in study['banks']}
    require(len(banks) == 36 and set(banks) == set(by_bank), 'Expected the 36 matched banks')
    picked = []
    for i, seed in enumerate(range(24, 28)):
        # Each order contributes three full and three split banks. Rotate the
        # omitted full method and alternate the two supervised split folds.
        for j, method in enumerate(('ggf', 'slics', 'supervised_vanilla', 'vanilla')):
            if i != j:
                picked.append((seed, 'full_category', method, 0))
        picked.extend((seed, 'category_split', method, fold) for method, fold in
                      [('ggf', i % 2), ('supervised_vanilla', (i + 1) % 2), ('vanilla', -1)])
    ids = []
    for seed, regime, method, fold in picked:
        matches = [bid for bid, b in banks.items()
                   if (b['seed'], b['regime'], b['kind']) == (seed, regime, method)
                   and (fold == -1 or b['fold'] == fold)]
        require(len(matches) == 1 and len(by_bank[matches[0]]) == 75, 'Unexpected bank census')
        ids.append(min(by_bank[matches[0]], key=lambda cid: sha(f'{SALT}/{cid}'.encode())))
    require(len(set(ids)) == 24, 'Unexpected duplicate pilot card')
    return sorted(ids, key=lambda cid: sha(f'{SALT}/order/{cid}'.encode()))


def evidence(cid, family, lock):
    index = int(cid[1:]) - 1
    require(cid == f'G{index + 1:04d}' and 0 <= index < 2700, 'Invalid opaque card ID')
    rel = f'{family}/context1/B{index // 12:03d}.json'
    raw = (PUBLIC / rel).read_bytes()
    require(sha(raw) == lock['packets'][rel], 'Frozen packet changed')
    packet = json.loads(raw)
    case = packet['cases'][index % 12]
    template = packet['output_template']['cases'][index % 12]
    require(case['case_id'] == template['case_id'] == cid, 'Packet case order changed')
    fields = ('pair_ids', 'invalid_pairs') if family == 'pair' else ('valid_high_views', 'valid_high_count')
    case = {k: case[k] for k in ('case_id', *fields)}
    card = (PUBLIC / 'cards' / f'{cid}.png').read_bytes()
    require(sha(card) == lock['cards'][f'cards/{cid}.png'], 'Frozen PNG changed')
    return case, template, packet['instructions'], card


def paid_schema(case, family):
    def obj(**properties):
        return dict(type='object', properties=properties, required=list(properties),
                    additionalProperties=False)

    def enum(*values):
        return dict(type='string', enum=list(values))

    text = dict(type='string')
    case_id = enum(case['case_id'])
    if family == 'pair':
        verdict = enum('yes', 'no', 'unclear')
        pair = obj(pair=dict(type='integer', minimum=0, maximum=2),
                   semantic=verdict, visual=verdict, shared_meaning=text, exceptions=text)
        return obj(case_id=case_id, pairs=dict(type='array', items=pair, minItems=3, maxItems=3))
    view = dict(type='integer', minimum=1, maximum=6)
    return obj(case_id=case_id, theme=text, fit_views=dict(type='array', items=view),
               high_fit_count=dict(type='integer', minimum=0, maximum=6),
               exceptions=dict(type='array', items=obj(view=view, reason=text)),
               coherence=enum(*WHOLE_CODES), specificity=enum(*WHOLE_SPEC),
               confidence=enum('high', 'medium', 'low'), high_low_contrast=enum('not_assessed'),
               observations=dict(type='string', description='Nonempty evidence about the numbered marked regions.'))


def request_body(case, template, rubric, card, family, *, paid=False, stronger=False, concise=False):
    # From prepare_evaluation_reviews.card: six 224px columns. The crops
    # include the numbered views, their marks and every zoom strip.
    with Image.open(io.BytesIO(card)) as image:
        require(image.size == (1344, 364), 'Unexpected card geometry')
        images = [card]
        for start in (0, 448, 896):
            crop = image.crop((start, 28, start + 448, 364))
            buf = io.BytesIO()
            crop.save(buf, format='PNG')
            images.append(buf.getvalue())
    explanation = (
        'Use the rubric above. This API call is one fresh context for one case. '
        'The first image is the full card. The next three show views 1–2, 3–4, and 5–6 '
        'with their zoom strips at the original pixel scale. They repeat the same views. '
        'Return exactly one JSON object matching the case template below. '
        'The local runner adds reviewer identity, provenance and file storage.'
    )
    if family == 'pair':
        explanation += (
            ' Pair IDs 0, 1 and 2 correspond to views 1–2, 3–4 and 5–6. '
            'For every pair, give a concrete meaning or explain the exception. '
            'If marked detail is unreadable, keep the relevant judgment unclear.'
        )
    else:
        explanation += ' Format each exception as {"view": 1, "reason": "concrete explanation"}.'
    if paid:
        explanation += (
            ' Inspect each numbered marked region and its zoom before deciding. '
            'A recognizable object in the surrounding photo is insufficient evidence. '
            'Only name an exact letter, part or texture as shared if the marks in both views support it. '
            'Describe what the marks actually show. Where they differ or are unreadable, explain that uncertainty.'
        )
        if family == 'pair':
            explanation += ' For each pair, cite the visible evidence in both numbered views in shared_meaning or exceptions.'
        else:
            explanation += ' Fill observations with concrete support from the numbered marked views, even when every view fits.'
    if stronger:
        explanation += (
            ' The pink squares, numbering, borders and layout are review overlays. '
            'They never count as a shared feature or theme. Judge the image content beneath the squares '
            'using the zoom strips. Scene context can help identify the content within the marks.'
        )
        if family == 'pair':
            explanation += (
                ' A visual yes needs a specific common visual feature supported in both sets of marked regions. '
                'If the marked appearances differ and there is no such feature, record visual=no. '
                'If evidence is unreadable, record unclear. State the actual shared feature for each yes.'
            )
        else:
            explanation += (
                ' Choose the fitting valid views first, then apply the frozen fit-count rule. '
                'Use one theme across those views and record an exception for every other valid view.'
            )
    if concise:
        if family == 'pair':
            explanation += (
                ' Judge semantic and visual coherence independently. A specifically named shared part, '
                'material or region qualifies semantically when visible in both sets of marks, even '
                'if the parent objects, actions or settings differ. Use scene context only to identify '
                'the marked content. If your own evidence establishes such a match, record semantic=yes. '
                'Keep shared_meaning and exceptions to at most 8 words each. Use an empty string '
                'for an inapplicable field. A yes on either axis needs a short concrete shared feature.'
            )
        else:
            explanation += (
                ' Keep theme to at most 8 words, observations to at most 12 words, and each '
                'exception reason to at most 5 words. Record each nonmatching valid view.'
            )
        explanation += ' Use compact JSON and follow the supplied JSON schema.'
    text = f'{rubric}\n\n{explanation}\n\nCase metadata\n{json.dumps(case)}'
    if not concise:
        text += f'\n\nCase template\n{json.dumps(template)}'
    content = [{'type': 'text', 'text': text}]
    for raw in images:
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')}})
    body = {
        'model': MODEL,
        'provider': {'only': [PROVIDER], 'allow_fallbacks': False},
        'temperature': 0,
        'max_tokens': 2048,
        'response_format': {'type': 'json_object'},
        'messages': [{'role': 'user', 'content': content}],
    }
    if paid:
        body['model'] = PAID_MODEL
        body['provider'] = {
            'only': [PAID_PROVIDER],
            'allow_fallbacks': False,
            'max_price': {'prompt': 0.10, 'completion': 0.40},
        }
        body['response_format'] = {
            'type': 'json_schema',
            'json_schema': {'name': f'blinded_{family}_review', 'strict': True,
                            'schema': paid_schema(case, family)},
        }
        body['usage'] = {'include': True}
    if stronger:
        body['model'] = STRONGER_MODEL
        body['provider']['only'] = [STRONGER_PROVIDER]
        body['provider']['max_price'] = {'prompt': 0.15, 'completion': 1.25}
        body['service_tier'] = 'flex'
        body['reasoning'] = {'effort': 'minimal'}
    return encode(body)


def job_inputs(job, lock):
    case, template, rubric, card = evidence(job['case_id'], job['family'], lock)
    body = request_body(case, template, rubric, card, job['family'], paid=job.get('paid', False),
                        stronger=job.get('stronger', False), concise=job.get('concise', False))
    return case, template, body


def prepare(lock, lock_hash):
    jobs = []
    for cid in select_cards(lock):
        for family in FAMILIES:
            job = {'case_id': cid, 'family': family}
            _, _, body = job_inputs(job, lock)
            jobs.append(dict(job, request_sha256=sha(body), request_bytes=len(body)))
    plan = encode(dict(model=MODEL, provider=PROVIDER, source_lock_sha256=lock_hash,
                       seeds=[24, 25, 26, 27], cards=24, jobs=jobs))
    if PLAN.exists():
        require(PLAN.read_bytes() == plan, 'Pilot plan already exists with different inputs')
    else:
        save(PLAN, plan)
    print(f'Frozen 24 cards and 48 offline-checked requests. Plan: {PLAN}')


def validate(response, case, template, family, *, paid=False, stronger=False):
    require(isinstance(response.get('id'), str) and response['id'], 'Missing generation ID')
    model = response.get('model', '').removesuffix(':free')
    expected = STRONGER_MODEL if stronger else PAID_MODEL if paid else MODEL.removesuffix(':free')
    require(model == expected, 'Unexpected response model')
    choices = response.get('choices', [])
    require(len(choices) == 1 and choices[0].get('finish_reason') == 'stop', 'Incomplete API response')
    result = json.loads(choices[0]['message']['content'])
    require(isinstance(result, dict) and result.keys() == template.keys()
            and result.get('case_id') == case['case_id'], 'Case or schema differs from template')
    if family == 'whole':
        validate_case(result, case)
        require(all(isinstance(ex, dict) and isinstance(ex.get('reason'), str)
                    and ex['reason'].strip() for ex in result['exceptions']), 'Exception reasons required')
    else:
        validate_batch(dict(public_sha256='pilot', method_blind=True, reviewer=response['id'],
                            cases=[result]), {'cases': [case]}, {'cases': [case]},
                       {'public_sha256': 'pilot'})
        require(all(p.keys() == template['pairs'][0].keys() and
                    all(isinstance(p[k], str) for k in ('shared_meaning', 'exceptions')) and
                    (p['shared_meaning'].strip() or p['exceptions'].strip())
                    for p in result['pairs']), 'Pair evidence or schema missing')
    return result


def raw_path(job):
    folder = 'flash_lite_smoke_v2_responses' if job.get('paid') else 'responses'
    if job.get('stronger'):
        folder = 'flash_35_lite_flex_responses'
    if job.get('concise'):
        folder = 'flash_35_lite_concise_responses'
    return OUT / folder / job['family'] / f"{job['case_id']}.json"


def review_one(job, lock, key):
    case, template, body = job_inputs(job, lock)
    require(sha(body) == job['request_sha256'], 'Frozen API request changed')
    request = Request(URL, data=body, headers={
        'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=150) as reply:
            raw = reply.read().decode('utf-8')
    except HTTPError as e:
        retry = e.headers.get('Retry-After', '')
        wait = f' Retry-After: {retry} seconds.' if retry.isdecimal() else ''
        detail = ''
        try:
            message = json.loads(e.read(8192))['error']['message']
            if isinstance(message, str):
                message = ' '.join(message.replace(key, '[redacted]').split())
                if 'base64,' not in message and 'sk-or-' not in message:
                    detail = f' {message[:300]}'
        except (ValueError, KeyError, TypeError):
            pass
        raise ValueError(f'OpenRouter HTTP {e.code} for {job["case_id"]}/{job["family"]}. '
                         f'Completed requests are saved.{wait}{detail} Check Activity before rerunning.') from None
    except (URLError, TimeoutError):
        raise ValueError('OpenRouter connection failed. Check Activity before rerunning.') from None
    # Preserve the exact response text even when it fails local validation.
    save(raw_path(job), encode(dict(job, recorded_at=datetime.now(timezone.utc).isoformat(),
                                  elapsed_seconds=round(time.perf_counter() - started, 3),
                                  raw_response=raw)))
    label, error = checked(job, case, template)
    print(f'{job["case_id"]}/{job["family"]} ' + ('valid' if label is not None else f'invalid ({error})'),
          flush=True)
    return label is not None


def run(jobs, lock, limit, *, parallel=False):
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    require(key, 'Set OPENROUTER_API_KEY in your local shell first')
    active = OUT / 'RUNNING'
    try:
        active.touch(exist_ok=False)
    except FileExistsError:
        raise ValueError(f'Another run may be active. Check before removing {active}') from None
    try:
        pending = [job for job in jobs if not raw_path(job).exists()]
        if parallel and pending:
            started = time.perf_counter()
            accepted = 0
            selected = pending[:limit]
            with ThreadPoolExecutor(max_workers=6) as pool:
                for offset in range(0, len(selected), 6):
                    futures = [pool.submit(review_one, job, lock, key)
                               for job in selected[offset:offset + 6]]
                    errors = []
                    for future in as_completed(futures):
                        try:
                            accepted += future.result()
                        except ValueError as error:
                            errors.append(str(error))
                    if errors:
                        raise ValueError("\n".join(errors))
            elapsed = time.perf_counter() - started
            print(f'{accepted}/{len(selected)} new schema-valid judgments in {elapsed:.1f}s '
                  f'({accepted * 60 / elapsed:.1f} per minute).')
        else:
            for i, job in enumerate(pending[:limit]):
                if not review_one(job, lock, key):
                    print('Paused on the first invalid response. Review the local report before continuing.')
                    break
                if i + 1 < min(limit, len(pending)):
                    time.sleep(3.1)
    finally:
        active.unlink()


def checked(job, case, template):
    raw = json.loads(raw_path(job).read_bytes())
    require(all(raw.get(k) == v for k, v in job.items()), 'Saved response has wrong request identity')
    try:
        response = json.loads(raw['raw_response'])
        if response.get('error'):
            return None, f"Provider error {response['error'].get('code', 'unknown')}"
        return validate(response, case, template, job['family'], paid=job.get('paid', False),
                        stronger=job.get('stronger', False)), None
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return None, str(e)


def report(jobs, lock, *, paid=False, pilot=False, stronger=False, concise=False):
    counts = Counter()
    costs = []
    sections = {}
    for job in jobs:
        cid, family = job['case_id'], job['family']
        section = sections.setdefault(cid, [])
        path = raw_path(job)
        if not path.exists():
            counts[family, 'pending'] += 1
            section.append(f'<h3>{family}</h3><p>Pending</p>')
            continue
        case, template, _ = job_inputs(job, lock)
        label, error = checked(job, case, template)
        counts[family, 'valid' if label is not None else 'invalid'] += 1
        data = json.loads(path.read_bytes())
        try:
            response = json.loads(data['raw_response'])
            metadata = {k: response.get(k) for k in ('id', 'model', 'provider', 'usage')}
        except (ValueError, AttributeError):
            metadata = {}
        usage = metadata.get('usage')
        if isinstance(usage, dict) and isinstance(usage.get('cost'), (int, float)):
            costs.append(usage['cost'])
        metadata['elapsed_seconds'] = data.get('elapsed_seconds')
        text = json.dumps(dict(metadata=metadata, label=label, error=error), ensure_ascii=False, indent=2)
        section.append(f'<h3>{family}</h3><pre>{html.escape(text)}</pre>'
                       f'<a href="{path.relative_to(OUT).as_posix()}">Raw API response</a>')
    summary = ' · '.join(f'{family}: ' + ', '.join(f'{counts[family, s]} {s}'
                         for s in ('valid', 'invalid', 'pending')) for family in FAMILIES)
    title = 'Blinded Gemma pilot'
    if paid:
        title = 'Blinded Flash-Lite pilot' if pilot else 'Blinded Flash-Lite smoke test'
    if stronger:
        title = 'Blinded Gemini 3.5 Flash-Lite pilot'
    if concise:
        title = 'Blinded concise-evidence diagnostic pilot'
    billed = f'Reported cost so far: ${sum(costs):.6f} across {len(costs)} replies.' if costs else ''
    parts = [f'<!doctype html><meta charset="utf-8"><title>{title}</title>',
             '<style>body{font:16px system-ui;margin:2rem}pre{white-space:pre-wrap}'
             'img{max-width:100%}section{border-top:1px solid #bbb;margin:2rem 0}</style>',
             f'<h1>{title}</h1>', f'<p>{summary}</p>', f'<p>{billed}</p>',
             '<p>Valid means the response passed the rubric schema. Inspect the marked evidence '
             'against the stated meanings, matching views and exceptions before scaling.</p>']
    for cid, section in sections.items():
        card = Path(os.path.relpath(PUBLIC / 'cards' / f'{cid}.png', OUT)).as_posix()
        parts += [f'<section><h2>{cid}</h2><a href="{card}"><img loading="lazy" '
                  f'src="{card}" alt="Blinded card {cid}"></a>', *section, '</section>']
    filename = 'PAID_PILOT_REPORT.html' if pilot else ('PAID_SMOKE_V2_REPORT.html' if paid else 'PILOT_REPORT.html')
    if stronger:
        filename = 'STRONGER_PILOT_REPORT.html'
    if concise:
        filename = 'SHORT_PILOT_REPORT.html'
    path = OUT / filename
    save(path, '\n'.join(parts).encode('utf-8'), replace=True)
    print(summary)
    if billed:
        print(billed)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'report'))
    parser.add_argument('--limit', type=int, default=2, help='Maximum new API requests for run')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--paid-smoke', action='store_true',
                        help='Opt in to up to two paid Gemini Flash-Lite calls on the first card')
    modes.add_argument('--paid-pilot', action='store_true',
                       help='Opt in to the 24-card Flash-Lite pilot, with up to six concurrent calls')
    modes.add_argument('--stronger-pilot', action='store_true',
                       help='Opt in to the 24-card Gemini 3.5 Flash-Lite pilot')
    modes.add_argument('--short-pilot', action='store_true',
                       help='Opt in to the 8-card concise-evidence diagnostic')
    args = parser.parse_args()
    stronger = args.stronger_pilot or args.short_pilot
    paid = args.paid_smoke or args.paid_pilot or stronger
    parallel = args.paid_pilot or stronger
    require(not paid or args.action != 'prepare', 'Paid options support run or report')
    require(1 <= args.limit <= 48, '--limit must be between 1 and 48')
    require(not args.paid_smoke or (args.action != 'prepare' and args.limit <= 2),
            '--paid-smoke supports run or report and at most two calls')
    raw = (SOURCE / 'REVIEW_LOCK.json').read_bytes()
    lock, lock_hash = json.loads(raw), sha(raw)
    if args.action == 'prepare':
        prepare(lock, lock_hash)
        return
    plan = json.loads(PLAN.read_bytes())
    require(plan['source_lock_sha256'] == lock_hash and plan['model'] == MODEL
            and plan['provider'] == PROVIDER, 'Pilot provenance differs from plan')
    expected = [(cid, family) for cid in select_cards(lock) for family in FAMILIES]
    jobs = plan['jobs']
    require([(job['case_id'], job['family']) for job in jobs] == expected, 'Pilot queue changed')
    if args.short_pilot:
        by_case = {(job['case_id'], job['family']): job for job in jobs}
        jobs = [by_case[cid, family] for cid in SHORT_CARDS for family in FAMILIES]
    if paid:
        paid_jobs = []
        for original in (jobs if parallel else jobs[:2]):
            _, _, free_body = job_inputs(original, lock)
            require(sha(free_body) == original['request_sha256'], 'Frozen smoke-test evidence changed')
            job = dict(original, paid=True)
            if stronger:
                job['stronger'] = True
            if args.short_pilot:
                job['concise'] = True
            _, _, body = job_inputs(job, lock)
            paid_jobs.append(dict(job, request_sha256=sha(body), request_bytes=len(body)))
        jobs = paid_jobs
    if stronger:
        path = OUT / ('SHORT_PILOT_PLAN.json' if args.short_pilot else 'STRONGER_FLEX_PLAN.json')
        frozen = encode(dict(model=STRONGER_MODEL, provider=STRONGER_PROVIDER,
                             source_lock_sha256=lock_hash, jobs=jobs))
        if path.exists():
            require(path.read_bytes() == frozen, 'Stronger pilot plan already exists with different inputs')
        else:
            save(path, frozen)
    if args.action == 'run':
        if args.paid_pilot:
            for job in jobs[:2]:
                case, template, _ = job_inputs(job, lock)
                require(raw_path(job).exists() and checked(job, case, template)[0] is not None,
                        'Complete both schema-valid Flash-Lite smoke calls first')
        try:
            run(jobs, lock, args.limit, parallel=parallel)
        finally:
            report(jobs, lock, paid=paid, pilot=parallel, stronger=stronger, concise=args.short_pilot)
    else:
        report(jobs, lock, paid=paid, pilot=parallel, stronger=stronger, concise=args.short_pilot)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from None
