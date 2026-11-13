"""Frozen rubric validators extracted from the original scorer. No networking."""

import json,math,re

PAIR_CODES=('yes','no','unclear')

WHOLE_CODES=('clear','partial','no_clear_theme','insufficient_evidence')

WHOLE_SPEC=('specific','broad_but_discriminating','generic_nondiscriminating','unclear')

MODEL='google/gemma-4-26b-a4b-it:free'

PAID_MODEL='google/gemini-2.5-flash-lite'

STRONGER_MODEL='google/gemini-3.5-flash-lite'

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
                if pair.get(endpoint) not in PAIR_CODES:
                    raise ValueError('Unfilled or invalid research judgment')
                if by_id[cid]['invalid_pairs'][pair['pair']] and pair[endpoint] != 'unclear':
                    raise ValueError('Missing evidence must remain unclear')
                if pair[endpoint] == 'yes' and not pair.get('shared_meaning', '').strip():
                    raise ValueError('Positive judgments require concrete meaning')

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
    require(v.get('coherence') in WHOLE_CODES and v.get('specificity') in WHOLE_SPEC and
            v.get('confidence') in ('high', 'medium', 'low'), 'Invalid label')
    require(v.get('high_low_contrast') == 'not_assessed', 'No LOW evidence supplied')
    require(isinstance(v.get('theme'), str) and len(v['theme'].split()) <= 30 and
            isinstance(v.get('observations'), str) and v['observations'].strip(), 'Theme/rationale missing or too long')
    if v['coherence'] != 'insufficient_evidence':
        require(v['coherence'] == expected_label(c['valid_high_count'], len(fit)), 'Five-of-six rubric mismatch')
        require(c['valid_high_count'] >= 4, 'Fewer than four valid HIGH views')
        if len(fit) >= math.ceil(c['valid_high_count']/2):
            require(v['theme'].strip(), 'Positive motif must be stated')

def require(condition, message):
    if not condition:
        raise ValueError(message)

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
