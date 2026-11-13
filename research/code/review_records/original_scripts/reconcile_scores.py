import hashlib,json,math
from pathlib import Path
p=Path(__file__).resolve().parent
independent_raw=(p/'INDEPENDENT_RECOUNT.json').read_bytes()
official_raw=(p/'RESULTS.json').read_bytes()
a,b=json.loads(independent_raw),json.loads(official_raw)
assert a['manifest_sha256']==b['manifest_sha256']
assert a['response_sha256']==b['response_hashes']
assert a['statuses']==b['response_status']
assert (a['groups'],a['planned_calls'])==(b['groups'],b['planned_calls'])
assert math.isclose(a['billed_production'],b['costs'],abs_tol=1e-10)
keys={'semantic':'semantic','visual':'visual','whole_clear':'coherence','whole_clear_and_discriminating':'clear_discriminating'}
bb={r['bank_id']:r for r in b['banks']}
for row in a['banks']:
    for formal,short in keys.items():
        assert math.isclose(row['metrics'][short]['lower'],bb[row['bank_id']]['rates'][formal],abs_tol=1e-12)
for endpoint in ('semantic','visual','coherence','specificity'):
    assert a['agreement'][endpoint]['confusion']==b['agreement'][endpoint]
bm={(r['regime'],r['method']):r for r in b['methods']}
for r in a['methods']:
    match=bm[r['regime'],r['method']]
    for formal,short in keys.items():
        for seed,scores in r['per_seed'].items():
            assert math.isclose(scores[short]['lower'],match['per_seed'][seed][formal],abs_tol=1e-12)
        assert math.isclose(r['summary'][short]['lower'],match['seven_order'][formal]['mean'],abs_tol=1e-12)
        assert math.isclose(r['summary'][short]['sample_sd'],match['seven_order'][formal]['sample_sd'],abs_tol=1e-12)
print('Independent and production scoring match for 9,450 raw hashes, 63 banks, seven seeds, four endpoints, and billed usage.')
result=dict(complete=True,independent_sha256=hashlib.sha256(independent_raw).hexdigest(),
    production_sha256=hashlib.sha256(official_raw).hexdigest(),response_hashes=9450,banks=63,seeds=7)
with (p/'SCORE_VERIFICATION.json').open('x',encoding='utf8') as h:
    json.dump(result,h,indent=2)
