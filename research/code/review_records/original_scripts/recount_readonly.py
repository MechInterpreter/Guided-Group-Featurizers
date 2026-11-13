import hashlib,json,sys,statistics
from collections import Counter,defaultdict
from pathlib import Path

sys.path.insert(0,'scripts/ggf_all75')
import openrouter_coherence_full as f
from openrouter_coherence_pilot import require,save,encode

manifest=f.catalog()
pair_refs,whole_refs,late=f.gpt_reference()
rates=defaultdict(lambda: defaultdict(Counter))
match=defaultdict(Counter)
status=Counter()
costs=[]
ids=set()
hashes={}
invalid=Counter()
missing={'provider_blocked','invalid_rubric'}
for group in manifest['groups']:
    cid,bid,era=(group[k] for k in ('case_id','bank_id','era'))
    for family in ('pair','whole'):
        path=f.response_path(group,family)
        raw=path.read_bytes()
        hashes[f.rel(path)]=hashlib.sha256(raw).hexdigest()
        rec=json.loads(raw)
        source=group['evidence'][family]
        assert (rec['group'],rec['era'],rec['family'],rec['evidence_sha256'])==(cid,era,family,source['sha256'])
        response=json.loads(rec['raw_response'])
        label=None
        stage='valid'
        if response.get('error'):
            assert response['error']['metadata']['error_type']=='content_policy_violation'
            stage='provider_blocked'
        else:
            assert response['model']==f.pilot.STRONGER_MODEL
            assert response['provider']=='Google AI Studio' and response['service_tier']=='flex'
            assert response['id'] and response['id'] not in ids
            ids.add(response['id'])
            costs.append(response['usage']['cost'])
            c=source['case']
            props=f.pilot.paid_schema(c,family)['properties']
            template=dict.fromkeys(props)
            if family=='pair':
                template['pairs']=[dict.fromkeys(props['pairs']['items']['properties'])]
            try:
                label=f.pilot.validate(response,c,template,family,paid=True,stronger=True)
            except (ValueError,TypeError,KeyError,AttributeError) as e:
                stage='invalid_rubric'
                invalid[family,str(e)]+=1
        status[family,stage]+=1
        bank=rates[bid]
        def add(endpoint,gpt,gemini):
            bank[endpoint][gemini]+=1
            bank['gpt_'+endpoint][gpt]+=1
            match[endpoint][gpt,gemini]+=1
        def vote(labels,fallback):
            c=Counter(labels)
            return next((k for k,n in c.items() if n>=2),fallback)
        if family=='pair':
            refs=[(pair_refs[ctx] if era=='early' else late[family,ctx])[cid] for ctx in f.CONTEXTS]
            model={x['pair']:x for x in label['pairs']} if label else {}
            for endpoint in ('semantic','visual'):
                for pair in range(3):
                    gt=vote([next(x for x in v['pairs'] if x['pair']==pair)[endpoint] for v in refs],'unclear')
                    add(endpoint,gt,model[pair][endpoint] if label else stage)
        else:
            if era=='early':
                ref=whole_refs[cid]
                gt,gts,gtd=(ref[k] for k in ('majority','specificity_majority','clear_and_discriminating'))
            else:
                refs=[late[family,ctx][cid] for ctx in f.CONTEXTS]
                gt=vote([v['coherence'] for v in refs],'unresolved_vote')
                gts=vote([v['specificity'] for v in refs],'unresolved_vote')
                gtd=gt=='clear' and sum(v['specificity'] in ('specific','broad_but_discriminating') for v in refs)>=2
            add('coherence',gt,label['coherence'] if label else stage)
            add('specificity',gts,label['specificity'] if label else stage)
            add('clear_discriminating','yes' if gtd else 'no',
                ('yes' if label['coherence']=='clear' and label['specificity'] in ('specific','broad_but_discriminating')
                 else 'no') if label else stage)
assert sum(status.values())==9450
assert len(hashes)==9450 and len(rates)==63
files=list((f.OUT/'responses').rglob('*.json'))
assert {f.rel(path) for path in files}==set(hashes)
identity={g['bank_id']:{k:g[k] for k in ('bank_id','seed','regime','method','fold')} for g in manifest['groups']}
endpoints={'semantic':('yes',225),'visual':('yes',225),'coherence':('clear',75),'clear_discriminating':('yes',75)}
grouped=defaultdict(list)
banks=[]
for bid,bank in sorted(rates.items()):
    meta=identity[bid]
    row=dict(meta,metrics={})
    for e,(positive,denom) in endpoints.items():
        c=bank[e]; gt=bank['gpt_'+e]
        assert sum(c.values())==sum(gt.values())==denom
        n=sum(c[k] for k in missing)
        row['metrics'][e]=dict(lower=c[positive]/denom,upper=(c[positive]+n)/denom,
                              valid_only=c[positive]/(denom-n),missing=n/denom,gpt=gt[positive]/denom)
    row['counts']={e:dict(c) for e,c in bank.items()}
    grouped[meta['regime'],meta['method'],meta['seed']].append(row['metrics'])
    banks.append(row)
merged=defaultdict(dict)
for (regime,method,seed),folds in grouped.items():
    merged[regime,method][seed]={e:{key:statistics.mean(t[e][key] for t in folds) for key in folds[0][e]} for e in endpoints}
methods=[]
for (regime,method),per_seed in sorted(merged.items()):
    assert sorted(per_seed)==list(range(21,28))
    summary={}
    for e in endpoints:
        summary[e]={key:statistics.mean(t[e][key] for t in per_seed.values()) for key in per_seed[21][e]}
        summary[e]['sample_sd']=statistics.stdev(t[e]['lower'] for t in per_seed.values())
    methods.append(dict(regime=regime,method=method,per_seed=per_seed,summary=summary))
    print(regime,method,[(e,round(m['lower']*100,2),round(m['sample_sd']*100,2),round(m['upper']*100,2),round(m['gpt']*100,2)) for e,m in summary.items()])
agreement={}
for e,t in match.items():
    common=sum(n for (g,a),n in t.items() if a not in missing)
    agrees=sum(n for (g,a),n in t.items() if a==g)
    agreement[e]=dict(exact_matches=agrees,comparable=common,planned=sum(t.values()),
                      confusion={f'{g}/{a}':n for (g,a),n in sorted(t.items())})
    print(e,'agreement',agrees,common,'of',sum(t.values()))
audit=dict(complete=True,manifest_sha256=f.sha(f.MANIFEST.read_bytes()),groups=4725,planned_calls=9450,
           billed_production=statistics.fsum(costs) if hasattr(statistics,'fsum') else sum(costs),
           statuses={f'{family}/{s}':n for (family,s),n in status.items()},
           invalid_reasons={f'{family}/{why}':n for (family,why),n in invalid.items()},
           response_sha256=hashes,agreement=agreement,banks=banks,methods=methods)
output=f.OUT/'INDEPENDENT_RECOUNT.json'
if output.exists():
    require(output.read_bytes()==encode(audit),'Existing independent recount differs')
else:
    save(output,encode(audit))
print('validity',dict(status),'invalid reasons',dict(invalid),'billed',audit['billed_production'])
print(output)
