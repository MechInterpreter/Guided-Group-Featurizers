"""Partial/final readout with explicit missing endpoints and paired method effects."""
import argparse,html,json
from pathlib import Path
import numpy as np
from eval_common import *

def mean(values):
    v=[float(x) for x in values if x is not None and np.isfinite(x)];return float(np.mean(v)) if v else None
def fmt(v):return 'pending / unsupported' if v is None else f'{v:.1%}'
def summarize_numeric(root):
    path=Path(root)/'NUMERIC_COMPLETE.json'
    if not path.exists():return []
    inputs=read(Path(root)/'INPUTS.json');folds={f['fold']:f for f in inputs['folds']};rows=[]
    for bank in read(path)['results']:
        for f in bank['evaluation_folds']:
            for stratum in (['all_categories'] if bank['regime']=='full_category' else ['excluded','guided']):
                cats=inputs['categories'] if stratum=='all_categories' else folds[f]['excluded' if stratum=='excluded' else 'guide']
                rr=[r for r in bank['categories'] if r['category_index'] in cats]
                rows.append(dict(bank_id=bank['bank_id'],regime=bank['regime'],kind=bank['kind'],seed=bank['seed'],fold=f,stratum=stratum,
                    capacity_matched=bank['capacity_matched'],planned_categories=len(cats),AP_supported=sum(r['patch_AP'] is not None for r in rr),
                    mean_AP_supported=mean([r['patch_AP'] for r in rr]),mean_regional_Capture_supported=mean([r['regional_Capture'] for r in rr]),
                    AP_by_category={str(r['category_index']):r['patch_AP'] for r in rr}))
    return rows

def aggregate_numeric(rows):
    groups={}
    for row in rows:groups.setdefault((row['regime'],row['kind'],row['stratum'],row['capacity_matched']),[]).append(row)
    out=[]
    for (regime,kind,stratum,matched),rr in sorted(groups.items()):
        # Equal category weight after fold aggregation within each training
        # order; folds37/38 must not get equal weight per category accidentally.
        seeds={}
        for r in rr:
            for c,val in r['AP_by_category'].items():seeds.setdefault(r['seed'],{}).setdefault(c,[]).append(val)
        per_seed={str(s):mean([mean(v) for v in cc.values()]) for s,cc in seeds.items()}
        out.append(dict(regime=regime,kind=kind,stratum=stratum,capacity_matched=matched,training_orders=sorted(seeds),
            mean_AP_supported=mean(per_seed.values()),per_order_AP=per_seed,
            planned_category_fold_cells=sum(r['planned_categories'] for r in rr),supported_category_fold_cells=sum(r['AP_supported'] for r in rr),
            dependence='Shared Vanilla checkpoint across category folds is not an additional trained replicate'))
    return out

def concentration_summary(root):
    path=Path(root)/'CONCENTRATION_COMPLETE.json'
    if not path.exists():return []
    collection=read(Path(root)/'COLLECTION_COMPLETE.json');banks={r['bank_id']:r for r in collection['banks']};inputs=read(Path(root)/'INPUTS.json');folds={r['fold']:r for r in inputs['folds']};allrows=[]
    # Eligibility derives only from shared presence labels, never from responses.
    # Keep every75-category support table in the raw result; summaries show N.
    for result in read(path)['results']:
        bank=banks[result['bank_id']]
        for f in bank['evaluation_folds']:
            for category in result['rows']:
                c=category['category_index'];stratum='all_categories' if bank['regime']=='full_category' else ('guided' if c in folds[f]['guide'] else 'excluded')
                for curve in category['curves']:
                    if curve['negative_quantile']!=.95:continue
                    allrows.append(dict(regime=bank['regime'],kind=bank['kind'],capacity_matched=bank['capacity_matched'],seed=bank['seed'],fold=f,
                        category_index=c,stratum=stratum,budget=curve['budget'],status=curve['status'],metrics=curve['metrics']))
    groups={}
    for r in allrows:groups.setdefault((r['regime'],r['kind'],r['capacity_matched'],r['stratum'],r['budget']),[]).append(r)
    return [dict(regime=k[0],kind=k[1],capacity_matched=k[2],stratum=k[3],budget=k[4],planned=len(rr),supported=sum(r['metrics'] is not None for r in rr),
        coverage=mean([r['metrics']['OR_coverage'] for r in rr if r['metrics']]),FPR=mean([r['metrics']['OR_FPR'] for r in rr if r['metrics']])) for k,rr in sorted(groups.items())]

def report(root):
    root=Path(root);numeric_rows=summarize_numeric(root);summary=aggregate_numeric(numeric_rows);states={}
    collection=read(root/'COLLECTION_COMPLETE.json') if (root/'COLLECTION_COMPLETE.json').exists() else {};pilot=collection.get('scope')=='seed21_pilot'
    for name,file in dict(responses='COLLECTION_COMPLETE.json',numeric='NUMERIC_COMPLETE.json',presence_calibration='presence/calibration_LOCK.json',presence_audit='presence/audit_LOCK.json',
        presence_QA='PRESENCE_QA.json',concentration='CONCENTRATION_COMPLETE.json',coherence='COHERENCE_COMPLETE.json',target_alignment_calibration='ALIGNMENT_CALIBRATION_LOCK.json',target_alignment_audit='ALIGNMENT_AUDIT_LOCK.json').items():
        states[name]='complete' if (root/file).exists() and read(root/file).get('complete') else 'pending'
    coherence=read(root/'COHERENCE_COMPLETE.json') if states['coherence']=='complete' else None
    concentration=concentration_summary(root);target=read(root/'ALIGNMENT_AUDIT_LOCK.json') if states['target_alignment_audit']=='complete' else None
    comparisons=[]
    for regime in ['full_category','category_split']:
        stratum='all_categories' if regime=='full_category' else 'excluded'
        lookup={(r['kind'],r['seed'],r['fold']):r for r in numeric_rows if r['regime']==regime and r['stratum']==stratum and r['capacity_matched']}
        for baseline in ['vanilla','supervised_vanilla']+(['slics'] if regime=='full_category' else []):
            differences=[]
            for key,a in lookup.items():
                if key[0]!='ggf' or (baseline,key[1],key[2]) not in lookup:continue
                b=lookup[baseline,key[1],key[2]]
                for c,ap in a['AP_by_category'].items():
                    bp=b['AP_by_category'].get(c)
                    differences.append(dict(seed=key[1],fold=key[2],category_index=int(c),delta=None if ap is None or bp is None else ap-bp))
            per_seed={str(s):mean([r['delta'] for r in differences if r['seed']==s]) for s in [21,22,23]}
            comparisons.append(dict(regime=regime,baseline=baseline,endpoint='category_mask_AP',per_order_difference=per_seed,
                mean_difference=mean(per_seed.values()),supported_pairs=sum(r['delta'] is not None for r in differences),planned_pairs=len(differences),
                claim='Single-seed pilot; no training-variability estimate, seeds22-23 pending' if pilot else 'Descriptive matched effect; no broad superiority inference from three training orders sharing initialization'))
    result=dict(schema='all75-comparison-readout-v1',scope='seed21_pilot' if pilot else 'three_seed_comparison',full_three_seed_study_complete=not pilot and all(v=='complete' for v in states.values()),complete=all(v=='complete' for v in states.values()),stages=states,numeric=summary,presence_QA=read(root/'PRESENCE_QA.json') if states['presence_QA']=='complete' else None,
        matched_numeric_effects=comparisons,concentration_q95=concentration,coherence=coherence,target_alignment=target,
        limits=['Native category-split SLiCS37/38 is a separate reference, not a75-group baseline','Category assignments are not unit semantic labels',
            'Full-category and category-split experiments remain separate','AI contexts and repeated image pairs are not independent human/model replications',
            'Previously inspected image pools; fresh generalization requires a new allocation','Pending/unsupported outcomes are not zeros; unresolved bounds are not confidence intervals'])
    save(root/'COMPARISON_RESULTS.json',result,immutable=False)
    table=''.join(f'<tr><td>{r["regime"]}</td><td>{r["kind"]}</td><td>{r["stratum"]}</td><td>{fmt(r["mean_AP_supported"])}</td><td>{r["supported_category_fold_cells"]}/{r["planned_category_fold_cells"]}</td><td>{"matched75" if r["capacity_matched"] else "native reference"}</td></tr>' for r in summary)
    effects=''.join(f'<tr><td>{r["regime"]}</td><td>GGF minus {r["baseline"]}</td><td>{fmt(r["mean_difference"])}</td><td>{r["supported_pairs"]}/{r["planned_pairs"]}</td></tr>' for r in comparisons)
    cc=''
    if coherence:
        cc='<table><tr><th>Native bank</th><th>Specific semantic</th><th>Visual property</th></tr>'+''.join(f'<tr><td>{html.escape(b)}</td><td>{fmt(v["semantic"]["confirmed"])} ({v["semantic"]["counts"]["yes"]}/{v["semantic"]["denominator"]})</td><td>{fmt(v["visual"]["confirmed"])}</td></tr>' for b,v in coherence['bank_results'].items())+'</table>'
    text='<!doctype html><html lang="en"><meta charset="utf-8"><title>All75 comparison</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#183348}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:9px;text-align:left;border-bottom:1px solid #ccc}.note{background:#edf2f7;padding:16px}</style><h1>All75 comparison dashboard</h1><p class="note">'+('All required endpoint stages are present.' if result['complete'] else 'Partial readout: pending endpoints remain explicit. Numerical category AP is not semantic coherence or visual target alignment.')+'</p><h2>Stage status</h2><ul>'+''.join('<li>'+k+': <b>'+v+'</b></li>' for k,v in states.items())+'</ul><h2>Annotated category alignment</h2><table><tr><th>Experiment</th><th>Method</th><th>Stratum</th><th>Mean AP over supported categories</th><th>Supported cells</th><th>Capacity</th></tr>'+table+'</table><h2>Matched AP effects</h2><p>Differences are percentage-point proportions. Equal category weight within each training order, then equal order weight. These are descriptive effects; no method-population significance claim.</p><table><tr><th>Experiment</th><th>Comparison</th><th>Difference</th><th>Supported cells</th></tr>'+effects+'</table><h2>Blinded coherence</h2>'+cc+'<p>Specific-semantic and visual-property scores remain separate; unknowns retained. Full exact concentration curves, overlap/unique/incremental coverage, alignment counts, reconstruction, activity and resource details are in the source JSON files.</p><h2>Limits</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in result['limits'])+'</ul><p><a href="COMPARISON_RESULTS.json">Exact comparative results and pending statuses</a></p></html>'
    if pilot:text=text.replace('<h1>All75 comparison dashboard</h1>','<h1>All75 seed21 pilot dashboard</h1><p>Single-seed exploratory results. Seeds22–23 remain pending for replication; this is not a completed three-seed study.</p>')
    (root/'COMPARISON_REPORT.html').write_text(text,encoding='utf8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();report(a.root)
