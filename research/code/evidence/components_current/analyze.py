"""Recompute completed component-ablation summaries from verified evidence."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.stats import t
R=Path(__file__).resolve().parent;D=R
def csvread(name):return list(csv.DictReader((D/name).open(newline='')))
models=csvread('test_models.csv');means=csvread('test_means.csv');cats=csvread('test_categories.csv')
paired=csvread('component_paired_images.csv');seed_pairs=csvread('component_paired_seeds.csv')
summary=json.loads((D/'component_summary.json').read_text());frozen=json.loads((D/'components_frozen.json').read_text())
assert summary['complete'] and len(models)==len(frozen['models'])==12
assert {(int(r['seed']),r['kind']) for r in models}=={(s,k) for s in [21,22,23] for k in ['released_vanilla','mask_only','fixed_guided','agf_g']}
assert len(cats)==900 and len(paired)==9 and len(seed_pairs)==3
assert all(int(r['valid_bootstraps'])==1000 and int(r['invalid_bootstraps'])==0 for r in paired)
assert all(r['total_training_updates']==10640 and r['groups']==4096 and r['rank']==12 and r['active_groups']==32 for r in frozen['models'])
assert all(next(x for x in frozen['models'] if x['seed']==seed and x['kind']=='mask_only')['targets_sha']==next(x for x in frozen['models'] if x['seed']==seed and x['kind']=='agf_g')['targets_sha']==next(x for x in frozen['models'] if x['seed']==seed and x['kind']=='fixed_guided')['targets_sha'] for seed in [21,22,23])
for m in means:
    rr=[r for r in models if r['kind']==m['kind']]
    for key in ['capture_pct','patch_AP_pct','VE_pct']:
        a=np.array([float(r[key]) for r in rr]);assert np.isclose(a.mean(),float(m[key]),atol=1e-12)
        assert np.isclose(a.std(ddof=1),float(m[key+'_seed_sd']),atol=1e-12)
mapping={r['kind']:r for r in means};index={(int(r['seed']),r['kind'],int(r['category_index'])):r for r in cats}
comparisons=[]
for control,treatment,label in [('released_vanilla','mask_only','Initialization'),('mask_only','fixed_guided','Fixed object guidance'),('fixed_guided','agf_g','Adaptive weighting')]:
    comparison=dict(component=label,treatment=treatment,control=control,mean_gains_pp={},per_seed={},category_mean_wins={})
    for key in ['capture_pct','patch_AP_pct','VE_pct']:
        comparison['mean_gains_pp'][key]=float(mapping[treatment][key])-float(mapping[control][key])
    for seed in [21,22,23]:
        a=next(r for r in models if int(r['seed'])==seed and r['kind']==treatment);b=next(r for r in models if int(r['seed'])==seed and r['kind']==control)
        p=next(r for r in paired if int(r['seed'])==seed and r['treatment']==treatment and r['control']==control)
        comparison['per_seed'][str(seed)]=dict(gains_pp={k:float(a[k])-float(b[k]) for k in ['capture_pct','patch_AP_pct','VE_pct']},capture_ci95=[float(p['capture_ci_low']),float(p['capture_ci_high'])],AP_ci95=[float(p['AP_ci_low']),float(p['AP_ci_high'])])
    category_ids=sorted({int(r['category_index']) for r in cats})
    for key in ['capture','ap']:
        diffs=[np.mean([float(index[(s,treatment,c)][key])-float(index[(s,control,c)][key]) for s in [21,22,23]]) for c in category_ids]
        comparison['category_mean_wins'][key]=dict(positive=int(np.sum(np.array(diffs)>1e-12)),negative=int(np.sum(np.array(diffs)<-1e-12)),tied=int(np.sum(np.abs(diffs)<=1e-12)),total=75)
    comparisons.append(comparison)
total={k:float(mapping['agf_g'][k])-float(mapping['released_vanilla'][k]) for k in ['capture_pct','patch_AP_pct','VE_pct']}
fractions={c['component']:{k:100*v/total[k] for k,v in c['mean_gains_pp'].items()} for c in comparisons}
result=dict(verified=True,completion=summary['complete'],configuration=dict(groups=4096,width=12,topk=32,cap=.6,seeds=[21,22,23],updates=10640),means=means,comparisons=comparisons,total_guided_vs_vanilla_gain_pp=total,sequential_gain_fractions_pct=fractions,seed_intervals=seed_pairs,development_MSE_passes={k:sum(r['native_warm_eligible']=='True' for r in models if r['kind']==k) for k in mapping},
    scope='Same previously inspected 1,500-image test cohort. No fresh holdout, new warm-ups or retraining of six controls. Fixed weights are per-seed means of six logged TRAIN weight vectors; no validation or test tuning. Three-seed intervals are distinct from paired whole-image bootstrap intervals. Matched capacity, activity, updates and image/noise streams; not matched FLOPs. Sequential gain fractions describe this ablation path, not unique independent causal shares.',
    artifact_zip_sha256='2c2ec70992d9823acc6f079c20731d0e183a917f6cc7fcd8752648a5ba06307c')
(R/'ANALYSIS.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(comparisons=comparisons,total=total,sequential_gain_fractions_pct=fractions,development_MSE_passes=result['development_MSE_passes']),indent=2))
