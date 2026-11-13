"""All frozen endpoints, exact paired image intervals and separate seed variation."""
import gc
import numpy as np
from scipy.stats import t as student_t
from coco_baselines import training
from coco_ggf import study,matched_baselines as matched
from coco_coordinated import run as memory
from coco_baselines.storage import identity
from scale_models import model
from scale_metrics import evaluate
from scale_plan import FINAL_SEEDS


def verify_freeze(ctx):
    frozen=memory.required(ctx['store'],'all_models_frozen')
    expected={(seed,kind) for seed in FINAL_SEEDS for kind in ['agf_g','agf_x','mask_only','fixed_guided',*ctx['rule']['methods']]}
    if not frozen['complete'] or {(r['seed'],r['kind']) for r in frozen['models']}!=expected or len(frozen['models'])!=len(expected):
        raise ValueError('All seven seeds and requested arms must freeze before evaluation')
    for row in frozen['models']:
        st=memory.required(ctx['store'],row['key'])
        if training.weight_sha(st['weights'])!=row['weights_sha'] or identity(row['targets'])!=row['targets_sha']:
            raise ValueError('Frozen weights/readouts changed')
        del st
    matched.assert_sources(ctx);return frozen


def seed_interval(values):
    a=np.asarray(values,float)
    if len(a)<2 or not np.isfinite(a).all():raise ValueError('Need finite paired seed values')
    mean=float(a.mean());sd=float(a.std(ddof=1));half=float(student_t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
    return dict(mean=mean,seed_sd=sd,ci95=[mean-half,mean+half],seeds=len(a),
                scope='Student-t interval over training seeds, conditional on this fixed evaluation cohort')


def run(ctx,pack):
    frozen=verify_freeze(ctx);s=ctx['store'];old=s.load('test_complete')
    stamp=identity(dict(ids=pack['ids'],normalization=identity(ctx['norm']),cfg=ctx['cfg'],fresh=ctx['rule']['fresh']))
    if old is not None:
        if old['cohort_sha']!=stamp:raise ValueError('Evaluation cohort changed')
        return old
    reports=[];comparisons=[]
    for seed in FINAL_SEEDS:
        lookup={}
        for row in [r for r in frozen['models'] if r['seed']==seed]:
            tag=f'{row["kind"]}_s{seed}';signature=identity(dict(weights=row['weights_sha'],targets=row['targets_sha'],cohort=stamp))
            saved=s.load('test_'+tag)
            if saved is None:
                state=memory.required(s,row['key']);m=model(row['method'],row['groups'],row['rank'],row['active_groups'],pack['x'].shape[-1],row['coefficient']).to(ctx['device'])
                m.load_state_dict(state['weights']);del state
                report=evaluate(m,pack,row['targets'],ctx['cfg'],ctx['device'],compact=True)
                compact=report.pop('_compact');s.save('compact_'+tag,compact)
                report.update(seed=seed,kind=row['kind'],weights_sha=row['weights_sha'],groups=row['groups'],rank=row['rank'],k=row['active_groups'],
                    parameters=sum(p.numel() for p in m.parameters())+row['head_parameters'],cap=row['cap'],updates=row['total_training_updates'],
                    own_warm_eligible=row['development']['reconstruction']['mse']<=row['reference_warm_mse']*1.02 if 'development' in row else None,
                    common_native_warm_eligible=row['development']['reconstruction']['mse']<=row['common_native_warm_mse']*1.02 if 'common_native_warm_mse' in row else None,
                    sparsity_matched=abs(report['reconstruction']['active_groups']-row['active_groups'])<=.2*row['active_groups'])
                saved=dict(signature=signature,report=report);s.save('test_'+tag,saved);s.write('reports/'+tag+'.json',report)
                s.event('test_scored',seed=seed,kind=row['kind'],capture=report['macro_capture'],AP=report['macro_ap'],VE=report['reconstruction']['ve'])
                del m,compact;gc.collect()
            if saved['signature']!=signature:raise ValueError('Saved test model identity differs')
            reports.append(saved['report']);lookup[row['kind']]=saved['report']
        for a,b in [('agf_g',m) for m in ctx['rule']['methods']]+[('agf_x','agf_g'),('agf_g','mask_only'),('agf_g','fixed_guided')]:
            key=f'pair_s{seed}_{a}_vs_{b}';ci=s.load(key)
            if ci is None:
                s.event('paired_intervals',seed=seed,treatment=a,control=b,replicates=ctx['cfg']['bootstrap'])
                ci=study.paired(memory.required(s,f'compact_{a}_s{seed}'),memory.required(s,f'compact_{b}_s{seed}'),ctx['cfg']);s.save(key,ci)
            ar,br=lookup[a],lookup[b]
            comparisons.append(dict(seed=seed,treatment=a,control=b,capture_gain_pp=100*(ar['macro_capture']-br['macro_capture']),
                AP_gain_pp=100*(ar['macro_ap']-br['macro_ap']),VE_gain_pp=100*(ar['reconstruction']['ve']-br['reconstruction']['ve']),
                capture_ci_low=100*ci['capture_gain_ci95'][0],capture_ci_high=100*ci['capture_gain_ci95'][1],
                AP_ci_low=100*ci['ap_gain_ci95'][0],AP_ci_high=100*ci['ap_gain_ci95'][1],valid_image_bootstraps=ci['valid_bootstrap_replicates'],
                invalid_image_bootstraps=ci['invalid_full_cohort_replicates']))
    rr=[dict(seed=r['seed'],kind=r['kind'],capture_pct=100*r['macro_capture'],patch_AP_pct=100*r['macro_ap'],VE_pct=100*r['reconstruction']['ve'],
             active_groups=r['reconstruction']['active_groups'],sparsity_matched=r['sparsity_matched'],parameters=r['parameters'],updates=r['updates']) for r in reports]
    means=[]
    for kind in sorted({r['kind'] for r in rr}):
        rows=[r for r in rr if r['kind']==kind];entry=dict(kind=kind,seeds=len(rows))
        for metric in ['capture_pct','patch_AP_pct','VE_pct']:
            stat=seed_interval([r[metric] for r in rows]);entry[metric]=stat['mean'];entry[metric+'_seed_sd']=stat['seed_sd'];entry[metric+'_seed_ci_low']=stat['ci95'][0];entry[metric+'_seed_ci_high']=stat['ci95'][1]
        entry['mean_active_groups']=float(np.mean([r['active_groups'] for r in rows]));entry['sparsity_matched_seeds']=sum(r['sparsity_matched'] for r in rows);means.append(entry)
    seed_pairs=[]
    for a,b in sorted({(r['treatment'],r['control']) for r in comparisons}):
        rows=[r for r in comparisons if (r['treatment'],r['control'])==(a,b)];entry=dict(treatment=a,control=b)
        for metric in ['capture_gain_pp','AP_gain_pp','VE_gain_pp']:
            stat=seed_interval([r[metric] for r in rows]);entry[metric]=stat['mean'];entry[metric+'_seed_sd']=stat['seed_sd'];entry[metric+'_seed_ci_low']=stat['ci95'][0];entry[metric+'_seed_ci_high']=stat['ci95'][1]
        seed_pairs.append(entry)
    for name,rows in [('test_models.csv',rr),('test_means.csv',means),('test_paired_images.csv',comparisons),('test_paired_seeds.csv',seed_pairs),
                      ('test_categories.csv',[dict(seed=r['seed'],kind=r['kind'],**c) for r in reports for c in r['categories']])]:matched.csv_write(s.results/name,rows)
    out=dict(complete=True,cohort_sha=stamp,fresh=ctx['rule']['fresh'],images=len(pack['ids']),models=reports,paired_comparisons=comparisons,
             means=means,seed_pairs=seed_pairs,uncertainty='Seed intervals and conditional paired image intervals reported separately. No multiplicity adjustment; primary contrasts predeclared in protocol.')
    s.save('test_complete',out);s.write('test_summary.json',out)
    plot(s.results/'figures',means,ctx['rule']['fresh']);return out


def plot(folder,means,fresh):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,7),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,['capture_pct','patch_AP_pct','VE_pct'],['Capture','Patch AP','Reconstruction VE']):
        for i,r in enumerate(means):
            color='#087f83' if r['kind'].startswith('agf') else '#9375b2' if r['kind'].startswith('paper') else '#879cad'
            ax.barh(i,r[key],color=color,height=.65,xerr=r[key+'_seed_sd'],capsize=2)
            ax.text(r[key]+1,i,f'{r[key]:.2f}%',va='center',fontsize=9)
        ax.set_yticks(range(len(means)),[r['kind'].replace('_',' ') for r in means]);ax.set_title(title);ax.set_xlabel('Percent; error bars = seed SD')
        ax.set_xlim(min(0,min(r[key] for r in means)-5),max(105,max(r[key] for r in means)+10));ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis();fig.suptitle('Seven seeds / '+('new coverage-enriched holdout' if fresh else 'previously inspected evaluation cohort'))
    for ext in ['png','pdf']:fig.savefig(folder/('final_comparison.'+ext),dpi=200)
    plt.close(fig)
