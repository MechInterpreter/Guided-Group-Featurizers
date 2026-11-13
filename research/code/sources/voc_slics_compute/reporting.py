"""Evidence receipts, seed uncertainty, paired comparisons and paper-ready plots."""
import contextlib,csv,gc,json,platform,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from experiment import PRIMARY,EXTRA,SEEDS,LABELS

def stat(values):
    a=np.asarray(values,float)
    if len(a)!=3 or not np.isfinite(a).all():raise ValueError('Three finite paired seeds required')
    sd=float(a.std(ddof=1));h=float(t.ppf(.975,2)*sd/np.sqrt(3));mean=float(a.mean())
    return dict(mean=mean,sd=sd,ci95=[mean-h,mean+h],seeds=3)

def csv_file(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

@contextlib.contextmanager
def cost(ctx,name):
    started=time.monotonic();cuda=str(ctx['device']).startswith('cuda')
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    try:yield
    finally:
        if cuda:torch.cuda.synchronize()
        row=dict(stage=name,wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,device=torch.cuda.get_device_name() if cuda else platform.processor(),
            scope='This invocation including I/O/evaluation in this stage; reused computation is not charged again; not a hardware-normalized method speed comparison')
        path=ctx['store'].results/'cost_events.jsonl'
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print('STAGE COST',json.dumps(row),flush=True)

def paired_curves(a,b,seed):
    if a['image_ids']!=b['image_ids'] or a['category_indices']!=b['category_indices'] or not np.array_equal(a['den'],b['den']):raise ValueError('Paired consolidation cohorts differ')
    den=a['den'];ea=a['errors'];eb=b['errors'];N=den.shape[1];rng=np.random.default_rng(170000+seed);draw=[]
    for _ in range(1000):
        w=np.bincount(rng.integers(N,size=N),minlength=N);dd=den@w
        if np.any(dd<=1e-12):continue
        diff=np.mean(np.einsum('cki,i->ck',eb-ea,w)/dd[:,None],0);draw.append(diff[[0,1,3]])
    point=np.mean((eb.sum(2)-ea.sum(2))/den.sum(1)[:,None],0)[[0,1,3]]
    return dict(difference_1_2_4=point.tolist(),image_bootstrap95=np.quantile(draw,[.025,.975],axis=0).tolist(),replicates_valid=len(draw))

def summary(ctx,cohort):
    s=ctx['store'];flat=[];means=[];interprets=[];paired=[];rankrows=[]
    for kind in PRIMARY+EXTRA:
        reports=[];inter=[]
        for seed in SEEDS:
            point=s.load('point_'+cohort+'_'+str(seed)+'_'+kind)
            endpoint=s.load('endpoint_'+str(seed)+'_'+kind)
            if point is not None:
                r=point['report'];reports.append(r);flat.append(dict(method=LABELS[kind],seed=seed,capture=100*r['macro_capture'],patch_AP=100*r['macro_ap'],VE=100*r['reconstruction']['ve'],parameters=r['total_learned_parameters']))
            if endpoint is None:continue
            item=s.load('interpretability_'+cohort+'_'+endpoint['key'])
            if item is not None:inter.append(item)
            ranks=s.load('rank_'+endpoint['key'])
            if ranks is not None:
                for label in ['concept_slots','reserve_slots']:
                    n=len(ctx['cfg']['category_indices']);blocks=[r for r in ranks['blocks'] if (r['group']<n)==(label=='concept_slots')]
                    valid=[r for r in blocks if r.get('entropy_utilization') is not None]
                    rankrows.append(dict(method=LABELS[kind],seed=seed,slots=label,allocated_blocks=len(blocks),estimated_nonzero_variance=len(valid),
                        inactive_blocks=sum(r['state']=='inactive' for r in blocks),insufficient_sample_blocks=sum(r['state']=='insufficient-samples' for r in blocks),
                        zero_variance_blocks=sum(r.get('coefficient',{}).get('zero_variance',False) for r in blocks),
                        mean_active_samples=float(np.mean([r['active_samples'] for r in blocks])),
                        entropy_rank=float(np.mean([r['coefficient']['entropy_rank'] for r in valid])) if valid else None,
                        utilization_percent=100*float(np.mean([r['entropy_utilization'] for r in valid])) if valid else None,
                        coefficient_matrix_stable_rank=float(np.mean([r['coefficient']['coefficient_matrix_stable_rank'] for r in valid])) if valid else None,
                        contribution_matrix_stable_rank=float(np.mean([r['decoded_contribution']['coefficient_matrix_stable_rank'] for r in valid if r['decoded_contribution']['coefficient_matrix_stable_rank'] is not None])) if any(r['decoded_contribution']['coefficient_matrix_stable_rank'] is not None for r in valid) else None))
        if len(reports)==3:
            out=dict(kind=kind,label=LABELS[kind])
            for k,get in [('capture',lambda r:r['macro_capture']),('patch_AP',lambda r:r['macro_ap']),('VE',lambda r:r['reconstruction']['ve'])]:out[k]=stat([100*get(r) for r in reports])
            means.append(out)
        if len(inter)==3:
            interprets.append(dict(kind=kind,label=LABELS[kind],capture_1_2_4=[stat([100*r['macro_capture_1_2_4'][j] for r in inter]) for j in range(3)],
                target_AP_drop=stat([100*r['macro_target_AP_drop'] for r in inter]),random_AP_drop=stat([100*r['macro_random_AP_drop'] for r in inter])))
    cs=[s.load(f'concept_sae_{cohort}_s{seed}') for seed in SEEDS]
    if all(r is not None for r in cs):
        means.append(dict(kind='concept_sae',label='Concept-SAE (released; DINO/mask adaptation)',**{k:stat([100*get(r['report']) for r in cs]) for k,get in [('capture',lambda r:r['macro_capture']),('patch_AP',lambda r:r['macro_ap']),('VE',lambda r:r['reconstruction']['ve'])]}))
        for seed,rr in zip(SEEDS,cs):
            r=rr['report'];flat.append(dict(method='Concept-SAE (released; DINO/mask adaptation)',seed=seed,capture=100*r['macro_capture'],patch_AP=100*r['macro_ap'],VE=100*r['reconstruction']['ve'],parameters=r['model_parameters']))
    for control in ['released_vanilla','supervised_vanilla',*EXTRA]:
        trials=[]
        for seed in SEEDS:
            a=s.load(f'endpoint_{seed}_agf_g');b=s.load(f'endpoint_{seed}_{control}')
            if a is None or b is None:continue
            ac=s.load('consolidation_compact_'+cohort+'_'+a['key']);bc=s.load('consolidation_compact_'+cohort+'_'+b['key'])
            if ac is not None and bc is not None:
                cachekey=f'paired_consolidation_{cohort}_{seed}_{control}'
                sig=identity(dict(a=a['weights_sha'],b=b['weights_sha'],images=ac['image_ids'],categories=ac['category_indices'],training=ctx['packs']['train']['identity'],replicates=1000))
                saved=s.load(cachekey)
                if saved is None:saved=dict(signature=sig,intervals=paired_curves(ac,bc,seed));s.save(cachekey,saved)
                if saved['signature']!=sig:raise ValueError('Paired consolidation cache identity changed')
                trials.append(dict(seed=seed,**saved['intervals']))
        if len(trials)==3:paired.append(dict(comparison='GGF minus '+LABELS[control],per_seed=trials,paired_seed_t95=[stat([100*r['difference_1_2_4'][j] for r in trials]) for j in range(3)]))
    rankmeans=[]
    for kind in PRIMARY+EXTRA:
        for slot in ['concept_slots','reserve_slots']:
            rr=[r for r in rankrows if r['method']==LABELS[kind] and r['slots']==slot]
            if len(rr)!=3:continue
            item=dict(method=LABELS[kind],slots=slot,seeds=3)
            for k in ['entropy_rank','utilization_percent','coefficient_matrix_stable_rank','contribution_matrix_stable_rank','mean_active_samples']:
                item[k]=stat([r[k] for r in rr]) if all(r[k] is not None for r in rr) else dict(undefined=True,reason='At least one seed lacks nonzero-variance blocks; denominator not silently reduced')
            rankmeans.append(item)
    paired_metrics=[]
    for kind in ['released_vanilla','supervised_vanilla',*EXTRA,'concept_sae']:
        a=[s.load(f'point_{cohort}_{seed}_agf_g') for seed in SEEDS]
        b=[s.load(f'concept_sae_{cohort}_s{seed}' if kind=='concept_sae' else f'point_{cohort}_{seed}_{kind}') for seed in SEEDS]
        if not all(r is not None for r in a+b):continue
        diff=dict(comparison='GGF minus '+('Concept-SAE task adaptation' if kind=='concept_sae' else LABELS[kind]))
        for metric,get in [('capture',lambda r:r['macro_capture']),('patch_AP',lambda r:r['macro_ap']),('VE',lambda r:r['reconstruction']['ve'])]:
            diff[metric]=stat([100*(get(aa['report'])-get(bb['report'])) for aa,bb in zip(a,b)])
        paired_metrics.append(diff)
    out=dict(cohort=cohort,means=means,paired_metric_differences=paired_metrics,consolidation=interprets,paired_consolidation=paired,rank=rankrows,rank_means=rankmeans,
        uncertainty='SD describes variation over three seeds; t95 intervals use df2. Image bootstrap conditions on fitted models; it does not replace seed uncertainty.',
        previously_inspected_test=cohort=='coco',evaluation_scope='Previously inspected COCO audit' if cohort=='coco' else 'VOC official validation; held out from fitting/tuning, not official test')
    s.write('evidence_summary.json',out);csv_file(s.results/'paper_results_per_seed.csv',flat);csv_file(s.results/'rank_per_seed.csv',rankrows)
    table=[]
    for r in means:
        table.append(dict(method=r['label'],**{k:f"{r[k]['mean']:.2f} ± {r[k]['sd']:.2f}" for k in ['capture','patch_AP','VE']}))
    csv_file(s.results/'paper_results_mean_sd.csv',table)
    text='\\begin{tabular}{lrrr}\n\\toprule\nMethod & Capture (\\%) & Patch AP (\\%) & VE (\\%) \\\\\n\\midrule\n'
    for r in table:text+=r['method'].replace('&','\\&')+' & '+' & '.join(r[k].replace('±',r'$\pm$') for k in ['capture','patch_AP','VE'])+r' \\'+'\n'
    text+='\\bottomrule\n\\end{tabular}\n';(s.results/'paper_results.tex').write_text(text,encoding='utf-8')
    if means:plot(s.results,means,interprets)
    return out

def plot(folder,means,interpret):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'agf_g':'#008b80','released_vanilla':'#84939b','supervised_vanilla':'#cd8b2a','concept_sae':'#7757aa'}
    fig,axes=plt.subplots(1,3,figsize=(10,3.3),layout='constrained');y=np.arange(len(means))
    for ax,metric,title in zip(axes,['capture','patch_AP','VE'],['Object contrast capture (%)','Patch AP (%)','Reconstruction VE (%)']):
        ax.barh(y,[r[metric]['mean'] for r in means],xerr=[r[metric]['sd'] for r in means],color=[colors.get(r['kind'],'#8e78ad') for r in means],capsize=2,height=.64)
        ax.set_yticks(y,[r['label'] for r in means] if ax is axes[0] else ['']*len(means));ax.invert_yaxis();ax.set_xlabel(title);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Mean over three seeds; error bars: seed standard deviation',fontsize=9)
    fig.savefig(folder/'baseline_comparison.pdf');fig.savefig(folder/'baseline_comparison.png',dpi=200);plt.close(fig)
    if not interpret:return
    fig,axes=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    for r in [v for v in interpret if v['kind'] in PRIMARY]:
        x=np.array([1,2,4]);mu=np.array([p['mean'] for p in r['capture_1_2_4']]);low=[p['ci95'][0] for p in r['capture_1_2_4']];high=[p['ci95'][1] for p in r['capture_1_2_4']];color=colors.get(r['kind'],'#8e78ad')
        axes[0].plot(x,mu,'o-',label=r['label'],color=color);axes[0].fill_between(x,low,high,alpha=.12,color=color)
        delta=r['target_AP_drop']['mean']-r['random_AP_drop']['mean'];axes[1].scatter(r['random_AP_drop']['mean'],r['target_AP_drop']['mean'],label=r['label'],color=color,s=55)
    axes[0].set(xlabel='Native groups selected on TRAIN',ylabel='Held-out contrast capture (%)',xticks=[1,2,4]);axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Random-group removal: AP drop (pp)',ylabel='Target-group removal: AP drop (pp)')
    fig.suptitle('Mean over three seeds; shaded bands: 95% seed t intervals',fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.savefig(folder/'consolidation_and_interventions.pdf');fig.savefig(folder/'consolidation_and_interventions.png',dpi=200);plt.close(fig)

def checklist(ctx,cohort,complete=False):
    s=ctx['store'];summ=summary(ctx,cohort);core_complete=all(any(r['kind']==k for r in summ['means']) for k in PRIMARY)
    interp_complete=all(any(r['kind']==k for r in summ['consolidation']) for k in PRIMARY)
    wanted={'person','potted plant','vase','fork','spoon','pizza'}
    expected=3*sum(ctx['provenance']['categories'][c]['name'] in wanted for c in ctx['cfg']['category_indices'])
    atlas_files=[p for k in PRIMARY for p in (s.results/'atlas').glob(cohort+'_'+k+'_*.png') if not p.stem.endswith('_cloud')]
    coverage=[dict(item='Supervised Vanilla with both object objectives, heads, fixed development-selected weights',status='completed' if core_complete else 'pending',evidence='paper_results_mean_sd.csv; fixed_weight_selection.json; protocol.json'),
        dict(item='Numerical consolidation across matched BSF models',status='completed' if interp_complete else 'pending',evidence='evidence_summary.json; consolidation_plans/; rank/'),
        dict(item='Direction-level atlas across matched BSF models',status='completed' if expected>0 and len(atlas_files)==expected else 'partial/pending or no eligible requested atlas category',images_completed=len(atlas_files),images_expected=expected,evidence='atlas/index.html and adjacent PNG/PDF/JSON files'),
        dict(item='Functional group-removal diagnostic with frozen TRAIN probe and TRAIN-matched random controls',status='completed' if interp_complete else 'pending',evidence='interpretability/; TRAIN_linear_mask_probe checkpoint'),
        dict(item='Additional dataset training without new GGF sweep',status='completed' if cohort=='voc' and core_complete or s.load('VOC_completed_evidence') is not None else 'pending VOC training stage',evidence='VOC_result_location.json; voc_training/results/.../training_protocol.json'),
        dict(item='Additional published supervised method beyond existing SLiCS',status='completed task adaptation' if any(r['kind']=='concept_sae' for r in summ['means']) else 'pending',evidence='concept_sae/protocol.json; native CLIP benchmark replication NOT claimed'),
        dict(item='Mean/SD, paired seed intervals and paired image-bootstrap consolidation intervals',status='completed' if core_complete and interp_complete else 'partial',evidence='paper_results_mean_sd.csv; evidence_summary.json; coco/voc paired output'),
        dict(item='Practical runtime, GPU memory, model parameter counts',status='recorded for executed stages',evidence='cost_events.jsonl; per-seed results; hardware.json'),
        dict(item='Optional supervised Grassmannian/Group Lasso',status='not included; four UNSUPERVISED implementation variants remain supporting comparisons',evidence='protocol.json'),
        dict(item='Rewrite abstract/introduction/method/discussion/conclusion, integrate tables/figures and fix references',status='manual paper revision remains necessary',evidence='paper_results.tex and generated figures are inputs, not completed writing'),
        dict(item='Venue formatting/page limit, arXiv/conference versions, mentor approval and resource coordination',status='manual review remains necessary',evidence='No automatic sharing, publishing, checklist ticks or submission performed')]
    s.write('checklist_coverage.json',dict(source_document='private-study-checklist-not-distributed',complete=complete,items=coverage))
