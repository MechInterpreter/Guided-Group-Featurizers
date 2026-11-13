"""Common fixed readouts; unchanged capture/AP/VE and explicit paired intervals."""
import gc,json,os,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from coco_baselines import data,training
from coco_baselines.storage import identity,sha_file
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched,study
from scale_models import model
from scale_metrics import evaluate
from experiment import SEEDS,PRIMARY,EXTRA,LABELS

def coco_test(ctx):
    if ctx['store'].load('primary_frozen') is None:raise ValueError('Freeze primary endpoints before evaluation')
    source=ctx['sources']['strong'];ids=ctx['test_ids'];stamp=ctx['test_stamp'];norm=ctx['norm'];n=len(ids)
    folder=ctx['local']/'normalized';folder.mkdir(exist_ok=True)
    path=folder/('test_'+stamp['normalized_sha']+'.npy');meta=path.with_suffix('.json')
    valid=path.exists() and meta.exists() and json.loads(meta.read_text()).get('file_sha')==sha_file(path)
    cached=ctx['store'].load('cached_coco_test_masks') if valid else None
    if cached is not None:
        if cached['identity']!=identity(stamp) or cached['ids']!=ids:raise ValueError('Cached COCO masks differ')
        x=np.load(path,mmap_mode='r')
        if x.shape!=(n,196,768) or x.dtype!=np.float32 or memory.streaming_array_sha(x)!=stamp['normalized_sha']:raise ValueError('Cached normalized test bytes differ')
        memory.assert_immutable(ctx['stamps'])
        return dict(x=x,ids=ids,masks=cached['masks'],identity=identity(stamp))
    target=None
    if not valid:target=np.lib.format.open_memmap(path.with_suffix('.pending.npy'),mode='w+',dtype=np.float32,shape=(n,196,768))
    masks=[]
    for lo in range(0,n,ctx['test_chunk']):
        shard=memory.required(source,f'features_test_{lo:06d}');want=ids[lo:lo+ctx['test_chunk']]
        if shard['ids']!=want or shard['normalization_sha']!=identity(norm) or len(shard['masks'])!=len(want):raise ValueError('Cached test shard differs')
        raw=shard['x'].float().numpy() if torch.is_tensor(shard['x']) else np.asarray(shard['x'],np.float32)
        if raw.shape!=(len(want),196,768) or not np.isfinite(raw).all():raise ValueError('Invalid test features')
        if target is not None:target[lo:lo+len(want)]=data.normalize(raw,norm)
        masks.extend(shard['masks']);del shard,raw
        if lo%200==0:ctx['store'].event('loading_cached_test',completed=min(lo+ctx['test_chunk'],n),total=n)
    if target is not None:target.flush();del target;os.replace(path.with_suffix('.pending.npy'),path)
    x=np.load(path,mmap_mode='r')
    if x.shape!=(n,196,768) or x.dtype!=np.float32 or memory.streaming_array_sha(x)!=stamp['normalized_sha']:raise ValueError('Test activation bytes differ')
    meta.write_text(json.dumps(dict(file_sha=sha_file(path))))
    ctx['store'].save('cached_coco_test_masks',dict(ids=ids,identity=identity(stamp),masks=masks))
    memory.assert_immutable(ctx['stamps']);return dict(x=x,ids=ids,masks=masks,identity=identity(stamp))

def cfg_for(ctx,cohort):
    cfg=dict(ctx['cfg'])
    if cohort=='voc':cfg['category_indices']=memory.required(ctx['store'],'voc_plan')['category_indices']
    elif cohort!='coco':raise ValueError('Unknown evaluation cohort')
    return cfg

def stat(values):
    a=np.asarray(values,float)
    if len(a)!=3 or not np.isfinite(a).all():raise ValueError('Exactly three finite seeds required')
    mean=float(a.mean());sd=float(a.std(ddof=1));h=float(t.ppf(.975,2)*sd/np.sqrt(3))
    return dict(mean=mean,sd=sd,low=mean-h,high=mean+h)

def points(ctx,cohort):
    reports=[]
    for kind in PRIMARY+EXTRA:
        for seed in SEEDS:
            r=ctx['store'].load('point_'+cohort+'_'+str(seed)+'_'+kind)
            if r is not None:reports.append(r['report'])
    flat=[dict(seed=r['seed'],kind=r['kind'],label=LABELS[r['kind']],capture=100*r['macro_capture'],
        patch_AP=100*r['macro_ap'],VE=100*r['reconstruction']['ve'],active_groups=r['reconstruction']['active_groups']) for r in reports]
    if flat:matched.csv_write(ctx['store'].results/(cohort+'_models.csv'),flat)
    means=[]
    for kind in PRIMARY+EXTRA:
        rows=[r for r in flat if r['kind']==kind]
        if len(rows)!=3:continue
        out=dict(kind=kind,label=LABELS[kind],seeds=3)
        for metric in ['capture','patch_AP','VE']:
            ss=stat([r[metric] for r in rows]);out.update({metric+'_'+k:v for k,v in ss.items()})
        means.append(out)
    if means:
        matched.csv_write(ctx['store'].results/(cohort+'_means.csv'),means);plot(ctx,means,cohort)
    ctx['store'].write(cohort+'_summary.json',dict(models=reports,means=means,cohort=cohort,
        categories=cfg_for(ctx,cohort)['category_indices'],test_previously_inspected=cohort=='coco',
        VOC_official_test=False,scope='frozen-model external transfer' if cohort=='voc' else 'historical inspected test audit'))
    return reports,means

def score(ctx,rows,pack,cohort):
    cfg=cfg_for(ctx,cohort);s=ctx['store']
    if s.load('primary_frozen') is None:raise ValueError('Freeze primary models first')
    for row in rows:
        key=cohort+'_'+str(row['seed'])+'_'+row['kind'];targets=[r for r in row['targets'] if r['category_index'] in cfg['category_indices']]
        if [r['category_index'] for r in targets]!=cfg['category_indices']:raise ValueError('Missing fixed COCO-trained readout')
        signature=identity(dict(weights=row['weights_sha'],targets=targets,cohort=pack['identity'],cfg=cfg))
        saved=s.load('point_'+key)
        if saved is None:
            weights=torch.load(ctx['local']/'frozen_weights'/(row['weights_sha']+'.pt'),map_location='cpu',weights_only=True)
            if training.weight_sha(weights)!=row['weights_sha']:raise ValueError('Frozen inference weights changed')
            m=model(row['method'],row['groups'],row['rank'],row['active_groups'],768,row['coefficient']).to(ctx['device']);m.load_state_dict(weights);del weights;gc.collect()
            started=time.monotonic();s.event('evaluation_start',cohort=cohort,seed=row['seed'],kind=row['kind'],images=len(pack['ids']))
            with matched.progress(ctx,m,key):report=evaluate(m,pack,targets,cfg,ctx['device'],compact=True)
            compact=report.pop('_compact');report.update(seed=row['seed'],kind=row['kind'],elapsed_seconds=time.monotonic()-started,
                weights_sha=row['weights_sha'],targets_sha=identity(targets),model_parameters=row['model_parameters'],
                head_parameters=row['head_parameters'],controller_parameters=row['controller_parameters'],
                total_learned_parameters=row['total_learned_parameters'])
            native=memory.required(s,'endpoint_'+str(row['seed'])+'_released_vanilla')
            limit=1.02*native['reference_warm_mse']
            report.update(common_native_warm_selection_MSE=native['reference_warm_mse'],development_MSE_limit=limit,
                within_common_development_limit=row['development']['reconstruction']['mse']<=limit,
                limit_does_not_filter_reported_models=True)
            # Separate compact cache avoids loading arrays just to display point estimates.
            s.save('compact_'+key,dict(signature=signature,audit=compact));s.save('point_'+key,dict(signature=signature,report=report))
            del compact,m;gc.collect();torch.cuda.empty_cache();saved=dict(signature=signature,report=report)
        if saved['signature']!=signature:raise ValueError('Saved evaluation protocol changed')
        r=saved['report'];print('RESULT',cohort,LABELS[row['kind']],row['seed'],
            json.dumps(dict(capture=100*r['macro_capture'],patch_AP=100*r['macro_ap'],VE=100*r['reconstruction']['ve'])),flush=True)
        points(ctx,cohort)
    _,means=points(ctx,cohort);print('THREE-SEED MEANS',cohort,json.dumps(means),flush=True)

def intervals(ctx,rows,cohort):
    s=ctx['store'];cfg=cfg_for(ctx,cohort)
    controls=['released_vanilla','supervised_vanilla'] if {r['kind'] for r in rows}==set(PRIMARY) else sorted({r['kind'] for r in rows})
    for control in controls:
        pairs=[]
        for seed in SEEDS:
            akey=cohort+'_'+str(seed)+'_agf_g';bkey=cohort+'_'+str(seed)+'_'+control
            a=memory.required(s,'point_'+akey);b=memory.required(s,'point_'+bkey)
            signature=identity(dict(a=a['signature'],b=b['signature'],replicates=cfg['bootstrap'],seed=cfg['fresh_test_seed']))
            key='paired_'+cohort+'_'+str(seed)+'_'+control;out=s.load(key)
            if out is None:
                s.event('paired_bootstrap_start',cohort=cohort,seed=seed,control=control,replicates=cfg['bootstrap'])
                ac=memory.required(s,'compact_'+akey);bc=memory.required(s,'compact_'+bkey)
                if (ac['signature'],bc['signature'])!=(a['signature'],b['signature']):raise ValueError('Compact evidence identity changed')
                from bootstrap_fast import paired
                ci=paired(ac['audit'],bc['audit'],cfg,ctx['device'])
                out=dict(signature=signature,image_intervals=ci,seed=seed,control=control,
                    capture_gain=100*(a['report']['macro_capture']-b['report']['macro_capture']),
                    patch_AP_gain=100*(a['report']['macro_ap']-b['report']['macro_ap']),
                    VE_gain=100*(a['report']['reconstruction']['ve']-b['report']['reconstruction']['ve']))
                s.save(key,out);s.write('intervals/'+key+'.json',out);del ac,bc;gc.collect()
            if out['signature']!=signature:raise ValueError('Paired interval protocol changed')
            pairs.append(out)
        summary=dict(cohort=cohort,control=control,seed_count=3,
            differences_pp={metric:stat([r[metric] for r in pairs]) for metric in ['capture_gain','patch_AP_gain','VE_gain']},
            image_intervals_by_seed=pairs,
            interpretation='Seed t intervals describe training variation. Image bootstrap intervals are conditional on one fitted seed. No multiplicity correction; no causal mediation claim.')
        s.write('intervals/'+cohort+'_vs_'+control+'.json',summary)

def plot(ctx,means,cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10,max(2.8,.38*len(means)+.9)),sharey=True,layout='constrained')
    y=np.arange(len(means))
    for ax,metric,title in zip(axes,['capture','patch_AP','VE'],['Capture (%)','Patch AP (%)','Reconstruction VE (%)']):
        vals=[r[metric+'_mean'] for r in means];sd=[r[metric+'_sd'] for r in means]
        ax.barh(y,vals,xerr=sd,color=['#008f83' if r['kind']=='agf_g' else '#d49a40' if r['kind']=='supervised_vanilla' else '#8397a3' for r in means],height=.65,capsize=2)
        ax.set_yticks(y,[r['label'] for r in means],fontsize=8);ax.set_xlabel(title);ax.set_xlim(min(-2,min(vals)-8),105)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].invert_yaxis();fig.suptitle(cohort.upper()+' / 1024 groups /12D /Top16 /three seeds (bars: mean ± SD)',fontsize=10)
    folder=ctx['store'].results/'figures';folder.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(folder/(cohort+'_comparison.'+suffix),dpi=180)
    plt.close(fig)
