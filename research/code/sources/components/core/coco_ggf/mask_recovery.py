"""A bounded test of reconstruction pressure from saved guided endpoints."""
import copy,csv,gc,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training,metrics
from coco_baselines.storage import identity
from coco_adaptation import core,inputs
from coco_coordinated import run as memory
from . import large_mask,study,slics_closeout as close
from .mask_recovery_fit import fit_recovery
from .mapping_export import install_mapping_export_fix

PARENT_ID='26d38cb987619ca46846630664c35eb2a0e2808147557ed37f22d571474fde6e'
PARENT_SHA='a035298cd89bba1493e0893ddc563b4db5e318c1f8ce586d928a1f08327c3475'
LABELS=dict(continued_object='Continued Vanilla',ggf='Previous GGF',mask_start='Mask initialization + reconstruction',
            endpoint='Unrecovered guided endpoint',control='Endpoint + continued GGF',recovery='Endpoint + reconstruction pressure')

def settings():
    return dict(version=1,study='ggf_mask_reconstruction_recovery',seeds=[21,22,23],
                updates=500,evaluate_every=50,save_every=50,bootstrap=200,
                reconstruction_weights=dict(control=1.0,recovery=4.0))

def prepare(source_output,parent_output,comparison_output,confirmation_output,output,local,sha,versions,cfg,device,require_drive=True):
    if cfg!=settings():raise ValueError('Keep the frozen two-arm, three-start recovery protocol')
    dest=Path(output).resolve();parent_root=Path(parent_output).resolve()
    if dest==parent_root or parent_root in dest.parents or dest in parent_root.parents:
        raise ValueError('Recovery output must be separate from its read-only parent')
    ctx=large_mask.prepare(source_output,comparison_output,confirmation_output,
        Path(output)/'input_verification',local,sha,versions,large_mask.settings(),device,require_drive)
    parent=memory.FrozenStore(Path(parent_output)/'checkpoints'/PARENT_ID[:20],Path(local)/'read_mask_parent')
    if parent.run_id!=PARENT_ID or parent.config['source_sha']!=PARENT_SHA:
        raise ValueError('Wrong completed larger mask-initialization source')
    if parent.config['source_run']!=ctx['source'].run_id or parent.config['activation_run']!=ctx['original'].run_id:
        raise ValueError('Parent activation/control lineage differs')
    if large_mask.training_versions(parent.config['versions'])!=large_mask.training_versions(versions) or parent.config['device']!=device:
        raise ValueError('Use matching training numerical versions and device category')
    if not memory.required(parent,'large_summary')['complete']:
        raise ValueError('Complete the parent evaluation before this recovery study')
    cfg_fit=dict(ctx['cfg'],updates=cfg['updates'],evaluate_every=cfg['evaluate_every'],
                 diagnostic_grad_every=cfg['evaluate_every'],local_save_every=cfg['save_every'])
    ctx.update(cfg=cfg_fit,recovery=cfg,parent_mask=parent)
    ctx['stamps']+=memory.source_snapshot([parent])
    ctx['store']=inputs.JournalStore(output,dict(config=cfg,fit_config=cfg_fit,source_sha=sha,
        versions=versions,device=device,parent_run=parent.run_id,original_control_run=ctx['source'].run_id),
        Path(local)/'recovery_rescue',require_drive)
    s=ctx['store'];catalog=[]
    for seed in cfg['seeds']:
        st=memory.required(parent,f'large_mask_ggf_s{seed}')
        if st['updates']!=2500 or st['target']!=2500 or st['reference_mse'] is None:
            raise ValueError('Expected complete guided endpoint with common reference floor')
        targets=memory.required(parent,f'large_mask_targets_s{seed}')
        if identity(targets)!=st['target_sha'] or sorted(t['category_index'] for t in targets)!=ctx['cfg']['category_indices']:
            raise ValueError('Saved training target mapping differs')
        if tuple(st['head'].shape)!=(75,2):raise ValueError('Trained head shape differs')
        if not all(torch.isfinite(w).all() for w in st['weights'].values()) or not torch.isfinite(st['head']).all():
            raise ValueError('Nonfinite source endpoint or head')
        if not np.isclose(st['reference_mse'],memory.required(ctx['source'],f'ggf_s{seed}')['initial_mse'],rtol=0,atol=0):
            raise ValueError('Original reconstruction reference changed')
        catalog.append(dict(seed=seed,key=f'large_mask_ggf_s{seed}',field='weights',updates=2500,
            weights_sha=training.weight_sha(st['weights']),head_sha=training.weight_sha({'head':st['head']}),
            targets_sha=identity(targets),common_reference_mse=st['reference_mse'],endpoint_validation=st['history'][-1]['validation']))
        del st;gc.collect()
    ctx['endpoints']=catalog
    s.write('input_catalog.json',catalog)
    s.write('protocol_settings.json',dict(config=cfg,fit_config=cfg_fit,
        change='Recovery multiplies only reconstruction loss by4. Control continues the original GGF loss. Same endpoints, trained heads, fixed target groups, batches/noise,500 updates, fresh Adam and original learning rate.',
        reset='Both arms intentionally reset Adam to isolate a new recovery phase; neither resets model weights or the trained detector head.',
        coefficient='4 is a predeclared diagnostic strength, not a claimed optimal setting or a fit to test scores.',
        guard='UNCHANGED: original native development MSE +2%; maximum capture+.2AP among eligible checkpoints. An ineligible endpoint cannot win by default.',
        preservation='Original contrast and detection losses remain active; preservation is measured, not guaranteed. Every category loss and reconstruction rejection is retained.',
        controls='Previous GGF, continued Vanilla and parent selected mask-model reports are read-only and reused. No baselines, SVD, backbone or warm-up are trained.',
        test='Same previously inspected1500-image holdout; all six recovery fits freeze before calibration/test. Unrecovered endpoints are audited after freeze and remain marked ineligible.',
        scope='GGF recovery only. No group exchanges or dimension changes. Calibration/test never chooses weights or recovery coefficient.'))
    memory.assert_immutable(ctx['stamps']);return ctx

def endpoint(ctx,seed):
    st=memory.required(ctx['parent_mask'],f'large_mask_ggf_s{seed}')
    expected=next(r for r in ctx['endpoints'] if r['seed']==seed)
    if training.weight_sha(st['weights'])!=expected['weights_sha'] or training.weight_sha({'head':st['head']})!=expected['head_sha']:
        raise ValueError('Frozen endpoint changed')
    return st,memory.required(ctx['parent_mask'],f'large_mask_targets_s{seed}')

def preflight(ctx):
    s=ctx['store'];old=s.load('preflight')
    if old is not None:return old
    seed=ctx['recovery']['seeds'][0];st,targets=endpoint(ctx,seed)
    m=core.new_model('vanilla',ctx['base'],st['weights'],ctx['device']);head=torch.nn.Parameter(st['head'].to(ctx['device']).clone())
    rng=np.random.default_rng(82000+seed);idx=rng.integers(len(ctx['packs']['train']['ids']),size=ctx['cfg']['images_per_update'])
    x=torch.as_tensor(np.array(ctx['packs']['train']['x'][idx],copy=True),device=ctx['device'])
    masks=[ctx['packs']['train']['masks'][i] for i in idx];opt=torch.optim.Adam(list(m.parameters())+[head],lr=ctx['cfg']['lr'])
    times=[]
    for i in range(6):
        study.sync(ctx['device']);t=time.monotonic();m.train();xx=x.reshape(-1,m.d)
        noisy=xx+ctx['cfg']['snr']*xx.std()*torch.randn_like(xx)
        native,_=m.loss(noisy,target=xx);cap,det,_=core.supervised_loss(m,x,masks,targets,head,ctx['cfg'])
        loss=4*native+ctx['cfg']['lambda_capture']*cap+ctx['cfg']['lambda_detection']*det
        if not torch.isfinite(loss):raise FloatingPointError('Preflight objective is nonfinite')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(m.parameters())+[head],ctx['cfg']['grad_clip'])
        opt.step();m.normalize_decoder();study.sync(ctx['device'])
        if i>=2:times.append(time.monotonic()-t)
    t=time.monotonic();v=core.validation(m,ctx['packs']['select'],targets,ctx['cfg'],ctx['device']);secs=time.monotonic()-t
    out=dict(numerical_ok=all(np.isfinite(v['reconstruction'][k]) for k in ('mse','ve')),
        new_fits=6,new_updates=3000,seconds_per_update=float(np.median(times)),development_seconds=secs,
        gradient_minutes=3000*float(np.median(times))/60,
        gradient_and_development_minutes=(3000*float(np.median(times))+66*secs)/60,
        excludes='Calibration,9 new model audits,paired bootstraps,Drive saves,input verification and plots. Not a guaranteed ETA.',
        baseline_retraining=0,backbone_extraction=0,source_endpoint_unchanged=True)
    del m,head,opt,x,st;gc.collect()
    if ctx['device'].startswith('cuda'):torch.cuda.empty_cache()
    s.save('preflight',out);s.write('preflight.json',out);return out

def train(ctx,max_updates=None):
    s=ctx['store'];old=s.load('all_models_frozen')
    if old is not None:return old
    rows=[]
    for seed in ctx['recovery']['seeds']:
        st,targets=endpoint(ctx,seed);initial=st['weights'];head=st['head'];ref=st['reference_mse']
        first=st['history'][-1]['validation'];parent_catalog=memory.required(ctx['parent_mask'],'all_models_frozen')['models']
        common=dict(seed=seed,parameters=sum(w.numel() for w in initial.values()),common_reference_mse=ref)
        rows.append(dict(**common,arm='endpoint',origin='new',store_kind='parent',key=f'large_mask_ggf_s{seed}',field='weights',
            weights_sha=training.weight_sha(initial),eligible=bool(first['reconstruction']['mse']<=ref*1.02),
            best_update=2500,recovery_updates=0,selection_score=first['score']))
        streams=[];noise_states=[];choices=[]
        for arm,weight in ctx['recovery']['reconstruction_weights'].items():
            key=f'recovery_s{seed}_{arm}';s.event('recovery_arm_start',seed=seed,arm=arm,reconstruction_weight=weight)
            fit=fit_recovery(ctx,key,initial,targets,True,82000+seed,max_updates=max_updates,
                reference_mse=ref,initial_head=head,reconstruction_weight=weight)
            if fit['updates']!=ctx['recovery']['updates']:return dict(complete=False,seed=seed,arm=arm)
            eligible=any(v['object_checkpoint_eligible'] for v in fit['history'])
            field='best' if eligible else 'weights'
            row=dict(**common,arm=arm,origin='new',store_kind='current',key=key,field=field,
                weights_sha=training.weight_sha(fit[field]),eligible=eligible,best_update=fit['best_update'] if eligible else None,
                recovery_updates=fit['updates'],selection_score=fit['best_score'] if eligible else None,
                reconstruction_weight=weight,initial_weights_sha=fit['initial_sha'],initial_head_sha=fit['recovery_identity']['initial_head_sha'])
            rows.append(row);choices.append(row);streams.append(fit['sampled_batches_sha']);noise_states.append(fit['torch_rng'].clone())
            s.event('recovery_arm_complete',seed=seed,arm=arm,eligible=eligible,best_update=row['best_update'])
            del fit;gc.collect()
        if len(set(streams))!=1 or not torch.equal(noise_states[0],noise_states[1]):raise ValueError('Matched batch/noise streams diverged')
        eligible_choices=[r for r in choices if r['eligible']]
        chosen=max(eligible_choices,key=lambda r:r['selection_score'])['arm'] if eligible_choices else None
        s.write(f'paired_training_s{seed}.json',dict(seed=seed,common_weights_sha=training.weight_sha(initial),
            common_head_sha=training.weight_sha({'head':head}),batch_sha=streams[0],equal_noise_RNG=True,
            updates_per_arm=ctx['recovery']['updates'],development_selected_arm=chosen,
            original_MSE_limit=ref*1.02,test_used=False))
        mask_row=copy.deepcopy(next(r for r in parent_catalog if r['seed']==seed and r['arm']=='mask_ggf'))
        mask_row.update(arm='mask_start',origin='saved',store_kind='parent',source_tag=f's{seed}_mask_ggf');rows.append(mask_row)
        for arm in ('ggf','continued_object'):
            row=copy.deepcopy(next(r for r in ctx['source_frozen']['models'] if r['seed']==seed and r['arm']==arm))
            row.update(origin='saved',store_kind='main',source_tag=f's{seed}_{arm}',eligible=True);rows.append(row)
        del initial,head,st;gc.collect()
    out=dict(complete=True,models=rows,test_used=False,structural_edits=0,new_updates=3000)
    memory.assert_immutable(ctx['stamps']);s.save('all_models_frozen',out);s.write('all_models_frozen.json',out);return out

def score_models(ctx):
    s=ctx['store'];frozen=memory.required(s,'all_models_frozen')
    if not frozen['complete']:raise RuntimeError('Freeze all six fits before audit')
    install_mapping_export_fix(ctx)
    mapping,test=large_mask.load_audit(ctx);reports=[]
    for row in frozen['models']:
        tag=f"s{row['seed']}_{row['arm']}";r=s.load('report_'+tag)
        if r is not None:reports.append(r);continue
        source=ctx['parent_mask'] if row['store_kind']=='parent' else ctx['source'] if row['store_kind']=='main' else s
        st=memory.required(source,row['key']);w=st[row['field']];del st
        if training.weight_sha(w)!=row['weights_sha']:raise ValueError('Frozen audit weights changed')
        if row['origin']=='saved':
            r=copy.deepcopy(memory.required(source,'report_'+row['source_tag']));ca=memory.required(source,'compact_'+row['source_tag'])
            if r['summary']['weights_sha']!=row['weights_sha'] or ca['image_ids']!=test['ids'] or ca['category_indices']!=ctx['cfg']['category_indices']:
                raise ValueError('Cached audit identity differs')
            r['summary'].update(**row);s.save('compact_'+tag,ca)
            maps=memory.required(source,'mapping_'+row['source_tag']);s.save('mapping_'+tag,maps);s.write('mappings/'+tag+'.json',maps)
            del ca,w
        else:
            m=core.new_model('vanilla',ctx['base'],w,ctx['device']);del w
            maps=s.load('mapping_'+tag)
            if maps is None:
                s.event('calibration_start',model=tag,images=400)
                with close.progress(ctx,m,tag+'_calibration'):a=metrics.audit_arrays(m,mapping['x'],mapping['masks'],mapping['ids'],ctx['cfg'],ctx['device'])
                maps=metrics.calibrate(a,ctx['cfg'],ctx['provenance']['categories']);del a;gc.collect();s.save('mapping_'+tag,maps)
            s.write('mappings/'+tag+'.json',maps)
            s.event('test_scoring_start',model=tag,images=1500)
            with close.progress(ctx,m,tag+'_test'):a=metrics.audit_arrays(m,test['x'],test['masks'],test['ids'],ctx['cfg'],ctx['device'])
            per=metrics.evaluate(a,maps,dict(ctx['cfg'],bootstrap=0),ctx['cfg']['fresh_test_seed'])
            by={v['category_index']:v for v in per};cohort=[by[c] for c in ctx['cfg']['category_indices']]
            if any(v['contrast'] is None or v['detection'] is None for v in cohort):raise ValueError('Missing frozen test category; denominator cannot shrink')
            rec=training.measure(m,test['x'].reshape(-1,m.d),ctx['device'],ctx['cfg']['batch'])
            if rec['active_groups']>16.00001:raise ValueError('Top16 budget exceeded')
            ca=study.compact_audit(a,maps,ctx['cfg']);s.save('compact_'+tag,ca)
            summary=dict(**row,groups=1024,block_rank=3,images=1500,categories=75,reconstruction=rec,
                mean_single_group_contrast_capture=float(np.mean([v['contrast']['single_group_capture'] for v in cohort])),
                mean_single_group_ap=float(np.mean([v['detection']['ap'] for v in cohort])),
                mean_multi_group_contrast_capture=float(np.mean([v['contrast']['multi_group_capture'] for v in cohort])),
                mean_all_group_contrast_capture=float(np.mean([v['contrast']['all_group_capture'] for v in cohort])))
            r=dict(summary=summary,categories=per,skipped_objects=a['skipped'])
            del m,a,ca;gc.collect()
            if ctx['device'].startswith('cuda'):torch.cuda.empty_cache()
        s.save('report_'+tag,r);s.write('reports/'+tag+'.json',r);reports.append(r)
        s.event('model_audit_complete',seed=row['seed'],arm=row['arm'],eligible=row['eligible'])
    return reports

def evaluate(ctx):
    s=ctx['store'];old=s.load('recovery_summary')
    if old is not None:
        reports=[memory.required(s,f"report_s{r['seed']}_{r['arm']}") for r in old['models']]
        tables(ctx,reports,old);memory.assert_immutable(ctx['stamps']);return old
    reports=score_models(ctx);comparisons=[]
    for seed in ctx['recovery']['seeds']:
        lookup={r['summary']['arm']:r for r in reports if r['summary']['seed']==seed}
        for arm,control in [('recovery','control'),('recovery','endpoint'),('recovery','ggf'),('control','ggf')]:
            a,b=lookup[arm]['summary'],lookup[control]['summary'];key=f'paired_s{seed}_{arm}_vs_{control}';ci=s.load(key)
            if ci is None:
                s.event('paired_bootstrap_start',seed=seed,arm=arm,control=control)
                ca=memory.required(s,f'compact_s{seed}_{arm}');cb=memory.required(s,f'compact_s{seed}_{control}')
                ci=study.paired(ca,cb,dict(ctx['cfg'],bootstrap=ctx['recovery']['bootstrap']));s.save(key,ci);del ca,cb;gc.collect()
            cg=a['mean_single_group_contrast_capture']-b['mean_single_group_contrast_capture']
            ag=a['mean_single_group_ap']-b['mean_single_group_ap'];vc=b['reconstruction']['ve']-a['reconstruction']['ve']
            comparisons.append(dict(seed=seed,arm=arm,control=control,capture_gain=cg,ap_gain=ag,ve_cost=vc,
                eligible=a['eligible'],positive_within_predeclared_cost=bool(a['eligible'] and cg>0 and ag>=0 and vc<=ctx['cfg']['max_reconstruction_ve_drop']),paired_intervals=ci))
    out=dict(complete=True,models=[r['summary'] for r in reports],comparisons=comparisons,
        eligible_recoveries=sum(r['summary']['eligible'] for r in reports if r['summary']['arm']=='recovery'),
        positives_vs_previous_GGF=sum(c['positive_within_predeclared_cost'] for c in comparisons if c['arm']=='recovery' and c['control']=='ggf'),
        starts=3,scope='500-update two-arm recovery,trained endpoint heads retained,fresh Adam,unchanged2%MSE floor. Same inspected1500-image holdout. No structural adaptation.',
        interpretation='Separate feasibility restoration, retention of endpoint gains, and advantage over previousGGF. Failed starts and category losses remain visible; no test-driven coefficient selection.')
    tables(ctx,reports,out);diagnostics(ctx,reports);memory.assert_immutable(ctx['stamps'])
    s.save('recovery_summary',out);s.write('summary.json',out);return out

def tables(ctx,reports,summary):
    s=ctx['store'];rows=[];categories=[]
    for r in reports:
        v=r['summary'];rows.append(dict(seed=v['seed'],arm=v['arm'],method=LABELS[v['arm']],eligible=v['eligible'],
            capture_pct=100*v['mean_single_group_contrast_capture'],ap_pct=100*v['mean_single_group_ap'],
            ve_pct=100*v['reconstruction']['ve'],selected_update=v.get('best_update')))
        for c in r['categories']:
            if c['category_index'] not in ctx['cfg']['category_indices']:continue
            categories.append(dict(seed=v['seed'],arm=v['arm'],category=c['category'],
                capture_pct=100*c['contrast']['single_group_capture'],ap_pct=100*c['detection']['ap'],
                capture_group=c['contrast']['groups'][0],detector_group=c['detection']['group'],objects=c['contrast']['object_count']))
    agg=[]
    for arm,label in LABELS.items():
        q=[r for r in rows if r['arm']==arm]
        if len(q)!=3:raise ValueError('Report every arm and all three starts')
        agg.append(dict(arm=arm,method=label,**{k:float(np.mean([r[k] for r in q])) for k in ['capture_pct','ap_pct','ve_pct']},eligible=sum(r['eligible'] for r in q)))
    for name,values in [('comparison',rows),('aggregate',agg),('per_category',categories)]:
        with (s.results/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    folder=s.results/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['capture_pct','ap_pct','ve_pct'],['Object contrast capture','Detection AP','Reconstruction VE']):
        for j,r in enumerate(agg):
            ax.bar(j,r[key],color='#008c82' if r['arm']=='recovery' else '#a7b7c9',hatch='//' if not r['eligible'] else None)
            ax.scatter(np.linspace(j-.12,j+.12,3),[v[key] for v in rows if v['arm']==r['arm']],c='black',s=15)
        ax.set_xticks(range(len(agg)),[r['method'] for r in agg],rotation=35,ha='right',fontsize=7)
        ax.set(title=title,ylabel='%');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Read guard eligibility alongside scores; unrecovered endpoints are diagnostic references');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/('comparison.'+ext),dpi=150,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(11,8))
    for i,seed in enumerate(ctx['recovery']['seeds']):
        for arm in ('control','recovery'):
            st=memory.required(s,f'recovery_s{seed}_{arm}')
            for ax,k in zip(axes[i],['macro_capture','macro_ap','mse']):
                vals=[h['validation']['reconstruction']['mse'] if k=='mse' else 100*h['validation'][k] for h in st['history']]
                ax.plot([h['update'] for h in st['history']],vals,label=LABELS[arm]);ax.set(title=f'Seed{seed}: {k}',xlabel='Recovery updates');ax.grid(alpha=.15)
            axes[i,2].axhline(st['reference_mse']*1.02,color='black',ls='--',lw=.7)
            del st
        axes[i,0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/('learning_curves.'+ext),dpi=150,bbox_inches='tight')
    plt.close(fig)

def diagnostics(ctx,reports):
    s=ctx['store'];out=[]
    for seed in ctx['recovery']['seeds']:
        old=next(r for r in reports if r['summary']['seed']==seed and r['summary']['arm']=='ggf')
        ref={c['category_index']:c for c in old['categories']}
        for arm in ('control','recovery'):
            st=memory.required(s,f'recovery_s{seed}_{arm}');h=st['history'];model=next(r for r in reports if r['summary']['seed']==seed and r['summary']['arm']==arm)
            out.append(dict(seed=seed,arm=arm,selected_update=st['best_update'] if model['summary']['eligible'] else None,
                eligible=model['summary']['eligible'],MSE_limit=st['reference_mse']*1.02,
                rejection_count=sum(not v['object_checkpoint_eligible'] for v in h),
                first_feasible_update=next((v['update'] for v in h if v['object_checkpoint_eligible']),None),
                final_validation=h[-1]['validation'],history=h,
                interpretation='Compare reconstruction restoration and preservation of original endpoint capture/AP; then assess held-out gains versus exact-budget continuation and previousGGF.'))
            del st
            deltas=[dict(category=c['category'],capture_gain=c['contrast']['single_group_capture']-ref[c['category_index']]['contrast']['single_group_capture'],
                         ap_gain=c['detection']['ap']-ref[c['category_index']]['detection']['ap'])
                for c in model['categories'] if c['category_index'] in ctx['cfg']['category_indices']]
            s.write(f'category_changes_s{seed}_{arm}.json',dict(categories=75,all_changes=deltas,
                capture_wins=sum(d['capture_gain']>0 for d in deltas),AP_wins=sum(d['ap_gain']>0 for d in deltas),
                capture_losses=sorted([d for d in deltas if d['capture_gain']<0],key=lambda d:d['capture_gain'])))
    s.write('component_diagnostics.json',dict(starts=out,scope='Observed failures and measured gradient conflicts; not proof of every causal explanation.'))

def export(ctx):
    s=ctx['store'];p=s.output/f'ggf_mask_recovery_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for f in s.results.rglob('*'):
            if f.is_file():z.write(f,'results/'+f.relative_to(s.results).as_posix())
    return p
