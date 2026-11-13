"""Width/activity trade-offs with matched total and active direction budgets."""
import copy
import csv
import gc
import time
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import eigh
from coco_baselines import metrics, training
from coco_baselines.storage import identity
from coco_adaptation import core, inputs
from . import capacity_sweep as previous, loss_weight_sweep as lineage, study

GRID = [dict(key='b4_k12',rank=4,active=12,groups=768),
        dict(key='b8_k6',rank=8,active=6,groups=384),
        dict(key='b16_k3',rank=16,active=3,groups=192)]

def settings():
    return dict(version=1,study='ggf_matched_capacity_wider_groups',seeds=[21],
        grid=copy.deepcopy(GRID),total_directions=3072,active_coordinates=48,
        native_epochs=40,updates=2500,evaluate_every=250,local_save_every=50,
        images_per_update=8,lambda_capture=.05,lambda_detection=.01,
        min_capture_gain=.005,max_ap_drop=.001,max_ve_drop=.001,
        headroom_widths=[3,4,8,16],
        scope='One-seed screen on previously inspected development images; no final-test access or structural adaptation.')

def validate(cfg):
    if set(cfg)!=set(settings()):raise ValueError('Unknown or missing settings')
    frozen=['total_directions','active_coordinates','native_epochs','updates','evaluate_every',
            'local_save_every','images_per_update','lambda_capture','lambda_detection',
            'min_capture_gain','max_ap_drop','max_ve_drop','headroom_widths','grid']
    for k in frozen:
        if cfg[k]!=settings()[k]:raise ValueError('Keep the predeclared matched experiment: '+k)
    if not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']) or any(s not in (21,22,23) for s in cfg['seeds']):
        raise ValueError('Use unique saved seed IDs 21,22,23')
    for r in cfg['grid']:
        if r['groups']*r['rank']!=cfg['total_directions'] or r['active']*r['rank']!=cfg['active_coordinates']:
            raise ValueError('Total/active direction budgets differ')
    return cfg

def prepare(source_output,output,local,sha,versions,cfg,device,require_drive=True):
    validate(cfg)
    old=lineage.settings();old.update(seeds=cfg['seeds'],include_scratch=False)
    ctx=lineage.prepare(source_output,Path(output)/'input_verification',local,sha,versions,old,device,require_drive)
    ctx['cfg']=dict(ctx['cfg'],device=device,updates=2500,evaluate_every=250,
        local_save_every=50,images_per_update=8,diagnostic_grad_every=250)
    ctx['sweep']=copy.deepcopy(cfg);ctx['grid']=copy.deepcopy(cfg['grid'])
    ctx['store']=inputs.JournalStore(output,dict(config=cfg,source_sha=sha,versions=versions,
        device=device,source_run=ctx['source'].run_id,activation_run=ctx['original'].run_id),
        Path(local)/'wider_rescue',require_drive)
    ctx['store'].write('protocol_settings.json',dict(config=cfg,
        control='Reevaluate saved successful 1024x3 Top16 GGF on the same 400 development images; no retraining.',
        variants='Fresh native 40 epochs and 2500 GGF updates per architecture; original Goodfire Vanilla class/init/loss.',
        matching='3072 total decoder directions, 48 maximum selected coordinates and equal parameter counts; group count varies.',
        gate='Original capture + .2 AP checkpoint rule with 2% MSE allowance relative to each architecture native start.',
        no_test='Load only saved 2000 training and 400 inspected development images.',
        diagnostics='Geometric ceilings, training-fitted projection, native/selected/final scores, per-category harms, sharing and AP profiles.',
        limitations='Different group counts and widths are inseparable here. Same epochs/updates does not mean equal runtime. No semantic-manifold or exact splitting-rate claim.'))
    return ctx

def control(ctx,seed):return previous.control(ctx,seed)

def branch(ctx,spec):
    return dict(ctx,base=dict(ctx['base'],groups=spec['groups'],block_rank=spec['rank'],
        active_groups=spec['active'],epochs=ctx['sweep']['native_epochs']),
        cfg=dict(ctx['cfg'],groups=spec['groups'],block_rank=spec['rank'],active_groups=spec['active']))

def contrasts(pack,cfg):
    """Exact audit contrast convention without allocating group codes or a model."""
    values=[];objects=[];skips=dict(foreground=0,background=0)
    for i,mm in enumerate(pack['masks']):
        known=np.asarray(mm['crowd'])<=cfg['negative_fraction']
        for obj in mm['instances']:
            fg=np.asarray(obj['fraction'],np.float64)*known
            bg=(mm['fraction'][:,obj['category']].astype(float)/255<=cfg['negative_fraction']) & known
            if fg.sum()<cfg['min_object_patch_mass']:skips['foreground']+=1;continue
            if bg.sum()<cfg['min_background_patches']:skips['background']+=1;continue
            w=fg/fg.sum()-bg/bg.sum()
            values.append(np.asarray(w@pack['x'][i],np.float32))
            objects.append(dict(image_id=int(pack['ids'][i]),image_index=i,annotation_id=int(obj['annotation_id']),
                category=int(obj['category']),foreground_mass=float(fg.sum()),background_patches=int(bg.sum())))
    return dict(dx=np.asarray(values,np.float32).reshape(-1,pack['x'].shape[-1]),objects=objects,skips=skips)

def spectrum(train,development,widths):
    """No mean subtraction: the capture denominator is contrast energy about zero."""
    a=np.asarray(train,np.float64);b=np.asarray(development,np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or not len(a) or not len(b):
        raise ValueError('Nonempty compatible contrasts required')
    d=a.shape[1];q=min(max(widths),d);ga=a.T@a;gb=b.T@b
    ta=float(np.trace(ga));tb=float(np.trace(gb))
    if min(ta,tb)<=1e-12:raise ValueError('Zero contrast energy')
    va,u=eigh(ga,subset_by_index=(d-q,d-1));vb=eigh(gb,subset_by_index=(d-q,d-1),eigvals_only=True)
    va=np.maximum(va[::-1],0);vb=np.maximum(vb[::-1],0);u=u[:,::-1]
    rows=[]
    for width in widths:
        k=min(width,d);basis=u[:,:k]
        rows.append(dict(width=width,train_optimistic_ceiling=float(np.clip(va[:k].sum()/ta,0,1)),
            development_optimistic_ceiling=float(np.clip(vb[:k].sum()/tb,0,1)),
            development_capture_with_training_basis=float(np.clip(np.square(b@basis).sum()/tb,0,1))))
    return dict(rows=rows,train_objects=len(a),development_objects=len(b),train_energy=ta,development_energy=tb)

def headroom(ctx):
    s=ctx['store'];old=s.load('geometric_headroom')
    if old is not None:return old
    packs={}
    for name in ('train','select'):
        key='contrast_vectors_'+name;packs[name]=s.load(key)
        if packs[name] is None:
            s.event('headroom_contrasts_start',split=name)
            packs[name]=contrasts(ctx['packs'][name],ctx['cfg']);s.save(key,packs[name])
    rows=[];cohort=ctx['cfg']['category_indices']
    for c in cohort:
        key=f'headroom_category_{c}';r=s.load(key)
        if r is None:
            groups={n:np.array([o['category']==c for o in p['objects']],bool) for n,p in packs.items()}
            a=packs['train']['dx'][groups['train']];b=packs['select']['dx'][groups['select']]
            s.event('headroom_category_start',category_index=c)
            if not len(a) or not len(b):raise ValueError('Missing supported category in headroom: '+str(c))
            r=dict(category_index=c,category=ctx['provenance']['categories'][c]['name'],
                train_images=len({o['image_id'] for o in packs['train']['objects'] if o['category']==c}),
                development_images=len({o['image_id'] for o in packs['select']['objects'] if o['category']==c}),
                **spectrum(a,b,ctx['sweep']['headroom_widths']))
            s.save(key,r);s.event('headroom_category_complete',category_index=c)
        rows.append(r)
    out=dict(categories=rows,macro=[dict(width=w,**{k:float(np.mean([next(x for x in r['rows'] if x['width']==w)[k] for r in rows]))
        for k in ('train_optimistic_ceiling','development_optimistic_ceiling','development_capture_with_training_basis')})
        for w in ctx['sweep']['headroom_widths']],
        formula='sum of top b eigenvalues of X.T@X / trace(X.T@X); X contains uncentered object-minus-comparison patch contrasts.',
        meaning='Optimistic rank-b linear-subspace ceiling on each finite sample set. Ignores gating, encoder, joint reconstruction, dictionary sharing and optimization.',
        caution='The development-fitted ceiling is an audit bound, never a trained representation. The training-fitted projection tests geometry transfer; neither establishes population headroom or manifold recovery.')
    s.save('geometric_headroom',out);s.write('geometric_headroom.json',out)
    del packs;gc.collect();return out

def preflight(ctx):
    s=ctx['store'];old=s.load('preflight')
    if old is not None:return old
    rows=[];device=ctx['device'];pack=ctx['packs']['train']
    x=torch.as_tensor(np.array(pack['x'][:8],copy=True),device=device);masks=pack['masks'][:8]
    for spec in ctx['grid']:
        c=branch(ctx,spec);m=training.make_model('vanilla',c['base']).to(device)
        # Timing only. Actual training assignments are selected after native fitting.
        targets=copy.deepcopy(ctx['targets'][ctx['sweep']['seeds'][0]])
        for t in targets:
            t['contrast_group']%=spec['groups'];t['detector_group']%=spec['groups']
        head=torch.nn.Parameter(torch.zeros(len(targets),2,device=device))
        opt=torch.optim.Adam(list(m.parameters())+[head],lr=c['cfg']['lr']);times=[]
        for guided in (False,True):
            study.sync(device);started=time.monotonic()
            for _ in range(3):
                xx=x.reshape(-1,x.shape[-1]);loss,_=m.loss(xx+c['base']['snr']*xx.std()*torch.randn_like(xx),target=xx)
                if guided:
                    cap,det,_=core.supervised_loss(m,x,masks,targets,head,c['cfg']);loss=loss+.05*cap+.01*det
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preflight loss: '+spec['key'])
                opt.zero_grad();loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(list(m.parameters())+[head],c['cfg']['grad_clip'])
                if not torch.isfinite(grad):raise FloatingPointError('Nonfinite preflight gradient: '+spec['key'])
                opt.step();m.normalize_decoder()
                if any(not bool(torch.isfinite(p).all()) for p in m.parameters()):
                    raise FloatingPointError('Nonfinite preflight parameters: '+spec['key'])
            study.sync(device);times.append((time.monotonic()-started)/3)
        updates=len(pack['x'].reshape(-1,x.shape[-1]))//c['base']['batch']*c['base']['epochs']
        native=times[0]*c['base']['batch']/len(xx)
        rows.append(dict(**spec,native_updates=updates,guided_updates=2500,
            parameters=sum(p.numel() for p in m.parameters()),native_seconds_per_update=native,
            guided_seconds_per_update=times[1],gradient_hours=(native*updates+times[1]*2500)/3600))
        del m,head,opt;gc.collect()
    total=sum(r['gradient_hours'] for r in rows)*len(ctx['sweep']['seeds'])
    out=dict(numerical_ok=all(np.isfinite(r['gradient_hours']) for r in rows),measurements=rows,
        rough_gradient_hours=[total,total*2],fits=len(rows)*len(ctx['sweep']['seeds']),
        exclusions='Headroom, mappings, validation, specificity, checkpointing, input reads and Drive stalls. Not an ETA.',
        timing_only_assignments=True,actual_assignment_policy='Training-mask mappings chosen separately for each native architecture before guided training.')
    s.save('preflight',out);s.write('preflight.json',out);return out

def qualification(a,b,cfg):
    if any(not np.isfinite(v[k]) for v in (a,b) for k in ('capture','ap','ve')):
        raise ValueError('Nonfinite comparison metrics')
    gains={k:float(a[k]-b[k]) for k in ('capture','ap','ve')};reasons=[]
    if gains['capture']<cfg['min_capture_gain']:reasons.append('capture_gain_below_0_5_pp')
    if gains['ap']<-cfg['max_ap_drop']:reasons.append('AP_drop_exceeds_0_1_pp')
    if gains['ve']<-cfg['max_ve_drop']:reasons.append('VE_drop_exceeds_0_1_pp')
    return dict(passed=not reasons,gains=gains,reasons=reasons)

def checked_report(report,cohort):
    """Never improve a macro score by silently dropping a difficult category."""
    rows=report['categories']
    if sorted(r['category_index'] for r in rows)!=sorted(cohort):
        raise ValueError('Evaluation category cohort differs')
    if any(r[k] is None or not np.isfinite(r[k]) for r in rows for k in ('capture','ap')):
        raise ValueError('Unsupported/nonfinite category; do not change the evaluation cohort')
    return report

def specificity(model,pack,targets,cfg,device):
    audit=metrics.audit_arrays(model,pack['x'],pack['masks'],pack['ids'],cfg,device)
    y,valid=metrics.binary_labels(audit,cfg);cols=sorted({t['detector_group'] for t in targets})
    matrix=metrics.ap_matrix(audit['heat'][:,cols],y,valid);lookup={g:i for i,g in enumerate(cols)}
    cohort=cfg['category_indices'];role_counts=Counter(t['contrast_group'] for t in targets);out=[]
    for t in targets:
        g=t['detector_group'];v=matrix[lookup[g]];order=sorted(cohort,key=lambda c:-(v[c] if np.isfinite(v[c]) else -1))
        out.append(dict(category=t['category'],category_index=t['category_index'],detector_group=g,
            contrast_group=t['contrast_group'],categories_sharing_contrast_assignment=role_counts[t['contrast_group']],
            top_category_APs=[dict(category=next(t2['category'] for t2 in targets if t2['category_index']==c),
                category_index=c,ap=None if not np.isfinite(v[c]) else float(v[c])) for c in order[:5]]))
    return dict(categories=out,unique_contrast_groups=len(role_counts),unique_detector_groups=len(cols),
        meaning='Category-response profiles and shared training assignments. Co-occurring objects can share responses; neither is an exact feature-merging or splitting rate.')

def run(ctx):
    s=ctx['store'];old=s.load('wider_complete')
    if old is not None:return old
    controls={seed:control(ctx,seed) for seed in ctx['sweep']['seeds']};rows=[]
    for spec in ctx['grid']:
        c=branch(ctx,spec)
        for seed in ctx['sweep']['seeds']:
            key=f'{spec["key"]}_s{seed}';done=s.load('completed_'+key)
            if done is not None:rows.append(done);continue
            s.event('variant_start',key=key,setting=spec['key'],rank=spec['rank'],active=spec['active'],groups=spec['groups'])
            d=ctx['packs']['train']['x'].shape[-1]
            x=ctx['packs']['train']['x'].reshape(-1,d);val=ctx['packs']['select']['x'].reshape(-1,d)
            native=training.fit(s,'native_'+key,'vanilla',x,val,c['base'],seed,c['base']['lrs'][0],ctx['device'])
            initial=native['best']
            if initial is None:raise RuntimeError('No native checkpoint')
            targets=s.load('targets_'+key)
            if targets is None:
                s.event('training_mapping_start',key=key)
                m=core.new_model('vanilla',c['base'],initial,ctx['device']);pack=ctx['packs']['train']
                audit=metrics.audit_arrays(m,pack['x'],pack['masks'],pack['ids'],c['cfg'],ctx['device'])
                targets=core.choose_targets(audit,c['cfg'],ctx['provenance']['categories'])
                if sorted(t['category_index'] for t in targets)!=c['cfg']['category_indices']:raise ValueError('Missing supported category')
                s.save('targets_'+key,targets);s.write('targets/'+key+'.json',targets)
                del m,audit;gc.collect();s.event('training_mapping_complete',key=key)
            guided=study.fit(c,'ggf_'+key,initial,targets,True,71000+seed)
            m=core.new_model('vanilla',c['base'],guided['best'],ctx['device'])
            selected=core.validation(m,ctx['packs']['select'],targets,c['cfg'],ctx['device'])
            spe=s.load('specificity_selected_'+key)
            if spe is None:
                spe=specificity(m,ctx['packs']['select'],targets,c['cfg'],ctx['device']);s.save('specificity_selected_'+key,spe)
            m.load_state_dict(guided['weights']);final=core.validation(m,ctx['packs']['select'],targets,c['cfg'],ctx['device'])
            sf=s.load('specificity_final_'+key)
            if sf is None:
                sf=spe if guided['best_update']==guided['updates'] else specificity(m,ctx['packs']['select'],targets,c['cfg'],ctx['device'])
                s.save('specificity_final_'+key,sf)
            a=previous.scalar(selected);f=previous.scalar(final);b=controls[seed]['metrics'];q=qualification(a,b,ctx['sweep'])
            for report in (selected,final,controls[seed]['report']):checked_report(report,c['cfg']['category_indices'])
            fm=guided['initial_mse'];final_eligible=f['mse']<=fm*(1+c['cfg']['fit_reconstruction_tolerance'])
            old_categories={r['category_index']:r for r in controls[seed]['report']['categories']}
            per=[dict(category=u['category'],category_index=u['category_index'],**{k:u[k]-old_categories[u['category_index']][k] for k in ('capture','ap')})
                 for u in selected['categories']]
            init_report=guided['history'][0]['validation']
            done=dict(seed=seed,setting=spec['key'],**{k:spec[k] for k in ('rank','groups','active')},
                dictionary_atoms=spec['groups']*spec['rank'],max_selected_coordinates=spec['active']*spec['rank'],
                parameters=sum(p.numel() for p in m.parameters()),native_updates=native['updates'],guided_updates=guided['updates'],
                native_selected_update=native['best_update'],selected_update=guided['best_update'],
                native_metrics=previous.scalar(init_report),metrics=a,final_metrics=f,
                native_report=init_report,selected_report=selected,final_report=final,
                qualified=q['passed'],gains_vs_frozen_success=q['gains'],rejection_reasons=q['reasons'],
                final_vs_control=qualification(f,b,ctx['sweep']),final_checkpoint_eligible=bool(final_eligible),
                final_relative_native_MSE_growth=f['mse']/fm-1,original_native_MSE_guard=.02,
                guard_rejections=sum(not r.get('object_checkpoint_eligible',True) for r in guided['history']),
                per_category_gains=per,specificity_selected=spe,specificity_final=sf,
                selected_weights_sha=training.weight_sha(guided['best']),initial_sha=training.weight_sha(initial),
                final_weights_sha=training.weight_sha(guided['weights']),training_batch_sha=guided['sampled_batches_sha'],
                native_seconds=native['seconds'],guided_seconds=guided['seconds'],
                same_capacity=True,same_training_budget=True,same_compute=False,scope=ctx['sweep']['scope'])
            if done['parameters']!=controls[seed]['parameters']:raise RuntimeError('Parameter capacity unexpectedly differs')
            s.save('completed_'+key,done);s.write('comparisons/'+key+'.json',done);rows.append(done)
            s.event('variant_complete',key=key,completed=len(rows),total=len(ctx['grid'])*len(controls),
                selected_metrics=a,final_metrics=f,qualified=q['passed'],final_eligible=bool(final_eligible))
            del m,native,guided,initial;gc.collect()
            if ctx['device'].startswith('cuda'):torch.cuda.empty_cache()
    for seed in ctx['sweep']['seeds']:
        if len({r['training_batch_sha'] for r in rows if r['seed']==seed})!=1:
            raise RuntimeError('Architectures did not use identical guided minibatch schedules')
        expected=next((r['expected_batches_sha'] for r in ctx.get('starts',[]) if r['seed']==seed),None)
        if expected is not None and any(r['training_batch_sha']!=expected for r in rows if r['seed']==seed):
            raise RuntimeError('Guided minibatches differ from the successful saved GGF recipe')
    out=dict(complete=True,controls=list(controls.values()),models=rows,no_test_access=True,
        qualified_settings=[dict(seed=r['seed'],setting=r['setting']) for r in rows if r['qualified']],
        interpretation='Primary winners must satisfy the original native-relative checkpoint guard. Final outcomes remain diagnostic; do not retrofit the guard to accept them.')
    lineage.memory.assert_immutable(ctx['stamps'])
    s.save('wider_complete',out);s.write('summary.json',out);return out

def figures(ctx,summary,room):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=ctx['store'];folder=s.results/'figures';folder.mkdir(exist_ok=True)
    for seed in ctx['sweep']['seeds']:
        control_row=next(r for r in summary['controls'] if r['seed']==seed)
        rows=[control_row]+[r for r in summary['models'] if r['seed']==seed];labels=['Saved 3/16']+[r['setting'] for r in rows[1:]]
        fig,axes=plt.subplots(1,3,figsize=(12,3.6))
        for ax,k,title in zip(axes,('capture','ap','ve'),('Object contrast capture','Detection AP','Reconstruction VE')):
            x=np.arange(len(rows));ax.bar(x-.16,[100*r['metrics'][k] for r in rows],.32,label='Selected',color='#008578')
            ax.bar(x+.16,[100*r.get('final_metrics',r['metrics'])[k] for r in rows],.32,label='Final (diagnostic)',color='#a8c8dc')
            ax.set_xticks(x,labels);ax.set_title(title);ax.set_ylabel('%');ax.spines[['top','right']].set_visible(False)
        axes[0].legend(fontsize=8);fig.suptitle('Matched capacity: 3072 dictionary directions, at most 48 active coordinates; development only',fontsize=10)
        fig.tight_layout();fig.savefig(folder/f'comparison_s{seed}.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.5))
    for k,label in [('train_optimistic_ceiling','Training optimistic ceiling'),('development_optimistic_ceiling','Development optimistic ceiling'),('development_capture_with_training_basis','Training-fitted basis on development')]:
        ax.plot([r['width'] for r in room['macro']],[100*r[k] for r in room['macro']],marker='o',label=label)
    ax.set(xlabel='Directions in one group',ylabel='Mean object-contrast capture (%)',title='Geometric headroom, without encoder or gating constraints')
    ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(folder/'headroom.png',dpi=180);plt.close(fig)
    flat=[]
    for r in summary['models']:
        for checkpoint in ('metrics','final_metrics'):
            flat.append(dict(seed=r['seed'],setting=r['setting'],checkpoint=checkpoint,**r[checkpoint],
                selected_update=r['selected_update'],qualified=r['qualified'],final_eligible=r['final_checkpoint_eligible']))
    with open(s.results/'comparison.csv','w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    return folder

def export(ctx):
    s=ctx['store'];p=s.output/f'ggf_wider_groups_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for q in s.results.rglob('*'):
            if q.is_file():z.write(q,'results/'+q.relative_to(s.results).as_posix())
    return p
