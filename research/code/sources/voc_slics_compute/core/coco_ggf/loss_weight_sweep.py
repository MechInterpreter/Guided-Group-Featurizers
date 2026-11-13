"""Loss-weight screening from frozen native BSF starts, before GGF training.

Use the original GGF engine, losses, seeded batches and checkpoint rule.
No source retraining, routing changes, structural edits or final-test access.
"""
import copy
import csv
import gc
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import core, inputs
from coco_baselines import metrics, training
from coco_baselines.storage import identity, serial
from coco_coordinated import run as memory
from . import loss_ablation as lineage, study

GRID = [
    dict(name='control', capture=.05, detection=.01),
    dict(name='contrast_half', capture=.025, detection=.01),
    dict(name='contrast_double', capture=.10, detection=.01),
    dict(name='detection_half', capture=.05, detection=.005),
    dict(name='detection_double', capture=.05, detection=.02),
    dict(name='both_double', capture=.10, detection=.02),
]


def settings():
    return dict(version=1, study='ggf_native_start_loss_weight_sweep', seeds=[21],
        grid=copy.deepcopy(GRID), updates=2500, evaluate_every=250,
        local_save_every=50, diagnostic_grad_every=250,
        bootstrap=200, training_seed_offset=71000,
        include_scratch=True, scratch_initialization_seed=104092,
        recommendation_max_ap_drop=.001, recommendation_max_ve_drop=.001,
        recommendation_max_relative_mse_growth=.02,
        scope='Exploratory weight tuning on previously inspected development images. '
              'No final test; confirm the frozen winner on other starts and new images.')


def validate_settings(cfg):
    if set(cfg)!=set(settings()):
        raise ValueError('Unknown or missing sweep settings')
    seeds=cfg['seeds']
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in (21,22,23) for s in seeds):
        raise ValueError('Use unique saved native seeds from 21, 22, 23')
    if cfg['updates']!=2500 or cfg['evaluate_every']!=250:
        raise ValueError('Keep the successful 2500-update horizon and 250-update audits')
    for k in ('local_save_every','diagnostic_grad_every','bootstrap'):
        if not isinstance(cfg[k],int) or cfg[k]<=0:
            raise ValueError('Positive integer required: '+k)
    if cfg['evaluate_every']%cfg['diagnostic_grad_every']:
        raise ValueError('Gradient diagnostics must include every evaluation')
    if cfg['training_seed_offset']!=71000:
        raise ValueError('Keep the original GGF minibatch seed')
    if type(cfg['include_scratch']) is not bool:
        raise ValueError('include_scratch must be a boolean')
    if cfg['scratch_initialization_seed']!=104092:
        raise ValueError('Keep the predeclared independent random initialization seed')
    for k in ('recommendation_max_ap_drop','recommendation_max_ve_drop','recommendation_max_relative_mse_growth'):
        if not isinstance(cfg[k],(int,float)) or not np.isfinite(cfg[k]) or cfg[k]<0:
            raise ValueError('Invalid recommendation margin: '+k)
    names=[]
    if not cfg['grid']:
        raise ValueError('A control and at least one alternative are required')
    for row in cfg['grid']:
        if set(row)!= {'name','capture','detection'}:
            raise ValueError('Each setting needs name, capture and detection')
        name=row['name']
        if not isinstance(name,str) or not name or not name.replace('_','').isalnum():
            raise ValueError('Simple setting names required')
        if name=='scratch_control':
            raise ValueError('scratch_control is reserved for the random-start diagnostic')
        names.append(name)
        for key in ('capture','detection'):
            v=row[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0:
                raise ValueError('Loss weights must be finite and nonnegative')
    if len(names)<2 or len(set(names))!=len(names):
        raise ValueError('Use at least two unique setting names')
    control=[r for r in cfg['grid'] if r['name']=='control']
    if control!=[GRID[0]]:
        raise ValueError('Retain control: reconstruction 1, contrast .05, detection .01')
    if len({(r['capture'],r['detection']) for r in cfg['grid']})!=len(names):
        raise ValueError('Duplicate weight combinations waste fits')
    return cfg


def prepare(source_output, output, local, source_sha, versions, cfg=None,
            device='cuda', require_drive=True):
    sweep=validate_settings(copy.deepcopy(settings() if cfg is None else cfg))
    if device not in ('cpu','cuda'):
        raise ValueError('Choose cpu or cuda before starting')
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('A GPU runtime is required for cuda')
    local=Path(local)
    source=memory.FrozenStore(Path(source_output)/'checkpoints'/lineage.EXPECTED[:20],local/'read_native')
    if source.run_id!=lineage.EXPECTED or source.config['source_sha']!=lineage.EXPECTED_SOURCE:
        raise ValueError('Wrong original GGF confirmation source; no retraining fallback')
    activation_output=Path(source_output).parent/'AGF_COCO_stratified_confirmation_v1'
    original=memory.FrozenStore(activation_output/'checkpoints'/source.config['parent_run_id'][:20],local/'read_activations')
    if original.run_id!=source.config['parent_run_id']:
        raise ValueError('Activation source lineage differs')
    dest=Path(output).resolve()
    for root in (Path(source_output).resolve(),activation_output.resolve()):
        if dest==root or root in dest.parents or dest in root.parents:
            raise ValueError('All inputs are read-only; choose a separate output folder')
    frozen=memory.required(source,'all_models_frozen')
    if not frozen['complete']:
        raise ValueError('The parent run is incomplete')
    inherited=copy.deepcopy(source.config['config']);base=copy.deepcopy(source.config['native_config'])
    if (base['groups'],base['block_rank'],base['active_groups'],inherited['updates'])!=(1024,3,16,2500):
        raise ValueError('Successful source architecture or training horizon differs')
    if (inherited['lambda_capture'],inherited['lambda_detection'])!=(.05,.01):
        raise ValueError('Successful recipe loss weights differ')
    if inherited['score_ap_weight']!=.2 or inherited['fit_reconstruction_tolerance']!=.02:
        raise ValueError('Original checkpoint selection rule differs')
    cfg_fit=dict(inherited,device=device,updates=sweep['updates'],evaluate_every=sweep['evaluate_every'],
        local_save_every=sweep['local_save_every'],diagnostic_grad_every=sweep['diagnostic_grad_every'],
        audit_image_batch=2,preview_images=0,batch=1024)
    store=inputs.JournalStore(output,dict(config=sweep,fit_config=cfg_fit,source_sha=source_sha,
        versions=versions,device=device,source_run=source.run_id,activation_run=original.run_id,
        original_numerical_versions=source.config['numerical_versions']),local/'rescue',require_drive)
    memory.memory_safe_metrics()
    provenance=json.loads((original.results/'provenance.json').read_text())
    norm=memory.required(original,'normalization')
    if not np.isfinite(norm['mean']).all() or not np.isfinite(norm['scale']) or norm['scale']<=0:
        raise ValueError('Saved training normalization is invalid')
    sets=[set(v) for v in provenance['splits'].values()]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError('Historical splits overlap')
    packs={}
    for split in ('train','select'):
        store.event('loading_verified_activation_cache',split=split)
        packs[split]=memory.normalized_pack(original,split,provenance,norm,local,inherited['extract_chunk'])
    if set(packs['train']['ids'])&set(packs['select']['ids']):
        raise ValueError('Training/development overlap')
    if (len(packs['train']['ids']),len(packs['select']['ids']))!=(2000,400):
        raise ValueError('Expected the same 2000 training and 400 development images')
    if any(tuple(p['x'].shape[1:])!=(196,768) for p in packs.values()):
        raise ValueError('Expected the frozen DINOv3 final-layer 196 by 768 patch cache')
    targets={};starts=[]
    for seed in sweep['seeds']:
        t=memory.required(source,f'targets_s{seed}')
        if sorted(r['category_index'] for r in t)!=inherited['category_indices'] or len(t)!=75:
            raise ValueError('All original 75 training-chosen category mappings are required')
        targets[seed]=t
        row=next(r for r in frozen['models'] if r['seed']==seed and r['arm']=='native')
        st=memory.required(source,row['key'])
        if training.weight_sha(st[row['field']])!=row['weights_sha']:
            raise ValueError('Saved native checkpoint hash differs')
        if set(st[row['field']])!={'W_enc','W_dec','b_enc'}:
            raise ValueError('Need the unchanged native Vanilla BSF state, before GGF training')
        if tuple(st[row['field']]['W_enc'].shape)!=(768,3072) or tuple(st[row['field']]['W_dec'].shape)!=(3072,768):
            raise ValueError('Native dictionary capacity differs')
        del st;gc.collect()
        pair=json.loads((source.results/f'paired_training_s{seed}.json').read_text())
        if pair['common_start_sha']!=row['weights_sha'] or pair['matched_gradient_updates']!=2500:
            raise ValueError('Historical start or matched-training protocol differs')
        starts.append(dict(seed=seed,key=row['key'],field=row['field'],weights_sha=row['weights_sha'],
            expected_batches_sha=pair['minibatches_sha'],target_sha=identity(t),native_pretraining_updates=row['updates']))
    ctx=dict(store=store,source=source,original=original,cfg=cfg_fit,base=base,packs=packs,
        targets=targets,sweep=sweep,starts=starts,local=local,device=device,provenance=provenance,
        stamps=memory.source_snapshot([source,original]))
    store.write('protocol_settings.json',dict(sweep=sweep,fit_recipe=cfg_fit,starts=starts,
        start='Native BSF checkpoints BEFORE GGF training. Fresh zero detector heads and fresh common Adam.',
        losses='Reconstruction weight remains 1. Vary the two supervised weights relative to reconstruction.',
        checkpoint_rule='Original capture + 0.2 AP selection, within 2% relative MSE of the native start.',
        no_test='Only training and development caches are loaded. Final test remains closed.',
        architecture='Fixed groups and native Top-K. No learned routing, reorganization, anchoring, GMM or EM.',
        source_reuse='Reevaluate all settings, including the control, in this environment; never reuse historical metric values.'))
    return ctx


def start(ctx,seed):
    row=next(r for r in ctx['starts'] if r['seed']==seed)
    st=memory.required(ctx['source'],row['key']);w=st[row['field']]
    if training.weight_sha(w)!=row['weights_sha']:
        raise ValueError('Native source weights changed')
    return w,row


def scratch(ctx,seed):
    """Random native architecture; training masks choose its own mappings once."""
    key=f'scratch_initial_s{seed}';cached=ctx['store'].load(key)
    if cached is None:
        ctx['store'].event('scratch_training_mapping_start',seed=seed,images=len(ctx['packs']['train']['ids']),
            note='Training masks only; full original category cohort; no BSF or GGF weights.')
        begun=time.monotonic()
        torch.manual_seed(ctx['sweep']['scratch_initialization_seed']+seed)
        m=training.make_model('vanilla',ctx['base'],ctx['packs']['train']['x'].shape[-1]).to(ctx['device'])
        p=ctx['packs']['train']
        audit=metrics.audit_arrays(m,p['x'],p['masks'],p['ids'],ctx['cfg'],ctx['device'])
        t=core.choose_targets(audit,ctx['cfg'],ctx['provenance']['categories'])
        if sorted(r['category_index'] for r in t)!=ctx['cfg']['category_indices']:
            raise ValueError('Random-start training mapping lacks a category; do not change the cohort')
        # Record shared and inactive category assignments before supervision.
        report=core.category_scores(audit,t,ctx['cfg'])
        cached=dict(weights=training.weights(m),targets=t,weights_sha=training.weight_sha(training.weights(m)),
            target_sha=identity(t),mapping_seconds=time.monotonic()-begun,training_mapping_report=report,
            initialization_seed=ctx['sweep']['scratch_initialization_seed']+seed)
        ctx['store'].save(key,cached);ctx['store'].write(f'assignments/scratch_s{seed}.json',
            {k:v for k,v in cached.items() if k!='weights'})
        del m,audit;gc.collect()
    if training.weight_sha(cached['weights'])!=cached['weights_sha'] or identity(cached['targets'])!=cached['target_sha']:
        raise ValueError('Random-start weights or assignments changed')
    return cached


def fit_context(ctx,row):
    fit=dict(ctx);fit['cfg']=dict(ctx['cfg'],lambda_capture=row['capture'],lambda_detection=row['detection'])
    return fit


def scalar(report):
    return dict(capture=report['macro_capture'],ap=report['macro_ap'],
        ve=report['reconstruction']['ve'],mse=report['reconstruction']['mse'],score=report['score'])


def check_report(report,targets):
    if {r['category_index'] for r in report['categories']}!={t['category_index'] for t in targets}:
        raise ValueError('The evaluation category cohort changed')
    if not all(np.isfinite(v) for v in scalar(report).values()):
        raise FloatingPointError('Nonfinite development metric')
    if any(r['capture'] is None or r['ap'] is None or not np.isfinite([r['capture'],r['ap']]).all()
           for r in report['categories']):
        raise ValueError('Missing category evidence; do not silently change the denominator')


def preflight(ctx):
    s=ctx['store'];old=s.load('preflight')
    if old is not None:return old
    cfg=ctx['cfg'];seed=ctx['sweep']['seeds'][0];initial,row=start(ctx,seed);t=ctx['targets'][seed]
    times={};gradients={};validation_seconds=0.;save_seconds=0.
    x=torch.as_tensor(np.array(ctx['packs']['train']['x'][:cfg['images_per_update']],copy=True),device=ctx['device'])
    masks=ctx['packs']['train']['masks'][:cfg['images_per_update']]
    for weights in ctx['sweep']['grid']:
        model=core.new_model('vanilla',ctx['base'],initial,ctx['device'])
        head=torch.nn.Parameter(torch.zeros(len(t),2,device=ctx['device']))
        opt=torch.optim.Adam(list(model.parameters())+[head],lr=cfg['lr']);elapsed=[]
        for i in range(3):
            study.sync(ctx['device']);begun=time.monotonic();xx=x.reshape(-1,model.d)
            noisy=xx+cfg['snr']*xx.std()*torch.randn_like(xx) if cfg['snr'] else xx
            native,_=model.loss(noisy,target=xx)
            cap,det,_=core.supervised_loss(model,x,masks,t,head,cfg)
            terms=dict(native=native,capture=weights['capture']*cap,detection=weights['detection']*det)
            loss=sum(terms.values())
            if not torch.isfinite(loss):raise FloatingPointError('Invalid preflight objective')
            if i==0:gradients[weights['name']]=study.component_gradients(model,terms)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+[head],cfg['grad_clip'])
            opt.step();model.normalize_decoder();study.sync(ctx['device'])
            if i:elapsed.append(time.monotonic()-begun)
        times[weights['name']]=float(np.median(elapsed))
        if weights['name']=='control':
            begun=time.monotonic();v=core.validation(model,ctx['packs']['select'],t,cfg,ctx['device'])
            check_report(v,t);validation_seconds=time.monotonic()-begun
            begun=time.monotonic();s.save('timing_checkpoint',dict(weights=training.weights(model)))
            save_seconds=time.monotonic()-begun
        del model,head,opt;gc.collect()
    if training.weight_sha(initial)!=row['weights_sha']:
        raise RuntimeError('Preflight mutated the saved native start')
    n=len(ctx['sweep']['seeds']);fits=n*(len(times)+int(ctx['sweep']['include_scratch']));calls=1+math.ceil(cfg['updates']/cfg['evaluate_every'])
    seconds=n*cfg['updates']*(sum(times.values())+int(ctx['sweep']['include_scratch'])*times['control'])+fits*calls*(validation_seconds+save_seconds)
    out=dict(numerical_ok=True,training_fits=fits,updates_per_fit=cfg['updates'],
        total_gradient_updates=fits*cfg['updates'],seconds_per_update=times,
        full_development_seconds=validation_seconds,drive_weights_save_seconds=save_seconds,
        rough_training_and_scheduled_audit_hours=[seconds/3600,2*seconds/3600],
        gradient_balance=gradients,preflight_updates_discarded=True,native_pretraining_fits=0,
        final_test_opened=False,starts_before_ggf=True,
        scratch_included=ctx['sweep']['include_scratch'],
        excludes='Random-start training-mask mapping, cache/Drive stalls, large optimizer saves, final diagnostics and bootstrap. Scenarios, not guaranteed times.')
    s.save('preflight',out);s.write('preflight.json',out);del initial;gc.collect()
    return out


def train(ctx,max_updates=None):
    store=ctx['store'];rows=[];proofs=[]
    for seed in ctx['sweep']['seeds']:
        initial,source_row=start(ctx,seed);paired=[]
        for weights in ctx['sweep']['grid']:
            key=f's{seed}_{weights["name"]}'
            store.event('weight_fit_start',seed=seed,key=key,weights=weights,target=ctx['cfg']['updates'])
            st=study.fit(fit_context(ctx,weights),key,initial,ctx['targets'][seed],True,
                ctx['sweep']['training_seed_offset']+seed,max_updates=max_updates)
            for h in st['history']:check_report(h['validation'],ctx['targets'][seed])
            if st['initial_sha']!=source_row['weights_sha'] or st['target_sha']!=identity(ctx['targets'][seed]):
                raise RuntimeError('Start weights or training assignments differ')
            if st['updates']==ctx['cfg']['updates'] and st['sampled_batches_sha']!=source_row['expected_batches_sha']:
                raise RuntimeError('Minibatches differ from the successful GGF training protocol')
            paired.append((st['updates'],st['sampled_batches_sha']))
            rows.append(dict(seed=seed,key=key,setting=weights['name'],reconstruction_weight=1.,
                capture_weight=weights['capture'],detection_weight=weights['detection'],
                updates=st['updates'],selected_update=st['best_update'],weights_sha=training.weight_sha(st['best']),
                source_weight_sha=st['initial_sha'],target_sha=st['target_sha'],
                native_pretraining_updates=source_row['native_pretraining_updates'],
                comparison_scope='Identical frozen native pretraining and guided-stage budgets across loss weights.'))
            store.write('histories/'+key+'.json',st['history'])
            del st;gc.collect()
        if len(set(paired))!=1:raise RuntimeError('Fits saw different minibatches or update counts')
        proofs.append(dict(seed=seed,updates=paired[0][0],batches_sha=paired[0][1],
            common_native_start_sha=source_row['weights_sha'],heads='Fresh zero heads in every fit',
            loss_weights_only_changed=True))
        if ctx['sweep']['include_scratch']:
            random=scratch(ctx,seed);scratch_ctx=dict(fit_context(ctx,GRID[0]),targets=dict(ctx['targets']))
            scratch_ctx['targets'][seed]=random['targets']
            key=f's{seed}_scratch_control'
            st=study.fit(scratch_ctx,key,random['weights'],random['targets'],True,
                ctx['sweep']['training_seed_offset']+seed,max_updates=max_updates)
            for h in st['history']:check_report(h['validation'],random['targets'])
            if (st['updates'],st['sampled_batches_sha'])!=paired[0]:
                raise RuntimeError('Random-start arm saw different guided-stage batches')
            rows.append(dict(seed=seed,key=key,setting='scratch_control',reconstruction_weight=1.,
                capture_weight=.05,detection_weight=.01,updates=st['updates'],selected_update=st['best_update'],
                weights_sha=training.weight_sha(st['best']),source_weight_sha=st['initial_sha'],
                target_sha=st['target_sha'],native_pretraining_updates=0,
                comparison_scope='Same guided-stage updates; lacks the pretrained arms native-training budget. Not a total-compute-matched test.'))
            ctx['store'].write('histories/'+key+'.json',st['history'])
            del random,st;gc.collect()
        del initial;gc.collect()
    out=dict(complete=all(r['updates']==ctx['cfg']['updates'] for r in rows),models=rows,
        pairing_proofs=proofs,final_test_opened=False)
    store.save('models_frozen',out);store.write('models_frozen.json',out)
    memory.assert_immutable(ctx['stamps']);return out


def comparison_guard(candidate,control,cfg):
    reasons=[]
    if candidate['ap']<control['ap']-cfg['recommendation_max_ap_drop']:reasons.append('AP_drop_vs_control')
    if candidate['ve']<control['ve']-cfg['recommendation_max_ve_drop']:reasons.append('VE_drop_vs_control')
    if candidate['mse']>control['mse']*(1+cfg['recommendation_max_relative_mse_growth']):reasons.append('MSE_growth_vs_control')
    return dict(passed=not reasons,reasons=reasons,
        note='Predeclared exploratory trade-off margins; not calibrated statistical equivalence tests.')


def pareto(rows):
    # All three desirable metrics increase. Do not hide trade-offs inside one number.
    return [r['setting'] for r in rows if not any(all(q[k]>=r[k] for k in ('capture','ap','ve'))
        and any(q[k]>r[k] for k in ('capture','ap','ve')) for q in rows)]


def summarize(ctx):
    s=ctx['store'];frozen=s.load('models_frozen')
    if frozen is None or not frozen['complete']:
        raise ValueError('Finish the entire sweep before choosing weights')
    rows=[];diagnostics=[];categories=[];comparisons=[]
    for item in frozen['models']:
        st=memory.required(s,item['key']);picked=next(h for h in st['history'] if h['update']==st['best_update'])
        r=dict(item,**scalar(picked['validation']),final=scalar(st['history'][-1]['validation']),
            native=scalar(st['history'][0]['validation']),train_seconds=st['seconds'],
            guard_rejections=sum(h.get('object_checkpoint_eligible') is False for h in st['history']))
        rows.append(r)
        for h in st['history']:
            for c in h['validation']['categories']:
                categories.append(dict(seed=item['seed'],setting=item['setting'],update=h['update'],**c))
        diagnostics.append(dict(seed=item['seed'],setting=item['setting'],selected_update=st['best_update'],
            selected_is_native_fallback=st['best_update']==0,guard_rejections=r['guard_rejections'],
            selected_validation=picked['validation'],final_validation=st['history'][-1]['validation'],
            gradient_diagnostics=[dict(update=h['update'],raw_native=h['native_loss'],raw_capture=h['capture_loss'],
                raw_detection=h['detection_loss'],total=h['total_loss'],parameter_gradients=h['parameter_gradient_norms'],
                component_gradients=h['component_gradients']) for h in st['history'] if h.get('component_gradients')]))
        del st;gc.collect()
    recommendations=[]
    for seed in ctx['sweep']['seeds']:
        subset=[r for r in rows if r['seed']==seed];control=next(r for r in subset if r['setting']=='control')
        for r in subset:
            guard=comparison_guard(r,control,ctx['sweep'])
            r['quality_vs_control']=guard
            comparisons.append(dict(seed=seed,setting=r['setting'],capture_gain=r['capture']-control['capture'],
                ap_gain=r['ap']-control['ap'],ve_gain=r['ve']-control['ve'],quality_vs_control=guard))
        feasible=[r for r in subset if r['quality_vs_control']['passed'] and r['setting']!='scratch_control']
        winner=max(feasible,key=lambda r:(r['capture'],r['ap'],r['ve']))
        recommendations.append(dict(seed=seed,setting=winner['setting'],weights=dict(reconstruction=1.,
            capture=winner['capture_weight'],detection=winner['detection_weight']),
            capture_gain_vs_control=winner['capture']-control['capture'],pareto_settings=pareto([r for r in subset if r['setting']!='scratch_control']),
            next='Freeze these weights, then confirm on other starts and newly reserved images.'))
    # Fixed training mappings, whole-image resampling, no test and no model re-selection in bootstrap.
    from . import owner_routing
    intervals=[]
    for seed in ctx['sweep']['seeds']:
        a=next(r for r in rows if r['seed']==seed and r['setting']=='control')
        state=memory.required(s,a['key']);control=core.new_model('vanilla',ctx['base'],state['best'],ctx['device'])
        del state;gc.collect()
        for r in [r for r in rows if r['seed']==seed and r['setting']!='control']:
            st=memory.required(s,r['key']);model=core.new_model('vanilla',ctx['base'],st['best'],ctx['device']);del st;gc.collect()
            probe=dict(ctx,cfg=dict(ctx['sweep'],training_seed=ctx['sweep']['training_seed_offset']),eval_cfg=ctx['cfg'])
            # Helper expects model.targets; native model inference remains unchanged.
            model.targets=scratch(ctx,seed)['targets'] if r['setting']=='scratch_control' else ctx['targets'][seed]
            control.targets=ctx['targets'][seed]
            ci=owner_routing.paired_capture_interval(probe,control,model,seed)
            intervals.append(dict(seed=seed,setting=r['setting'],**ci,
                interpretation='Descriptive development interval; excludes weight/checkpoint selection bias and is not fresh confirmation.'))
            del model;gc.collect()
        del control;gc.collect()
    summary=dict(complete=True,models=rows,comparisons=comparisons,recommendations=recommendations,
        development_intervals=intervals,pairing_proofs=frozen['pairing_proofs'],final_test_opened=False,
        scope=ctx['sweep']['scope'],architecture='Fixed-group GGF; every fit starts before GGF supervision.',
        metric_scope='Object-vs-background contrast capture, mask detection AP and activation VE. '
                     'No complete manifold recovery or exact feature-splitting claim.',
        diagnostics_scope='Measured gradients, learning curves, guard rejections and category losses; not a guarantee of identifying every causal failure.')
    s.write('summary.json',summary);s.save('summary',summary)
    s.write('component_diagnostics.json',diagnostics)
    s.write('checkpoint_index.json',dict(root=str(s.root),models=frozen['models']))
    flat=[]
    for r in rows:
        flat.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))})
    for name,records in [('weight_comparison.csv',flat),('per_category_learning_curves.csv',categories)]:
        with (s.results/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    plots(ctx,rows)
    memory.assert_immutable(ctx['stamps']);return summary


def plots(ctx,rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    folder=ctx['store'].results/'figures';folder.mkdir(exist_ok=True)
    for seed in ctx['sweep']['seeds']:
        rr=[r for r in rows if r['seed']==seed]
        fig=Figure(figsize=(13,4));FigureCanvasAgg(fig);axes=fig.subplots(1,3)
        for ax,key,title in zip(axes,('capture','ap','ve'),('Object contrast capture','Detection AP','Reconstruction VE')):
            ax.bar([r['setting'] for r in rr],[100*r[key] for r in rr],color=['#596f7c']+['#008e86']*(len(rr)-1))
            ax.tick_params(axis='x',labelrotation=35);ax.set_title(title);ax.set_ylabel('%');ax.grid(axis='y',alpha=.15)
        fig.suptitle(f'Native-start weight screen: seed {seed}, development only');fig.tight_layout()
        fig.savefig(folder/f'weight_comparison_s{seed}.png',dpi=160)
        fig=Figure(figsize=(13,4));FigureCanvasAgg(fig);axes=fig.subplots(1,3)
        for r in rr:
            st=ctx['store'].load(r['key']);history=st['history']
            for ax,key,title in zip(axes,('capture','ap','ve'),('Object contrast capture','Detection AP','Reconstruction VE')):
                ax.plot([h['update'] for h in history],[100*scalar(h['validation'])[key] for h in history],label=r['setting'])
                ax.set(title=title,xlabel='GGF updates',ylabel='%');ax.grid(alpha=.15)
            del st;gc.collect()
        axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(folder/f'learning_curves_s{seed}.png',dpi=160)


def export(ctx):
    s=ctx['store'];inventory=[]
    for p in sorted(s.results.rglob('*')):
        if p.is_file() and p.name!='artifact_manifest.json':
            inventory.append(dict(path=p.relative_to(s.results).as_posix(),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    s.write('artifact_manifest.json',dict(run_id=s.run_id,files=inventory,checkpoint_root=str(s.root),final_test_opened=False))
    root=Path('/content') if s.require_drive and Path('/content').exists() else s.output
    dest=root/f'ggf_loss_weight_sweep_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in s.results.rglob('*'):
            if p.is_file():z.write(p,'results/'+p.relative_to(s.results).as_posix())
    return dest
