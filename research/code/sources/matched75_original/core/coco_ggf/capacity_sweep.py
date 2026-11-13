"""Change GGF width/activity; reuse frozen successful GGF as the control.

New variants repeat native 40-epoch pretraining and 2500 guided updates.
No historical baseline/control retraining, no SLiCS, no final-test access.
"""
import copy,gc,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import metrics,training
from coco_baselines.storage import identity
from coco_adaptation import core,inputs
from coco_coordinated import run as memory
from . import loss_weight_sweep as lineage,study

def settings():
    return dict(version=1,study='ggf_width_activity_screen',seeds=[21],groups=1024,
        block_sizes=[2,3,4],active_groups=[8,16,32],native_epochs=40,updates=2500,
        evaluate_every=250,local_save_every=50,images_per_update=8,
        lambda_capture=.05,lambda_detection=.01,min_capture_gain=.005,
        max_ap_drop=.001,max_ve_drop=.001,
        scope='Exploratory development sweep; confirm any selected setting on new starts and a reserved holdout.')

def prepare(source_output,output,local,sha,versions,cfg,device,require_drive=True):
    if set(cfg)!=set(settings()) or cfg['groups']!=1024 or cfg['native_epochs']!=40 or cfg['updates']!=2500:
        raise ValueError('Keep the successful dictionary group count and full training horizons')
    if cfg['lambda_capture']!=.05 or cfg['lambda_detection']!=.01:raise ValueError('Isolate architecture; retain successful loss weights')
    if len(set(cfg['block_sizes']))!=len(cfg['block_sizes']) or len(set(cfg['active_groups']))!=len(cfg['active_groups']):raise ValueError('Duplicate sweep settings')
    if not cfg['block_sizes'] or any(type(r) is not int or r<=0 for r in cfg['block_sizes']):raise ValueError('Positive block sizes')
    if not cfg['active_groups'] or any(type(k) is not int or not 1<=k<=1024 for k in cfg['active_groups']):raise ValueError('Valid active-group counts')
    check=lineage.settings();check['seeds']=cfg['seeds'];check['include_scratch']=False
    ctx=lineage.prepare(source_output,Path(output)/'input_verification',local,sha,versions,check,device,require_drive)
    ctx['cfg']=dict(ctx['cfg'],device=device,images_per_update=cfg['images_per_update'],updates=cfg['updates'],
        evaluate_every=cfg['evaluate_every'],local_save_every=cfg['local_save_every'],
        diagnostic_grad_every=cfg['evaluate_every'],lambda_capture=.05,lambda_detection=.01)
    ctx['store']=inputs.JournalStore(output,dict(config=cfg,source_sha=sha,versions=versions,device=device,
        source_run=ctx['source'].run_id,activation_run=ctx['original'].run_id),Path(local)/'capacity_rescue',require_drive)
    ctx['sweep']=cfg;ctx['device']=device
    ctx['grid']=[dict(rank=r,active=k,key=f'b{r}_k{k}') for r in cfg['block_sizes'] for k in cfg['active_groups'] if (r,k)!=(3,16)]
    ctx['store'].write('protocol_settings.json',dict(config=cfg,grid=ctx['grid'],
        control='Frozen successful GGF with 1024 groups x 3 directions, Top-16; reevaluated, not retrained.',
        variants='Every other setting receives fresh native 40-epoch pretraining and 2500 GGF updates. No gate change midway through the old model.',
        capacity='Group count stays 1024. Width changes total atoms/parameters; activity changes maximum coefficient count. These are resource-performance trade-offs, not isolated geometry effects.',
        selection='Same capture + .2 AP under 2% MSE of each variant native start; qualified gain additionally checks AP and VE against the frozen successful GGF.',
        no_test='Only the same 2000 training and 400 historical development images are loaded. Existing 1500-image scores are not mixed into this table.',
        claim='A one-start screening result is not independent confirmation, global optimality or an AGF structural-adaptation result.'))
    return ctx

def scalar(report):
    return dict(capture=report['macro_capture'],ap=report['macro_ap'],ve=report['reconstruction']['ve'],
                mse=report['reconstruction']['mse'],active=report['reconstruction']['active_groups'],
                dead=report['reconstruction']['dead_groups'])

def control(ctx,seed):
    s=ctx['store'];key=f'frozen_ggf_s{seed}';old=s.load(key)
    if old is not None:return old
    frozen=memory.required(ctx['source'],'all_models_frozen')
    row=next(r for r in frozen['models'] if r['seed']==seed and r['arm']=='ggf')
    state=memory.required(ctx['source'],row['key']);w=state[row['field']]
    if training.weight_sha(w)!=row['weights_sha']:raise ValueError('Successful GGF checkpoint hash mismatch')
    m=core.new_model('vanilla',ctx['base'],w,ctx['device']);report=study.core.validation(m,ctx['packs']['select'],ctx['targets'][seed],ctx['cfg'],ctx['device'])
    out=dict(seed=seed,setting='b3_k16_frozen_success',rank=3,active_budget=16,groups=1024,
        dictionary_atoms=3072,max_selected_coordinates=48,parameters=sum(p.numel() for p in m.parameters()),
        source_key=row['key'],source_field=row['field'],weights_sha=row['weights_sha'],selected_update=row['best_update'],
        metrics=scalar(report),report=report,role='Previously successful saved GGF; no additional optimization; same development evaluation')
    s.save(key,out);s.write('controls/'+key+'.json',out);s.event('frozen_control_evaluated',seed=seed,metrics=out['metrics'])
    del m,state;gc.collect();return out

def branch(ctx,spec):
    out=dict(ctx)
    out['base']=dict(ctx['base'],groups=1024,block_rank=spec['rank'],active_groups=spec['active'],epochs=40)
    out['cfg']=dict(ctx['cfg'],block_rank=spec['rank'],active_groups=spec['active'])
    return out

def preflight(ctx):
    s=ctx['store'];old=s.load('preflight')
    if old is not None:return old
    specs=ctx['grid'];measurements=[]
    p=ctx['packs']['train'];x=torch.as_tensor(np.array(p['x'][:8],copy=True),device=ctx['device']); masks=p['masks'][:8]
    targets=ctx['targets'][ctx['sweep']['seeds'][0]]
    for spec in specs:
        local=branch(ctx,spec);m=training.make_model('vanilla',local['base']).to(ctx['device'])
        head=torch.nn.Parameter(torch.zeros(75,2,device=ctx['device']));opt=torch.optim.Adam(list(m.parameters())+[head],lr=.0003)
        timings=[]
        for guided in (False,True):
            started=time.monotonic()
            for _ in range(3):
                xx=x.reshape(-1,768);loss,_=m.loss(xx+.1*xx.std()*torch.randn_like(xx),target=xx)
                if guided:
                    cap,det,_=core.supervised_loss(m,x,masks,targets,head,local['cfg']);loss=loss+.05*cap+.01*det
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(m.parameters())+[head],5.);opt.step();m.normalize_decoder()
            study.sync(ctx['device']);timings.append((time.monotonic()-started)/3)
        # Native actual minibatches contain 2048 patches, not this 1568-patch timing batch.
        estimate=timings[0]*(2048/1568)*7640+timings[1]*2500
        measurements.append(dict(**spec,native_seconds_per_update_estimate=timings[0]*(2048/1568),
            guided_seconds_per_update=timings[1],gradient_hours=estimate/3600,
            native_parameters=sum(p.numel() for p in m.parameters()),dictionary_atoms=1024*spec['rank'],
            max_selected_coordinates=spec['rank']*spec['active']))
        del m,head,opt;gc.collect()
    total=sum(r['gradient_hours'] for r in measurements)*len(ctx['sweep']['seeds'])
    out=dict(numerical_ok=all(np.isfinite(r['gradient_hours']) for r in measurements),measurements=measurements,
        variants=len(specs),seeds=ctx['sweep']['seeds'],rough_gradient_hours=[total,total*2],
        excludes='Training mappings, validation, data reads, initialization, checkpoints, inference and Drive latency; scenario range, not an ETA.',
        original_control_retraining=False)
    s.save('preflight',out);s.write('preflight.json',out);return out

def run(ctx):
    s=ctx['store'];old=s.load('sweep_complete')
    if old is not None:return old
    controls={seed:control(ctx,seed) for seed in ctx['sweep']['seeds']};rows=[]
    for spec in ctx['grid']:
        local=branch(ctx,spec)
        for seed in ctx['sweep']['seeds']:
            key=f'{spec["key"]}_s{seed}';done=s.load('completed_'+key)
            if done is not None:rows.append(done);continue
            s.event('variant_start',key=key,rank=spec['rank'],active=spec['active'])
            x=ctx['packs']['train']['x'].reshape(-1,768);val=ctx['packs']['select']['x'].reshape(-1,768)
            native=training.fit(s,'native_'+key,'vanilla',x,val,local['base'],seed,local['base']['lrs'][0],ctx['device'])
            initial=native['best']
            if initial is None:raise RuntimeError('No eligible native start')
            targets=s.load('targets_'+key)
            if targets is None:
                m=core.new_model('vanilla',local['base'],initial,ctx['device']);pack=ctx['packs']['train']
                audit=metrics.audit_arrays(m,pack['x'],pack['masks'],pack['ids'],local['cfg'],ctx['device'])
                targets=core.choose_targets(audit,local['cfg'],ctx['provenance']['categories'])
                if sorted(t['category_index'] for t in targets)!=local['cfg']['category_indices']:raise ValueError('Do not silently drop unsupported categories')
                s.save('targets_'+key,targets);s.write('targets/'+key+'.json',targets);del m,audit;gc.collect()
            native_report=s.load('native_report_'+key)
            if native_report is None:
                m=core.new_model('vanilla',local['base'],initial,ctx['device'])
                native_report=core.validation(m,ctx['packs']['select'],targets,local['cfg'],ctx['device'])
                s.save('native_report_'+key,native_report);del m;gc.collect()
            guided=study.fit(local,'ggf_'+key,initial,targets,True,71000+seed)
            m=core.new_model('vanilla',local['base'],guided['best'],ctx['device'])
            selected=core.validation(m,ctx['packs']['select'],targets,local['cfg'],ctx['device'])
            m.load_state_dict(guided['weights']);final=core.validation(m,ctx['packs']['select'],targets,local['cfg'],ctx['device'])
            a=scalar(selected);b=controls[seed]['metrics'];gains={k:a[k]-b[k] for k in ('capture','ap','ve')}
            limits=ctx['sweep'];reasons=[]
            if gains['capture']<limits['min_capture_gain']:reasons.append('capture_gain_below_0_5_percentage_points')
            if gains['ap']<-limits['max_ap_drop']:reasons.append('AP_drop_exceeds_0_1_percentage_points')
            if gains['ve']<-limits['max_ve_drop']:reasons.append('VE_drop_exceeds_0_1_percentage_points')
            done=dict(seed=seed,setting=spec['key'],rank=spec['rank'],active_budget=spec['active'],groups=1024,
                dictionary_atoms=1024*spec['rank'],max_selected_coordinates=spec['rank']*spec['active'],
                parameters=sum(p.numel() for p in m.parameters()),native_updates=native['updates'],guided_updates=guided['updates'],
                native_selected_update=native['best_update'],selected_update=guided['best_update'],
                selected_weights_sha=training.weight_sha(guided['best']),training_batch_sha=guided['sampled_batches_sha'],
                native_seconds=native['seconds'],guided_seconds=guided['seconds'],metrics=a,final_metrics=scalar(final),
                native_metrics=scalar(native_report),native_report=native_report,
                selected_report=selected,final_report=final,gains_vs_frozen_success=gains,
                qualified=not reasons,rejection_reasons=reasons,
                guard_rejections=sum(not r.get('object_checkpoint_eligible',True) for r in guided['history']),
                same_training_budget=True,same_compute=False,scope=limits['scope'])
            s.save('completed_'+key,done);s.write('comparisons/'+key+'.json',done);rows.append(done)
            s.event('variant_complete',key=key,completed=len(rows),total=len(ctx['grid'])*len(controls),metrics=a,qualified=done['qualified'],reasons=reasons)
            del m,native,guided,initial;gc.collect()
            if ctx['device']=='cuda':torch.cuda.empty_cache()
    out=dict(complete=True,controls=list(controls.values()),models=rows,no_test_access=True,
        qualified_settings=[dict(seed=r['seed'],setting=r['setting']) for r in rows if r['qualified']],
        meaning='Screening margins are engineering criteria, not statistical equivalence or proof of optimality.')
    s.save('sweep_complete',out);s.write('summary.json',out);return out

def export(ctx):
    s=ctx['store'];p=s.output/f'ggf_capacity_sweep_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for q in s.results.rglob('*'):
            if q.is_file():z.write(q,'results/'+q.relative_to(s.results).as_posix())
    return p
