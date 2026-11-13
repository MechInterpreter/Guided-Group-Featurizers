"""Bounded supervised learn -> graph -> exchange cycles from saved AGF starts.

No baseline training or extraction. Reused holdout is audited once after every
model freezes. Additional search costs are separate from matched update counts.
"""
import copy
import csv
import gc
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import inputs
from coco_baselines.storage import identity, serial, sha_file
from coco_coordinated import run as memory
from . import downstream_fidelity as fidelity, mature_regroup as mature
from . import accumulating_regroup as accum, direction_regroup as prior
from . import bank_experiment as bank, owner_routing as old, study

AUDIT_RUN='c06f8e727b721be48ef71b4043004d50727af15f58c3e49f6b962376d44a6ff4'
AUDIT_SHA='0adbcd465099672cf8970ead318c64664930c2d05d3e8cf95858c3470a78a52f'

def settings():
    recipe=mature.settings()
    recipe.update(recovery_updates=100,evaluate_every=100,local_save_every=25,
        gradient_every=100,restricted_updates=0)
    return dict(version=1,study='bounded_repeated_supervised_agf_cycles',seeds=[22,23],
        cycles=2,updates_per_cycle=100,bootstrap=200,training_seed=106071,
        task_ap_margin=.001,task_category_margin=.01,recipe=recipe,
        scope='Two further cycles after the already accepted one-pass AGF. Same training data and historical development; final audit reuses the inspected 1500-image holdout. No fresh replication or equal-wall-time claim.')

class StageStore:
    """A full input signature separates each model/round's resumable statistics."""
    def __init__(self,parent,prefix,signature):
        self.parent=parent;self.prefix=prefix;self.config=parent.config
        self.results=parent.results/prefix;self.results.mkdir(parents=True,exist_ok=True)
        old_sig=self.load('input_identity')
        if old_sig is not None and old_sig!=signature:raise ValueError('Stage inputs changed: '+prefix)
        if old_sig is None:self.save('input_identity',signature)
    def load(self,key):return self.parent.load(self.prefix+'_'+key)
    def save(self,key,value,*args,**kwargs):return self.parent.save(self.prefix+'_'+key,value,*args,**kwargs)
    def write(self,path,value):return self.parent.write(self.prefix+'/'+path,value)
    def event(self,name,**kw):
        child=kw.pop('stage_prefix',None)
        kw['stage_prefix']=self.prefix+'/'+str(child) if child else self.prefix
        return self.parent.event(name,**kw)

def stage(ctx,name,initial,cycle):
    cfg=dict(ctx['protocol'],training_seed=ctx['study_cfg']['training_seed']+1000*ctx['seed']+100*cycle)
    sig=identity(dict(weights=accum.weight_sha(initial),spec=initial['spec'],cycle=cycle,
        train_ids=ctx['packs']['train']['ids'],select_ids=ctx['packs']['select']['ids'],protocol=cfg))
    return dict(ctx,initial=initial,protocol=cfg,prior=None,
        source_weights_sha=accum.weight_sha(initial),store=StageStore(ctx['store'],name,sig))

class ProgressSource:
    """Read-only proxy: show shard verification progress without changing inputs."""
    def __init__(self,source,split,total):self.source=source;self.split=split;self.total=total
    def __getattr__(self,key):return getattr(self.source,key)
    def load(self,key):
        result=self.source.load(key)
        if key.startswith('features_'+self.split+'_') and result is not None:
            done=int(key.rsplit('_',1)[-1])+len(result['ids'])
            print('Verified activation shards:',self.split,done,'/',self.total,flush=True)
        return result

def prepare(parent_output,audit_output,output,local,sha,versions,device='cuda',require_drive=True):
    output=Path(output);local=Path(local);cfg=settings()
    for root in (Path(parent_output),Path(audit_output),Path(parent_output).parent/'GGF_COCO_scaled_confirmation_v1',Path(parent_output).parent/'AGF_COCO_stratified_confirmation_v1'):
        a=output.resolve();b=root.resolve()
        if a==b or a in b.parents or b in a.parents:raise ValueError('Output overlaps a read-only source: '+str(root))
    # Reuse verified provenance/model discovery without loading a backbone.
    src=fidelity.prepare(parent_output,output/'verified_inputs',local/'verified_inputs',sha,versions,device,require_drive)
    audit=fidelity.open_source(audit_output,AUDIT_RUN,local/'read_audit')
    if audit.config['source_sha']!=AUDIT_SHA or not memory.required(audit,'complete')['complete']:
        raise ValueError('Wrong or incomplete downstream-preservation parent')
    dest=output.resolve();source=Path(audit_output).resolve()
    if dest==source or dest in source.parents or source in dest.parents:raise ValueError('Output overlaps read-only audit')
    frozen_probe=memory.required(audit,'frozen_probe');clean_select=memory.required(audit,'clean_select')
    if frozen_probe['test_used'] or clean_select['ids']!=src['ids']['select']:
        raise ValueError('Frozen probe or development image identities changed')
    actual_probe_sha=identity({k:memory.streaming_array_sha(v) if isinstance(v,np.ndarray) else v for k,v in frozen_probe['probe'].items()})
    if actual_probe_sha!=frozen_probe['weights_sha']:raise ValueError('Frozen probe tensors changed')
    trained_source=fidelity.open_source(Path(parent_output).parent/'GGF_COCO_scaled_confirmation_v1',old.EXPECTED,local/'read_scaled')
    if trained_source.config['source_sha']!=old.EXPECTED_SOURCE:raise ValueError('Unexpected successful GGF source')
    eval_cfg=dict(trained_source.config['config'],device=device,audit_image_batch=2,preview_images=0,batch=1024)
    if eval_cfg['category_indices']!=src['cohort']:raise ValueError('Evaluation cohort changed')
    memory.memory_safe_metrics()
    packs={}
    for split in ('train','select'):
        packs[split]=memory.normalized_pack(ProgressSource(src['original'],split,len(src['ids'][split])),split,src['provenance'],src['norm'],local,20)
    store=inputs.JournalStore(output,dict(config=cfg,source_sha=sha,versions=versions,device=device,
        independent_parent=fidelity.PARENT_RUN,audit_parent=AUDIT_RUN,probe_sha=frozen_probe['weights_sha'],
        source_models=[{k:v for k,v in r.items() if k!='store'} for r in src['models'] if r['seed'] in cfg['seeds']]),local/'rescue',require_drive)
    ctx=dict(source=src,audit=audit,trained_source=trained_source,store=store,packs=packs,
        cfg=eval_cfg,base=trained_source.config['native_config'],provenance=src['provenance'],
        original=src['original'],norm=src['norm'],local=local,device=device,study_cfg=cfg,
        protocol=cfg['recipe'],probe=frozen_probe,clean_select=clean_select,
        cohort=src['cohort'],stamps=src['stamps']+memory.source_snapshot([audit,trained_source]),seeds={})
    starts={}
    for seed in cfg['seeds']:
        selected=[r for r in src['models'] if r['seed']==seed];snapshots={}
        for row in selected:
            snap=memory.required(row['store'],row['key'])
            if accum.weight_sha(snap)!=row['sha']:raise ValueError('Starting snapshot hash changed')
            snapshots[row['arm']]=snap
        if [t['category_index'] for t in snapshots['AGF']['spec']['targets']]!=src['cohort']:
            raise ValueError('Fixed group readout/category ordering changed')
        if identity(snapshots['GGF']['spec']['targets'])!=identity(snapshots['AGF']['spec']['targets']):
            raise ValueError('GGF/AGF object readouts differ')
        seed_store=StageStore(store,'seed'+str(seed),identity({arm:accum.weight_sha(snap) for arm,snap in snapshots.items()}))
        seed_ctx=dict(ctx,seed=seed,store=seed_store,initial=snapshots['AGF'],root_agf=snapshots['AGF'],root_ggf=snapshots['GGF'])
        seed_store.save('root_agf',snapshots['AGF']);seed_store.save('root_ggf',snapshots['GGF'])
        ctx['seeds'][seed]=seed_ctx
        starts[seed]={arm:accum.weight_sha(snap) for arm,snap in snapshots.items()}
    store.write('protocol_settings.json',dict(settings=cfg,starts=starts,probe_sha=frozen_probe['weights_sha'],
        supervision='COCO masks guide GGF losses, dependencies and acceptance. Inference uses activations alone.',
        comparisons='Same-layout no-further-edit continuation isolates repeated adaptation. Original mature GGF continuation checks the overall method. All receive the same attempted 100 updates per executed cycle, batches and noise, fresh Adam per cycle.',
        capacity='1024 groups, width 3, native Top-16. No scalar bank, Gaussian model, EM, new routing, width changes or new readout assignments.',
        acceptance='Positive mean capture on development; root-relative reconstruction/group category guards plus frozen image-readout preservation. No test score chooses a candidate.',
        task_guard='Mean image AP may fall at most 0.1 pp versus current, one-pass AGF and mature GGF; each image-category AP may fall at most 1 pp versus current and one-pass AGF. These are explicit engineering tolerances, not calibrated equivalence claims.',
        stop='At most two further cycles; stop after a whole cycle accepts neither training nor regrouping. This bounded stop does not prove global convergence.',
        costs='Matched attempted gradient updates, not equal wall-time; extra AGF graph/scoring/selection costs are recorded. No equal-compute superiority claim.',
        test='Freeze all seeds/control choices before reading the inspected 1500-image holdout. No retries, no fresh-test claim.'))
    store.write('split_roles.json',dict(ids=src['ids'],train='Losses/profiles/graph and proposal shortlisting',
        select='Historical development: checked stages and strongest safe control checkpoint',
        test='Reused inspected 1500-image audit only after global freeze'))
    return ctx

@torch.inference_mode()
def task_readout(ctx,snap,pack):
    m=prior.model(ctx,snap);m.eval();pooled=[];t0=time.perf_counter()
    signature=identity(dict(weights=accum.weight_sha(snap),spec=snap['spec'],ids=pack['ids']))
    part_key='readout_'+signature[:24]
    resumed=ctx['store'].load(part_key)
    if resumed is not None:
        if resumed['signature']!=signature or resumed['done']!=len(resumed['pooled']):raise ValueError('Task pooling resume changed')
        pooled=list(resumed['pooled'])
    start=len(pooled)
    for i in range(start,len(pack['ids'])):
        x=pack['x'][i]
        xx=torch.as_tensor(np.array(x,copy=True),device=ctx['device']);rec,_=m(xx)
        if not torch.isfinite(rec).all():raise FloatingPointError('Nonfinite task reconstruction')
        pooled.append(rec.double().mean(0).cpu().numpy())
        if (i+1)%20==0 or i+1==len(pack['ids']):
            ctx['store'].save(part_key,dict(signature=signature,done=i+1,pooled=np.array(pooled)))
        if (i+1)%100==0:ctx['store'].event('image_readout_progress',done=i+1,total=len(pack['ids']))
    scores=fidelity.predict(ctx['probe']['probe'],np.array(pooled))
    y=fidelity.presence(pack['masks'],ctx['cohort']);ap=fidelity.category_ap(y,scores)
    del m;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(macro_ap=float(ap.mean()),per_category_ap=ap.tolist(),scores=scores,labels=y,
        ids=pack['ids'],category_indices=ctx['cohort'],seconds=time.perf_counter()-t0,weights_sha=accum.weight_sha(snap))

def measured(ctx,key,snap,split='select',with_task=True):
    store=ctx['store'];sig=identity(dict(weights=accum.weight_sha(snap),spec=snap['spec'],
        ids=ctx['packs'][split]['ids'],split=split,probe_sha=ctx['probe']['weights_sha'],with_task=with_task))
    cached=store.load(key)
    if cached is not None:
        if cached['signature']!=sig:raise ValueError('Cached evaluation source changed: '+key)
        return cached
    t0=time.perf_counter();store.event('measurement_start',key=key,split=split)
    report=accum.measured(ctx,key+'_group',snap,split)
    task=task_readout(ctx,snap,ctx['packs'][split]) if with_task else None
    result=dict(signature=sig,report=report,task=task,weights_sha=accum.weight_sha(snap),seconds=time.perf_counter()-t0)
    store.save(key,result);store.write('measurements/'+key+'.json',compact(result))
    store.event('measurement_complete',key=key,seconds=result['seconds'],**bank.scalar(report),
        image_ap=task['macro_ap'] if task else None)
    return result

def compact(value):
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items() if k not in ('scores','labels','snapshot')}
    if isinstance(value,list):return [compact(v) for v in value]
    return serial(value)

def preflight(ctx):
    cached=ctx['store'].load('preflight')
    if cached is not None:return cached
    reports={str(seed):prior.preflight(view) for seed,view in ctx['seeds'].items()}
    first=next(iter(ctx['seeds'].values()));timed=prior.timing(first)
    cfg=ctx['study_cfg'];graphs=len(cfg['seeds'])*cfg['cycles']
    result=dict(numerical_ok=all(r['numerical_ok'] for r in reports.values()),starts=reports,
        maximum_training_updates_all_three_arms=len(cfg['seeds'])*cfg['cycles']*cfg['updates_per_cycle']*3,
        maximum_fresh_graphs=graphs,maximum_shortlisted_exchanges=graphs*cfg['recipe']['max_attempts'],
        maximum_accepted_exchanges=graphs*cfg['recipe']['max_accepted'],
        seconds_per_profile_image=timed['seconds_per_profile_image'],
        seconds_per_transient_gradient_update=timed['transient_seconds_per_gradient_update'],
        profile_only_minutes=timed['seconds_per_profile_image']*len(first['packs']['train']['ids'])*graphs/60,
        gradient_only_minutes=timed['transient_seconds_per_gradient_update']*len(cfg['seeds'])*cfg['cycles']*cfg['updates_per_cycle']*3/60,
        excludes='Dependency graph construction, all candidate screening, group metrics, image readouts, bootstrap, checkpoints and Drive. Components only, NOT an ETA.',
        transient_benchmark_updates_discarded=3,baseline_training=False,backbone_extraction=False)
    ctx['store'].save('preflight',result);ctx['store'].write('preflight.json',result);return result

def stage_gate(candidate,current,root_agf,root_ggf,cfg,require_gain=True):
    refs=dict(current=current['report'],loop_root=root_agf['report'],mature_GGF=root_ggf['report'])
    gate=accum.quality_gate(candidate['report'],refs,cfg['recipe'])
    gain=candidate['report']['macro_capture']-current['report']['macro_capture']
    if require_gain:
        gate['margins']['positive_mean_capture']=gain-cfg['recipe']['positive_epsilon']
    a=np.array(candidate['task']['per_category_ap'])
    cats=candidate['task']['category_indices']
    if len(cats)!=len(a):raise ValueError('Task category dimensions changed')
    for name,ref in [('current',current),('loop_root',root_agf),('mature_GGF',root_ggf)]:
        if ref['task']['category_indices']!=cats:raise ValueError('Task category ordering changed')
        gate['margins'][name+'_image_AP']=candidate['task']['macro_ap']-ref['task']['macro_ap']+cfg['task_ap_margin']
        if name!='mature_GGF':
            for c,value in zip(cats,a-np.array(ref['task']['per_category_ap'])):
                gate['margins'][name+'_image_category_'+str(c)]=float(value)+cfg['task_category_margin']
    gate['failures']=[k for k,v in gate['margins'].items() if not np.isfinite(v) or v < -1e-12]
    if require_gain and (not np.isfinite(gain) or gain<=cfg['recipe']['positive_epsilon']):
        if 'positive_mean_capture' not in gate['failures']:gate['failures'].append('positive_mean_capture')
    gate['passed']=not gate['failures'];gate['capture_gain']=gain
    return gate

def pick_control(rows,root_agf,root_ggf,cfg):
    """Same criterion; each continuation preserves its own starting layout. The mature GGF root remains a legal no-training fallback."""
    checked=[]
    for row in rows:
        gate=stage_gate(row['measurement'],row['root_measurement'],row['root_measurement'],root_ggf,cfg,False)
        checked.append(dict(row,gate=gate))
    eligible=[r for r in checked if r['gate']['passed']]
    if not eligible:raise ValueError('No unchanged root remains eligible; inspect numerical replay/guards')
    chosen=max(eligible,key=lambda r:(r['measurement']['report']['macro_capture'],
        r['measurement']['report']['macro_ap'],-r['measurement']['report']['reconstruction']['mse']))
    return chosen,checked

def run_seed(ctx):
    s=ctx['store'];cached=s.load('seed_frozen')
    if cached is not None:return cached
    cfg=ctx['study_cfg'];agf=ctx['root_agf'];control_fixed=copy.deepcopy(agf);control_ggf=ctx['root_ggf']
    base_agf=measured(ctx,'development_root_agf',agf);base_ggf=measured(ctx,'development_root_ggf',control_ggf)
    current=base_agf;controls={arm:[dict(checkpoint='root_'+('agf' if arm=='fixed_layout' else 'ggf'),
        measurement=base_agf if arm=='fixed_layout' else base_ggf,root_measurement=base_agf if arm=='fixed_layout' else base_ggf)] for arm in ('fixed_layout','continued_ggf')}
    rounds=[];accepted_edits=0;accepted_training=0;started=time.perf_counter()
    for cycle in range(1,cfg['cycles']+1):
        key='cycle'+str(cycle);saved=s.load(key+'_decision')
        if saved is not None:
            if saved['before_sha']!=accum.weight_sha(agf):raise ValueError('Cycle resume chain changed')
            agf=memory.required(s,saved['final_checkpoint']);current=measured(ctx,key+'_selected',agf)
            control_fixed=memory.required(s,saved['fixed_endpoint']);control_ggf=memory.required(s,saved['ggf_endpoint'])
            for arm in controls:controls[arm].append(saved['controls'][arm])
            rounds.append(saved);accepted_edits+=saved['development_accepted_exchanges'];accepted_training+=int(saved['training_adopted'])
            if saved['stop']:break
            continue
        before=accum.weight_sha(agf);cycle_start=time.perf_counter();s.event('cycle_start',seed=ctx['seed'],cycle=cycle)
        fits={};contexts={}
        for arm,snap in [('adaptive',agf),('fixed_layout',control_fixed),('continued_ggf',control_ggf)]:
            view=stage(ctx,key+'_'+arm,snap,cycle);contexts[arm]=view
            fits[arm]=prior.fit(view,'learning',snap,updates=cfg['updates_per_cycle'])
            if fits[arm]['updates']!=cfg['updates_per_cycle']:raise ValueError('Incomplete learning phase')
        if len({fit['batch_sha'] for fit in fits.values()})!=1:raise ValueError('Paired training image/noise schedules differ')
        trained=fits['adaptive']['current'];train_report=measured(ctx,key+'_trained',trained)
        training_gate=stage_gate(train_report,current,base_agf,base_ggf,cfg)
        training_adopted=training_gate['passed']
        if training_adopted:agf=trained;current=train_report;accepted_training+=1
        s.save(key+'_training_selected',agf)
        s.write('decisions/'+key+'_training.json',dict(adopted=training_adopted,gate=training_gate,
            attempted_updates=cfg['updates_per_cycle'],candidate_sha=accum.weight_sha(trained),selected_sha=accum.weight_sha(agf)))
        graph_ctx=stage(ctx,key+'_graph',agf,cycle)
        mature.screen_and_graph(graph_ctx);batch=mature.accumulate(graph_ctx)
        candidate=memory.required(graph_ctx['store'],'accumulated_snapshot')
        edited=measured(ctx,key+'_edited',candidate)
        gate=stage_gate(edited,current,base_agf,base_ggf,cfg)
        adopted=bool(batch['accepted_proposals']) and gate['passed']
        n=len(batch['accepted_proposals']) if adopted else 0
        if adopted:agf=candidate;current=edited;accepted_edits+=n
        else:gate['reason']='No train-adopted edits' if not batch['accepted_proposals'] else 'Whole batch failed development; no best-prefix retry'
        s.save(key+'_selected_snapshot',agf)
        controls_now={}
        for arm in controls:
            endpoint=fits[arm]['current'];name=key+'_'+arm+'_endpoint';s.save(name,endpoint)
            obs=measured(ctx,key+'_'+arm,endpoint)
            row=dict(checkpoint=name,measurement=obs,root_measurement=base_agf if arm=='fixed_layout' else base_ggf)
            controls[arm].append(row);controls_now[arm]=row
        control_fixed=fits['fixed_layout']['current'];control_ggf=fits['continued_ggf']['current']
        stop=not training_adopted and not adopted
        record=dict(cycle=cycle,before_sha=before,after_sha=accum.weight_sha(agf),training_adopted=training_adopted,
            training_gate=training_gate,graph_weights_sha=graph_ctx['source_weights_sha'],
            training_accepted_exchanges=len(batch['accepted_proposals']),development_accepted_exchanges=n,
            accepted_proposals=batch['accepted_proposals'] if adopted else [],batch_gate=gate,
            final_checkpoint=key+'_selected_snapshot',fixed_endpoint=key+'_fixed_layout_endpoint',
            ggf_endpoint=key+'_continued_ggf_endpoint',controls=controls_now,
            learning_updates_per_arm=cfg['updates_per_cycle'],batch_sha=fits['adaptive']['batch_sha'],
            seconds=time.perf_counter()-cycle_start,fit_compute_seconds={k:v['seconds'] for k,v in fits.items()},
            stop=stop,stop_reason='No training or structural stage passed development' if stop else None)
        s.save(key+'_decision',record);s.write('decisions/'+key+'.json',compact(record));rounds.append(record)
        s.event('cycle_complete',seed=ctx['seed'],cycle=cycle,training_adopted=training_adopted,
            development_accepted_exchanges=n,stop=stop,**bank.scalar(current['report']))
        if stop:break
    chosen={};control_checks={}
    for arm,rows in controls.items():chosen[arm],control_checks[arm]=pick_control(rows,base_agf,base_ggf,cfg)
    s.save('final_agf',agf)
    for arm,row in chosen.items():s.save('final_'+arm,memory.required(s,row['checkpoint']))
    models=dict(final_agf=accum.weight_sha(agf),root_agf=accum.weight_sha(ctx['root_agf']),root_ggf=accum.weight_sha(ctx['root_ggf']))
    for arm in chosen:models['final_'+arm]=accum.weight_sha(memory.required(s,'final_'+arm))
    out=dict(complete=True,seed=ctx['seed'],models=models,cycles=rounds,
        model_identities={name:identity(dict(weights=sha,spec=memory.required(s,name)['spec'])) for name,sha in models.items()},
        new_development_accepted_exchanges=accepted_edits,accepted_learning_phases=accepted_training,
        attempted_updates_per_arm=cfg['updates_per_cycle']*len(rounds),total_attempted_updates=3*cfg['updates_per_cycle']*len(rounds),
        selected_controls=chosen,all_control_checks=control_checks,development_final=current,
        development_changes_vs_AGF=accum.category_deltas(current['report'],base_agf['report']),development_changes_vs_GGF=accum.category_deltas(current['report'],base_ggf['report']),
        seconds_including_control_training=sum(r['seconds'] for r in rounds),timing_scope='Sum of completed-cycle wall times; interrupted work before a resumed cycle is not included. Per-fit accumulated compute and events are retained.',
        stop_reason=rounds[-1]['stop_reason'] or 'Declared two-cycle cap reached; not evidence of a plateau',
        test_used_for_selection=False,scope=cfg['scope'])
    s.save('seed_frozen',out);s.write('frozen_models.json',compact(out));return out

def run(ctx):
    s=ctx['store'];cached=s.load('all_models_frozen')
    if cached is not None:memory.assert_immutable(ctx['stamps']);return cached
    roots={seed:run_seed(view) for seed,view in ctx['seeds'].items()}
    out=dict(complete=True,seeds={str(seed):compact(row) for seed,row in roots.items()},test_used=False)
    s.save('all_models_frozen',out);s.write('all_models_frozen.json',out)
    memory.assert_immutable(ctx['stamps']);return out

def audit(ctx):
    s=ctx['store'];cached=s.load('complete')
    if cached is not None:memory.assert_immutable(ctx['stamps']);return cached
    frozen=memory.required(s,'all_models_frozen')
    if not frozen['complete'] or frozen['test_used']:raise ValueError('All models must freeze before audit')
    # The parent holdout is already inspected. This additional repeated-loop audit
    # never creates or reallocates images, changes a nominee, or tunes a threshold.
    provenance=copy.deepcopy(ctx['provenance']);provenance['splits']['test']=ctx['source']['ids']['test']
    test=memory.normalized_pack(ProgressSource(ctx['source']['parent'],'test',len(provenance['splits']['test'])),'test',provenance,ctx['norm'],ctx['local'],20)
    if memory.streaming_array_sha(test['x'])!=json.loads((ctx['source']['parent'].results/'fresh_activation_identity.json').read_text())['normalized_sha']:
        raise ValueError('Reused test activations changed')
    rows=[];pair_rows=[];seed_results=[]
    for seed,view in ctx['seeds'].items():
        view['packs']['test']=test;models={};reports={}
        for name in ('root_ggf','root_agf','final_continued_ggf','final_fixed_layout','final_agf'):
            snap=memory.required(view['store'],name)
            if accum.weight_sha(snap)!=frozen['seeds'][str(seed)]['models'][name]:raise ValueError('Frozen nominee changed')
            if identity(dict(weights=accum.weight_sha(snap),spec=snap['spec']))!=frozen['seeds'][str(seed)]['model_identities'][name]:raise ValueError('Frozen model specification changed')
            models[name]=snap;reports[name]=measured(view,'audit_'+name,snap,'test')
            rows.append(dict(seed=seed,model=name,**bank.scalar(reports[name]['report']),
                image_ap=reports[name]['task']['macro_ap'],weight_sha=accum.weight_sha(snap)))
        pairs=[]
        for comparator in ('root_agf','final_fixed_layout','final_continued_ggf'):
            key='final_vs_'+comparator
            ci=accum.paired_audit(view,models['final_agf'],models[comparator],'test',key)
            ci['capture_gain']=reports['final_agf']['report']['macro_capture']-reports[comparator]['report']['macro_capture']
            ci['ap_gain']=reports['final_agf']['report']['macro_ap']-reports[comparator]['report']['macro_ap']
            ci['scope']='Conditional whole-image intervals on a reused inspected holdout; no fresh replication or seed-uncertainty claim.'
            task=fidelity.paired_ap(reports['final_agf']['task']['labels'],reports['final_agf']['task']['scores'],
                reports[comparator]['task']['scores'],ctx['study_cfg']['bootstrap'],106072)
            gate=stage_gate(reports['final_agf'],reports[comparator],reports['root_agf'],reports['root_ggf'],ctx['study_cfg'])
            row=dict(seed=seed,comparator=comparator,group_paired=ci,image_task_paired=task,gate=gate,
                point_checks_passed=gate['passed'],all_category_changes=accum.category_deltas(reports['final_agf']['report'],reports[comparator]['report']),capture_interval_positive=ci['capture_gain_ci95'][0]>0,
                mean_image_interval_preserved=task['gain_ci95'][0]>=-ctx['study_cfg']['task_ap_margin'])
            view['store'].write('audit_pairs/'+key+'.json',compact(row));pairs.append(row);pair_rows.append(compact(row))
        np.savez_compressed(view['store'].results/'image_readout_scores.npz',image_ids=np.array(test['ids']),
            category_indices=np.array(ctx['cohort']),labels=reports['final_agf']['task']['labels'],
            **{name:r['task']['scores'] for name,r in reports.items()})
        seed_results.append(dict(seed=seed,cycles=frozen['seeds'][str(seed)]['cycles'],
            accepted_exchanges=frozen['seeds'][str(seed)]['new_development_accepted_exchanges'],
            comparisons=compact(pairs),scope=ctx['study_cfg']['scope']))
    with (s.results/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    memory.assert_immutable(ctx['stamps'])
    result=dict(complete=True,models=rows,seeds=seed_results,paired=pair_rows,
        test_images=len(test['ids']),primary_seeds=[22,23],scope=ctx['study_cfg']['scope'],
        interpretation='Repeated adaptation adds value only if new exchanges pass development and the final model improves beyond the same-layout training control. Gains shared by fixed controls are training gains. A failed audit never changes the frozen nominee.',
        equal_gradient_budget=True,equal_wall_time=False,new_backbone_extraction=False,new_baseline_training=False)
    plots(ctx,result);s.save('complete',result);s.write('summary.json',result)
    s.event('complete',accepted_exchanges={r['seed']:r['accepted_exchanges'] for r in seed_results});return result

def plots(ctx,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=result['models'];fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    for ax,metric,label in zip(axes,('capture','ap','ve'),('Object capture','Patch detection AP','Reconstruction VE')):
        for seed,offset,color in [(22,-.18,'#008578'),(23,.18,'#15313b')]:
            subset=[r for r in rows if r['seed']==seed]
            ax.bar(np.arange(len(subset))+offset,[r[metric]*100 for r in subset],width=.34,label=str(seed),color=color)
        ax.set_xticks(np.arange(5),['GGF root','One-pass AGF','Cont. GGF','Fixed layout','Loop AGF'],rotation=35,ha='right',fontsize=8)
        ax.set_ylabel(label+' (%)');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(title='Start');fig.suptitle('Reused 1500-image audit: all frozen nominees and controls')
    folder=ctx['store'].results/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)

def export(ctx):
    s=ctx['store'];path=s.output/f'agf_repeated_loop_{s.run_id[:12]}_results.zip';hashes={}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(s.results.rglob('*')):
            if p.is_file():
                name='results/'+p.relative_to(s.results).as_posix();z.write(p,name);hashes[name]=sha_file(p)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise OSError('Evidence export corruption')
    return path
