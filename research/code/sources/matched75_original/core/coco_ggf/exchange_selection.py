"""Fixed training order, cumulative per-exchange development checks; no training."""
import copy
import csv
import json
import time
from pathlib import Path
import numpy as np
from coco_adaptation import inputs
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from . import repeated_loop as loop, accumulating_regroup as accum
from . import dependency_regroup as dep, direction_regroup as prior

PARENT_RUN='27285ff9e435b8d3a3adafbed35ff6051cfd0d4ac3f1c4abd7a4017a49c7bf7c'
PARENT_SHA='482c51d87d022a91d5a80e2364412783d29de2fccb5cda0e58eac1d84fe563ef'
STAGES={22:2,23:1}

class ReadPrefix:
    def __init__(self,parent,prefix):
        self.parent=parent;self.prefix=prefix
        self.results=parent.results/prefix
    def load(self,key):return self.parent.load(self.prefix+'_'+key)

def settings():
    return dict(version=1,study='per_exchange_development_selection',seeds=[22,23],
        max_attempts=8,max_accepted=5,gradient_updates=0,new_graphs=0,
        order='Saved training-only standalone mean gain, target gain, then proposal ID; never reordered with development or test scores.',
        scope='Diagnostic on historical development and reused inspected test images. Per-edit selection is a new rule, not independent confirmation.')

def prepare(parent_output,audit_output,loop_output,output,local,sha,versions,device='cuda',require_drive=True):
    output=Path(output);local=Path(local)
    a=output.resolve();b=Path(loop_output).resolve()
    if a==b or a in b.parents or b in a.parents:raise ValueError('Output overlaps read-only loop parent')
    ctx=loop.prepare(parent_output,audit_output,output/'verified_inputs',local,sha,versions,device,require_drive)
    source=loop.fidelity.open_source(loop_output,PARENT_RUN,local/'read_loop')
    if source.config['source_sha']!=PARENT_SHA or not memory.required(source,'complete')['complete']:
        raise ValueError('Wrong or incomplete loop source; no retraining fallback')
    if identity(source.config['config'])!=identity(loop.settings()):raise ValueError('Loop settings changed')
    if source.config['probe_sha']!=ctx['probe']['weights_sha']:raise ValueError('Frozen probe changed')
    cfg=settings()
    store=inputs.JournalStore(output,dict(config=cfg,source_sha=sha,versions=versions,device=device,
        loop_parent=PARENT_RUN,probe_sha=ctx['probe']['weights_sha']),local/'rescue',require_drive)
    ctx['store']=store;ctx['loop_source']=source;ctx['selection_cfg']=cfg
    ctx['stamps']+=memory.source_snapshot([source])
    seeds={}
    for seed,original in ctx['seeds'].items():
        parent=ReadPrefix(source,'seed'+str(seed));cycle=STAGES[seed]
        graph=ReadPrefix(parent,'cycle'+str(cycle)+'_graph')
        initial=memory.required(parent,'cycle'+str(cycle)+'_training_selected')
        if accum.weight_sha(initial)!=accum.weight_sha(memory.required(parent,'final_agf')):
            raise ValueError('Failed-pass source is not the retained final model')
        screen=memory.required(graph,'fresh_screen')
        if screen['weights_sha']!=accum.weight_sha(initial):raise ValueError('Graph source weights changed')
        graph_ids=memory.required(graph,'dependency_graph')['image_ids'] if screen['rows'] else []
        protocol=dict(ctx['protocol'],training_seed=ctx['study_cfg']['training_seed']+1000*seed+100*cycle)
        signature=identity(dict(weights=accum.weight_sha(initial),spec=initial['spec'],cycle=cycle,
            train_ids=ctx['packs']['train']['ids'],select_ids=ctx['packs']['select']['ids'],protocol=protocol))
        if memory.required(parent,'cycle'+str(cycle)+'_graph_input_identity')!=signature:
            raise ValueError('Graph model/specification/splits/protocol changed')
        s=loop.StageStore(store,'seed'+str(seed),dict(parent=PARENT_RUN,cycle=cycle,start=accum.weight_sha(initial)))
        view=dict(original,store=s,initial=initial,source_weights_sha=accum.weight_sha(initial),
            dependency=graph,saved_screen=screen,parent=parent,parent_graph=graph,graph_ids=graph_ids)
        for name in ('root_agf','root_ggf'):s.save(name,memory.required(parent,name));view[name]=memory.required(parent,name)
        s.save('start',initial);s.save('old_batch',memory.required(graph,'accumulated_snapshot'))
        s.save('final_fixed_layout',memory.required(parent,'final_fixed_layout'))
        s.save('final_continued_ggf',memory.required(parent,'final_continued_ggf'))
        seeds[seed]=view
    ctx['seeds']=seeds
    store.write('protocol_settings.json',dict(settings=cfg,guards=ctx['study_cfg'],
        dimensions='1024 groups, width 3, Top-16; no scalar bank, GMM, EM, new losses or readout reassignment.',
        source='Reuse failed pass cycle2 of start22 and cycle1 of start23; exact retained start and training-ranked shortlist.',
        selection='Train-screen positive target/mean capture and preservation; then cumulative development mean gain and all unchanged guards. Rejection reserves no groups.',
        multiplicity='Up to 8 attempts per start; rejected edits are not retried. Development is a selection set; holdout intervals are conditional and exploratory.',
        controls='Exact starting checkpoint is primary zero-update control. Old whole batch is diagnostic only. Prior eligible continued controls remain historical controls; no equal-wall-time claim.',
        inference='Activations alone; COCO masks supervise training statistics and development acceptance.'))
    return ctx

def select(ctx):
    s=ctx['store'];cached=s.load('selection_frozen')
    if cached is not None:return cached
    initial=ctx['initial'];current=initial;cfg=ctx['study_cfg'];protocol=cfg['recipe']
    screen=dep.training_screen(ctx)
    if ctx['graph_ids'] and ctx['graph_ids']!=screen['ids']:raise ValueError('Training-screen image identities changed')
    train_view=dict(ctx,packs=dict(ctx['packs'],train_screen=screen))
    train_root=accum.measured(train_view,'train_root',initial,'train_screen');train_current=train_root
    root_agf=loop.measured(ctx,'development_root_agf',ctx['root_agf'])
    root_ggf=loop.measured(ctx,'development_root_ggf',ctx['root_ggf'])
    start=loop.measured(ctx,'development_start',initial);dev_current=start
    order=accum.proposal_order(ctx['saved_screen'],protocol)
    s.write('proposal_order.json',order)
    steps=[];accepted=[];groups=set();targets=set();started=time.perf_counter()
    for i,p in enumerate(order):
        key=f'exchange{i:02d}';before=accum.weight_sha(current);saved=s.load(key)
        if saved is not None:
            if saved['before_sha']!=before or saved['proposal']!=p:raise ValueError('Resume chain changed')
            row=saved
        else:
            why=accum.conflict(p,groups,targets)
            if len(accepted)>=protocol['max_accepted']:why='maximum_accepted_reached'
            row=dict(proposal=p,before_sha=before,accepted=False,skip=why,snapshot=None)
            if why is None:
                candidate=prior.permute_snapshot(current,p['outgoing'],p['incoming'])
                s.save(key+'_candidate',candidate)
                train=accum.measured(train_view,key+'_train',candidate,'train_screen')
                train_gate=accum.positive_gate(train,train_current,train_root,protocol,p['category_index'])
                row.update(candidate_sha=accum.weight_sha(candidate),train_gate=train_gate,train_report=train,
                    train_deltas=accum.category_deltas(train,train_current))
                if train_gate['passed']:
                    dev=loop.measured(ctx,key+'_development',candidate)
                    gate=loop.stage_gate(dev,dev_current,root_agf,root_ggf,cfg)
                    start_guard=loop.stage_gate(dev,start,root_agf,root_ggf,cfg,False)
                    gate['margins'].update({'pass_start_'+k:v for k,v in start_guard['margins'].items()})
                    gate['failures']+=['pass_start_'+k for k in start_guard['failures']]
                    gate['passed']=not gate['failures']
                    row.update(development_gate=gate,development=dev,
                        development_deltas=accum.category_deltas(dev['report'],dev_current['report']),
                        root_deltas=accum.category_deltas(dev['report'],start['report']),
                        image_ap_deltas=[dict(category_index=c,change=a-b) for c,a,b in zip(
                            dev['task']['category_indices'],dev['task']['per_category_ap'],dev_current['task']['per_category_ap'])])
                    model=prior.model(ctx,candidate)
                    graph=memory.required(ctx['dependency'],'dependency_graph')
                    row['gate_effects']=dep.gate_effects(ctx,model,screen,graph,p)
                    del model
                    if gate['passed']:
                        row['accepted']=True;row['snapshot']=key+'_accepted'
                        s.save(row['snapshot'],candidate)
                else:row['skip']='training_gate_failed'
            row['after_sha']=accum.weight_sha(candidate) if row['accepted'] else before
            # Accepted snapshot and reports are committed BEFORE the decision.
            s.save(key,row)
        if row['accepted']:
            current=memory.required(s,row['snapshot'])
            if accum.weight_sha(current)!=row['after_sha']:raise ValueError('Accepted checkpoint changed')
            train_current=row['train_report'];dev_current=row['development'];accepted.append(p)
            groups.update((p['receiver_group'],p['donor_group']));targets.add(p['category_index'])
        elif accum.weight_sha(current)!=before:raise AssertionError('Rejected candidate modified current')
        steps.append(row);s.write('steps/'+key+'.json',loop.compact(row))
        s.event('exchange_decision',seed=ctx['seed'],done=i+1,total=len(order),proposal=p['id'],
            category=p['category'],accepted=row['accepted'],accepted_total=len(accepted),
            train_failures=row.get('train_gate',{}).get('failures',[]),
            development_failures=row.get('development_gate',{}).get('failures',[]),skip=row.get('skip'))
    final_gate=loop.stage_gate(dev_current,start,root_agf,root_ggf,cfg) if accepted else None
    if accepted and not final_gate['passed']:raise AssertionError('Accepted sequence failed cumulative final guard')
    s.save('final_agf',current)
    old=loop.measured(ctx,'development_old_batch',memory.required(s,'old_batch'))
    out=dict(complete=True,accepted_proposals=accepted,steps=steps,final_sha=accum.weight_sha(current),
        final_identity=identity(dict(weights=accum.weight_sha(current),spec=current['spec'])),
        final_gate=final_gate,development_final=loop.compact(dev_current),
        old_batch_gate=loop.stage_gate(old,start,root_agf,root_ggf,cfg),
        seconds=time.perf_counter()-started,gradient_updates=0,test_used=False)
    s.save('selection_frozen',out);s.write('selection_frozen.json',loop.compact(out));return out

def run(ctx):
    cached=ctx['store'].load('all_models_frozen')
    if cached is not None:memory.assert_immutable(ctx['stamps']);return cached
    seeds={str(seed):loop.compact(select(view)) for seed,view in ctx['seeds'].items()}
    out=dict(complete=True,seeds=seeds,test_used=False)
    ctx['store'].save('all_models_frozen',out);ctx['store'].write('all_models_frozen.json',out)
    memory.assert_immutable(ctx['stamps']);return out

def audit(ctx):
    s=ctx['store'];cached=s.load('complete')
    if cached is not None:memory.assert_immutable(ctx['stamps']);return cached
    frozen=memory.required(s,'all_models_frozen')
    if not frozen['complete'] or frozen['test_used']:raise ValueError('Freeze both starts before audit')
    provenance=copy.deepcopy(ctx['provenance']);provenance['splits']['test']=ctx['source']['ids']['test']
    test=memory.normalized_pack(loop.ProgressSource(ctx['source']['parent'],'test',len(provenance['splits']['test'])),
        'test',provenance,ctx['norm'],ctx['local'],20)
    expected=json.loads((ctx['source']['parent'].results/'fresh_activation_identity.json').read_text())['normalized_sha']
    if memory.streaming_array_sha(test['x'])!=expected:raise ValueError('Holdout activation bytes changed')
    rows=[];pairs=[]
    for seed,view in ctx['seeds'].items():
        view['packs']['test']=test;reports={}
        if accum.weight_sha(memory.required(view['store'],'final_agf'))!=frozen['seeds'][str(seed)]['final_sha']:
            raise ValueError('Frozen selected model changed')
        nominated=memory.required(view['store'],'final_agf')
        if identity(dict(weights=accum.weight_sha(nominated),spec=nominated['spec']))!=frozen['seeds'][str(seed)]['final_identity']:
            raise ValueError('Frozen specification/readout changed')
        for name in ('start','old_batch','final_agf','final_fixed_layout','final_continued_ggf'):
            snap=memory.required(view['store'],name);r=loop.measured(view,'audit_'+name,snap,'test');reports[name]=r
            rows.append(dict(seed=seed,model=name,**loop.bank.scalar(r['report']),image_ap=r['task']['macro_ap'],weights_sha=accum.weight_sha(snap)))
        np.savez_compressed(view['store'].results/'image_readout_scores.npz',image_ids=np.array(test['ids']),
            category_indices=np.array(ctx['cohort']),labels=reports['final_agf']['task']['labels'],
            **{name:r['task']['scores'] for name,r in reports.items()})
        selected=memory.required(view['store'],'final_agf');start=memory.required(view['store'],'start')
        ci=accum.paired_audit(view,selected,start,'test','selected_vs_start')
        task=loop.fidelity.paired_ap(reports['final_agf']['task']['labels'],reports['final_agf']['task']['scores'],
            reports['start']['task']['scores'],ctx['study_cfg']['bootstrap'],106082)
        gate=loop.stage_gate(reports['final_agf'],reports['start'],
            loop.measured(view,'audit_root_agf',view['root_agf'],'test'),
            loop.measured(view,'audit_root_ggf',view['root_ggf'],'test'),ctx['study_cfg'])
        pair=dict(seed=seed,group_paired=ci,image_paired=task,gate=gate,
            category_changes=accum.category_deltas(reports['final_agf']['report'],reports['start']['report']))
        pairs.append(loop.compact(pair));view['store'].write('audit_comparison.json',loop.compact(pair))
    with (s.results/'comparison.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=dict(complete=True,models=rows,paired=pairs,accepted={k:len(v['accepted_proposals']) for k,v in frozen['seeds'].items()},
        gradient_updates=0,new_graphs=0,test_images=len(test['ids']),scope=ctx['selection_cfg']['scope'],
        interpretation='Positive guarded gains versus exact start support per-exchange selection. Zero accepted exchanges indicate the unchanged guards or shortlist remain limiting. Old batch is a diagnostic, never an eligible control. No test-based retries.')
    memory.assert_immutable(ctx['stamps']);s.save('complete',out);s.write('summary.json',out);return out

def export(ctx):
    path=loop.export(ctx)
    target=path.with_name('agf_exchange_selection_'+ctx['store'].run_id[:12]+'_results.zip')
    path.replace(target);return target
