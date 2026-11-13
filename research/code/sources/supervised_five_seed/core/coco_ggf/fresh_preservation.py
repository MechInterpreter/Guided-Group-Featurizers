"""Fixed frozen exchanges, independent development preservation, untouched audit."""
import copy
import csv
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import block_diag, csc_matrix, eye, hstack, vstack
from coco_adaptation import inputs, coverage
from coco_baselines import data
from coco_baselines.storage import identity, sha_file
from coco_stratified import dataset
from coco_coordinated import run as memory
from . import exchange_selection as previous, repeated_loop as loop
from . import accumulating_regroup as accum, direction_regroup as prior
from . import dependency_regroup as dep

PARENT_RUN='3e4a59c4ce2e26bb49d93a76323f7d45187b02c96fa9e1ada830f5d1f89dee29'
PARENT_SHA='4299ce832ce570d8f0eaed7e98018fdc243c17d4909d714411afa45e6219efca'

def settings():
    return dict(version=1,study='independent_development_preservation_confirmation',
        seeds=[22,23],development_images=2000,test_images=1500,
        development_min_images=40,test_min_images=20,plan_seed=106101,
        coverage_reserve=1.2,extension_limit_per_category=1000,
        solver_seconds=60,extract_chunk=20,extract_batch=8,gradient_updates=0,new_graphs=0,
        max_attempts=8,max_accepted=5,
        scope='New coverage-enriched development selects among an unchanged training-ranked shortlist, in addition to historical development checks. All models freeze before untouched test extraction; no test-based retries or threshold changes.')

def prepare(parent_output,audit_output,loop_output,selection_output,output,local,sha,versions,device='cuda',require_drive=True):
    output=Path(output);local=Path(local)
    for root in (parent_output,audit_output,loop_output,selection_output):
        a=output.resolve();b=Path(root).resolve()
        if a==b or a in b.parents or b in a.parents:raise ValueError('Output overlaps read-only parent')
    ctx=previous.prepare(parent_output,audit_output,loop_output,output/'verified_inputs',local,sha,versions,device,require_drive)
    source=loop.fidelity.open_source(selection_output,PARENT_RUN,local/'read_selection')
    if source.config['source_sha']!=PARENT_SHA or not memory.required(source,'complete')['complete']:
        raise ValueError('Wrong or incomplete exchange-selection parent; no retraining fallback')
    if identity(source.config['config'])!=identity(previous.settings()):raise ValueError('Historical selector changed')
    if source.config['probe_sha']!=ctx['probe']['weights_sha']:raise ValueError('Frozen image readout changed')
    cfg=settings()
    s=inputs.JournalStore(output,dict(config=cfg,source_sha=sha,versions=versions,device=device,
        selection_parent=PARENT_RUN,probe_sha=ctx['probe']['weights_sha'],guards=ctx['study_cfg']),local/'rescue',require_drive)
    ctx.update(store=s,selection_source=source,fresh_cfg=cfg)
    ctx['stamps']+=memory.source_snapshot([source])
    for seed,view in ctx['seeds'].items():
        old=previous.ReadPrefix(source,'seed'+str(seed))
        if accum.weight_sha(memory.required(old,'start'))!=accum.weight_sha(view['initial']):raise ValueError('Exact starting checkpoint changed')
        store=loop.StageStore(s,'seed'+str(seed),identity(dict(start=accum.weight_sha(view['initial']),parent=PARENT_RUN)))
        view.update(store=store,fresh_cfg=cfg)
        for key in ('start','root_agf','root_ggf','old_batch','final_fixed_layout','final_continued_ggf'):
            store.save(key,memory.required(old,key))
        store.save('historical_selected',memory.required(old,'final_agf'))
    s.write('protocol_settings.json',dict(settings=cfg,guards=ctx['study_cfg'],
        primary='Exact retained starting checkpoint, zero extra gradient updates; frozen historical controls are descriptive, not equal-wall-time controls.',
        selection='Original training-ranked order; train, historical development and separate fresh development must all pass. No new proposal generation, metric or threshold.',
        labels='COCO masks supervise screening and acceptance; activations alone at inference. Object evidence and detection are proxies, not complete concept recovery.',
        audit='Allocate both fresh sets using masks only. Exclude all historical images. Extract/score test only after both seeds freeze. Report every final failure; no audit-driven fallback/retry.',
        cost='New DINOv3 extraction and scoring costs are recorded; no baseline retraining, gradients, graph builds, Gaussian fitting or 90-minute timer.'))
    return ctx

def excluded_ids(ctx):
    groups=copy.deepcopy(ctx['provenance']['splits'])
    groups['prior_scaled_test']=memory.required(ctx['trained_source'],'fresh_plan')['splits']['test']
    groups['prior_independent_test']=ctx['source']['ids']['test']
    return groups,set(int(i) for ids in groups.values() for i in ids)

def validate_plan(plan,used,cohort,cfg):
    seen=set(used)
    for split,size,quota in [('confirm',cfg['development_images'],cfg['development_min_images']),('test',cfg['test_images'],cfg['test_min_images'])]:
        ids=plan['splits'][split]
        if len(ids)!=size or len(set(ids))!=size or set(ids)&seen:raise ValueError('Fresh split size/identity/overlap failed: '+split)
        counts=np.asarray(plan['coverage']['split_counts'][split])
        if np.any(counts[cohort]<quota):raise ValueError('Fixed category coverage missing: '+split)
        seen.update(ids)
    if plan['coverage']['category_indices']!=cohort:raise ValueError('Fresh cohort changed')
    if plan['excluded_ids_sha']!=identity(sorted(used)):raise ValueError('Historical exclusions changed')

def joint_allocate(ids,evidence,cohort,cfg,event=lambda *a,**k:None):
    """Joint mask-only binary cover reserves scarce images for both disjoint sets."""
    ids=np.asarray(ids);usable=np.asarray(evidence).all(2).astype(np.int32)
    sizes=[cfg['development_images'],cfg['test_images']]
    quotas=[cfg['development_min_images'],cfg['test_min_images']]
    if len(ids)!=len(set(ids.tolist())) or len(ids)<sum(sizes):raise ValueError('Insufficient unique annotation pool for both fresh sets')
    if np.any(usable[:,cohort].sum(0)<sum(quotas)):raise ValueError('Insufficient category supply for both quotas; no categories are dropped')
    rng=np.random.default_rng(cfg['plan_seed']);order=rng.permutation(len(ids));priority=np.empty(len(ids),int);priority[order]=np.arange(len(ids))
    patterns={}
    for i in order:
        bits=usable[i,cohort]
        if bits.any():
            bucket=patterns.setdefault(bits.tobytes(),[])
            if len(bucket)<sum(quotas):bucket.append(int(i))
    kernel=np.array([i for v in patterns.values() for i in v],int);n=len(kernel);c=len(cohort)
    cover=csc_matrix(usable[kernel][:,cohort].T.astype(float))
    matrix=vstack([block_diag((cover,cover)),hstack([eye(n),eye(n)]),
        block_diag((csc_matrix(np.ones((1,n))),csc_matrix(np.ones((1,n)))))],format='csc')
    lower=np.r_[np.repeat(quotas,c),np.zeros(n),np.zeros(2)]
    upper=np.r_[np.full(2*c,np.inf),np.ones(n),sizes]
    event('joint_quota_solver_start',candidates=len(ids),kernel_images=n,quotas=quotas,sizes=sizes)
    result=milp(np.ones(2*n),integrality=np.ones(2*n),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,lower,upper),options=dict(time_limit=cfg['solver_seconds'],mip_rel_gap=.02))
    event('joint_quota_solver_result',status=int(result.status),message=str(result.message))
    if result.x is None:raise ValueError('No joint feasible quota assignment found. Inspect mask supply/solver log; no metrics, quotas or categories were changed.')
    chosen=[kernel[np.flatnonzero(result.x[j*n:(j+1)*n]>.5)].tolist() for j in range(2)]
    free=np.ones(len(ids),bool)
    for take,quota,size in zip(chosen,quotas,sizes):
        if len(take)>size or not take or np.any(usable[take][:,cohort].sum(0)<quota) or not free[take].all():raise ValueError('Invalid joint solver incumbent')
        free[take]=False
    # All quota cores are reserved before either split is filled.
    target=usable[:,cohort].mean(0)
    for j,take in enumerate(chosen):
        have=usable[take][:,cohort].sum(0);goal=np.maximum(target,quotas[j]/sizes[j])*sizes[j]
        while len(take)<sizes[j]:
            pool=np.flatnonzero(free);pool=pool[np.argsort(priority[pool])[:1024]]
            if not len(pool):raise ValueError('No unique images remain to fill both sets')
            bits=usable[pool][:,cohort];deficit=goal-have
            cost=np.sum(((deficit-bits)**2-deficit**2)/np.maximum(goal,1),axis=1)
            i=int(pool[np.lexsort((priority[pool],cost))[0]]);take.append(i);free[i]=False;have+=usable[i,cohort]
    splits={name:ids[rng.permutation(take)].tolist() for name,take in zip(('confirm','test'),chosen)}
    report=dict(category_indices=cohort,split_counts={name:usable[take].sum(0).tolist() for name,take in zip(('confirm','test'),chosen)},
        minimums=dict(confirm=quotas[0],test=quotas[1]),sizes=dict(confirm=sizes[0],test=sizes[1]),
        candidate_usable_counts=usable.sum(0).tolist(),allocator='joint_binary_quota_cover_then_prevalence_fill',solver_status=int(result.status))
    return splits,report

def fresh_plan(ctx,coco):
    s=ctx['store'];cfg=ctx['fresh_cfg'];groups,used=excluded_ids(ctx);old=s.load('fresh_plan')
    if old is not None:validate_plan(old,used,ctx['cohort'],cfg);return old
    source=memory._discover(ctx['original'].output/'coverage',
        lambda c:c['source_sha']==ctx['original'].config['source_sha'] and c['annotation_sha']==ctx['provenance']['annotation_sha'],ctx['local']/'read_coverage')
    ids=[];rows=[]
    folders=sorted(source.root.glob('coverage_scan_*'))
    for index,folder in enumerate(folders):
        p=memory.required(source,folder.name)
        for i,e in zip(p['ids'],p['evidence']):
            if i not in used:ids.append(int(i));rows.append(e)
        if index%20==0:s.event('mask_pool_read',done=index+1,total=len(folders),unused=len(ids))
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('Missing/duplicated coverage pool')
    supply=np.stack(rows).all(2).sum(0);seen=set(ids)|used
    reserve=int(np.ceil((cfg['development_min_images']+cfg['test_min_images'])*cfg['coverage_reserve']))
    for c in ctx['cohort']:
        if supply[c]>=reserve:continue
        cat=ctx['provenance']['categories'][c]['id']
        candidates=np.random.default_rng(cfg['plan_seed']+c+100).permutation(sorted(set(coco.getImgIds(catIds=[cat]))-seen))[:cfg['extension_limit_per_category']].tolist()
        for lo in range(0,len(candidates),25):
            chosen=candidates[lo:lo+25];key=f'mask_extra_c{c}_{lo:06d}';part=s.load(key)
            if part is None:
                ev=[]
                for i in chosen:
                    im=coco.imgs[i];mask=data.masks_for_image(coco,i,data.geometry(im['width'],im['height']),sorted(coco.cats))
                    ev.append(coverage.evidence(dict(ids=[i],masks=[mask]),ctx['cfg'])[0])
                part=dict(ids=chosen,evidence=np.stack(ev));s.save(key,part)
            if part['ids']!=chosen:raise ValueError('Mask extension order changed')
            ids.extend(chosen);rows.extend(part['evidence']);seen.update(chosen);supply+=part['evidence'].all(2).sum(0)
            s.event('mask_pool_extension',category_index=c,done=lo+len(chosen),usable=int(supply[c]),required=reserve)
            if supply[c]>=reserve:break
        if supply[c]<reserve:raise ValueError(f'Category-index {c} has {supply[c]} usable images, needs {reserve} including reserve. Stop before extraction; never silently drop a category.')
    splits,report=joint_allocate(ids,np.stack(rows),ctx['cohort'],cfg,s.event)
    plan=dict(splits=splits,coverage=report,excluded_ids_sha=identity(sorted(used)),excluded_split_sizes={k:len(v) for k,v in groups.items()},
        annotation_sha=ctx['provenance']['annotation_sha'],rule='Masks only; fixed cohort and disjoint images. Confirmation is selection data; test remains closed until global freeze.')
    validate_plan(plan,used,ctx['cohort'],cfg);s.save('fresh_plan',plan);s.write('fresh_split_plan.json',plan);return plan

def extract(ctx,coco,token_callback,split):
    if split not in ('confirm','test'):raise ValueError('Unknown extraction split')
    s=ctx['store'];cfg=ctx['fresh_cfg'];plan=memory.required(s,'fresh_plan')
    groups,used=excluded_ids(ctx);validate_plan(plan,used,ctx['cohort'],cfg)
    if split=='test':
        frozen=memory.required(s,'all_models_frozen')
        if not frozen['complete'] or frozen['test_used']:raise RuntimeError('Freeze every model before extracting test')
        verify_frozen(ctx,frozen)
    ids=plan['splits'][split];path=ctx['local']/('fresh_'+split+'_normalized.npy');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        x=np.lib.format.open_memmap(path,mode='r+')
        if x.shape!=(len(ids),196,768) or x.dtype!=np.float32:
            x._mmap.close();raise ValueError('Local activation mmap shape/dtype changed')
    else:x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ids),196,768))
    processor=model=None;masks=[];cats=sorted(coco.cats);completed=False
    try:
        for lo in range(0,len(ids),cfg['extract_chunk']):
            chosen=ids[lo:lo+cfg['extract_chunk']];key=f'features_{split}_{lo:06d}';part=s.load(key)
            if part is None:
                if model is None:
                    processor,model=dataset.backbone(ctx['provenance'],ctx['device'],token_callback)
                    s.write(split+'_backbone_verification.json',getattr(model,'_agf_backbone_verification',{}))
                tensors=[];mm=[];geometries=[];hashes=[]
                for b in range(0,len(chosen),cfg['extract_batch']):
                    images=[]
                    for iid in chosen[b:b+cfg['extract_batch']]:
                        im=coco.imgs[iid];file=data.download(data.COCO_BUCKET+'train2017/'+im['file_name'],ctx['local']/'images'/im['file_name'])
                        g=data.geometry(im['width'],im['height'])
                        with Image.open(file) as image:
                            if image.size!=(im['width'],im['height']):raise ValueError('Image/mask geometry mismatch')
                            images.append(data.transform_image(image.convert('RGB'),g))
                        geometries.append(g);hashes.append(sha_file(file));mm.append(data.masks_for_image(coco,iid,g,cats))
                    xx=processor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt').to(ctx['device'])
                    with torch.inference_mode():features=model(**xx).last_hidden_state[:,5:].float().cpu()
                    if tuple(features.shape[1:])!=(196,768) or not bool(torch.isfinite(features).all()):raise ValueError('Invalid DINO patch extraction')
                    tensors.append(features.half());del xx,features
                part=dict(ids=chosen,x=torch.cat(tensors),masks=mm,geometry=geometries,image_hashes=hashes,normalization_sha=identity(ctx['norm']))
                s.save(key,part);del tensors
            if part['ids']!=chosen or part['normalization_sha']!=identity(ctx['norm']):raise ValueError('Extraction shard identity changed')
            x[lo:lo+len(chosen)]=data.normalize(part['x'].float().numpy(),ctx['norm']);x.flush()
            masks.extend(part['masks']);del part;gc.collect()
            s.event('extraction',split=split,done=lo+len(chosen),total=len(ids))
        pack=dict(ids=ids,x=x,masks=masks);observed=coverage.evidence(pack,ctx['cfg']).all(2).sum(0)
        if not np.array_equal(observed,np.asarray(plan['coverage']['split_counts'][split])):raise ValueError('Extracted mask coverage differs from frozen plan')
        # Readout AP requires both positive and negative images for every category.
        y=loop.fidelity.presence(masks,ctx['cohort'])
        if not ((y.sum(0)>0)&(y.sum(0)<len(y))).all():raise ValueError('Missing positive/negative image-readout support')
        stamp=dict(ids=ids,normalized_sha=memory.streaming_array_sha(x),normalization_sha=identity(ctx['norm']),shape=list(x.shape),coverage=observed.tolist(),image_positive_counts=y.sum(0).tolist())
        old=s.load('activation_identity_'+split)
        if old is not None and old!=stamp:raise ValueError('Resumed normalized activations changed')
        s.save('activation_identity_'+split,stamp);s.write(split+'_activation_identity.json',stamp)
        completed=True;return pack
    finally:
        del processor,model
        if not completed:x._mmap.close()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def cumulative_gate(candidate,current,start,root_agf,root_ggf,cfg):
    gate=loop.stage_gate(candidate,current,root_agf,root_ggf,cfg)
    guard=loop.stage_gate(candidate,start,root_agf,root_ggf,cfg,False)
    gate['margins'].update({'pass_start_'+k:v for k,v in guard['margins'].items()})
    gate['failures']+=['pass_start_'+k for k in guard['failures']];gate['passed']=not gate['failures']
    return gate

def measured(ctx,key,snap,split):
    """Score identical weights/readouts/images once, even under multiple roles."""
    signature=identity(dict(weights=accum.weight_sha(snap),spec=snap['spec'],ids=ctx['packs'][split]['ids'],
        split=split,probe_sha=ctx['probe']['weights_sha']))
    r=loop.measured(ctx,'shared_'+signature[:24],snap,split)
    ctx['store'].save(key,r);ctx['store'].write('measurements/'+key+'.json',loop.compact(r))
    return r

def select(ctx):
    s=ctx['store'];cached=s.load('selection_frozen')
    if cached is not None:return cached
    initial=ctx['initial'];current=initial;cfg=ctx['study_cfg'];protocol=cfg['recipe']
    screen=dep.training_screen(ctx)
    if ctx['graph_ids'] and ctx['graph_ids']!=screen['ids']:raise ValueError('Training screen changed')
    train_view=dict(ctx,packs=dict(ctx['packs'],train_screen=screen))
    train_root=accum.measured(train_view,'train_root',initial,'train_screen');train_current=train_root
    roots={}
    for split in ('select','confirm'):
        roots[split]={key:measured(ctx,split+'_'+key,snap,split) for key,snap in [('start',initial),('root_agf',ctx['root_agf']),('root_ggf',ctx['root_ggf'])]}
    dev_current={split:r['start'] for split,r in roots.items()}
    order=accum.proposal_order(ctx['saved_screen'],protocol);s.write('proposal_order.json',order)
    steps=[];accepted=[];groups=set();targets=set();started=time.perf_counter()
    for i,p in enumerate(order):
        key=f'exchange{i:02d}';before=accum.weight_sha(current);row=s.load(key)
        if row is not None:
            if row['before_sha']!=before or row['proposal']!=p:raise ValueError('Resume decision chain changed')
        else:
            why=accum.conflict(p,groups,targets)
            if len(accepted)>=protocol['max_accepted']:why='maximum_accepted_reached'
            row=dict(proposal=p,before_sha=before,accepted=False,skip=why,snapshot=None,checks={})
            if why is None:
                candidate=prior.permute_snapshot(current,p['outgoing'],p['incoming']);s.save(key+'_candidate',candidate)
                train=accum.measured(train_view,key+'_train',candidate,'train_screen')
                gate=accum.positive_gate(train,train_current,train_root,protocol,p['category_index'])
                row.update(candidate_sha=accum.weight_sha(candidate),train_gate=gate,train_report=train,train_deltas=accum.category_deltas(train,train_current))
                if gate['passed']:
                    for split in ('select','confirm'):
                        r=measured(ctx,key+'_'+split,candidate,split);root=roots[split]
                        gate=cumulative_gate(r,dev_current[split],root['start'],root['root_agf'],root['root_ggf'],cfg)
                        row['checks'][split]=dict(gate=gate,measurement=r,
                            category_deltas=accum.category_deltas(r['report'],dev_current[split]['report']),
                            image_ap_deltas=[dict(category_index=c,change=a-b) for c,a,b in zip(r['task']['category_indices'],r['task']['per_category_ap'],dev_current[split]['task']['per_category_ap'])])
                        if not gate['passed']:break
                    model=prior.model(ctx,candidate)
                    row['gate_effects']=dep.gate_effects(ctx,model,screen,memory.required(ctx['dependency'],'dependency_graph'),p)
                    del model
                    if gate['passed']:
                        row['accepted']=True;row['snapshot']=key+'_accepted';s.save(row['snapshot'],candidate)
                else:row['skip']='training_gate_failed'
            row['after_sha']=accum.weight_sha(candidate) if row['accepted'] else before
            s.save(key,row)
        if row['accepted']:
            current=memory.required(s,row['snapshot'])
            if accum.weight_sha(current)!=row['after_sha']:raise ValueError('Accepted checkpoint changed')
            train_current=row['train_report'];dev_current={split:row['checks'][split]['measurement'] for split in roots}
            accepted.append(p);groups.update((p['receiver_group'],p['donor_group']));targets.add(p['category_index'])
        elif accum.weight_sha(current)!=before:raise AssertionError('Rejected edit changed current model')
        steps.append(row);s.write('steps/'+key+'.json',loop.compact(row))
        s.event('exchange_decision',seed=ctx['seed'],done=i+1,total=len(order),proposal=p['id'],category=p['category'],
            accepted=row['accepted'],accepted_total=len(accepted),train_failures=row.get('train_gate',{}).get('failures',[]),
            development_failures={split:check['gate']['failures'] for split,check in row['checks'].items()},skip=row.get('skip'))
    final_gates={}
    if accepted:
        for split,r in roots.items():
            gate=loop.stage_gate(dev_current[split],r['start'],r['root_agf'],r['root_ggf'],cfg)
            if not gate['passed']:raise AssertionError('Final cumulative preservation failed on '+split)
            final_gates[split]=gate
    s.save('final_agf',current)
    out=dict(complete=True,accepted_proposals=accepted,steps=steps,final_sha=accum.weight_sha(current),
        final_identity=identity(dict(weights=accum.weight_sha(current),spec=current['spec'])),final_gates=final_gates,
        seconds=time.perf_counter()-started,gradient_updates=0,test_used=False)
    s.save('selection_frozen',out);s.write('selection_frozen.json',loop.compact(out));return out

def run(ctx,confirm):
    plan=memory.required(ctx['store'],'fresh_plan')
    if confirm['ids']!=plan['splits']['confirm']:raise ValueError('Wrong confirmation split')
    expected=memory.required(ctx['store'],'activation_identity_confirm')
    if memory.streaming_array_sha(confirm['x'])!=expected['normalized_sha']:raise ValueError('Confirmation activations changed')
    for view in ctx['seeds'].values():view['packs']=dict(view['packs'],confirm=confirm)
    old=ctx['store'].load('all_models_frozen')
    if old is not None:verify_frozen(ctx,old);memory.assert_immutable(ctx['stamps']);return old
    rows={str(seed):loop.compact(select(view)) for seed,view in ctx['seeds'].items()}
    out=dict(complete=True,seeds=rows,test_used=False,confirmation_ids_sha=identity(confirm['ids']))
    ctx['store'].save('all_models_frozen',out);ctx['store'].write('all_models_frozen.json',out);memory.assert_immutable(ctx['stamps']);return out

def verify_frozen(ctx,frozen):
    if set(frozen['seeds'])!=set(map(str,ctx['seeds'])):raise ValueError('Frozen seed set changed')
    for seed,view in ctx['seeds'].items():
        snap=memory.required(view['store'],'final_agf')
        actual=identity(dict(weights=accum.weight_sha(snap),spec=snap['spec']))
        if actual!=frozen['seeds'][str(seed)]['final_identity']:raise ValueError('Frozen weights/readouts changed before test')
    if frozen['confirmation_ids_sha']!=identity(memory.required(ctx['store'],'fresh_plan')['splits']['confirm']):raise ValueError('Frozen confirmation image IDs changed')

def audit(ctx,test):
    s=ctx['store'];frozen=memory.required(s,'all_models_frozen')
    if not frozen['complete'] or frozen['test_used']:raise ValueError('Both seeds must freeze before test')
    verify_frozen(ctx,frozen)
    if test['ids']!=memory.required(s,'fresh_plan')['splits']['test']:raise ValueError('Wrong untouched test split')
    if memory.streaming_array_sha(test['x'])!=memory.required(s,'activation_identity_test')['normalized_sha']:raise ValueError('Test activations changed')
    old=s.load('complete')
    if old is not None:memory.assert_immutable(ctx['stamps']);return old
    rows=[];pairs=[]
    for seed,view in ctx['seeds'].items():
        view['packs']['test']=test;reports={}
        selected=memory.required(view['store'],'final_agf');start=memory.required(view['store'],'start')
        if identity(dict(weights=accum.weight_sha(selected),spec=selected['spec']))!=frozen['seeds'][str(seed)]['final_identity']:raise ValueError('Frozen weights/readouts changed')
        for name,snap in [('start',start),('final_agf',selected),('root_agf',view['root_agf']),('root_ggf',view['root_ggf'])]:
            r=measured(view,'audit_'+name,snap,'test');reports[name]=r
            rows.append(dict(seed=seed,model=name,**loop.bank.scalar(r['report']),image_ap=r['task']['macro_ap'],weights_sha=accum.weight_sha(snap)))
        np.savez_compressed(view['store'].results/'image_readout_scores.npz',image_ids=np.array(test['ids']),category_indices=np.array(ctx['cohort']),
            labels=reports['final_agf']['task']['labels'],**{name:r['task']['scores'] for name,r in reports.items()})
        gate=loop.stage_gate(reports['final_agf'],reports['start'],reports['root_agf'],reports['root_ggf'],ctx['study_cfg'],bool(frozen['seeds'][str(seed)]['accepted_proposals']))
        ci=accum.paired_audit(view,selected,start,'test','selected_vs_start')
        ci['scope']='Paired whole-image uncertainty on untouched test, conditional on these fixed starts and development-selected models; not a new independent training-seed study.'
        view['store'].save('paired_selected_vs_start',ci);view['store'].write('paired/selected_vs_start.json',ci)
        image_ci=loop.fidelity.paired_ap(reports['final_agf']['task']['labels'],reports['final_agf']['task']['scores'],reports['start']['task']['scores'],ctx['study_cfg']['bootstrap'],106102)
        pair=dict(seed=seed,group_paired=ci,image_paired=image_ci,gate=gate,
            category_changes=accum.category_deltas(reports['final_agf']['report'],reports['start']['report']),
            image_ap_changes=[dict(category_index=c,change=a-b) for c,a,b in zip(ctx['cohort'],reports['final_agf']['task']['per_category_ap'],reports['start']['task']['per_category_ap'])])
        pairs.append(loop.compact(pair));view['store'].write('audit_comparison.json',loop.compact(pair))
    with (s.results/'comparison.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=dict(complete=True,models=rows,paired=pairs,accepted={k:len(v['accepted_proposals']) for k,v in frozen['seeds'].items()},
        gradient_updates=0,new_graphs=0,development_images=len(ctx['packs'].get('confirm',{}).get('ids',[])) or ctx['fresh_cfg']['development_images'],
        test_images=len(test['ids']),scope=ctx['fresh_cfg']['scope'],
        interpretation='An added independent development check is useful only if selected gains and every unchanged preservation guard generalize to this untouched test. Zero accepted edits is a safe fallback, not an adaptation win. No test-triggered retries or post-test threshold changes.')
    memory.assert_immutable(ctx['stamps']);s.save('complete',out);s.write('summary.json',out);return out

def export(ctx):
    path=loop.export(ctx);target=path.with_name('agf_fresh_preservation_'+ctx['store'].run_id[:12]+'_results.zip');path.replace(target);return target
