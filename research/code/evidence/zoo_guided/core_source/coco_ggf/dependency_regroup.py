"""Supervised direction dependencies, exact exchanges and ordinary GGF recovery.

The graph measures fixed-gate removal effects at the saved model. It is a
shortlisting instrument; actual exchanges must check the changed Top-K gate.
Training masks supply category evidence. Readouts stay fixed, and evaluation
images never choose proposals. No population or optimality claim is implied.
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
from coco_adaptation import core, inputs
from coco_baselines import metrics, training
from coco_baselines.storage import identity, serial
from coco_coordinated import run as memory
from . import direction_regroup as prior, bank_experiment as bank, owner_routing as old, study

PRIOR_RUN='c2b0c91ffcd32f0a43e27618d7ada6f2266a93233a26f38144f91c589713e3fb'
PRIOR_SHA='48915740977e2e78a4cd158111d8c8373afe34f34a17a87eb4f1d910f68569bf'


def settings():
    cfg=prior.settings()
    cfg.update(version=1, study='dependency_graph_direction_exchange',
        candidate_pool=64, per_category_pool=2, screen_per_policy=8,
        graph_recovery_candidates=2, capture_recovery_candidates=1,
        recovery_candidates=3, restricted_updates=0, evaluate_every=100,
        scope='One saved seed, supervised historical development. Separate local repair and overall advantage; final test stays closed.')
    return cfg


def prepare(parent, previous_output, output, local, sha, versions, device, require_drive=True):
    cfg=settings();output=Path(output);local=Path(local)
    ctx=old.prepare(parent,output/'inputs',local,sha,versions,device=device,require_drive=require_drive)
    ctx['protocol']=cfg
    ctx['cfg']=dict(ctx['eval_cfg'],**{k:cfg[k] for k in ('lr','snr','lambda_capture','lambda_detection','grad_clip','images_per_update')})
    w,h,t,row=old.source_start(ctx,cfg['seed'])
    ctx['initial']=bank.native_snapshot(ctx,w,t,h,1024,3,16)
    ctx['source_weights_sha']=row['weights_sha'];ctx['prior']=None
    previous_output=Path(previous_output)
    if output.resolve()==previous_output.resolve() or previous_output.resolve() in output.resolve().parents or output.resolve() in previous_output.resolve().parents:
        raise ValueError('The direction-analysis parent must remain read-only')
    root=previous_output/'checkpoints'/PRIOR_RUN[:20]
    if root.exists():
        source=memory.FrozenStore(root,local/'read_direction_profiles')
        if source.run_id!=PRIOR_RUN or source.config['source_sha']!=PRIOR_SHA:
            raise ValueError('Wrong direction-analysis parent')
        roles=json.loads((source.results/'split_roles.json').read_text())
        if roles['source_weights_sha']!=row['weights_sha'] or roles['category_readouts']!=t:
            raise ValueError('Direction profiles belong to different weights or readouts')
        for split in ('train','select'):
            if roles['ids'][split]!=ctx['packs'][split]['ids']:raise ValueError('Profile image identities changed')
        ctx['prior']=source;ctx['stamps']+=memory.source_snapshot([source])
    ctx['store']=inputs.JournalStore(output,dict(config=cfg,source_sha=sha,versions=versions,
        device=device,parent_run=ctx['source'].run_id,activation_run=ctx['original'].run_id,
        direction_profile_parent=PRIOR_RUN if ctx['prior'] else None),local/'dependency_rescue',require_drive)
    ctx['store'].write('split_roles.json',dict(ids={k:p['ids'] for k,p in ctx['packs'].items()},
        category_readouts=t,source_weights_sha=row['weights_sha'],
        train='Profiles, dependency graph, proposal ranking and frozen exchange screen; 2000 images for recovery',
        select='400 historical development images for checkpoints and gates; previously inspected',
        confirm='Only one frozen nominee, if eligible: historical disjoint confirmation, not fresh test',
        final_test_opened=False))
    ctx['store'].event('prepared',profile_reuse_available=ctx['prior'] is not None,restricted_updates=0)
    return ctx


def profiles(ctx,split):
    s=ctx['store'];key='profiles_'+split;value=s.load(key)
    if value is None and ctx.get('prior') is not None:
        value=ctx['prior'].load(key)
        if value is not None:
            targets=ctx['initial']['spec']['targets']
            if value['gain'].shape!=(len(targets),ctx['initial']['spec']['groups']*3):raise ValueError('Wrong profile matrix shape')
            if value['category_indices'].tolist()!=[t['category_index'] for t in targets]:raise ValueError('Profile category order changed')
            s.save(key,value);s.event('verified_profile_import',split=split,parent_run=PRIOR_RUN,
                numerical_versions_differ=ctx['prior'].config['versions']!=s.config['versions'],
                role='Historical training statistics reused for proposals; graph and all models are rescored now')
    if value is None:value=prior.profiles(ctx,split)
    folder=s.results/'directions';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/(split+'_profiles.npz'),**value)
    return value


def timing(ctx):
    result=prior.timing(ctx)
    updates=ctx['protocol']['recovery_updates']*(1+ctx['protocol']['graph_recovery_candidates']+ctx['protocol']['capture_recovery_candidates'])
    result.update(maximum_initial_recovery_updates=updates,conditional_search_budget_control_updates=updates,
        rough_initial_gradient_compute_minutes=result['transient_seconds_per_gradient_update']*updates/60,
        maximum_frozen_exchanges=2*ctx['protocol']['screen_per_policy'],maximum_exchange_pool=ctx['protocol']['candidate_pool'],
        note='Shared ordinary no-edit control; duplicate selected edits are trained once. Graph and exact screening time are excluded.')
    ctx['store'].write('timing.json',result)
    return result


def candidate_pool(train,targets,cfg):
    ranked=prior.rank_proposals(train,targets,cfg)
    # Keep high capture-only candidates, then broaden object coverage instead of
    # filling all slots with exchanges for a single category.
    chosen=ranked[:cfg['screen_per_policy']];seen={r['id'] for r in chosen}
    per={t['category_index']:{} for t in targets}
    for p in ranked:per[p['category_index']].setdefault(p['incoming'],[]).append(p)
    bundles={c:sorted(v.values(),key=lambda b:(-max(r['mean_predicted_gain'] for r in b),-max(r['target_predicted_gain'] for r in b),b[0]['id'])) for c,v in per.items()}
    for depth in range(cfg['per_category_pool']):
        rows=sorted((v[depth] for v in bundles.values() if len(v)>depth),
            key=lambda b:(-max(r['mean_predicted_gain'] for r in b),b[0]['id']))
        for bundle in rows:
            additions=[p for p in bundle if p['id'] not in seen]
            if len(chosen)+len(additions)>cfg['candidate_pool']:continue
            chosen.extend(additions);seen.update(p['id'] for p in additions)
    return chosen


def training_screen(ctx):
    cfg=dict(old.settings(),assignment_images=ctx['protocol']['screening_images'])
    targets=ctx['initial']['spec']['targets'];p=ctx['packs']['train']
    ix,coverage=old.assignment_indices(p,[t['category_index'] for t in targets],cfg,ctx['cfg'])
    screen=dict(x=np.array(p['x'][ix],copy=True),ids=[p['ids'][i] for i in ix],masks=[p['masks'][i] for i in ix])
    ctx['store'].write('training_screen_coverage.json',coverage)
    return screen


@torch.no_grad()
def dependency_shard(ctx,pack,lo,hi,atoms):
    """Remove one coefficient after the original gate; exact conditional effect.

    The gate is held fixed ONLY for this graph. Capture and AP use original
    readout groups. Actual exchanging and gate displacement are tested later.
    """
    m=prior.model(ctx,ctx['initial']);m.eval();targets=m.targets;C=len(targets);J=len(atoms)
    D=m.W_dec.detach();DD=D[atoms];dn=DD.square().sum(1)
    num=np.zeros((C,J));den=np.zeros(C);err=np.zeros(C);counts=np.zeros(C,int)
    rec=np.zeros(J);energy=0.;xsum=np.zeros(m.d);patches=0
    hs=[];zs=[];fs=[];cs=[];ims=[];annotations=[];gates=[]
    lookup={t['category_index']:k for k,t in enumerate(targets)};ag=np.array(atoms)//3
    for i in range(lo,hi):
        x=torch.as_tensor(np.array(pack['x'][i],copy=True),device=ctx['device'])
        z=m.encode(x);flat=z.reshape(len(x),-1);residual=x-flat@D
        gates.append(np.packbits((z.norm(dim=-1)>1e-6).cpu().numpy(),axis=1))
        rec+=(2*flat[:,atoms]*(residual@DD.T)+flat[:,atoms].square()*dn).double().sum(0).cpu().numpy()
        energy+=float(x.double().square().sum());xsum+=x.double().sum(0).cpu().numpy();patches+=len(x)
        pick=np.random.default_rng(ctx['cfg']['patch_sample_seed']+int(pack['ids'][i])).choice(len(x),min(len(x),ctx['cfg']['detection_patches']),replace=False)
        hs.append(z[pick][:,m.detector_ids].norm(dim=-1).cpu().numpy())
        zs.append(flat[pick][:,atoms].cpu().numpy())
        fs.append(pack['masks'][i]['fraction'][pick]);cs.append(pack['masks'][i]['crowd'][pick]);ims.extend([i]*len(pick))
        ww,_,cats=core.object_weights([pack['masks'][i]],list(lookup),ctx['cfg'])
        if not len(cats):continue
        wt=torch.as_tensor(ww,dtype=x.dtype,device=x.device);dx=wt@x;dz=wt@flat
        for k,cat in enumerate(cats):
            c=lookup[int(cat)];g=targets[c]['contrast_group'];ids=list(range(3*g,3*g+3))
            r=dx[k]-dz[k,ids]@D[ids];take=np.flatnonzero(ag==g)
            if len(take):
                a=[atoms[j] for j in take];v=dz[k,a];dd=D[a]
                num[c,take]+=(2*v*(dd@r)+v.square()*dd.square().sum(1)).double().cpu().numpy()
            den[c]+=float(dx[k].double().square().sum());err[c]+=float(r.double().square().sum());counts[c]+=1
        annotations.extend([dict(image_id=int(pack['ids'][i]),annotation_id=int(o['annotation_id']),category=int(o['category'])) for o in pack['masks'][i]['instances']])
    out=dict(capture_numerator=num,denominator=den,error=err,object_counts=counts,
        reconstruction_numerator=rec,raw_energy=energy,xsum=xsum,patches=patches,
        heat=np.concatenate(hs),coefficients=np.concatenate(zs),fraction=np.concatenate(fs),
        crowd=np.concatenate(cs),patch_image_index=np.array(ims),image_ids=pack['ids'][lo:hi],annotations=annotations,
        base_gate_packed=np.stack(gates))
    del m;gc.collect()
    return out


def build_graph(ctx,pool,screen):
    s=ctx['store'];cached=s.load('dependency_graph')
    if cached is not None:return cached
    atoms=sorted({p[k] for p in pool for k in ('outgoing','incoming')})
    if not atoms:raise ValueError('No positive training contribution candidates')
    totals=[];pool_sha=identity(pool)
    for lo in range(0,len(screen['ids']),ctx['protocol']['shard_images']):
        hi=min(lo+ctx['protocol']['shard_images'],len(screen['ids']));key=f'dependency_shard_{lo:05d}'
        shard=s.load(key)
        if shard is None:
            shard=dependency_shard(ctx,screen,lo,hi,atoms)
            shard.update(atom_ids=atoms,pool_sha=pool_sha);s.save(key,shard)
        if shard['atom_ids']!=atoms or shard['pool_sha']!=pool_sha or shard['image_ids']!=screen['ids'][lo:hi]:raise ValueError('Dependency shard identity changed')
        totals.append(shard);s.event('dependency_graph_progress',done=hi,total=len(screen['ids']),directions=len(atoms))
    summed=lambda key:sum((t[key] for t in totals))
    den=summed('denominator');error=summed('error');counts=summed('object_counts')
    heat=np.concatenate([t['heat'] for t in totals]);coeff=np.concatenate([t['coefficients'] for t in totals])
    frac=np.concatenate([t['fraction'] for t in totals]);crowd=np.concatenate([t['crowd'] for t in totals])
    labels,valid=metrics.binary_labels(dict(fraction=frac,crowd=crowd),ctx['cfg'])
    targets=ctx['initial']['spec']['targets'];cats=[t['category_index'] for t in targets]
    if (den<=1e-12).any() or (counts==0).any():raise ValueError('Graph has missing object contrasts; no category is dropped')
    cap_damage=summed('capture_numerator')/den[:,None]
    baseline_ap=np.array([metrics.ap(heat[:,c],labels[:,cat],valid[:,cat]) for c,cat in enumerate(cats)],float)
    ap_damage=np.zeros_like(cap_damage);positive_images=[];negative_images=[]
    images=np.concatenate([t['patch_image_index'] for t in totals])
    for c,cat in enumerate(cats):
        p,n=metrics.image_counts(labels[:,cat],valid[:,cat],images);positive_images.append(p);negative_images.append(n)
        if min(p,n)<ctx['cfg']['decision_min_images']:raise ValueError('Graph detection coverage insufficient')
        for j,atom in enumerate(atoms):
            if atom//3==targets[c]['detector_group']:
                h=np.sqrt(np.maximum(heat[:,c].astype(float)**2-coeff[:,j].astype(float)**2,0.))
                ap_damage[c,j]=baseline_ap[c]-metrics.ap(h,labels[:,cat],valid[:,cat])
    energy=summed('raw_energy')-float(np.square(summed('xsum')).sum())/summed('patches')
    recon_damage=summed('reconstruction_numerator')/max(energy,1e-12)
    for a in (cap_damage,ap_damage,recon_damage,baseline_ap):
        if not np.isfinite(a).all():raise FloatingPointError('Nonfinite dependency evidence')
    graph=dict(atom_ids=atoms,category_indices=cats,category_names=[t['category'] for t in targets],
        capture_damage=cap_damage,ap_damage=ap_damage,reconstruction_ve_damage=recon_damage,
        original_capture=1-error/den,original_ap=baseline_ap,object_counts=counts,
        positive_images=positive_images,negative_images=negative_images,image_ids=screen['ids'],pool_sha=pool_sha,
        base_gate_packed=np.concatenate([t['base_gate_packed'] for t in totals]),
        scope='Training-only, fixed-gate one-direction removal at original readouts; unannotated roles and direction interactions may be missing')
    s.save('dependency_graph',graph)
    folder=s.results/'dependencies';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'graph.npz',**{k:v for k,v in graph.items() if isinstance(v,np.ndarray)},atom_ids=np.array(atoms),category_indices=np.array(cats),image_ids=np.array(screen['ids']))
    edges=[]
    for c,t in enumerate(targets):
        for j,atom in enumerate(atoms):
            if atom//3 in (t['contrast_group'],t['detector_group']):
                edges.append(dict(direction=atom,group=atom//3,category=t['category'],category_index=cats[c],
                    capture_loss_if_removed_pp=100*cap_damage[c,j],AP_loss_if_removed_pp=100*ap_damage[c,j],
                    object_count=int(counts[c]),positive_images=positive_images[c],negative_images=negative_images[c]))
    csv_write(folder/'direction_object_dependencies.csv',edges)
    s.write('dependencies/graph_scope.json',{k:v for k,v in graph.items() if not isinstance(v,np.ndarray)})
    plot_graph(ctx,graph)
    return graph


def graph_rank(pool,graph,cfg):
    index={a:j for j,a in enumerate(graph['atom_ids'])};rows=[]
    for p in pool:
        q=copy.deepcopy(p);js=[index[p[k]] for k in ('outgoing','incoming')]
        cap=np.maximum(graph['capture_damage'][:,js],0).sum(1)
        det=np.maximum(graph['ap_damage'][:,js],0).sum(1)
        # Receiver's net target capture is already in the proposal algebra.
        cap[graph['category_indices'].index(p['category_index'])]=0
        harm=(cap+det)/cfg['max_category_drop']
        reconstruction=max(float(np.maximum(graph['reconstruction_ve_damage'][js],0).sum()),0)/cfg['max_ve_drop']
        # Both vectors stay in the dictionary. Removal reconstruction importance
        # is diagnostic, not a predicted exchange loss; actual VE is screened.
        cost=float(harm.mean()+harm.max())
        q.update(dependency_cost=cost,dependency_score=float(p['target_predicted_gain']/cfg['minimum_target_gain']-cost),
            capture_damage_estimate=cap.tolist(),AP_damage_estimate=det.tolist(),
            reconstruction_removal_cost=reconstruction,
            dependent_categories=[graph['category_indices'][c] for c in np.flatnonzero(harm>0)],
            high_risk_categories=[graph['category_indices'][c] for c in np.flatnonzero(harm>1)],
            incoming_dependencies=[dict(category=graph['category_names'][c],category_index=graph['category_indices'][c],
                capture_damage_pp=100*float(graph['capture_damage'][c,js[1]]),AP_damage_pp=100*float(graph['ap_damage'][c,js[1]]))
                for c in range(len(harm)) if abs(graph['capture_damage'][c,js[1]])+abs(graph['ap_damage'][c,js[1]])>1e-10])
        rows.append(q)
    rows.sort(key=lambda q:(-q['dependency_score'],-q['mean_predicted_gain'],q['id']))
    return rows


def csv_write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(serial(rows))


def plot_graph(ctx,graph):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    weights=np.maximum(graph['capture_damage'],0)+np.maximum(graph['ap_damage'],0)
    atoms=np.argsort(-weights.max(0))[:24];cats=np.argsort(-weights[:,atoms].max(1))[:20]
    fig,ax=plt.subplots(figsize=(12,6));im=ax.imshow(weights[np.ix_(cats,atoms)]*100,aspect='auto',cmap='magma')
    ax.set_xticks(range(len(atoms)),[str(graph['atom_ids'][j]) for j in atoms],rotation=90)
    ax.set_yticks(range(len(cats)),[graph['category_names'][j] for j in cats]);ax.set_xlabel('Direction ID')
    ax.set_title('Which objects lose capture or detection if a direction is removed?')
    fig.colorbar(im,ax=ax,label='Positive capture loss + AP loss (percentage points)')
    fig.tight_layout();folder=ctx['store'].results/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'dependency_graph.png',dpi=160);plt.close(fig)


def local_gate(report,base,control,cfg,target):
    """Predeclared diagnostic gate: target improves, mean does not fall.

    Original overall gain threshold is still assessed separately; passing this
    local gate alone never establishes or adopts an overall AGF advantage.
    """
    rule=dict(cfg,minimum_mean_gain=0.)
    g=bank.acceptance(report,base,control,base,rule,target)
    a={r['category_index']:r for r in report['categories']}
    for name,ref in [('original',base),('matched_control',control)]:
        b={r['category_index']:r for r in ref['categories']}
        for c in a:g['margins'][f'{name}_detector_{c}']=a[c]['ap']-b[c]['ap']+cfg['max_category_drop']
    g['failures']=[k for k,v in g['margins'].items() if not np.isfinite(v) or v < -1e-12]
    g['passed']=not g['failures'];return g


def overall_gate(report,base,control,cfg,target):
    local=local_gate(report,base,control,cfg,target)
    g=bank.acceptance(report,base,control,base,cfg,target)
    g['margins'].update({k:v for k,v in local['margins'].items() if 'detector_' in k})
    g['failures']=[k for k,v in g['margins'].items() if not np.isfinite(v) or v < -1e-12]
    g['passed']=not g['failures'];return g


@torch.no_grad()
def gate_effects(ctx,m,screen,graph,proposal):
    targets=m.targets;C=len(targets);totals=np.zeros((C,5));global_sums=np.zeros(5);patches=0
    changed_groups={proposal['receiver_group'],proposal['donor_group']}
    for i,mm in enumerate(screen['masks']):
        original=np.unpackbits(graph['base_gate_packed'][i],axis=1,count=m.n_groups).astype(bool)
        x=torch.as_tensor(np.array(screen['x'][i],copy=True),device=ctx['device'])
        edited=(m.encode(x).norm(dim=-1)>1e-6).cpu().numpy()
        added=edited&~original;lost=original&~edited
        unrelated=np.ones(m.n_groups,bool);unrelated[list(changed_groups)]=False
        global_sums+=np.array([(original!=edited).any(1).sum(),added.sum(),lost.sum(),added[:,unrelated].sum(),lost[:,unrelated].sum()]);patches+=len(x)
        known=mm['crowd']<=ctx['cfg']['negative_fraction']
        for c,t in enumerate(targets):
            fg=(mm['fraction'][:,t['category_index']]/255.)*known;mass=fg.sum()
            if mass<=0:continue
            cg=t['contrast_group'];dg=t['detector_group']
            totals[c]+=np.array([mass,fg@original[:,cg],fg@edited[:,cg],fg@original[:,dg],fg@edited[:,dg]])
        if (i+1)%ctx['protocol']['shard_images']==0:
            ctx['store'].event('exchange_gate_progress',id=proposal['id'],done=i+1,total=len(screen['ids']))
    rows=[]
    for c,t in enumerate(targets):
        mass=totals[c,0]
        rows.append(dict(category=t['category'],category_index=t['category_index'],foreground_mass=float(mass),
            original_capture_admission=float(totals[c,1]/max(mass,1e-12)),edited_capture_admission=float(totals[c,2]/max(mass,1e-12)),
            original_detector_admission=float(totals[c,3]/max(mass,1e-12)),edited_detector_admission=float(totals[c,4]/max(mass,1e-12))))
    return dict(patches=patches,patch_fraction_with_changed_gates=float(global_sums[0]/patches),
        mean_groups_added=float(global_sums[1]/patches),mean_groups_lost=float(global_sums[2]/patches),
        mean_unrelated_groups_added=float(global_sums[3]/patches),mean_unrelated_groups_lost=float(global_sums[4]/patches),
        categories=rows,scope='All screen patches; foreground mass weighted admission. Secondary gate diagnostic, not AP sampling or capture.')


def frozen_screen(ctx,pool,graph,screen):
    s=ctx['store'];cached=s.load('dependency_screen')
    if cached is not None:return cached
    ranked=graph_rank(pool,graph,ctx['protocol']);n=ctx['protocol']['screen_per_policy']
    capture_order=sorted(pool,key=lambda p:(-p['mean_predicted_gain'],-p['target_predicted_gain'],p['id']))[:n]
    graph_order=ranked[:n];ids={p['id'] for p in capture_order+graph_order}
    by_id={p['id']:p for p in ranked};view=dict(ctx,packs=dict(ctx['packs'],train_screen=screen))
    m=prior.model(ctx,ctx['initial']);ref=bank.evaluate(view,m,'train_screen');del m
    s.write('dependencies/ranked_exchanges.json',ranked)
    rows=[]
    for pid in sorted(ids):
        p=by_id[pid];key='dependency_frozen_'+pid;row=s.load(key)
        if row is None:
            start=prior.permute_snapshot(ctx['initial'],p['outgoing'],p['incoming'])
            m=prior.model(ctx,start);m.eval();gating=gate_effects(ctx,m,screen,graph,p)
            report=bank.evaluate(view,m,'train_screen');del m
            a={r['category_index']:r for r in report['categories']};b={r['category_index']:r for r in ref['categories']}
            deltas=[dict(category=k,category_index=k,category_name=a[k]['category'],
                capture=a[k]['capture']-b[k]['capture'],ap=a[k]['ap']-b[k]['ap']) for k in a]
            c=p['category_index'];row=dict(proposal=p,report=report,
                mean_capture_delta=report['macro_capture']-ref['macro_capture'],target_capture_delta=a[c]['capture']-b[c]['capture'],
                ap_delta=report['macro_ap']-ref['macro_ap'],ve_delta=report['reconstruction']['ve']-ref['reconstruction']['ve'],
                category_deltas=deltas,local_gate=local_gate(report,ref,ref,ctx['protocol'],c),
                overall_gate=overall_gate(report,ref,ref,ctx['protocol'],c),
                gate_changes=gating,
                readouts='Original capture and detector groups; no reassignment after edits',
                slot_to_original_atom_id=start['spec']['atom_ids'])
            s.save(key,row)
        row.update(in_capture_shortlist=pid in {p['id'] for p in capture_order},in_graph_shortlist=pid in {p['id'] for p in graph_order})
        rows.append(row);s.write('frozen_screen/'+pid+'.json',row)
        s.event('frozen_dependency_exchange',id=pid,done=len(rows),total=len(ids),target_gain_pp=100*row['target_capture_delta'],
            AP_change_pp=100*row['ap_delta'],locally_safe=row['local_gate']['passed'],failures=row['local_gate']['failures'])
    by_row={r['proposal']['id']:r for r in rows}
    safe=[by_row[p['id']] for p in graph_order if by_row[p['id']]['local_gate']['passed']]
    safe.sort(key=lambda r:(-r['mean_capture_delta'],-r['target_capture_delta'],r['proposal']['id']))
    chosen=[];seen=set();targets=set()
    for r in safe:
        c=r['proposal']['category_index']
        if c in targets:continue
        chosen.append(dict(policy='dependency_graph',**r));seen.add(r['proposal']['id']);targets.add(c)
        if len(chosen)>=ctx['protocol']['graph_recovery_candidates']:break
    # Retain one previous-style counterfactual even if unsafe: a diagnosis arm,
    # never an automatic adoption. A duplicate is trained only once.
    originals=[by_row[p['id']] for p in capture_order]
    originals.sort(key=lambda r:(-r['mean_capture_delta'],-r['target_capture_delta'],r['proposal']['id']))
    for r in originals[:ctx['protocol']['capture_recovery_candidates']]:
        if r['proposal']['id'] not in seen:chosen.append(dict(policy='capture_only_reference',**r))
        else:next(q for q in chosen if q['proposal']['id']==r['proposal']['id'])['policy']='both_policies'
    policy_summary={name:dict(shortlist=[p['id'] for p in order],locally_safe=sum(by_row[p['id']]['local_gate']['passed'] for p in order),
        screened=len(order)) for name,order in [('capture_only',capture_order),('dependency_graph',graph_order)]}
    out=dict(rows=rows,chosen=chosen,reference=ref,policy_summary=policy_summary,
        selection_rule='Same training pool and equal shortlist limits. Graph-ranked candidates must pass exact frozen local safeguards; one capture-only reference retained.')
    s.save('dependency_screen',out);s.write('frozen_screen.json',out)
    return out


def run_recovery(ctx,screen):
    s=ctx['store'];cached=s.load('dependency_recovery')
    if cached is not None:return cached
    baseline=s.load('baseline_report')
    if baseline is None:
        m=prior.model(ctx,ctx['initial']);baseline=bank.evaluate(ctx,m);del m;s.save('baseline_report',baseline)
    full=prior.fit(ctx,'continued_ggf',ctx['initial'],affected=None)
    controls={r['update']:r for r in full['history']};decisions=[];eligible=[];local_eligible=[]
    for index,row in enumerate(screen['chosen']):
        p=row['proposal'];start=prior.permute_snapshot(ctx['initial'],p['outgoing'],p['incoming'])
        edited=prior.fit(ctx,f'exchange{index}_edited',start,affected=None)
        if edited['batch_sha']!=full['batch_sha']:raise AssertionError('Continuation samples differ')
        checks=[]
        for e in edited['history']:
            c=controls[e['update']]
            lg=local_gate(e['report'],baseline,c['report'],ctx['protocol'],p['category_index'])
            og=overall_gate(e['report'],baseline,c['report'],ctx['protocol'],p['category_index'])
            q=dict(update=e['update'],checkpoint=e['checkpoint'],local_gate=lg,overall_gate=og,
                actual_deltas={k:bank.scalar(e['report'])[k]-bank.scalar(c['report'])[k] for k in ('capture','ap','ve','mse')})
            checks.append(q)
            item=dict(proposal=p,policy=row['policy'],checkpoint=e['checkpoint'],report=e['report'],update=e['update'])
            if lg['passed']:local_eligible.append(item)
            if og['passed']:eligible.append(item)
        decision=dict(proposal=p,policy=row['policy'],checks=checks,edited=edited['history'],control=full['history'],batch_sha=edited['batch_sha'])
        decisions.append(decision);s.write('decisions/'+p['id']+'.json',decision)
    used=ctx['protocol']['recovery_updates']*(1+len(decisions))
    candidates=eligible or local_eligible
    nominee=max(candidates,key=lambda r:(r['report']['macro_capture'],r['report']['macro_ap'],-r['report']['reconstruction']['mse'])) if candidates else None
    budget=None;budget_gates=None;confirmation=None;accepted=False;confirmed_local=False
    if nominee:
        budget=prior.fit(ctx,'search_budget_continued_ggf',ctx['initial'],updates=used)
        strong=max(budget['history'],key=lambda r:(r['report']['macro_capture']+.2*r['report']['macro_ap'],-r['report']['reconstruction']['mse']))
        target=nominee['proposal']['category_index']
        budget_gates=dict(local=local_gate(nominee['report'],baseline,strong['report'],ctx['protocol'],target),
            overall=overall_gate(nominee['report'],baseline,strong['report'],ctx['protocol'],target))
        if budget_gates['local']['passed']:
            norm=memory.required(ctx['original'],'normalization')
            ctx['packs']['confirm']=memory.normalized_pack(ctx['original'],'confirm',ctx['provenance'],norm,ctx['local'],20)
            sets=[set(p['ids']) for p in ctx['packs'].values()]
            if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)):raise ValueError('Confirmation overlap')
            reports={}
            for name,snap in [('original',ctx['initial']),('edited',s.load(nominee['checkpoint'])),('budget_control',s.load(strong['checkpoint']))]:
                m=prior.model(ctx,snap);reports[name]=bank.evaluate(ctx,m,'confirm');del m
            lg=local_gate(reports['edited'],reports['original'],reports['budget_control'],ctx['protocol'],target)
            og=overall_gate(reports['edited'],reports['original'],reports['budget_control'],ctx['protocol'],target)
            confirmed_local=lg['passed'];accepted=og['passed'] and budget_gates['overall']['passed'] and bool(eligible)
            confirmation=dict(reports=reports,local_gate=lg,overall_gate=og,scope='Historical disjoint confirmation; previously used in research, not fresh test')
    out=dict(complete=True,accepted=accepted,confirmed_local_repair=confirmed_local,
        baseline=baseline,continued=full['history'],decisions=decisions,nominee=nominee,
        eligible_checkpoints=len(eligible),locally_eligible_checkpoints=len(local_eligible),
        search_update_cost=used,search_budget_control=budget['history'] if budget else None,
        search_budget_gates=budget_gates,confirmation=confirmation,final_test_opened=False,
        policy_summary=screen['policy_summary'],scope=ctx['protocol']['scope'],
        interpretation='Graph safety, local repair and overall advantage are distinct outcomes. All recoveries use ordinary full-model GGF updates.')
    s.save('dependency_recovery',out);s.write('summary.json',out);memory.assert_immutable(ctx['stamps'])
    return out


def comparisons(ctx,out):return prior.comparisons(ctx,out)
def paired_diagnostics(ctx,out):return prior.paired_diagnostics(ctx,out)


def export(ctx):
    s=ctx['store'];path=s.output/f'agf_dependency_regroup_{s.run_id[:12]}_results.zip';manifest={}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(s.results.rglob('*')):
            if p.is_file():
                n='results/'+p.relative_to(s.results).as_posix();z.write(p,n);manifest[n]=hashlib.sha256(p.read_bytes()).hexdigest()
        z.writestr('artifact_hashes.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise OSError('Export corruption')
    return path
