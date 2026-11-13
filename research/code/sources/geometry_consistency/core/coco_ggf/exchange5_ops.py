"""Width-aware, frozen-weight direction analysis derived from the earlier exchange implementation.
Algebra and fixed-gate removal are proposal diagnostics only. Exact changed Top-16
predictions determine acceptance. No optimization functions are called here.
"""
import copy,csv,gc
import numpy as np
import torch
from coco_adaptation import core
from coco_baselines import metrics,training
from coco_baselines.storage import identity,serial
from coco_coordinated import run as memory

def model(ctx,snap):
    spec=snap['spec'];base=dict(ctx['base'],groups=spec['groups'],block_rank=spec['rank'],active_groups=spec['active'])
    m=core.new_model('vanilla',base,snap['weights'],ctx['device'])
    m.targets=copy.deepcopy(spec['targets']);m.total_atoms=m.n_groups*m.group_size
    m.detector_ids=torch.tensor([t['detector_group'] for t in m.targets],device=ctx['device'],dtype=torch.long)
    return m

def permute_snapshot(snap,outgoing,incoming):
    rank=snap['spec']['rank'];n=snap['spec']['groups']*rank
    if type(outgoing) is not int or type(incoming) is not int or not (0<=outgoing<n and 0<=incoming<n) or outgoing//rank==incoming//rank:
        raise ValueError('Exchange requires two valid slots in different groups')
    result=copy.deepcopy(snap);p=torch.arange(n)
    p[outgoing],p[incoming]=p[incoming].clone(),p[outgoing].clone()
    w=result['weights'];w['W_enc']=w['W_enc'][:,p].clone();w['W_dec']=w['W_dec'][p].clone();w['b_enc']=w['b_enc'][p].clone()
    previous=snap['spec'].get('atom_ids',list(range(n)))
    result['spec']['atom_ids']=[previous[j] for j in p.tolist()]
    return result

def empty_stats(C,A,rank):
    return dict(gain=np.zeros((C,A)),remove=np.zeros((C,rank)),cross=np.zeros((C,rank,A)),
        raw_fg=np.zeros((C,A)),raw_bg=np.zeros((C,A)),gated_fg=np.zeros((C,A)),
        signed_delta=np.zeros((C,A)),exclusion=np.zeros((C,A)),energy=np.zeros(C),counts=np.zeros(C,dtype=np.int64))

@torch.no_grad()
def shard_stats(m,pack,lo,hi,cfg):
    rank=m.group_size;C=len(m.targets);A=m.total_atoms;out=empty_stats(C,A,rank)
    lookup={t['category_index']:c for c,t in enumerate(m.targets)}
    D=m.W_dec.detach();norm=D.square().sum(1)
    for i in range(lo,hi):
        x=torch.as_tensor(np.array(pack['x'][i],copy=True),device=m.W_dec.device)
        raw=x@m.W_enc+m.b_enc;z=m.encode(x).reshape(len(x),A)
        ww,_,cc=core.object_weights([pack['masks'][i]],list(lookup),cfg)
        if not len(cc):continue
        wt=torch.as_tensor(ww,dtype=x.dtype,device=x.device)
        dx=wt@x;da=wt@raw;dz=wt@z
        for k,cat in enumerate(cc):
            c=lookup[int(cat)];g=m.targets[c]['contrast_group'];ids=list(range(rank*g,rank*g+rank))
            r=dx[k]-dz[k,ids]@D[ids];v=da[k];v0=dz[k,ids]
            out['gain'][c]+=(2*v*(D@r)-v.square()*norm).double().cpu().numpy()
            out['remove'][c]+=(-2*v0*(D[ids]@r)-v0.square()*norm[ids]).double().cpu().numpy()
            out['cross'][c]+=(2*v0[:,None]*v[None,:]*(D[ids]@D.T)).double().cpu().numpy()
            # Positive/negative normalized instance weights; no background semantics assumed.
            fw=wt[k].clamp_min(0);bw=(-wt[k]).clamp_min(0)
            fw=fw/fw.sum().clamp_min(1e-12);bw=bw/bw.sum().clamp_min(1e-12)
            out['raw_fg'][c]+=(fw@raw.abs()).double().cpu().numpy()
            out['raw_bg'][c]+=(bw@raw.abs()).double().cpu().numpy()
            out['gated_fg'][c]+=(fw@z.abs()).double().cpu().numpy()
            out['signed_delta'][c]+=v.double().cpu().numpy()
            excluded=(raw.abs()>1e-6)&(z.abs()<=1e-6)
            exclusion_rate=(fw@excluded.float())/(fw@(raw.abs()>1e-6).float()).clamp_min(1e-12)
            out['exclusion'][c]+=exclusion_rate.double().cpu().numpy()
            out['energy'][c]+=float(dx[k].square().sum());out['counts'][c]+=1
    return out

def rank_proposals(train,targets,cfg):
    """Exact two-vector exchange algebra under RAW/additive approximation.

Includes donor cost across all categories currently read from either group.
Actual changed TopK must subsequently be measured, not inferred from this score.
"""
    rank=cfg['block_rank'];groups=train['readout_groups'];A=train['gain'].shape[1];rows=[];seen=set()
    for c,t in enumerate(targets):
        receiver=int(groups[c]);outs=range(rank*receiver,rank*receiver+rank)
        order=np.argsort(-train['gain'][c],kind='stable')
        incoming=[int(j) for j in order if int(j)//rank!=receiver][:cfg['top_incoming']]
        for j in incoming:
            donor=j//rank
            for i in outs:
                target_gain=train['gain'][c,j]+train['remove'][c,i%rank]+train['cross'][c,i%rank,j]
                if target_gain<=0:continue
                pair=tuple(sorted((i,j)))
                changes=np.zeros(len(groups))
                for k,g in enumerate(groups):
                    if g==receiver:changes[k]=train['gain'][k,j]+train['remove'][k,i%rank]+train['cross'][k,i%rank,j]
                    elif g==donor:changes[k]=train['gain'][k,i]+train['remove'][k,j%rank]+train['cross'][k,j%rank,i]
                row=dict(category=t['category'],category_index=int(t['category_index']),
                    outgoing=i,incoming=j,receiver_group=receiver,donor_group=donor,
                    target_predicted_gain=float(target_gain),mean_predicted_gain=float(changes.mean()),
                    worst_predicted_category_gain=float(changes.min()),
                    excluded_fraction=float(train['exclusion'][c,j]),
                    affected_categories=[int(targets[k]['category_index']) for k in np.flatnonzero((groups==receiver)|(groups==donor))],
                    instances=int(train['counts'][c]))
                # Retain the target orientation with the strongest mean/target score for a duplicate pair.
                if pair in seen:
                    ix=next(k for k,r in enumerate(rows) if tuple(sorted((r['outgoing'],r['incoming'])))==pair)
                    if (row['mean_predicted_gain'],row['target_predicted_gain'])>(rows[ix]['mean_predicted_gain'],rows[ix]['target_predicted_gain']):rows[ix]=row
                else:seen.add(pair);rows.append(row)
    rows.sort(key=lambda r:(-r['mean_predicted_gain'],-r['target_predicted_gain'],r['outgoing'],r['incoming']))
    for j,r in enumerate(rows):r['id']=f'exchange{j:03d}'
    return rows

def candidate_pool(train,targets,cfg):
    ranked=rank_proposals(train,targets,cfg)
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

@torch.no_grad()
def dependency_shard(ctx,pack,lo,hi,atoms):
    """Remove one coefficient after the original gate; exact conditional effect.

    The gate is held fixed ONLY for this graph. Capture and AP use original
    readout groups. Actual exchanging and gate displacement are tested later.
    """
    m=model(ctx,ctx['initial']);m.eval();targets=m.targets;rank=m.group_size;C=len(targets);J=len(atoms)
    D=m.W_dec.detach();DD=D[atoms];dn=DD.square().sum(1)
    num=np.zeros((C,J));den=np.zeros(C);err=np.zeros(C);counts=np.zeros(C,int)
    rec=np.zeros(J);energy=0.;xsum=np.zeros(m.d);patches=0
    hs=[];zs=[];fs=[];cs=[];ims=[];annotations=[];gates=[]
    lookup={t['category_index']:k for k,t in enumerate(targets)};ag=np.array(atoms)//rank
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
            c=lookup[int(cat)];g=targets[c]['contrast_group'];ids=list(range(rank*g,rank*g+rank))
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
    rank=ctx['protocol']['block_rank'];s=ctx['store'];cached=s.load('dependency_graph')
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
            if atom//rank==targets[c]['detector_group']:
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
            if atom//rank in (t['contrast_group'],t['detector_group']):
                edges.append(dict(direction=atom,group=atom//rank,category=t['category'],category_index=cats[c],
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
        q.update(dependency_cost=cost,dependency_score=float(p['target_predicted_gain']/cfg['proposal_score_scale']-cost),
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
