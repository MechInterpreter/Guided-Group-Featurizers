"""TRAIN-frozen native-contribution consolidation, rank, atlas and interventions.

These are object-proxy and frozen linear-probe diagnostics, not a ground-truth
feature-splitting rate, causal object ontology, or generative intervention.
"""
import gc,hashlib,html,json,os,time
from pathlib import Path
import numpy as np
import torch
from coco_baselines import data,metrics,training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from scale_models import model

def objects(pack,cfg):
    out=[]
    for i,mm in enumerate(pack['masks']):
        known=mm['crowd']<=cfg['negative_fraction']
        for o in mm['instances']:
            if o['category'] not in cfg['category_indices']:continue
            fg=np.asarray(o['fraction'],float)*known;bg=(mm['fraction'][:,o['category']].astype(float)/255<=cfg['negative_fraction'])&known
            if fg.sum()>=cfg['min_object_patch_mass'] and bg.sum()>=cfg['min_background_patches']:
                out.append(dict(image_index=i,category=o['category'],annotation_id=o['annotation_id'],weights=(fg/fg.sum()-bg/bg.sum()).astype(np.float32)))
    return out

def rank_stats(cov):
    e=np.maximum(np.linalg.eigvalsh((cov+cov.T)/2),0);total=float(e.sum())
    if total<=1e-12:return dict(entropy_rank=None,participation_ratio=None,coefficient_matrix_stable_rank=None,covariance_stable_rank=None,spectrum=e.tolist(),zero_variance=True)
    p=e[e>0]/total
    return dict(entropy_rank=float(np.exp(-(p*np.log(p)).sum())),participation_ratio=float(total**2/(e@e)),
        coefficient_matrix_stable_rank=float(total/e[-1]),covariance_stable_rank=float((e@e)/e[-1]**2),spectrum=e[::-1].tolist(),zero_variance=False)

def selected_indices(pack,count=256):
    return sorted(range(len(pack['ids'])),key=lambda i:hashlib.sha256(('ggf-common-cohort-'+str(pack['ids'][i])).encode()).hexdigest())[:count]

def curve(target,contributions):
    den=np.square(target,dtype=np.float64).sum(1);prediction=np.zeros_like(target,dtype=np.float64);errors=[]
    for part in contributions:
        prediction+=part;errors.append(np.square(target-prediction).sum(1))
    return den,np.asarray(errors)

def load_model(ctx,row):
    path=ctx['local']/'frozen_weights'/(row['weights_sha']+'.pt')
    w=torch.load(path,map_location='cpu',weights_only=True)
    if training.weight_sha(w)!=row['weights_sha']:raise ValueError('Analysis model differs from frozen endpoint')
    m=model(row['method'],row['groups'],row['rank'],row['active_groups'],768,row['coefficient']).to(ctx['device']);m.load_state_dict(w);m.eval();return m

@torch.no_grad()
def train_plan(ctx,row,m):
    key='consolidation_plan_'+row['key'];s=ctx['store'];saved=s.load(key)
    fingerprint=identity(dict(weights=row['weights_sha'],train=ctx['packs']['train']['identity'],groups=[1,2,4],version=1))
    if saved is not None:
        if saved['fingerprint']!=fingerprint:raise ValueError('Analysis training plan changed')
        return saved
    p=ctx['packs']['train'];obs=objects(p,ctx['cfg']);G,b,d=m.n_groups,m.group_size,m.d
    folder=ctx['local']/'consolidation_mapping'/row['key'];folder.mkdir(parents=True,exist_ok=True)
    zpath=folder/'object_coefficients.npy';dz=np.lib.format.open_memmap(zpath,mode='w+',dtype=np.float32,shape=(len(obs),G,b));dx=np.empty((len(obs),d),np.float32)
    by={}
    for j,o in enumerate(obs):by.setdefault(o['image_index'],[]).append(j)
    active=np.zeros(G,np.float64);energy=np.zeros(G,np.float64);cohort=set(selected_indices(p));cohort_patches=0
    D=m.atoms().cpu().numpy();gram=np.einsum('grd,gsd->grs',D,D)
    for i in range(len(p['ids'])):
        x=np.array(p['x'][i],copy=True);z=m.encode(torch.as_tensor(x,device=ctx['device'])).cpu().numpy()
        for j in by.get(i,[]):dx[j]=obs[j]['weights']@x;dz[j]=np.einsum('p,pgr->gr',obs[j]['weights'],z)
        if i in cohort:
            active+=(np.linalg.norm(z,axis=-1)>1e-6).sum(0);energy+=np.einsum('pgr,grs,pgs->g',z,gram,z);cohort_patches+=len(z)
        if (i+1)%100==0:s.event('consolidation_TRAIN_mapping',model=row['kind'],seed=row['seed'],done=i+1,total=len(p['ids']))
    dz.flush();cats=np.asarray([o['category'] for o in obs]);plans=[]
    for c in ctx['cfg']['category_indices']:
        take=np.flatnonzero(cats==c)
        if not len(take):raise ValueError('Missing declared training concept in consolidation')
        out=metrics.contrast_mapping(dx[take],dz[take],D,4)
        if out is None:raise ValueError('Undefined training object contrast')
        g=out['groups'][0]
        # Matching uses TRAIN activation frequency and decoded contribution RMS only.
        freq=active/max(cohort_patches,1);rms=np.sqrt(energy/max(cohort_patches,1))
        distance=(np.log(freq+1e-5)-np.log(freq[g]+1e-5))**2+(np.log(rms+1e-5)-np.log(rms[g]+1e-5))**2
        pool=[int(q) for q in np.argsort(distance) if q!=g][:32]
        controls=np.random.default_rng(310000+row['seed']+c).choice(pool,min(4,len(pool)),replace=False).tolist()
        plans.append(dict(category_index=c,groups=out['groups'],training_curve=out['calibration_capture'],training_objects=len(take),
            random_control_groups=controls,target_training_activity=float(freq[g]),target_training_contribution_rms=float(rms[g]),
            control_training_activity=freq[controls].tolist(),control_training_contribution_rms=rms[controls].tolist()))
    saved=dict(fingerprint=fingerprint,models=row['weights_sha'],training_identity=p['identity'],plans=plans,
        cohort_ids=[p['ids'][i] for i in selected_indices(p)],group_activity=(active/max(cohort_patches,1)).tolist(),
        group_contribution_rms=np.sqrt(energy/max(cohort_patches,1)).tolist(),
        rule='Post-fit greedy TRAIN selection across all native groups; stop before non-positive marginal TRAIN gain; pad unchanged prediction for larger budgets. No decoder/code fitting; common cohort, each model owns active subset.')
    s.save(key,saved);s.write('consolidation_plans/'+row['key']+'.json',saved)
    del dz;gc.collect();zpath.unlink(missing_ok=True);return saved

@torch.no_grad()
def ranks(ctx,row,m,plan):
    key='rank_'+row['key'];saved=ctx['store'].load(key)
    if saved is not None:return saved
    G,b=m.n_groups,m.group_size;n=np.zeros(G,np.int64);summ=np.zeros((G,b),np.float64);cross=np.zeros((G,b,b),np.float64)
    for i in selected_indices(ctx['packs']['train']):
        z=m.encode(torch.as_tensor(np.array(ctx['packs']['train']['x'][i],copy=True),device=ctx['device'])).cpu().numpy().astype(float)
        active=np.linalg.norm(z,axis=-1)>1e-6;z=np.where(active[:,:,None],z,0.)
        n+=active.sum(0);summ+=z.sum(0);cross+=np.einsum('pgr,pgs->grs',z,z)
    D=m.atoms().cpu().numpy();records=[];reserved=len(ctx['cfg']['category_indices'])
    for g in range(G):
        r=dict(group=g,active_samples=int(n[g]),allocated_width=b,slot='initialized-category-slot' if g<reserved and row['kind']=='agf_g' else 'positional-category-slot' if g<reserved else 'reserve',
            state='inactive' if n[g]==0 else 'insufficient-samples' if n[g]<2 else 'estimated')
        if n[g]>=2:
            cov=(cross[g]-np.outer(summ[g],summ[g])/n[g])/(n[g]-1);cc=rank_stats(cov)
            e,V=np.linalg.eigh((cov+cov.T)/2);root=(V*np.sqrt(np.maximum(e,0)))@V.T
            contribution=rank_stats(root@(D[g]@D[g].T)@root)
            r.update(coefficient=cc,decoded_contribution=contribution,entropy_utilization=None if cc['entropy_rank'] is None else cc['entropy_rank']/b)
        records.append(r)
    out=dict(cohort_ids=plan['cohort_ids'],blocks=records,definitions=dict(entropy='exp(-sum p log p), p=lambda/sum lambda',participation='sum(lambda)^2/sum(lambda^2)',
        matrix_stable='sum(lambda)/max(lambda)',covariance_stable='sum(lambda^2)/max(lambda)^2',utilization='coefficient entropy effective rank / allocated width'),
        caveats='Active means nonzero coefficient norm after native selection; Group Lasso has soft gating. Zero-variance ranks undefined. Vanilla category slots are positional. Ranks do not establish intrinsic concept dimension or MDL efficiency.')
    s=ctx['store'];s.save(key,out);s.write('rank/'+row['key']+'.json',out);return out

def fit_probe(ctx):
    s=ctx['store'];old=s.load('TRAIN_linear_mask_probe')
    if old is not None:return old
    p=ctx['packs']['train'];d=p['x'].shape[-1];cats=ctx['cfg']['category_indices'];dev=ctx['device'];A=torch.zeros((d+1,d+1),device=dev,dtype=torch.float64);B=torch.zeros((d+1,len(cats)),device=dev,dtype=torch.float64);count=0
    for i in range(len(p['ids'])):
        pick=np.random.default_rng(ctx['cfg']['patch_sample_seed']+int(p['ids'][i])).choice(196,32,replace=False)
        take=pick[p['masks'][i]['crowd'][pick]<=ctx['cfg']['negative_fraction']]
        if not len(take):continue
        X=torch.as_tensor(np.array(p['x'][i,take],copy=True),device=dev,dtype=torch.float64);X=torch.cat([X,torch.ones((len(X),1),device=dev,dtype=torch.float64)],1)
        y=torch.as_tensor(p['masks'][i]['fraction'][take][:,cats]/255,device=dev,dtype=torch.float64)
        A+=X.T@X;B+=X.T@y;count+=len(X)
        if (i+1)%100==0:s.event('TRAIN_frozen_probe',done=i+1,total=len(p['ids']),patches=count)
    penalty=torch.eye(d+1,device=dev,dtype=torch.float64)*.001;penalty[-1,-1]=0
    W=torch.linalg.solve(A/count+penalty,B/count).float().cpu()
    out=dict(W=W,category_indices=cats,train_identity=p['identity'],samples=count,ridge=.001,
        objective='TRAIN-only ridge regression of patch mask fractions, fixed lambda .001; no featurizer/readout fitting on evaluation')
    s.save('TRAIN_linear_mask_probe',out);return out

def ap_columns(scores,fraction,crowd,cfg,cats):
    f=np.asarray(fraction,float)/255;out=[]
    for j,c in enumerate(cats):
        y=f[:,c]>=cfg['positive_fraction'];valid=((f[:,c]>=cfg['positive_fraction'])|(f[:,c]<=cfg['negative_fraction']))&(np.asarray(crowd)<=cfg['negative_fraction'])
        out.append(metrics.ap(scores[:,j],y,valid))
    if any(v is None for v in out):raise ValueError('Evaluation probe category denominator undefined')
    return np.asarray(out)

@torch.no_grad()
def heldout(ctx,row,m,pack,plan,probe,cohort):
    key='interpretability_'+cohort+'_'+row['key'];sig=identity(dict(weights=row['weights_sha'],pack=pack['identity'],plan=plan['fingerprint'],probe_train=probe['train_identity'],version=1));old=ctx['store'].load(key)
    if old is not None:
        if old['signature']!=sig:raise ValueError('Frozen interpretation protocol changed')
        return old
    cfg=ctx['cfg'];cats=cfg['category_indices'];C=len(cats);N=len(pack['ids']);obs=objects(pack,cfg);by={}
    for o in obs:by.setdefault(o['image_index'],[]).append(o)
    den=np.zeros((C,N));errors=np.zeros((C,4,N));allerror=np.zeros((C,N));cindex={c:j for j,c in enumerate(cats)}
    plans={p['category_index']:p for p in plan['plans']};D=m.atoms().cpu().numpy();Gids=sorted({g for p in plan['plans'] for g in p['groups']+p['random_control_groups']});W=probe['W'].numpy();weights=W[:-1];bias=W[-1]
    # Project decoder atoms into the frozen probe ONCE. Avoid constructing full
    # 768-D contributions for every category/random control on every image.
    projected=np.einsum('grd,dc->grc',D[Gids],weights,optimize=True);gindex={g:j for j,g in enumerate(Gids)}
    # Only 32 label-blind sampled patches/image enter AP and interventions.
    orig=[];recon=[];fraction=[];crowd=[];samegroup=[]
    target_scores={c:[] for c in cats};other_change={c:[] for c in cats};rand_scores={c:[] for c in cats};rand_other={c:[] for c in cats}
    for i in range(N):
        x=np.array(pack['x'][i],copy=True);tx=torch.as_tensor(x,device=ctx['device']);pred,z=m(tx);z=z.cpu().numpy();pred=pred.cpu().numpy()
        pick=np.random.default_rng(cfg['patch_sample_seed']+int(pack['ids'][i])).choice(196,min(32,cfg['detection_patches']),replace=False)
        original_scores=x[pick]@weights+bias;recon_scores=pred[pick]@weights+bias;orig.append(original_scores);recon.append(recon_scores)
        fraction.append(pack['masks'][i]['fraction'][pick]);crowd.append(pack['masks'][i]['crowd'][pick]);samegroup.append(np.stack([np.linalg.norm(z[pick,plans[c]['groups'][0]],axis=-1) for c in cats],1))
        projected_changes=np.einsum('pgr,grc->pgc',z[pick][:,Gids],projected,optimize=True)
        # Compare raw target-class probe response and off-target response magnitude.
        # These do not require a separate score matrix per removed group.
        for j,c in enumerate(cats):
            p=plans[c];g=p['groups'][0];change=projected_changes[:,gindex[g]]
            target_scores[c].append(original_scores[:,j]-change[:,j]);other_change[c].append(np.mean(np.abs(np.delete(change,j,axis=1)),axis=1))
            changes=projected_changes[:,[gindex[q] for q in p['random_control_groups']]].transpose(1,0,2)
            rand_scores[c].append(original_scores[:,j][None,:]-changes[:,:,j]);rand_other[c].append(np.mean(np.abs(np.delete(changes,j,axis=2)),axis=(0,2)))
        for o in by.get(i,[]):
            j=cindex[o['category']];w=o['weights'];target=w@x;dz=np.einsum('p,pgr->gr',w,z);p=plans[o['category']];parts=[dz[g]@D[g] for g in p['groups']]
            dd,ee=curve(target[None],[(v[None]) for v in parts]);den[j,i]+=dd[0]
            for k in range(4):errors[j,k,i]+=ee[min(k,len(ee)-1),0]
            allerror[j,i]+=np.square(target-w@pred,dtype=np.float64).sum()
        if (i+1)%100==0:ctx['store'].event('consolidation_and_interventions',cohort=cohort,model=row['kind'],seed=row['seed'],done=i+1,total=N)
    frac=np.concatenate(fraction);cr=np.concatenate(crowd);raw=np.concatenate(orig);rec=np.concatenate(recon);heat=np.concatenate(samegroup)
    base_ap=ap_columns(raw,frac,cr,cfg,cats);rec_ap=ap_columns(rec,frac,cr,cfg,cats);rows=[];rng=np.random.default_rng(991000+row['seed'])
    bootstrap=np.zeros((1000,4));gaps=[]
    for j,c in enumerate(cats):
        total=den[j].sum()
        if total<=1e-12:raise ValueError('Declared category has no held-out contrast')
        captures=1-errors[j].sum(1)/total;allcap=1-allerror[j].sum()/total
        y=frac[:,c]/255>=cfg['positive_fraction'];valid=((frac[:,c]/255>=cfg['positive_fraction'])|(frac[:,c]/255<=cfg['negative_fraction']))&(cr<=cfg['negative_fraction'])
        after=metrics.ap(np.concatenate(target_scores[c]),y,valid)
        random_after=[metrics.ap(v,y,valid) for v in np.concatenate(rand_scores[c],axis=1)]
        if after is None or any(v is None for v in random_after):raise ValueError('Undefined intervention AP')
        rows.append(dict(category_index=c,groups=plans[c]['groups'],capture_1_2_4=[float(captures[k-1]) for k in [1,2,4]],
            all_group_capture=float(allcap),extra_group_gain_4_minus_1=float(captures[3]-captures[0]),same_group_AP=metrics.ap(heat[:,j],y,valid),
            probe_original_AP=float(base_ap[j]),probe_reconstruction_AP=float(rec_ap[j]),target_group_removed_AP=float(after),target_AP_drop=float(base_ap[j]-after),
            random_removed_AP=list(map(float,random_after)),random_mean_AP_drop=float(base_ap[j]-np.mean(random_after)),
            off_target_absolute_logit_change=float(np.concatenate(other_change[c]).mean()),random_off_target_absolute_logit_change=float(np.concatenate(rand_other[c]).mean()),
            usable_objects=int(sum(o['category']==c for o in obs)),usable_images=int((den[j]>0).sum())))
    # Whole-image bootstrap reuses identical image draws across every category and prefix.
    for b in range(1000):
        count=np.bincount(rng.integers(N,size=N),minlength=N);dd=den@count
        if np.any(dd<=1e-12):bootstrap[b]=np.nan;continue
        bootstrap[b]=np.mean(1-np.einsum('cki,i->ck',errors,count)/dd[:,None],axis=0)
    result=dict(signature=sig,seed=row['seed'],kind=row['kind'],weights_sha=row['weights_sha'],categories=rows,
        macro_capture_1_2_4=np.mean([r['capture_1_2_4'] for r in rows],axis=0).tolist(),
        macro_capture_image_bootstrap95=np.nanquantile(bootstrap[:,[0,1,3]],[.025,.975],axis=0).tolist(),
        macro_target_AP_drop=float(np.mean([r['target_AP_drop'] for r in rows])),macro_random_AP_drop=float(np.mean([r['random_mean_AP_drop'] for r in rows])),
        probe='Frozen TRAIN-only linear ridge mask predictor, applied to original x and residual-preserving x-native_group_contribution. Removal diagnoses a fitted readout; not causal ground-truth concept manipulation.',
        consolidation='Greedy native-code subsets chosen on TRAIN only; held-out sum of native group contributions. Extra-group gain is not a feature-splitting rate; overlap/cancellation and context remain possible.',
        controls='Four random groups sampled from 32 nearest TRAIN activity/contribution-RMS neighbors; matching distances/statistics retained, not exact energy matching.',
        cohort=cohort,denominator_categories=cats,bootstrap='1000 paired whole-image draws within model; seed SD/t intervals separate')
    s=ctx['store'];s.save(key,result);s.write('interpretability/'+key+'.json',result)
    s.save('consolidation_compact_'+cohort+'_'+row['key'],dict(den=den,errors=errors,image_ids=pack['ids'],category_indices=cats))
    return result

def image_path(ctx,pack,index,coco):
    image_id=pack['ids'][index]
    if 'voc_assets' in ctx:
        r=ctx['voc_assets']['records'][image_id];return ctx['voc_assets']['root']/'JPEGImages'/(r['image']+'.jpg'),r['geometry']
    info=coco.imgs[image_id];path=data.download(data.COCO_BUCKET+'train2017/'+info['file_name'],ctx['local']/'images'/info['file_name'])
    return path,data.geometry(info['width'],info['height'])

@torch.no_grad()
def atlas(ctx,row,m,pack,plan,coco,cohort):
    if row['seed']!=21:return
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ctx['store'].results/'atlas';folder.mkdir(exist_ok=True)
    wanted=['person','potted plant','vase','fork','spoon','pizza'];names={i:c['name'] for i,c in enumerate(ctx['provenance']['categories'])}
    pp=[p for p in plan['plans'] if names[p['category_index']] in wanted];cohort_idx=selected_indices(ctx['packs']['train'],128);D=m.atoms().cpu().numpy();links=[]
    for p in pp:
        c=p['category_index'];g=p['groups'][0];stem=cohort+'_'+row['kind']+'_'+str(c);png=folder/(stem+'.png')
        if png.exists():links.append(png.name);continue
        candidates=[i for i,mm in enumerate(pack['masks']) if np.max(mm['fraction'][:,c]/255)>=ctx['cfg']['positive_fraction']]
        if not candidates:continue
        i=min(candidates,key=lambda i:hashlib.sha256(('fixed-example-'+str(pack['ids'][i])).encode()).hexdigest())
        samples=[]
        for ti in cohort_idx:
            z=m.encode(torch.as_tensor(np.array(ctx['packs']['train']['x'][ti],copy=True),device=ctx['device'])).cpu().numpy()[:,g]
            samples.append(z[np.linalg.norm(z,axis=-1)>1e-6])
        samples=np.concatenate(samples)
        if len(samples)<3:continue
        # PCA of DECODED contributions, not arbitrary raw coefficient coordinates.
        mean=samples.mean(0);centered=samples-mean;cov=centered.T@centered/max(len(samples)-1,1)
        ev,U=np.linalg.eigh((cov+cov.T)/2);root=(U*np.sqrt(np.maximum(ev,0)))@U.T
        _,_,V=np.linalg.svd(root@D[g],full_matrices=False);V=V[:3]
        for v in V:
            if v[np.argmax(np.abs(v))]<0:v*=-1
        train_pc=(samples-mean)@(D[g]@V.T);lim=np.maximum(np.quantile(np.abs(train_pc),.99,axis=0),1e-6)
        coefflim=np.maximum(np.quantile(np.abs(samples),.99,axis=0),1e-6);normlim=max(float(np.quantile(np.linalg.norm(samples,axis=-1),.99)),1e-6)
        x=np.array(pack['x'][i],copy=True);z=m.encode(torch.as_tensor(x,device=ctx['device'])).cpu().numpy()[:,g];norm=np.linalg.norm(z,axis=-1);on=norm>1e-6
        path,geom=image_path(ctx,pack,i,coco)
        with Image.open(path) as im:crop=np.asarray(data.transform_image(im.convert('RGB'),geom))/255
        fig,axes=plt.subplots(3,5,figsize=(11,6.8),layout='constrained');axes=axes.ravel()
        axes[0].imshow(crop);axes[0].set_title('Original')
        def overlay(ax,field,title,cmap,vmin,vmax,alpha):
            ax.imshow(crop);masked=np.ma.array(field.reshape(14,14),mask=(~on).reshape(14,14));ax.imshow(masked,cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha,extent=(-.5,223.5,223.5,-.5),interpolation='nearest');ax.set_title(title)
        overlay(axes[1],norm,'Group presence','winter',0,normlim,.65)
        pc=(z-mean)@(D[g]@V.T);rgb=np.clip(.5+.5*pc/lim,0,1);rgb[~on]=0
        axes[2].imshow(crop);axes[2].imshow(rgb.reshape(14,14,3),alpha=(on*.65).reshape(14,14),extent=(-.5,223.5,223.5,-.5),interpolation='nearest');axes[2].set_title('Contribution PCA → RGB')
        for j in range(min(12,z.shape[1])):overlay(axes[3+j],z[:,j],f'Signed direction {j+1}','coolwarm',-coefflim[j],coefflim[j],.65)
        for ax in axes:ax.axis('off')
        fig.suptitle(f'{names[c]} · {row["kind"]} · group {g} · fixed example {pack["ids"][i]}',fontsize=12)
        fig.savefig(png,dpi=170);fig.savefig(folder/(stem+'.pdf'));plt.close(fig);links.append(png.name)
        record=dict(category=c,image_id=pack['ids'][i],group=g,weights_sha=row['weights_sha'],cohort_ids=[ctx['packs']['train']['ids'][k] for k in cohort_idx],
            coefficient_display_limits=coefflim.tolist(),contribution_PCA_basis=V.tolist(),PCA_display_limits=lim.tolist(),group_norm_limit=normlim,
            selection='First SHA-ranked annotation-positive evaluation image; no performance filtering',
            colors='Signed coefficient coordinates are basis-dependent; contribution PCA fitted on TRAIN active patches; RGB encodes PCs, not photographic color. Display normalization is per model/group; intensity not comparable absolutely.',
            interpolation='Nearest-patch primary overlay, inactive patches transparent; no smoothing changes metrics')
        (folder/(stem+'.json')).write_text(json.dumps(record,indent=2))
        # A genuine learned contribution cloud with a shared TRAIN PCA basis.
        fig=plt.figure(figsize=(4,3.6));ax=fig.add_subplot(projection='3d');take=np.random.default_rng(21+c).choice(len(train_pc),min(1800,len(train_pc)),replace=False)
        ax.scatter(*train_pc[take].T,c=np.clip(.5+.5*train_pc[take]/lim,0,1),s=2,alpha=.6,rasterized=True);ax.set(xlabel='PC1',ylabel='PC2',zlabel='PC3',title=names[c]+' · decoded contribution PCA');fig.tight_layout();fig.savefig(folder/(stem+'_cloud.png'),dpi=170);plt.close(fig)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GGF direction atlas</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}img{width:100%}</style><h1>Direction atlas</h1><p>Fixed annotation-positive examples, shared images across methods. TRAIN-only display scales and contribution PCA. Raw coefficient directions depend on basis. See adjacent JSON for provenance.</p>'+''.join('<figure><img src="'+html.escape(p.name)+'"><figcaption>'+html.escape(p.stem)+'</figcaption></figure>' for p in sorted(folder.glob('*.png')) if not p.stem.endswith('_cloud')))

def run(ctx,rows,pack,cohort,coco):
    probe=fit_probe(ctx)
    for row in rows:
        m=load_model(ctx,row)
        try:
            plan=train_plan(ctx,row,m);ranks(ctx,row,m,plan);heldout(ctx,row,m,pack,plan,probe,cohort)
            if coco is not None or 'voc_assets' in ctx:atlas(ctx,row,m,pack,plan,coco,cohort)
        finally:del m;gc.collect();torch.cuda.empty_cache()
