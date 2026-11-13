"""Read-only checkpoint reuse. Evaluation probes are separate from training heads."""
import argparse, gc, hashlib, json, os, shutil, time, zipfile, platform
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cv_math import *

NAMES=['person','car','chair','potted plant','vase','dog']
PER_CATEGORY=32
METHODS=['agf_g','released_vanilla','supervised_vanilla']
LABELS=dict(agf_g='GGF',released_vanilla='Vanilla BSF',supervised_vanilla='Supervised Vanilla')
SEEDS=[21,22,23]

def log(stage,**kw):print(stage,json.dumps(kw,default=str),flush=True)

class ReadThrough:
    def __init__(self,job,source):self.job=job;self.source=source
    def __getattr__(self,k):return getattr(self.job,k)
    def load(self,k):
        v=self.job.load(k)
        if v is None and (k in ['primary_complete','primary_frozen','TRAIN_linear_mask_probe','cached_coco_test_masks'] or k.startswith('consolidation_plan_')):v=self.source.load(k)
        return v

def discover_primary(root,local,ctx):
    from coco_baselines.storage import identity
    from coco_ggf import matched_baselines as matched
    import experiment as exp
    hits=[]
    for path in (Path(root)/'results').glob('*/manifest.json'):
        m=json.loads(path.read_text());c=m['config']
        if c.get('study')=='ggf_checklist_evidence_v1' and c.get('source_run')==exp.SOURCE_RUN and c.get('versions')==matched.versions() and c.get('normalization_sha')==identity(ctx['norm']) and c.get('training_identity')==ctx['packs']['train']['identity']:
            st=matched.discover(root,local,run=m['run_id']);rec=st.load('primary_complete')
            if rec and rec.get('complete'):hits.append((st,rec))
    if len(hits)!=1:raise ValueError('Need exactly one completed compatible COCO primary comparison; no fitting fallback')
    source,receipt=hits[0];rows=receipt['models']
    if len(rows)!=9 or {(r['seed'],r['kind']) for r in rows}!={(s,k) for s in SEEDS for k in METHODS}:raise ValueError('Incomplete source catalog')
    for r in rows:exp.validate_row(r)
    return source,sorted(rows,key=lambda r:(r['seed'],METHODS.index(r['kind'])))

def select_cohort(pack,ctx,split):
    import analysis
    names={c['name']:i for i,c in enumerate(ctx['provenance']['categories'])}
    available=analysis.objects(pack,ctx['cfg']);out=[];used=set()
    # One instance per base image; prevent image leakage and bootstrap inflation.
    for name in NAMES:
        if name not in names:raise ValueError('Category unavailable: '+name)
        candidates=[o for o in available if o['category']==names[name]]
        candidates.sort(key=lambda o:digest([split,int(pack['ids'][o['image_index']]),int(o['annotation_id'])]))
        chosen=[]
        for o in candidates:
            image_id=int(pack['ids'][o['image_index']])
            if image_id in used:continue
            used.add(image_id);chosen.append(dict(split=split,image_id=image_id,annotation_id=int(o['annotation_id']),category=int(o['category']),category_name=name))
            if len(chosen)==PER_CATEGORY:break
        if len(chosen)<12:raise ValueError('Too few predeclared eligible images for '+split+'/'+name)
        out+=chosen
    return out

def source_crop(ctx,coco,rec):
    from coco_baselines import data
    info=coco.imgs[rec['image_id']];p=data.download(data.COCO_BUCKET+'train2017/'+info['file_name'],ctx['local']/'images'/info['file_name'])
    geom=data.geometry(info['width'],info['height'])
    with Image.open(p) as im:rgb=np.asarray(data.transform_image(im.convert('RGB'),geom))
    ann=coco.anns[rec['annotation_id']]
    if ann['image_id']!=rec['image_id'] or ann.get('iscrowd',0):raise ValueError('Invalid declared instance')
    mask=np.asarray(data.transform_image(Image.fromarray(coco.annToMask(ann)),geom,mask=True))
    return rgb,mask

def geometry_preflight(ctx,coco,records):
    from coco_baselines import data
    accepted=[];rejected=[]
    for rec in records:
        im=coco.imgs[rec['image_id']];mask=coco.annToMask(coco.anns[rec['annotation_id']])
        mask=np.asarray(data.transform_image(Image.fromarray(mask),data.geometry(im['width'],im['height']),mask=True))
        fractions=np.stack([render(np.zeros((224,224,3),np.uint8),mask,a,v)[1] for a,v in VARIANTS])
        mass=float(fractions.sum(1).min());bg=int((fractions<=.1).sum(1).min())
        (accepted if mass>=1 and bg>=4 else rejected).append(rec)
    for split in ['train','eval']:
        for name in NAMES:
            if sum(r['split']==split and r['category_name']==name for r in accepted)<12:raise ValueError('Annotation-only transformed support is insufficient: '+split+'/'+name)
    ctx['store'].write('annotation_only_geometry_exclusions.json',rejected)
    log('GEOMETRY PRECHECK',accepted=len(accepted),rejected=len(rejected),rule='at least one full-patch foreground mass and four comparison patches at EVERY declared level; no model scores')
    return accepted

def record_key(rec):return rec['split']+'_'+str(rec['image_id'])+'_'+str(rec['annotation_id'])

def extract(ctx,coco,records,cache,asset):
    from transformers import AutoImageProcessor,AutoModel
    from coco_stratified import dataset
    from coco_baselines.storage import identity
    processor=AutoImageProcessor.from_pretrained(str(asset));m=AutoModel.from_pretrained(str(asset),dtype=torch.float32).eval().to(ctx['device'])
    dataset.verify_backbone_state(m.state_dict(),ctx['provenance']['backbone_weights']['tensor_sha'])
    started=time.monotonic();fresh=0;inference_seconds=0.
    for i,rec in enumerate(records):
        key=record_key(rec)
        if cache.get(key) is None:
            rgb,mask=source_crop(ctx,coco,rec);views=[];fracs=[]
            for a,v in VARIANTS:
                im,f=render(rgb,mask,a,v);views.append(im);fracs.append(f)
            fracs=np.asarray(fracs,dtype=np.float32)
            if np.any(fracs.sum(1)<1) or np.any((fracs<=.1).sum(1)<4):raise ValueError('Transformation removes usable support: '+key+'; change only a versioned annotation-only protocol, never silently skip')
            xs=[];torch.cuda.synchronize();t=time.monotonic()
            with torch.inference_mode():
                for lo in range(0,len(views),8):
                    x=processor(images=views[lo:lo+8],do_resize=False,do_center_crop=False,return_tensors='pt').to(ctx['device'])
                    raw=m(**x).last_hidden_state[:,5:].float().cpu().numpy()
                    if raw.shape[1:]!=(196,768):raise ValueError('Unexpected DINO patch layout')
                    xs.append((raw-np.asarray(ctx['norm']['mean']))*ctx['norm']['scale'])
            torch.cuda.synchronize();inference_seconds+=time.monotonic()-t;fresh+=1
            cache.put(key,x=np.concatenate(xs).astype(np.float32),fraction=fracs)
        if (i+1)%5==0:
            elapsed=time.monotonic()-started;log('FEATURES',objects=i+1,total=len(records),new_objects=fresh,seconds=round(elapsed,1),ETA_minutes=round((len(records)-i-1)*elapsed/(i+1)/60,1))
        if fresh and fresh%50==0:log('MEASURED DINO PILOT',new_objects=fresh,forward_seconds_per_view=inference_seconds/(fresh*len(VARIANTS)),scope='Extraction only; remaining model analysis and summaries are separate')
    ctx['store'].write('feature_timing.json',dict(objects=len(records),new_objects=fresh,views_per_object=len(VARIANTS),wall_seconds=time.monotonic()-started,forward_seconds=inference_seconds,GPU=torch.cuda.get_device_name()))
    del m,processor;gc.collect();torch.cuda.empty_cache()

def get_plan(ctx,row,m,source):
    import analysis
    plan=analysis.train_plan(ctx,row,m)
    plans={p['category_index']:dict(p) for p in plan['plans']}
    freq=np.asarray(plan['group_activity']);rms=np.asarray(plan['group_contribution_rms'])
    for c,p in plans.items():
        g=p['groups'][0]
        dist=(np.log(freq+1e-5)-np.log(freq[g]+1e-5))**2+(np.log(rms+1e-5)-np.log(rms[g]+1e-5))**2
        pool=[int(q) for q in np.argsort(dist) if q not in p['groups']][:32]
        p['random_control_groups']=np.random.default_rng(310000+row['seed']+c).choice(pool,4,replace=False).tolist()
    ctx['store'].write('controlled_group_plan_'+row['kind']+'_'+str(row['seed'])+'.json',dict(plans=list(plans.values()),control_rule='TRAIN activity/decoded-RMS nearest32; sample4 excluding every selected target group'))
    for c in [r['category'] for r in ctx['cohort']]:
        if not plans[c]['groups']:raise ValueError('TRAIN mapping has no positive-gain group; retain failure, do not select on evaluation')
    return plans

def collect(ctx,row,m,records,features,result_cache,plans):
    D=m.atoms().detach().cpu().numpy().astype(np.float64);summaries=[]
    stats={}
    for c in {r['category'] for r in records}:
        for g in plans[c]['groups']:
            stats[(c,g)]=[0,np.zeros(D.shape[1]),np.zeros((D.shape[1],D.shape[1]))]
    start=time.monotonic()
    for i,rec in enumerate(records):
        key=record_key(rec);saved=result_cache.get(key)
        if saved is None:
            f=features.get(key);x=f['x'];frac=f['fraction'];p=plans[rec['category']]
            gids=list(dict.fromkeys(p['groups']+p['random_control_groups']+[plans[c]['groups'][0] for c in sorted({r['category'] for r in records}) if plans[c]['groups']]));inds=[gids.index(g) for g in p['groups']]
            norm=[];pooled=[];dc=[];den=[];err=[];ap=[];ns=[];ss=[];cc=[];normrms=[]
            from coco_baselines.metrics import ap as average_precision
            with torch.inference_mode():
                # Bound encoder activations to one transformed image at a time.
                for j in range(len(x)):
                    z=m.encode(torch.as_tensor(x[j].copy(),device=ctx['device'])).detach().cpu().numpy()[:,gids].astype(np.float64)
                    fg=frac[j]/frac[j].sum();bg=(frac[j]<=.1).astype(float);bg/=bg.sum()
                    dz=np.einsum('p,pgr->gr',fg,z);pooled.append(dz)
                    dx=(fg-bg)@x[j];dzc=np.einsum('p,pgr->gr',fg-bg,z)
                    native=np.einsum('gr,grd->gd',dzc,D[gids]);parts=np.cumsum(native[inds],axis=0)
                    den.append(float(dx@dx));err.append(np.sum((parts-dx)**2,axis=1))
                    selected=z[:,inds];active=np.linalg.norm(selected,axis=-1)>1e-6
                    ns.append(active.sum(0));ss.append((selected*active[:,:,None]).sum(0));cc.append(np.einsum('pgr,pgs->grs',selected*active[:,:,None],selected))
                    norm.append(np.einsum('p,pg->g',fg,np.linalg.norm(z,axis=-1)))
                    normrms.append(np.sqrt(np.einsum('p,pg->g',fg,np.linalg.norm(z,axis=-1)**2)))
                    ap.append(average_precision(np.linalg.norm(z[:,inds[0]],axis=-1),frac[j]>=.5,(frac[j]>=.5)|(frac[j]<=.1)))
            saved=result_cache.put(key,pooled=np.asarray(pooled,dtype=np.float32),raw=np.einsum('vp,vpd->vd',frac/frac.sum(1,keepdims=True),x),
                norm=np.asarray(norm,dtype=np.float32),norm_rms=np.asarray(normrms,dtype=np.float32),den=np.asarray(den),err=np.asarray(err),
                ap=np.asarray([np.nan if v is None else v for v in ap]),gids=np.asarray(gids),selected=np.asarray(inds),
                n=np.sum(ns,0),s=np.sum(ss,0),cross=np.sum(cc,0))
        if rec['split']=='train':
            for j,g in enumerate(plans[rec['category']]['groups']):
                v=stats[(rec['category'],g)];v[0]+=saved['n'][j];v[1]+=saved['s'][j];v[2]+=saved['cross'][j]
        summaries.append((rec,saved))
        if (i+1)%10==0:log('ENCODE',method=row['kind'],seed=row['seed'],objects=i+1,total=len(records),elapsed_minutes=round((time.monotonic()-start)/60,2))
    pcas={}
    for (c,g),(n,s,cross) in stats.items():pcas[(c,g)]=pca_from_stats(n,s,cross,D[g])
    return summaries,pcas,D

def readout_features(s,p,pcas,D,category,k):
    selected=s['selected'][:k];out=[]
    for j in selected:
        g=int(s['gids'][j]);pc=pcas[(category,g)]
        out+=[project(s['pooled'][:,j],D[g],pc),s['norm'][:,j,None]]
    return np.concatenate(out,axis=1)

def evaluate_model(ctx,row,summaries,pcas,D,plans):
    from scipy.stats import spearmanr
    metrics=[];readouts=[];interventions=[];trajectories=[];pca_rows=[]
    maskW=ctx['mask_probe']['W']
    if torch.is_tensor(maskW):maskW=maskW.cpu().numpy()
    category_columns=ctx['mask_probe']['category_indices']
    for (c,g),pc in pcas.items():
        pca_rows.append(dict(model=row['kind'],seed=row['seed'],category=c,group=g,valid=pc['valid'],active_TRAIN_patches=pc.get('samples',0),three_PC_variance_fraction=pc.get('variance_fraction'),eigenvalues=pc['variance'].tolist()))
    for name in NAMES:
        cohort=[(r,s) for r,s in summaries if r['category_name']==name];c=cohort[0][0]['category'];plan=plans[c]
        for a,levels in LEVELS.items():
            ix=[i for i,(att,v) in enumerate(VARIANTS) if att==a];Y=np.asarray([target(a,v) for v in levels])
            train=[s for r,s in cohort if r['split']=='train'];test=[(r,s) for r,s in cohort if r['split']=='eval']
            # Shared raw-DINO probe is independently fitted on identical TRAIN arrays.
            # It stays fixed during every group edit and across all methods/seeds.
            rawprobe=fit_ridge(np.concatenate([s['raw'][ix] for s in train]),np.tile(Y,len(train)))
            raw_y=np.tile(Y,len(test));raw_pred=predict(rawprobe,np.concatenate([s['raw'][ix] for r,s in test]));raw_r2=r2(raw_y,raw_pred)
            for k in [1,2,4]:
                actual=min(k,len(plan['groups']))
                X=np.concatenate([readout_features(s,plan,pcas,D,c,actual)[ix] for s in train]);yy=np.tile(Y,len(train))
                probe=fit_ridge(X,yy)
                xt=np.concatenate([readout_features(s,plan,pcas,D,c,actual)[ix] for r,s in test]);yt=np.tile(Y,len(test));pr=predict(probe,xt)
                rho=spearmanr(yt,pr).statistic
                readouts.append(dict(model=row['kind'],seed=row['seed'],category=name,attribute=a,group_budget=k,actual_groups=actual,readout_dimensions=xt.shape[1],heldout_R2=r2(yt,pr),raw_DINO_R2=raw_r2,spearman=None if not np.isfinite(rho) else float(rho),evaluation_images=len(test)))
                # Counterfactual donor is the SAME held-out object at a prespecified
                # transformation level. This is not probe optimization or generation.
                for r,s in test:
                    source=ix[2];groupids=s['gids'];selected=s['selected'][:actual]
                    contributions=np.einsum('vgr,grd->vgd',s['pooled'],D[groupids]);base=s['raw'][source]
                    original=float(predict(rawprobe,base[None])[0]);comp=contributions[:,selected].sum(1)
                    random_idx=[j for j,g in enumerate(groupids) if j not in s['selected']][:actual]
                    random=contributions[:,random_idx].sum(1)
                    for j in ix:
                        delta=comp[j]-comp[source];rd=random[j]-random[source]
                        matched=energy_match(rd,delta);edited=transplant(base,comp[source],comp[j])
                        oracle=float(predict(rawprobe,s['raw'][j,None])[0])-original
                        shift=float(predict(rawprobe,edited[None])[0])-original
                        random_shift=None if matched is None else float(predict(rawprobe,(base+matched)[None])[0])-original
                        logits=delta@maskW[:-1];random_logits=None if matched is None else matched@maskW[:-1];ci=category_columns.index(c)
                        interventions.append(dict(model=row['kind'],seed=row['seed'],category=name,image_id=r['image_id'],attribute=a,level=VARIANTS[j][1],group_budget=k,actual_groups=actual,
                            known_change=float(Y[ix.index(j)]-Y[2]),oracle_probe_change=oracle,group_probe_change=shift,random_probe_change=random_shift,
                            edit_norm=float(np.linalg.norm(delta)),target_category_probe_change=float(logits[ci]),off_target_absolute_logit_change=float(np.mean(np.abs(np.delete(logits,ci)))),random_off_target_absolute_logit_change=None if random_logits is None else float(np.mean(np.abs(np.delete(random_logits,ci)))),residual_preserved=True,control_valid=matched is not None))
                    pred=predict(probe,readout_features(s,plan,pcas,D,c,actual)[ix])
                    for j,v,q in zip(ix,levels,pred):trajectories.append(dict(model=row['kind'],seed=row['seed'],category=name,image_id=r['image_id'],attribute=a,level=v,group_budget=k,prediction=float(q),truth=float(target(a,v))))
            for r,s in test:
                for j in ix:
                    for k in [1,2,4]:
                        index=min(k,len(plan['groups']))-1
                        metrics.append(dict(model=row['kind'],seed=row['seed'],category=name,image_id=r['image_id'],attribute=a,level=VARIANTS[j][1],group_budget=k,den=float(s['den'][j]),error=float(s['err'][j,index]),patch_AP=None if not np.isfinite(s['ap'][j]) else float(s['ap'][j]),group_norm=float(s['norm'][j,0])))
    return dict(metrics=metrics,readouts=readouts,interventions=interventions,trajectories=trajectories,pca=pca_rows)

def atlas(ctx,row,m,summaries,pcas,D,plans,cache,coco):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    for name in NAMES:
        choices=[(r,s) for r,s in summaries if r['split']=='eval' and r['category_name']==name]
        rec,s=min(choices,key=lambda item:digest(['fixed-display',item[0]['image_id']]))
        c=rec['category'];g=plans[c]['groups'][0];pc=pcas[(c,g)]
        rgb,mask=source_crop(ctx,coco,rec);f=cache.get(record_key(rec));fig,axes=plt.subplots(3,5,figsize=(9,5.4))
        # TRAIN-only norm limit, shared across every displayed transformation.
        train_norm=np.concatenate([v['norm_rms'][:,0] for r,v in summaries if r['split']=='train' and r['category']==c])
        limit=max(float(np.quantile(train_norm,.99)),1e-6)
        for j,level in enumerate(LEVELS['rotation']):
            v=VARIANTS.index(('rotation',level));image,fr=render(rgb,mask,'rotation',level)
            with torch.inference_mode():z=m.encode(torch.as_tensor(f['x'][v].copy(),device=ctx['device'])).cpu().numpy()[:,g]
            norm=np.linalg.norm(z,axis=-1);active=norm>1e-6
            color=np.clip(.5+.5*project(z,D[g],pc)/pc['limits'],0,1).reshape(14,14,3)
            alpha=zoom(np.clip(norm/limit,0,.85).reshape(14,14),16,order=1)
            base=np.asarray(image)/255;teal=np.array([0.,.58,.53]);overlay=base*(1-alpha[:,:,None])+teal*alpha[:,:,None]
            rgbmap=zoom(color,(16,16,1),order=1);rgbalpha=zoom(active.reshape(14,14).astype(float),16,order=0)*.8
            axes[0,j].imshow(image);axes[0,j].set_title(str(level)+' degrees',fontsize=9)
            axes[1,j].imshow(overlay);axes[2,j].imshow(base*(1-rgbalpha[:,:,None])+rgbmap*rgbalpha[:,:,None])
        for ax in axes.flat:ax.axis('off')
        for ax,label in zip(axes[:,0],['Input','Norm','Contribution PCA RGB']):ax.text(-.08,.5,label,transform=ax.transAxes,rotation=90,va='center',fontsize=8)
        fig.suptitle(name+' | '+LABELS[row['kind']]+' | seed '+str(row['seed'])+' | group '+str(g),fontsize=10)
        fig.tight_layout();out=ctx['out']/'figures';out.mkdir(exist_ok=True)
        for ext in ['pdf','png']:fig.savefig(out/(row['kind']+'_'+str(row['seed'])+'_'+name.replace(' ','_')+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)

def finish(ctx,all_results,records,rows,protocol):
    import pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import t as student
    out=ctx['out'];combined={k:[r for result in all_results for r in result[k]] for k in ['metrics','readouts','interventions','trajectories','pca']}
    for k,v in combined.items():pd.DataFrame(v).to_csv(out/(k+'.csv'),index=False)
    read=pd.DataFrame(combined['readouts']);group=['model','attribute','group_budget']
    # Equal category weight within seed, then mean/SD across training starts.
    per_seed=read.groupby(group+['seed']).heldout_R2.mean().reset_index();summary=per_seed.groupby(group).heldout_R2.agg(['mean','std','count']).reset_index()
    summary['t95_halfwidth']=summary.apply(lambda r:float(student.ppf(.975,r['count']-1)*r['std']/np.sqrt(r['count'])) if r['count']>1 else np.nan,axis=1)
    summary.to_csv(out/'property_R2_seed_summary.csv',index=False)
    metric=pd.DataFrame(combined['metrics']);caps=metric.groupby(['model','seed','category','attribute','group_budget'])[['den','error']].sum().reset_index();caps['capture']=1-caps.error/caps.den
    caps.to_csv(out/'capture_by_category.csv',index=False)
    capseed=caps.groupby(['model','seed','attribute','group_budget']).capture.mean().reset_index()
    capsummary=capseed.groupby(['model','attribute','group_budget']).capture.agg(['mean','std','count']).reset_index()
    capsummary['t95_halfwidth']=capsummary.apply(lambda r:float(student.ppf(.975,r['count']-1)*r['std']/np.sqrt(r['count'])) if r['count']>1 else np.nan,axis=1)
    capsummary.to_csv(out/'capture_seed_summary.csv',index=False)
    inter=pd.DataFrame(combined['interventions']);inter=inter[inter.known_change!=0].copy()
    inter['group_oracle_absolute_error']=abs(inter.group_probe_change-inter.oracle_probe_change)
    inter['random_oracle_absolute_error']=abs(inter.random_probe_change-inter.oracle_probe_change)
    imsummary=inter.groupby(['model','seed','attribute','group_budget','image_id'])[['group_oracle_absolute_error','random_oracle_absolute_error','off_target_absolute_logit_change','random_off_target_absolute_logit_change']].mean().reset_index()
    imsummary.to_csv(out/'intervention_image_summary.csv',index=False)
    cis=[]
    for keys,df in imsummary.groupby(['model','seed','attribute','group_budget']):
        records_=df.to_dict('records')
        for field in ['group_oracle_absolute_error','random_oracle_absolute_error','off_target_absolute_logit_change','random_off_target_absolute_logit_change']:
            clean=[r for r in records_ if np.isfinite(r[field])]
            lo,hi=image_bootstrap(clean,field) if clean else [None,None]
            cis.append(dict(zip(['model','seed','attribute','group_budget'],keys),metric=field,mean=float(np.mean([r[field] for r in clean])) if clean else None,image_cluster95_low=lo,image_cluster95_high=hi))
    pd.DataFrame(cis).to_csv(out/'intervention_image_cluster_intervals.csv',index=False)
    trajectory=pd.DataFrame(combined['trajectories']);ci=[]
    for keys,df in trajectory.groupby(['model','seed','attribute','level','group_budget']):
        rows_=df.to_dict('records');lo,hi=image_bootstrap(rows_,'prediction')
        ci.append(dict(zip(['model','seed','attribute','level','group_budget'],keys),mean=float(df.prediction.mean()),image_cluster95_low=lo,image_cluster95_high=hi))
    pd.DataFrame(ci).to_csv(out/'trajectory_image_cluster_intervals.csv',index=False)
    fig,axs=plt.subplots(1,4,figsize=(9.2,2.4));colors=['#008b80','#89969b','#9f6cb8']
    # Mean across categories/images for each seed, bands show training-start SD.
    for ax,a in zip(axs,LEVELS):
        for method,color in zip(METHODS,colors):
            q=trajectory[(trajectory.model==method)&(trajectory.attribute==a)&(trajectory.group_budget==1)].groupby(['seed','level']).prediction.mean().unstack(0)
            x=q.index.values;mu=q.mean(1).values;sd=q.std(1).values
            ax.plot(x,mu,color=color,label=LABELS[method]);ax.fill_between(x,mu-sd,mu+sd,color=color,alpha=.16)
        ax.set_title(a,fontsize=9);ax.set_xlabel('Controlled level',fontsize=8);ax.tick_params(labelsize=7);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Frozen TRAIN readout',fontsize=8);axs[-1].legend(fontsize=6,frameon=False)
    fig.tight_layout();folder=out/'figures';folder.mkdir(exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(folder/('controlled_trajectories.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    text='''# Controlled variation and group diagnostics

All models are frozen saved COCO checkpoints. Group choices come from the original TRAIN-only contrast mapping. PCA, display scales, and ridge readouts use TRAIN images only. Evaluation uses disjoint previously inspected COCO audit images, not a new untouched test set. The cohort is selected by annotation and stable hash, never performance.

Each annotated instance is cropped from the pinned image crop, resized with aspect ratio preserved to a 144-pixel longest side, centered, and composited onto a neutral canvas. Rotation and scale therefore act about its fixed center. Foreground rotation, scale and brightness are controlled 2D image transformations, not 3D pose or physical illumination. Background brightness is a negative control for object-specific responses. Annotation-only geometry checks require usable support at all levels before any inference. Context removal, centering and compositing create distribution shift. These results assess this controlled setting, not all natural-image variation. High foreground-property recovery is useful; high response to background changes is a sensitivity diagnostic, not automatically an improvement.

Contribution PCA uses decoded contributions, not raw native coefficients. The first three components can omit variation. Invalid or inactive groups remain visible in pca.csv. RGB axes are independently fitted per group; hues do not identify the same semantic factor across models. Norm display limits use TRAIN statistics; displayed brightness is not an absolute cross-model scale.

One/two/four-group readouts retain the original encoder Top-k and use fixed TRAIN group prefixes. More groups also increase readout dimension. Missing additional positive-gain groups are not replaced by evaluation-selected groups.

Interventions transplant pooled native contributions between a prespecified pair of views of the same held-out object, holding the source residual fixed. A shared TRAIN-fitted raw-DINO attribute probe measures the change; four TRAIN activity/energy-neighbor groups supply deterministic random control prefixes. Rescaled controls match edit norm; zero-norm controls remain explicitly invalid. These are frozen-readout diagnostics, not native-backbone causal interventions, generated images, semantic exclusivity, or proof of recursive generalization.

CSV files retain every category, model, training start, object and transformation. Seed_summary reports sample SD and Student-t intervals across training starts. Trajectory_image_cluster_intervals resamples base images, keeping their views together; it conditions on fitted models. Colored plot bands are seed SD. Review failures as well as gains before choosing paper panels.

Research precedents: Fel et al., Structuring Sparsity (arXiv:2606.25234), Sections 3.5, 5.2 and Appendix G.4, for contribution PCA, controlled variation and frozen probes; Alain and Bengio (arXiv:1610.01644) for linear probes; PANet (ICCV 2019) for mask-pooled features; Everingham et al. (IJCV 2015) for paired image-level bootstrap evaluation. The exact compositing and transplantation protocol here is a new diagnostic adaptation.
'''
    (out/'README.md').write_text(text,encoding='utf8')
    manifest=dict(complete=True,protocol=protocol,models=rows,base_objects=len(records),variants_per_object=len(VARIANTS),no_featurizer_training=True,files={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='completion.json'})
    ctx['store'].write('completion.json',manifest)
    archive=out.parent/'controlled_variation_results.zip';pending=archive.with_suffix('.pending.zip')
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(pending) as z:
        if z.testzip() is not None:raise OSError('Archive verification failed')
    os.replace(pending,archive)
    Path(ctx['options']['receipt']).write_text(json.dumps(dict(complete=True,results=str(out),archive=str(archive))))
    log('2D ANALYSES COMPLETE',results=out,archive=archive)

def main(options):
    import experiment as exp,analysis,evaluation,backbone,resume_weights
    from coco_baselines import training
    from coco_baselines.storage import identity
    from coco_coordinated import run as memory
    from coco_ggf import matched_baselines as matched
    from coco_adaptation.inputs import JournalStore
    if not torch.cuda.is_available():raise RuntimeError('Choose a GPU runtime for DINO and frozen-model inference')
    torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
    safe_output(options['output'],[options['study_source'],options['source_output'],options['slics_source'],options['concept_source_root'],options['legacy_output'],*options['paths'].values()])
    log('VERIFY saved data lineage and normalization; no training will run')
    ctx=exp.prepare(dict(options,output=str(Path(options['output'])/'verified_setup')))
    source,rows=discover_primary(options['study_source'],Path(options['local'])/'read_primary',ctx)
    import native_sources,native_coherence,legacy_cache
    native_stores,native_rows=native_sources.discover(ctx,options)
    source_stamps=memory.source_snapshot([source,*native_stores.values()]);protocol=dict(version=3,semantic_protocol='five native methods; worst-condition object-proxy AP; native-output trajectory cosine; independent blinded ratings',source_run=source.run_id,source_sha=options['analysis_sha'],weights=[r['weights_sha'] for r in rows],native_models=native_rows,normalization=identity(ctx['norm']),categories=NAMES,per_category_limit=PER_CATEGORY,transformations=LEVELS,train_mapping='saved TRAIN-only greedy capture mapping for BSFs; fixed native category slots for comparators',ridge=.001,pca='shared coherence: TRAIN pooled native-output vectors, first3 for every method; BSF-specific diagnostics retain active-patch PCA',no_training=True)
    job=JournalStore(Path(options['output'])/'analysis',protocol,Path(options['local'])/'analysis_rescue',True)
    ctx['store']=ReadThrough(job,source);ctx['options']=options;ctx['out']=job.results
    ctx['store'].write('native_source_catalog.json',dict(models=native_rows,source_runs={k:v.run_id for k,v in native_stores.items()},no_training=True))
    # Restore only a weight snapshot into disposable local storage. Source is FrozenStore.
    resume_weights.restore(dict(ctx,store=source),rows,memory.required,training.weight_sha)
    pack=evaluation.coco_test(ctx);coco=backbone.annotations(ctx,options['source_output'])
    backbone.backbone_preflight(ctx,coco,options['source_output'])
    asset=Path(ctx['fresh_provenance']['backbone']['model'])
    records=select_cohort(ctx['packs']['train'],ctx,'train')+select_cohort(pack,ctx,'eval')
    records=geometry_preflight(ctx,coco,records);ctx['cohort']=records
    if {r['image_id'] for r in records if r['split']=='train'}&{r['image_id'] for r in records if r['split']=='eval'}:raise ValueError('TRAIN/evaluation overlap')
    ctx['store'].write('cohort.json',records);ctx['store'].write('protocol.json',protocol)
    categories=sorted({r['category'] for r in records});legacy=legacy_cache.find(ctx,options,records)
    feature_id=identity(dict(protocol=protocol,cohort=records,backbone=ctx['provenance']['backbone_weights']))
    features=Cache(Path(options['local'])/'new_features'/feature_id,Path(options['output'])/'features'/feature_id,feature_id)
    features=legacy_cache.cache(features,legacy,'features',legacy['feature_id'] if legacy else '',options['local'])
    log('SHARED NATIVE AUDIT',methods=native_coherence.ALL_METHODS,native_models=len(native_rows),reusing_v2_features=bool(legacy),note='No featurizer training or new joint 900-direction SLiCS reconstruction')
    extract(ctx,coco,records,features,asset)
    ctx['mask_probe']=analysis.fit_probe(ctx)
    all_results=[];semantic_results=[]
    for row in rows:
        start=time.monotonic();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();m=analysis.load_model(ctx,row);plans=get_plan(ctx,row,m,source)
        sig=identity(dict(features=feature_id,weights=row['weights_sha'],plans=plans))
        results=Cache(Path(options['local'])/'model_analysis'/sig,Path(options['output'])/'model_analysis'/sig,sig)
        oldsig=identity(dict(features=legacy['feature_id'],weights=row['weights_sha'],plans=plans)) if legacy else ''
        results=legacy_cache.cache(results,legacy,'model_analysis',oldsig,options['local'])
        summaries,pcas,D=collect(ctx,row,m,records,features,results,plans)
        result=evaluate_model(ctx,row,summaries,pcas,D,plans);all_results.append(result)
        from native_units import BSFUnits
        common_row,observations=native_coherence.bsfunits(row,summaries,D,plans,categories)
        adapter=BSFUnits(m,common_row['units'],ctx['device'])
        semantic_results.append(native_coherence.process(ctx,common_row,adapter,observations,categories,features,coco))
        del adapter,observations
        pca_folder=ctx['out']/'pca_bases';pca_folder.mkdir(exist_ok=True)
        for (c,g),pc in pcas.items():np.savez_compressed(pca_folder/(row['kind']+'_'+str(row['seed'])+'_'+str(c)+'_'+str(g)+'.npz'),**{k:v for k,v in pc.items() if k!='reason'})
        if row['seed']==21:atlas(ctx,row,m,summaries,pcas,D,plans,features,coco)
        torch.cuda.synchronize();ctx['store'].write('model_'+row['kind']+'_'+str(row['seed'])+'.json',dict(weights=row['weights_sha'],seconds=time.monotonic()-start,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),cost_scope='Frozen-model analysis and displays in this notebook; not original training cost',heldout_results=result))
        log('MODEL COMPLETE',model=row['kind'],seed=row['seed'],minutes=round((time.monotonic()-start)/60,2),remaining_models=len(rows)-len(all_results))
        del m,summaries;gc.collect();torch.cuda.empty_cache()
    memory.assert_immutable(source_stamps);matched.assert_sources(ctx)
    ctx['options']=dict(options,receipt=options['receipt']+'.twoD')
    finish(ctx,all_results,records,rows,protocol)
    for row in native_rows:
        start=time.monotonic();adapter=native_sources.load(ctx,row,native_stores[row['kind']],categories)
        common_row=dict(row,units=[row['slots'][c] for c in categories],interface=adapter.interface)
        observations=native_coherence.collect(ctx,common_row,adapter,records,features)
        semantic_results.append(native_coherence.process(ctx,common_row,adapter,observations,categories,features,coco))
        log('NATIVE MODEL COMPLETE',method=row['kind'],seed=row['seed'],minutes=round((time.monotonic()-start)/60,2))
        del adapter,observations;gc.collect();torch.cuda.empty_cache()
    native_coherence.finish(ctx,semantic_results)
    log('SEMANTIC AUTOMATED TESTS COMPLETE',methods=native_coherence.ALL_METHODS,note='Common five-method blinded review pack ready; human ratings not yet assessed')
    from lighting_study import run as physical_lighting
    log('START PHYSICAL LIGHTING',note='Frozen checkpoints; renderer, inference, discovery-only probes and PCA; no featurizer training')
    physical_lighting(ctx,rows,options)
    memory.assert_immutable(source_stamps);matched.assert_sources(ctx)
    out=ctx['out'];ctx['store'].write('whole_notebook_completion.json',dict(complete=True,scope=['controlled2D','contributionPCA','frozen_readout_interventions','Blender_physical_lighting','semantic_operational_tests','blinded_review_materials'],semantic_methods=native_coherence.ALL_METHODS,BSF_specific_methods=METHODS,human_ratings_complete=False,no_featurizer_training=True))
    archive=out.parent/'controlled_variation_and_lighting_results.zip';pending=archive.with_suffix('.pending.zip')
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(pending) as z:
        if z.testzip() is not None:raise OSError('Combined result archive verification failed')
    os.replace(pending,archive)
    Path(options['receipt']).write_text(json.dumps(dict(complete=True,results=str(out),archive=str(archive),physical_lighting_complete=True,twoD_complete=True,semantic_automated_complete=True,human_ratings_complete=False,review_pack=str(out/'semantic_coherence/blinded_review_public.zip'))))
    log('WHOLE NOTEBOOK COMPLETE',archive=archive)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();main(json.loads(Path(args.config).read_text()))
