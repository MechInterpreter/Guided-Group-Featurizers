"""Shared semantic diagnostics in each method's native output space."""
import hashlib,json,secrets,time
from pathlib import Path
import numpy as np
from cv_math import VARIANTS,LEVELS,Cache,digest,fit_ridge,predict,r2,target
from semantic_math import FOREGROUND,average_precision

ALL_METHODS=['agf_g','released_vanilla','supervised_vanilla','slics','concept_sae']
LABELS=['GGF','Vanilla BSF','Supervised Vanilla','SLiCS','Concept-SAE']

def decoded_consistency(delta,ids):
    z=np.asarray(delta,float);ids=np.asarray(ids)
    if z.ndim!=3 or len(ids)!=len(z):raise ValueError('Native trajectory shape')
    norm=np.linalg.norm(z,axis=-1);valid=norm>1e-10*max(float(norm.max(initial=0)),1.)
    u=np.divide(z,norm[:,:,None],out=np.zeros_like(z),where=valid[:,:,None])
    pairs=(ids[:,None]!=ids[None,:])&np.triu(np.ones((len(z),len(z)),bool),1)
    cos=np.einsum('ikd,jkd->ijk',u,u).clip(-1,1)
    return dict(consistency=float(cos[pairs].mean()) if pairs.any() else None,
        zero_change_fraction=float((~valid).mean()),decoded_change_RMS=float(np.sqrt(np.mean(norm**2))))

def fit_pca(train):
    x=np.asarray(train,float)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise ValueError('TRAIN PCA input')
    mean=x.mean(0);centered=x-mean;_,s,V=np.linalg.svd(centered,full_matrices=False)
    axes=np.zeros((3,x.shape[1]));axes[:min(3,len(V))]=V[:3]
    for a in axes:
        if a[np.argmax(abs(a))]<0:a*=-1
    variance=np.zeros(3);variance[:min(3,len(s))]=s[:3]**2/(len(x)-1)
    return dict(mean=mean,V=axes,variance=variance,valid=bool(variance[0]>1e-12),fraction=float(variance.sum()/max(float(np.sum(s*s)/(len(x)-1)),1e-30)))

def numeric(row,observations,categories):
    """No choices depend on evaluation data. Scores stay in native units."""
    evaluation=[(r,s) for r,s in observations if r['split']=='eval'];training=[(r,s) for r,s in observations if r['split']=='train']
    ids=np.array([r['image_id'] for r,s in evaluation]);truth=np.array([r['category'] for r,s in evaluation])
    if len(set(ids))!=len(ids):raise ValueError('One instance per evaluation image')
    scores=np.stack([s['scores'] for r,s in evaluation]);summary=[];conditions=[];pcas={};readouts=[]
    for j,c in enumerate(categories):
        tr=[s for r,s in training if r['category']==c];te=[(r,s) for r,s in evaluation if r['category']==c]
        pc=fit_pca(np.concatenate([s['vectors'][:,j] for s in tr]));pcas[c]=pc
        aps=[];cons=[];zero=[];mag=[]
        for attribute in FOREGROUND:
            ix=[i for i,(a,v) in enumerate(VARIANTS) if a==attribute]
            for i in ix:
                ap=average_precision(truth==c,scores[:,i,j]);aps.append(ap)
                conditions.append(dict(model=row['kind'],seed=row['seed'],category=c,attribute=attribute,level=VARIANTS[i][1],category_AP=ap))
            z=np.stack([s['vectors'][ix,j] for r,s in te]);changes=np.delete(z-z[:,2:3],2,axis=1)
            q=decoded_consistency(changes,[r['image_id'] for r,s in te]);cons.append(q['consistency']);zero.append(q['zero_change_fraction']);mag.append(q['decoded_change_RMS'])
            conditions.append(dict(model=row['kind'],seed=row['seed'],category=c,attribute=attribute,**q))
        bgix=[i for i,(a,v) in enumerate(VARIANTS) if a=='background']
        bg=np.stack([s['scores'][bgix,j] for r,s in te]);scale=max(float(np.sqrt(np.mean([s['scores'][:,j]**2 for s in tr]))),1e-10)
        summary.append(dict(model=row['kind'],seed=row['seed'],category=c,group=int(row['units'][j]),category_name=te[0][0]['category_name'],
            worst_condition_AP=float(min(aps)),mean_condition_AP=float(np.mean(aps)),identity_AP=average_precision(truth==c,scores[:,2,j]),
            trajectory_consistency=float(np.mean(cons)),zero_change_fraction=float(np.mean(zero)),decoded_change_RMS=float(np.mean(mag)),
            background_norm_sensitivity=float(np.mean(abs(bg-bg[:,2:3]))/scale),evaluation_objects=len(te),comparison_objects=len(evaluation)-len(te),
            interface=row['interface'],common_PCA_variance_fraction=pc['fraction'],common_PCA_valid=pc['valid']))
        for attribute,levels in LEVELS.items():
            ix=[i for i,(a,v) in enumerate(VARIANTS) if a==attribute]
            X=np.concatenate([(s['vectors'][ix,j]-pc['mean'])@pc['V'].T for s in tr]);Y=np.tile([target(attribute,v) for v in levels],len(tr))
            probe=fit_ridge(X,Y);values=[]
            for r,s in te:
                y=np.array([target(attribute,v) for v in levels]);pred=predict(probe,(s['vectors'][ix,j]-pc['mean'])@pc['V'].T)
                values.append(r2(y,pred))
            readouts.append(dict(model=row['kind'],seed=row['seed'],category=c,attribute=attribute,common_PC_dimensions=3,heldout_mean_object_R2=float(np.mean(values)),valid_PCA=pc['valid']))
    return dict(summary=summary,conditions=conditions,readouts=readouts),pcas

def bsfunits(row,summaries,D,plans,categories):
    """Reuse already computed BSF codes, without another encoder pass."""
    groups=[plans[c]['groups'][0] for c in categories];observations=[]
    for rec,s in summaries:
        ix=[int(np.flatnonzero(s['gids']==g)[0]) for g in groups]
        observations.append((rec,dict(scores=s['norm'][:,ix],vectors=np.einsum('vgr,grd->vgd',s['pooled'][:,ix],D[groups]))))
    return dict(row,units=groups,interface='gated BSF norm / native additive contribution'),observations

def collect(ctx,row,adapter,records,features):
    from native_units import summarize
    from cv_study import record_key,log
    sig=digest(dict(version=3,weights=row['weights_sha'],cohort=records,interface=row['interface'],source=ctx['options']['analysis_sha']))
    cache=Cache(Path(ctx['options']['local'])/'native_units'/sig,Path(ctx['options']['output'])/'native_units'/sig,sig)
    observations=[];started=time.monotonic()
    for i,rec in enumerate(records):
        key=record_key(rec);obs=cache.get(key)
        if obs is None:
            f=features.get(key);scores,parts=adapter.maps(f['x']);q=summarize(f['x'],f['fraction'],scores,parts)
            obs=cache.put(key,scores=q['scores'].astype(np.float32),vectors=q['vectors'].astype(np.float32))
        observations.append((rec,obs))
        if (i+1)%5==0:
            elapsed=time.monotonic()-started;log('NATIVE CONCEPT ENCODE',method=row['kind'],seed=row['seed'],objects=i+1,total=len(records),minutes=round(elapsed/60,2),ETA_minutes=round(elapsed/(i+1)*(len(records)-i-1)/60,2),solver=getattr(adapter,'last_solver',None))
    return observations

def make_cards(ctx,row,adapter,observations,pcas,categories,features,coco):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    from cv_study import source_crop,record_key
    from cv_math import render
    root=ctx['out']/'semantic_coherence';private=root/'private';public=root/'blinded_review'
    private.mkdir(parents=True,exist_ok=True);public.mkdir(exist_ok=True)
    keypath=private/'review_key.json';key=json.loads(keypath.read_text()) if keypath.exists() else {}
    ev=[(r,s) for r,s in observations if r['split']=='eval'];tr=[(r,s) for r,s in observations if r['split']=='train']
    for j,c in enumerate(categories):
        token=row['kind']+'_'+str(row['seed'])+'_'+str(c)
        if token not in key:
            key[token]=dict(card_id=secrets.token_hex(6),model=row['kind'],seed=row['seed'],category=c,group=row['units'][j])
            tmp=keypath.with_suffix('.pending');tmp.write_text(json.dumps(key,indent=2));tmp.replace(keypath)
        card=key[token]['card_id'];path=public/(card+'.png');vp=public/(card+'_variation.png')
        if path.exists() and vp.exists():
            from PIL import Image
            try:
                for p in [path,vp]:
                    with Image.open(p) as im:im.verify()
                continue
            except (OSError,ValueError):pass
        trscore=np.concatenate([s['scores'][:,j] for r,s in tr]);mu=float(trscore.mean());sd=max(float(trscore.std()),1e-8)
        top=sorted(ev,key=lambda rs:(-float(rs[1]['scores'][2,j]),digest(rs[0]['image_id'])))[:8]
        fig,ax=plt.subplots(2,8,figsize=(14,4.1))
        for k,(rec,s) in enumerate(top):
            rgb,mask=source_crop(ctx,coco,rec);im,_=render(rgb,mask,*VARIANTS[2]);f=features.get(record_key(rec))
            response,_=adapter.maps(f['x'][2:3]);scaled=np.clip((response[0,:,j]-mu)/sd,-2.5,2.5).reshape(14,14)
            up=zoom(scaled,16,order=1);alpha=abs(up)/2.5*.8;color=np.where((up>=0)[:,:,None],np.array([0.,.58,.53]),np.array([.52,.34,.7]))
            base=np.asarray(im)/255;ax[0,k].imshow(im);ax[1,k].imshow(base*(1-alpha[:,:,None])+color*alpha[:,:,None])
        for a in ax.flat:a.axis('off')
        fig.suptitle('Card '+card+': strongest distinct object views | relative native response',fontsize=12);fig.tight_layout()
        tmp=path.with_suffix('.pending.png');fig.savefig(tmp,dpi=140);tmp.replace(path);plt.close(fig)
        pc=pcas[c];values=np.stack([(s['vectors'][2,j]-pc['mean'])@pc['V'].T for r,s in ev]);fig,ax=plt.subplots(3,8,figsize=(14,6.4))
        for axis in range(3):
            ordered=sorted(range(len(ev)),key=lambda i:(float(values[i,axis]),digest(ev[i][0]['image_id'])))
            for k,i in enumerate(ordered[:4]+ordered[-4:]):
                rgb,mask=source_crop(ctx,coco,ev[i][0]);im,_=render(rgb,mask,*VARIANTS[2]);ax[axis,k].imshow(im);ax[axis,k].axis('off')
                if k in [0,4]:ax[axis,k].set_title(('Low' if k==0 else 'High')+' PCA '+str(axis+1),fontsize=9)
        fig.suptitle('Card '+card+': native-output variation, TRAIN PCA'+(' | constant output' if not pc['valid'] else ''),fontsize=12);fig.tight_layout()
        tmp=vp.with_suffix('.pending.png');fig.savefig(tmp,dpi=140);tmp.replace(vp);plt.close(fig)

def process(ctx,row,adapter,observations,categories,features,coco):
    result,pcas=numeric(row,observations,categories);make_cards(ctx,row,adapter,observations,pcas,categories,features,coco)
    folder=ctx['out']/'semantic_coherence/private/pca';folder.mkdir(parents=True,exist_ok=True)
    for c,pc in pcas.items():np.savez_compressed(folder/(row['kind']+'_'+str(row['seed'])+'_'+str(c)+'.npz'),**pc)
    return result

def finish(ctx,results):
    import pandas as pd
    models={r['model'] for q in results for r in q['summary']}
    if models!=set(ALL_METHODS):raise ValueError('Every declared method must finish; no dropping a failed competitor')
    from semantic_study import finish as summary
    summary(ctx,results)
    root=ctx['out']/'semantic_coherence';pd.DataFrame([r for q in results for r in q['readouts']]).to_csv(root/'common_3PC_property_readouts.csv',index=False)
    p=root/'completion.json';done=json.loads(p.read_text());done.update(expected_methods=ALL_METHODS,common_native_units=True,comparators=['SLiCS','Concept-SAE'],review_scope='same images and pooling, native responses; TRAIN pooled-output PCA, three coordinates for every method; method budgets differ')
    p.write_text(json.dumps(done,indent=2))
