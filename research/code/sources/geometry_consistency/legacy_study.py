"""Geometry first, whole-bank discovery, then input interventions. Frozen fits only."""
import argparse,ast,csv,gc,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from study_math import panel,balanced_indices,geometry,pca_fit,pca_rgb,background_composites
from cv_math import Cache,digest

METHODS=['agf_g','supervised_vanilla','released_vanilla','slics','concept_sae']
RUNS=[21,22,23]
SYN_REV='61ec39e676af44a7a33146f78f54019b8a0f97fc'

def log(s,**q):print(s,json.dumps(q,default=str),flush=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,q):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending');t.write_text(json.dumps(q,indent=2,allow_nan=False),encoding='utf8');os.replace(t,p)
def csv_write(p,rows):
    if not rows:return
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending')
    with tmp.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
def done_json(p,q):
    atomic_json(p,q);atomic_json(str(p)+'.sha.json',dict(sha=sha(p)))
def read_done(p):
    p=Path(p);s=Path(str(p)+'.sha.json')
    if not p.exists() or not s.exists():return None
    if sha(p)!=json.loads(s.read_text())['sha']:raise ValueError('Corrupt committed result '+str(p))
    return json.loads(p.read_text())

def source_models(ctx,options):
    from cv_study import discover_primary
    from coco_baselines.storage import identity
    from coco_baselines import training
    from coco_ggf import matched_baselines as matched
    import native_sources,endpoint_restore
    source,rows=discover_primary(options['study_source'],Path(options['local'])/'read_primary',ctx)
    # A completed primary journal catalogs endpoints without copying endpoints
    # reused from its upstream frozen stores. Restore verified inference-only
    # weights locally instead of requiring every endpoint in that one journal.
    endpoints=[source,ctx['store'],*ctx.get('sources',{}).values()]
    frozen_sources=[source,*ctx.get('sources',{}).values()]
    prior_root=Path(options['study_source']).parent/'GGF_COCO_1024_supervised_VOC_v1'
    for manifest_path in sorted((prior_root/'results').glob('*/manifest.json')):
        record=json.loads(manifest_path.read_text());cfg=record['config']
        if (cfg.get('training_identity')==ctx['packs']['train']['identity']
                and cfg.get('selection_identity')==ctx['packs']['select']['identity']
                and cfg.get('normalization_sha')==identity(ctx['norm'])
                and cfg.get('versions')==matched.versions()
                and cfg.get('source_sha')==options['source_sha']):
            store=matched.discover(prior_root,Path(options['local'])/'read_compatible_endpoints',run=record['run_id'])
            endpoints.append(store);frozen_sources.append(store)
    endpoint_restore.restore(ctx,rows,None,training.weight_sha,endpoints)
    ctx['endpoint_sources']=frozen_sources
    log('BSF ENDPOINT PREFLIGHT COMPLETE',models=len(rows),note='All nine catalog hashes verified. Only local inference snapshots written.')
    stores,native=native_sources.discover(ctx,options)
    return source,stores,rows+native

def load(ctx,row,source,stores):
    from coco_coordinated.run import required
    from coco_baselines import training
    from scale_models import model
    import native_sources,concept_sae
    if row['kind'] in METHODS[:3]:
        path=Path(ctx['local'])/'frozen_weights'/(row['weights_sha']+'.pt')
        weights=torch.load(path,map_location='cpu',weights_only=True)
        if training.weight_sha(weights)!=row['weights_sha']:raise ValueError('Frozen local BSF weights differ')
        m=model('released_vanilla',1024,12,16,768).to(ctx['device']);m.load_state_dict(weights);m.eval()
        state=weights
        D=m.atoms().detach().cpu().numpy();A=None
        units=[int(next(t for t in row['targets'] if t['category_index']==c)['contrast_group']) for c in ctx['cats']]
        if row['kind']=='agf_g' and sorted(units)!=list(range(75)):raise ValueError('GGF guided-bank inventory changed. Inspect the saved training targets before labeling unit cohorts.')
    elif row['kind']=='slics':
        state=required(stores['slics'],row['key'])
        if state['updates']!=10640 or training.weight_sha({'D':state['D']})!=row['weights_sha']:raise ValueError('Frozen SLiCS weights differ')
        D=state['D'].reshape(75,12,768).detach().cpu().numpy();m=None;A=None
        units=[int(row['slots'][c]) for c in ctx['cats']]
    else:
        state=required(stores['concept_sae'],row['key']);native_sources.validate_concept(state,row['seed'])
        if training.weight_sha(state['weights'])!=row['weights_sha']:raise ValueError('Frozen Concept-SAE weights differ')
        m=concept_sae.build(concept_sae.source(ctx['options']['concept_source']),75).to(ctx['device']);m.load_state_dict(state['weights']);m.eval();D=None
        A=torch.cat([m.concept_model.token_aggregator.weight,m.free_model.token_aggregator.weight],1).detach().cpu().numpy()
        units=[int(row['slots'][c]) for c in ctx['cats']]
    if m is not None:
        for p in m.parameters():p.requires_grad_(False)
    del state
    return m,D,A,units

@torch.inference_mode()
def observe(kind,m,D,A,x,picks,device):
    from native_units import partial_nnls
    xx=torch.as_tensor(np.array(x,copy=True),dtype=torch.float32,device=device)
    if kind in METHODS[:3]:
        z=m.encode(xx)[picks].float().cpu().numpy()
        return dict(z=z,score=np.linalg.norm(z,axis=-1))
    if kind=='slics':
        z,solver=partial_nnls(xx[picks],torch.tensor(D,device=device));z=z.float().cpu().numpy()
        return dict(z=z,score=np.linalg.norm(z,axis=-1))
    masks,presence=m.concept_model.tokenizer(xx[None]);concept=masks*presence;free=m.free_model.tokenizer(xx[None])
    cp=torch.cat([torch.stack([p(concept[:,j]) for j,p in enumerate(m.concept_model.output_projectors)],1),torch.stack([p(free[:,j]) for j,p in enumerate(m.free_model.output_projectors)],1)],1)[0].float().cpu().numpy()
    response=np.concatenate([concept[0].T.float().cpu().numpy(),np.abs(A[:,75:])*np.linalg.norm(cp[75:],axis=-1)[None]],axis=1)
    return dict(cp=cp,score=response[picks])

def encode_panel(ctx,row,pack,pn,m,D,A):
    sig=digest(dict(weights=row['weights_sha'],pack=pack['identity'],ids=pn['ids'].tolist(),patch=pn['patch'].tolist(),revision=ctx['signature']))
    cache=Cache(Path(ctx['options']['local'])/'encodings'/sig,ctx['out']/'encodings'/sig,sig)
    chunks=[]
    for i,image_id in enumerate(pack['ids']):
        ix=np.flatnonzero(pn['images']==i);key=str(int(image_id));q=cache.get(key)
        if q is None:q=cache.put(key,**observe(row['kind'],m,D,A,pack['x'][i],pn['patch'][ix],ctx['device']))
        chunks.append(q)
        if (i+1)%25==0:log('ENCODE',method=row['kind'],seed=row['seed'],images=i+1,total=len(pack['ids']))
    score=np.concatenate([q['score'] for q in chunks])
    return dict(score=score,z=np.concatenate([q['z'] for q in chunks]) if D is not None else None,cp=np.stack([q['cp'] for q in chunks]) if D is None else None)

def embed(obs,pn,g,ix,D,A):
    if D is not None:
        # The SVD map preserves decoded Euclidean geometry exactly, without a
        # 768-dimensional allocation. It is not a fitted semantic readout.
        u,s,_=np.linalg.svd(D[g].astype(float),full_matrices=False)
        return obs['z'][ix,g]@(u*s)
    return obs['cp'][pn['images'][ix],g]*A[pn['patch'][ix],g,None]

def ap_profiles(scores,pn):
    from sklearn.metrics import average_precision_score
    out=np.full((scores.shape[1],pn['y'].shape[1]),np.nan)
    for c in range(out.shape[1]):
        use=pn['valid'][:,c];y=pn['y'][use,c]
        if not y.any() or y.all():continue
        # Exact, shared tied-score AP implementation.
        for g in range(out.shape[0]):out[g,c]=average_precision_score(y,scores[use,g])
    prevalence=np.array([pn['y'][pn['valid'][:,c],c].mean() for c in range(out.shape[1])])
    return out,prevalence

def unit_geometry(ctx,row,g,c,ref,test,rp,tp,D,A,scope):
    ri=balanced_indices(rp['y'][:,c],rp['valid'][:,c],256,61023+c)
    ti=balanced_indices(tp['y'][:,c],tp['valid'][:,c],128,71023+c)
    q=dict(method=row['kind'],seed=row['seed'],unit=int(g),category=int(ctx['cats'][c]),scope=scope,reference_patches=len(ri),evaluation_patches=len(ti))
    if not len(ri) or not len(ti):return dict(q,status='insufficient_annotation_support')
    r=embed(ref,rp,g,ri,D,A);t=embed(test,tp,g,ti,D,A)
    for normalized,name in [(False,'raw'),(True,'direction')]:
        for k in [10,30]:
            v=geometry(r,t,rp['y'][ri,c],tp['y'][ti,c],normalized,k)
            q.update({f'{name}_k{k}_{n}':x for n,x in v.items()})
    # PCA is fitted to ALL sampled calibration patches firing this unit,
    # not only to examples carrying its assigned category.
    active=np.flatnonzero(ref['score'][:,g]>1e-10)
    if len(active)>1024:active=np.random.default_rng(818+g).choice(active,1024,replace=False)
    allr=embed(ref,rp,g,active,D,A);pc=pca_fit(allr)
    if pc is not None:
        folder=ctx['out']/'pca'/row['kind']/str(row['seed']);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/f'unit_{g}.npz',**pc,reference_ids=rp['ids'][active],reference_patch=rp['patch'][active],reference_category_labels=rp['y'][active],reference_coordinates=(allr-pc['mean'])@pc['axes'].T,evaluation_ids=tp['ids'][ti],evaluation_patch=tp['patch'][ti],evaluation_coordinates=(t-pc['mean'])@pc['axes'].T,evaluation_rgb=pca_rgb(t,pc))
        q['PCA_variance_fraction']=pc['variance_fraction']
    return dict(q,status='complete')

def atlas(ctx,row,m,D,A,units,ref,test,rp,tp):
    """Same person images across methods plus predeclared bank locations.

    Display choices never select units by a favorable GGF comparison. These
    selected illustrations do not replace every-unit quantitative results.
    """
    if row['seed']!=21:return
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from coco_baselines import data
    candidates=[units[0]]
    G=ref['score'].shape[1]
    candidates+=([75,256,512,768,1023] if G==1024 else [75,90,110] if G==111 else [])
    folder=ctx['out']/'preview';folder.mkdir(exist_ok=True)
    for g in dict.fromkeys(candidates):
        output=folder/f"{row['kind']}_{g}.png"
        if output.exists():continue
        active=np.flatnonzero(ref['score'][:,g]>1e-10)
        if len(active)<3:continue
        if len(active)>1024:active=np.random.default_rng(818+g).choice(active,1024,replace=False)
        v=embed(ref,rp,g,active,D,A);pc=pca_fit(v)
        if pc is None:continue
        # Person images are mask-only chosen and shared. Other units use their
        # highest calibration-image responses, and are explicitly exploratory.
        pack=ctx['packs']['confirm'] if g==units[0] else ctx['packs']['select']
        pn=tp if g==units[0] else rp;obs=test if g==units[0] else ref
        if g==units[0]:
            possible=np.unique(pn['images'][pn['y'][:,0]])[:3]
        else:
            image_scores=np.array([obs['score'][pn['images']==i,g].max(initial=0) for i in range(len(pack['ids']))])
            possible=np.argsort(-image_scores,kind='stable')[:3]
        fig,axes=plt.subplots(3,4,figsize=(12,8),layout='constrained',squeeze=False)
        scale=max(float(np.quantile(ref['score'][:,g],.99)),1e-8)
        evalvec=[]
        for rr,i in enumerate(possible):
            image_id=int(pack['ids'][i]);p=data.download(data.COCO_BUCKET+f'train2017/{image_id:012d}.jpg',Path(ctx['options']['local'])/'images'/f'{image_id:012d}.jpg')
            with Image.open(p) as im:im=im.convert('RGB');rgb=np.array(data.transform_image(im,data.geometry(*im.size)))
            q=observe(row['kind'],m,D,A,pack['x'][i],np.arange(196),ctx['device'])
            tmp=dict(z=q.get('z'),cp=q.get('cp')[None] if D is None else None)
            full=dict(images=np.zeros(196,int),patch=np.arange(196))
            values=embed(tmp,full,g,np.arange(196),D,A);evalvec.append(values)
            heat=q['score'][:,g].reshape(14,14);colors=pca_rgb(values,pc).reshape(14,14,3)
            axes[rr,0].imshow(rgb);axes[rr,0].set_title(f'Image {image_id}')
            axes[rr,1].imshow(rgb);hm=axes[rr,1].imshow(heat,extent=(0,224,224,0),alpha=.55,cmap='viridis',vmin=0,vmax=scale);axes[rr,1].set_title('Native response')
            axes[rr,2].imshow(rgb);rgba=np.dstack([colors,(q['score'][:,g].reshape(14,14)>1e-10)*.65]);axes[rr,2].imshow(rgba,extent=(0,224,224,0));axes[rr,2].set_title('Contribution PCA RGB')
            uv=(values-pc['mean'])@pc['axes'].T;on=q['score'][:,g]>1e-10
            axes[rr,3].scatter(uv[on,0],uv[on,1] if uv.shape[1]>1 else np.zeros(on.sum()),c=pca_rgb(values[on],pc),s=8);axes[rr,3].set_title('Contribution PC1 / PC2')
            for ax in axes[rr,:3]:ax.axis('off')
        fig.colorbar(hm,ax=axes[:,1],shrink=.6,label='Native score, calibration-fixed limits')
        fig.suptitle(f"{row['kind']} unit {g}. PCA and colors fitted on calibration only.")
        fig.savefig(output,dpi=180);plt.close(fig)
    links='\n'.join(f'<h3>{p.stem}</h3><img src="{p.name}" style="max-width:100%">' for p in sorted(folder.glob('*.png')))
    (folder/'index.html').write_text('<html><meta charset="utf-8"><h1>Selected response and contribution geometry examples</h1><p>Person examples use the same annotation-selected evaluation images across methods. Other bank positions were fixed in advance and use high calibration responses. RGB indicates PCA location, not semantic correctness or comparable strength across methods.</p>'+links+'</html>',encoding='utf8')

def summarize(ctx,stage):
    rows=[]
    for p in (ctx['out']/stage).glob('*.json'):
        q=read_done(p)
        if q is not None:rows+=q if isinstance(q,list) else [q]
    csv_write(ctx['out']/f'{stage}_all_units.csv',rows)
    import pandas as pd
    if not rows:return
    df=pd.DataFrame(rows)
    if stage in ['discovery','dictionary_geometry']:
        all_units=df.copy();all_units['scope']='whole_dictionary';df=pd.concat([df,all_units],ignore_index=True)
    metrics=[c for c in df if c.endswith('coverage_adjusted_purity') or c in ['test_AP','test_uplift','stable_category','test_focus','target_drop','untouched_change','appearance_change','target_drop_above_untouched','background_change','alternate_background_target_drop']]
    metrics=[c for c in metrics if pd.api.types.is_numeric_dtype(df[c])]
    if not metrics:return
    # Average units/categories within run first. Runs, not patches, supply SD.
    group=['method','seed']+(['scope'] if 'scope' in df else [])+(['dataset'] if 'dataset' in df else [])
    if stage=='controlled_consistency':
        # First average repeated edits per unit/category, then units within
        # category, then categories equally. Pair counts are not category weights.
        base=df.groupby(group+['category','unit'],dropna=False)[metrics].mean().reset_index()
        base=base.groupby(group+['category'],dropna=False)[metrics].mean().reset_index()
        per=base.groupby(group,dropna=False)[metrics].mean().reset_index()
    else:per=df.groupby(group,dropna=False)[metrics].mean().reset_index()
    per.to_csv(ctx['out']/f'{stage}_per_run.csv',index=False)
    stats=per.groupby([c for c in group if c!='seed'])[metrics].agg(['mean','std','count'])
    stats.to_csv(ctx['out']/f'{stage}_mean_SD.csv')
    from scipy.stats import t
    paired=[]
    comparison_scope=per['scope'].astype(str) if 'scope' in per else pd.Series('all',index=per.index)
    if 'dataset' in per:comparison_scope=comparison_scope+' / '+per['dataset'].astype(str)
    for scope in comparison_scope.unique():
        sub=per[comparison_scope==scope]
        for other in METHODS[1:]:
            left=sub[sub.method=='agf_g'].set_index('seed');right=sub[sub.method==other].set_index('seed')
            ids=left.index.intersection(right.index)
            for metric in metrics:
                d=(left.loc[ids,metric]-right.loc[ids,metric]).dropna().to_numpy()
                if len(d)<2:continue
                mean=float(d.mean());sd=float(d.std(ddof=1));half=float(t.ppf(.975,len(d)-1)*sd/np.sqrt(len(d)))
                paired.append(dict(scope=scope,comparator=other,metric=metric,paired_runs=len(d),GGF_minus_comparator=mean,run_SD=sd,CI95_low=mean-half,CI95_high=mean+half))
    csv_write(ctx['out']/f'{stage}_paired_run_intervals.csv',paired)
    if len(per):
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        display=per[per.scope.isin(['fixed_category_readout','whole_dictionary'])] if 'scope' in per else per
        if len(display):
            metric=metrics[0];fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
            for scope,part in display.groupby(['scope','dataset'] if 'dataset' in display else 'scope'):
                v=part.groupby('method')[metric].agg(['mean','std']).reindex(METHODS)
                ax.errorbar(np.arange(5),v['mean'],yerr=v['std'].fillna(0),marker='o',capsize=3,label=str(scope))
            ax.set_xticks(range(5),['GGF','Supervised\nVanilla','Vanilla','SLiCS','Concept-SAE']);ax.set_ylabel(metric.replace('_',' '));ax.legend(fontsize=8)
            fig.savefig(ctx['out']/f'{stage}_overview.png',dpi=200);plt.close(fig)
    log('EARLY RESULTS',stage=stage,table=per.to_dict('records'))

def run_geometry(ctx,row,source,stores,rp,tp):
    path=ctx['out']/'guided_geometry'/f"{row['kind']}_{row['seed']}.json"
    allpath=ctx['out']/'dictionary_geometry'/path.name
    if read_done(path) is not None and read_done(allpath) is not None:return
    m,D,A,units=load(ctx,row,source,stores)
    ref=encode_panel(ctx,row,ctx['packs']['select'],rp,m,D,A)
    test=encode_panel(ctx,row,ctx['packs']['confirm'],tp,m,D,A)
    # Fixed calibration RMS puts pair changes in response-scale units without
    # using intervention outcomes or per-image display normalization.
    scale_path=ctx['out']/'response_scales'/f"{row['kind']}_{row['seed']}.npy"
    scales=np.maximum(np.sqrt(np.mean(ref['score'].astype(float)**2,axis=0)),1e-8)
    temp=scale_path.with_suffix('.pending.npy');np.save(temp,scales);os.replace(temp,scale_path)
    if read_done(path) is None:
        result=[unit_geometry(ctx,row,g,c,ref,test,rp,tp,D,A,'fixed_category_readout') for c,g in enumerate(units)]
        done_json(path,result);summarize(ctx,'guided_geometry')
    if ctx['options'].get('phase')=='guided':
        atlas(ctx,row,m,D,A,units,ref,test,rp,tp)
        del m,ref,test;gc.collect();torch.cuda.empty_cache();return
    # One independently assigned category per native unit, including every
    # non-readout BSF group and every Concept-SAE free token. Never name on test.
    discover_path=ctx['out']/'discovery'/path.name
    discovery=read_done(discover_path)
    if discovery is None:
        ap,prev=ap_profiles(ref['score'],rp);bp,bprev=ap_profiles(test['score'],tp)
        uplift=np.maximum((ap-prev)/(1-prev),0);testup=np.maximum((bp-bprev)/(1-bprev),0)
        discovery=[]
        for g in range(len(ap)):
            a=np.nan_to_num(uplift[g]);c=int(a.argmax());assigned=bool(a.max()>0 and np.any(ref['score'][:,g]>1e-10))
            total=float(np.nansum(testup[g]));cohort=('directly_guided' if g<75 else 'not_directly_guided') if row['kind']=='agf_g' else ('guided_token' if g<75 else 'free_token') if row['kind']=='concept_sae' else ('category_dictionary' if row['kind']=='slics' else 'other_groups' if g not in units else 'category_readout')
            val=float(bp[g,c]);pre=float(bprev[c])
            if not np.isfinite(val):val=0.
            discovery.append(dict(method=row['kind'],seed=row['seed'],unit=g,scope=cohort,category=int(ctx['cats'][c]) if assigned else None,assigned=assigned,test_AP=val if assigned else 0.,test_uplift=(val-pre)/(1-pre) if assigned and np.isfinite(pre) else 0.,test_focus=float(np.nan_to_num(testup[g,c])/total) if assigned and total>0 else 0.,stable_category=bool(assigned and c==np.nan_to_num(testup[g]).argmax()),calibration_activity=float(np.mean(ref['score'][:,g]>1e-10))))
        done_json(discover_path,discovery);summarize(ctx,'discovery')
    # Every unit is retained. Category-unassigned units are reported, not
    # silently omitted or declared semantically meaningless.
    result=[]
    for entry in discovery:
        g=entry['unit'];p=ctx['out']/'dictionary_units'/row['kind']/str(row['seed'])/f'{g}.json'
        q=read_done(p)
        if q is None:
            if entry['assigned']:
                q=dict(entry,**{k:v for k,v in unit_geometry(ctx,row,g,ctx['cats'].index(entry['category']),ref,test,rp,tp,D,A,entry['scope']).items() if k not in entry})
            else:
                q=dict(entry,status='no_positive_object_association')
                # Zero *evidence* in unconditional bank summaries. This does
                # not label a possibly non-object concept meaningless.
                for mode in ['raw','direction']:
                    for k in [10,30]:q[f'{mode}_k{k}_coverage_adjusted_purity']=0.
            done_json(p,q)
        result.append(q)
        if (g+1)%64==0:log('DICTIONARY GEOMETRY',method=row['kind'],seed=row['seed'],unit=g+1,total=len(discovery))
    done_json(allpath,result);summarize(ctx,'dictionary_geometry')
    del m,ref,test;gc.collect();torch.cuda.empty_cache()

def syn_pairs(ctx):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    root=ctx['out']/'syncoco';root.mkdir(exist_ok=True)
    meta=hf_hub_download('jokl/syncoco','metadata.csv',repo_type='dataset',revision=SYN_REV,local_dir=root)
    df=pd.read_csv(meta);wanted=set(ctx['cats']);chosen=[]
    for c,part in df.groupby('removed_cls_idx'):
        if int(c) not in wanted:continue
        part=part.copy();part['order']=part['pair_id'].map(lambda x:digest(['pair',int(x)]));part=part.sort_values('order').head(20)
        chosen+=part.drop(columns='order').to_dict('records')
    atomic_json(ctx['out']/'controlled_cohort.json',dict(dataset='jokl/syncoco',revision=SYN_REV,metadata_sha=sha(meta),pairs=chosen,source_image_identity='coco_idx is a dataset index, not a verified COCO image ID. Source overlap is unknown. This is exploratory functional validation.',support={str(c):sum(int(p['removed_cls_idx'])==c for p in chosen) for c in ctx['cats']}))
    return root,chosen

def extract_pairs(ctx,root,pairs):
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor,AutoModel
    from coco_baselines import data
    from coco_stratified import dataset
    import backbone
    from PIL import Image,ImageEnhance
    coco=backbone.annotations(ctx,ctx['options']['source_output']);backbone.backbone_preflight(ctx,coco,ctx['options']['source_output'])
    asset=Path(ctx['fresh_provenance']['backbone']['model'])
    proc=AutoImageProcessor.from_pretrained(str(asset));m=AutoModel.from_pretrained(str(asset),dtype=torch.float32).eval().to(ctx['device'])
    dataset.verify_backbone_state(m.state_dict(),ctx['provenance']['backbone_weights']['tensor_sha'])
    sig=digest(dict(dataset=SYN_REV,norm=ctx['normalization'],backbone=ctx['provenance']['backbone_weights'],variants=['original','target_removed','brightness_0.8','brightness_1.2']))
    cache=Cache(Path(ctx['options']['local'])/'pair_features'/sig,ctx['out']/'pair_features'/sig,sig)
    for n,p in enumerate(pairs):
        key=str(p['pair_id'])
        if cache.get(key) is not None:continue
        paths=[hf_hub_download('jokl/syncoco',p[k],repo_type='dataset',revision=SYN_REV,local_dir=root) for k in ['orig_image','syn_image']]
        images=[]
        for path in paths:
            with Image.open(path) as im:
                im=im.convert('RGB');images.append(data.transform_image(im,data.geometry(*im.size)))
        images+=[ImageEnhance.Brightness(images[0]).enhance(s) for s in [.8,1.2]]
        with torch.inference_mode():
            raw=m(**proc(images=images,do_resize=False,do_center_crop=False,return_tensors='pt').to(ctx['device'])).last_hidden_state[:,5:].float().cpu().numpy()
        if raw.shape!=(4,196,768):raise ValueError('Unexpected DINO feature layout')
        x=data.normalize(raw,ctx['norm']);cache.put(key,x=x,source_image_sha=np.array([sha(a) for a in paths]))
        if (n+1)%10==0:log('PAIR FEATURES',complete=n+1,total=len(pairs))
    # Supplement synthetic removals with an exact pixel-controlled background
    # swap. The foreground pixels and their position are identical in both
    # composites. This isolates context sensitivity without generative edits.
    pack=ctx['packs']['confirm'];cat_ids=sorted(coco.getCatIds());context=[]
    def rgb(image_id):
        info=coco.imgs[image_id];p=data.download(data.COCO_BUCKET+'train2017/'+info['file_name'],Path(ctx['options']['local'])/'images'/info['file_name'])
        with Image.open(p) as im:return np.array(data.transform_image(im.convert('RGB'),data.geometry(info['width'],info['height'])))
    for c in ctx['cats']:
        candidates=[i for i,a in enumerate(pack['masks']) if np.any(np.asarray(a['fraction'])[:,c]>=128)]
        candidates.sort(key=lambda i:digest(['context',c,int(pack['ids'][i])]))
        backgrounds=[i for i,a in enumerate(pack['masks']) if not np.any(np.asarray(a['fraction'])[:,c]>0)]
        backgrounds.sort(key=lambda i:digest(['background',c,int(pack['ids'][i])]))
        count=0
        for i in candidates:
            if count>=10 or len(backgrounds)<2:break
            image_id=int(pack['ids'][i]);info=coco.imgs[image_id]
            ann_ids=coco.getAnnIds(imgIds=[image_id],catIds=[cat_ids[c]],iscrowd=False)
            if not ann_ids:continue
            mask=np.maximum.reduce([coco.annToMask(coco.anns[j]) for j in ann_ids])
            mask=np.asarray(data.transform_image(Image.fromarray(mask),data.geometry(info['width'],info['height']),mask=True))>0
            if not .01<=mask.mean()<=.6:continue
            pair_id=9000000+c*100+count;count+=1
            p=dict(pair_id=pair_id,removed_cls_idx=c,orig_labels=str([c]),syn_labels='[]',dataset='pixel_controlled_background',source_image_id=image_id,background_ids=[int(pack['ids'][j]) for j in backgrounds[count-1:count+1]])
            # Cycle background indices if a category has few available negatives.
            p['background_ids']=[int(pack['ids'][backgrounds[(count-1+j)%len(backgrounds)]]) for j in range(2)]
            context.append(p)
            key=str(pair_id)
            if cache.get(key) is not None:continue
            fg=rgb(image_id);bg1,bg2=[rgb(a) for a in p['background_ids']]
            composite1,composite2=background_composites(fg,mask,bg1,bg2)
            assert np.array_equal(composite1[mask],composite2[mask])
            im=Image.fromarray(composite1)
            images=[im,Image.fromarray(bg1),ImageEnhance.Brightness(im).enhance(.8),ImageEnhance.Brightness(im).enhance(1.2),Image.fromarray(composite2),Image.fromarray(bg2)]
            with torch.inference_mode():raw=m(**proc(images=images,do_resize=False,do_center_crop=False,return_tensors='pt').to(ctx['device'])).last_hidden_state[:,5:].float().cpu().numpy()
            cache.put(key,x=data.normalize(raw,ctx['norm']))
        log('PIXEL CONTROLLED CONTEXT',category=c,objects=count)
    atomic_json(ctx['out']/'pixel_controlled_context_cohort.json',dict(pairs=context,construction='Union of target-category masks, fixed position and identical foreground pixels, two annotated target-absent background donors. Composite boundaries may be artificial. No physical-pose claim.',source_image_status='Existing confirmation images, separate from original training but previously inspected.'))
    for p in pairs:p['dataset']='syncoco_removal'
    pairs.extend(context)
    del m;gc.collect();torch.cuda.empty_cache();return cache

def run_controls(ctx,row,source,stores,pairs,features):
    path=ctx['out']/'controlled_consistency'/f"{row['kind']}_{row['seed']}.json"
    if read_done(path) is not None:return
    m,D,A,units=load(ctx,row,source,stores)
    # Discovery categories stay fixed. Native unit meaning is never renamed
    # after examining a removal. Include all native units in saved pair scores.
    discovery=read_done(ctx['out']/'discovery'/path.name)
    target_by_unit={q['unit']:q['category'] for q in discovery if q['assigned']}
    sig=digest(dict(protocol=ctx['signature'],weights=row['weights_sha'],dataset=SYN_REV))
    cache=Cache(Path(ctx['options']['local'])/'pair_scores'/sig,ctx['out']/'pair_scores'/sig,sig)
    results=[]
    example_pairs={}
    for p in pairs:example_pairs.setdefault((p.get('dataset','syncoco_removal'),int(p['removed_cls_idx'])),int(p['pair_id']))
    for i,p in enumerate(pairs):
        key=str(p['pair_id']);score=cache.get(key)
        if score is None:
            x=features.get(key)['x'];scores=[]
            for view in x:scores.append(observe(row['kind'],m,D,A,view,np.arange(196),ctx['device'])['score'])
            s=np.stack(scores).astype(np.float32)
            pooled=np.sort(s,axis=1)[:,-10:].mean(1)
            keep=example_pairs[(p.get('dataset','syncoco_removal'),int(p['removed_cls_idx']))]==int(p['pair_id'])
            score=cache.put(key,pooled=pooled,native_maps=s if keep else np.empty((0,),np.float32))
        pooled=score['pooled'] # fixed top-ten-patch mean
        orig_labels=set(ast.literal_eval(p['orig_labels']));syn_labels=set(ast.literal_eval(p['syn_labels']))
        removed=int(p['removed_cls_idx']);j=ctx['cats'].index(removed)
        unchanged=[a for a,c in enumerate(ctx['cats']) if c in orig_labels&syn_labels]
        delta=pooled[0]-pooled[1];appearance=np.mean(np.abs(pooled[2:4]-pooled[0]),axis=0)
        # Response scales are fixed on calibration features, never on pairs.
        scales=np.load(ctx['out']/'response_scales'/f"{row['kind']}_{row['seed']}.npy")
        delta=delta/scales;appearance=appearance/scales
        for g,c,scope in [(units[j],removed,'fixed_category_readout')]+[(g,c,'dictionary_discovered') for g,c in target_by_unit.items() if c==removed]:
            unrelated=[units[a] for a in unchanged]
            off=float(np.mean(np.abs(delta[unrelated]))) if unrelated else None
            results.append(dict(method=row['kind'],seed=row['seed'],unit=int(g),category=c,scope=scope,dataset=p.get('dataset','syncoco_removal'),pair_id=int(p['pair_id']),target_drop=float(delta[g]),target_decreased=bool(delta[g]>0),untouched_change=off,target_drop_above_untouched=float(delta[g]-off) if off is not None else None,appearance_change=float(appearance[g]),background_change=float(abs(pooled[4,g]-pooled[0,g])/scales[g]) if len(pooled)>4 else None,alternate_background_target_drop=float((pooled[4,g]-pooled[5,g])/scales[g]) if len(pooled)>4 else None))
        if (i+1)%20==0:log('CONTROLLED PAIRS',method=row['kind'],seed=row['seed'],complete=i+1,total=len(pairs))
    done_json(path,results);summarize(ctx,'controlled_consistency')
    del m;gc.collect();torch.cuda.empty_cache()

def main(options):
    import experiment as exp
    from coco_baselines.storage import identity
    from coco_coordinated import run as memory
    if not torch.cuda.is_available():raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ctx=exp.prepare(dict(options,output=str(Path(options['output'])/'verified_setup')))
    ctx['options']=options;ctx['cats']=list(ctx['cfg']['category_indices']);ctx['normalization']=identity(ctx['norm'])
    source,stores,rows=source_models(ctx,options)
    if len(ctx['cats'])!=75 or len(rows)!=15:raise ValueError('Expected 75 eligible categories and fifteen frozen models')
    a=ctx['packs']['select'];b=ctx['packs']['confirm']
    if set(a['ids'])&set(b['ids']):raise ValueError('Calibration/evaluation image overlap')
    protocol=dict(revision=options.get('analysis_identity',options['analysis_sha']),normalization=ctx['normalization'],models=[{k:r[k] for k in ['kind','seed','weights_sha']} for r in rows],calibration=a['identity'],evaluation=b['identity'],patches_per_image=32,reference_limit=256,query_limit=128,k=[10,30],concept_interface='native mask presence for response, true token contribution for geometry',dataset_revision=SYN_REV,geometry='exact decoded distances and separately unit-normalized contribution directions',selection='saved fixed category readouts; every native unit additionally named on calibration only',cohort_status='Previously inspected COCO development/confirmation images. Exploratory evidence, not new confirmation.',no_training=True)
    ctx['signature']=digest(protocol);ctx['out']=Path(options['output'])/'results'/ctx['signature'];ctx['out'].mkdir(parents=True,exist_ok=True)
    for name in ['guided_geometry','dictionary_geometry','discovery','controlled_consistency','response_scales']:(ctx['out']/name).mkdir(exist_ok=True)
    atomic_json(ctx['out']/'PROTOCOL.json',protocol)
    stamps=memory.source_snapshot([*ctx['endpoint_sources'],*stores.values()])
    rp=panel(a,ctx['cats']);tp=panel(b,ctx['cats'])
    # All fifteen fixed-category comparisons finish before the slower bank audit.
    for phase in ['guided','dictionary']:
        options['phase']=phase
        for seed in RUNS:
            for kind in METHODS:
                row=next(r for r in rows if r['kind']==kind and r['seed']==seed)
                run_geometry(ctx,row,source,stores,rp,tp)
        atomic_json(ctx['out']/f'{phase}_geometry_complete.json',dict(complete=True,models=15))
        log('STAGE COMPLETE',stage=phase,results=ctx['out'])
        memory.assert_immutable(stamps)
    root,pairs=syn_pairs(ctx);features=extract_pairs(ctx,root,pairs)
    for seed in RUNS:
        for kind in METHODS:
            row=next(r for r in rows if r['kind']==kind and r['seed']==seed)
            run_controls(ctx,row,source,stores,pairs,features);memory.assert_immutable(stamps)
    atomic_json(ctx['out']/'COMPLETE.json',dict(complete=True,geometry_models=15,controlled_models=15,no_training=True))
    atomic_json(options['receipt'],dict(complete=True,results=str(ctx['out'])))
    log('WHOLE NOTEBOOK COMPLETE',results=ctx['out'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(json.loads(Path(p.parse_args().config).read_text()))
