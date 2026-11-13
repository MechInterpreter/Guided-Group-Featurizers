"""Direct controlled-property tests and anonymous review materials."""
import csv,hashlib,json,secrets,zipfile
from pathlib import Path
import numpy as np
from cv_math import VARIANTS,LEVELS,project,digest,render
from semantic_math import FOREGROUND,average_precision,trajectory_consistency,paired_summary,holm

def numeric(row,summaries,D,plans,out):
    """Evaluation only; consumes frozen TRAIN choices, never selects a group."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    categories=sorted({r['category'] for r,s in summaries})
    evaluation=[(r,s) for r,s in summaries if r['split']=='eval']
    ids=np.array([r['image_id'] for r,s in evaluation]);truth=np.array([r['category'] for r,s in evaluation])
    if len(np.unique(ids))!=len(ids):raise ValueError('One instance per base image required')
    scores=np.zeros((len(evaluation),len(VARIANTS),len(categories)))
    rows=[];conditions=[]
    for ci,c in enumerate(categories):
        g=plans[c]['groups'][0]
        for i,(rec,s) in enumerate(evaluation):
            index=np.flatnonzero(s['gids']==g)
            if len(index)!=1:raise ValueError('Missing frozen cross-category group')
            scores[i,:,ci]=s['norm'][:,index[0]]
        selected=[(r,s) for r,s in evaluation if r['category']==c]
        own=[(r,s) for r,s in summaries if r['split']=='train' and r['category']==c]
        training_scale=max(float(np.sqrt(np.mean([s['norm_rms'][:,np.flatnonzero(s['gids']==g)[0]]**2 for r,s in own]))),1e-10)
        aps=[];cons=[];deg=[];rms=[]
        for a in FOREGROUND:
            ix=[j for j,(att,v) in enumerate(VARIANTS) if att==a]
            for j in ix:
                ap=average_precision(truth==c,scores[:,j,ci]);aps.append(ap)
                conditions.append(dict(model=row['kind'],seed=row['seed'],category=c,attribute=a,level=VARIANTS[j][1],category_AP=ap))
            changes=[]
            for rec,s in selected:
                zi=np.flatnonzero(s['gids']==g)[0];z=s['pooled'][ix,zi]
                changes.append(np.delete(z-z[2],2,axis=0))
            changes=np.stack(changes)
            metric=trajectory_consistency(changes,D[g],[r['image_id'] for r,s in selected])
            cons.append(metric['consistency']);deg.append(metric['zero_change_fraction']);rms.append(metric['decoded_change_RMS'])
            conditions.append(dict(model=row['kind'],seed=row['seed'],category=c,attribute=a,**metric))
        bgix=[j for j,(a,v) in enumerate(VARIANTS) if a=='background']
        bg=np.stack([s['norm'][bgix,np.flatnonzero(s['gids']==g)[0]] for r,s in selected])
        bg_sensitivity=float(np.mean(abs(bg-bg[:,2:3]))/training_scale)
        rows.append(dict(model=row['kind'],seed=row['seed'],category=c,group=g,
            category_name=selected[0][0]['category_name'],identity_AP=average_precision(truth==c,scores[:,2,ci]),
            mean_condition_AP=float(np.mean(aps)),worst_condition_AP=float(min(aps)),
            trajectory_consistency=float(np.mean(cons)),zero_change_fraction=float(np.mean(deg)),
            decoded_change_RMS=float(np.mean(rms)),background_norm_sensitivity=bg_sensitivity,
            evaluation_objects=len(selected),comparison_objects=len(evaluation)-len(selected)))
    stem=row['kind']+'_'+str(row['seed'])
    np.savez_compressed(out/(stem+'_category_responses.npz'),scores=scores,image_ids=ids,categories=categories,true_category=truth)
    (out/(stem+'_records.json')).write_text(json.dumps([r for r,s in evaluation],indent=2))
    result=dict(summary=rows,conditions=conditions)
    (out/(stem+'_metrics.json')).write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

def make_cards(ctx,row,m,summaries,pcas,D,plans,features,coco):
    """Complete panel inventory, no per-method display curation or judgment."""
    import torch
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    from cv_study import source_crop,record_key
    root=ctx['out']/'semantic_coherence';private=root/'private';public=root/'blinded_review'
    private.mkdir(parents=True,exist_ok=True);public.mkdir(parents=True,exist_ok=True)
    keypath=private/'review_key.json'
    key=json.loads(keypath.read_text()) if keypath.exists() else {}
    identity=2
    evaluation=[(r,s) for r,s in summaries if r['split']=='eval']
    for c in sorted(plans):
        matching=[(r,s) for r,s in evaluation if r['category']==c]
        if not matching:continue
        g=plans[c]['groups'][0];pc=pcas[(c,g)]
        token=row['kind']+'_'+str(row['seed'])+'_'+str(c)
        if token not in key:
            key[token]=dict(card_id=secrets.token_hex(6),model=row['kind'],seed=int(row['seed']),category=int(c),group=int(g))
            temporary=keypath.with_suffix('.pending');temporary.write_text(json.dumps(key,indent=2));temporary.replace(keypath)
        card=key[token]['card_id'];path=public/(card+'.png')
        variation_path=public/(card+'_variation.png')
        if path.exists() and variation_path.exists():
            from PIL import Image
            try:
                for p in [path,variation_path]:
                    with Image.open(p) as im:im.verify()
                continue
            except (OSError,ValueError):pass
        # Distinct images; order ties by predeclared stable image hash.
        scored=sorted(evaluation,key=lambda item:(-float(item[1]['norm'][identity,np.flatnonzero(item[1]['gids']==g)[0]]),digest(item[0]['image_id'])))
        top=scored[:8]
        train_norm=np.concatenate([s['norm_rms'][:,np.flatnonzero(s['gids']==g)[0]] for r,s in summaries if r['split']=='train'])
        limit=max(float(np.quantile(train_norm,.99)),1e-8)
        fig,ax=plt.subplots(2,8,figsize=(14,4.1))
        for j,(rec,s) in enumerate(top):
            rgb,mask=source_crop(ctx,coco,rec);image,fr=render(rgb,mask,*VARIANTS[identity])
            f=features.get(record_key(rec))
            with torch.inference_mode():z=m.encode(torch.as_tensor(f['x'][identity].copy(),device=ctx['device'])).cpu().numpy()[:,g]
            norm=np.linalg.norm(z,axis=1);alpha=zoom(np.clip(norm/limit,0,.85).reshape(14,14),16,order=1)
            base=np.asarray(image)/255
            ax[0,j].imshow(image);ax[1,j].imshow(base*(1-alpha[:,:,None])+np.array([0.,.58,.53])*alpha[:,:,None])
        for a in ax.flat:a.axis('off')
        ax[0,0].set_ylabel('Input');ax[1,0].set_ylabel('Group response')
        fig.suptitle('Card '+card+': eight strongest distinct object views',fontsize=12)
        fig.tight_layout();pending=path.with_suffix('.pending.png');fig.savefig(pending,dpi=140);pending.replace(path);plt.close(fig)
        # Show native contribution PCA poles on identical evaluation pools.
        # Extreme examples are descriptive, never used to choose a group/axis.
        pools=[]
        for rec,s in evaluation:
            z=s['pooled'][identity,np.flatnonzero(s['gids']==g)[0]]
            pools.append(project(z[None],D[g],pc)[0] if pc['valid'] else np.zeros(3))
        pools=np.stack(pools);fig,ax=plt.subplots(3,8,figsize=(14,6.4))
        for axis in range(3):
            order=sorted(range(len(evaluation)),key=lambda j:(float(pools[j,axis]),digest(evaluation[j][0]['image_id'])))
            indices=order[:4]+order[-4:]
            for j,index in enumerate(indices):
                rec,s=evaluation[index];rgb,mask=source_crop(ctx,coco,rec);image,_=render(rgb,mask,*VARIANTS[identity]);ax[axis,j].imshow(image);ax[axis,j].axis('off')
                if j in [0,4]:ax[axis,j].set_title(('Low' if j==0 else 'High')+' PCA '+str(axis+1),fontsize=9)
        fig.suptitle('Card '+card+': contribution PCA poles (TRAIN axes)'+(' | inactive PCA' if not pc['valid'] else ''),fontsize=12)
        fig.tight_layout();pending=variation_path.with_suffix('.pending.png');fig.savefig(pending,dpi=140);pending.replace(variation_path);plt.close(fig)

def finish(ctx,results):
    import pandas as pd
    root=ctx['out']/'semantic_coherence';root.mkdir(exist_ok=True)
    summary=pd.DataFrame([r for result in results for r in result['summary']])
    summary.to_csv(root/'category_metrics.csv',index=False)
    pd.DataFrame([r for result in results for r in result['conditions']]).to_csv(root/'condition_metrics.csv',index=False)
    metrics=['worst_condition_AP','trajectory_consistency','identity_AP','mean_condition_AP','zero_change_fraction','decoded_change_RMS','background_norm_sensitivity']
    per=summary.groupby(['model','seed'])[metrics].mean().reset_index();per.to_csv(root/'per_start_metrics.csv',index=False)
    per.groupby('model')[metrics].agg(['mean','std','count']).to_csv(root/'mean_SD.csv')
    comparisons=[]
    for metric in ['worst_condition_AP','trajectory_consistency']:
        a=per[per.model=='agf_g'].set_index('seed')[metric].sort_index()
        for control in [m for m in ['supervised_vanilla','released_vanilla','slics','concept_sae'] if m in set(per.model)]:
            b=per[per.model==control].set_index('seed')[metric].sort_index()
            if list(a.index)!=list(b.index):raise ValueError('Unpaired models')
            comparisons.append(dict(metric=metric,control=control,**paired_summary(a.values,b.values)))
    for q,p in zip(comparisons,holm([q['one_sided_p'] for q in comparisons])):q['Holm_p']=float(p)
    pd.DataFrame(comparisons).to_csv(root/'paired_primary_comparisons.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=[m for m in ['released_vanilla','supervised_vanilla','slics','concept_sae','agf_g'] if m in set(per.model)]
    names=dict(released_vanilla='Vanilla',supervised_vanilla='Supervised\nVanilla',slics='SLiCS',concept_sae='Concept-SAE',agf_g='GGF')
    palette=dict(released_vanilla='#89969b',supervised_vanilla='#be8b29',slics='#567ba4',concept_sae='#7b5aa6',agf_g='#008b80')
    labels=[names[m] for m in methods];colors=[palette[m] for m in methods]
    fig,axs=plt.subplots(1,2,figsize=(9,2.8))
    for ax,metric,title,mult in zip(axs,['worst_condition_AP','trajectory_consistency'],['Worst-condition object AP (%)','Decoded-trajectory consistency'],[100,1]):
        for i,(method,color) in enumerate(zip(methods,colors)):
            values=per[per.model==method][metric].to_numpy()*mult
            ax.bar(i,values.mean(),yerr=values.std(ddof=1),color=color,width=.65,capsize=3)
            ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=12,c='black',zorder=3)
        ax.set_xticks(range(len(methods)),labels,fontsize=8);ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Controlled diagnostics: means and sample SD; points are trained models',fontsize=10);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(root/('coherence_diagnostics.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    key=json.loads((root/'private/review_key.json').read_text());cards=[v['card_id'] for v in key.values()]
    expected=len(per)*len(summary.category.unique())
    if len(cards)!=expected or len(set(cards))!=expected:raise ValueError('Incomplete blind inventory')
    public=root/'blinded_review'
    fields=['card_id','common_concept_explanation','group_coherence_1_to_5','variation_explanation','variation_coherence_1_to_5','confidence_1_to_3']
    for rater in ['A','B']:
        p=public/('rater_'+rater+'_blank.csv')
        if not p.exists():
            order=sorted(cards,key=lambda c:hashlib.sha256((rater+c).encode()).hexdigest())
            with p.open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fields);w.writeheader();w.writerows(dict(card_id=c) for c in order)
    rubric='''# Anonymous group review

Two reviewers independently rate every card before opening any method key.
Keep the private folder away from reviewers. Do not discuss scores until
both sheets are locked. Compare all cards, including inactive/poor groups.

The same six-category object pool is used throughout. A feature unit can
be a native concept dictionary, a category token or a BSF block; the
review does not assume that their architectures are identical. A shared shape,
texture, material, part, or other intelligible property across categories
can be coherent. Category purity alone is not the definition of coherence.
Teal marks native response above its TRAIN foreground mean and purple
below that mean, with the same clipped TRAIN-standard-deviation display
rule for every unit. These are not probabilities or comparable absolute
magnitudes. Primary metrics retain the unmodified native scores. Top
responses are selected mechanically from distinct images. PCA uses
TRAIN pooled native-output vectors, with three coordinates for every
method. Signs have no shared meaning across cards. Method names are
hidden, but response patterns may allow a knowledgeable rater to guess;
this is identity blinding, not guaranteed architectural indistinguishability.

For group_coherence, describe the common explanation, then rate 1=no
consistent interpretable pattern, 2=weak/mostly mixed, 3=partly coherent,
4=clear shared pattern with exceptions, 5=clear pattern across nearly all
examples. For variation_coherence, use the three PCA-pole rows: rate
whether opposite ends express a repeatable interpretable change, with
the same 1-5 anchors. Inactive/uninterpretable groups receive 1, not an
omitted score. Confidence is 1=low, 2=medium, 3=high; it does not filter
the primary analysis. Give a short explanation for each judgment.

These are predefined human judgments, not ground-truth semantic labels.
Report agreement, all scores and failures, and sensitivity to individual
reviewers. Do not infer recursive generalization from this review.
'''
    (public/'README.md').write_text(rubric)
    links='\n'.join('<article><h2>'+c+'</h2><img src="'+c+'.png"><img src="'+c+'_variation.png"></article>' for c in sorted(cards))
    (public/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Anonymous group review</title><style>body{font:16px sans-serif;margin:24px;background:#f7f7f5}img{width:100%;max-width:1500px}article{margin-bottom:48px}</style><h1>Anonymous group review</h1><p>Read README.md first. Complete your independent rating sheet.</p>'+links)
    with zipfile.ZipFile(root/'blinded_review_public.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in public.iterdir():z.write(p,p.name)
    report='''# Semantic-coherence study: interpretation

The automated tests operationalize two narrower properties: a fixed group
recognizes its annotated object proxy across controlled conditions, and
the same controlled change produces consistent decoded displacement
across different instances. They do not establish broad semantic purity.
Worst-condition AP uses only the six predeclared categories as negatives.
Pairwise cosine is basis invariant; tiny/zero changes contribute zero
and remain in the reported inactivity rate. Nonlinear but meaningful
changes can have low cosine, so it is not a universal manifold score.
Category means are equally weighted within a training start. Primary
comparisons are paired across starts, with two metrics by four controls
and Holm correction. These intervals are limited by the small number
of trained models; object populations and semantic concepts are broader.

The shared five-method analysis uses native responses and output-vector
variation without converting tokens into BSF codes. SLiCS uses independent
category projections; these are not summed as a joint reconstruction or
subtracted as if they were joint additive components. The existing
subset/intervention and physical-lighting analyses remain BSF-family
diagnostics and do not rank the external methods on those tasks. Higher rank,
higher Capture, larger probe effects or a single favorable heatmap do
not by themselves prove semantic coherence. Background sensitivity and
off-target intervention changes remain visible alongside positive results.

The blinded review is required for the broader semantic-coherence claim.
Automated completion means the review materials are ready, not that human
ratings support GGF. Run the CPU scorer after both complete sheets arrive.
The sampled object-associated groups and previously inspected COCO images
limit the claim; this is not independent confirmation on every group.

Precedents: Fel et al., arXiv:2606.25234, for decoded-contribution PCA and
controlled lighting; Bau et al., CVPR 2017, arXiv:1704.05796, for alignment
to semantic annotations. The exact consistency metrics and review rubric
are this study's operational definitions, not established universal scores.
'''
    (root/'README.md').write_text(report)
    (root/'completion.json').write_text(json.dumps(dict(automated_complete=True,human_ratings_complete=False,cards=expected,primary_comparisons=comparisons,scope='Six predeclared object-proxy groups; broad semantic coherence requires completed independent review'),indent=2))
