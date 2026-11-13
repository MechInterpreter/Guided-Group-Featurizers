"""Development-selected longer training for the pinned released Concept-SAE adapter.

The architecture, losses and evaluation are imported unchanged from the original
audited bundle. This is a new protocol, not continuation of the 10-epoch screen.
"""
import copy, gc, hashlib, json, signal, sys, time, zipfile, os
from pathlib import Path
import numpy as np
import torch
import concept_sae as native
from coco_adaptation.inputs import JournalStore
from coco_baselines import metrics, training
from coco_baselines.storage import identity, sha_file
from coco_ggf.drive_recovery import install_drive_recovery

SPEC = dict(revision='concept-sae-development-repair-v2', epochs=[30,50,30],
    tokenizer_LRs=[1e-4,3e-4,1e-3], aggregator_LR=1e-3, free_LR=1e-3,
    step_sizes=[20,30,None], gammas=[.1,.1,1.], effective_batch=64, microbatch=8,
    selection_seed=21, seeds=[21,22,23],
    selection='COCO seed21 tokenizer LR and epoch: maximum development native mask-times-presence macro AP among checkpoints with mask MSE below the all-zero baseline and AP above constant-score AP; ties prefer lower mask MSE, earlier epoch, then earlier LR. Freeze LR for remaining starts and VOC. Each tokenizer uses the same checkpoint rule. Stages2/3: minimum development reconstruction MSE.',
    gate='Before freezing any tokenizer, require mask MSE below all-zero MSE and native patch AP above constant-score AP. Failure saves diagnostics and stops before downstream fitting/held-out scoring; no failed start is dropped.',
    losses='Unchanged pinned RELEASED code: linear presence/mask, tokenizer merge L1=.01, aggregator KL=.01 at T=.02 and mean weight L1=1, free aggregation weight L1=1. No sigmoid, reweighting, focal loss or new heads.',
    paper_schedule='30/50/30 epochs and tokenizer/aggregator StepLR timing follow paper Appendix B; native released objective/output conventions retained where paper and code differ. Free stage LR constant because paper specifies no decay.',
    task='Normalized frozen DINO patches, human category masks/presence,14x14 grid,36 free tokens; explicit task adaptation, not native CLIP replication or matched BSF capacity.',
    evaluation='Unchanged original category-assigned contribution capture and native mask-times-presence patch AP. No readout selected on held-out scores.',
    scope='COCO held-out audit was previously inspected; VOC official validation, not official test. Development repair motivated by failed localization diagnostics, not independent confirmation.',
    no_time_limit=True)

class TokenizerQualityError(RuntimeError): pass

_CONTEXTS = {}
_STOP = False

def _signal_stop(signum, frame):
    global _STOP
    _STOP = True
    print('Concept-SAE graceful stop requested; finishing the current batch/validation and verified save.',flush=True)

def new_context(parent, cohort, selection=None):
    payload=dict(protocol=SPEC, code_sha256=sha_file(Path(__file__)),
        original_wrapper=parent['options']['wrapper_sha'], native_commit=native.COMMIT,
        native_source_sha256=native.SHA, cohort=cohort, parent_run=parent['store'].run_id,
        train=parent['packs']['train']['identity'], development=parent['packs']['select']['identity'],
        categories=parent['cfg']['category_indices'], normalization=identity(parent['norm']),
        versions=parent['store'].config['versions'], COCO_selection=selection)
    root=Path(parent['options']['output'])/'concept_sae_protocol_v2'
    local=parent['local']/'concept_sae_protocol_v2'
    s=JournalStore(root,payload,local/'rescue',parent['store'].require_drive)
    install_drive_recovery(s)
    ctx=dict(parent,store=s,local=local,active_key=None)
    ctx['local'].mkdir(parents=True,exist_ok=True)
    s.write('protocol.json',payload)
    s.check_mount();(s.results/'concept_sae_v2.py').write_bytes(Path(__file__).read_bytes())
    _CONTEXTS[cohort]=ctx
    return ctx

def constant_masks(pack,cats):
    # FIT only: per-category, per-position mean; no development fitting.
    total=np.zeros((len(cats),pack['x'].shape[1]),np.float64)
    count=np.zeros(pack['x'].shape[1],np.float64)
    for m in pack['masks']:
        known=np.asarray(m['crowd'])<=.1
        total+=np.asarray(m['fraction'])[:,cats].T/255*known
        count+=known
    return (total/np.maximum(count,1)[None,:]).astype(np.float32)

@torch.no_grad()
def tokenizer_diagnostics(m,pack,cats,device,cfg,constant,require_all=True):
    m.eval();mask_error=presence_error=zero_error=constant_error=0.;known_count=0.
    pred=[];targets=[];valids=[];presence=[];exists=[]
    per_error=np.zeros(len(cats));per_zero=np.zeros(len(cats))
    for lo in range(0,len(pack['ids']),8):
        idx=list(range(lo,min(lo+8,len(pack['ids']))))
        x,y,e,k=native.batch(pack,idx,cats,device)
        raw,score=m.concept_model.tokenizer(x)
        if not torch.isfinite(raw).all() or not torch.isfinite(score).all():
            raise FloatingPointError('Nonfinite tokenizer predictions')
        error=((raw-y).double().square()*k[:,None,:])
        mask_error+=float(error.sum());presence_error+=float((score-e).double().square().sum())
        ze=(y.double().square()*k[:,None,:]);zero_error+=float(ze.sum())
        base=torch.as_tensor(constant,device=device)
        constant_error+=float(((base-y).double().square()*k[:,None,:]).sum())
        known_count+=float(k.sum())*len(cats)
        per_error+=error.sum((0,2)).cpu().numpy();per_zero+=ze.sum((0,2)).cpu().numpy()
        for j,i in enumerate(idx):
            # Same fixed per-image patch sample and annotation rules as final AP.
            pick=np.random.default_rng(cfg['patch_sample_seed']+int(pack['ids'][i])).choice(raw.shape[-1],min(32,raw.shape[-1]),replace=False)
            f=np.asarray(pack['masks'][i]['fraction'])[pick][:,cats]/255
            cr=np.asarray(pack['masks'][i]['crowd'])[pick]
            pred.append((raw[j]*score[j])[:,pick].T.cpu().numpy())
            targets.append(f>=cfg['positive_fraction'])
            valids.append(((f>=cfg['positive_fraction'])|(f<=cfg['negative_fraction']))&(cr[:,None]<=cfg['negative_fraction']))
        presence.append(score.squeeze(-1).cpu().numpy());exists.append(e.squeeze(-1).cpu().numpy())
    pp=np.concatenate(pred);yy=np.concatenate(targets);vv=np.concatenate(valids)
    ap=[metrics.ap(pp[:,j],yy[:,j],vv[:,j]) for j in range(len(cats))]
    baseline=[metrics.ap(np.zeros(len(pp)),yy[:,j],vv[:,j]) for j in range(len(cats))]
    all_defined=all(v is not None for v in ap+baseline)
    if require_all and not all_defined:raise ValueError('Undefined development AP; category denominator may not be reduced')
    presence=np.concatenate(presence);exists=np.concatenate(exists)
    image_ap=[metrics.ap(presence[:,j],exists[:,j]>0) for j in range(len(cats))]
    out=dict(mask_MSE=mask_error/max(known_count,1),presence_MSE=presence_error/(len(pack['ids'])*len(cats)),
        zero_mask_MSE=zero_error/max(known_count,1),FIT_constant_mask_MSE=constant_error/max(known_count,1),
        macro_patch_AP=float(np.mean(ap)) if all_defined else None,constant_score_AP=float(np.mean(baseline)) if all_defined else None,
        AP_defined_categories=sum(v is not None for v in ap),AP_total_categories=len(cats),
        per_category_AP=ap,per_category_mask_MSE=(per_error/max(known_count/len(cats),1)).tolist(),
        per_category_zero_MSE=(per_zero/max(known_count/len(cats),1)).tolist(),
        per_category_image_AP=image_ap,negative_presence_fraction=float(np.mean(presence<0)))
    out['quality_pass']=quality_pass(out)
    return out

def quality_pass(d):
    keys=['mask_MSE','zero_mask_MSE','macro_patch_AP','constant_score_AP']
    return bool(all(d[k] is not None for k in keys) and np.isfinite([d[k] for k in keys]).all()
        and d['mask_MSE']<d['zero_mask_MSE'] and d['macro_patch_AP']>d['constant_score_AP'])

def quality_key(d):
    return (bool(d['quality_pass']),d['macro_patch_AP'],-d['mask_MSE'])

def stage_protocol(ctx,seed,stage,lr,parent_sha):
    return dict(spec=SPEC,code_sha256=ctx['store'].config['code_sha256'],seed=seed,stage=stage,lr=lr,
        train=ctx['packs']['train']['identity'],development=ctx['packs']['select']['identity'],parent_weights_sha=parent_sha)

def fit_stage(ctx,m,seed,stage,lr,key,parent_sha):
    """Resume completed Adam updates; checkpoint current AND selected weights."""
    s=ctx['store'];part=native.configure(m,stage)
    params=[p for p in part.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=lr)
    step=SPEC['step_sizes'][stage-1]
    sched=None if step is None else torch.optim.lr_scheduler.StepLR(opt,step_size=step,gamma=SPEC['gammas'][stage-1])
    signature=identity(stage_protocol(ctx,seed,stage,lr,parent_sha))
    n=len(ctx['packs']['train']['ids']);nb=(n+63)//64;epochs=SPEC['epochs'][stage-1];target=nb*epochs
    rng=np.random.default_rng(730000+seed+stage*10000)
    st=s.load(key)
    if st is not None:
        if st['signature']!=signature or st['target']!=target:raise ValueError('Revised Concept-SAE resume identity differs')
        part.load_state_dict(st['weights']);opt.load_state_dict(st['optimizer'])
        if sched is not None:sched.load_state_dict(st['scheduler'])
        rng.bit_generator.state=st['sample_rng'];torch.set_rng_state(st['torch_rng'])
        if str(ctx['device']).startswith('cuda'):torch.cuda.set_rng_state_all(st['cuda_rng'])
        print('RESUMED REVISED CONCEPT-SAE',key,st['updates'],'/',target,flush=True)
    else:
        st=dict(signature=signature,updates=0,target=target,permutation=None,history=[],best=None,
            best_diagnostics=None,best_loss=None,best_epoch=None,seconds=0.,epoch_sum=0.,epoch_images=0)
    constant=constant_masks(ctx['packs']['train'],ctx['cfg']['category_indices']) if stage==1 else None
    ctx['active_key']=key
    def save(drive):
        st.update(weights=training.weights(part),optimizer=copy.deepcopy(opt.state_dict()),
            scheduler=None if sched is None else copy.deepcopy(sched.state_dict()),
            sample_rng=copy.deepcopy(rng.bit_generator.state),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if str(ctx['device']).startswith('cuda') else [])
        s._width_drive_recovery_original_save(key,st,drive=drive)
        if drive:print('VERIFIED REVISED CONCEPT-SAE DRIVE SAVE',key,st['updates'],'/',target,flush=True)
    def gradient_check(g):
        if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite Concept-SAE gradient')
        return g
    hooks=[p.register_hook(gradient_check) for p in params]
    try:
        while st['updates']<target or len(st['history'])<epochs:
            # A stop is deferred out of Adam/validation/Drive writes. An epoch
            # boundary saved just before validation is processed on resume.
            if _STOP:save(True);raise KeyboardInterrupt('Concept-SAE stopped after a verified complete-batch save')
            u=st['updates']
            if u and u%nb==0 and len(st['history'])<u//nb:
                epoch=u//nb
                if stage==1:
                    diag=tokenizer_diagnostics(m,ctx['packs']['select'],ctx['cfg']['category_indices'],ctx['device'],ctx['cfg'],constant)
                    value=diag['mask_MSE']+diag['presence_MSE']
                    better=st['best_diagnostics'] is None or quality_key(diag)>quality_key(st['best_diagnostics'])
                else:
                    value=native.validation(m,ctx['packs']['select'],ctx['cfg']['category_indices'],ctx['device'],stage)
                    diag=dict(development_reconstruction_MSE=value)
                    better=st['best_loss'] is None or value<st['best_loss']
                if not np.isfinite(value):raise FloatingPointError('Nonfinite validation objective')
                if better:st.update(best=training.weights(part),best_loss=value,best_diagnostics=diag,best_epoch=epoch)
                if sched is not None:sched.step()
                row=dict(epoch=epoch,updates=u,development=diag,training_epoch_objective=st['epoch_sum']/max(st['epoch_images'],1),
                    next_learning_rate=opt.param_groups[0]['lr'])
                st['history'].append(row);st.update(epoch_sum=0.,epoch_images=0)
                # State is complete before an interrupt can be honored.
                save(True)
                s.write('concept_sae/'+key+'.json',{k:v for k,v in st.items() if k in ['history','updates','target','best_epoch','best_loss','best_diagnostics','seconds']})
                s.event('Concept_SAE_v2_epoch',seed=seed,native_stage=stage,epoch=epoch,total_epochs=epochs,tokenizer_LR=lr,
                    updates=u,total=target,development=value,mask_MSE=diag.get('mask_MSE'),
                    zero_mask_MSE=diag.get('zero_mask_MSE'),patch_AP=diag.get('macro_patch_AP'),quality_pass=diag.get('quality_pass'))
                continue
            epoch,bi=divmod(u,nb)
            if bi==0:st['permutation']=rng.permutation(n)
            idx=st['permutation'][bi*64:(bi+1)*64].tolist();m.train();started=time.monotonic()
            value=native.optimizer_step(m,part,opt,ctx['packs']['train'],idx,ctx['cfg']['category_indices'],ctx['device'],stage,SPEC['microbatch'])
            if str(ctx['device']).startswith('cuda'):torch.cuda.synchronize()
            st.update(updates=u+1,seconds=st['seconds']+time.monotonic()-started,
                epoch_sum=st['epoch_sum']+value*len(idx),epoch_images=st['epoch_images']+len(idx))
            if (u+1)%50==0:save(False)
            if (u+1)%10==0 or (u+1)%nb==0:
                s.event('Concept_SAE_v2_progress',seed=seed,native_stage=stage,update=u+1,total=target,
                    seconds_per_update=st['seconds']/(u+1),learning_rate=opt.param_groups[0]['lr'])
        if st['best'] is None:raise ValueError('No completed validation checkpoint')
        part.load_state_dict(st['best'])
        summary=dict(key=key,signature=signature,best_epoch=st['best_epoch'],best_loss=st['best_loss'],
            best_diagnostics=st['best_diagnostics'],epochs_completed=epochs,updates=target,
            weights_sha=training.weight_sha(training.weights(part)),seconds=st['seconds'],
            final_epoch_is_best=st['best_epoch']==epochs,
            convergence_note='Fixed schedule completed; boundary best or improving final trajectory is flagged, not proof of convergence.')
        s.write('stage_receipts/'+key+'.json',summary)
        return summary
    finally:
        for h in hooks:h.remove()
        ctx['active_key']=None
        del opt,sched,st;gc.collect();torch.cuda.empty_cache()

def make_model(ctx,seed):
    torch.manual_seed(seed)
    mod=native.source(Path(ctx['options']['concept_source']))
    return native.build(mod,len(ctx['cfg']['category_indices'])).to(ctx['device'])

def lr_key(lr):return 'lr'+str(round(lr*10**7))

def select_tokenizer_lr(ctx):
    s=ctx['store'];old=s.load('tokenizer_LR_selection')
    if old is not None:
        if old['spec_sha']!=identity(SPEC):raise ValueError('Frozen Concept-SAE LR rule changed')
        return old
    rows=[]
    for lr in SPEC['tokenizer_LRs']:
        m=make_model(ctx,21)
        try:
            r=fit_stage(ctx,m,21,1,lr,'tokenizer_search_s21_'+lr_key(lr),'native-seeded-initialization')
            rows.append(dict(lr=lr,**r))
        except FloatingPointError as exc:
            failed=dict(lr=lr,failed=True,error=str(exc),selection_eligible=False)
            rows.append(failed);s.write('failed_candidates/'+lr_key(lr)+'.json',failed)
            print('TOKENIZER LR CANDIDATE FAILED NUMERIC CHECKS',lr,str(exc),flush=True)
        finally:del m;gc.collect();torch.cuda.empty_cache()
    eligible=[r for r in rows if not r.get('failed',False) and r['best_diagnostics']['quality_pass']]
    receipt=dict(spec_sha=identity(SPEC),candidates=rows,uses='COCO TRAIN and development only',complete=True,
        selection_rule=SPEC['selection'],success=bool(eligible))
    if eligible:
        # max is stable: exact ties keep earlier LR; epoch tie keeps earlier epoch.
        winner=max(eligible,key=lambda r:quality_key(r['best_diagnostics']))
        receipt.update(lr=winner['lr'],selected_key=winner['key'])
        s.save('tokenizer_LR_selection',receipt)
    s.write('tokenizer_LR_selection.json',receipt)
    if not eligible:
        raise TokenizerQualityError('No revised tokenizer passed the predeclared development mask/AP sanity checks. All candidates are saved. No aggregator/free fitting or held-out score is accepted; inspect tokenizer_LR_selection.json.')
    print('FROZEN COCO DEVELOPMENT TOKENIZER LR',receipt['lr'],flush=True)
    return receipt

def fit(ctx,seed,selection):
    s=ctx['store'];key=f'concept_sae_final_s{seed}'
    protocol=dict(SPEC,native_commit=native.COMMIT,native_source_sha256=native.SHA,
        train=ctx['packs']['train']['identity'],development=ctx['packs']['select']['identity'],
        tokenizer_LR=selection['lr'],selection_sha=identity(selection),code_sha256=s.config['code_sha256'])
    signature=identity(protocol);old=s.load(key)
    m=make_model(ctx,seed)
    if old is not None:
        if old['signature']!=signature:raise ValueError('Revised final protocol changed')
        m.load_state_dict(old['weights']);print('REUSED REVISED CONCEPT-SAE FINAL',seed,flush=True);return m,old
    lr=selection['lr']
    stagekey='tokenizer_search_s21_'+lr_key(lr) if seed==21 and s.config['cohort']=='coco' else f'tokenizer_s{seed}_'+lr_key(lr)
    receipts=[fit_stage(ctx,m,seed,1,lr,stagekey,'native-seeded-initialization')]
    if not receipts[0]['best_diagnostics']['quality_pass']:
        s.write(f'quality_failure_s{seed}.json',receipts[0])
        del m;gc.collect();torch.cuda.empty_cache()
        raise TokenizerQualityError(f'Seed {seed} tokenizer failed development mask/AP checks; saved diagnostics, no downstream fit or held-out score. Do not drop this start.')
    parent=receipts[-1]['weights_sha']
    for stage,lr in [(2,SPEC['aggregator_LR']),(3,SPEC['free_LR'])]:
        r=fit_stage(ctx,m,seed,stage,lr,f'concept_sae_s{seed}_stage{stage}',parent)
        receipts.append(r);parent=identity([r['weights_sha'] for r in receipts])
    # Generalization diagnostics on a fixed TRAIN subset; these never select LR,
    # weights, thresholds or readouts. Development measurements above select.
    train=ctx['packs']['train'];idx=sorted(range(len(train['ids'])),key=lambda i:hashlib.sha256(str(train['ids'][i]).encode()).hexdigest())[:256]
    subset=dict(x=np.array(train['x'][idx]),ids=[train['ids'][i] for i in idx],masks=[train['masks'][i] for i in idx])
    diag=tokenizer_diagnostics(m,subset,ctx['cfg']['category_indices'],ctx['device'],ctx['cfg'],constant_masks(train,ctx['cfg']['category_indices']),require_all=False)
    s.write(f'tokenizer_TRAIN_diagnostic_s{seed}.json',diag)
    weights=training.weights(m)
    out=dict(signature=signature,weights=weights,weights_sha=training.weight_sha(weights),seed=seed,
        parameters=sum(p.numel() for p in m.parameters()),protocol=protocol,stage_receipts=receipts)
    s.save(key,out)
    return m,out

def plot_learning_curves(ctx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    any_curves=False
    for p in sorted((ctx['store'].results/'concept_sae').glob('*.json')):
        rows=json.loads(p.read_text()).get('history',[])
        if not rows:continue
        any_curves=True;x=[r['epoch'] for r in rows];name=p.stem.replace('concept_sae_','').replace('tokenizer_search_','').replace('tokenizer_','')
        if 'mask_MSE' in rows[0]['development']:
            axes[0].plot(x,[r['development']['mask_MSE'] for r in rows],label=name)
            axes[1].plot(x,[100*r['development']['macro_patch_AP'] for r in rows],label=name)
            axes[0].axhline(rows[0]['development']['zero_mask_MSE'],color='.5',linestyle=':',linewidth=.6)
            axes[1].axhline(100*rows[0]['development']['constant_score_AP'],color='.5',linestyle=':',linewidth=.6)
        else:axes[2].plot(x,[r['development']['development_reconstruction_MSE'] for r in rows],label=name)
    for ax,title in zip(axes,['Development mask MSE','Development native patch AP (%)','Development reconstruction MSE']):
        ax.set(xlabel='Epoch',ylabel=title);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
        if ax.lines:ax.legend(fontsize=6,ncol=2)
    if any_curves:
        fig.savefig(ctx['store'].results/'training_diagnostics.pdf')
        fig.savefig(ctx['store'].results/'training_diagnostics.png',dpi=180)
    plt.close(fig)

def archive(ctx,complete=False):
    s=ctx['store'];s.write('completion_status.json',dict(complete=complete,protocol='Concept-SAE v2',legacy_results_unchanged=True))
    plot_learning_curves(ctx)
    out=s.output/('concept_sae_v2_'+s.run_id[:12]+'_evidence.zip');tmp=out.with_suffix('.pending.zip')
    hashes={}
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(s.results.rglob('*')):
            if p.is_file():
                name=p.relative_to(s.results).as_posix();hashes[name]=sha_file(p);z.write(p,name)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None or any(hashlib.sha256(z.read(k)).hexdigest()!=v for k,v in hashes.items()):raise ValueError('Revised archive verification failed')
    os.replace(tmp,out);print('VERIFIED REVISED CONCEPT-SAE EVIDENCE ZIP',out,flush=True)
    return out

def run(parent,pack,cohort):
    global _STOP
    if cohort=='voc':
        if 'coco' not in _CONTEXTS:raise ValueError('Run the COCO development selection first')
        selection=_CONTEXTS['coco']['store'].load('tokenizer_LR_selection')
        if selection is None:raise ValueError('No frozen successful COCO tokenizer selection')
    else:selection=None
    ctx=new_context(parent,cohort,selection)
    _STOP=False;previous=signal.getsignal(signal.SIGINT);signal.signal(signal.SIGINT,_signal_stop)
    import reporting
    cost=reporting.cost(ctx,'Concept_SAE_v2_calibration_training_evaluation')
    cost.__enter__()
    try:
        if cohort=='coco':selection=select_tokenizer_lr(ctx)
        for seed in SPEC['seeds']:
            if _STOP:raise KeyboardInterrupt('Stopped between completed revised fits')
            m=None
            try:
                m,meta=fit(ctx,seed,selection)
                if _STOP:raise KeyboardInterrupt('Stopped before held-out evaluation')
                result=native.evaluate(ctx,seed,m,meta,pack,cohort)
                result['report'].update(training_protocol=meta['protocol'],stage_receipts=meta['stage_receipts'])
                ctx['store'].save(f'concept_sae_{cohort}_s{seed}',result)
                ctx['store'].write(f'concept_sae/concept_sae_{cohort}_s{seed}.json',result)
                archive(ctx)
            finally:del m;gc.collect();torch.cuda.empty_cache()
        ctx['store'].save('revised_concept_complete',dict(complete=True,seeds=SPEC['seeds'],selection_sha=identity(selection)))
        parent['concept_v2_receipt']=dict(results=str(ctx['store'].results),archive=str(archive(ctx,True)))
    finally:
        try:
            cost.__exit__(*sys.exc_info())
            # Keep diagnostic evidence on a quality failure or graceful stop.
            archive(ctx,ctx['store'].load('revised_concept_complete') is not None)
        finally:signal.signal(signal.SIGINT,previous)

class ReportView:
    """Read old BSF evidence, new Concept evidence; write all new reports in v2."""
    def __init__(self,base,new):self.base,self.new=base,new
    def __getattr__(self,k):return getattr(self.new,k)
    def load(self,key):
        if key.startswith('concept_sae_'):return self.new.load(key)
        value=self.new.load(key)
        return value if value is not None else self.base.load(key)
    def save(self,*a,**kw):return self.new.save(*a,**kw)

def revised_checklist(parent,cohort,complete=False):
    import reporting
    # Do not overwrite legacy summary/plot files or combine old and new fits.
    if cohort not in _CONTEXTS:return
    ctx=_CONTEXTS[cohort]
    view=dict(parent,store=ReportView(parent['store'],ctx['store']))
    reporting._pre_v2_checklist(view,cohort,complete)
    # Atlas image files intentionally remain in the original BSF result store.
    coverage=json.loads((ctx['store'].results/'checklist_coverage.json').read_text())
    wanted={'person','potted plant','vase','fork','spoon','pizza'}
    expected=3*sum(parent['provenance']['categories'][c]['name'] in wanted for c in parent['cfg']['category_indices'])
    images=[p for k in reporting.PRIMARY for p in (parent['store'].results/'atlas').glob(cohort+'_'+k+'_*.png') if not p.stem.endswith('_cloud')]
    for item in coverage['items']:
        if item['item']=='Direction-level atlas across matched BSF models':
            item.update(images_completed=len(images),images_expected=expected,evidence=str(parent['store'].results/'atlas/index.html'),
                status='completed (saved BSF atlas reused)' if expected>0 and len(images)==expected else 'partial/pending or no eligible requested atlas category')
    ctx['store'].write('checklist_coverage.json',coverage)
    ctx['store'].write('comparison_provenance.json',dict(BSF_run=parent['store'].run_id,
        BSF_results=str(parent['store'].results),Concept_SAE_run=ctx['store'].run_id,
        Concept_SAE_protocol=SPEC,legacy_10epoch_Concept_SAE_excluded=True))
    done=ctx['store'].load('revised_concept_complete') is not None
    parent['concept_v2_receipt']=dict(results=str(ctx['store'].results),archive=str(archive(ctx,done)))

def install_reporting():
    import reporting
    if not hasattr(reporting,'_pre_v2_checklist'):
        reporting._pre_v2_checklist=reporting.checklist
        reporting.checklist=revised_checklist
