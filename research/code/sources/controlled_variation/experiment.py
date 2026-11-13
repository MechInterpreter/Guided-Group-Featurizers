"""Frozen 1024-group, three-seed supervised control and external VOC evaluation."""
import argparse,copy,gc,hashlib,json,os,signal,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity,sha_file
from coco_adaptation import inputs
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched
from coco_ggf.drive_recovery import install_drive_recovery
import scale_data,scale_run
from scale_fit import fit
from scale_metrics import train_targets
from scale_models import model,BASELINE_AUDIT

CORE_SHA='8a29854a50c94420686f9b904f7eaecd88f9324bcf2b016c1ce6ad4057aa0b9c'
SOURCE_RUN='bd3eb87f9fb5ffc0a41963b4e8154c881e32b96cb44b8c40e8730665a72fff32'
SEEDS=[21,22,23]
PRIMARY=['agf_g','released_vanilla','supervised_vanilla']
EXTRA=['released_grassmannian','released_group_lasso','paper_grassmannian','paper_group_lasso']
G,K,B,WARM,GUIDED,CAP=1024,16,12,7640,3000,.40
LABELS=dict(agf_g='GGF',released_vanilla='Vanilla BSF',supervised_vanilla='Supervised Vanilla',
    released_grassmannian='Grassmannian (released)',released_group_lasso='Group Lasso (released)',
    paper_grassmannian='Grassmannian (paper equations)',paper_group_lasso='Group Lasso (paper equations)')
PINS={
 'g1024_k16_s21_mask_lr3000_p10000_agf_g_c040':'19a482d4fd65080c409e71bc12bbec35ba56e7a0460bc401d85ba216801045f5',
 'g1024_k16_s21_released_vanilla_lr3000_p10000_released_vanilla':'5cdab92b39a78217ad87a0c607b5803d43e87468e8d255cf35ba6cb5efcc482f',
 'g1024_k16_s22_mask_lr3000_p10000_agf_g_c040':'2d4c780f107bce3c35ec7e134631525b857ba408e0f36c13b2739f6e233b4a46'}

def recipe():
    return dict(version=2,seeds=SEEDS,groups=G,width=B,top_k=K,cap=CAP,warm_updates=WARM,guided_updates=GUIDED,
        total_updates=WARM+GUIDED,lr=.0003,methods=PRIMARY+EXTRA,
        supervised_vanilla='Random warm-up; contrast+detection; same scalar heads; fixed loss pair selected on seed21 development; no controller or mask initialization',
        fixed_weight_candidates=[[.05,.01],[.20,.20],[.40,.40]],
        extra_variants='Both released and paper-equation unsupervised adapters, COCO supporting stage',
        train_mapping='Same TRAIN-only group selection policy after reconstruction warm-up; category slots not hardwired into supervision',
        endpoints='Fixed 10640 updates; fixed-weight hyperparameter selection on development only, not endpoint cherry-picking',
        baseline_settings='Three full-budget fixed-weight candidates; extra unsupervised variants at common Adam LR0.0003 and paper lasso coefficient0.001, NOT tuned or native ImageNet replication',
        evaluation='Previously inspected COCO1500 audit; independently trained VOC2012 models evaluated on official validation',
        voc='Official train partitioned FIT/development by fixed hash; own FIT-only normalization, mask initialization and mappings; COCO architecture/cap/update defaults; no GGF VOC sweep',
        interpretability='TRAIN-selected 1/2/4 native contribution groups; held-out capture, same-group AP, fixed linear-probe removal with activity/RMS-neighbor random controls; active coefficient/contribution rank and direction atlas',
        additional_supervised='Pinned released Concept-SAE task adaptation, all three stages/three seeds on COCO and VOC; separate native budget/parameter count',
        uncertainty='Seed SD and Student-t95; paired whole-image bootstrap1000 and seed differences; retain all declared categories',
        order='COCO primary+analysis, VOC training+analysis, Concept-SAE, supporting unsupervised variants',
        no_exchanges=True,no_capacity_sweep=True,no_time_limit=True)

class ReuseStore:
    """Only identical 1024 computations read through; original source never written."""
    def __init__(self,job,source,stamps):self.job,self.source,self.stamps=job,source,stamps;self.seen=set()
    def __getattr__(self,key):return getattr(self.job,key)
    @staticmethod
    def reusable(key):
        import re
        return bool(re.fullmatch(r'(?:dev_)?g1024_k16_s(?:21|22|23)_(?:mask|released_vanilla|released_grassmannian|released_group_lasso|paper_grassmannian|paper_group_lasso)_[A-Za-z0-9_]+',key)
            or re.fullmatch(r'initializer_cat[0-9]+',key) or key=='mask_dictionary12') and 'supervised_vanilla' not in key
    def load(self,key):
        value=self.job.load(key)
        if value is None and self.reusable(key):
            value=self.source.load(key)
            if value is not None and key not in self.seen:
                self.stamps.append(matched.source_key_stamp(self.source,key));self.seen.add(key)
                print('SOURCE REUSE',key,'updates',value.get('updates','n/a') if isinstance(value,dict) else 'targets',flush=True)
                for section in ['histories','targets','development']:
                    path=self.source.results/section/(key+'.json')
                    if path.exists():self.job.write(section+'/'+path.name,json.loads(path.read_text()))
        if isinstance(value,dict) and 'updates' in value:
            limit=WARM if key.endswith('_warm') else GUIDED
            if value['updates']>limit:raise ValueError('Source exceeds this fixed endpoint; cannot substitute longer training: '+key)
        if value is not None and key in PINS:
            if value['updates']!=GUIDED or training.weight_sha(value['weights'])!=PINS[key]:raise ValueError('Pinned completed reference changed: '+key)
        return value
    def save(self,key,value,*args,**kwargs):
        if key in PINS:raise ValueError('Cannot overwrite a pinned completed source endpoint')
        return self.job.save(key,value,*args,**kwargs)
    @property
    def _width_drive_recovery_original_save(self):return self.force_save
    def force_save(self,key,value,*args,**kwargs):
        if key in PINS:raise ValueError('Cannot overwrite a pinned completed source endpoint')
        return self.job._width_drive_recovery_original_save(key,value,*args,**kwargs)

def prepare(options,require_drive=True):
    if options['source_sha']!=CORE_SHA:raise ValueError('Wrong pinned core')
    from coco_ggf.twelve_capacity import cache
    cache.safe_destination(options['output'],[options['source_output'],*options['paths'].values()])
    ctx=scale_data.prepare(dict(options,output=str(Path(options['output'])/'verified_inputs')),require_drive)
    source=matched.discover(options['source_output'],Path(options['local'])/'read_scale',run=SOURCE_RUN)
    if source.config['source_sha']!=CORE_SHA or source.config['versions']!=matched.versions():raise ValueError('Source code or numerical environment differs')
    for key,value in [('recipe',ctx['rule']),('normalization_sha',identity(ctx['norm'])),
                      ('training_identity',ctx['packs']['train']['identity']),('selection_identity',ctx['packs']['select']['identity'])]:
        if identity(source.config[key])!=identity(value):raise ValueError('Source lineage mismatch: '+key)
    payload=dict(checklist_revision='2026-10-19-fixed-supervision-VOC-training-consolidation',study='ggf_checklist_evidence_v1',protocol=recipe(),wrapper_sha=options['wrapper_sha'],source_sha=CORE_SHA,
        source_run=SOURCE_RUN,versions=matched.versions(),normalization_sha=identity(ctx['norm']),
        training_identity=ctx['packs']['train']['identity'],selection_identity=ctx['packs']['select']['identity'])
    job=inputs.JournalStore(options['output'],payload,Path(options['local'])/'rescue',require_drive);install_drive_recovery(job)
    ctx['sources']['scale']=source;ctx['stamps']+=memory.source_snapshot([source])
    ctx['store']=ReuseStore(job,source,ctx['checkpoint_stamps']);ctx['options']=options
    # Preserve the pinned core's rule verbatim: it participates in old fit fingerprints.
    # Its historical target=12000 is NEVER reached; every call is capped at3000.
    ctx['cfg'].update(bootstrap=1000,fresh_test_seed=126101,preview_images=0)
    audit=dict(BASELINE_AUDIT,paper_group_lasso='Soft block shrinkage, learned thresholds, excess-activity L2,1 and AuxK. Unspecified penalty details are explicit adapters; coefficient0.001 is FIXED, not tuned in this job.')
    ctx['store'].write('protocol.json',dict(payload,implementation_differences=audit))
    for key in PINS:
        value=ctx['store'].load(key)
        if value is None:raise FileNotFoundError('Restore completed source reference: '+key)
        del value;gc.collect()
    return ctx

def supervised_vanilla(ctx,seed):
    from supervised_control import endpoint as fixed_endpoint
    return fixed_endpoint(ctx,seed)


def validate_row(row):
    if (row['seed'] not in SEEDS or row['kind'] not in PRIMARY+EXTRA or
        (row['groups'],row['rank'],row['active_groups'],row['additional_updates'],row['total_training_updates'])!=(G,B,K,GUIDED,WARM+GUIDED)):
        raise ValueError('Wrong model endpoint or budget')
    if row['targets_sha']!=identity(row['targets']):raise ValueError('Readout changed')
    if row['kind']=='agf_g' and row['cap']!=CAP:raise ValueError('Wrong guided cap')
    if row['kind']=='supervised_vanilla' and row['cap'] is not None:raise ValueError('Fixed supervision has no adaptive cap')
    return row

def check_pairing(ctx,rows):
    # Compact proofs are cached only within this single worker. Resume revalidates
    # checkpoint payloads. Avoid rereading both large Adam slots at every freeze.
    proofs=ctx.setdefault('_pairing_proofs',{})
    def proof(key):
        if key not in proofs:
            st=memory.required(ctx['store'],key)
            proofs[key]=dict(updates=st['updates'],weights_sha=training.weight_sha(st['weights']),
                streams=(st['sampled_batches_sha'],st['sampled_noise_sha']),
                head_shape=None if st['head'] is None else tuple(st['head'].shape),balance_present=st['balance'] is not None)
            del st;gc.collect()
        return proofs[key]
    for seed in sorted({r['seed'] for r in rows}):
        selected=[r for r in rows if r['seed']==seed];streams=set();warm_streams=set();warm_hashes={}
        for row in selected:
            validate_row(row);st=proof(row['key'])
            if st['updates']!=GUIDED or st['weights_sha']!=row['weights_sha']:raise ValueError('Endpoint weight mismatch')
            streams.add(st['streams'])
            supervised=row['kind'] in ['agf_g','supervised_vanilla']
            if supervised and (st['head_shape']!=(len(ctx['cfg']['category_indices']),2) or st['balance_present']!=(row['kind']=='agf_g')):raise ValueError('Missing classifier/controller state')
            if not supervised and (st['head_shape'] is not None or st['balance_present']):raise ValueError('Unsupervised baseline acquired heads')
            warmkey=f'g1024_k16_s{seed}_'+('mask' if row['kind']=='agf_g' else row['method'])+'_warm'
            if warmkey not in warm_hashes:
                warm=proof(warmkey)
                if warm['updates']!=WARM:raise ValueError('Warm-up budget differs')
                warm_streams.add(warm['streams'])
                warm_hashes[warmkey]=warm['weights_sha']
        if len(streams)!=1:raise ValueError('Within-seed continuation streams differ')
        if len(warm_streams)!=1:raise ValueError('Within-seed warm-up streams differ')
        sup=next((r for r in selected if r['kind']=='supervised_vanilla'),None)
        if sup:
            vanilla=next(r for r in selected if r['kind']=='released_vanilla')
            if sup['targets_sha']!=vanilla['targets_sha']:raise ValueError('Random warm-up branch readouts differ')
            if sup['warm_weights_sha']!=warm_hashes[sup['warm_key']]:raise ValueError('Supervised Vanilla did not branch from the same random warm-up')
    memory.assert_immutable(ctx['stamps'])

def endpoint(ctx,seed,kind):
    receipt='endpoint_'+str(seed)+'_'+kind;row=ctx['store'].load(receipt)
    if row is None:
        ctx['store'].event('endpoint_start',seed=seed,kind=kind)
        row=supervised_vanilla(ctx,seed) if kind=='supervised_vanilla' else scale_run.run_branch(ctx,G,K,seed,kind,CAP,limit=GUIDED,lr=.0003,coefficient=.001)
        validate_row(row);row['development']=scale_run.development(ctx,row)['report']
        count_model=model(row['method'],G,B,K,ctx['packs']['train']['x'].shape[-1],.001)
        row['model_parameters']=sum(p.numel() for p in count_model.parameters() if p.requires_grad)
        row['controller_parameters']=2 if kind=='agf_g' else 0
        row['head_parameters']=2*len(ctx['cfg']['category_indices']) if kind in ['agf_g','supervised_vanilla'] else 0
        row['total_learned_parameters']=row['model_parameters']+row['head_parameters']+row['controller_parameters']
        del count_model;gc.collect()
        ctx['store'].save(receipt,row);ctx['store'].write('endpoints/'+receipt+'.json',row)
    validate_row(row)
    cache=ctx['local']/'frozen_weights';cache.mkdir(exist_ok=True);weight_path=cache/(row['weights_sha']+'.pt')
    if not weight_path.exists():
        st=memory.required(ctx['store'],row['key'])
        if st['updates']!=GUIDED or training.weight_sha(st['weights'])!=row['weights_sha']:raise ValueError('Cannot cache changed frozen weights')
        pending=weight_path.with_suffix('.pending.pt');torch.save(st['weights'],pending);os.replace(pending,weight_path)
        ctx.setdefault('_pairing_proofs',{})[row['key']]=dict(updates=st['updates'],weights_sha=row['weights_sha'],
            streams=(st['sampled_batches_sha'],st['sampled_noise_sha']),head_shape=None if st['head'] is None else tuple(st['head'].shape),balance_present=st['balance'] is not None)
        del st;gc.collect()
    print('DEVELOPMENT ENDPOINT',json.dumps(dict(seed=seed,model=LABELS[kind],capture=100*row['development']['macro_capture'],
        patch_AP=100*row['development']['macro_ap'],VE=100*row['development']['reconstruction']['ve'])),flush=True)
    return row

def freeze(ctx,rows,name):
    check_pairing(ctx,rows);out=dict(models=rows,complete=True,configuration_fixed_before_VOC=True)
    old=ctx['store'].load(name)
    if old is not None and identity(old)!=identity(out):raise ValueError('Frozen endpoint catalog changed')
    ctx['store'].save(name,out);ctx['store'].write(name+'.json',out)

def archive(ctx,complete=False):
    s=ctx['store'];s.write('completion_status.json',dict(complete=complete,primary_complete=s.load('primary_complete') is not None,
        source_run=SOURCE_RUN,checkpoints_included=False))
    s.write('checkpoint_locations.json',dict(current=str(s.root),sources={k:dict(run=v.run_id,root=str(v.root)) for k,v in ctx['sources'].items()}))
    s.write('source_checkpoint_hashes.json',ctx['checkpoint_stamps'])
    s.write('source_manifest_hashes.json',ctx['stamps'])
    path=s.output/('ggf_checklist_evidence_'+s.run_id[:12]+'_results.zip');pending=path.with_suffix('.pending.zip');hashes={}
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(s.results.rglob('*')):
            if p.is_file():
                name='results/'+p.relative_to(s.results).as_posix();z.write(p,name);hashes[name]=sha_file(p)
        for option,name in [('wrapper_zip','experiment_source.zip'),('source_zip','pinned_core_source.zip')]:
            z.write(ctx['options'][option],name);hashes[name]=sha_file(ctx['options'][option])
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(pending) as z:
        if z.testzip() is not None or any(hashlib.sha256(z.read(k)).hexdigest()!=v for k,v in hashes.items()):raise ValueError('Archive verification failed')
    os.replace(pending,path);print('VERIFIED RESULTS ARCHIVE',path,flush=True);return path

