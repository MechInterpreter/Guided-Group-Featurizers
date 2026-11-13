"""Discover only completed, normalization-compatible frozen competitors."""
import json
from pathlib import Path
import torch

def check_context(config,ctx,concept=False):
    from coco_baselines.storage import identity
    train_key='train' if concept else 'train_identity'
    norm_key='normalization' if concept else 'normalization_sha'
    if config.get(train_key)!=ctx['packs']['train']['identity'] or config.get(norm_key)!=identity(ctx['norm']):raise ValueError('Comparator training cohort/normalization differs')
    if concept and config.get('cohort')!='coco':raise ValueError('Use completed COCO Concept-SAE, not VOC')
    dev='development' if concept else 'development_identity'
    if config.get(dev)!=ctx['packs']['select']['identity']:raise ValueError('Comparator development split differs')

def discover(ctx,options):
    from coco_baselines.storage import identity
    from coco_coordinated.run import FrozenStore,required
    from coco_baselines import training
    local=Path(options['local'])/'native_sources';rows=[];stores={}
    root=Path(options['slics_source']);catalogs=[]
    for p in (root/'results').glob('*/all_models_frozen.json'):
        o=json.loads(p.read_text());manifest=json.loads((p.parent/'manifest.json').read_text())
        if o.get('complete') and manifest['config'].get('study')=='ggf_slics_five_seed_evaluation':catalogs.append((p,o,manifest))
    if len(catalogs)!=1:raise ValueError('Need exactly one completed five-order SLiCS catalog')
    path,catalog,audit=catalogs[0];parent=audit['config']['parent_run']
    store=FrozenStore(root/'checkpoints'/parent[:20],local/'slics');check_context(store.config,ctx)
    stores['slics']=store
    for seed in [21,22,23]:
        choices=[r for r in catalog['models'] if r['seed']==seed and r['arm']=='slics_final_partial']
        if len(choices)!=1:raise ValueError('Missing completed SLiCS order')
        r=choices[0]
        if (r['selected_total_updates'],r['total_training_updates'])!=(10640,10640):raise ValueError('Do not reuse SLiCS initialization/selected prefix')
        if r.get('solver',{}).get('nnls_tolerance')!=1e-5:raise ValueError('SLiCS solver protocol changed')
        slots={int(t['category_index']):int(t['contrast_group']) for t in r['targets']}
        if sorted(slots.values())!=list(range(75)):raise ValueError('Invalid native SLiCS category slots')
        rows.append(dict(kind='slics',seed=seed,key=r['key'],weights_sha=r['weights_sha'],slots=slots,source_run=store.run_id))
    root=Path(options['concept_source_root']);candidates=[]
    for p in (root/'results').glob('*/manifest.json'):
        m=json.loads(p.read_text());c=m['config']
        if c.get('cohort')=='coco' and c.get('protocol',{}).get('revision')=='concept-sae-development-repair-v2':
            check_context(c,ctx,True);st=FrozenStore(root/'checkpoints'/m['run_id'][:20],local/'concept')
            done=st.load('revised_concept_complete')
            if done and done.get('complete'):candidates.append(st)
    if len(candidates)!=1:raise ValueError('Need exactly one completed repaired COCO Concept-SAE run')
    store=candidates[0];stores['concept_sae']=store;cats=store.config['categories']
    for seed in [21,22,23]:
        state=required(store,f'concept_sae_final_s{seed}')
        validate_concept(state,seed)
        rows.append(dict(kind='concept_sae',seed=seed,key=f'concept_sae_final_s{seed}',weights_sha=state['weights_sha'],slots={int(c):j for j,c in enumerate(cats)},source_run=store.run_id))
        del state
    return stores,rows

def validate_concept(state,seed):
    from coco_baselines import training
    p=state['protocol']
    if p.get('revision')!='concept-sae-development-repair-v2' or p.get('epochs')!=[30,50,30]:raise ValueError('Do not reuse the failed or short-schedule Concept-SAE run')
    if state.get('seed')!=seed or [s['epochs_completed'] for s in state['stage_receipts']]!=[30,50,30]:raise ValueError('Incomplete Concept-SAE stages')
    if not state['stage_receipts'][0].get('best_diagnostics',{}).get('quality_pass'):raise ValueError('Concept-SAE mask-quality check failed')
    if training.weight_sha(state['weights'])!=state['weights_sha']:raise ValueError('Concept-SAE weight checksum mismatch')

def load(ctx,row,store,categories):
    from coco_coordinated.run import required
    from coco_baselines import training
    from native_units import SLiCSUnits,ConceptUnits
    slots=[row['slots'][int(c)] for c in categories];st=required(store,row['key'])
    if row['kind']=='slics':
        if st['updates']!=10640 or training.weight_sha({'D':st['D']})!=row['weights_sha']:raise ValueError('SLiCS final weights changed')
        return SLiCSUnits(st['D'],slots,ctx['device'])
    validate_concept(st,row['seed'])
    import concept_sae as native
    model=native.build(native.source(ctx['options']['concept_source']),len(store.config['categories'])).to(ctx['device'])
    model.load_state_dict(st['weights']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return ConceptUnits(model,slots,ctx['device'])
