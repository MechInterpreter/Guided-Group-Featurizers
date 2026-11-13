"""CPU checks for reuse, cap changes, resume, seed expansion, and reporting."""
import copy
import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core
from coco_ggf import weight_bound as screen, width_balance_fit as engine
from test_width_balance import fixture, cheap_validation


def test_recipe_changes_only_two_bounds():
    old=screen.previous.settings();new=screen.settings()
    changed=copy.deepcopy(new['balance'])
    assert changed['bounds']==dict(capture=[.0125,.4],detection=[.0025,.4])
    changed['bounds']=old['balance']['bounds']
    assert changed==old['balance']
    assert new['groups']==1024 and new['active_groups']==16 and new['phases']==[2500,500]
    assert new['test_access'] is False
    assert screen.seeds_checked([23,21])==[21,23]
    for bad in [[],[21,21],[1],[21.,22],[True]]:
        with pytest.raises(ValueError):screen.seeds_checked(bad)


def test_higher_bound_can_exceed_old_cap():
    b=engine.Balancer(screen.settings()['balance'],'cpu')
    for _ in range(20):b.observe([torch.tensor(1.)]*3)
    for _ in range(600):b.adapt([torch.tensor(1.)]*3,torch.tensor([1.,.001,.001]))
    weights=b.weights().detach()
    assert weights[0]==1 and torch.all(weights[1:]>.2) and torch.all(weights[1:]<=.4+1e-7)


def test_higher_bound_resume_is_exact_and_changed_recipe_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'full40',4);b,_=fixture(tmp_path,'resume40',4)
    for ctx in [a,b]:ctx['recipe']=screen.settings()
    torch.manual_seed(21);initial=training.weights(training.make_model('vanilla',a['base'],8))
    full=engine.fit(a,'fit',initial,t,30,71021,True,True,reference_mse=1.)
    engine.fit(b,'fit',initial,t,30,71021,True,True,reference_mse=1.,max_updates=23)
    resumed=engine.fit(b,'fit',initial,t,30,71021,True,True,reference_mse=1.)
    for key in ['weights','balance','optimizer']:
        assert engine.state_identity(full[key])==engine.state_identity(resumed[key])
    assert full['sampled_batches_sha']==resumed['sampled_batches_sha']
    assert full['sampled_noise_sha']==resumed['sampled_noise_sha']
    b['recipe']['balance']['bounds']['capture'][1]=.8
    with pytest.raises(ValueError,match='identity changed'):
        engine.fit(b,'fit',initial,t,30,71021,True,True,reference_mse=1.)


def test_same_sampling_noise_and_phase_controller(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'cap20',4);b,_=fixture(tmp_path,'cap40',4)
    b['recipe']=screen.settings()
    torch.manual_seed(21);initial=training.weights(training.make_model('vanilla',a['base'],8))
    old=engine.fit(a,'first',initial,t,30,71021,True,True,reference_mse=1.)
    new=engine.fit(b,'first',initial,t,30,71021,True,True,reference_mse=1.)
    assert old['sampled_batches_sha']==new['sampled_batches_sha'] and old['sampled_noise_sha']==new['sampled_noise_sha']
    second=engine.fit(b,'second',new['weights'],t,12,82021,True,True,reference_mse=1.,
        initial_head=new['head'],initial_balance=new['balance'])
    assert second['balance']['reference_count']==20 and second['balance']['adaptations']==new['balance']['adaptations']+1
    assert second['history'][0]['validation']==new['history'][-1]['validation']


def test_expanding_requested_seeds_does_not_change_fit_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'validation',cheap_validation)
    ctx,t=fixture(tmp_path,'expand',4);ctx['recipe']=screen.settings();ctx['seeds']=[21]
    torch.manual_seed(21);initial=training.weights(training.make_model('vanilla',ctx['base'],8))
    first=engine.fit(ctx,'first',initial,t,4,71021,True,True,reference_mse=1.)
    ctx['seeds']=[21,22,23]
    again=engine.fit(ctx,'first',initial,t,4,71021,True,True,reference_mse=1.)
    assert first['fingerprint']==again['fingerprint'] and training.weight_sha(first['weights'])==training.weight_sha(again['weights'])


def metrics(capture=.3,ap=.7,ve=.65,mse=.35):
    return dict(macro_capture=capture,macro_ap=ap,reconstruction=dict(ve=ve,mse=mse))


def test_screen_needs_all_three_metrics_and_original_guard():
    baseline=metrics();good=metrics(.4,.8,.66,.34)
    assert screen.compare(21,baseline,good,.35)['passes_screen']
    assert not screen.compare(21,baseline,metrics(.4,.8,.64,.36),.35)['passes_screen']
    assert not screen.compare(21,baseline,good,.30)['passes_screen']
    assert not screen.compare(21,baseline,baseline,.35)['passes_screen']
    with pytest.raises(FloatingPointError):screen.compare(21,baseline,metrics(ap=float('nan')),.35)


def test_export_keeps_losses_guards_and_development_label(tmp_path):
    ctx,_=fixture(tmp_path,'reports',4);s=ctx['store'];r=dict(seed=21,baseline=metrics(),treatment=metrics(.4,.8,.64,.36))
    r['comparison']=screen.compare(21,r['baseline'],r['treatment'],.35)
    for phase in [1,2]:
        for key in [f'reused_cap020_s21_phase{phase}',f'phase{phase}_cap040_s21']:
            s.write('histories/'+key+'.json',[dict(update=250,weights_used=[1.,.2,.2],
                diagnostics=dict(weighted_encoder_gradient_norms=[1.,.01,.01],adaptation=dict(target_norms=[.3,.3,.3])))])
    screen.tables_and_plots(ctx,dict(results=[r],requested_seeds=[21]))
    with (s.results/'development_comparison.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and rows[1]['native_eligible']=='False'
    with (s.results/'development_gains.csv').open() as f:gains=list(csv.DictReader(f))
    assert gains[0]['split']=='development' and gains[0]['passes_screen']=='False'
    with (s.results/'bound_pressure_snapshots.csv').open() as f:pressure=list(csv.DictReader(f))
    assert any(x['at_upper_bound']=='True' and x['target_exceeds_current_strength']=='True' for x in pressure)
    source=tmp_path/'source.zip';source.write_bytes(b'source')
    output=screen.export(dict(ctx,sources={}),source)
    import zipfile,hashlib
    with zipfile.ZipFile(output) as z:
        assert not any(n.endswith('.pt') for n in z.namelist())
        for name,digest in json.loads(z.read('artifact_hashes.json')).items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest


def test_parent_seed_rejects_wrong_phase_lineage(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'lineage',4)
    torch.manual_seed(21);w=training.weights(training.make_model('vanilla',ctx['base'],8))
    target_id=identity(t);head=torch.zeros(1,2)
    warm=dict(updates=7640,target=7640,best=w)
    p1=dict(updates=2500,target=2500,weights=w,head=head,balance=None,fingerprint='wrong',history=[])
    p2=dict(updates=500,target=500,weights=w,head=head,balance=None,fingerprint='wrong',history=[dict(update=500,validation=metrics())])
    payloads=dict(warm4_mask_s21=warm,phase1_mask4_balanced_s21=p1,phase2_mask4_balanced_s21=p2)
    class Parent:
        def load(self,key):return payloads[key]
    provenance=dict(splits=dict(train=ctx['packs']['train']['ids'],select=ctx['packs']['select']['ids']))
    original=type('Original',(),dict(results=tmp_path))()
    (tmp_path/'provenance.json').write_text(json.dumps(provenance))
    row=dict(rank=4,key='phase2_mask4_balanced_s21',balanced=True,supervised=True,targets=t,targets_sha=target_id,
        weights_sha=training.weight_sha(w),head_sha=training.weight_sha({'head':head}),reference_mse=1.)
    ctx.update(parent=Parent(),sources=dict(original=original),catalog={(21,'mask4_balanced'):row},checkpoint_stamps=[])
    monkeypatch.setattr(screen.matched,'check_targets',lambda *args:None)
    monkeypatch.setattr(screen.matched,'valid_weights',lambda *args:None)
    with pytest.raises(ValueError,match='fingerprint differs'):screen.verify_seed(ctx,21)
    for state,initial,n,seed,h,b in [(p1,w,2500,71021,None,None),(p2,w,500,82021,head,None)]:
        state['fingerprint']=identity(dict(initial=training.weight_sha(initial),targets=t,updates=n,seed=seed,
            balanced=True,supervised=True,reference_mse=1.,initial_head_sha=None if h is None else training.weight_sha({'head':h}),
            initial_balance=engine.state_identity(b),base=ctx['base'],cfg=ctx['cfg'],balance=screen.previous.settings()['balance'],
            train_ids=provenance['splits']['train'],select_ids=provenance['splits']['select']))
    monkeypatch.setattr(screen.matched,'source_key_stamp',lambda parent,key:dict(key=key))
    assert screen.verify_seed(ctx,21)[0]==row
    row['weights_sha']='changed'
    with pytest.raises(ValueError,match='identities differ'):screen.verify_seed(ctx,21)


def test_orchestration_reuses_warmup_and_paired_control_without_test(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'pipeline',4);ctx['recipe']=screen.settings();ctx['seeds']=[21]
    torch.manual_seed(21);w=training.weights(training.make_model('vanilla',ctx['base'],8))
    h=torch.zeros(1,2);balance=dict(marker='carried')
    warm=dict(best=w)
    base1=dict(updates=2500,target=2500,sampled_batches_sha='batch1',sampled_noise_sha='noise1')
    base2=dict(updates=500,target=500,sampled_batches_sha='batch2',sampled_noise_sha='noise2',
        history=[dict(update=500,validation=metrics())])
    row=dict(targets=t,reference_mse=.35,weights_sha='saved-endpoint',targets_sha=identity(t))
    monkeypatch.setattr(screen,'verify_seed',lambda c,s:(row,warm,base1,base2))
    monkeypatch.setattr(screen.matched,'assert_sources',lambda c:None)
    monkeypatch.setattr(screen,'tables_and_plots',lambda c,o:None)
    calls=[]
    def fitted(c,key,initial,targets,n,seed,balanced,supervised,**kwargs):
        calls.append(dict(key=key,initial=initial,kwargs=kwargs,seed=seed,n=n))
        assert initial is w and targets is t and balanced and supervised
        assert c['recipe']['balance']['bounds']['capture'][1]==.4
        if n==500:
            assert kwargs['initial_head'] is h and kwargs['initial_balance'] is balance
        parent=base1 if n==2500 else base2
        return dict(parent,weights=w,head=h,balance=balance,
            history=[dict(update=n,validation=metrics(.4,.8,.66,.34),weights_next=[1.,.4,.4])])
    monkeypatch.setattr(screen,'fit',fitted)
    out=screen.run(ctx)
    assert [x['seed'] for x in calls]==[71021,82021]
    assert [x['n'] for x in calls]==[2500,500]
    assert out['complete'] and not out['study_complete'] and out['all_requested_seeds_pass_screen']
    assert out['test_access'] is False and 'test' not in ctx['packs']
    assert ctx['store'].load('requested_screen_complete')['results'][0]['new_updates']==3000
    base2['sampled_noise_sha']='different'
    # The copied new state differs from the altered saved control.
    original_fitted=fitted
    def divergent(*args,**kwargs):
        result=original_fitted(*args,**kwargs)
        if args[4]==500:result['sampled_noise_sha']='original-noise'
        return result
    monkeypatch.setattr(screen,'fit',divergent)
    with pytest.raises(ValueError,match='noise diverged'):screen.run(ctx)


def test_prepare_opens_only_train_development_and_expansion_reuses_run(tmp_path,monkeypatch):
    train=list(range(2000));select=list(range(2000,2400))
    provenance=dict(splits=dict(train=train,select=select,test=list(range(2400,3900))))
    original_results=tmp_path/'original_results';original_results.mkdir()
    for name in ['provenance.json','coverage_plan.json']:
        (original_results/name).write_text(json.dumps(provenance))
    norm=dict(mean=np.zeros((196,768)),scale=1.,train_sha='saved')
    class Original:
        run_id=screen.memory.EXPECTED_PARENT
        results=original_results
        config=dict(config=dict(extract_chunk=100))
        def load(self,key):
            assert key=='normalization';return norm
    class Parent:
        run_id=screen.PARENT_RUN
        results=tmp_path/'parent_results'
        config=dict(recipe=screen.previous.settings(),versions=screen.matched.versions(),
            base=dict(groups=1024,active_groups=16,block_rank=3),
            fit_config=dict(device='cuda',images_per_update=8,lr=3e-4,snr=.1),
            source_runs=dict(original=screen.memory.EXPECTED_PARENT))
        def load(self,key):
            if key=='preflight':return dict(hardware=dict(gpu='Tesla T4'))
            assert key=='all_models_frozen'
            return dict(complete=True,models=[dict(seed=s,arm=a) for s in [21,22,23] for a in screen.previous.ARMS])
    Parent.results.mkdir();(Parent.results/'summary.json').write_text('{"complete":true}')
    monkeypatch.setattr(screen.matched,'discover',lambda path,*a,**k:Parent() if 'parent' in str(path) else Original())
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'Tesla T4')
    monkeypatch.setattr(screen,'verify_seed',lambda *a:None)
    monkeypatch.setattr(screen.memory,'source_snapshot',lambda stores:[])
    monkeypatch.setattr(screen.memory,'memory_safe_metrics',lambda:None)
    monkeypatch.setattr(screen.matched,'source_key_stamp',lambda source,key:dict(key=key))
    monkeypatch.setattr(screen.matched,'assert_sources',lambda ctx:None)
    opened=[]
    def normalized(source,split,*a,**k):
        opened.append(split)
        assert split in ['train','select']
        return dict(ids=provenance['splits'][split],x=np.empty((0,196,768)),masks=[])
    monkeypatch.setattr(screen.memory,'normalized_pack',normalized)
    args=[tmp_path/'parent',tmp_path/'activation',tmp_path/'new_output',tmp_path/'local',tmp_path/'cache','source-hash']
    a=screen.prepare(*args,seeds=[21],require_drive=False)
    b=screen.prepare(*args,seeds=[21,22,23],require_drive=False)
    assert a['store'].run_id==b['store'].run_id
    assert opened==['train','select','train','select'] and 'test' not in b['packs']
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'Different GPU')
    with pytest.raises(RuntimeError,match='same GPU model'):
        screen.prepare(*args,require_drive=False)
    assert opened==['train','select','train','select']
    with pytest.raises(ValueError,match='overlaps'):
        screen.prepare(args[0],args[1],args[0]/'new',*args[3:],require_drive=False)
