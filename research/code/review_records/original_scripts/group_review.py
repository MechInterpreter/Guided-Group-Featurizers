"""Separate all75 five-of-six group review; never changes frozen pair outcomes."""
import argparse, collections, datetime, hashlib, html, json, math, re, shutil
from pathlib import Path
import review_ready as original

WORK=Path(__file__).resolve().parents[2]
STUDY=WORK/'output/ggf_all75_semantic_review_20261027/study'
ROOT=WORK/'output/ggf_all75_group_review_20261028'
PUBLIC=ROOT/'reviewer_public'
STREAMS=('context1','context2','context3')
CODES=('clear','partial','no_clear_theme','insufficient_evidence')
SPEC=('specific','broad_but_discriminating','generic_nondiscriminating','unclear')
HISTORICAL=WORK/'output/ggf_matched_semantic_review_20261027/REVIEW_PROTOCOL.txt'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(p,v):
    p=Path(p);raw=json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False).encode('utf8')
    if p.exists() and p.read_bytes()!=raw:raise ValueError('Frozen content differs: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def key(*s):return hashlib.sha256(('all75-group-six-v1|'+'|'.join(map(str,s))).encode()).hexdigest()

INSTRUCTIONS="""Read ONLY this public packet and its referenced original PNGs in a fresh context. Do not read private mappings, registries, method/native-unit/category identities, previous descriptions, numerical scores, other raters' labels or unrelated files. No browsing or subagents. Actually inspect every full sheet and every zoom strip at original resolution before rating. The PNG header mentions image pairs from an earlier task: IGNORE that pair instruction. Judge ONE overarching motif across ALL SIX numbered marked regions together, not three separate pair themes. Objects outside the marked patches do not establish a match. Parts, textures, geometric structures, materials, contexts, relations and cross-category motifs are legitimate themes. An unnamed repeated motif is not automatically incoherent, but do not stretch a vague theme to cover unrelated examples. Choose a single best-supported theme of at most30 words, list every matching view (1–6), and explain exceptions. Broad or generic motifs may be coherent; rate specificity separately.

Use the earlier five-of-six group rubric: valid HIGH examples have nonzero native responses (the frozen per-view metadata supplies validity). If fewer than4 valid HIGH examples, coherence=insufficient_evidence. With adequate readable evidence, clear means one recognizable motif fits at least ceil(5*valid_HIGH/6); partial means a motif fits at least ceil(valid_HIGH/2) but does not reach clear; no_clear_theme means no recognizable motif fits half the valid HIGH regions after allowing parts, texture, geometry, context and relations. With six valid views: clear=5or6 matches; partial=3or4; no_clear_theme=0to2. Resolution/occlusion that prevents judgment is insufficient_evidence, never an invented no or yes. A collection of unrelated pair-specific themes does NOT make the overall group clear.

Specificity is independent: specific / broad_but_discriminating / generic_nondiscriminating / unclear. Record confidence high / medium / low. Exceptions must list all valid views not fitting the proposed single theme, and explain competing meanings or uncertain marked detail. LOW-response images are not supplied in this supplemental task: high_low_contrast must remain not_assessed; do not infer contrast or specificity against negatives. Do not infer model identity. Return the exact output template with your actual fresh reviewer ID and provenance. Write your completed actual JSON atomically via .tmp then rename, so unfinished labels cannot be accepted. These are AI review replicates, not independent human raters."""

def prepare():
    collection,pub,private,lock,human=original.verify_review(STUDY)
    banks={b['bank_id']:b for b in collection['banks']}
    chosen=[c for c in private['cases'] if banks[c['bank_id']]['capacity_matched']]
    assert len(chosen)==675
    images={r['image_id']:r for r in read(STUDY/'IMAGE_ASSETS.json')['images']}
    viewlocks={b['bank_id']:b for b in read(STUDY/'PORTABLE_LOCK.json')['banks']}
    viewmaps={}
    for b in banks:
        p=STUDY/'portable'/b/'VIEWS.json'
        if sha(p)!=viewlocks[b]['views_sha256']:raise ValueError('View cache changed')
        viewmaps[b]=read(p)
    order=sorted(chosen,key=lambda c:key('display',c['case_id']))
    public=[];private_new=[];hashes={}
    for c in order:
        cid=c['case_id'];p=STUDY/'reviews/coherence/cards'/f'{cid}.png'
        if sha(p)!=lock['cards'][f'cards/{cid}.png']:raise ValueError('Original card changed')
        target=PUBLIC/'cards'/p.name;original.copy_exact(p,target);hashes[f'cards/{p.name}']=sha(p)
        views=viewmaps[c['bank_id']]['roles']['audit'][str(c['unit'])]
        valid=[bool(j<len(views) and views[j].get('patches') and images.get(views[j].get('image_id'),{}).get('status')=='ready') for j in c['source_slots']]
        assert sum(valid)==6
        public.append(dict(case_id=cid,card=f'cards/{cid}.png',valid_high_views=[j+1 for j,v in enumerate(valid) if v],valid_high_count=sum(valid)))
        private_new.append(c)
    protocol=dict(schema='all75-group-five-of-six-v1',scope='seed21_supplemental_group_coherence',
      status='frozen_before_new_group_ratings_after_pair_results',groups=675,matched_banks=9,contexts=3,
      sampling='Every native group from all matched75 seed21 banks; full-category4 and category-split5. Native SLiCS37/38 and ConceptSAE111 excluded from this matched endpoint.',
      evidence='Exact original strongest-six audit PNGs and patch positions, copied byte-for-byte. No image replacement or new model inference. Printed pair header superseded by whole-group instructions.',
      primary='Groups rated clear by at least two of three fresh blinded AI reviewers / all native groups. Unknowns remain in denominator.',
      secondary='Majority-clear AND at least two independent specificity ratings specific or broad_but_discriminating / all native groups; report both component distributions, no weighted composite.',
      historical_rubric_sha256=sha(HISTORICAL),instructions=INSTRUCTIONS,
      differences='Same five-of-six HIGH rubric, different checkpoints and image pool from old pilot. LOW images absent so high_low_contrast not_assessed. Supplemental endpoint requested after pair results; not an independent confirmatory allocation.',
      immutable_pair_result_sha256=sha(STUDY/'COHERENCE_COMPLETE.json'),original_review_lock_sha256=sha(STUDY/'reviews/coherence/REVIEW_LOCK.json'),
      reporting='Separate full-category and category split, each fold and combined equal-fold means; shared Vanilla counted once. Single seed, AI-majority labels not independent human validation. Do not change prior pair scores or paper automatically.')
    freeze(ROOT/'PROTOCOL.json',protocol)
    freeze(PUBLIC/'PUBLIC.json',dict(schema=protocol['schema'],instructions=INSTRUCTIONS,cases=public))
    freeze(ROOT/'PRIVATE.json',dict(cases=private_new,banks=[b for b in collection['banks'] if b['capacity_matched']]))
    lock2=dict(protocol_sha256=sha(ROOT/'PROTOCOL.json'),public_sha256=sha(PUBLIC/'PUBLIC.json'),private_sha256=sha(ROOT/'PRIVATE.json'),cards=hashes)
    freeze(ROOT/'REVIEW_LOCK.json',lock2)
    queue=[]
    for ctx in STREAMS:
        for start in range(0,len(public),12):
            batch=f'B{start//12:03d}';cases=public[start:start+12]
            template=dict(public_sha256=lock2['public_sha256'],method_blind=True,reviewer='',cases=[dict(case_id=c['case_id'],theme='',fit_views=[],high_fit_count=None,exceptions=[],coherence='',specificity='',confidence='',high_low_contrast='not_assessed',observations='') for c in cases])
            packet=dict(context=ctx,batch=batch,instructions=INSTRUCTIONS,cases=cases,card_root='../../',output_template=template)
            freeze(PUBLIC/'packets'/ctx/(batch+'.json'),packet);queue.append(dict(context=ctx,batch=batch,cases=len(cases)))
    freeze(PUBLIC/'QUEUE.json',dict(batches=queue,total_groups=675,streams=3,groups_per_stream=675))
    return sync()

def validate_labels(path,ctx,batch):
    lock=read(ROOT/'REVIEW_LOCK.json')
    for p,h in [(ROOT/'PROTOCOL.json',lock['protocol_sha256']),(PUBLIC/'PUBLIC.json',lock['public_sha256']),(ROOT/'PRIVATE.json',lock['private_sha256'])]:
        if sha(p)!=h:raise ValueError('Locked source changed')
    packet=read(PUBLIC/'packets'/ctx/(batch+'.json'));d=read(path)
    if d.get('public_sha256')!=lock['public_sha256'] or d.get('method_blind') is not True or not d.get('reviewer','').startswith('agent:/root/'):raise ValueError('Actual reviewer provenance required')
    cases=d.get('cases',[]);by_id={c['case_id']:c for c in cases}
    if len(by_id)!=len(cases) or set(by_id)!={c['case_id'] for c in packet['cases']}:raise ValueError('Fixed case coverage differs')
    for c in packet['cases']:
        if sha(PUBLIC/c['card'])!=lock['cards'][c['card']]:raise ValueError('Card changed')
        v=by_id[c['case_id']];fit=v['fit_views'];n=c['valid_high_count']
        if len(set(fit))!=len(fit) or not set(fit)<=set(c['valid_high_views']) or v['high_fit_count']!=len(fit):raise ValueError('Invalid fit count')
        exception_ids=[]
        for ex in v['exceptions']:
            if isinstance(ex,int):exception_ids.append(ex)
            elif isinstance(ex,dict) and isinstance(ex.get('view'),int) and str(ex.get('reason','')).strip():exception_ids.append(ex['view'])
            elif isinstance(ex,str) and (m:=re.match(r'^View\s+([1-6])\b',ex)):exception_ids.append(int(m.group(1)))
            else:raise ValueError('Each exception must identify its numbered view')
        if sorted(exception_ids)!=sorted(set(c['valid_high_views'])-set(fit)):raise ValueError('All nonmatching views must remain exceptions')
        if v['coherence'] not in CODES or v['specificity'] not in SPEC or v['confidence'] not in ('high','medium','low'):raise ValueError('Invalid label')
        if v['high_low_contrast']!='not_assessed':raise ValueError('No LOW evidence supplied')
        if len(v['theme'].split())>30 or not v['observations'].strip():raise ValueError('Theme/rationale missing or too long')
        if v['coherence']!='insufficient_evidence':
            expected='clear' if len(fit)>=math.ceil(5*n/6) else 'partial' if len(fit)>=math.ceil(n/2) else 'no_clear_theme'
            if v['coherence']!=expected or n<4:raise ValueError('Five-of-six rubric mismatch')
            if len(fit)>=math.ceil(n/2) and not v['theme'].strip():raise ValueError('Positive motif must be stated')
    return d

def sync(start=None):
    p=ROOT/'WORK_STATE.json';state=read(p) if p.exists() else dict(total_batches=171,total_groups=675,created_at_utc=now(),active=[],accepted=[])
    accepted=[];seen=set()
    for ctx in STREAMS:
        for raw in sorted((ROOT/'returned_reviews'/ctx).glob('B*.json')):
            d=validate_labels(raw,ctx,raw.stem)
            if d['reviewer'] in seen:raise ValueError('Fresh reviewer required per packet')
            seen.add(d['reviewer']);target=ROOT/'accepted'/ctx/raw.name;original.copy_exact(raw,target)
            accepted.append(dict(context=ctx,batch=raw.stem,cases=len(d['cases']),reviewer=d['reviewer'],sha256=sha(raw)))
    done={(r['context'],r['batch']) for r in accepted}
    active=[r for r in state['active'] if (r['context'],r['batch']) not in done]
    if start:
        ctx,batch,agent=start
        if len(active)>=3 or (ctx,batch) in done or any((r['context'],r['batch'])==(ctx,batch) for r in active):raise ValueError('Duplicate or too many assignments')
        if not (PUBLIC/'packets'/ctx/(batch+'.json')).is_file():raise ValueError('Unknown packet')
        active.append(dict(context=ctx,batch=batch,agent=agent,started_at_utc=now()))
    assigned={(r['context'],r['batch']) for r in active}
    nexts={ctx:next((f'B{i:03d}' for i in range(57) if (ctx,f'B{i:03d}') not in done|assigned),None) for ctx in STREAMS}
    state.update(accepted=accepted,accepted_batches=len(accepted),accepted_group_ratings=sum(r['cases'] for r in accepted),active=active,next_batch_by_context=nexts,complete=len(accepted)==171,updated_at_utc=now())
    tmp=p.with_suffix('.tmp');tmp.parent.mkdir(parents=True,exist_ok=True);tmp.write_text(json.dumps(state,indent=2),encoding='utf8');tmp.replace(p)
    return {k:state[k] for k in ('accepted_batches','accepted_group_ratings','active','next_batch_by_context','complete')}

def score():
    state=sync()
    if not state['complete']:raise ValueError('All171 actual packets required')
    private=read(ROOT/'PRIVATE.json');banks={b['bank_id']:b for b in private['banks']};streams=[]
    for ctx in STREAMS:
        allcases={}
        for p in sorted((ROOT/'accepted'/ctx).glob('*.json')):
            d=validate_labels(p,ctx,p.stem)
            for c in d['cases']:
                if c['case_id'] in allcases:raise ValueError('Duplicate native group')
                allcases[c['case_id']]=c
        if set(allcases)!={c['case_id'] for c in private['cases']}:raise ValueError('Missing group')
        streams.append(allcases)
    results=[];outcomes=[]
    for bank_id,b in banks.items():
        cases=[c for c in private['cases'] if c['bank_id']==bank_id];counts=collections.Counter();spec_counts=collections.Counter();joint=0;votes=[];disagreements=0
        for c in cases:
            raw=[s[c['case_id']] for s in streams];labels=[r['coherence'] for r in raw];sp=[r['specificity'] for r in raw]
            winner=next((k for k in CODES if labels.count(k)>=2),'unresolved_vote');counts[winner]+=1
            sw=next((k for k in SPEC if sp.count(k)>=2),'unresolved_vote');spec_counts[sw]+=1
            specific=sum(k in ('specific','broad_but_discriminating') for k in sp)>=2
            joint+=winner=='clear' and specific;disagreements+=len(set(labels))>1
            outcomes.append(dict(case_id=c['case_id'],bank_id=bank_id,unit=c['unit'],majority=winner,specificity_majority=sw,clear_and_discriminating=winner=='clear' and specific,raw=raw))
        if len(cases)!=75 or sum(counts.values())!=75:raise ValueError('Native bank denominator mismatch')
        results.append(dict(bank_id=bank_id,regime=b['regime'],method=b['kind'],fold=b['fold'],groups=75,counts={k:counts[k] for k in (*CODES,'unresolved_vote')},specificity_counts={k:spec_counts[k] for k in (*SPEC,'unresolved_vote')},clear_rate=counts['clear']/75,clear_and_discriminating=joint,clear_and_discriminating_rate=joint/75,disagreements=disagreements))
    methods=[]
    for regime,method in sorted({(b['regime'],b['kind']) for b in banks.values()}):
        rr=[r for r in results if (r['regime'],r['method'])==(regime,method)]
        methods.append(dict(regime=regime,method=method,groups=sum(r['groups'] for r in rr),native_banks=len(rr),counts={k:sum(r['counts'][k] for r in rr) for k in (*CODES,'unresolved_vote')},clear_rate=sum(r['clear_rate'] for r in rr)/len(rr),clear_and_discriminating_rate=sum(r['clear_and_discriminating_rate'] for r in rr)/len(rr)))
    proof=dict(complete=True,scope='seed21_supplemental_group_coherence',protocol_sha256=sha(ROOT/'PROTOCOL.json'),banks=results,methods=methods,group_outcomes=outcomes,sources=read(ROOT/'WORK_STATE.json')['accepted'],limits=read(ROOT/'PROTOCOL.json')['differences'])
    if sha(STUDY/'COHERENCE_COMPLETE.json')!=read(ROOT/'PROTOCOL.json')['immutable_pair_result_sha256']:raise ValueError('Original pair result changed')
    freeze(ROOT/'RESULTS.json',proof)
    rows=''.join(f'<tr><td>{r["regime"]}</td><td>{r["method"]}</td><td>{r["groups"]}</td><td>{r["counts"]["clear"]}</td><td>{r["clear_rate"]:.1%}</td><td>{r["clear_and_discriminating_rate"]:.1%}</td></tr>' for r in methods)
    (ROOT/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>Whole-group coherence</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:20px}th,td{padding:14px;border-bottom:1px solid #ccc;text-align:left}table{border-collapse:collapse}</style><h1>Whole-group coherence · seed21</h1><p>All675 matched native groups. Three fresh blinded review streams; clear requires a single motif fitting5of6 high-response examples. Geometry and texture count; specificity remains separate. No LOW-response contrast supplied. Supplemental endpoint requested after pair scores; prior pair outcomes unchanged.</p><table><tr><th>Experiment</th><th>Method</th><th>Groups</th><th>Majority clear</th><th>Clear rate</th><th>Clear + discriminating</th></tr>'+rows+'</table><p>Category folds averaged equally; shared Vanilla counted once. AI-majority labels, one training seed, previously inspected image pool. Not an independent human or unseen-image confirmation.</p><a href="RESULTS.json">Full counts, exceptions, themes, disagreements and provenance</a>',encoding='utf8')
    return {k:proof[k] for k in ('complete','scope','methods')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('prepare','sync','score'),required=True);p.add_argument('--start',nargs=3);a=p.parse_args()
    print(json.dumps(prepare() if a.phase=='prepare' else score() if a.phase=='score' else sync(a.start),indent=2))
