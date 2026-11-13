"""CPU-only scoring after two independent, complete blinded reviews."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from semantic_math import paired_summary,holm,quadratic_kappa

SCORES=['group_coherence_1_to_5','variation_coherence_1_to_5']

def read_ratings(path,cards):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(cards) or {r['card_id'] for r in rows}!=set(cards):raise ValueError('Every card must be rated exactly once')
    out={}
    for r in rows:
        if r['card_id'] in out:raise ValueError('Duplicate card')
        for field in SCORES+['confidence_1_to_3']:
            try:value=int(r[field])
            except (ValueError,KeyError):raise ValueError('Missing integer rating: '+field)
            if value not in range(1,4 if field.startswith('confidence') else 6):raise ValueError('Rating out of range')
            r[field]=value
        if not r.get('common_concept_explanation','').strip() or not r.get('variation_explanation','').strip():raise ValueError('Both explanations are required, including failures')
        out[r['card_id']]=r
    return out

def score(root,a_path,b_path):
    root=Path(root);key=json.loads((root/'private/review_key.json').read_text());cards={v['card_id']:v for v in key.values()}
    a=read_ratings(a_path,cards);b=read_ratings(b_path,cards)
    if hashlib.sha256(Path(a_path).read_bytes()).hexdigest()==hashlib.sha256(Path(b_path).read_bytes()).hexdigest():raise ValueError('Two independent sheets are required; files are byte-identical')
    import pandas as pd
    records=[]
    for card,k in cards.items():
        r=dict(k)
        for field in SCORES:
            r[field]=(a[card][field]+b[card][field])/2
            r[field+'_A']=a[card][field];r[field+'_B']=b[card][field]
        records.append(r)
    df=pd.DataFrame(records);out=root/'review_results';out.mkdir(exist_ok=True)
    completion=root/'completion.json'
    expected=json.loads(completion.read_text()).get('expected_methods') if completion.exists() else None
    if expected and set(df.model)!=set(expected):raise ValueError('Incomplete declared method inventory')
    controls=[m for m in ['supervised_vanilla','released_vanilla','slics','concept_sae'] if m in set(df.model)]
    df.to_csv(out/'all_card_ratings.csv',index=False)
    comparisons=[]
    for field in SCORES:
        p=df.groupby(['model','seed'])[field].mean().unstack(0).sort_index()
        for control in controls:
            comparisons.append(dict(metric=field,control=control,**paired_summary(p.agf_g,p[control])))
    for r,p in zip(comparisons,holm([r['one_sided_p'] for r in comparisons])):r['Holm_p']=float(p)
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    per_reviewer=[]
    for reviewer in ['A','B']:
        for field in SCORES:
            p=df.groupby(['model','seed'])[field+'_'+reviewer].mean().unstack(0).sort_index()
            for control in controls:
                per_reviewer.append(dict(reviewer=reviewer,metric=field,control=control,**paired_summary(p.agf_g,p[control])))
    pd.DataFrame(per_reviewer).to_csv(out/'individual_reviewer_comparisons.csv',index=False)
    agreement={f:dict(quadratic_weighted_kappa=quadratic_kappa([a[c][f] for c in cards],[b[c][f] for c in cards]),exact_agreement=float(np.mean([a[c][f]==b[c][f] for c in cards]))) for f in SCORES}
    result=dict(complete=True,raters=2,cards=len(cards),rating_SHA256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a_path,b_path]},agreement=agreement,comparisons=comparisons,scope='Human judgments of predefined panels; no universal semantic-coherence or recursive-generalization claim')
    (out/'completion.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--rater-a',required=True);p.add_argument('--rater-b',required=True);a=p.parse_args()
    print(json.dumps(score(a.root,a.rater_a,a.rater_b),indent=2))
