'use client';
import {ArrowLeft,ArrowRight} from 'lucide-react';

export type PairView={imageId:number;imageIndex:number;patches:number[];pooled:number};
type Verdict='yes'|'no'|'unclear';
export type OriginalPair={pair:number;views:PairView[];unavailable:boolean;majority:{semantic:Verdict;visual:Verdict};reviews:{context:string;pair:number;semantic:Verdict;visual:Verdict;shared_meaning:string;exceptions:string}[]};
export type PairGroup={group:number;case_id:string;pairs:OriginalPair[]};
export type PairArchive={seed:number;steps:number;total_pairs:number;methods:Record<string,PairGroup[]>;crops:Record<string,{src:string;pixel_sha256:string}>;contexts:{context:string;sha256:string}[];rubrics:{semantic_rubric:string;visual_rubric:string;unavailable:string}};

export function PairNavigator({cursor,onChange}:{cursor:number;onChange:(cursor:number)=>void}){
 return <section className="pair-navigation" data-tour="examples" aria-label="Original pair walkthrough">
  <div><span className="eyebrow">ORIGINAL BLINDED PAIRS · SEED 21</span><h2>Pair {cursor+1} <span>of 225 per method</span></h2><p>Every group, every fixed pair, including exceptions. Each method has its own groups and image selections.</p></div>
  <div className="pair-navigation-controls"><button aria-label="Previous reviewed pair" disabled={cursor===0} onClick={()=>onChange(cursor-1)}><ArrowLeft size={16}/>Previous</button><label>Group<select aria-label="Walkthrough group" value={Math.floor(cursor/3)} onChange={e=>onChange(Number(e.target.value)*3+cursor%3)}>{Array.from({length:75},(_,g)=><option key={g} value={g}>{String(g).padStart(2,'0')}</option>)}</select></label><label>Pair<select aria-label="Fixed review pair" value={cursor%3} onChange={e=>onChange(Math.floor(cursor/3)*3+Number(e.target.value))}>{[0,1,2].map(p=><option key={p} value={p}>{p+1} / 3</option>)}</select></label><button aria-label="Next reviewed pair" disabled={cursor===224} onClick={()=>onChange(cursor+1)}>Next<ArrowRight size={16}/></button></div>
  <div className="pair-progress" aria-hidden="true"><i style={{width:(cursor+1)/225*100+'%'}}/></div>
 </section>;
}

function ReviewView({view,cropSrc,slot,active,patch,onPatch}:{view:PairView;cropSrc:string;slot:number;active:boolean;patch:number;onPatch:(patch:number)=>void}){
 const rect=(p:number)=>({x:p%14*16,y:Math.floor(p/14)*16,width:15,height:15});
 return <div className={'pair-view'+(active?' active':'')} data-image-id={view.imageId} data-patches={view.patches.join(',')}>
  <button className="pair-view-title" onClick={()=>onPatch(view.patches[0])}>View {slot+1}<span>Image {view.imageId}</span></button>
  <svg className="pair-crop" viewBox="0 0 224 224" role="img" aria-label={'Original review view '+(slot+1)+', image '+view.imageId+', four marked patches'}>
   <image href={cropSrc} width={224} height={224}/>
   {view.patches.map(p=><rect key={p} {...rect(p)} fill="transparent" stroke="#f31567" strokeWidth={2} onClick={()=>onPatch(p)}><title>Original marked patch {p}</title></rect>)}
   {active&&view.patches.includes(patch)&&<rect {...rect(patch)} fill="none" stroke="white" strokeWidth={1} pointerEvents="none"/>}
  </svg>
  <div className="pair-zooms">{view.patches.map(p=>{const b=rect(p);return <button key={p} aria-label={'Inspect original view '+(slot+1)+' patch '+p} aria-pressed={active&&patch===p} onClick={()=>onPatch(p)}><svg viewBox={[b.x,b.y,16,16].join(' ')}><image href={cropSrc} width={224} height={224}/></svg><span>{p}</span></button>})}</div>
 </div>;
}

export default function PairEvidence({entry,pairIndex,crops,imageId,patch,selected,onPatch,primary}:{entry:PairGroup;pairIndex:number;crops:PairArchive['crops'];imageId:number;patch:number;selected:boolean;onPatch:(view:PairView,patch:number)=>void;primary:boolean}){
 const pair=entry.pairs[pairIndex];
 return <div className="pair-evidence" data-case-id={entry.case_id} data-pair={pairIndex}>
  <div className="pair-images">{pair.views.map((view,index)=><ReviewView key={index} view={view} cropSrc={crops[String(view.imageId)].src} slot={pairIndex*2+index} active={selected&&imageId===view.imageId} patch={patch} onPatch={p=>onPatch(view,p)}/>)}</div>
  <div className="pair-verdicts" data-tour={primary?'method-groups':undefined}>{(['semantic','visual'] as const).map(endpoint=>{const values=pair.reviews.map(r=>r[endpoint]),agreed=new Set(values).size===1;return <div key={endpoint} data-endpoint={endpoint} data-verdict={pair.majority[endpoint]}><span>{endpoint==='semantic'?'Semantic meaning':'Visual structure'}</span><strong className={'verdict-'+pair.majority[endpoint]}>{pair.majority[endpoint]==='unclear'?'Unclear':pair.majority[endpoint]==='yes'?'Yes':'No'}</strong><small>{values.filter(v=>v===pair.majority[endpoint]).length}/3 {agreed?'agree':'· disagreement'}</small></div>})}</div>
  {pair.unavailable&&<p className="pair-unavailable">Unavailable evidence · retained in the planned total.</p>}
  <details className="pair-original-reviews" key={entry.case_id+'-'+pairIndex}><summary>All 3 original judgments &amp; explanations</summary><p>Labels and notes are preserved verbatim. Semantic and visual judgments are separate; the summary uses a two-of-three majority.</p>{pair.reviews.map((review,index)=><article key={review.context} data-review-context={review.context}><h3>Reviewer {index+1}<span>Semantic: {review.semantic} · Visual: {review.visual}</span></h3><p className="original-meaning">{review.shared_meaning||'No shared meaning recorded.'}</p><p className="original-exceptions"><strong>Exceptions: </strong>{review.exceptions||'None recorded.'}</p></article>)}<small>{entry.case_id} · fixed pair {pairIndex+1} · original views {pairIndex*2+1}–{pairIndex*2+2}</small></details>
 </div>;
}
