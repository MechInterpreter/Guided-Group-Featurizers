'use client';
import {useEffect,useRef,useState} from 'react';
import {X} from 'lucide-react';
type Metric={mean:number;sd:number};
type Row={method:string;name:string;meaning:Metric;visual:Metric;whole:Metric;rejected?:number[]};
type Evidence={trainingOrders:number[];reviewers:Record<string,{name:string;apiId?:string;scores:Record<string,Row[]>}>;concentration:Record<string,{method:string;name:string;coverage:Metric;falsePositive:Metric;precision:Metric}[]>;sources:{file:string;sha256:string}[]};
const percentage=(metric:Metric)=>`${metric.mean.toFixed(2)} ± ${metric.sd.toFixed(2)}%`;
export default function BenchmarkEvidence({onClose}:{onClose:()=>void}){
 const [data,setData]=useState<Evidence|null>(null),[error,setError]=useState(''),[setting,setSetting]=useState('all');
 const dialog=useRef<HTMLElement>(null),close=useRef<HTMLButtonElement>(null);
 useEffect(()=>{let alive=true;fetch('/data/paper-evidence.json').then(async r=>{if(!r.ok)throw Error();return r.json() as Promise<Evidence>}).then(d=>{if(alive)setData(d)}).catch(()=>{if(alive)setError('The research evidence could not be loaded. Please reload.');});return()=>{alive=false}},[]);
 useEffect(()=>{const prior=document.activeElement as HTMLElement|null;close.current?.focus();const overflow=document.body.style.overflow;document.body.style.overflow='hidden';return()=>{document.body.style.overflow=overflow;prior?.focus()}},[]);
 return <div className="modal-backdrop" onClick={onClose}><article className="connection-dialog benchmark-dialog" data-tour="benchmark" role="dialog" aria-modal="true" aria-label="Research benchmark evidence" ref={dialog} onClick={e=>e.stopPropagation()} onKeyDown={e=>{
  if(e.key==='Escape'){e.stopPropagation();onClose()}
  if(e.key==='Tab'){const targets=[...dialog.current!.querySelectorAll<HTMLElement>('button,select,summary')];const first=targets[0],last=targets.at(-1);if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus()}}
 }}>
 <button className="modal-close icon-btn" aria-label="Close benchmark evidence" ref={close} onClick={onClose}><X size={20}/></button>
 <div className="eyebrow">REPORTED RESEARCH RESULTS</div><h2>Coherence and concentration, measured.</h2>
 <p>Seven training orders (21–27), 75 groups per method, 12 directions per group. Values are mean ± sample SD. The interactive clouds use the frozen seed-21 models; these tables summarize the paper's seven-order evaluation.</p>
 <label className="benchmark-setting">Guidance setting <select aria-label="Benchmark guidance setting" value={setting} onChange={e=>setSetting(e.target.value)}><option value="all">All category labels</option><option value="split">Category-split guidance</option></select></label>
 {error&&<p role="alert">{error}</p>}{!data&&!error&&<p role="status">Loading reported results…</p>}
 {data&&<>
 <div className="benchmark-table-pair">{Object.entries(data.reviewers).map(([key,reviewer])=><section key={key}>
  <h3>{reviewer.name}</h3>{reviewer.apiId&&<small>{reviewer.apiId}</small>}
  <div className="benchmark-scroll"><table data-reviewer={key}><caption>Blinded coherence{setting==='split'?', all groups in each split-guidance fit':''}</caption><thead><tr><th>Method</th><th>Specific pair meaning</th><th>Visual structure</th><th>Whole-group theme</th></tr></thead><tbody>{reviewer.scores[setting].map(row=><tr key={row.method} className={row.method==='ggf'?'benchmark-ggf':''}><th>{row.name}</th><td>{percentage(row.meaning)}</td><td>{percentage(row.visual)}</td><td>{percentage(row.whole)}</td></tr>)}</tbody></table></div>
  {key==='gemini'&&<p className="benchmark-footnote">Rejected pair-task / whole-group replies: {reviewer.scores[setting].map(row=>row.name+' '+row.rejected?.join(' / ')).join('; ')}. Missing judgments remain in the planned totals.</p>}
 </section>)}</div>
 <p className="benchmark-footnote">Specific meaning asks whether fixed pairs share an object type, part or material. Visual structure is judged separately. A clear whole-group theme must fit at least five of six views. Planned pairs and groups stay in the denominators, including unclear judgments. GPT and Gemini results are kept separate.</p>
 <section><h3>Coverage with one selected group per category</h3><p>{setting==='split'?'Only categories withheld from guidance are evaluated here.':'All category labels are available during guidance.'} Readouts and thresholds are selected on calibration crops and frozen before the audit.</p>
 <div className="benchmark-scroll"><table data-endpoint="concentration"><caption>Category presence on audit crops</caption><thead><tr><th>Method</th><th>Target-containing crops covered</th><th>False-positive rate</th></tr></thead><tbody>{data.concentration[setting].map(row=><tr key={row.method} className={row.method==='ggf'?'benchmark-ggf':''}><th>{row.name}</th><td>{percentage(row.coverage)}</td><td>{percentage(row.falsePositive)}</td></tr>)}</tbody></table></div></section>
 <p className="benchmark-footnote">The display cutoff and co-activation links are exploration controls, separate from these fixed research endpoints. Shared initialization and image pools make these descriptive training-order comparisons. The split study withholds supervision categories, not all visual exposure to those objects. SLiCS has no matched 75-group split-guidance condition.</p>
 <details className="benchmark-provenance"><summary>Source tables and reproducibility</summary><p>Values are imported directly from the paper's table sources. Their SHA-256 digests identify the exact source versions.</p>{data.sources.map(source=><div key={source.file}><strong>{source.file}</strong><code>{source.sha256}</code></div>)}<button className="quiet-btn" onClick={()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='ggf-reported-evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}}>Download reported evidence</button></details>
 </>}
 </article></div>;
}
