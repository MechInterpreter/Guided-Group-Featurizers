from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
H=Path(__file__).parent;DATA=H;OUT=H/'reproduced';OUT.mkdir(exist_ok=True)
METHODS=['Vanilla BSF','Supervised Vanilla','GGF','Concept-SAE']
def clean(x):return 'Concept-SAE' if x.startswith('Concept-SAE') else x
frames={}
for cohort in ['coco','voc']:
 f=pd.read_csv(DATA/(cohort+'_per_run.csv'));f['method']=f.method.map(clean);frames[cohort]=f
 summary=json.loads((DATA/(cohort+'_summary.json')).read_text())
 for record in summary['means']:
  method=clean(record['label']);q=f[f.method==method]
  assert len(q)==3
  for col,key in [('capture','capture'),('patch_AP','patch_AP'),('VE','VE')]:
   assert abs(q[col].mean()-record[key]['mean'])<1e-10
   assert abs(q[col].std(ddof=1)-record[key]['sd'])<1e-10
def events(cohort):return [json.loads(s) for s in (DATA/(cohort+'_events.jsonl')).read_text().splitlines() if s]
timings={}
for cohort in ['coco','voc']:
 by={}
 for e in events(cohort):
  if e.get('key') and 'seconds_per_update' in e:by[e['key']]=e
 timings[cohort]=by
def measured(by,key,updates):
 q=by.get(key)
 if q is None or q['update']!=updates:return None
 return q['update']*q['seconds_per_update']
costs=[];stage_rows=[]
for cohort in ['coco','voc']:
 by=timings[cohort]
 for seed in [21,22,23]:
  for method in METHODS:
   if method=='Concept-SAE':
    report=json.loads((DATA/(cohort+'_CS'+str(seed)+'.json')).read_text())['report']
    stages=report['stage_receipts'];assert len(stages)==3 and all(s['epochs_completed']==n for s,n in zip(stages,[30,50,30]))
    secs=sum(s['seconds'] for s in stages)
    for i,s in enumerate(stages,1):stage_rows.append(dict(cohort=cohort,method=method,seed=seed,stage=i,seconds=s['seconds'],updates=s['updates'],best_epoch=s['best_epoch'],epochs=s['epochs_completed']))
    params=report['model_parameters'];covered=True
   else:
    prefix=f'g1024_k16_s{seed}_';warm=prefix+('mask_warm' if method=='GGF' else 'released_vanilla_warm')
    key=prefix+('mask_lr3000_p10000_agf_g_c040' if method=='GGF' else 'released_vanilla_lr3000_p10000_released_vanilla' if method=='Vanilla BSF' else 'released_vanilla_fixed_supervision_candidate1')
    w=measured(by,warm,7640);g=measured(by,key,3000);covered=w is not None and g is not None;secs=w+g if covered else None
    if w is not None:stage_rows.append(dict(cohort=cohort,method=method,seed=seed,stage='warm',seconds=w,updates=7640,best_epoch=None,epochs=None))
    if g is not None:stage_rows.append(dict(cohort=cohort,method=method,seed=seed,stage='continuation',seconds=g,updates=3000,best_epoch=None,epochs=None))
    params=int(frames[cohort].query('method==@method and seed==@seed').parameters.iloc[0])
   costs.append(dict(cohort=cohort,method=method,seed=seed,learned_parameters=params,update_loop_seconds=secs,complete_loop_coverage=covered))
cost=pd.DataFrame(costs);cost.to_csv(OUT/'resource_per_run.csv',index=False);pd.DataFrame(stage_rows).to_csv(OUT/'stage_timing_receipts.csv',index=False)
assert cost[cost.cohort=='voc'].complete_loop_coverage.all()
voc_log=[json.loads(s) for s in (DATA/'voc_cost.jsonl').read_text().splitlines() if s]
peaks={}
for method,kind in [('Vanilla BSF','released_vanilla'),('Supervised Vanilla','supervised_vanilla'),('GGF','agf_g')]:
 rows=[r for r in voc_log if r['stage'].startswith('voc_') and r['stage'].endswith('_'+kind+'_endpoint')]
 assert rows and all(r['device']=='Tesla T4' for r in rows)
 peaks[method]=dict(allocated_GiB=max(r['peak_allocated_bytes'] for r in rows)/2**30,reserved_GiB=max(r['peak_reserved_bytes'] for r in rows)/2**30,invocations=len(rows))
r=json.loads((DATA/'voc_concept_cost.jsonl').read_text().splitlines()[-1]);assert r['device']=='Tesla T4'
peaks['Concept-SAE']=dict(allocated_GiB=r['peak_allocated_bytes']/2**30,reserved_GiB=r['peak_reserved_bytes']/2**30,invocations=1)
resources=[]
for method in METHODS:
 q=cost[(cost.cohort=='voc')&(cost.method==method)]
 resources.append(dict(method=method,parameters_M=q.learned_parameters.iloc[0]/1e6,loop_min_mean=q.update_loop_seconds.mean()/60,loop_min_SD=q.update_loop_seconds.std(ddof=1)/60,**peaks[method]))
pd.DataFrame(resources).to_csv(OUT/'voc_resources.csv',index=False)
slics=json.loads((DATA/'slics_training_receipts.json').read_text())
small_by={r['key']:r for r in slics['ggf_timings']}
small_rows=[]
for r in slics['slics']:
 seed=r['seed'];warm=small_by[f'small75_s{seed}_warm'];guided=small_by[f'small75_s{seed}_lr10000_ggf']
 assert (warm['update'],guided['update'])==(7640,3000)
 small_rows.extend([dict(method='SLiCS',seed=seed,parameters=691200,loop_min=r['training_seconds']/60),dict(method='GGF',seed=seed,parameters=1383452,loop_min=(warm['update']*warm['seconds_per_update']+guided['update']*guided['seconds_per_update'])/60)])
small=pd.DataFrame(small_rows);small.to_csv(OUT/'slics_resource_per_run.csv',index=False)
small_means=[dict(method=m,parameters=int(small[small.method==m].parameters.iloc[0]),loop_min_mean=small[small.method==m].loop_min.mean(),loop_min_SD=small[small.method==m].loop_min.std(ddof=1)) for m in ['SLiCS','GGF']]
derived=dict(VOC=json.loads((DATA/'voc_summary.json').read_text()),COCO=json.loads((DATA/'coco_summary.json').read_text()),VOC_resources=resources,
 VOC_parameter_ratio=92000983/18886696,COCO_parameter_ratio=175125423/18886808,
 VOC_loop_ratio_Concept_to_GGF=resources[-1]['loop_min_mean']/resources[2]['loop_min_mean'],
 resource_scope='T4 implementation timings under each method protocol; BSF timers exclude pre-loop batch/noise preparation while Concept timers include internal microbatch handling. Both exclude validation, checkpointing, feature extraction and hyperparameter search. Shared Vanilla warm-up is charged to both individual methods; selected fixed-weight candidate only. Peaks are observed allocated high-water marks over logged fit invocations, not standardized memory/inference benchmarks.',
 SLiCS_comparison_resources=small_means,
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.iterdir() if p.is_file()})
(OUT/'derived.json').write_text(json.dumps(derived,indent=2))
tex=[r'\begin{table}[t]\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\caption{T4 resources in two separate studies. Time: recorded update loops (minutes, mean $\pm$ SD), excluding validation and saving. GPU peaks: allocated/reserved GiB; NR: not recorded. Protocols and timer boundaries differ (Appendix~\ref{app:compute-reporting}).}',r'\label{tab:compute-main}',r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Method & Params (M) & Update min & GPU peaks\\\midrule',r'\multicolumn{4}{l}{VOC: 1,024 groups for BSF arms}\\']
for r in resources:tex.append(f"{r['method']} & {r['parameters_M']:.2f} & ${r['loop_min_mean']:.2f}\\pm{r['loop_min_SD']:.2f}$ & {r['allocated_GiB']:.2f}/{r['reserved_GiB']:.2f}\\\\")
tex += [r'\midrule\multicolumn{4}{l}{COCO: separate 75-group SLiCS study}\\']
for r in small_means:tex.append(f"{r['method']} & {r['parameters']/1e6:.2f} & ${r['loop_min_mean']:.2f}\\pm{r['loop_min_SD']:.2f}$ & NR\\\\")
tex += [r'\bottomrule\end{tabular}',r'\end{table}']
(OUT/'table_compute.tex').write_text('\n'.join(tex)+'\n')
# Performance versus cost, with each dataset's own protocol and cohort.
colors={'GGF':'#008b80','Vanilla BSF':'#89969b','Supervised Vanilla':'#c28a24','Concept-SAE':'#7657a6'}
fig,grid=plt.subplots(2,2,figsize=(7.0,5.4));axs=grid.ravel()
for ax,cohort in zip(axs[:2],['coco','voc']):
 f=frames[cohort]
 for method in METHODS:
  q=f[f.method==method];x=q.parameters.iloc[0]/1e6;y=q.patch_AP.mean();sd=q.patch_AP.std(ddof=1)
  ax.errorbar(x,y,yerr=sd,fmt='o',ms=5,color=colors[method],capsize=3,label=method)
  # Separate the equal-parameter BSF labels vertically, without moving data.
  dy={'GGF':7,'Vanilla BSF':-11,'Supervised Vanilla':-11,'Concept-SAE':7}[method]
  dx=-5 if method=='Concept-SAE' else 6;ha='right' if method=='Concept-SAE' else 'left'
  ax.annotate(method,(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=7,ha=ha)
 ax.set_xscale('log');ax.set_xlim(13,250 if cohort=='coco' else 135);ax.set_ylim(20,102);ax.set_title(cohort.upper(),fontsize=10);ax.set_xlabel('Learned parameters (millions)',fontsize=8)
 ax.xaxis.set_major_locator(FixedLocator([20,50,100,200] if cohort=='coco' else [20,50,100]));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));ax.xaxis.set_minor_locator(NullLocator())
axs[0].set_ylabel('Patch AP (%)',fontsize=9)
for r in resources:
 q=frames['voc'][frames['voc'].method==r['method']]
 axs[2].errorbar(r['loop_min_mean'],q.patch_AP.mean(),xerr=r['loop_min_SD'],yerr=q.patch_AP.std(ddof=1),fmt='o',color=colors[r['method']],capsize=3,ms=5)
 axs[2].annotate(r['method'],(r['loop_min_mean'],q.patch_AP.mean()),xytext=(5,5 if r['method']!='Supervised Vanilla' else -13),textcoords='offset points',fontsize=7)
axs[2].set_xlim(5,24);axs[2].set_ylim(60,101);axs[2].set_title('VOC: recorded update loops',fontsize=10);axs[2].set_xlabel('Update-loop time (minutes)',fontsize=8)
for r in small_means:
 old=pd.read_csv(DATA/'slics_means.csv');q=old[(old.arm==('slics_final_partial' if r['method']=='SLiCS' else 'matched_ggf_final'))&(old.seed_group=='all_five')].iloc[0]
 y=q.patch_AP_pct;sd=q.patch_AP_pct_sd
 axs[3].errorbar(r['loop_min_mean'],y,xerr=r['loop_min_SD'],yerr=sd,fmt='o',color=colors.get(r['method'],'#7657a6'),capsize=3,ms=5)
 axs[3].annotate(r['method'],(r['loop_min_mean'],y),xytext=(5,6),textcoords='offset points',fontsize=7)
axs[3].set_xlim(0,150);axs[3].set_ylim(45,82);axs[3].set_title('COCO: 75-group comparison',fontsize=10);axs[3].set_xlabel('Update-loop time (minutes)',fontsize=8)
axs[2].set_ylabel('Patch AP (%)',fontsize=9)
for ax in axs:
 ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8);ax.grid(alpha=.15);ax.set_axisbelow(True)
fig.tight_layout(w_pad=2)
for ext in ['pdf','png']:fig.savefig(OUT/('ggf_final_resources.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
print(pd.DataFrame(resources).to_string(index=False));print('VOC Concept-SAE gains:',derived['VOC']['paired_metric_differences'][-1]);print('VOC parameter ratio:',derived['VOC_parameter_ratio'])
print('SLiCS comparison resources:',small_means)
