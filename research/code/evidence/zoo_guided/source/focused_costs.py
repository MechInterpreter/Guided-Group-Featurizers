"""Synchronized forward timing only; no new training, readouts, rank or MDL sweep."""
import gc,time
import numpy as np
import torch
from coco_baselines import training
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched
from scale_models import model
from focused_evaluate import verify_freeze

@torch.inference_mode()
def measure(m,x,device):
    m.eval();xb=torch.as_tensor(np.array(x[:196],copy=True),device=device)
    for _ in range(5):m(xb)
    if device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(30):
        if device.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter();m(xb)
        if device.startswith('cuda'):torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    return dict(core_parameters=sum(p.numel() for p in m.parameters()),
        median_ms_per_196_patch_image=1000*float(np.median(times)),p95_ms=1000*float(np.quantile(times,.95)),
        peak_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30 if device.startswith('cuda') else None,
        scope='Synchronized featurizer forward, one image196 patches,5 warm-up/30 timed repetitions. Excludes backbone, masks, input loading, readout fitting and training.')

def run(ctx):
    s=ctx['store'];frozen=verify_freeze(ctx);old=s.load('costs_complete')
    if old is not None:return old
    x=ctx['packs']['train']['x'].reshape(-1,ctx['packs']['train']['x'].shape[-1]);records=[]
    for row in frozen['models']:
        key='cost_'+row['key'];saved=s.load(key)
        if saved is not None and saved['weights_sha']!=row['weights_sha']:raise ValueError('Timing model identity changed')
        if saved is None:
            state=memory.required(s,row['key'])
            if training.weight_sha(state['weights'])!=row['weights_sha']:raise ValueError('Timing weights changed')
            m=model(row['method'],row['groups'],row['rank'],row['active_groups'],x.shape[-1],row['coefficient']).to(ctx['device'])
            m.load_state_dict(state['weights']);del state;gc.collect()
            saved=dict(kind=row['kind'],seed=row['seed'],weights_sha=row['weights_sha'],training_head_parameters=row['head_parameters'],
                additional_optimizer_step_seconds=row.get('additional_optimizer_step_seconds'),**measure(m,x,ctx['device']))
            # Training seconds are actual saved optimizer-step times, not wall
            # time including mapping/evaluation or an equal-compute claim.
            s.save(key,saved);del m;gc.collect();torch.cuda.empty_cache()
        records.append(saved);matched.csv_write(s.results/'forward_costs.csv',records)
        s.event('forward_cost',kind=row['kind'],seed=row['seed'],median_ms=saved['median_ms_per_196_patch_image'])
    out=dict(complete=True,models=len(records),gradient_updates=0,extra_holdout_inference=False,records=records)
    s.save('costs_complete',out);s.write('forward_cost_summary.json',out);return out
