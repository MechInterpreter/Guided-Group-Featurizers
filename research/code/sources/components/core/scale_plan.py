"""Frozen schedules and development-only selection; no test metric enters here."""
import math
import numpy as np

SCREEN_SEEDS=[21,22,23]
FINAL_SEEDS=[21,22,23,24,25,26,27]
GRID=[(1024,16),(1024,32),(4096,16),(4096,32)]
CAPS=[.4,.6,1.]
METHODS=['released_vanilla','released_grassmannian','released_group_lasso',
         'paper_vanilla','paper_grassmannian','paper_group_lasso']


def settings(baselines='both',fresh=True):
    if baselines not in ('both','released','paper_plus_vanilla'):raise ValueError('Unknown baseline set')
    methods=METHODS if baselines=='both' else METHODS[:3] if baselines=='released' else [METHODS[0],*METHODS[3:]]
    return dict(version=1,width=12,screen_seeds=SCREEN_SEEDS,final_seeds=FINAL_SEEDS,
        configurations=GRID,caps=CAPS,methods=methods,baselines=baselines,fresh=bool(fresh),
        warm_updates=7640,screen_updates=3000,final_guided_updates=12000,
        total_final_updates=19640,images_per_update=8,lr=.0003,warm_lr=.0003,
        snr=.1,clip=5.,evaluate_every=500,local_save_every=50,drive_save_every=250,
        reconstruction_weight=1.,fixed_weights=[1.,.05,.01],
        AP_preservation_margin=.005,VE_preservation_margin=.001,score_tie_margin=.001,
        selector='capture + 0.2 AP; preserve AP within 0.5pp and VE within 0.1pp of the reference; prefer lower cost within 0.1pp score',
        final_holdout_images=1500,fresh_seed=126101,bootstrap=1000,
        rank_min_active_patches=32,mdl_distortions=[.01,.05,.10,.20],
        rank_and_mdl_are_diagnostics=True,baseline_learning_rates=[.0001,.0003,.001],
        paper_lasso_coefficients=[.0001,.001,.01],reference_only_prior_runs=True,
        training_schedule='fixed endpoints, common warm reconstruction + common additional-update budgets; not original paper optimizer/epoch replication')


def group_tag(groups,active):return f'g{groups}_k{active}'
def cap_tag(cap):return f'c{int(round(cap*100)):03d}'


def select(records,reference,rule):
    """Records must be complete three-seed DEVELOPMENT means."""
    if not records or reference not in [r['id'] for r in records]:raise ValueError('Missing reference')
    ref=next(r for r in records if r['id']==reference)
    for r in records:
        if r['seeds']!=SCREEN_SEEDS or not np.isfinite([r['capture'],r['ap'],r['ve']]).all():
            raise ValueError('Selection requires all three finite development scores')
    feasible=[r for r in records if r['ap']>=ref['ap']-rule['AP_preservation_margin'] and r['ve']>=ref['ve']-rule['VE_preservation_margin']]
    for r in feasible:r['selection_score']=r['capture']+.2*r['ap']
    best=max(r['selection_score'] for r in feasible)
    near=[r for r in feasible if r['selection_score']>=best-rule['score_tie_margin']]
    chosen=min(near,key=lambda r:(r['cost'],r['id']))
    return dict(chosen=chosen['id'],reference=reference,records=records,feasible=[r['id'] for r in feasible],
                rule=rule['selector'],scope='development-only heuristic; not proof of optimal configuration')


def parameter_counts(groups,width,d=768):
    return dict(free_core=2*groups*width*d+groups*width,tied_core=groups*width*d+1,
                maximum_active_coordinates=lambda k:k*width)
