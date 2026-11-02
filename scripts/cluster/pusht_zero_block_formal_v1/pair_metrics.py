"""Predeclared development-only interpretation of0 versus existing8 blocks."""
import numpy as np
from base_verifier import validate_metrics


def compare(zero,eight,base,contract):
    metrics={'zero_blocks':validate_metrics(zero,base),'eight_blocks215999':validate_metrics(eight,base)}
    if zero['checkpoint_selection']!='EMA at exactly84168 updates; no best-of-development selection':
        raise ValueError('Wrong endpoint selection')
    z=np.asarray([r['max_overlap_reward'] for r in zero['episodes']])
    e=np.asarray([r['max_overlap_reward'] for r in eight['episodes']])
    d=z-e
    draws=np.random.RandomState(contract['paired_bootstrap_seed']).randint(0,len(d),(20000,len(d)))
    ci=np.quantile(d[draws].mean(1),[.025,.975])
    margin=contract['diagnostic_reward_margin']
    zero_better=bool(d.mean()>=margin and ci[0]>0 and metrics['zero_blocks']['full_reward_fraction']>=metrics['eight_blocks215999']['full_reward_fraction'])
    eight_better=bool(d.mean()<=-margin and ci[1]<0 and metrics['zero_blocks']['full_reward_fraction']<=metrics['eight_blocks215999']['full_reward_fraction'])
    interpretation=('Evidence favors removing the8-block stack in this fixed-language Push-T recipe'
        if zero_better else 'Evidence favors retaining the8-block stack in this fixed-language Push-T recipe'
        if eight_better else 'Inconclusive under the predeclared development criteria; no automatic extra budget or seeds')
    return dict(scores=metrics,zero_minus_eight_mean_reward=float(d.mean()),paired_episode_bootstrap95=ci.tolist(),
        zero_better_worse_equal=[int((d>0).sum()),int((d<0).sum()),int((d==0).sum())],
        per_seed_gain=[dict(environment_seed=r['environment_seed'],zero_minus_eight=float(v)) for r,v in zip(zero['episodes'],d)],
        zero_better_criteria_passed=zero_better,eight_better_criteria_passed=eight_better,interpretation=interpretation,
        scope='Reused dev20, one training seed, old8 trainedA6000/new0 trainedA40; whole stack includes capacity/dropout. Not final confirmation or a replacement multilingual policy.',
        main_eight_block_model_replaced=False,automatic_following_action=False,final50_accessed=False)
