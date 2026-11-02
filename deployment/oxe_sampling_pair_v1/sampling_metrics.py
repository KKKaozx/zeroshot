"""Sampling-policy effect separated from reaching basic static competence."""
import numpy as np

KEYS=['path_position_normalized','path_rotation_deg','endpoint_position_normalized','endpoint_rotation_deg']


def summarize(outputs,target,mask,selection):
    from metrics import metrics
    assert set(outputs)=={'replacement','cycle','source232587','static'}
    rows=selection['train']+selection['validation'];groups={}
    for part in ['train','validation']:
        groups[part]={}
        for source in ['Bridge','Language Table','BC-Z','Fractal']:
            ids=[i for i,r in enumerate(rows) if r['role']==part and r['source']==source]
            groups[part][source]={name:metrics(pred,target,mask,ids) for name,pred in outputs.items()}
    primary=groups['validation']['Bridge'];assert primary['cycle']['windows']==64
    assert len({rows[r['selection_index']]['group_id'] for r in primary['cycle']['per_window']})==64
    rng=np.random.default_rng(20261029);boot=rng.integers(64,size=(10000,64));comparisons={}
    for baseline in ['replacement','source232587','static']:
        comparisons['cycle_minus_'+baseline]={}
        for key in KEYS:
            d=np.array([a[key]-b[key] for a,b in zip(primary['cycle']['per_window'],primary[baseline]['per_window'])])
            comparisons['cycle_minus_'+baseline][key]={'mean_difference':float(d.mean()),
                'paired_group_bootstrap95':np.quantile(d[boot].mean(1),[.025,.975]).tolist()}
    c=primary['cycle'];r=primary['replacement'];s=primary['static'];old=primary['source232587']
    gates={'bridge_path_position_better_control':c[KEYS[0]]<r[KEYS[0]],
        'bridge_other_three_metrics_no_worse_control':all(c[k]<=r[k] for k in KEYS[1:]),
        'bridge_path_position_difference_ci_upper_negative':comparisons['cycle_minus_replacement'][KEYS[0]]['paired_group_bootstrap95'][1]<0}
    competence={'bridge_all_four_better_static':all(c[k]<s[k] for k in KEYS),
        'bridge_all_four_no_worse_source':all(c[k]<=old[k] for k in KEYS)}
    regressions={source:[k for k in KEYS if g['cycle'][k] is not None and g['cycle'][k]>g['replacement'][k]]
        for source,g in groups['validation'].items()}
    return {'by_source':groups,'primary_bridge64':primary,'paired_comparisons':comparisons,
        'sampling_effect_gates':gates,'sampling_effect_gate_passed':all(gates.values()),
        'basic_competence_gates':competence,'basic_competence_gate_passed':all(competence.values()),
        'validation_source_regression_flags':regressions,
        'candidate_for_review_only':all(gates.values()) and all(competence.values()) and not any(regressions.values()),
        'scope':'One source training seed/checkpoint; repeateddev64groups/3sampling draws; source10240is unequal-budget regression reference, not causal arm',
        'position_units':'Normalized recorded source units; LT XY only; no new calibration',
        'main_model_replaced':False,'automatic_following_action':False}
