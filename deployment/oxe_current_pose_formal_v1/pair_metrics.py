"""Unchanged geometry, fixed groups; separate state effect from basic competence."""
import numpy as np

KEYS=['path_position_normalized','path_rotation_deg','endpoint_position_normalized','endpoint_rotation_deg']


def summarize(outputs,targets,masks,selection):
    from metrics import metrics
    assert set(outputs)=={'mask_control','real_pose','unchanged231297','static'}
    rows=selection['train']+selection['validation'];by_source={}
    for role in ['train','validation']:
        by_source[role]={}
        for source in ['Bridge','Language Table','BC-Z','Fractal']:
            ids=[i for i,r in enumerate(rows) if r['role']==role and r['source']==source]
            by_source[role][source]={name:metrics(p,targets,masks,ids) for name,p in outputs.items()}
    primary=by_source['validation']['Bridge'];assert all(v['windows']==64 for v in primary.values())
    assert len({rows[r['selection_index']]['group_id'] for r in primary['real_pose']['per_window']})==64
    rng=np.random.default_rng(20261028);boot=rng.integers(64,size=(10000,64));comparisons={}
    for baseline in ['mask_control','unchanged231297','static']:
        comparisons['real_minus_'+baseline]={}
        for key in KEYS:
            delta=np.asarray([a[key]-b[key] for a,b in zip(primary['real_pose']['per_window'],primary[baseline]['per_window'])])
            comparisons['real_minus_'+baseline][key]={'mean_difference':float(delta.mean()),
                'paired_group_bootstrap95':np.quantile(delta[boot].mean(1),[.025,.975]).tolist()}
    r=primary['real_pose'];c=primary['mask_control'];u=primary['unchanged231297'];s=primary['static']
    gates={'path_position_better_than_mask_control':r[KEYS[0]]<c[KEYS[0]],
        'endpoint_position_better_than_mask_control':r[KEYS[2]]<c[KEYS[2]],
        'rotation_no_worse_than_mask_control':all(r[k]<=c[k] for k in [KEYS[1],KEYS[3]]),
        'paired_path_position_interval_upper_below_zero':comparisons['real_minus_mask_control'][KEYS[0]]['paired_group_bootstrap95'][1]<0,
        'all_four_means_no_worse_than_unchanged_source':all(r[k]<=u[k] for k in KEYS),
        'all_four_means_better_than_static':all(r[k]<s[k] for k in KEYS)}
    passed=all(gates.values())
    return {'by_source':by_source,'primary_bridge64':primary,'paired_comparisons':comparisons,
        'gates':gates,'candidate_for_next_qualification':passed,
        'decision':'Candidate for next qualification only; no automatic action' if passed else 'This bounded test did not establish required competence; no automatic extra budget or variant',
        'scope':'One training seed, reused development groups; equal2048continuation for real/control, source8192only a regression reference; not LIBERO/task success',
        'uncertainty_scope':'Group bootstrap after averaging3sampling draws, not training-seed uncertainty',
        'position_units':'Normalized recorded source units, divisor0.1; no new calibration; Language Table XY only',
        'main_model_replaced':False,'automatic_following_action':False}
