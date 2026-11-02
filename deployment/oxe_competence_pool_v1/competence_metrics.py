"""Predeclared32training-anchor competence; no whole512pool or LIBERO claim."""
import numpy as np
KEYS=['path_position_normalized','path_rotation_deg','endpoint_position_normalized','endpoint_rotation_deg']


def motion(pred,target,mask,indices):
    amplitudes=[];directions=[];positive=[];expert=[]
    for j in indices:
        m=mask[j,:,:3]>.5;valid=m.any(-1);assert valid.any()
        a=pred[:,j,valid,:3].astype(np.float64)*m[valid];b=target[j,valid,:3].astype(np.float64)*m[valid]
        an=np.linalg.norm(a,axis=-1);bn=np.linalg.norm(b,axis=-1)
        active=(an>1e-12)&(bn[None]>1e-12)
        if active.any():
            cos=np.clip((a*b).sum(-1)[active]/(an*bn)[active],-1,1)
            directions.append(float(cos.mean()));positive.append(float((cos>0).mean()))
        amplitudes.append(float(an.mean()));expert.append(float(bn.mean()))
    magnitude=float(np.mean(amplitudes));reference=float(np.mean(expert))
    return {'windows':len(indices),'prediction_amplitude':magnitude,'expert_amplitude':reference,
        'prediction_to_expert_amplitude_ratio':magnitude/reference if reference>1e-12 else None,
        'mean_direction_cosine_nonzero_pairs':float(np.mean(directions)) if directions else None,
        'positive_direction_fraction_nonzero_pairs':float(np.mean(positive)) if positive else None,
        'scope':'Descriptive window-equal masked position norms/cosines; zero vectors excluded from direction; not an added admission gate'}


def summarize(outputs,target,mask,selection):
    from metrics import metrics
    assert set(outputs)=={'moderate_pool','replacement','source232587','static'}
    rows=selection['train']+selection['validation'];groups={};gates={};control_gates={};movements={}
    for part in ['train','validation']:
        groups[part]={};movements[part]={}
        for source in ['Bridge','Language Table','BC-Z','Fractal']:
            ids=[i for i,row in enumerate(rows) if row['role']==part and row['source']==source]
            groups[part][source]={name:metrics(pred,target,mask,ids) for name,pred in outputs.items()}
            movements[part][source]={name:motion(pred,target,mask,ids) for name,pred in outputs.items()}
            if part=='train':
                a=groups[part][source];assert len(ids)==8
                gates[source]={k:a['moderate_pool'][k]<a['static'][k] for k in KEYS if a['static'][k] is not None}
                control_gates[source]={k:a['moderate_pool'][k]<=a['replacement'][k] for k in KEYS if a['replacement'][k] is not None}
    comparisons={};rng=np.random.default_rng(20261029)
    for source,g in groups['train'].items():
        boot=rng.integers(8,size=(10000,8));comparisons[source]={}
        for base in ['replacement','static','source232587']:
            comparisons[source][base]={}
            for key in KEYS:
                if g[base][key] is None:continue
                d=np.asarray([a[key]-b[key] for a,b in zip(g['moderate_pool']['per_window'],g[base]['per_window'])])
                comparisons[source][base][key]={'mean_difference':float(d.mean()),
                    'paired_training_group_bootstrap95':np.quantile(d[boot].mean(1),[.025,.975]).tolist()}
    basic=all(all(v.values()) for v in gates.values());matched=all(all(v.values()) for v in control_gates.values())
    return {'by_source':groups,'motion':movements,'training_basic_gates_by_source':gates,'training_no_worse_control_gates':control_gates,
        'all_source_training_anchor_basic_passed':basic,'all_source_training_anchor_no_worse_control':matched,
        'limited_training_anchor_support':basic and matched,'training_anchor_comparisons':comparisons,
        'scope':'Predeclared32seen-training anchors within512pool; not all512evaluated, not generalization. Reused88development windows descriptive only. One source training seed, different allocation from old control; bootstrap excludes training-seed uncertainty.',
        'position_units':'Normalized recorded source units; LT XY only; no new calibration',
        'interpretation':'Only pool membership/diversity/exposure change; not pure exposure causal proof or main-model admission',
        'main_model_replaced':False,'automatic_following_action':False}
