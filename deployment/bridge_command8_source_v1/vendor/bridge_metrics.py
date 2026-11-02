"""Per-source masked generation errors, never invent LT Z/orientation targets."""
import numpy as np

KEYS=['path_position_normalized','path_rotation_deg','endpoint_position_normalized','endpoint_rotation_deg']


def metrics(pred,target,mask,indices):
    assert pred.ndim==4 and pred.shape[1:]==target.shape==mask.shape
    rows=[]
    for j in indices:
        p=pred[:,j].astype(np.float64);t=target[j].astype(np.float64);m=mask[j]>.5
        valid=m[:,:3].any(-1);assert valid.any()
        axes=m[valid,:3];assert (axes==axes[0]).all(),'Position coordinate availability varies within window'
        distance=np.sqrt(((p[:,valid,:3]-t[valid,:3])**2*axes).sum(-1))
        qvalid=m[:,3:7].all(-1)
        assert np.array_equal(m[:,3:7].any(-1),qvalid),'Partial quaternion target is invalid'
        rotation=None
        if qvalid.any():
            pq=p[:,qvalid,3:7];tq=t[qvalid,3:7]
            assert (np.linalg.norm(pq,axis=-1)>1e-12).all() and (np.linalg.norm(tq,axis=-1)>1e-12).all()
            pq=pq/np.linalg.norm(pq,axis=-1,keepdims=True);tq=tq/np.linalg.norm(tq,axis=-1,keepdims=True)
            rotation=np.degrees(2*np.arccos(np.clip(np.abs((pq*tq).sum(-1)),0,1)))
        row={'selection_index':int(j),'position_axes':[a for a,on in zip('xyz',axes[0]) if on],
            'path_position_normalized':float(distance.mean()),'endpoint_position_normalized':float(distance[:,-1].mean()),
            'path_rotation_deg':float(rotation.mean()) if rotation is not None else None,
            'endpoint_rotation_deg':float(rotation[:,-1].mean()) if rotation is not None else None}
        assert np.isfinite([v for k,v in row.items() if k in KEYS and v is not None]).all()
        rows.append(row)
    assert rows
    result={'windows':len(rows),'position_axes':rows[0]['position_axes'],'per_window':rows}
    assert all(row['position_axes']==result['position_axes'] for row in rows)
    for key in KEYS:
        values=[row[key] for row in rows]
        assert all(v is None for v in values) or all(v is not None for v in values)
        result[key]=None if values[0] is None else float(np.mean(values))
    return result


def summarize(outputs,targets,masks,selection):
    rows=selection['train']+selection['validation'];by_source={}
    for role in ['train','validation']:
        by_source[role]={}
        for source in ['Bridge','Language Table','BC-Z','Fractal']:
            ids=[i for i,row in enumerate(rows) if row['role']==role and row['source']==source]
            by_source[role][source]={name:metrics(p,targets,masks,ids) for name,p in outputs.items()}
    bridge=by_source['validation']['Bridge'];a,b=bridge['epsilon'],bridge['sample']
    rng=np.random.default_rng(20261028);n=a['windows']
    assert n==64 and len({rows[r['selection_index']]['group_id'] for r in a['per_window']})==n
    boot=rng.integers(n,size=(10000,n));differences={};ci={}
    for key in KEYS:
        delta=np.asarray([y[key]-x[key] for x,y in zip(a['per_window'],b['per_window'])])
        differences[key]=float(delta.mean());ci[key]=np.quantile(delta[boot].mean(1),[.025,.975]).tolist()
    return {'by_source':by_source,'primary_bridge64':bridge,'sample_minus_epsilon_bridge64':differences,
        'paired_bridge_group_bootstrap95':ci,
        'uncertainty_scope':'Reused validation groups, averaged three sampling draws, one training seed; excludes training-seed variability',
        'units':'Normalized recorded position units, divisor0.1; no new physical calibration; Language Table XY only',
        'scope':'Fixed8192-update current-main prefix, not final65536model or LIBERO/task success',
        'automatic_following_action':False,'main_model_replaced':False}
