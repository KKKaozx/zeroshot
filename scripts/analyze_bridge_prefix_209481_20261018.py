"""Bounded CPU analysis of SAVED209481 predictions, no model/data/weights load.

All ten development episodes retained. Never infer execution, semantic phases,
world-frame offsets or an unobserved future from these tool-relative arrays.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/bridge-prefix-formal-209481'
OUT=ROOT/'reports/bridge_prefix_failure_analysis_209481_20261018'
METHODS=('unchanged207558','control','candidate','static')
SEGMENTS={'first1':slice(0,1),'first4':slice(0,4),'full16':slice(0,16),'target16':slice(15,16)}


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def errors(pred,truth):
    # Independent acos formulation; published runner used atan2.
    p=np.asarray(pred,dtype=np.float64);t=np.asarray(truth,dtype=np.float64)
    assert p.shape[1:]==t.shape and p.shape[-1]==7
    assert np.isfinite(p).all() and np.isfinite(t).all()
    position=np.linalg.norm(p[...,:3]-t[...,:3],axis=-1)*10.
    q,r=p[...,3:7],t[...,3:7]
    qn,rn=np.linalg.norm(q,axis=-1),np.linalg.norm(r,axis=-1)
    assert qn.min()>1e-8 and rn.min()>1e-8
    dot=abs(np.sum(q*r,axis=-1)/(qn*rn))
    return position,np.rad2deg(2.*np.arccos(np.clip(dot,0.,1.)))


def weights(rows):
    identity=[r['shard']+'::'+str(r['record_index']) for r in rows]
    counts={e:identity.count(e) for e in set(identity)}
    w=np.array([1./(len(counts)*counts[e]) for e in identity])
    assert abs(w.sum()-1.)<1e-12
    return w


def average(values,w):
    # values: seeds x windows x targets, uniform seeds and targets,
    # equal episodes then equal windows inside each episode.
    return float(np.einsum('snk,n->sk',values,w).mean())


def profile(values,w):return np.einsum('snk,n->sk',values,w).mean(axis=0)


def motion_description(pred,truth,w,sl):
    p=pred[:,:,sl,:3].astype(np.float64)*10.
    t=truth[:,sl,:3].astype(np.float64)*10.
    residual=p-t
    pn,tn=np.linalg.norm(p,axis=-1),np.linalg.norm(t,axis=-1)
    mse=average(np.sum(residual**2,axis=-1),w)
    mean=p.mean(axis=0)
    spread=average(np.sum((p-mean)**2,axis=-1),w)
    mean_error=average(np.sum((mean-t)**2,axis=-1)[None],w)
    assert abs(mse-spread-mean_error)<=max(1e-10,mse*1e-12)
    signed=np.array([average(residual[...,axis],w) for axis in range(3)])
    # Directions only defined for non-negligible true AND predicted motion.
    eligible=(pn>.01)&(tn[None]>.01)
    denominator=pn*tn
    cosine=np.divide(np.sum(p*t,axis=-1),denominator,
        out=np.zeros_like(pn),where=eligible)
    eligible_mass=average(eligible.astype(float),w)
    return dict(mean_predicted_displacement_cm=average(pn,w),
        mean_true_displacement_cm=average(tn[None],w),
        signed_tool_axis_residual_cm=signed.tolist(),
        tool_axis_mean_absolute_error_cm=[average(abs(residual[...,a]),w) for a in range(3)],
        position_RMSE_cm=float(np.sqrt(mse)),
        squared_position_error_cm2=mse,
        sampling_spread_cm2=spread,sampling_spread_fraction=spread/mse if mse else 0.,
        seed_mean_position_error_cm2=mean_error,
        global_tool_offset_squared_error_fraction=float(np.dot(signed,signed)/mse) if mse else 0.,
        direction_eligible_weight=eligible_mass,
        direction_eligible_observations=int(eligible.sum()),
        direction_cosine=(average(np.where(eligible,cosine,0.),w)/eligible_mass if eligible_mass else None),
        opposite_direction_fraction=(average((eligible&(cosine<0.)).astype(float),w)/eligible_mass if eligible_mass else None))


def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=False)
    report=read(SOURCE/'report.json');rows=read(SOURCE/'validation_rows.json')
    assert report['execution_checks_passed'] and report['optimization_operations']==8192
    assert report['both_weights_frozen_before_development_scoring']
    assert not report['final25_payloads_read'] and not report['old40_or_consumed13_rescored']
    assert not report['gripper_trained'] and report['clip_unchanged'] and report['gripper_unchanged']
    assert report['protocol_sha256']==sha(ROOT/'cluster_packages/bridge_prefix_pair_v1/protocol.json')
    assert report['formal_source_sums_sha256']==sha(ROOT/'cluster_packages/bridge_prefix_formal_v1/SHA256SUMS')
    assert rows==read(ROOT/'cluster_packages/bridge_prefix_pair_v1/input_reference/validation_rows.json')
    assert len(rows)==50
    with np.load(SOURCE/'validation_inputs.npz',allow_pickle=False) as z:truth=z['targets'].copy()[...,:7]
    w=weights(rows);predictions={}
    for method in METHODS[:-1]:
        with np.load(SOURCE/f'{method}_full16.npz',allow_pickle=False) as z:
            assert np.array_equal(z['targets'][...,:7],truth)
            assert np.array_equal(z['sampling_seeds'],[1101,1102,1103])
            predictions[method]=z['predictions'].copy()
            assert predictions[method].shape==(3,50,16,7)
    identity=np.zeros((1,50,16,7));identity[...,6]=1.;predictions['static']=identity
    error_arrays={m:errors(p,truth) for m,p in predictions.items()}
    episode_ids=list(dict.fromkeys(r['shard']+'::'+str(r['record_index']) for r in rows))
    tasks=sorted({r['task'] for r in rows})
    assert len(episode_ids)==10 and len(tasks)==5
    task_rows=[];episode_rows=[];target_rows=[];all_motion={};profiles={}
    all_summaries={};maximum_report_difference=0.
    for method,(position,rotation) in error_arrays.items():
        summaries={}
        for segment,sl in SEGMENTS.items():
            summaries[segment]=dict(position_cm=average(position[:,:,sl],w),rotation_deg=average(rotation[:,:,sl],w),
                window_position_cm=float(position[:,:,sl].mean()),window_rotation_deg=float(rotation[:,:,sl].mean()))
        all_summaries[method]=summaries
        expected=report['results_full16'][method]
        for observed,saved in [(summaries['full16']['position_cm'],expected['episode_equal']['path_position_cm']),
            (summaries['full16']['rotation_deg'],expected['episode_equal']['path_rotation_deg']),
            (summaries['target16']['position_cm'],expected['episode_equal']['endpoint_position_cm']),
            (summaries['target16']['rotation_deg'],expected['episode_equal']['endpoint_rotation_deg']),
            (float(position.mean()),expected['window_weighted']['path_position_cm']),
            (float(rotation.mean()),expected['window_weighted']['path_rotation_deg'])]:
            delta=abs(observed-saved);maximum_report_difference=max(delta,maximum_report_difference);assert delta<1e-7
        profiles[method]=dict(position_cm=profile(position,w).tolist(),rotation_deg=profile(rotation,w).tolist())
        for k in range(16):
            target_rows.append(dict(method=method,target=k+1,position_episode_equal_cm=profiles[method]['position_cm'][k],
                rotation_episode_equal_deg=profiles[method]['rotation_deg'][k],
                position_window_weighted_cm=float(position[:,:,k].mean()),rotation_window_weighted_deg=float(rotation[:,:,k].mean())))
        for episode in episode_ids:
            ix=[i for i,r in enumerate(rows) if r['shard']+'::'+str(r['record_index'])==episode]
            for segment,sl in SEGMENTS.items():
                episode_rows.append(dict(method=method,episode=episode,task=rows[ix[0]]['task'],windows=len(ix),segment=segment,
                    position_cm=float(position[:,ix,sl].mean()),rotation_deg=float(rotation[:,ix,sl].mean())))
        for task in tasks:
            ix=[i for i,r in enumerate(rows) if r['task']==task];tw=weights([rows[i] for i in ix])
            for segment,sl in SEGMENTS.items():
                task_rows.append(dict(method=method,task=task,episodes=2,windows=len(ix),segment=segment,
                    position_cm=average(position[:,ix,sl],tw),rotation_deg=average(rotation[:,ix,sl],tw)))
        all_motion[method]={s:motion_description(predictions[method],truth,w,sl) for s,sl in SEGMENTS.items()}
    # Check analytical summaries against each task/episode, not only grand mean.
    for table,field,label in [(task_rows,'per_task','task'),(episode_rows,'per_episode','episode')]:
        for row in table:
            if row['segment']!='full16':continue
            saved=next(e for e in report['results_full16'][row['method']][field] if e[label]==row[label])
            if field=='per_task':saved=saved['episode_equal']
            assert abs(row['position_cm']-saved['path_position_cm'])<1e-7
            assert abs(row['rotation_deg']-saved['path_rotation_deg'])<1e-7
    auxiliary={}
    for method in ('control','candidate'):
        with np.load(SOURCE/f'{method}_K1_auxiliary.npz',allow_pickle=False) as z:
            assert z['predictions'].shape==(3,50,1,7) and np.array_equal(z['targets'][...,:7],truth[:,:1])
            pos,rot=errors(z['predictions'],truth[:,:1])
            auxiliary[method]=dict(position_cm=average(pos,w),rotation_deg=average(rot,w))
        assert abs(auxiliary[method]['position_cm']-report['auxiliary_K1'][method]['episode_equal']['path_position_cm'])<1e-7
    def wins(table,method,reference,label):
        m=[r for r in table if r['method']==method and r['segment']=='full16']
        other={r[label]:r for r in table if r['method']==reference and r['segment']=='full16'}
        return dict(position=sum(r['position_cm']<other[r[label]]['position_cm'] for r in m),
            rotation=sum(r['rotation_deg']<other[r[label]]['rotation_deg'] for r in m),
            both=sum(r['position_cm']<other[r[label]]['position_cm'] and r['rotation_deg']<other[r[label]]['rotation_deg'] for r in m))
    source_files={p.name:sha(p) for p in sorted(SOURCE.iterdir()) if p.suffix in ('.json','.npz')}
    result=dict(job=209481,date='2026-10-18',saved_arrays_only=True,GPU_used=False,model_loaded=False,
        weights_available_locally=False,training_or_inference_executed=False,raw_dataset_read=False,
        final25_payloads_read=False,episodes=10,windows=50,source_sha256=source_files,
        maximum_recomputed_summary_difference=maximum_report_difference,
        weighting='episode-equal; window-weighted also saved',summaries=all_summaries,per_target=profiles,
        original_vs_static_episode_wins=wins(episode_rows,'unchanged207558','static','episode'),
        original_vs_static_task_wins=wins(task_rows,'unchanged207558','static','task'),
        candidate_vs_original_episode_wins=wins(episode_rows,'candidate','unchanged207558','episode'),
        motion_diagnostics=all_motion,K1_auxiliary=auxiliary,
        adoption_unchanged=report['adoption'],
        caveats=['No new independent test: these10 are consumed development data.',
            'Target index is future horizon, not a semantic grasp/release phase.',
            'Jointly predicted relative poses: increasing horizon error is not evidence of recursive integration drift.',
            'Signed residuals are in each input tool frame; no world-camera calibration inferred.',
            'Mean-prediction/spread decomposition is descriptive, not a deployed ensemble, refit or improvement verdict.',
            'Three sampling seeds are not three training seeds; no causal root-cause proof from residuals.'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    write_csv('per_task.csv',task_rows);write_csv('per_episode.csv',episode_rows);write_csv('per_target.csv',target_rows)
    motion_rows=[]
    for method,v in all_motion.items():
        for segment,x in v.items():
            motion_rows.append(dict(method=method,segment=segment,
                **{k:v for k,v in x.items() if not isinstance(v,list)}))
    write_csv('motion_diagnostics.csv',motion_rows)
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':180})
    colors=dict(unchanged207558='#1670a5',control='#888888',candidate='#d77b11',static='#292929')
    names=dict(unchanged207558='Original 207558',control='Control',candidate='Added late inputs',static='No motion')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,metric,title in zip(axes,('position_cm','rotation_deg'),('Position error (cm)','Rotation error (degrees)')):
        for method in METHODS:
            ax.plot(range(1,17),profiles[method][metric],label=names[method],color=colors[method],
                linestyle='--' if method=='static' else '-',linewidth=2)
        ax.set(xlabel='Future target index',ylabel=title,xticks=[1,4,8,12,16]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Saved 209481 predictions: 10 development episodes, episode-equal errors')
    fig.savefig(OUT/'error_by_prediction_target.png');fig.savefig(OUT/'error_by_prediction_target.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    labels={'close microwave':'Close\nmicrowave','close the drawer':'Close\ndrawer','open the drawer':'Open\ndrawer',
        'take broccoli out of pan':'Remove\nbroccoli','turn lever vertical to front':'Turn\nlever'}
    for ax,metric,title in zip(axes,('position_cm','rotation_deg'),('Position error (cm)','Rotation error (degrees)')):
        for j,method in enumerate(METHODS):
            values=[next(r[metric] for r in task_rows if r['task']==task and r['method']==method and r['segment']=='full16') for task in tasks]
            ax.bar(np.arange(5)+(j-1.5)*.19,values,width=.19,color=colors[method],label=names[method])
        ax.set(xticks=range(5),xticklabels=[labels[t] for t in tasks],ylabel=title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Full16 errors by task: only two development episodes per task')
    fig.savefig(OUT/'error_by_task.png');fig.savefig(OUT/'error_by_task.pdf');plt.close(fig)
    print(json.dumps(dict(episode_wins=result['original_vs_static_episode_wins'],task_wins=result['original_vs_static_task_wins'],
        summaries=all_summaries,original_motion=all_motion['unchanged207558'],K1_auxiliary=auxiliary,
        maximum_report_difference=maximum_report_difference),indent=2))


if __name__=='__main__':main()
