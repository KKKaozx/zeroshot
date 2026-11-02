"""One missing current-pool baseline: train-only task mean vs archived readouts.

No model, ridge refit, encoding, sampling, raw-data read, or reserved test.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bridge_806_task_prior_20261017'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    assert not OUT.exists(),'Completed baseline must not be repeated or tuned'
    row_path=ROOT/'cluster_packages/bridge_demonstration_expansion_v1/rows.json'
    train_path=ROOT/'reports/bridge-demo806-fit-208207/reduced_u8192_train.npz'
    dev_path=ROOT/'reports/bridge-demo806-fit-208207/reduced_u8192_development.npz'
    readout_path=ROOT/'reports/bridge_806_context_readout_20261017/report.json'
    prediction_path=ROOT/'reports/bridge_806_context_readout_20261017/predictions_and_coefficients.npz'
    audit_path=ROOT/'reports/bridge_demo806_analysis_208207/analysis.json'
    readout_audit_path=ROOT/'reports/bridge_806_context_readout_20261017/independent_saved_array_audit.json'
    audit=read(audit_path);readout=read(readout_path);ra=read(readout_audit_path)
    assert audit['verified'] and readout['verified'] and ra['verified']
    assert sha(readout_path)==ra['report_sha256']
    for p in (train_path,dev_path):assert sha(p)==audit['downloaded_files'][p.name]['sha256']
    assert sha(prediction_path)==readout['output_sha256'][prediction_path.name]
    rows=read(row_path)
    ids=lambda split:{(r['shard'],r['record_index']) for r in rows[split]}
    assert len(rows['train'])==3624 and len(ids('train'))==806
    assert len(rows['development'])==40 and len(ids('development'))==10
    assert not ids('train') & ids('development')
    OUT.mkdir()
    plan=dict(question='How much held-out XYZ advantage does the existing806-context linear readout retain over a predictor that knows only the task name?',
        difference_from_old='Task means were assessed on128 training windows in20261013/15. No806-pool task-mean baseline was included in the current context-only readout. No old baseline or ridge fit is rerun.',
        sources={str(p.relative_to(ROOT)):sha(p) for p in (row_path,train_path,dev_path,readout_path,prediction_path,audit_path,readout_audit_path,Path(__file__))},
        rule='Exactly one arithmetic mean XYZ trajectory for each of5 tasks using all its806-pool training windows with equal window weights,16 future targets,centimeters. Task comes from the known instruction/task label. No image,state,phase,index,episode ID or development label used to construct predictions.',
        archived_readouts='Compare both fixed archived8192/16170 global readouts; no fitting or choosing a winner.',
        evaluation='All40 development windows/10 disjoint episodes. Window and episode-equal path/endpoint position,all5 tasks. Residual squared-error identity relative to task mean is descriptive only.',
        interpretation_gate='Readout supplies incremental position evidence beyond task-only prior when BOTH window and episode-equal path error are lower and at least6/10 episodes improve. No task-success/generalization-fix verdict. Known task label may be stronger than model task inference.',
        limits=['Reused development set,not independent final testing.','Context also includes current state and history; advantage cannot be uniquely attributed to image or Adapter.','A task-only prior ignores actual scene; lower mean error alone does not make it a valid executable controller.','XYZ only; no rotation/gripper/closed-loop claims.'],
        stopping='One baseline, no per-task weighting variants,phase baselines,parameter sweep,new ridge fit,NN/GPU jobs or reserved-test reads.')
    write(OUT/'plan.json',plan)
    # No target values are read until the complete fixed plan above is saved.
    with np.load(train_path,allow_pickle=False) as z:train=z['targets'][...,:3].astype(np.float64)*10
    with np.load(dev_path,allow_pickle=False) as z:truth=z['targets'][...,:3].astype(np.float64)*10
    tasks=sorted({r['task'] for r in rows['train']})
    assert len(tasks)==5 and set(r['task'] for r in rows['development'])==set(tasks)
    centers={t:train[[i for i,r in enumerate(rows['train']) if r['task']==t]].mean(0) for t in tasks}
    prior=np.stack([centers[r['task']] for r in rows['development']])
    ep_groups=defaultdict(list);task_groups=defaultdict(list)
    for i,row in enumerate(rows['development']):
        ep_groups[row['shard']+'::'+str(row['record_index'])].append(i)
        task_groups[row['task']].append(i)
    def summary(e):
        def mean(at):return dict(path_position_cm=float(e[at].mean()),endpoint_position_cm=float(e[at,-1].mean()))
        ep=[dict(episode=k,task=rows['development'][ix[0]]['task'],windows=len(ix),**mean(ix)) for k,ix in ep_groups.items()]
        return dict(window_weighted=mean(list(range(40))),
                    episode_equal={k:float(np.mean([r[k] for r in ep])) for k in ('path_position_cm','endpoint_position_cm')},
                    per_episode=ep,per_task=[dict(task=k,windows=len(ix),**mean(ix)) for k,ix in sorted(task_groups.items())])
    eprior=np.linalg.norm(prior-truth,axis=-1)
    summaries={'task_only_mean':summary(eprior)}
    paired={};gates={};decomposition={};windows=[]
    with np.load(prediction_path,allow_pickle=False) as z:
        for stage in ('8192','16170'):
            pred=z[f'u{stage}_development_prediction_cm']
            error=np.linalg.norm(pred-truth,axis=-1)
            s=summary(error);summaries['readout_'+stage]=s
            for weighting in ('window_weighted','episode_equal'):
                for key in ('path_position_cm','endpoint_position_cm'):
                    assert abs(s[weighting][key]-readout['results'][stage]['summaries']['development']['linear'][weighting][key])<1e-10
            paired[stage]=[]
            for a,b in zip(s['per_episode'],summaries['task_only_mean']['per_episode']):
                assert a['episode']==b['episode']
                paired[stage].append(dict(episode=a['episode'],task=a['task'],windows=a['windows'],
                    readout_cm=a['path_position_cm'],task_only_cm=b['path_position_cm'],delta_cm=a['path_position_cm']-b['path_position_cm']))
            better=sum(r['delta_cm']<0 for r in paired[stage])
            gates[stage]=dict(window_better=s['window_weighted']['path_position_cm']<summaries['task_only_mean']['window_weighted']['path_position_cm'],
                episode_equal_better=s['episode_equal']['path_position_cm']<summaries['task_only_mean']['episode_equal']['path_position_cm'],
                episodes_better=better,incremental_evidence=False)
            gates[stage]['incremental_evidence']=gates[stage]['window_better'] and gates[stage]['episode_equal_better'] and better>=6
            # E||f-y||² = E||m-y||² + E||f-m||² -2E[(y-m).(f-m)].
            b=truth-prior;d=pred-prior
            prior_sq=np.sum(b*b,axis=-1);departure_sq=np.sum(d*d,axis=-1);alignment=2*np.sum(b*d,axis=-1)
            reconstructed=prior_sq+departure_sq-alignment
            actual=np.sum((pred-truth)**2,axis=-1)
            delta=float(np.max(abs(reconstructed-actual)));assert delta<1e-9
            decomposition[stage]=dict(task_prior_mse_cm2=float(prior_sq.mean()),
                readout_departure_mse_cm2=float(departure_sq.mean()),twice_residual_alignment_cm2=float(alignment.mean()),
                readout_mse_cm2=float(actual.mean()),identity_max_abs_difference_cm2=delta,
                scope='Finite40-window640-target identity,not a causal or population variance decomposition.')
            for i,row in enumerate(rows['development']):
                windows.append(dict(stage=stage,episode=row['shard']+'::'+str(row['record_index']),task=row['task'],
                    start_index=row['start_index'],readout_cm=float(error[i].mean()),task_only_cm=float(eprior[i].mean())))
    np.savez_compressed(OUT/'task_prior_predictions.npz',task_names=np.asarray(tasks),
                        training_task_means_cm=np.stack([centers[t] for t in tasks]),development_prediction_cm=prior)
    for stage,records in paired.items():
        with (OUT/f'episodes_{stage}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with (OUT/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(windows[0]));w.writeheader();w.writerows(windows)
    result=dict(verified=True,plan_sha256=sha(OUT/'plan.json'),source_sha256=sha(Path(__file__)),summaries=summaries,
                gates=gates,residual_decomposition=decomposition,training_windows_per_task={t:sum(r['task']==t for r in rows['train']) for t in tasks},
                neural_training=False,ridge_refit=False,new_inference=False,gpu_used=False,reserved_test_read=False,
                limitations=plan['limits'])
    write(OUT/'report.json',result)
    print(json.dumps(dict(means={k:{g:s[g] for g in ('window_weighted','episode_equal')} for k,s in summaries.items()},gates=gates,
                         tasks={k:s['per_task'] for k,s in summaries.items()},decomposition=decomposition),indent=2))


if __name__=='__main__':main()
