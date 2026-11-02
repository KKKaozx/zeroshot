"""One fixed linear readout on existing train-only features; no model inference."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/panda_feature_readout_20261015'
PROBE=ROOT/'reports/panda_initial_approach_probe_20261015'
FIT=ROOT/'reports/panda_initial_only_fit_20261015'
LAMBDA=1e-3


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def errors(pred,target):
    distance=np.linalg.norm((pred-target)*100,axis=-1)
    return dict(path_position_cm=float(distance.mean()),endpoint_position_cm=float(distance[:,-1].mean()),
        future_step12_position_cm=float(distance[:,11].mean()),max_window_path_cm=float(distance.mean(1).max()))


def ridge(x,y,test):
    # Every statistic is fitted to the current training fold only. Scalar feature
    # scale equalizes total variance across representations without selecting axes.
    mean=x.mean(0);scale=np.sqrt(np.square(x-mean).sum(1).mean())
    assert scale>1e-12
    z=(x-mean)/scale;query=(test-mean)/scale;my=y.mean(0);centered=y-my
    dual=query @ z.T @ np.linalg.solve(z@z.T+LAMBDA*np.eye(len(z)),centered)+my
    u,s,v=np.linalg.svd(z,full_matrices=False)
    svd=(query@v.T) @ ((s/(s*s+LAMBDA))[:,None]*(u.T@centered))+my
    discrepancy=float(np.abs(dual-svd).max());assert discrepancy<1e-9
    return dual,discrepancy


def main():
    OUT.mkdir(exist_ok=False)
    p=read(PROBE/'plan.json');r=read(PROBE/'report.json');f=read(FIT/'plan.json')
    assert read(PROBE/'independent_audit.json')['verified'] and read(FIT/'independent_audit.json')['verified']
    assert read(FIT/'reload_report.json')['predictions_exact']
    assert sha(PROBE/'outputs.npz')==r['output_sha256']
    assert f['selection']==p['selection'][:16] and all(row['split']=='train' and row['start']==0 for row in f['selection'])
    sources=[PROBE/'outputs.npz',PROBE/'plan.json',FIT/'final_train_predictions.npz',FIT/'plan.json',FIT/'reload_report.json']
    manifest=dict(stage='train_feature_linear_readout',source_sha256={str(s.relative_to(ROOT)):sha(s) for s in sources},
        entry_sha256=sha(Path(__file__)),selection=f['selection'],ridge_lambda=LAMBDA,
        target='16 future tool-relative XYZ values in meters; target data never enters input features',
        feature_definitions={'clip_pooled':'.5*(saved frozen CLIP CLS + saved patch-token mean), before Adapter',
            'adapter_mixed':'Saved1024D output from original8192-update model',
            'adapter_initial_only':'Saved1024D output from1024-update initial-stage capacity check'},
        algorithm='Centered ridge, scalar total-variance normalization on training fold, fixed lambda1e-3, no search',
        validation='Leave1 of16 original training positions out from the auxiliary readout fit; not untouched policy generalization',
        negative_control='Within each training fold rotate feature rows by7 while preserving label order; omitted label stays excluded',
        duplicate_review='Prior Bridge cosine nearest-neighbor and Panda image swaps do not compare these3 representations with the same linear readout',
        stop='One fixed readout, no lambda/model/feature search after results',
        neural_training_updates=0,model_inference=False,images_reencoded=False,development_used=False,test_used=False,
        storage='Small numerical predictions/plots only; no checkpoints or downloads')
    write(OUT/'plan.json',manifest)
    with np.load(PROBE/'outputs.npz') as z:
        target=z['targets'][:16,:,:3].astype(float)*.1
        visual=z['visual_cls_patch_mean'][:16].astype(float)
        features={'clip_pooled':.5*(visual[:,:1024]+visual[:,1024:]),'adapter_mixed':z['contexts'][:16].astype(float)}
        old=z['saved_native_predictions'][:,:16,:,:3].astype(float)*.1
    with np.load(FIT/'final_train_predictions.npz') as z:
        assert np.array_equal(target,z['targets'][...,:3].astype(float)*.1)
        features['adapter_initial_only']=z['contexts'].astype(float)
        specialist=z['predictions'][...,:3].astype(float)*.1
    assert all(x.shape==(16,1024) and np.isfinite(x).all() for x in features.values())
    y=target.reshape(16,-1);allrows=np.arange(16);predictions={'targets_m':target};result={};per_window=[];delta=0.
    mean_loo=np.stack([y[allrows!=i].mean(0) for i in allrows]).reshape(target.shape)
    baseline=errors(mean_loo,target);predictions['fold_mean']=mean_loo
    for name,x in features.items():
        resub,d=ridge(x,y,x);delta=max(delta,d)
        correct=[];shuffled=[]
        for i in allrows:
            train=allrows[allrows!=i]
            a,d=ridge(x[train],y[train],x[i:i+1]);delta=max(delta,d);correct.append(a[0])
            b,d=ridge(np.roll(x[train],7,axis=0),y[train],x[i:i+1]);delta=max(delta,d);shuffled.append(b[0])
        correct=np.asarray(correct).reshape(target.shape);shuffled=np.asarray(shuffled).reshape(target.shape)
        result[name]=dict(resubstitution=errors(resub.reshape(target.shape),target),leave_one_position_out=errors(correct,target),
            wrong_alignment_control=errors(shuffled,target),
            interpretations='Training fit alone is not evidence; cross-validation evaluates an auxiliary readout of original training-policy representations')
        predictions[name+'_resub']=resub.reshape(target.shape);predictions[name+'_loo']=correct;predictions[name+'_wrong']=shuffled
        for i,row in enumerate(f['selection']):
            per_window.append(dict(representation=name,episode=row['episode'],
                loo_path_cm=float(np.linalg.norm((correct[i]-target[i])*100,axis=-1).mean()),
                wrong_path_cm=float(np.linalg.norm((shuffled[i]-target[i])*100,axis=-1).mean())))
    native=dict(original_mixed={k:float(np.mean([errors(v,target)[k] for v in old])) for k in errors(old[0],target)},
        initial_only={k:float(np.mean([errors(v,target)[k] for v in specialist])) for k in errors(specialist[0],target)})
    np.savez_compressed(OUT/'predictions.npz',**predictions)
    with (OUT/'per_window.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_window[0]));writer.writeheader();writer.writerows(per_window)
    assert all(sha(ROOT/name)==h for name,h in manifest['source_sha256'].items())
    report=dict(complete=True,source_unchanged=True,folds=16,feature_dimension=1024,readouts=result,
        mean_target_leave_one_out=baseline,native_policy_comparison=native,max_dual_svd_prediction_difference_m=delta,
        output_sha256=sha(OUT/'predictions.npz'),neural_training_updates=0,model_inference=False,
        development_used=False,test_used=False,limitations=[
            'Representations were learned using these16 policy-training observations; auxiliary heldout folds are not policy-heldout data.',
            'Success proves a simple geometric readout exists in the inspected pooled representation; not a complete policy or physical grasp.',
            'Failure of one fixed linear readout does not prove information absence or uniquely blame CLIP/Adapter.',
            'This is a posthoc module diagnostic, not replacement of the required diffusion method.'])
    write(OUT/'report.json',report)
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True);x=np.arange(3)
    for offset,key,label in [(-.25,'resubstitution','Fit all16'),(0,'leave_one_position_out','Leave1 position out'),(.25,'wrong_alignment_control','Wrong feature-label alignment')]:
        ax.bar(x+offset,[v[key]['path_position_cm'] for v in result.values()],width=.25,label=label)
    ax.axhline(baseline['path_position_cm'],color='#777777',linestyle='--',label='Fold training-target mean')
    ax.set_xticks(x,['CLIP pooled','Adapter mixed stages','Adapter initial-only']);ax.set(ylabel='Auxiliary XYZ trajectory error (cm)',title='One fixed readout; original16 training positions only');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'feature_readout.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
