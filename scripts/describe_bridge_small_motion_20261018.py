"""Descriptive first-target motion strata of existing207558 saved predictions.

Future truth defines bins ONLY for diagnosis, NEVER as a deployment gate.
All windows retained; no fitted model, parameter search or changed verdict.
"""
import csv
import json
from pathlib import Path
import numpy as np
from compare_bridge_near_train_development_20261018 import macro_weights,read
from analyze_bridge_prefix_209481_20261018 import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bridge_near_train_development_20261018'


def main():
    assert OUT.is_dir() and not (OUT/'small_motion.json').exists()
    verified=read(OUT/'analysis.json');rows_out=[];overall={}
    bins=[(0.,.5),(.5,1.),(1.,2.),(2.,float('inf'))]
    for domain,path,rows_path in [('train',ROOT/'reports/bridge-coverage-rep-fit-207558/reduced_u8192_train.npz',
            ROOT/'cluster_packages/bridge_prefix_pair_v1/old_rows.json'),
        ('new_development',ROOT/'reports/bridge-prefix-formal-209481/unchanged207558_full16.npz',
            ROOT/'reports/bridge-prefix-formal-209481/validation_rows.json')]:
        assert sha(path)==verified['source_sha256'][domain]['sha256']
        rows=read(rows_path);rows=rows['train'] if domain=='train' else rows
        w=macro_weights(rows,sorted({r['task'] for r in rows}))
        with np.load(path,allow_pickle=False) as z:
            p=z['predictions'][:,:,0,:3].astype(np.float64)*10.
            t=z['targets'][:,0,:3].astype(np.float64)*10.
        true_motion=np.linalg.norm(t,axis=-1);error=np.linalg.norm(p-t,axis=-1)
        pred_motion=np.linalg.norm(p,axis=-1)
        scores=dict(model_position_cm=float(np.sum(error*w[None])/3),
            static_position_cm=float(np.sum(true_motion*w)),
            model_squared_position_error_cm2=float(np.sum(error**2*w[None])/3),
            static_squared_position_error_cm2=float(np.sum(true_motion**2*w)))
        overall[domain]=scores
        partitions=[]
        for lo,hi in bins:
            chosen=(true_motion>=lo)&(true_motion<hi)
            mass=float(w[chosen].sum());partitions.append(chosen)
            assert mass>0
            normalized=w[chosen]/mass
            rows_out.append(dict(domain=domain,true_next_displacement_bin_cm=f'[{lo:g},{hi:g})',
                windows=int(chosen.sum()),task_macro_weight=mass,
                mean_predicted_displacement_cm=float(np.sum(pred_motion[:,chosen]*normalized[None])/3),
                model_position_cm=float(np.sum(error[:,chosen]*normalized[None])/3),
                static_position_cm=float(np.sum(true_motion[chosen]*normalized)),
                model_squared_position_error_cm2=float(np.sum(error[:,chosen]**2*normalized[None])/3),
                static_squared_position_error_cm2=float(np.sum(true_motion[chosen]**2*normalized))))
        assert np.all(np.stack(partitions).sum(axis=0)==1)
        same=[r for r in rows_out if r['domain']==domain]
        assert sum(r['windows'] for r in same)==len(rows)
        for metric,value in scores.items():
            assert abs(sum(r['task_macro_weight']*r[metric] for r in same)-value)<1e-10
    result=dict(date='2026-10-18',model=207558,target=1,scope='saved predictions only',
        weighting='Same five task weights; episode equal within task; bin means condition original weights, not new episode rebalance',
        metric='Final sampled XYZ error; squared-distance diagnostic is NOT denoising training loss and excludes quaternion',
        threshold_policy='Post-hoc descriptive fixed centimeter bins, not predeclared acceptance or deployment thresholds',
        global_scores=overall,motion_bins=rows_out,
        all_windows_retained=True,GPU_used=False,model_loaded=False,training_or_sampling=False,
        raw_dataset_read=False,final25_payloads_read=False,
        no_truth_based_deployment_gate=True,adoption_unchanged=False,
        caveats=['Near-zero measured next displacement does not identify a grasp/release/hold semantic phase.',
            'Measured displacement does not establish the native commanded motion or its delay.',
            'Three samples are repeated diffusion draws, not independent training seeds.',
            'Metric rank reversal does not prove an incorrect training objective or fix generalization.'])
    # adoption_unchanged means the failed candidate still is NOT admitted.
    result['adoption_unchanged']=True;result['candidate_admitted']=False
    (OUT/'small_motion.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (OUT/'small_motion.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows_out[0]));writer.writeheader();writer.writerows(rows_out)
    print(json.dumps(dict(overall=overall,bins=rows_out),indent=2))


if __name__=='__main__':main()
