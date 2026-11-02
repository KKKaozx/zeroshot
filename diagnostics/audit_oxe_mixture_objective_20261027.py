"""Existing train targets/history: source exposure, mask weights and label scales."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/oxe_mixture_objective_audit_20261027'
SOURCES=('Bridge','Language Table','BC-Z','Fractal')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def quantiles(values):
    values=np.asarray(values)
    return dict(zip(('min','p25','median','p75','p95','p99','max'),map(float,np.quantile(values,[0,.25,.5,.75,.95,.99,1]))))


def main():
    OUT.mkdir(exist_ok=False)
    folder=ROOT/'reports/oxe_nominal_targets_20261025'
    manifest=read(folder/'manifest.json')
    for name in ('actions.npy','supervision_masks.npy','window_rows.npy','episodes.json'):
        assert sha(folder/name)==manifest['artifacts'][name]['sha256'],name
    episodes=read(folder/'episodes.json')
    rows=np.load(folder/'window_rows.npy',mmap_mode='r')
    targets=np.load(folder/'actions.npy',mmap_mode='r')
    masks=np.load(folder/'supervision_masks.npy',mmap_mode='r')
    source=np.asarray([SOURCES.index(episodes[int(row[0])]['source']) for row in rows],dtype=np.int8)
    counts=masks.sum(axis=1,dtype=np.int64)
    pose=counts[:,:7].sum(axis=1)
    gripper=counts[:,7]
    exposure=np.zeros(len(rows),np.int64)
    shares=np.zeros((65536,4),np.float64)
    gripshares=np.zeros_like(shares)
    no_bridge=0
    training_path=ROOT/'reports/oxe_noaug_seed42_227223/cluster/reports/step_report.json'
    history=read(training_path)['updates']
    assert len(history)==65536
    for j,item in enumerate(history):
        assert item['update']==j+1
        ids=np.asarray(item['indices'],np.int64)
        assert np.array_equal(counts[ids].sum(axis=0),item['valid_dimensions'])
        np.add.at(exposure,ids,1)
        s=source[ids]
        if not np.any(s==0):no_bridge+=1
        total=pose[ids].sum();assert total>0
        total_g=gripper[ids].sum()
        for k in range(4):
            shares[j,k]=pose[ids][s==k].sum()/total
            if total_g:gripshares[j,k]=gripper[ids][s==k].sum()/total_g
    old=read(ROOT/'reports/oxe_conditioning_sampling_audit_20261027/attempt1/existing_training_coverage_and_text_overlap.json')
    result={}
    for k,name in enumerate(SOURCES):
        ids=np.flatnonzero(source==k)
        assert exposure[ids].sum()==old['source_exposure'][name]['actual_window_presentations']
        a=np.asarray(targets[ids]);m=np.asarray(masks[ids]);width=2 if name=='Language Table' else 3
        valid=m[...,0]>0
        xyz=a[...,:width][valid].astype(np.float64)
        assert np.isfinite(a).all()
        item=dict(windows=len(ids),window_presentations=int(exposure[ids].sum()),
            presentation_fraction=float(exposure[ids].sum()/exposure.sum()),
            previously_verified_coverage=old['source_exposure'][name],
            mean_pose_loss_denominator_mass_fraction=float(shares[:,k].mean()),
            mean_gripper_loss_denominator_mass_fraction=float(gripshares[:,k].mean()),
            valid_future_targets_per_window=quantiles(m[...,0].sum(axis=1)),
            valid_target_position_norm_normalized=quantiles(np.linalg.norm(xyz,axis=1)),
            valid_target_max_absolute_position_normalized=float(np.abs(xyz).max()),
            target_components_outside_sampler_axis_limit8=int((np.abs(xyz)>8).sum()),
            target_fraction_outside_sampler_axis_limit8=float((np.abs(xyz)>8).any(axis=1).mean()),
            source_position_units_verified=False,
            position_report_units='Normalized recorded units; divisor0.1, not a newly verified meter/cm conversion')
        if width==3:
            q=a[...,3:7][valid].astype(np.float64);norm=np.linalg.norm(q,axis=1)
            assert np.max(np.abs(norm-1))<1e-3
            item['rotation_degrees']=quantiles(np.degrees(2*np.arccos(np.clip(np.abs(q[:,3]/norm),0,1))))
        result[name]=item
    summary=dict(scope='All existing bound train labels and exact seed42 update indices only; no model or raw-source rerun',
        sources=result,total_window_presentations=int(exposure.sum()),updates_without_bridge=no_bridge,
        missing_pose_channels_enter_noisy_denoiser_input=True,mask_given_to_denoiser=False,
        missing_dimension_loss_masked=True,
        pose_objective='Sum squared epsilon errors over valid pose entries divided by total valid pose entries in each batch',
        denominator_shares_are_not_measured_loss_or_gradient_shares=True,
        cross_source_scale_or_coordinate_registration_proven=False,
        causal_negative_transfer_established=False,
        optimizer_updates=0,new_inference_calls=0,raw_source_records_read=0,
        training_history_sha256=sha(training_path),target_dataset_identity=manifest['dataset_identity'])
    (OUT/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(source_summary={name:{k:v[k] for k in ['presentation_fraction',
        'mean_pose_loss_denominator_mass_fraction','mean_gripper_loss_denominator_mass_fraction',
        'target_fraction_outside_sampler_axis_limit8','valid_target_max_absolute_position_normalized']} for name,v in result.items()},
        updates_without_bridge=no_bridge),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
