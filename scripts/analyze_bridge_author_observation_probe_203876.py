"""Independent CPU replay/sensitivity audit; no author model or checkpoint load."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reports/bridge-author-obs-203876'
PACK=ROOT/'cluster_packages/bridge_author_observation_probe_v1'
OUT=ROOT/'reports/bridge_author_observation_analysis_203876'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def paired(a,b,errors):
    pos,rot=[],[]
    for draw in range(3):
        p,r,_=errors(a[draw:draw+1],b[draw]);pos.append(p[0]);rot.append(r[0])
    pos,rot=np.stack(pos),np.stack(rot)
    return dict(position_cm=float(pos.mean()),rotation_deg=float(rot.mean()),
        per_window_position_cm=pos.mean((0,2)).tolist(),per_window_rotation_deg=rot.mean((0,2)).tolist(),
        per_draw_position_cm=pos.mean((1,2)).tolist(),per_draw_rotation_deg=rot.mean((1,2)).tolist())

def main():
    OUT.mkdir(exist_ok=True);r=read(DATA/'report.json');p=read(PACK/'protocol.json')
    assert r['execution_passed'] and r['trained_updates']==0 and r['parameters_unchanged']
    assert not r['simulation_executed'] and not r['reserved_test_used'] and not r['mixed_observation_accuracy_evaluated']
    assert r['normalizer_matches_training_statistics'] and r['source_verified'] and r['sampler_calls']==39
    assert r['origin_job']==203699 and r['protocol_sha256']==sha(PACK/'protocol.json')
    assert r['checkpoint_sha256']==p['checkpoint_sha256'] and r['sampling_seeds']==p['sampling_seeds']
    assert r['replay_tolerance']==p['replay_tolerance'] and r['peak_reserved_gib']<20 and 'error' not in r
    z=np.load(DATA/'predictions.npz',allow_pickle=False)
    expected_names={'original','image_only','state_only','both','full40_replay','original_targets','donor_targets','window_indices','donor_indices'}
    assert set(z.files)==expected_names
    windows=np.array([x['window'] for x in p['donors']]);donors=np.array([x['donor_window'] for x in p['donors']])
    assert np.array_equal(z['window_indices'],windows) and np.array_equal(z['donor_indices'],donors)
    assert len(windows)==9 and (windows==np.arange(138,147)).all() and (donors<128).all()
    assert all(z[k].shape==(3,9,16,7) and np.isfinite(z[k]).all() for k in ('original','image_only','state_only','both'))
    assert z['full40_replay'].shape==(3,40,16,7)
    original=ROOT/'reports/bridge-author-reference-203699'
    with np.load(original/'ema_development.npz',allow_pickle=False) as old:
        delta=float(np.max(np.abs(z['full40_replay']-old['predictions'])))
        assert delta==r['original_replay_max_absolute_difference']==0
        assert np.array_equal(z['original_targets'],old['targets'][windows-128])
        full_target=old['targets'].copy()
    with np.load(original/'ema_train.npz',allow_pickle=False) as old:
        assert np.array_equal(z['donor_targets'],old['targets'][donors])
    assert np.array_equal(z['original'],z['full40_replay'][:,windows-128])
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_bridge_author_reference_203699 import errors,basic,check_metric
    po,ro,_=errors(z['full40_replay'],full_target);check_metric(basic(po,ro),r['original_full40_metrics'])
    changes={}
    for case in ('image_only','state_only','both'):
        changes[case]=paired(z[case],z['original'],errors)
        for k,value in changes[case].items():assert np.max(np.abs(np.asarray(value)-r['output_changes'][case][k]))<1e-7
    quality={}
    for key,case,target in [('original','original','original_targets'),('both_donor','both','donor_targets')]:
        pos,rot,_=errors(z[case],z[target]);quality[key]=basic(pos,rot)
        check_metric(quality[key],r['valid_context_quality'][key])
    rows=read(ROOT/'cluster_packages/bridge_author_reference_v1/training_protocol.json')['rows']
    per_window=[]
    for j,(i,donor) in enumerate(zip(windows,donors)):
        item=dict(window=int(i),donor=int(donor),episode=rows[int(i)]['shard']+'::'+str(rows[int(i)]['record_index']),
            input_index=rows[int(i)]['start_index'],donor_episode=rows[int(donor)]['shard']+'::'+str(rows[int(donor)]['record_index']),
            output_changes={case:dict(position_cm=changes[case]['per_window_position_cm'][j],
                rotation_deg=changes[case]['per_window_rotation_deg'][j]) for case in changes})
        for name,case,target in [('original_quality','original','original_targets'),('both_donor_quality','both','donor_targets')]:
            pos,rot,_=errors(z[case][:,j:j+1],z[target][j:j+1]);item[name]=basic(pos,rot)
        pos,rot,_=errors(z['original'][:,j:j+1],z['donor_targets'][j:j+1])
        item['original_prediction_to_donor_target']=basic(pos,rot)
        pos,rot,_=errors(z['donor_targets'][None,j:j+1],z['original_targets'][j:j+1])
        item['true_target_difference']=basic(pos,rot)
        per_window.append(item)
    input_counts=sum(x['output_changes']['image_only']['position_cm']>x['output_changes']['state_only']['position_cm'] for x in per_window)
    project_checks={}
    for name,directory,analysis_path,prefix,description in [
        ('201663',ROOT/'reports/bridge-growth128-fit-201663',ROOT/'reports/bridge_growth128_analysis_201663_20261013/analysis.json','',
         'Adapter output with repeated-current history cancellation; no numeric pose input'),
        ('202278_real',ROOT/'reports/bridge-state-fit-202278/real',ROOT/'reports/bridge_state_analysis_202278_20261013/analysis.json','real/',
         'Fused Adapter output plus numeric current-state projection; not Adapter output alone')]:
        verified=read(analysis_path);assert verified['verified']
        arrays=[];hashes={}
        for split,expected_indices in [('train',np.arange(128)),('development',np.arange(128,168))]:
            file=directory/('reduced_u8192_'+split+'.npz')
            assert sha(file)==verified['artifact_sha256'][prefix+file.name]
            hashes[file.name]=sha(file)
            with np.load(file,allow_pickle=False) as old:
                assert np.array_equal(old['row_indices'],expected_indices)
                assert old['contexts'].shape==(len(expected_indices),1024) and np.isfinite(old['contexts']).all()
                arrays.append({k:old[k].copy() for k in ('contexts','predictions','targets')})
        c=np.concatenate([x['contexts'] for x in arrays]).astype(float)
        target=np.concatenate([x['targets'] for x in arrays])[...,:7]
        i,j=146,114
        assert np.array_equal(target[i],z['original_targets'][-1]) and np.array_equal(target[j],z['donor_targets'][-1])
        po,ro,_=errors(arrays[1]['predictions'][:,18:19,...,:7],target[i:i+1])
        project_checks[name]=dict(context_description=description,window=i,training_neighbor=j,
            context_vectors_exact_equal=bool(np.array_equal(c[i],c[j])),context_l2_difference=float(np.linalg.norm(c[i]-c[j])),
            context_cosine=float(c[i]@c[j]/np.linalg.norm(c[i])/np.linalg.norm(c[j])),
            original_window_quality=basic(po,ro),files_sha256=hashes,
            new_model_inference=False,paired_project_noise_intervention_executed=False,
            limit='Distinct vectors exclude exact collapse for this pair; do not prove usable geometry or uniquely identify a faulty decoder.')
        # One fixed train-only readout diagnostic; no metric/k search or development-label selection.
        normalized=c/np.linalg.norm(c,axis=-1,keepdims=True);nn=[]
        for index,row in enumerate(rows[128:],start=128):
            candidates=np.array([k for k,x in enumerate(rows[:128]) if x['task']==row['task']])
            nn.append(int(candidates[np.argmax(normalized[candidates]@normalized[index])]))
        po,ro,_=errors(target[nn][None],target[128:]);nearest=basic(po,ro)
        static=np.zeros_like(target[128:]);static[...,6]=1
        sp,sr,_=errors(static[None],target[128:])
        project_checks[name]['fixed_cosine_nearest_training_readout']=dict(
            development=nearest,static=basic(sp,sr),selected_training_indices=nn,
            trained=False,development_targets_used_in_neighbor_selection=False,
            no_k_or_distance_sweep=True,
            interpretation='Descriptive saved-feature readout; failure does not establish absent geometry or a correct action head.')
    result=dict(audit_passed=True,job=203876,trained_updates=0,local_model_loaded=False,
        original_full40_replay_exact=True,original_replay_max_absolute_difference=delta,
        all_paired_changes_and_valid_metrics_recomputed=True,image_change_exceeds_state_change_windows=input_counts,
        total_windows=9,independent_development_episodes=2,distinct_training_donors=len(set(map(int,donors))),
        reported_weights_unchanged=r['parameters_unchanged'],weight_contents_independently_verified=False,
        reported_nonselected_rows_difference=r['unchanged_batch_rows_max_absolute_difference'],
        nonselected_intervention_outputs_not_saved_for_independent_check=True,
        reported_normalizer_matches_training_statistics=r['normalizer_matches_training_statistics'],
        reported_peak_reserved_gib=r['peak_reserved_gib'],output_changes=changes,valid_context_quality=quality,
        per_window=per_window,project_saved_context_checks=project_checks,
        files_sha256={n:sha(DATA/n) for n in ('report.json','predictions.npz')},
        interpretation='Fixed-pair sensitivity under possible inconsistent crossed observations; no mixed-observation accuracy or improvement verdict.')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('INDEPENDENT PROBE AUDIT PASSED; original replay exact; image change larger:',input_counts,'/9')
    for item in per_window:
        print(item['window'],'image/state/both cm',*[round(item['output_changes'][x]['position_cm'],3) for x in changes],
              'original/donor quality cm',round(item['original_quality']['path_position_cm'],3),
              round(item['both_donor_quality']['path_position_cm'],3))

if __name__=='__main__':main()
