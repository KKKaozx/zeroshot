"""Independent checks of paired inputs, fitted windows and executed physics."""
import hashlib,json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
plan=json.loads((root/'preparation.json').read_text())
training={name:json.loads((root/name/'training_report.json').read_text()) for name in ['baseline','hold']}
assert training['baseline']['initial_trainable_sha256']==training['hold']['initial_trainable_sha256']
assert training['baseline']['source_sha256']==training['hold']['source_sha256']
assert all(t['batch_size']==9 and t['updates']==1000 and t['clip_unchanged'] for t in training.values())
base=np.load(root/'baseline/training_examples.npz');hold=np.load(root/'hold/training_examples.npz')
assert all(np.array_equal(base[k][:8],hold[k][:8]) for k in ['images','targets','currents','starts'])
assert np.array_equal(base['targets'][...,7],hold['targets'][...,7])
summary=dict(initial_parameters_identical=True,training_source_identical=True,original_eight_inputs_identical=True,
             all_gripper_target_labels_identical=True,same_batch_and_update_budget=True,groups={})
for name,train in training.items():
    roll=json.loads((root/f'{name}_rollout/rollout_report.json').read_text())
    checkpoint=hashlib.sha256((root/name/'final.pt').read_bytes()).hexdigest()
    assert checkpoint==train['checkpoint_sha256']==roll['checkpoint_sha256']
    cases=[]
    for result in roll['results']:
        data=np.load(root/f'{name}_rollout'/f"policy_seed_{result['sampling_seed']}.npz")
        h=data['object_heights'][-20:];contact=data['finger_contacts'][-20:].all(axis=1)
        error=np.abs(h-plan['hold_reference_object_height_m'])*100;spread=np.ptp(h)*100
        gate=bool(error.max()<=2 and spread<=1 and contact.all())
        assert np.isclose(error.max(),result['hold']['last_two_seconds_max_height_error_cm'])
        assert np.isclose(spread,result['hold']['last_two_seconds_height_range_cm'])
        assert int(contact.sum())==result['hold']['last_two_seconds_two_finger_contacts']
        assert gate==result['hold']['passed']
        command=data['actions'][:,7]>0
        switches=(np.flatnonzero(np.diff(command)!=0)+2).tolist()
        cases.append(dict(seed=result['sampling_seed'],grasp_passed=result['success'],hold_passed=gate,
            max_hold_height_error_cm=float(error.max()),hold_height_range_cm=float(spread),
            two_finger_contact_actions=int(contact.sum()),command_switch_actions=switches,
            close_to_open_switches=int(((~command[:-1])&command[1:]).sum())))
    fit=train['history'][-1]
    summary['groups'][name]=dict(pose_fit_passed=fit['pose_fit_passed'],
        observed_gripper_fit_passed=fit['gripper']['observed_state_and_events_fit_passed'],
        offline_position_mm=fit['position_cm']*10,offline_rotation_deg=fit['rotation_deg'],
        grasp_passes=sum(c['grasp_passed'] for c in cases),hold_passes=sum(c['hold_passed'] for c in cases),
        mean_max_hold_height_error_cm=float(np.mean([c['max_hold_height_error_cm'] for c in cases])),cases=cases)
summary['both_fit_gates_passed']=all(g['pose_fit_passed'] and g['observed_gripper_fit_passed'] for g in summary['groups'].values())
summary['scope']='One paired training initialization, same seen scene, three sampling seeds; no generalization claim.'
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
