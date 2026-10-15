"""Frozen readout counterfactuals on saved native simulator observations.

No training, new diffusion sampling, or robot control. Only executed prefixes
were saved; the production legacy head is pointwise across time, so the other
12 pose cells cannot affect the four inspected outputs. Verify that explicitly.
"""
import json
import time
from pathlib import Path

from fit_pybullet_fixed_grasp import (
    ROOT, CONTRACT, RobotAdapterModel, CLIPTokenizer, prepare_image, sim,
    torch, np, file_sha, trainable_hash, digest,
)

BASE = ROOT / 'reports/pybullet_hold_pair_20261011'
OUT = BASE / 'frozen_gripper_probe'


@torch.inference_mode()
def main():
    started = time.monotonic()
    original = json.loads((BASE/'hold_rollout/rollout_report.json').read_text())
    checkpoint = BASE/'hold/final.pt'
    assert file_sha(checkpoint) == original['checkpoint_sha256']
    for name, value in original['source_sha256'].items():
        assert file_sha(ROOT/'code'/name) == value, name
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['contract'] == CONTRACT
    model = RobotAdapterModel(payload['config']).cuda().eval()
    missing = model.load_state_dict(payload['trainable_state_dict'], strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith(('vision_encoder.', 'text_encoder.')) for k in missing.missing_keys)
    assert model.trajectory_conditioned_gripper and model.gripper_target_mode == 'state'
    assert model.gripper_head_type == 'legacy'
    before = [trainable_hash(model), digest(model.vision_encoder), digest(model.text_encoder)]
    tokenizer = CLIPTokenizer.from_pretrained(payload['config']['model']['name'], local_files_only=True)
    text = {k:v.cuda() for k,v in tokenizer(sim.INSTRUCTION, return_tensors='pt').items()}
    initial_current = float(np.load(ROOT/'reports/pybullet_hold_collection_20261011/demonstration.npz')['currents'][0])
    training = np.load(BASE/'hold/training_examples.npz')
    hold_index = int(np.flatnonzero(training['starts']==90)[0])
    hold_current = float(training['currents'][hold_index,0])
    hold_context = model.get_context_vector(prepare_image(training['images'][hold_index]).unsqueeze(0).cuda(),
                                            text['input_ids'],text['attention_mask'])
    rows = []
    replay_mismatches = 0
    padding_difference = 0.
    for seed in [1101, 1102, 1103]:
        trace = np.load(BASE/f'hold_rollout/policy_seed_{seed}.npz')
        for r in range(28):
            start = r*4
            current = initial_current if r == 0 else float(trace['measured_widths'][start-1])
            ctx = model.get_context_vector(prepare_image(trace['images'][r]).unsqueeze(0).cuda(),
                                           text['input_ids'], text['attention_mask'])
            width = torch.tensor([[current]], dtype=torch.float32, device='cuda')
            native = torch.zeros((1,model.chunk_size,7), device='cuda')
            native[...,6] = 1
            native[0,:4] = torch.as_tensor(trace['actions'][start:start+4,:7], device='cuda')
            variants = {'recorded_pose': native}
            hold = torch.zeros_like(native); hold[...,6] = 1
            variants['stationary_pose'] = hold
            zero_xyz = native.clone(); zero_xyz[...,:3] = 0
            variants['zero_translation'] = zero_xyz
            identity_q = native.clone(); identity_q[...,3:7] = hold[...,3:7]
            variants['identity_rotation'] = identity_q
            sign_q = native.clone(); sign_q[...,3:7] *= -1
            variants['equivalent_quaternion_sign_flip'] = sign_q
            canonical = native.clone()
            canonical[...,3:7] *= torch.where(canonical[...,6:7]<0, -1., 1.)
            variants['canonical_quaternion'] = canonical
            logits = {k:model.predict_gripper_logits(ctx,v,width)[0,:4].cpu().numpy()
                      for k,v in variants.items()}
            train_width = torch.tensor([[hold_current]],dtype=torch.float32,device='cuda')
            logits['training_hold_image_same_pose_width'] = model.predict_gripper_logits(hold_context,native,width)[0,:4].cpu().numpy()
            logits['training_hold_width_same_pose_image'] = model.predict_gripper_logits(ctx,native,train_width)[0,:4].cpu().numpy()
            logits['training_hold_image_and_width_same_pose'] = model.predict_gripper_logits(hold_context,native,train_width)[0,:4].cpu().numpy()
            predicted = np.where(logits['recorded_pose'] >= 0, 1., -1.)
            replay_mismatches += int(np.count_nonzero(predicted != trace['actions'][start:start+4,7]))
            changed_tail = native.clone(); changed_tail[:,4:] = 2.
            tail_logits = model.predict_gripper_logits(ctx,changed_tail,width)[0,:4].cpu().numpy()
            padding_difference = max(padding_difference, float(np.max(np.abs(tail_logits-logits['recorded_pose']))))
            for j in range(4):
                idx = start+j
                rows.append(dict(seed=seed, replan=r, chunk_offset=j, action=idx+1,
                    observed_current=current,
                    pre_action_contacts=(trace['finger_contacts'][idx-1].tolist() if idx else [False,False]),
                    pre_action_height_m=(float(trace['object_heights'][idx-1]) if idx else original['results'][0]['initial_height_m']),
                    recorded_pose=trace['actions'][idx,:7].tolist(),
                    logits={k:float(v[j]) for k,v in logits.items()},
                    open_probability={k:float(1/(1+np.exp(-float(v[j])))) for k,v in logits.items()}))
        print('READOUT_REPLAY seed',seed,'complete',flush=True)
    assert replay_mismatches == 0, f'Record replay differs: {replay_mismatches}'
    assert padding_difference == 0., 'Unobserved pose cells affect inspected outputs'
    bad = next(x for x in rows if x['seed']==1102 and x['action']==53)
    assert bad['pre_action_contacts'] == [True,True]
    assert bad['logits']['recorded_pose'] >= 0
    opening_while_contact = [x for x in rows if all(x['pre_action_contacts']) and x['logits']['recorded_pose'] >= 0]
    flips = {key:sum(x['logits'][key]<0 for x in opening_while_contact)
             for key in bad['logits'] if key != 'recorded_pose'}
    # Inspect all recorded training images at this SAME failed pose, width,
    # instruction and horizon offset. Labels describe their own training phase,
    # not the hypothetical true future of the failure observation.
    same_pose = torch.zeros((1,model.chunk_size,7),device='cuda');same_pose[...,6]=1
    same_pose[0,0]=torch.tensor(bad['recorded_pose'],device='cuda')
    same_width = torch.tensor([[bad['observed_current']]],dtype=torch.float32,device='cuda')
    image_rows=[]
    for i,start in enumerate(training['starts']):
        ctx = model.get_context_vector(prepare_image(training['images'][i]).unsqueeze(0).cuda(),text['input_ids'],text['attention_mask'])
        value=float(model.predict_gripper_logits(ctx,same_pose,same_width)[0,0].cpu())
        image_rows.append(dict(training_observation=int(start),own_first_target_command=float(training['targets'][i,0,7]),
                               failed_pose_width_first_offset_logit=value))
    after = [trainable_hash(model), digest(model.vision_encoder), digest(model.text_encoder)]
    assert before == after
    for name, value in original['source_sha256'].items():
        assert file_sha(ROOT/'code'/name) == value
    assert file_sha(checkpoint) == original['checkpoint_sha256']
    report = dict(stage='frozen_saved_observation_gripper_readout_counterfactual',
        trained=False, robot_controlled=False, new_diffusion_sampling=False,
        checkpoint_sha256=original['checkpoint_sha256'],
        source_sha256=original['source_sha256'], probe_source_sha256=file_sha(Path(__file__)),
        all_parameters_unchanged=before==after, parameter_hashes=before,
        replay=dict(executed_commands=336, mismatches=replay_mismatches,
                    unrecorded_tail_max_prefix_logit_difference=padding_difference,
                    initial_current_source='bit-identical initial physical collection state'),
        first_failed_open=bad,
        recorded_open_actions_with_preceding_dual_contact=len(opening_while_contact),
        those_actions_changed_to_close_by_input_replacement=flips,
        training_hold_reference=dict(observation=90,current=hold_current,
                                     original_first_four_target_commands=training['targets'][hold_index,:4,7].tolist()),
        training_image_only_comparison_at_failed_first_action=image_rows,
        limits=['Only recorded four-action prefixes inspected; no full saved chunks available.',
                'Immediate pre-action contact is an outcome annotation, never a network input.',
                'Counterfactual outputs did not control robot and establish no recovery success.',
                'One trained checkpoint, three sampling episodes, one fixed training scene.'],
        rows=rows, elapsed_seconds=time.monotonic()-started)
    OUT.mkdir(exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    from PIL import Image
    Image.fromarray(np.load(BASE/'hold_rollout/policy_seed_1102.npz')['images'][13]).save(OUT/'before_wrong_open.png')
    print('FIRST_FAILED_OPEN',json.dumps(bad),flush=True)
    print('CONTACT_OPEN_FLIPS',flips,flush=True)
    print('FROZEN GRIPPER READOUT PROBE COMPLETE',OUT/'report.json',flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)
    main()
