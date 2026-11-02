"""Fixed seed42 decoded-action diagnosis on existing source validation roles.

Eight origin/content groups per source, one metadata-selected window per group,
one predeclared DDPM draw per window. No fitting, simulator, checkpoint search,
target changes, or physical-unit assumption added by this diagnostic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/oxe_source_validation_decode_20261027/attempt1'
sys.path.insert(0, str(ROOT / 'code'))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', TF_CPP_MIN_LOG_LEVEL='3',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024**2), b''):
            h.update(part)
    return h.hexdigest()


def window_metrics(pred, target, mask, features):
    import numpy as np
    valid = mask[:, 0] > .5
    assert valid[0] and np.array_equal(valid, mask[:, 1] > .5)
    width = 3 if bool(mask[0, 2]) else 2
    assert np.array_equal(mask[:, :width] > .5, np.repeat(valid[:, None], width, axis=1))
    pos = np.linalg.norm(pred[valid, :width]-target[valid, :width], axis=1)
    static = np.linalg.norm(target[valid, :width], axis=1)
    rotation = bool(mask[0, 3])
    if rotation:
        assert np.array_equal(mask[:, 3:7] > .5, np.repeat(valid[:, None], 4, axis=1))
        q = target[valid, 3:7].astype(float)
        p = pred[valid, 3:7].astype(float)
        assert np.max(np.abs(np.linalg.norm(q, axis=1)-1)) < 1e-3
        assert np.max(np.abs(np.linalg.norm(p, axis=1)-1)) < 1e-3
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        angles = np.degrees(2*np.arccos(np.clip(np.abs(np.sum(p*q, axis=1)), 0, 1)))
        idle_angles = np.degrees(2*np.arccos(np.clip(np.abs(q[:, 3]), 0, 1)))
    else:
        assert not mask[:, 2:].any()
    result = {}
    for name, indices in (('first', slice(0, 1)), ('valid_path', slice(None))):
        r = dict(position_dimensions=width, valid_targets=1 if name == 'first' else int(valid.sum()),
                 model_position_normalized=float(pos[indices].mean()),
                 no_motion_position_normalized=float(static[indices].mean()),
                 model_position_recorded_units=float(pos[indices].mean()*.1),
                 no_motion_position_recorded_units=float(static[indices].mean()*.1))
        if rotation:
            r.update(model_rotation_deg=float(angles[indices].mean()), no_motion_rotation_deg=float(idle_angles[indices].mean()))
        result[name] = r
    g = mask[:, 7] > .5
    if g.any():
        actual, predicted = target[g, 7], pred[g, 7]
        assert np.isin(actual, [-1., 1.]).all() and np.isin(predicted, [-1., 1.]).all()
        result['gripper'] = dict(valid_targets=int(g.sum()), correct=int((actual == predicted).sum()),
                                 accuracy=float((actual == predicted).mean()),
                                 actual_open_count=int((actual > 0).sum()), predicted_open_count=int((predicted > 0).sum()),
                                 hold_baseline_available=bool(features[1]))
        if features[1]:
            result['gripper']['hold_known_command_correct'] = int((actual == features[0]).sum())
            result['gripper']['hold_known_command_accuracy'] = float((actual == features[0]).mean())
    return result


def aggregate(rows):
    import numpy as np
    out = {}
    for source in ('Bridge', 'Language Table', 'BC-Z', 'Fractal'):
        items = [r['metrics'] for r in rows if r['source'] == source]
        s = dict(independent_origin_content_groups=len(items), windows=len(items))
        for horizon in ('first', 'valid_path'):
            metrics = {key:float(np.mean([m[horizon][key] for m in items])) for key in (
                'model_position_normalized', 'no_motion_position_normalized',
                'model_position_recorded_units', 'no_motion_position_recorded_units')}
            metrics['model_to_no_motion_position_ratio'] = (metrics['model_position_normalized']/metrics['no_motion_position_normalized']
                                                          if metrics['no_motion_position_normalized'] > 0 else None)
            metrics['windows_position_better_than_no_motion'] = sum(m[horizon]['model_position_normalized'] < m[horizon]['no_motion_position_normalized'] for m in items)
            if 'model_rotation_deg' in items[0][horizon]:
                for key in ('model_rotation_deg', 'no_motion_rotation_deg'):
                    metrics[key] = float(np.mean([m[horizon][key] for m in items]))
                metrics['windows_rotation_better_than_no_motion'] = sum(m[horizon]['model_rotation_deg'] < m[horizon]['no_motion_rotation_deg'] for m in items)
            s[horizon] = metrics
        grippers = [m['gripper'] for m in items if 'gripper' in m]
        if grippers:
            known = [g for g in grippers if g['hold_baseline_available']]
            s['gripper'] = dict(window_equal_accuracy=float(np.mean([g['accuracy'] for g in grippers])),
                                 valid_targets=sum(g['valid_targets'] for g in grippers),
                                 actual_open_count=sum(g['actual_open_count'] for g in grippers),
                                 predicted_open_count=sum(g['predicted_open_count'] for g in grippers),
                                 known_command_windows=len(known),
                                 model_accuracy_on_known_windows=float(np.mean([g['accuracy'] for g in known])) if known else None,
                                 hold_accuracy_on_known_windows=float(np.mean([g['hold_known_command_accuracy'] for g in known])) if known else None)
        out[source] = s
    return out


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', choices=('validation', 'train'), default='validation')
    args = parser.parse_args()
    partition = args.partition
    if partition == 'train':
        OUT = ROOT / 'reports/oxe_source_train_decode_20261027/attempt1'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    memory, util = map(int, gpu.split(','))
    if memory > 1024 or util > 20:
        raise RuntimeError('Existing local GPU workload; no model started')
    OUT.mkdir(parents=True, exist_ok=False)
    report = dict(execution_complete=False, errors=[], model_windows_decoded=0, training_updates=0,
                  simulator_steps=0, checkpoint_search=False, formal_libero_episodes=0,
                  partition=partition,
                  scope=f'Small existing source-{partition} diagnosis under current reached-pose label conventions, one trained seed; no fitting, final or physical task-success claim')
    rows, predictions, targets, masks, features_all = [], [], [], [], []
    started = time.monotonic()
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from oxe_nominal_dataset import NominalOXEDataset
        from libero_project_policy import load_project_policy
        from libero_waypoint_execution import WaypointExecutionContract
        from train import tokenise, trainable_state_dict
        from step_training_runner import frozen_digest
        from completed_policy_export import model_sha
        from dataset import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD

        mapping = read(ROOT/'reports/oxe_bounded_pool_plan_20261025/candidate_shards.json')
        source_map = dict(format='oxe-nominal-source-locations-v1', files=[dict(relative_file=r['stage_relative_path'],path=r['source_path'],bytes=r['bytes']) for r in mapping['files']])
        write(OUT/'source_locations.json', source_map)
        target_dir = ('reports/oxe_nominal_targets_20261025' if partition == 'train'
                      else 'reports/oxe_nominal_validation_20261025')
        data = NominalOXEDataset(ROOT, ROOT/target_dir, OUT/'source_locations.json', partition=partition)
        train_groups = {(e['source'], e['group_id']) for e in read(ROOT/'reports/oxe_nominal_targets_20261025/episodes.json')}
        val_groups = {(e['source'], e['group_id']) for e in read(ROOT/'reports/oxe_nominal_validation_20261025/episodes.json')}
        assert not train_groups & val_groups
        presentations = None
        history_file = ROOT/'reports/oxe_noaug_seed42_227223/cluster/reports/step_report.json'
        if partition == 'train':
            history = read(history_file)
            assert history['complete'] and history['completed_updates'] == 65536
            presentations = np.zeros(len(data), dtype=np.int64)
            for update in history['updates']:
                for index in update['indices']:
                    presentations[index] += 1
            assert int(presentations.sum()) == 524276
            del history
        selected = []
        for source in ('Bridge', 'Language Table', 'BC-Z', 'Fractal'):
            candidates = [e for e in data.store.episodes if e['source'] == source]
            prefix = 'sourcetrain20261027|' if partition == 'train' else 'sourceval20261027|'
            candidates.sort(key=lambda e:hashlib.sha256((prefix+source+'|'+e['group_id']+'|'+e['episode_key']).encode()).hexdigest())
            used = set()
            for e in candidates:
                if e['group_id'] in used:
                    continue
                index = (e['window_first'] + e['window_stop'] - 1)//2
                if presentations is not None:
                    eligible = [wi for wi in range(e['window_first'], e['window_stop']) if presentations[wi] > 0]
                    if not eligible:
                        continue
                    index = eligible[(len(eligible)-1)//2]
                used.add(e['group_id'])
                meta = data.store.get(index)
                assert meta['episode_key'] == e['episode_key'] and e['partition'] == partition
                selected.append(dict(source=source, group_id=e['group_id'], episode_key=e['episode_key'],
                                     window_index=index,input_row=meta['input_row'],valid_targets=meta['layout']['valid_targets'],
                                     payload_sha256=e['payload_sha256'], sampling_seed=1101+len(selected),
                                     actual_training_presentations=int(presentations[index]) if presentations is not None else None))
                if len(used) == 8:
                    break
            assert len(used) == 8
        protocol = dict(selection='Metadata SHA rank,8 distinct origin/content groups per source; midpoint existing window of each representative episode; no motion/error selection',
                        partition=partition,
                        training_selection_rule='For train only: midpoint among actually sampled windows in the metadata-ranked episode, never filter by loss or exposure magnitude',
                        training_history_sha256=sha(history_file) if presentations is not None else None,
                        selected=selected, checkpoint='reports/oxe_noaug_seed42_227223/cluster/final_policy.pt',
                        seed42_only=True, draws_per_window=1, diffusion_steps=100,
                        target_binding=data.nominal_target_binding, validation_used_for_development=True,
                        comparison_validation_report='reports/oxe_source_validation_decode_20261027/attempt1/report.json' if partition == 'train' else None,
                        metrics='First target and window-equal mean over valid path targets, per source; mask missing dimensions and padded tails',
                        position_units='Normalized coordinates and original source recorded units (multiply existing labels by0.1), not calibrated meters/cm',
                        no_motion='Zero source displacement, identity left-relative quaternion; hold last command only when known',
                        gripper_unknown='No hold-baseline score for unknown history; LT has no gripper score',
                        gpu_diagnostic_wall_cap_seconds=600, new_training_updates=0, simulator_steps=0,
                        primary_final_tasks_accessed=0, script_sha256=sha(__file__))
        write(OUT/'selection_protocol.json', protocol)
        (OUT/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.manual_seed(1101)
        k = read(ROOT/'configs/libero_nominal_execution_candidate_20261025.json')['constructor_kwargs']
        k['target_times_seconds']=tuple(k['target_times_seconds']); k['output_scales']=tuple(k['output_scales'])
        policy = load_project_policy(ROOT/protocol['checkpoint'],
                    ROOT.parent/'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41',
                    WaypointExecutionContract(**k), device='cuda',sampling_seed=1101,
                    local_dino_path=ROOT/'baselines/dinov2_large_frozen_assets_20261023',
                    frozen_profile_path=ROOT/'reports/oxe_noaug_seed42_227223/cluster/preflight.json')
        assert policy.checkpoint_metadata['frozen_encoders_and_tokenizer_match_training']
        assert policy.config['action']['nominal_target_binding']['partition_bindings'][partition] == data.nominal_target_binding
        report['checkpoint_metadata']=policy.checkpoint_metadata
        report['initial_frozen_sha256']=frozen_digest(policy.model, trainable_state_dict(policy.model))
        report['initial_learned_sha256']=model_sha(trainable_state_dict(policy.model))
        assert report['initial_learned_sha256']=='2295d8add3e735a7340eb8e244fbc1d4b6abc09d95692120fde1573303abd66d'
        torch.cuda.reset_peak_memory_stats()
        inference_started=time.monotonic()
        thumbs=[]
        for j, chosen in enumerate(selected):
            if time.monotonic()-inference_started > 600:
                raise RuntimeError('Fixed diagnostic GPU budget reached; retain completed windows, do not extend')
            instruction,pixels,features,target,mask=data[chosen['window_index']]
            tokens=tokenise(policy.tokenizer,[instruction],torch.device('cuda'))
            with torch.random.fork_rng(devices=[0]),torch.inference_mode():
                torch.random.default_generator.manual_seed(chosen['sampling_seed'])
                torch.cuda.manual_seed(chosen['sampling_seed'])
                context=policy.model.get_context_vector(pixels[None].cuda(),tokens['input_ids'],attention_mask=tokens.get('attention_mask'))
                pred=policy.model.sample(context,current_gripper=features[None].cuda())[0].cpu().numpy()
            assert pred.shape==(16,8) and np.isfinite(pred).all()
            predictions.append(pred);targets.append(target.numpy());masks.append(mask.numpy());features_all.append(features.numpy())
            row=dict(chosen,metrics=window_metrics(pred,target.numpy(),mask.numpy(),features.numpy()),instruction=instruction)
            rows.append(row)
            rgb=((pixels*CLIP_IMAGE_STD+CLIP_IMAGE_MEAN).permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
            thumbs.append(Image.fromarray(rgb).resize((112,112)))
            write(OUT/'window_metrics.json',rows)
            report['model_windows_decoded']=len(rows)
            print('SOURCE_'+partition.upper()+'_DECODE',j+1,'/32',chosen['source'],flush=True)
        torch.cuda.synchronize()
        report['inference_seconds']=time.monotonic()-inference_started
        report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
        report['final_frozen_sha256']=frozen_digest(policy.model,trainable_state_dict(policy.model))
        report['final_learned_sha256']=model_sha(trainable_state_dict(policy.model))
        assert report['final_frozen_sha256']==report['initial_frozen_sha256']
        assert report['final_learned_sha256']==report['initial_learned_sha256']
        assert not policy.model.training and not any(e.requires_grad for encoder in (policy.model.vision_encoder,policy.model.text_encoder) for e in encoder.parameters())
        sheet=Image.new('RGB',(8*116,4*144),'white');draw=ImageDraw.Draw(sheet)
        for i,thumb in enumerate(thumbs):
            x=i%8*116;y=i//8*144
            sheet.paste(thumb,(x,y+28));draw.text((x+2,y+2),selected[i]['source'],fill='black');draw.text((x+2,y+14),'window '+str(selected[i]['window_index']),fill='black')
        sheet.save(OUT/'input_contact_sheet.png')
        report.update(execution_complete=True, source_metrics=aggregate(rows),
                      aggregation='Each window and its distinct source group has equal weight; sources never pooled',
                      source_validation_only=partition=='validation', source_train_only=partition=='train',
                      current_reached_label_contract_unchanged=True,
                      no_final_generalization_or_task_success_verdict=True)
    except Exception as exc:
        report['errors'].append(type(exc).__name__+': '+str(exc))
        raise
    finally:
        if predictions:
            import numpy as np
            np.savez_compressed(OUT/'predictions_and_targets.npz',predictions=np.stack(predictions),
                                targets=np.stack(targets),masks=np.stack(masks),current_features=np.stack(features_all))
        report['elapsed_seconds']=time.monotonic()-started
        write(OUT/'report.json',report)
        print('SOURCE_VALIDATION_DECODE_REPORT',OUT/'report.json',flush=True)
        print(json.dumps(report.get('source_metrics',report),indent=2),flush=True)


if __name__ == '__main__':
    main()
