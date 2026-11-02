"""One fresh Bridge trajectory: inner episode selection versus its final weights."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parent
STATE = CODE.parent/'bridge_state_conditioning_v1'
sys.path.insert(0, str(STATE))
os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TF_CPP_MIN_LOG_LEVEL='2')
from selection_contract import (sha, verify_split, observed_statistics, training_schedule,
                                selection_value, better, encode_or_reuse)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['data-root', 'data-reference', 'clip', 'token-cache', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--stage', choices=['preflight', 'fit'], default='preflight')
    ap.add_argument('--preflight-reference', type=Path)
    args = ap.parse_args()
    # Verify dependencies before importing executable frozen source.
    protocol = json.loads((CODE/'protocol.json').read_text(encoding='utf-8'))
    for name, expected in protocol['inherited_sha256'].items():
        assert sha(CODE.parent/name) == expected, name
    own_hashes = {name:sha(CODE/name) for name in ['run_selection.py', 'selection_contract.py', 'protocol.json']}
    import numpy as np
    import torch
    import fit_state_arm as B
    from state_input_contract import state_features
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest_path = CODE.parent/'bridge_growth128_v1/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = manifest['train']+manifest['development']
    indices = verify_split(rows, protocol['plan'])
    cfg = copy.deepcopy(manifest['config']); cfg['model']['name'] = str(args.clip)
    report = dict(stage=args.stage, execution_checks_passed=False, entry_sha256=own_hashes,
                  protocol_sha256=sha(CODE/'protocol.json'), manifest_sha256=sha(manifest_path),
                  data_checks=[], input_numeric_comparisons=[], selection_history=[],
                  downloads=False, gripper_trained=False, simulation_executed=False,
                  reserved_test_targets_read=False, outer_scored_during_selection=False,
                  outer_labels_loaded_for_identity_audit=True, optimizer_indices=indices['optimization_indices'])
    (args.output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'protocol.json').write_bytes((CODE/'protocol.json').read_bytes())
    def persist():
        report['elapsed_seconds'] = time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    try:
        assert torch.cuda.is_available(), 'GPU allocation required'
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        batches = training_schedule(indices['optimization_indices'])
        report['schedule_sha256'] = B.tensor_sha(batches)
        assert report['schedule_sha256'] == protocol['schedule_sha256']
        items = B.load_verified_items(rows, args.data_root, manifest, args.data_reference,
                                      report['data_checks'], report['input_numeric_comparisons'])
        # Existing raw-current-state contract is checked, then old128-row scaling is discarded.
        _, raw_checks = B.load_current_features(STATE/'protocol.json', rows, args.data_root, report['data_checks'])
        with np.load(STATE/'current_pose.npz', allow_pickle=False) as z:
            observed = z['current_pose'].copy()
        stats = observed_statistics(observed, indices['optimization_indices'])
        assert stats == protocol['state_statistics']
        features = torch.from_numpy(state_features(observed, stats)).cuda()
        report.update(input_contract_passed=True, raw_observed_state_checks=raw_checks,
                      effective_state_statistics=stats, state_features_sha256=B.tensor_sha(features),
                      raw_audit_old128_scaling_discarded=True)
        if args.stage == 'fit':
            assert args.preflight_reference
            pre = json.loads((args.preflight_reference/'report.json').read_text(encoding='utf-8'))
            assert pre['stage'] == 'preflight' and pre['execution_checks_passed']
            for key in ['entry_sha256', 'protocol_sha256', 'manifest_sha256', 'data_checks',
                        'effective_state_statistics', 'schedule_sha256']:
                assert pre[key] == report[key], key
            assert pre['observer_replay_exact'] and pre['selected_checkpoint_replay_exact']
        targets = torch.stack([v[3] for v in items]).cuda()
        current = torch.stack([v[2] for v in items]).cuda()
        B.seed(43)
        model = B.RobotAdapterModel(cfg).cuda().eval()
        texts = B.CLIPTokenizer.from_pretrained(str(args.clip), local_files_only=True)(
            [v[0] for v in items], padding=True, truncation=True, max_length=77, return_tensors='pt')
        sources = {n:sha(CODE.parent/'bridge_growth128_v1'/n) for n in manifest['source_sha256']}
        cache_identity = hashlib.sha256(json.dumps(dict(manifest=report['manifest_sha256'],
            checks=report['data_checks'], sources=sources), sort_keys=True).encode()).hexdigest()
        visual, language, reused = encode_or_reuse(model, items, texts, args.token_cache, cache_identity, B.digest)
        mask = texts['attention_mask'].cuda()
        model.vision_encoder.cpu(); model.text_encoder.cpu(); torch.cuda.empty_cache()
        clip_hashes = [B.digest(model.vision_encoder), B.digest(model.text_encoder)]
        model.history_projection = torch.nn.Linear(1024, 1024, bias=False).cuda()
        torch.nn.init.zeros_(model.history_projection.weight)
        for name, value in model.named_parameters():
            value.requires_grad_(name.startswith(B.PREFIXES))
        initial_hash = hashlib.sha256(''.join(n+B.tensor_sha(v) for n,v in sorted(model.state_dict().items())
            if not n.startswith(('vision_encoder.', 'text_encoder.'))).encode()).hexdigest()
        assert initial_hash == protocol['original_initial_shared_sha256'], 'Fresh initialization43 changed'
        B.add_zero_port(model)
        trainables = [p for p in model.parameters() if p.requires_grad]
        # Retain only frozen non-CLIP tensors for the final invariant check.
        frozen = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()
                  if not n.startswith(('vision_encoder.', 'text_encoder.')+B.PREFIXES)}
        optimizer = torch.optim.Adam(trainables, lr=1e-4)
        report.update(full_token_cache_reused=reused, initialization_seed=43,
                      initial_shared_sha256=initial_hash, config=cfg, sampling_steps=100,
                      sampling_seeds=manifest['sampling_seeds'], torch_version=str(torch.__version__),
                      gpu=torch.cuda.get_device_name(), state_projection_initial_zero=True)
        torch.cuda.reset_peak_memory_stats()
        end, switch, points = (8, 4, [4,8]) if args.stage == 'preflight' else (6144, 3072, protocol['plan']['selection_steps'])
        report.update(final_update=end, lr_switch_after_update=switch)
        def context(ix):
            now = model.adapter(visual[ix], language[ix], mask[ix])
            repeated = model.adapter(visual[ix], language[ix], mask[ix])
            return now+model.history_projection(repeated-now)+model.state_projection(features[ix])
        def check(label, update, split, chosen=None, save=True):
            ix = indices[split] if chosen is None else chosen
            if split == 'outer_development_indices':
                assert report.get('selection_locked'), 'Outer evaluation before selection lock'
            with B.observation_guard(model):
                ctx = torch.cat([context(torch.tensor(ix[i:i+2], device='cuda')) for i in range(0,len(ix),2)])
                truth = targets[ix].cpu().numpy()
                pred, gate = B.evaluate(model, ctx, current[ix], truth, manifest['sampling_seeds'], batch=2)
                result = B.metrics(pred, truth, [rows[i] for i in ix]); result['fit_gate'] = gate
            if save:
                np.savez_compressed(args.output/f'{label}_u{update:04d}.npz', predictions=pred, targets=truth,
                                    contexts=ctx.cpu().numpy(), row_indices=np.asarray(ix),
                                    sampling_seeds=np.asarray(manifest['sampling_seeds']))
            return result, pred
        def save_weights(name, update):
            payload = dict(stage='inner_selection_trainable_only', config=cfg, update=update,
                           initialization_seed=43, protocol_sha256=report['protocol_sha256'],
                           state_statistics=stats, trainable_state_dict=B.trainable_copy(model),
                           reconstruction='Original frozen growth128 model, zero history port and9-to1024 state port; use96-row state statistics; direct x0/DDPM100, no prior.')
            temporary = args.output/(name+'.partial')
            torch.save(payload, temporary); temporary.replace(args.output/name)
            assert sum(p.stat().st_size for p in args.output.glob('*.pt')) < 1200*1024**2
        def load_weights(name):
            payload = torch.load(args.output/name, map_location='cpu', weights_only=False)
            assert payload['protocol_sha256'] == report['protocol_sha256'] and payload['state_statistics'] == stats
            state = payload['trainable_state_dict']
            assert set(state) == set(B.trainable_copy(model))
            model.load_state_dict(state, strict=False)
            assert B.state_sha(B.trainable_copy(model)) == B.state_sha(state)
            return payload['update']
        def update_one(update):
            model.train()
            for group in optimizer.param_groups:
                group['lr'] = 1e-4 if update <= switch else 1e-5
            ix = batches[update-1].cuda()
            assert set(ix.cpu().tolist()) <= set(indices['optimization_indices'])
            B.seed(430000+update); ctx = context(ix)
            B.seed(420000+update)
            pred, truth, _ = model.diffusion_loss(targets[ix], ctx, current[ix])
            loss = torch.nn.functional.mse_loss(pred, truth)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grads = B.gradient_norms(model)
            assert all(p.grad is None for p in model.vision_encoder.parameters())
            assert all(p.grad is None for p in model.text_encoder.parameters())
            norm = torch.nn.utils.clip_grad_norm_(trainables, 1., error_if_nonfinite=True)
            optimizer.step()
            return dict(update=update, loss=float(loss.detach()), learning_rate=optimizer.param_groups[0]['lr'],
                        gradient_norms=grads, unclipped_global_norm=float(norm))
        if args.stage == 'preflight':
            # Same8 updates without observations versus with selection/checkpoint I/O.
            original = B.trainable_copy(model)
            control = [update_one(u) for u in range(1,end+1)]
            control_hash = B.state_sha(B.trainable_copy(model))
            control_adam_hash = B.state_sha(optimizer.state_dict())
            model.load_state_dict(original, strict=False); del original
            optimizer = torch.optim.Adam(trainables, lr=1e-4)
        incumbent = None
        with (args.output/'updates.jsonl').open('w', encoding='utf-8') as log:
            observed_updates = []
            for update in range(1,end+1):
                row = update_one(update); log.write(json.dumps(row)+'\n')
                if args.stage == 'preflight': observed_updates.append(row)
                if update == 1 or update % 256 == 0:
                    log.flush(); print('SELECTION_UPDATE', update, '/', end, 'loss', row['loss'], flush=True)
                if update in points:
                    metrics, _ = check('inner', update, 'inner_selection_indices')
                    criterion = selection_value(metrics)
                    record = dict(update=update, metrics=metrics, criterion=criterion)
                    report['selection_history'].append(record)
                    if better(criterion, incumbent):
                        incumbent = criterion; report['selected_update'] = update
                        save_weights('best.pt', update)
                    print('INNER_SELECTION', update, 'score', criterion['value'],
                          'cm', criterion['episode_position_cm'], 'deg', criterion['episode_rotation_deg'], flush=True)
                    persist()
        report['actual_draw_slots_per_row'] = torch.bincount(batches[:end].flatten(), minlength=168).tolist()
        assert all(report['actual_draw_slots_per_row'][i] == 0 for i in indices['inner_selection_indices']+indices['outer_development_indices'])
        if args.stage == 'fit':
            assert [report['actual_draw_slots_per_row'][i] for i in indices['optimization_indices']] == [512]*96
        else:
            report['observer_replay_exact'] = (control == observed_updates and
                control_hash == B.state_sha(B.trainable_copy(model)) and
                control_adam_hash == B.state_sha(optimizer.state_dict()))
            assert report['observer_replay_exact']
            report['actual_operational_updates_including_control'] = end*2
        # Selection is locked before any outer scoring, and its identity is persisted.
        report['selection_locked'] = True
        report['selected_checkpoint_sha256'] = sha(args.output/'best.pt')
        (args.output/'selection_lock.json').write_text(json.dumps(dict(selected_update=report['selected_update'],
            checkpoint_sha256=report['selected_checkpoint_sha256'], criterion=incumbent,
            rule=protocol['plan']['selection_rule']), indent=2)+'\n', encoding='utf-8')
        persist()
        if args.stage == 'fit':
            save_weights('final.pt', end)
            report['final_checkpoint_sha256'] = sha(args.output/'final.pt')
            report['final'] = {}
            for label, split in [('train','optimization_indices'), ('outer','outer_development_indices')]:
                report['final'][label], _ = check('final_'+label, end, split)
        loaded_update = load_weights('best.pt')
        assert loaded_update == report['selected_update']
        replay, prediction = check('selected_inner_replay', loaded_update, 'inner_selection_indices', save=False)
        with np.load(args.output/f'inner_u{loaded_update:04d}.npz', allow_pickle=False) as z:
            delta = float(np.abs(prediction-z['predictions']).max())
        report['selected_checkpoint_replay_max_difference'] = delta
        report['selected_checkpoint_replay_exact'] = delta == 0
        assert delta == 0, 'Selected weights did not replay original inner predictions'
        if args.stage == 'fit':
            report['selected'] = {}
            for label, split in [('train','optimization_indices'), ('outer','outer_development_indices')]:
                report['selected'][label], _ = check('selected_'+label, loaded_update, split)
            selected, final = report['selected']['outer'], report['final']['outer']
            report['both_outer_pose_errors_improved'] = all(selected[key] < final[key]
                for key in ['path_position_cm', 'path_rotation_deg'])
            print('SELECTION_OUTER_FINAL', json.dumps(final), flush=True)
            print('SELECTION_OUTER_SELECTED', json.dumps(selected), flush=True)
        else:
            # Preflight checkpoint round-trip is verified, then remove this known transient file.
            (args.output/'best.pt').unlink()
            report['preflight_weights_removed_after_verified_replay'] = True
        assert clip_hashes == [B.digest(model.vision_encoder), B.digest(model.text_encoder)]
        assert all(torch.equal(model.state_dict()[n].cpu(), value) for n,value in frozen.items())
        for name, expected in own_hashes.items(): assert sha(CODE/name) == expected
        report.update(execution_checks_passed=True, clip_unchanged=True, other_frozen_unchanged=True,
                      inner_and_outer_gradient_draws=0, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      scientific_verdict='One checkpoint-selection rule; execution success is not a generalization verdict.')
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        persist(); print('INNER SELECTION REPORT', args.output/'report.json', flush=True)


if __name__ == '__main__':
    main()
