"""One fixed paired continuation after209420; no preflight rerun or sweeps.

Uses immutable preflight sources and the already encoded1558-token cache.
Both final4096 weights are frozen BEFORE new-development policy scores.
The final25 payloads are never opened. No gripper training or grasp verdict.
"""
import os
os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1')
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
from bridge_prefix_pair_metrics_v1 import summarize, admission


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_sums(root, sha):
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert sha(root/name) == expected, (root, name)


def load_training_inputs(pre, train_root, old, delta, tokenizer, sha, tsha):
    """Reuse tokens, reload native numeric values; NO training image decoding.

    Numeric references originated onWindows. Recompute only labels/state with
    the already verifiedLinux route to avoid even tiny cross-platform shifts.
    This is input loading, not another audit/training experiment.
    """
    from dataset import relative_pose_action,euler_xyz_to_quaternion,bridge_reverse_scan_gripper
    from state_input_contract import state_features
    from bridge_prefix_input_contract_v1 import records_at,episode_arrays
    from bridge_partial_future_contract_v1 import build_numeric_sample
    rows = read(pre/'candidate_rows.json')['train']
    old_rows = read(pre/'old_rows.json')
    width = old['language'].shape[1]
    # Prove canonical spelling produces the SAME original1876 token inputs.
    original_text = tokenizer([r['task'] for r in old_rows['train']+old_rows['development']],
        padding='max_length', max_length=width, return_tensors='pt', truncation=False)
    for k in ('input_ids', 'attention_mask'):
        assert tsha(original_text[k]) == old['key']['tokenizer_inputs'][k], k
    with np.load(pre/'input_reference/old_training_reference.npz', allow_pickle=False) as z:
        old_targets, old_state = z['targets'].copy(), z['state_features'].copy()
    with np.load(pre/'input_reference/train_delta_reference.npz', allow_pickle=False) as z:
        expected_targets = np.concatenate([old_targets, z['targets']])
        expected_state = np.concatenate([old_state, z['state_features']])
        validity = np.concatenate([np.ones((1836,16), bool), z['validity']])
    targets=expected_targets.copy();state=np.empty((3394,9),dtype=np.float32)
    raw_texts=[None]*3394;grouped=defaultdict(list)
    for i,row in enumerate(rows):grouped[row['shard'],row['record_index']].append(i)
    audit=read(pre/'input_reference/report.json');pool=read(pre/'train_manifest.json')
    stats=read(pre/'state_protocol.json')['state_features']
    sources={(e['shard'],e['record_index']):e for e in audit['episodes'] if e['role']=='train_delta'}
    metadata={(e['shard'],e['record_index']):e for e in pool['partitions']['train']}
    by_shard=defaultdict(list)
    for shard,index in grouped:by_shard[shard].append(index)
    seen=0
    for shard,indices in sorted(by_shard.items()):
        for index,payload,values in records_at(train_root/shard,indices):
            identity=shard,index
            assert hashlib.sha256(payload).hexdigest()==sources[identity]['payload_sha256']
            raw,actions,grip=episode_arrays(values,metadata[identity],bridge_reverse_scan_gripper)
            for i in grouped[identity]:
                t=rows[i]['start_index'];raw_texts[i]=values['steps/language_instruction'][t].decode()
                state[i]=state_features(raw[t:t+1,:6],stats)[0]
                if i>=1836:
                    item=build_numeric_sample(raw,grip,t,relative_pose_action,euler_xyz_to_quaternion)
                    assert np.array_equal(item['valid'],validity[i]);targets[i]=item['target']
            seen+=1
    assert seen==405 and all(v is not None for v in raw_texts)
    assert float(abs(state-expected_state).max())<=1e-6
    assert np.array_equal(np.isnan(targets),np.isnan(expected_targets))
    assert float(abs(targets[validity]-expected_targets[validity]).max())<=1e-6
    texts = tokenizer(raw_texts, padding='max_length', max_length=width,
                      return_tensors='pt', truncation=False)
    assert texts['input_ids'].shape == (3394, width)
    lookup = {}
    for i in range(1836):
        lookup.setdefault(tuple(texts['input_ids'][i].tolist()), i)
    representatives = [lookup[tuple(texts['input_ids'][i].tolist())] for i in range(1836,3394)]
    key = delta['key']
    assert representatives == key['representative_language_rows']
    assert key['protocol_sha256'] == sha(pre/'protocol.json')
    assert key['input_audit_sha256'] == sha(pre/'input_reference/report.json')
    assert key['old_cache_key'] == old['key'] and key['old_cache_hashes'] == old['hashes']
    language_indices = torch.arange(3394)
    language_indices[1836:] = torch.tensor(representatives)
    assert torch.equal(texts['attention_mask'][1836:], original_text['attention_mask'][representatives])
    assert torch.equal(texts['input_ids'][:1836], original_text['input_ids'][:1836])
    assert targets.shape == (3394,16,8) and state.shape == (3394,9)
    assert np.isfinite(targets[validity]).all() and np.isfinite(state).all()
    return dict(rows=rows, targets=torch.from_numpy(targets).cuda(),
        state=torch.from_numpy(state).cuda(), validity=torch.from_numpy(validity).cuda(),
        language_indices=language_indices, attention=texts['attention_mask'])


def encode_validation(pre, model, tokenizer, sha, tsha, output):
    """Read/encode only the50 qualified development observations, exactly once."""
    from dataset import prepare_image,relative_pose_action,euler_xyz_to_quaternion,bridge_reverse_scan_gripper
    from state_input_contract import state_features
    from bridge_prefix_input_contract_v1 import records_at,image_contract,episode_arrays
    from bridge_partial_future_contract_v1 import build_numeric_sample
    audit = read(pre/'input_reference/report.json')
    rows = read(pre/'input_reference/validation_rows.json')
    checks = audit['rows']['validation']
    assert len(rows) == len(checks) == 50 and [c['row'] for c in checks] == rows
    images, texts = [None]*50, [None]*50
    grouped = defaultdict(list)
    for i,r in enumerate(rows):
        grouped[r['shard'], r['record_index']].append(i)
    with np.load(pre/'input_reference/validation_reference.npz', allow_pickle=False) as z:
        targets, states = z['targets'].copy(), z['state_features'].copy()
        grid_reference = z['normalized_image_grid'].copy()
        assert z['validity'].all()
    expected_targets,expected_states=targets.copy(),states.copy()
    metadata={(e['shard'],e['record_index']):e for e in read(pre/'protocol.json')['validation']['episodes']}
    stats=read(pre/'state_protocol.json')['state_features']
    assert len(grouped) == 10 and targets.shape == (50,16,8)
    for (shard,index), indices in grouped.items():
        path = pre/'validation_data'/shard
        assert sha(path) == audit['validation_export_sha256'][shard]
        record, payload, values = next(records_at(path, [index]))
        source = next(e for e in audit['episodes'] if
            (e['role'], e['shard'], e['record_index']) == ('validation', shard, index))
        assert record == index and hashlib.sha256(payload).hexdigest() == source['payload_sha256']
        raw,actions,grip=episode_arrays(values,metadata[shard,index],bridge_reverse_scan_gripper)
        for i in indices:
            t=rows[i]['start_index']
            item=build_numeric_sample(raw,grip,t,relative_pose_action,euler_xyz_to_quaternion)
            assert item['valid'].all();targets[i]=item['target']
            states[i]=state_features(raw[t:t+1,:6],stats)[0]
            image, actual, grid = image_contract(values,t,prepare_image)
            assert all(actual[k] == checks[i][k] for k in actual)
            assert float(abs(grid-grid_reference[i]).max()) <= 1e-6
            images[i], texts[i] = image, actual['instruction']
    assert all(v is not None for v in images+texts)
    assert float(abs(targets-expected_targets).max())<=1e-6
    assert float(abs(states-expected_states).max())<=1e-6
    tokenized = tokenizer(texts, padding=True, max_length=77, truncation=False, return_tensors='pt')
    assert tokenized['input_ids'].shape[1] <= 77
    model.vision_encoder.cuda().eval(); model.text_encoder.cuda().eval()
    visual, language = [], []
    with torch.no_grad():
        for start in range(0,50,2):
            visual.append(model.vision_encoder(pixel_values=torch.stack(images[start:start+2]).cuda()).last_hidden_state.cpu())
            tokens = {k:v[start:start+2].cuda() for k,v in tokenized.items()}
            language.append(model.text_encoder(**tokens).last_hidden_state.cpu())
    model.vision_encoder.cpu(); model.text_encoder.cpu(); torch.cuda.empty_cache()
    visual, language = torch.cat(visual), torch.cat(language)
    assert torch.isfinite(visual).all() and torch.isfinite(language).all()
    np.savez_compressed(output/'validation_inputs.npz', targets=targets, state_features=states)
    (output/'validation_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    return dict(rows=rows, targets=targets, state=torch.from_numpy(states).cuda(),
        visual=visual, language=language, attention=tokenized['attention_mask'],
        visual_sha256=tsha(visual), language_sha256=tsha(language))


def run_arm(interface, payload, trainables, context, data, schedule, horizons, arm, output, P, report):
    """The SAME8-step preflight calculation extended once to the fixed4096."""
    interface.base.load_state_dict(payload['trainable_state_dict'],strict=False)
    torch.nn.init.zeros_(interface.prefix.request_projection.weight)
    initial = P.fingerprint(interface.trainable_payload())
    optimizer = torch.optim.Adam(trainables, lr=1e-5, weight_decay=0.)
    assert not optimizer.state
    losses, seconds = [], []
    for update in range(1,4097):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        ids, K = schedule[update-1], int(horizons[update-1])
        interface.train()
        torch.manual_seed(620000+update); torch.cuda.manual_seed_all(620000+update)
        ctx = context(ids)
        torch.manual_seed(610000+update); torch.cuda.manual_seed_all(610000+update)
        loss, _, pred = interface.loss(data['targets'][ids,:,:7],data['validity'][ids],ctx,K)
        assert torch.isfinite(loss), (arm, update, 'nonfinite loss')
        loss.backward()
        assert all(v.grad is None for v in interface.base.parameters() if not v.requires_grad)
        torch.nn.utils.clip_grad_norm_(trainables,1.,error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        losses.append(float(loss.detach())); seconds.append(time.monotonic()-tick)
        del ctx, loss, pred
        if update <= 2 or update%256 == 0:
            print('PREFIX_FORMAL_UPDATE',arm,update,'/4096','K',K,'loss',losses[-1],flush=True)
    final = interface.trainable_payload()
    assert P.fingerprint(final) != initial
    assert torch.count_nonzero(final['request_projection_state_dict']['weight']) > 0
    path = output/f'{arm}_u4096.pt'
    assert not path.exists() and not path.with_suffix('.partial').exists()
    saved = dict(model=final, config=copy.deepcopy(payload['config']), arm=arm, updates=4096,
        protocol_sha256=report['protocol_sha256'], initial_checkpoint_sha256=report['initial_checkpoint_sha256'],
        formal_source_sums_sha256=report['formal_source_sums_sha256'],
        final_checkpoint_only=True, gripper_trained=False)
    torch.save(saved,path.with_suffix('.partial')); path.with_suffix('.partial').replace(path)
    restored = torch.load(path,map_location='cpu',weights_only=True)
    assert P.fingerprint(restored['model']) == P.fingerprint(final)
    interface.load_trainable_payload(restored['model'])
    assert P.fingerprint(interface.trainable_payload()) == P.fingerprint(final)
    np.savez_compressed(output/f'{arm}_losses.npz',losses=np.asarray(losses),requested_horizons=horizons)
    result = dict(arm=arm,logical_optimizer_updates=4096,K16_updates=2048,K1_updates=2048,
        optimizer_initial_state_empty=True,initial_trainable_sha256=initial,
        final_trainable_sha256=P.fingerprint(final),checkpoint=path.name,checkpoint_sha256=P.sha(path),
        checkpoint_roundtrip_exact=True,first8_losses=losses[:8],
        mean_loss_by_request={str(K):float(np.mean(np.asarray(losses)[horizons==K])) for K in (1,16)},
        warm_update_seconds=float(np.mean(seconds[8:])))
    del final, saved, restored, optimizer
    return result


def evaluate(interface, validation, K, seeds):
    interface.eval(); all_predictions = []
    with torch.no_grad():
        for seed in seeds:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            predictions = []
            for start in range(0,50,2):
                sl = slice(start,start+2)
                ctx = interface.context_from_tokens(validation['visual'][sl].cuda(),
                    validation['language'][sl].cuda(), validation['attention'][sl].cuda(),
                    validation['state'][sl])
                predictions.append(interface.sample_pose(ctx,K).cpu().numpy())
            all_predictions.append(np.concatenate(predictions))
            print('PREFIX_DEV_SAMPLE','K',K,'seed',seed,'50/50',flush=True)
    return np.stack(all_predictions)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    contract = read(CODE/'formal_contract.json')
    pre = args.project/'bridge_prefix_pair_v1'
    sys.path[:0] = [str(pre),str(pre/'frozen')]
    import run_preflight as P
    from transformers import CLIPTokenizer
    from models import RobotAdapterModel
    from bridge_prefix_model_interface_v1 import PrefixModelInterface
    args.output.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    r = dict(stage='formal_pair', execution_checks_passed=False, generalization_verdict_available=False,
        protocol_sha256=P.sha(pre/'protocol.json'), formal_source_sums_sha256=P.sha(CODE/'SHA256SUMS'),
        final25_payloads_read=False, old40_or_consumed13_rescored=False, gripper_trained=False,
        downloads=False, training_runs=[], preflight_job=209420, user_verified_preflight=True)
    def persist():
        r['elapsed_seconds'] = time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    try:
        check_sums(CODE,P.sha);check_sums(pre,P.sha)
        assert P.sha(pre/'SHA256SUMS') == contract['preflight_source_sums_sha256']
        assert r['protocol_sha256'] == contract['protocol_sha256']
        # Verify the ACTUAL complete report; the user terminal summary alone is not substituted.
        preflight_dir = args.project/'runs/bridge-prefix-preflight-209420'
        subprocess.run([sys.executable,str(pre/'verify_prefix_preflight.py'),str(preflight_dir)],check=True)
        preflight = read(preflight_dir/'report.json')
        r['preflight_report_sha256'] = P.sha(preflight_dir/'report.json')
        assert torch.cuda.is_available() and str(torch.__version__) == read(pre/'old_report.json')['torch_version']
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True);torch.cuda.reset_peak_memory_stats()
        p = read(pre/'protocol.json')
        assert p['recipe']['new_updates_per_arm'] == 4096 and p['recipe']['batch_size'] == 8
        checkpoint = args.project/'runs/bridge-coverage-rep-fit-207558/final_trainable.pt'
        r['initial_checkpoint_sha256'] = P.sha(checkpoint)
        assert r['initial_checkpoint_sha256'] == p['initialization']['sha256']
        payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg = copy.deepcopy(payload['config']); original = read(pre/'base_protocol.json')['config']
        a,b = copy.deepcopy(cfg),copy.deepcopy(original)
        a['model']['name']=b['model']['name']='<pinned_local_CLIP>';assert a == b
        old = torch.load(args.project/'bridge_training_coverage_v1/full_tokens.pt',map_location='cpu',weights_only=True)
        expected_key = read(pre/'old_report.json')['token_cache_key_sha256']
        assert hashlib.sha256(json.dumps(old['key'],sort_keys=True).encode()).hexdigest() == expected_key
        for k in ('visual','language'): assert P.tensor_sha(old[k]) == old['hashes'][k]
        delta = torch.load(pre/'delta_visual_tokens.pt',map_location='cpu',weights_only=True)
        assert delta['visual'].shape == (1558,257,1024)
        assert P.tensor_sha(delta['visual']) == delta['visual_sha256']
        assert hashlib.sha256(json.dumps(delta['key'],sort_keys=True).encode()).hexdigest() == preflight['delta_cache_key_sha256']
        cfg['model']['name'] = str(args.project/'baseline_setup/clip-vit-large-patch14-v1')
        torch.manual_seed(42);model = RobotAdapterModel(cfg).cpu().eval()
        clip_hash = [P.digest(model.vision_encoder),P.digest(model.text_encoder)]
        assert clip_hash == old['key']['clip']
        model.history_projection=torch.nn.Linear(1024,1024,bias=False)
        model.state_projection=torch.nn.Linear(9,1024,bias=False)
        prefixes=('adapter.','diffusion_decoder.','history_projection.','state_projection.')
        for n,v in model.named_parameters():v.requires_grad_(n.startswith(prefixes))
        assert {n for n,v in model.named_parameters() if v.requires_grad} == set(payload['trainable_state_dict'])
        assert not model.load_state_dict(payload['trainable_state_dict'],strict=False).unexpected_keys
        tokenizer=CLIPTokenizer.from_pretrained(cfg['model']['name'],local_files_only=True)
        data=load_training_inputs(pre,args.project/'bridge_expansion_training_v1/data',old,delta,tokenizer,P.sha,P.tensor_sha)
        for name in ('adapter','diffusion_decoder','history_projection','state_projection'):getattr(model,name).cuda()
        for name,value in list(model.named_buffers(recurse=False)):setattr(model,name,value.cuda())
        assert model.alpha_bars.is_cuda
        interface=PrefixModelInterface(model).eval()
        trainables=[v for v in interface.parameters() if v.requires_grad]
        assert sum(v.numel() for v in trainables) == 132326023
        frozen_hash=P.fingerprint({n:v for n,v in model.named_parameters() if not v.requires_grad})
        def context(ids):
            ids=torch.as_tensor(ids,dtype=torch.long)
            visual=torch.empty((len(ids),257,1024),dtype=old['visual'].dtype)
            was_old=ids<1836
            if was_old.any():visual[was_old]=old['visual'][ids[was_old]]
            if (~was_old).any():visual[~was_old]=delta['visual'][ids[~was_old]-1836]
            language=old['language'][data['language_indices'][ids]]
            return interface.context_from_tokens(visual.cuda(),language.cuda(),
                data['attention'][ids].cuda(),data['state'][ids.cuda()])
        with np.load(pre/'matched_schedule.npz',allow_pickle=False) as z:
            schedules={k:z[k].copy() for k in ('control','candidate')};horizons=z['requested_horizons'].copy()
        assert horizons.shape==(4096,) and np.array_equal(horizons,np.tile([16,1],2048))
        assert all(v.shape==(4096,8) for v in schedules.values())
        assert np.array_equal(schedules['control'][horizons==16],schedules['candidate'][horizons==16])
        assert schedules['control'].min()>=0 and schedules['control'].max()<1836
        assert schedules['candidate'].min()>=0 and schedules['candidate'].max()<3394
        # Admission resources are projected for750 sampled observations (full16+K1),
        # rather than silently using the preflight's smaller546-observation estimate.
        sample_seconds=(preflight['estimated_paired_formal_seconds']-120.)/1.2-preflight['warm_update_seconds']*8192
        estimate=120.+1.2*(preflight['warm_update_seconds']*8192+max(0.,sample_seconds)*750/546)
        assert estimate<=7200
        r.update(estimated_formal_seconds_including_auxiliary=estimate, formal_sampled_observations=750,
            delta_visual_cache_reused=True, old_visual_cache_reused=True,
            final_checkpoint_only=True, initialization='same207558 and zero request port; freshAdam each arm')
        persist()
        for arm in ('control','candidate'):
            arm_result=run_arm(interface,payload,trainables,context,data,schedules[arm],horizons,arm,args.output,P,r)
            r['training_runs'].append(arm_result)
            assert P.fingerprint({n:v for n,v in model.named_parameters() if not v.requires_grad})==frozen_hash
            persist()
        assert r['training_runs'][0]['initial_trainable_sha256']==r['training_runs'][1]['initial_trainable_sha256']
        assert r['training_runs'][0]['first8_losses'][0]==r['training_runs'][1]['first8_losses'][0]
        r['both_weights_frozen_before_development_scoring']=True;persist()
        # No current405 re-fit or old development rescoring. New50 only.
        validation=encode_validation(pre,model,tokenizer,P.sha,P.tensor_sha,args.output)
        r['validation_tokens']={k:validation[k] for k in ('visual_sha256','language_sha256')}
        results={};auxiliary={};seeds=p['recipe']['diffusion_sampling_seeds']
        for method in ('unchanged207558','control','candidate'):
            if method=='unchanged207558':
                assert not model.load_state_dict(payload['trainable_state_dict'],strict=False).unexpected_keys
                torch.nn.init.zeros_(interface.prefix.request_projection.weight)
            else:
                run=next(v for v in r['training_runs'] if v['arm']==method)
                path=args.output/run['checkpoint'];assert P.sha(path)==run['checkpoint_sha256']
                saved=torch.load(path,map_location='cpu',weights_only=True)
                interface.load_trainable_payload(saved['model']);del saved
            predictions=evaluate(interface,validation,16,seeds)
            np.savez_compressed(args.output/f'{method}_full16.npz',predictions=predictions,
                targets=validation['targets'],sampling_seeds=np.asarray(seeds))
            results[method]=summarize(predictions,validation['targets'],validation['rows'])
            print('PREFIX_FULL16_METRICS',method,json.dumps(results[method]['episode_equal']),flush=True)
            if method!='unchanged207558':
                one=evaluate(interface,validation,1,seeds)
                np.savez_compressed(args.output/f'{method}_K1_auxiliary.npz',predictions=one,
                    targets=validation['targets'][:,:1],sampling_seeds=np.asarray(seeds))
                auxiliary[method]=summarize(one,validation['targets'][:,:1],validation['rows'])
        identity=np.zeros((1,50,16,7),dtype=np.float32);identity[...,6]=1.
        results['static']=summarize(identity,validation['targets'],validation['rows'])
        assert [P.digest(model.vision_encoder),P.digest(model.text_encoder)]==clip_hash
        assert P.fingerprint({n:v for n,v in model.named_parameters() if not v.requires_grad})==frozen_hash
        r.update(results_full16=results, auxiliary_K1=auxiliary, adoption=admission(results),
            clip_unchanged=True,gripper_unchanged=True,optimization_operations=8192,
            development_episodes=10,development_windows=50, weights_selected_by_development=False,
            generalization_verdict_available=True,execution_checks_passed=True,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            limits='Same-task offline new-episode pose only; frozen gripper, no closed-loop/unseen-task claim.')
        assert r['peak_reserved_gib']<=7.5
        print('PREFIX_DEVELOPMENT_ADMISSION',json.dumps(r['adoption']),flush=True)
    except Exception as error:
        r['error']=repr(error);r['execution_checks_passed']=False;raise
    finally:
        persist();print('PREFIX_FORMAL_REPORT',args.output/'report.json',flush=True)
    print('PREFIX FORMAL PAIR COMPLETE; execution passing is not an improvement verdict',flush=True)


if __name__=='__main__':main()
