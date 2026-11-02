"""Exact old numeric inputs; replace frozen visual tokens only, keep text exact."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from dataset import UnifiedRobotDataset
from state_input_contract import state_features
from frozen_visual_backbone import FrozenDinoVision

BUNDLE = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tsha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_inputs(args, code, protocol, rows, batches, report):
    # The old report authenticates every image/target/current value and payload.
    baseline = read(BUNDLE/'clip207558_report.json')
    assert baseline['execution_checks_passed'] and baseline['initialization_seed'] == 42
    reference = {(r['shard'], r['record_index']): r for r in baseline['episode_payload_checks']}
    expected = {r['index']: r for r in baseline['input_row_checks']}
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault((row['shard'], row['record_index']), []).append(i)
    reader = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size = 16
    reader.bridge_gripper_policy = 'reverse_scan_valid_steps_v2'
    reader.bridge_current_gripper = 'continuous'
    items = [None]*1876
    observed = np.zeros((1876, 6), np.float32)
    row_checks, episode_checks = [], []
    for count, (episode, indices) in enumerate(grouped.items(), 1):
        example = UnifiedRobotDataset._load_tfrecord_example(str(args.data_root/episode[0]), episode[1])
        payload = hashlib.sha256(example.SerializeToString(deterministic=True)).hexdigest()
        assert payload == reference[episode]['payload_sha256'], ('Payload changed', episode)
        states = np.asarray(example.features.feature['steps/observation/state'].float_list.value,
                            dtype='<f4').reshape(-1, 7)
        reader._load_tfrecord_example = lambda *_: example
        for i in indices:
            row = rows[i]
            assert row['start_index']+16 < len(states)
            item = reader._get_tfrecord_bridge_state_action(dict(file_path=str(args.data_root/episode[0]),
                record_index=episode[1], start_index=row['start_index']))
            assert ' '.join(item[0].lower().split()) == row['task']
            actual = dict(index=i, image_sha256=tsha(item[1]), target_sha256=tsha(item[3]),
                current_sha256=tsha(item[2]), encoded_image_sha256=hashlib.sha256(
                    example.features.feature['steps/observation/image_0'].bytes_list.value[row['start_index']]).hexdigest())
            assert actual == expected[i], ('Old numeric input differs', i)
            items[i] = item
            observed[i] = states[row['start_index'], :6]
            row_checks.append(actual)
        episode_checks.append(dict(shard=episode[0], record_index=episode[1], payload_sha256=payload))
        if count % 50 == 0:
            print('DINO_ONLY_INPUT', count, '/', len(grouped), flush=True)
    assert len(items) == 1876 and len(expected) == 1876
    state_protocol = read(code.parent/'bridge_state_conditioning_v1/protocol.json')
    features = state_features(observed, state_protocol['state_features'])
    original = read(code.parent/'bridge_growth128_v1/manifest.json')
    old_rows = original['train']+original['development']
    retained = [next(i for i, row in enumerate(rows) if (row['shard'], row['record_index'], row['start_index']) ==
        (old['shard'], old['record_index'], old['start_index'])) for old in old_rows]
    assert retained == list(range(128))+list(range(1836, 1876))
    assert hashlib.sha256(features[retained].tobytes()).hexdigest() == baseline['retained_state_features_sha256']
    row_checks.sort(key=lambda r: r['index'])
    identity = hashlib.sha256(json.dumps(dict(rows=row_checks, episodes=episode_checks), sort_keys=True).encode()).hexdigest()
    assert identity == baseline['input_identity_sha256']
    report.update(retained_data_checks=baseline['retained_data_checks'],
        retained_numeric_comparisons=baseline['retained_numeric_comparisons'],
        retained_state_checks=baseline['retained_state_checks'],
        retained_state_features_sha256=baseline['retained_state_features_sha256'],
        input_row_checks=row_checks, episode_payload_checks=episode_checks,
        verified_row_indices=list(range(1876)), full_input_contract_passed=True,
        reused_complete_geometry_audit=True, retained_numeric_contract_passed=True,
        input_identity_sha256=identity, input_verification_method='all actual row hashes versus immutable207558 report')
    return items, features, [items[i] for i in retained], baseline['retained_data_checks'], retained


def tokens(args, code, protocol, model, tokenizer, items, old_items, checks, retained_indices, report, digest):
    started = time.monotonic()
    baseline = read(BUNDLE/'clip207558_report.json')
    text = tokenizer([item[0] for item in items], padding=True, truncation=True,
                     max_length=77, return_tensors='pt')
    expected_text = {k: tsha(v) for k, v in text.items()}
    old = torch.load(args.old_cache, map_location='cpu', weights_only=True)
    assert old['key']['input_identity_sha256'] == baseline['input_identity_sha256']
    assert hashlib.sha256(json.dumps(old['key'], sort_keys=True).encode()).hexdigest() == baseline['token_cache_key_sha256']
    assert old['key']['clip'] == [digest(model.vision_encoder), digest(model.text_encoder)]
    assert old['key']['tokenizer_inputs'] == expected_text
    assert old['hashes'] == {k: tsha(old[k]) for k in ['visual', 'language']}
    language = old['language']
    language_hash = old['hashes']['language']
    del old
    settings = read(BUNDLE/'contract.json')['dino_settings']
    # Use the explicit root supplied by the launcher, not an inferred assets path.
    import os
    settings['dino_assets'] = os.environ['BRIDGE_DINO_ASSETS']
    before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    model.vision_encoder.cpu()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model.vision_encoder = FrozenDinoVision(settings).cuda().eval()
    assert torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_cuda, torch.cuda.get_rng_state())
    identity = dict(input_identity_sha256=report['input_identity_sha256'], dino_settings=settings,
        visual_state_sha256=digest(model.vision_encoder), language_sha256=language_hash,
        text_encoder_sha256=digest(model.text_encoder), tokenizer_inputs=expected_text,
        torch_version=str(torch.__version__), contract_sha256=sha(BUNDLE/'contract.json'))
    if args.cache.exists():
        cached = torch.load(args.cache, map_location='cpu', weights_only=True)
        assert cached['key'] == identity
        assert cached['hashes'] == {k: tsha(cached[k]) for k in ['visual', 'language']}
        assert torch.equal(language, cached['language'])
        visual, reused = cached['visual'], True
        del cached
    else:
        assert args.stage == 'preflight', 'Formal requires the already checked DINO cache'
        visual = torch.empty((1876, 257, 1024), dtype=torch.float32)
        with torch.no_grad():
            for lo in range(0, len(items), 2):
                value = model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[lo:lo+2]]).cuda()).last_hidden_state
                assert value.dtype == torch.float32 and torch.isfinite(value).all()
                visual[lo:lo+len(value)] = value.cpu()
                if lo % 200 == 0:
                    print('DINO_ONLY_ENCODE', lo, '/1876', flush=True)
        cached = dict(key=identity, visual=visual, language=language)
        cached['hashes'] = {k: tsha(cached[k]) for k in ['visual', 'language']}
        temporary = args.cache.with_suffix('.partial')
        assert not temporary.exists()
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cached, temporary)
        assert temporary.stat().st_size < 3*1024**3
        temporary.replace(args.cache)
        del cached
        reused = False
    report.update(original_token_cache_reused=True, original_cache_reuse_scope='language only; CLIP visual tokens never used for DINO training',
        retained_visual_tokens_exact=False, retained_language_tokens_exact=True,
        full_token_cache_reused=reused, local_full_token_cache_written=False,
        token_rows=1876, active_token_rows=1876, encoding_seconds=time.monotonic()-started,
        token_cache_key_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        dino_assets=settings, dino_initialization_preserves_cpu_cuda_rng=True,
        language_tokens_exact_original207558=True)
    return visual.cuda(), language.cuda(), text['attention_mask'].cuda()
