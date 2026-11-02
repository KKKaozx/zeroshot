"""Validate one complete operational report; never submits jobs."""
import argparse
import hashlib
import json
from pathlib import Path
CODE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
r=json.loads((args.directory/'report.json').read_text())
p=json.loads((CODE/'protocol.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert r['stage']=='spatial_readout_preflight' and r['execution_checks_passed']
assert r['protocol_sha256']==sha(CODE/'protocol.json')
assert r['package_sums_sha256']==sha(CODE/'SHA256SUMS')
assert r['actual_checkpoint_sha256']==p['checkpoint']['sha256']
for flag in ('development_scored','final25_payloads_read','formal_training_executed','gripper_trained',
             'CLIP_training_allowed','downloads','final_weights_saved','validation_payloads_read'):
    assert not r[flag],flag
assert r['clip_unchanged'] and r['other_frozen_unchanged'] and r['optimization_operations']==16
assert r['images_encoded']==0 and r['numeric_inputs']['episodes']==405 and r['numeric_inputs']['windows']==1836
assert r['numeric_inputs']['target_max_abs']<=1e-6
for name in ('original207558_replay','zero_readout_full_sampler_replay'):
    assert r[name]['passed'] and r[name]['context_max_abs']<=1e-4 and r[name]['prediction_max_abs']<=1e-4
assert [v['arm'] for v in r['runs']]==['control','candidate']
for v in r['runs']:
    assert v['logical_updates']==len(v['losses'])==8
    assert v['base_parameters_changed'] and v['weights_Adam_and_next_calculation_restore_exact']
assert not r['runs'][0]['score_parameters_changed'] and r['runs'][1]['score_parameters_changed']
assert all(v>0 for v in r['runs'][1]['new_score_gradient_norms'].values())
assert r['runs'][0]['initial_base_sha256']==r['runs'][1]['initial_base_sha256']
assert r['runs'][0]['losses'][0]==r['runs'][1]['losses'][0]
assert r['runs'][1]['trainable_parameters']-r['runs'][0]['trainable_parameters']==1026
assert r['peak_reserved_gib']<=7.5 and r['estimated_paired_formal_seconds']<=7200
print('SPATIAL PREFLIGHT VERIFIED; no generalization verdict or automatic formal submission')
print(json.dumps({k:r[k] for k in ('peak_reserved_gib','warm_update_seconds','estimated_paired_formal_seconds')},indent=2))
