"""Explicitly bind the passed source engineering preflight to this A40 pair."""
import hashlib
import json
from pathlib import Path

BUNDLE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit(directory, actual=None):
    c=read(BUNDLE/'contract.json')
    old=read(BUNDLE/'clip207558_report.json')
    p=read(directory/'report.json')
    assert sha(BUNDLE/'source_dino_only_contract.json')==c['source_dino_only_contract_sha256']
    assert p['contract_sha256']==c['source_dino_only_contract_sha256']
    assert p['experiment']=='bridge_dino_only_v1' and p['stage']=='preflight'
    assert not p['local_bounded'] and p['execution_checks_passed']
    assert p['branch_restore_passed'] and p['optimization_operations']==16
    assert p['final_update']==8 and not p['development_scored']
    assert p['gpu']==old['gpu'] and p['torch_version']==old['torch_version']
    assert p['full_input_contract_passed'] and p['frozen_visual_and_text_unchanged']
    assert p['other_frozen_unchanged'] and not p['gripper_trained']
    assert p['dino_initialization_preserves_cpu_cuda_rng'] and p['language_tokens_exact_original207558']
    for k in ['initial_shared_sha256','augmented_initial_sha256','input_identity_sha256',
              'input_row_checks','episode_payload_checks','retained_state_features_sha256',
              'trainable_parameters','initialization_seed','sampling_steps','sampling_seeds']:
        assert p[k]==old[k], k
    assert p['entry_sha256']==sha(BUNDLE/'frozen/bridge_coverage_replication_v1/run_training_coverage.py')
    assert p['protocol_sha256']==sha(BUNDLE/'frozen/bridge_coverage_replication_v1/protocol.json')
    assert p['input_helper_sha256']==c['source_dino_helper_sha256']
    assert p['peak_reserved_gib']<7.5 and p['estimated_formal_seconds']<9000
    if actual is not None:
        for k in ['protocol_sha256','entry_sha256','input_identity_sha256','input_row_checks',
                  'episode_payload_checks','retained_state_features_sha256']:
            assert actual[k]==p[k], k
        actual['source_engineering_preflight_binding']=dict(job=228978,
            source_gpu=p['gpu'],training_gpu_expected='NVIDIA A40',
            new_a40_engineering_updates=0,legacy_helper_sha256=p['input_helper_sha256'],
            pair_helper_sha256=actual['input_helper_sha256'],
            binding_change='New declared pair helper; exact real input and original initial parameters remain enforced')
    return p


def validate_cache_bytes(root):
    """Check both full cached tensor payloads before spending either arm budget."""
    import torch
    old=read(BUNDLE/'clip207558_report.json')
    passed=admit(root/'runs/bridge-dino-only-preflight-228978')
    result={}
    for arm,path,expected in [
        ('clip',root/'bridge_training_coverage_v1/full_tokens.pt',old['token_cache_key_sha256']),
        ('dino',root/'runs/bridge-dino-only-cache-v1/full_tokens.pt',passed['token_cache_key_sha256'])]:
        cache=torch.load(path,map_location='cpu',weights_only=True)
        key=hashlib.sha256(json.dumps(cache['key'],sort_keys=True).encode()).hexdigest()
        assert key==expected,arm
        assert cache['key']['input_identity_sha256']==old['input_identity_sha256']
        for name in ('visual','language'):
            digest=hashlib.sha256(cache[name].contiguous().numpy().tobytes()).hexdigest()
            assert digest==cache['hashes'][name],(arm,name)
        assert tuple(cache['visual'].shape)==(1876,257,1024)
        result[arm]=dict(key_sha256=key,visual_sha256=cache['hashes']['visual'],
            language_sha256=cache['hashes']['language'],payload_hashes_checked=True)
        del cache
    assert result['clip']['language_sha256']==result['dino']['language_sha256']
    return result
