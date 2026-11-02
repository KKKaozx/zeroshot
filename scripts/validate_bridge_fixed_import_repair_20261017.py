"""Reproduce the collision, then exercise actual original fit with absolute binding.

Synthetic CPU fixture only. Stops on entry to advance; no project model/training.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_fixed_context_import_repair_v1'
OLD=ROOT/'cluster_packages/bridge_fixed_context_v1'
OUT=ROOT/'reports/bridge_fixed_import_repair_validation_20261017'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(),'Regression already completed'
    spec=importlib.util.spec_from_file_location('actual_import_repair',PACK/'run_repaired.py')
    repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
    contract=json.loads((PACK/'repair_contract.json').read_text())
    for name,expected in contract['original_files_sha256'].items():assert sha(OLD/name)==expected
    hooks=repair.load_exact('fixed_hooks',OLD/'fixed_hooks.py')
    validator=repair.load_exact('_synthetic_test_absolute_fixed_validator',OLD/'validate_preflight.py')
    p=json.loads((OLD/'protocol.json').read_text())
    project=OUT/'SYNTHETIC_PROJECT_NOT_CLUSTER_RESULT'
    directory=project/'runs/bridge-fixed-preflight-208897'
    directory.mkdir(parents=True)
    for name in ['protocol.json','run_fixed_context.py','fixed_hooks.py','validate_preflight.py']:
        (directory/name).write_bytes((OLD/name).read_bytes())
    (directory/'effective_runner_snapshot.py').write_bytes((OLD/'effective_runner.py.txt').read_bytes())
    f=dict(protocol_sha256=sha(OLD/'protocol.json'),effective_runner_sha256=p['effective_runner_sha256'],
        source_sha256=p['local_package_sha256'],prefix_replay={'passed':True},prefix_retrained=False,
        new_candidate_development_scored=False,joint_first4_replay_max_difference={'loss':0.,'global_unclipped_gradient_norm':0.,'module_gradients':0.})
    for key in ['first_masked_decoder_update_matches_joint','full_restored_masked_continuation_exact',
                'context_parameters_and_adam_unchanged','training_context_dropout_preserved',
                'global_clipping_before_update_mask','shadow_context_gradients_computed','context_update_mask_only']:
        f[key]=True
    r=dict(SYNTHETIC_CPU_FIXTURE_NOT_A_CLUSTER_RESULT=True,stage='preflight',local_bounded=False,
        execution_checks_passed=True,optimization_operations=12,development_scored=False,runs=[],checkpoints=[],
        fixed_context=f,input_row_checks=[{}]*3664,active_token_rows=3664,episode_payload_checks=[{}]*816,
        verified_row_indices=list(range(3664)),new_token_rows_encoded=0,old_token_rows_reencoded=0,
        torch_version='2.5.1+cu118',initialization_seed=42,peak_reserved_gib=1.,estimated_formal_seconds=1.,
        input_identity_sha256='SYNTHETIC_IDENTITY_ONLY')
    for key in ['full_input_contract_passed','retained_numeric_contract_passed','full_token_cache_reused','clip_unchanged','other_frozen_unchanged']:
        r[key]=True
    for key in ['downloads','gripper_trained','simulation_executed','reserved_test_targets_read','state_statistics_changed']:
        r[key]=False
    (directory/'report.json').write_text(json.dumps(r)+'\n',encoding='utf-8')
    validator.validate(directory)
    snapshot=directory/'fixed_hooks.py';saved=snapshot.read_bytes()
    snapshot.write_bytes(saved+b'\n# deliberate synthetic corruption\n')
    try:validator.validate(directory)
    except AssertionError:pass
    else:raise AssertionError('Original integrity gate was weakened')
    snapshot.write_bytes(saved)
    validator.validate(directory)
    collision=ModuleType('validate_preflight')
    collision.__file__=str(ROOT/'cluster_packages/bridge_demonstration_expansion_v1/validate_preflight.py')
    previous=sys.modules.get('validate_preflight');sys.modules['validate_preflight']=collision
    original=hooks.FixedHooks.fit
    dummy=SimpleNamespace(here=project/'bridge_fixed_context_v1',block=False,
                          context_hash=lambda model:'context',context_adam_hash=lambda model,optimizer:'adam')
    calls=[]
    class StoppedBeforeTraining(Exception):pass
    def advance(*args):
        calls.append(args)
        raise StoppedBeforeTraining()
    x=dict(model=None,optimizer=None,r={'input_identity_sha256':'SYNTHETIC_IDENTITY_ONLY'},advance=advance)
    previous_job=os.environ.get('FIXED_PREFLIGHT_JOB');os.environ['FIXED_PREFLIGHT_JOB']='208897'
    try:
        try:original(dummy,x)
        except ImportError as e:
            assert "cannot import name 'validate'" in str(e)
        else:raise AssertionError('Did not reproduce original collision')
        assert not calls
        repair.bind_fit_validator(hooks,validator)
        assert hooks.FixedHooks.fit.__wrapped__ is original
        try:hooks.FixedHooks.fit(dummy,x)
        except StoppedBeforeTraining:pass
        else:raise AssertionError('Expected stop before training')
        assert calls==[('fixed_context',8193,16170)] and dummy.block
        assert sys.modules['validate_preflight'] is collision
    finally:
        if previous is None:sys.modules.pop('validate_preflight',None)
        else:sys.modules['validate_preflight']=previous
        if previous_job is None:os.environ.pop('FIXED_PREFLIGHT_JOB',None)
        else:os.environ['FIXED_PREFLIGHT_JOB']=previous_job
    for name,expected in contract['original_files_sha256'].items():assert sha(OLD/name)==expected
    result=dict(passed=True,original_import_error_reproduced=True,
        repaired_actual_original_fit_reaches_advance_after_full_validator=True,
        original_fit_body_not_recompiled_or_changed=True,conflicting_module_restored_after_error=True,
        absolute_validator_path=str(Path(validator.__file__).resolve()),
        original_validator_rejects_corrupted_snapshot=True,fixture_is_synthetic=True,
        actual_cluster_preflight_report_verified_locally=False,project_model_executed=False,
        training_updates_executed=0,gpu_used=False,original_package_files_unchanged=True,
        repair_contract_sha256=sha(PACK/'repair_contract.json'),test_entry_sha256=sha(Path(__file__)))
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
