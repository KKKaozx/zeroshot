"""Repair validator resolution only; preserve the frozen experiment and preflight."""
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def load_exact(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        loaded = sys.modules[name]
        assert Path(loaded.__file__).resolve() == path, (name, loaded.__file__, str(path))
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def bind_fit_validator(hooks_module, validator_module):
    """Bind the absolute validator only during the unchanged original fit call."""
    original = hooks_module.FixedHooks.fit
    assert not hasattr(original, '__fixed_validator_repair__'), 'Already patched'

    @functools.wraps(original)
    def repaired(self, x):
        previous = sys.modules.get('validate_preflight')
        sys.modules['validate_preflight'] = validator_module
        try:
            # Original fit body, thresholds, validate call and optimization are unchanged.
            return original(self, x)
        finally:
            if previous is None:
                sys.modules.pop('validate_preflight', None)
            else:
                sys.modules['validate_preflight'] = previous
    repaired.__fixed_validator_repair__ = True
    hooks_module.FixedHooks.fit = repaired
    return original


def main():
    contract = read(HERE/'repair_contract.json')
    assert sha(Path(__file__)) == contract['repair_runner_sha256']
    original_dir = HERE.parent/'bridge_fixed_context_v1'
    for name, expected in contract['original_files_sha256'].items():
        assert sha(original_dir/name) == expected, name
    failed_dir = HERE.parent/'runs/bridge-fixed-fit-208926'
    failed = read(failed_dir/'report.json')
    assert not failed['execution_checks_passed'] and not failed['runs'] and not failed['checkpoints']
    assert failed.get('optimization_operations', 0) == 0
    assert "cannot import name 'validate'" in failed['error'] and 'validate_preflight' in failed['error']
    assert failed['fixed_context']['prefix_replay']['passed']
    assert not (failed_dir/'fixed_context_updates.jsonl').exists()
    assert not list(failed_dir.glob('*.pt')), 'Investigate any existing training result before retrying'
    assert sha(failed_dir/'protocol.json') == contract['experiment_protocol_sha256']
    assert sys.argv[sys.argv.index('--stage')+1] == 'fit'
    output = Path(sys.argv[sys.argv.index('--output')+1])
    assert not output.exists(), 'Preserve earlier outputs'
    job = os.environ['FIXED_PREFLIGHT_JOB']
    assert job == '208897', 'Reuse the already successful frozen preflight'
    hooks = load_exact('fixed_hooks', original_dir/'fixed_hooks.py')
    validator = load_exact('_bridge_fixed_context_absolute_preflight', original_dir/'validate_preflight.py')
    assert Path(validator.__file__).resolve() == (original_dir/'validate_preflight.py').resolve()
    pre = validator.validate(HERE.parent/('runs/bridge-fixed-preflight-'+job))
    print('ABSOLUTE_PREFLIGHT_VALIDATOR VERIFIED', validator.__file__, flush=True)
    bind_fit_validator(hooks, validator)
    original = load_exact('_bridge_fixed_context_original_entry', original_dir/'run_fixed_context.py')
    proof = dict(repair_contract_sha256=sha(HERE/'repair_contract.json'),
        repair_runner_sha256=sha(Path(__file__)), failed_job='208926',
        failed_job_no_new_training_updates_verified=True, reused_preflight_job=job,
        original_preflight_validator_passed=True, validator_absolute_path=str(Path(validator.__file__).resolve()),
        original_files_sha256=contract['original_files_sha256'], experiment_protocol_unchanged=True,
        training_and_evaluation_functions_unchanged=True,
        execution_checks_passed_before_formal=False, generalization_verdict=False)
    try:
        original.main()
    finally:
        if output.exists():
            (output/'import_repair_runner.py').write_bytes(Path(__file__).read_bytes())
            (output/'import_repair_contract.json').write_bytes((HERE/'repair_contract.json').read_bytes())
            (output/'import_repair_entry.json').write_text(json.dumps(proof, indent=2)+'\n')
            report_path = output/'report.json'
            if report_path.exists():
                report = read(report_path)
                report['import_resolution_repair'] = proof
                report_path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
