"""Stop the unmodified formal entry at its real identity guard; zero forwards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback


class GuardCaptured(BaseException):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_line(source):
    lines = source.splitlines()
    found = [i + 1 for i, line in enumerate(lines)
             if line.strip() == "if frozen!=bound['frozen_state_sha256'] or contract['tokenizer']!=bound['tokenizer']:"]
    if len(found) != 1:
        raise ValueError('Expected exactly one unchanged combined identity guard')
    return found[0]


def collect(frame, reference):
    import torch
    from frozen_identity_compare import compare, SCHEDULE_KEYS
    state = frame.f_locals
    model, saved = state['model'], state['saved']
    bound, contract = state['bound'], state['contract']
    tensors, values = {}, {}
    complete = model.state_dict()
    for key, value in complete.items():
        if key in saved:
            continue
        cpu = value.detach().cpu().contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        tensors[key] = {'dtype': str(cpu.dtype), 'shape': list(cpu.shape),
                        'sha256': hashlib.sha256(raw).hexdigest()}
        if key in SCHEDULE_KEYS:
            values[key] = cpu.tolist()
    actual = {'config': state['config'], 'legacy_frozen_sha256': state['frozen'],
        'tensors': tensors, 'schedule_values': values,
        'key_order': [key for key in complete if key not in saved],
        'tokenizer_before_model': contract['tokenizer'],
        'tokenizer_after_model': contract['tokenizer'], 'expected_tokenizer': bound['tokenizer'],
        'cpu_capability': torch.backends.cpu.get_cpu_capability()}
    report = compare(reference, actual, bound['frozen_state_sha256'])
    # This comparison's second profile is the actual CUDA formal guard, not a CPU reconstruction.
    report = {'cpu_reference_vs_actual_formal_guard': report,
        'actual_frozen_sha256': state['frozen'], 'expected_frozen_sha256': bound['frozen_state_sha256'],
        'frozen_matches_at_actual_guard': state['frozen'] == bound['frozen_state_sha256'],
        'tokenizer_matches_at_actual_guard': contract['tokenizer'] == bound['tokenizer'],
        'actual_tokenizer': contract['tokenizer'], 'expected_tokenizer': bound['tokenizer'],
        'tokenizer_changed_field_names': [key for key in sorted(set(contract['tokenizer']) | set(bound['tokenizer']))
                                        if contract['tokenizer'].get(key) != bound['tokenizer'].get(key)],
        'model_device': str(next(model.parameters()).device), 'gpu': torch.cuda.get_device_name(),
        'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'bound_preflight_job': bound['preflight_job_id'], 'stopped_at_guard_line': frame.f_lineno,
        'model_forward_calls': 0, 'optimizer_updates': 0, 'training_checkpoint_written': False,
        'original_code_modified': False, 'guard_bypassed': False,
        'scope': 'Exact formal initialization through unchanged run_stage.py; stopped BEFORE guard evaluation, no training',
        'formal_retry_submitted': False, 'automatic_following_action': False}
    report['would_fail_original_guard'] = not (report['frozen_matches_at_actual_guard']
                                              and report['tokenizer_matches_at_actual_guard'])
    if not report['would_fail_original_guard']:
        report['classification'] = 'Actual formal guard matches; earlier failure not reproduced; no training started'
    elif not report['tokenizer_matches_at_actual_guard']:
        report['classification'] = 'Actual formal tokenizer identity mismatch captured'
    else:
        report['classification'] = 'Actual formal frozen-state mismatch captured; inspect named tensor comparison'
    return report, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/projects/Zeroshot')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu-report-dir', default='/projects/Zeroshot/runs/oxe-id-cpu-226567')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    bundle = root / 'oxe_dino_noaug_training_v1'
    code = bundle / 'code'
    runner = (code / 'step_training_runner.py').resolve()
    line = guard_line(runner.read_text())
    for record in (bundle / 'SHA256SUMS').read_text().splitlines():
        expected, name = record.split('  ', 1)
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(bundle / name) != expected:
            raise ValueError('Original formal package checksum mismatch: ' + name)
    reference_dir = Path(args.cpu_report_dir)
    reference = json.loads((reference_dir / 'preflight.json').read_text())
    cpu_report = json.loads((reference_dir / 'report.json').read_text())
    binding = json.loads((bundle / 'cluster_preflight_binding.json').read_text())
    if (not cpu_report['preflight_reconstruction_matches_legacy_binding']
            or reference['legacy_frozen_sha256'] != binding['frozen_state_sha256']):
        raise ValueError('Need actual matching CPU reference from 226567')
    training_out = root / 'runs/oxe-dino-noaug-s42'
    if training_out.exists() and any(training_out.iterdir()):
        raise ValueError('Diagnostic requires no existing training outputs; nothing removed')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    # Training entry checks original bundle, data, device and package bindings itself.
    # Global tracing rejects any unexpected forward/optimizer step as a second stop condition.
    runner_path = os.path.normcase(str(runner))
    result = {'guard_captured': False, 'formal_retry_submitted': False}

    def local_trace(frame, event, arg):
        if event == 'line' and frame.f_lineno == line:
            sys.settrace(None)
            report, actual = collect(frame, reference)
            (out / 'actual_profile.json').write_text(json.dumps(actual, indent=2) + '\n')
            result.update(report, guard_captured=True)
            raise GuardCaptured()
        return local_trace

    def dispatch(frame, event, arg):
        filename = frame.f_code.co_filename.replace('\\', '/')
        name = frame.f_code.co_name
        if event == 'call':
            if (name in ('_call_impl', '_wrapped_call_impl') and filename.endswith('/torch/nn/modules/module.py')):
                raise RuntimeError('Unexpected forward blocked by diagnostic')
            if name == 'step' and '/torch/optim/' in filename:
                raise RuntimeError('Unexpected optimizer step blocked by diagnostic')
            if name == 'train_fixed_updates' and os.path.normcase(frame.f_code.co_filename) == runner_path:
                return local_trace
        return None

    original_argv = sys.argv
    sys.argv = [str(bundle / 'run_stage.py'), '--root', str(root), '--seed', '42', '--stage', '1']
    sys.path.insert(0, str(code))
    try:
        sys.settrace(dispatch)
        runpy.run_path(str(bundle / 'run_stage.py'), run_name='__main__')
        raise RuntimeError('Entry returned without reaching expected guard')
    except GuardCaptured:
        pass
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
        raise
    finally:
        sys.settrace(None)
        sys.argv = original_argv
        (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print('ACTUAL_FORMAL_GUARD_REPORT', out / 'report.json', flush=True)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
