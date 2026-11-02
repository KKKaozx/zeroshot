"""Failure-only evidence from the direct training entry; never changes state."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


SCHEDULE_KEYS = {'betas', 'alphas', 'alpha_bars', 'posterior_variance',
                 'posterior_mean_x0', 'posterior_mean_xt'}


def profile(model, saved):
    import torch
    digest = hashlib.sha256()
    tensors, order, values = {}, [], {}
    for key, value in model.state_dict().items():
        if key in saved:
            continue
        cpu = value.detach().cpu().contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        dtype, shape = str(cpu.dtype), list(cpu.shape)
        digest.update(json.dumps([key, dtype, shape]).encode())
        digest.update(raw)
        tensors[key] = {'dtype': dtype, 'shape': shape,
                        'sha256': hashlib.sha256(raw).hexdigest()}
        order.append(key)
        if key in SCHEDULE_KEYS:
            values[key] = cpu.tolist()
    return {'legacy_frozen_sha256': digest.hexdigest(), 'tensors': tensors,
            'key_order': order, 'schedule_values': values}


def write_failure(args, model, saved, frozen, contract, bound):
    import torch
    actual = profile(model, saved)
    report = {
        'scope': 'Failure branch of direct formal entry, before first forward/update',
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'actual_frozen_sha256_at_guard': frozen,
        'expected_frozen_sha256': bound['frozen_state_sha256'],
        'recomputed_frozen_sha256_in_failure_branch': actual['legacy_frozen_sha256'],
        'frozen_matches_at_guard': frozen == bound['frozen_state_sha256'],
        'frozen_changed_between_guard_and_capture': frozen != actual['legacy_frozen_sha256'],
        'actual_tokenizer': contract['tokenizer'], 'expected_tokenizer': bound['tokenizer'],
        'tokenizer_matches_at_guard': contract['tokenizer'] == bound['tokenizer'],
        'tokenizer_changed_field_names': [key for key in sorted(set(contract['tokenizer']) | set(bound['tokenizer']))
                                        if contract['tokenizer'].get(key) != bound['tokenizer'].get(key)],
        'config': contract['config'], 'preflight_job_id': bound['preflight_job_id'],
        'hostname': os.uname().nodename if hasattr(os, 'uname') else None,
        'model_device': str(next(model.parameters()).device),
        'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'cpu_capability': torch.backends.cpu.get_cpu_capability(),
        'intraop_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'python_executable': sys.executable, 'argv': sys.argv, 'cwd': os.getcwd(),
        'python_trace_active': sys.gettrace() is not None,
        'loaded_module_paths': {key: getattr(sys.modules.get(key), '__file__', None)
                                for key in ('train', 'models', 'adapter', 'frozen_visual_backbone',
                                            'step_training_runner', 'transformers', 'torch')},
        'execution_environment': {key: os.environ.get(key) for key in (
            'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'ATEN_CPU_CAPABILITY', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES',
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')},
        'model_forward_calls': 0, 'optimizer_updates': 0,
        'training_checkpoint_written': False, 'guard_bypassed': False,
        'automatic_following_action': False,
    }
    reference_path = os.environ.get('OXE_FROZEN_REFERENCE_JSON')
    report['reference_path'] = reference_path
    if reference_path:
        try:
            reference = json.loads(Path(reference_path).read_text(encoding='utf-8'))
            if reference['legacy_frozen_sha256'] != bound['frozen_state_sha256']:
                raise ValueError('Reference profile does not reproduce the immutable binding')
            left, right = reference['tensors'], actual['tensors']
            changed = [key for key in sorted(set(left) & set(right)) if left[key] != right[key]]
            report.update(
                same_key_order=reference['key_order'] == actual['key_order'],
                missing_keys_in_actual=sorted(set(left) - set(right)),
                extra_keys_in_actual=sorted(set(right) - set(left)),
                changed_encoder_tensors=[key for key in changed if key.startswith(('vision_encoder.', 'text_encoder.'))],
                changed_schedule_buffers=[key for key in changed if key in SCHEDULE_KEYS],
                other_changed_tensors=[key for key in changed if key not in SCHEDULE_KEYS
                                       and not key.startswith(('vision_encoder.', 'text_encoder.'))],
                changed_tensor_metadata={key: {'expected': left[key], 'actual': right[key]} for key in changed})
            import numpy as np
            report['schedule_numeric_differences'] = {}
            for key in changed:
                if key not in SCHEDULE_KEYS:
                    continue
                x = np.asarray(reference['schedule_values'][key])
                y = np.asarray(actual['schedule_values'][key])
                report['schedule_numeric_differences'][key] = (
                    {'max_abs': float(np.max(np.abs(x - y))), 'unequal_values': int(np.count_nonzero(x != y))}
                    if x.shape == y.shape else {'expected_shape': list(x.shape), 'actual_shape': list(y.shape)})
        except Exception as exc:
            report['reference_comparison_error'] = type(exc).__name__ + ': ' + str(exc)
    # Emit useful operands even if writing the optional evidence directory fails.
    print('FORMAL_IDENTITY_FAILURE ' + json.dumps(report, sort_keys=True), flush=True)
    job = os.environ.get('SLURM_JOB_ID', '')
    suffix = job if job.isdigit() else 'local-' + str(time.time_ns())
    output = Path(args.output_dir).resolve().parent / ('oxe-id-failure-' + suffix)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'actual_profile.json').write_text(json.dumps(actual, indent=2) + '\n', encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('FORMAL_IDENTITY_FAILURE_REPORT', output / 'report.json', flush=True)
    return report
