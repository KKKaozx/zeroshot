"""Verify downloaded completed policy/report bytes, without constructing models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43), default=42)
    args = parser.parse_args()
    endpoints = {
        42: (227223, 'b3b59ab65e0c422d5de774630749bc27afb943edf4fdfd854a2a3cd96daf76fc',
             '2295d8add3e735a7340eb8e244fbc1d4b6abc09d95692120fde1573303abd66d'),
        43: (227981, '243800bd3c304728b2ae96a38fd7ccd1e754f7030b160711fc6704acec4db29f',
             'f032cf513759c267d9e96dd42211a3d327c1e538f4d38d6caff9a0c1704fdecf'),
    }
    job, expected_contract, expected_learned = endpoints[args.seed]
    import torch
    from completed_policy_export import contract_sha, model_sha
    from step_training_state import UpdateSchedule
    folder = ROOT / f'reports/oxe_noaug_seed{args.seed}_{job}/cluster'
    archive = folder / f'oxe-noaug-s{args.seed}-{job}-reports.tar.gz'
    wanted = {'step_report.json', 'stage1_report.json', 'stage2_report.json',
              'completed_policy_export.json', 'final_policy.pt.sha256'}
    extracted = folder / 'reports'
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(wanted) or {m.name for m in members} != wanted:
            raise ValueError('Unexpected report archive members')
        for m in members:
            if not m.isfile() or Path(m.name).name != m.name:
                raise ValueError('Only the explicit regular report files may be extracted')
            data = tar.extractfile(m).read()
            output = extracted / m.name
            if output.exists() and output.read_bytes() != data:
                raise ValueError('Existing extracted report differs; preserve it')
            output.write_bytes(data)
    expected, filename = (extracted / 'final_policy.pt.sha256').read_text().strip().split()
    assert filename == 'final_policy.pt'
    policy_path = folder / filename
    actual = sha(policy_path)
    assert actual == expected
    checkpoint = torch.load(policy_path, map_location='cpu', weights_only=False, mmap=True)
    assert checkpoint['format'] == 'main-completed-fixed-policy-v1'
    contract = checkpoint['training_contract']
    assert contract_sha(contract) == checkpoint['training_contract_sha256'] == expected_contract
    learned = model_sha(checkpoint['trainable_state_dict'])
    assert learned == checkpoint['learned_state_sha256'] == expected_learned
    assert checkpoint['completed_updates'] == contract['arguments']['total_updates'] == 65536
    assert not checkpoint['optimizer_schedule_rng_resume_state_included']
    assert checkpoint['config'] == contract['config']
    assert contract['arguments']['seed'] == args.seed and contract['arguments']['language_condition'] == 'no_augmentation'
    assert not contract['arguments']['oxe_input_preflight']
    assert contract['offline_training_protocol']['sha256'] == '297e03087cf4efaf61b476b84b046404565014984d32c0d777a03b62fada51be'
    split = contract['split_indices']
    assert [len(split[k]) for k in ('train','validation','test')] == [145812, 17949, 0]
    records = {name: json.loads((extracted / name).read_text(encoding='utf-8')) for name in wanted if name.endswith('.json')}
    stage1, stage2, step, export = (records[k] for k in ('stage1_report.json','stage2_report.json','step_report.json','completed_policy_export.json'))
    assert stage1['passed'] and stage1['completed_updates'] == 32768
    assert stage2['passed'] and stage2['completed_updates'] == 65536 and stage2['policy_export'] == export
    assert export['passed'] and export['all_learned_tensors_preserved_sha256'] == learned
    assert export['policy_bytes'] == policy_path.stat().st_size == 526910896 and export['full_optimizer_state_discarded']
    assert step['complete'] and step['completed_updates'] == step['total_updates'] == 65536
    assert step['frozen_state_unchanged'] and step['formal_training_admitted'] and not step['input_preflight_only']
    assert not step['test_partition_evaluated'] and not step['formal_libero_evaluation_completed']
    assert step['config'] == checkpoint['config'] and step['generalization_verdict'] is None
    history = step['updates']
    assert len(history) == 65536
    schedule = UpdateSchedule(total_updates=65536, base_lrs=[.0002,.0002], kind='cosine', warmup_updates=1024)
    samples = 0
    lr_max_abs = 0.
    lr_nonexact_values = 0
    # This file audits Linux records on Windows. Double-precision libm cosine
    # can differ at its last bit. Bound the absolute forward rounding error
    # by four machine epsilons times base LR; no training/resume guard changes.
    lr_absolute_bound = 4 * math.ulp(1.0) * .0002
    for i, row in enumerate(history, 1):
        assert row['update'] == i
        assert all(math.isfinite(row[k]) for k in ('loss','pose_loss','gripper_loss','gradient_norm'))
        expected_rates = schedule.rates(i - 1)
        assert len(row['lr']) == len(expected_rates)
        for actual_lr, expected_lr in zip(row['lr'], expected_rates):
            error = abs(actual_lr - expected_lr)
            assert math.isfinite(actual_lr) and error <= lr_absolute_bound
            lr_max_abs = max(lr_max_abs, error)
            lr_nonexact_values += actual_lr != expected_lr
        assert len(row['indices']) in (4, 8) and all(type(j) is int and 0 <= j < 145812 for j in row['indices'])
        assert len(row['valid_dimensions']) == 8 and all(math.isfinite(x) and x >= 0 for x in row['valid_dimensions'])
        samples += len(row['indices'])
    assert [v['update'] for v in step['validation']] == [32768,65536]
    assert all(math.isfinite(v['loss']) for v in step['validation'])
    code = {name: {'training': contract['source_code_sha256'][name], 'local': sha(ROOT / 'code' / name)}
            for name in ('models.py','adapter.py','diffusion_decoder.py','cartesian_action_contract.py')}
    assert all(value['training'] == value['local'] for value in code.values())
    result = {'passed':True, 'seed':args.seed,'endpoint_job_id':job,'report_archive_sha256':sha(archive),
        'policy_bytes':policy_path.stat().st_size,'policy_file_sha256':actual,
        'all_learned_tensors_sha256':learned,'learned_tensor_count':len(checkpoint['trainable_state_dict']),
        'learned_values':sum(v.numel() for v in checkpoint['trainable_state_dict'].values()),
        'training_contract_sha256':checkpoint['training_contract_sha256'], 'completed_updates':65536,
        'all_update_counters_losses_indices_lr_checked':True, 'sample_exposures':samples,
        'cross_platform_lr_recompute':{'nonexact_values':lr_nonexact_values,'max_absolute_difference':lr_max_abs,
            'absolute_rounding_bound':lr_absolute_bound,'training_or_resume_guards_changed':False},
        'validation':step['validation'], 'source_split_counts':{k:len(v) for k,v in split.items()},
        'frozen_state_sha256':contract['frozen_state_sha256'],'runtime_source_code':code,
        'visual_backbone':checkpoint['config']['model']['visual_backbone'],
        'optimizer_resume_state_included':False,'frozen_encoder_reconstruction_performed':False,
        'model_forward_calls':0,'training_updates_executed':0,'libero_physics_steps':0,
        'libero_task_scoring_performed':False,'generalization_verdict':None}
    (folder.parent / 'download_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
