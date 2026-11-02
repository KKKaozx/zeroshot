"""Four-condition labels and actual train-loop prefix, synthetic reviewed fixtures."""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from language_training_conditions import LanguageAugmentor, validate_resume_contract


def rejected(fn):
    try:
        fn()
    except ValueError:
        return True
    raise AssertionError('Invalid bank or resume contract accepted')


def fixture():
    entries = {}
    for i, name in enumerate(('left', 'right')):
        entries[str(i)] = {'group_key': 'synthetic_train:' + str(i), 'instruction': 'move ' + name,
                           'suite_id': 'synthetic_only', 'goal_id': name,
                           'paraphrases': [f'synthetic wording {j} for {name}' for j in range(5)],
                           'paraphrase_review': {'status': 'verified', 'evidence': 'Test declaration only, not a real reviewed bank'},
                           'hold_gripper_command': {'open_positive': 1 if i == 0 else -1,
                                                    'source_evidence': 'Synthetic known command; not sensed opening'},
                           'counterfactuals': [{'donor_index': 1 - i,
                                               'instruction': 'move ' + ('right' if i == 0 else 'left'),
                                               'scene_review': {'status': 'verified_mismatch',
                                                                'another_valid_goal': False,
                                                                'evidence': 'Synthetic mismatch fixture only'}}]}
    return {'format': 'reviewed-training-language-bank-v1', 'scope': 'training_only',
            'dataset_identity': 'synthetic_identity', 'entries': entries}


def prefix(tree):
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    loop = next(n for n in ast.walk(train) if isinstance(n, ast.For)
                and isinstance(n.iter, ast.Name) and n.iter.id == 'epoch_loader')
    body = []
    for statement in loop.body:
        body.append(copy.deepcopy(statement))
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'text' for t in statement.targets):
            break
    else:
        raise AssertionError('Could not find original tokenizer boundary')
    return ast.For(target=copy.deepcopy(loop.target), iter=copy.deepcopy(loop.iter), body=body, orelse=[])


def main():
    torch.set_num_threads(1)
    before_path = ROOT / 'reports/language_condition_integration_20261020/train_before.py'
    after_path = ROOT / 'code/train.py'
    before, after = ast.parse(before_path.read_text(encoding='utf-8')), ast.parse(after_path.read_text(encoding='utf-8'))
    old_functions = {n.name: ast.dump(n) for n in before.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new_functions = {n.name: ast.dump(n) for n in after.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert all(value == new_functions[name] for name, value in old_functions.items() if name not in ('train', 'parse_args'))
    args_node = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
    args_namespace = {'argparse': argparse, 'Path': Path, '__file__': str(after_path)}
    exec(compile('from __future__ import annotations\n' + ast.get_source_segment(after_path.read_text(encoding='utf-8'), args_node),
                 str(after_path), 'exec'), args_namespace)
    assert args_namespace['parse_args']([]).language_condition == 'disabled'
    assert args_namespace['parse_args'](['--language-condition', 'combined', '--counterfactual-fraction', '.4']).counterfactual_fraction == .4

    with tempfile.TemporaryDirectory(dir=ROOT / 'reports') as temp:
        path = Path(temp) / 'synthetic_bank.json'
        bank = fixture()
        path.write_text(json.dumps(bank))
        groups = {0: 'synthetic_train:0', 1: 'synthetic_train:1'}
        make = lambda mode, fraction=None: LanguageAugmentor(path, mode, 42, 'synthetic_identity', [0, 1], groups, fraction)
        augmentors = {mode: make(mode, .4 if mode in ('counterfactual_only', 'combined') else None)
                      for mode in ('no_augmentation', 'paraphrase_only', 'counterfactual_only', 'combined')}
        torch.manual_seed(8)
        batch_size, horizon = 128, 16
        indices = [i % 2 for i in range(batch_size)]
        texts = ['move left' if i == 0 else 'move right' for i in indices]
        target = torch.randn(batch_size, horizon, 8)
        masks = torch.ones_like(target)
        global_python, global_numpy, global_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        outputs = {name: aug.apply(texts, target, masks, indices, epoch=2, step=3) for name, aug in augmentors.items()}
        assert random.getstate() == global_python and np.array_equal(np.random.get_state()[1], global_numpy[1])
        assert torch.equal(torch.get_rng_state(), global_torch)
        baseline = outputs['no_augmentation']
        assert baseline[0] == texts and torch.equal(baseline[1], target) and baseline[2] is masks
        paraphrase = outputs['paraphrase_only']
        assert torch.equal(paraphrase[1], target) and {r['kind'] for r in paraphrase[3]} == {'original', 'paraphrase'}
        cf, combined = outputs['counterfactual_only'], outputs['combined']
        assert [r['kind'] == 'counterfactual' for r in cf[3]] == [r['kind'] == 'counterfactual' for r in combined[3]]
        for row, record in enumerate(cf[3]):
            if record['kind'] == 'counterfactual':
                assert cf[0][row] != texts[row] and cf[0][row] == combined[0][row]
                assert torch.equal(cf[1][row], combined[1][row])
                assert bool((cf[1][row, :, :3] == 0).all())
                assert bool((cf[1][row, :, 3:6] == 0).all()) and bool((cf[1][row, :, 6] == 1).all())
                assert bool((cf[1][row, :, 7] == (1 if indices[row] == 0 else -1)).all())
            else:
                assert torch.equal(cf[1][row], target[row])
                assert combined[0][row] == paraphrase[0][row]
        repeated = make('combined', .4).apply(texts, target, masks, indices, epoch=2, step=3)
        assert repeated[0] == combined[0] and torch.equal(repeated[1], combined[1]) and repeated[3] == combined[3]
        validate_resume_contract(augmentors['combined'].contract, make('combined', .4).contract)
        assert rejected(lambda: validate_resume_contract(augmentors['combined'].contract, make('combined', .3).contract))
        assert rejected(lambda: validate_resume_contract(augmentors['combined'].contract, None))
        assert rejected(lambda: make('combined'))
        assert rejected(lambda: LanguageAugmentor(path, 'combined', 42, 'synthetic_identity', [0], {0: groups[0]}, .4))
        assert rejected(lambda: LanguageAugmentor(path, 'combined', 42, 'other_split', [0, 1], groups, .4))

        changed = copy.deepcopy(bank)
        changed['entries']['0']['counterfactuals'][0]['scene_review']['another_valid_goal'] = True
        path.write_text(json.dumps(changed))
        assert rejected(lambda: make('combined', .4))
        changed = copy.deepcopy(bank)
        changed['entries']['1']['goal_id'] = ''
        path.write_text(json.dumps(changed))
        assert rejected(lambda: make('combined', .4))
        changed = copy.deepcopy(bank)
        changed['entries']['0']['hold_gripper_command']['open_positive'] = True
        path.write_text(json.dumps(changed))
        boolean_command = make('counterfactual_only', .4)
        assert rejected(lambda: boolean_command.apply(texts, target, masks, indices, epoch=2, step=3))
        changed = copy.deepcopy(bank)
        changed['entries']['0']['hold_gripper_command'] = None
        path.write_text(json.dumps(changed))
        missing_command = make('counterfactual_only', .4)
        assert rejected(lambda: missing_command.apply(texts, target, masks, indices, epoch=2, step=3))
        partial = masks.clone()
        partial[:, :, 7] = 0
        _, nulls, same_mask, choices = missing_command.apply(texts, target, partial, indices, epoch=2, step=3)
        assert same_mask is partial and torch.equal(partial[:, :, 7], torch.zeros(batch_size, horizon))
        path.write_text(json.dumps(bank))

        images, measured_grippers = torch.randn(batch_size, 3, 2, 2), torch.randn(batch_size, 1) * 10
        batch = (texts, images, measured_grippers, target, masks, indices)
        def execute_prefix(node, augmentor):
            received = []
            namespace = {'epoch_loader': [batch], 'args': argparse.Namespace(max_steps_per_epoch=0, overfit_samples=0),
                         'epoch': 2, 'step_count': 3, 'device': 'cpu', 'language_augmentor': augmentor,
                         'language_condition_counts': Counter(), 'tokenizer': None,
                         'tokenise': lambda tokenizer, instructions, device: received.append(instructions[:])}
            ast.fix_missing_locations(node)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(after_path), 'exec'), namespace)
            assert namespace['images'] is images and namespace['current_grippers'] is measured_grippers
            return namespace, received
        old_ns, old_text = execute_prefix(prefix(before), None)
        new_ns, new_text = execute_prefix(prefix(after), None)
        assert old_text == new_text and old_ns['actions'] is new_ns['actions'] is target
        new_ns, new_text = execute_prefix(prefix(after), make('combined', .4))
        assert new_text == [combined[0]] and torch.equal(new_ns['actions'], combined[1])
        assert new_ns['language_condition_counts'] == Counter(r['kind'] for r in combined[3])

    report = {'checks_passed': True, 'fixtures_only_not_real_reviewed_augmentation_data': True,
              'default_training_prefix_and_unmodified_evaluation_functions_verified': True,
              'actual_train_loop_applies_interventions_before_tokenizer': True,
              'same_counterfactual_draws_in_counterfactual_and_combined_groups': True,
              'original_targets_preserved_for_paraphrases': True,
              'identity_quaternion_and_explicit_gripper_hold_for_nulls': True,
              'partial_supervision_mask_preserved': True,
              'heldout_bank_members_and_changed_contract_rejected': True,
              'known_valid_alternative_goal_rejected': True,
              'global_rng_unchanged_and_language_draws_replay_exact': True,
              'model_loaded': False, 'real_training_updates': 0,
              'train_before_sha256': hashlib.sha256(before_path.read_bytes()).hexdigest(),
              'train_after_sha256': hashlib.sha256(after_path.read_bytes()).hexdigest()}
    (ROOT / 'reports/language_condition_integration_20261020/report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
