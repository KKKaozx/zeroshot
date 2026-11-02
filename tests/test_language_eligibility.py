"""Synthetic scene declarations test intervention semantics, never approve data."""
import ast
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from language_training_conditions import (BANK_V1, BANK_V2, LanguageAugmentor,
    OriginalInstructionControl, language_record_counts, validate_resume_contract)


def fixture():
    entries = {}
    for i in range(5):
        entries[str(i)] = {
            'group_key': 'synthetic:' + str(i), 'instruction': 'synthetic goal ' + str(i),
            'suite_id': 'synthetic', 'goal_id': str(i),
            'paraphrases': ['synthetic goal %d wording %d' % (i, j) for j in range(5)],
            'paraphrase_review': {'status': 'verified', 'evidence': 'Synthetic fixture only'},
            'hold_gripper_command': {'open_positive': -1, 'source_evidence': 'Synthetic causal command'},
            'counterfactual_review': {'status': 'available', 'evidence': 'Synthetic scene declaration'},
            'counterfactuals': [{'donor_index': (i + 1) % 5,
                'instruction': 'synthetic goal ' + str((i + 1) % 5),
                'scene_review': {'status': 'verified_mismatch', 'another_valid_goal': False,
                                 'evidence': 'Synthetic mismatch only'}}],
        }
    entries['1']['hold_gripper_command'] = None
    entries['2']['counterfactuals'] = []
    entries['2']['counterfactual_review'] = {'status': 'unavailable',
        'reason': 'No reviewed scene', 'evidence': 'Synthetic unavailable fixture'}
    entries['3']['hold_gripper_command'] = None  # XY only: no gripper command needed.
    return {'format': BANK_V2, 'scope': 'training_only', 'dataset_identity': 'synthetic', 'entries': entries}


class LanguageEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'reports')
        self.path = Path(self.temp.name) / 'bank.json'
        self.bank = fixture()
        self.groups = {i: 'synthetic:' + str(i) for i in range(5)}
        self.indices = list(range(5)) * 64
        self.texts = ['synthetic goal ' + str(i) for i in self.indices]
        self.targets = torch.arange(len(self.indices) * 4 * 8, dtype=torch.float32).reshape(-1, 4, 8) / 1000
        self.masks = torch.ones_like(self.targets)
        for row, index in enumerate(self.indices):
            if index == 3:
                self.masks[row, :, 2:] = 0
            if index == 4:
                self.masks[row, 2:] = 0

    def tearDown(self):
        self.assertEqual(Path(self.temp.name).resolve().parent, (ROOT / 'reports').resolve())
        self.temp.cleanup()

    def make(self, mode='combined', bank=None, seed=42):
        self.path.write_text(json.dumps(bank if bank is not None else self.bank))
        return LanguageAugmentor(self.path, mode, seed, 'synthetic', list(self.groups), self.groups,
                                 .25 if mode in ('combined', 'counterfactual_only') else None)

    def apply(self, aug, step=11):
        return aug.apply(self.texts, self.targets, self.masks, self.indices, epoch=1, step=step)

    def test_paired_negatives_and_positive_fallback_preserve_targets_masks_and_rng(self):
        cf, combined, para = [self.make(mode) for mode in ('counterfactual_only', 'combined', 'paraphrase_only')]
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        original = self.targets.clone()
        a, b, p = self.apply(cf), self.apply(combined), self.apply(para)
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(np.random.get_state(), np_state)))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(torch.equal(self.targets, original))
        self.assertIs(a[2], self.masks)
        self.assertIs(b[2], self.masks)
        realized_by_index = Counter()
        for row, (ra, rb) in enumerate(zip(a[3], b[3])):
            index = self.indices[row]
            self.assertEqual(ra['counterfactual_requested'], rb['counterfactual_requested'])
            self.assertEqual(ra['counterfactual_eligible'], rb['counterfactual_eligible'])
            self.assertEqual(ra['kind'] == 'counterfactual', rb['kind'] == 'counterfactual')
            if ra['kind'] == 'counterfactual':
                realized_by_index[index] += 1
                self.assertEqual(a[0][row], b[0][row])
                self.assertTrue(torch.equal(a[1][row], b[1][row]))
                null = torch.zeros_like(original[row]); null[:, 6] = 1; null[:, 7] = -1
                expected = torch.where(self.masks[row] > 0, null, original[row])
                self.assertTrue(torch.equal(a[1][row], expected))
                if index == 3:
                    self.assertTrue(torch.equal(a[1][row, :, 2:], original[row, :, 2:]))
                if index == 4:
                    self.assertTrue(torch.equal(a[1][row, 2:], original[row, 2:]))
            else:
                self.assertEqual(a[0][row], self.texts[row])
                self.assertTrue(torch.equal(a[1][row], original[row]))
                self.assertTrue(torch.equal(b[1][row], original[row]))
                self.assertEqual(b[0][row], p[0][row])
            if index in (1, 2):
                self.assertFalse(ra['counterfactual_eligible'])
                self.assertNotEqual(ra['kind'], 'counterfactual')
        self.assertEqual(set(realized_by_index), {0, 3, 4})
        counts = language_record_counts(a[3])
        self.assertEqual(counts['counterfactual_opportunities'], 320)
        self.assertEqual(counts['counterfactual_eligible'], 192)
        self.assertEqual(counts['counterfactual_ineligible'], 128)
        self.assertEqual(counts['counterfactual_requested'],
                         counts['counterfactual'] + counts['counterfactual_fallback'])
        self.assertGreater(counts['counterfactual_fallback:unknown_current_gripper_command'], 0)
        self.assertGreater(counts['counterfactual_fallback:no_reviewed_scene_mismatch'], 0)
        self.assertEqual(sum(counts[k] for k in ('original', 'paraphrase', 'counterfactual')), 320)

    def test_reconstructed_controller_replays_and_contract_changes_rejected(self):
        first = self.make()
        a = self.apply(first, step=12)
        restored = self.make()
        validate_resume_contract(first.contract, restored.contract)
        b = self.apply(restored, step=12)
        self.assertEqual(a[0], b[0]); self.assertEqual(a[3], b[3])
        self.assertTrue(torch.equal(a[1], b[1]))
        for contract in (None, self.make(seed=43).contract, dict(first.contract, format='language-condition-training-v1')):
            with self.assertRaises(ValueError):
                validate_resume_contract(first.contract, contract)
        with self.assertRaises(ValueError):
            LanguageAugmentor(self.path, 'combined', 42, 'wrong-split', list(self.groups), self.groups, .25)
        with self.assertRaises(ValueError):
            LanguageAugmentor(self.path, 'combined', 42, 'synthetic', [0], {0: self.groups[0]}, .25)

    def test_invalid_review_or_hold_cannot_be_silently_reclassified(self):
        cases = []
        for mutation in (
            lambda b: b['entries']['0'].pop('counterfactual_review'),
            lambda b: b['entries']['0']['counterfactual_review'].update(status='unavailable', reason='pretend'),
            lambda b: b['entries']['2']['counterfactual_review'].update(status='available'),
            lambda b: b['entries']['0']['counterfactuals'][0]['scene_review'].update(another_valid_goal=True),
            lambda b: b['entries']['0']['counterfactuals'][0].update(donor_index=99),
            lambda b: b['entries']['0']['hold_gripper_command'].update(open_positive=True),
            lambda b: b['entries']['0']['hold_gripper_command'].update(source_evidence=''),
        ):
            b = copy.deepcopy(self.bank); mutation(b); cases.append(b)
        for index, bank in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.make(bank=bank)

    def test_legacy_v1_keeps_strict_requirement_and_old_records(self):
        bank = copy.deepcopy(self.bank); bank['format'] = BANK_V1
        with self.assertRaises(ValueError):
            self.make(bank=bank)
        bank['entries']['2']['counterfactuals'] = copy.deepcopy(self.bank['entries']['0']['counterfactuals'])
        # Fix the donor so entry 2 has a different goal.
        for e in bank['entries'].values():
            e['hold_gripper_command'] = {'open_positive': -1, 'source_evidence': 'Synthetic'}
        legacy = self.make(bank=bank)
        output = self.apply(legacy)
        self.assertEqual(legacy.contract['format'], 'language-condition-training-v1')
        self.assertTrue(all('counterfactual_eligible' not in r for r in output[3]))
        self.assertEqual(language_record_counts(output[3]), Counter(r['kind'] for r in output[3]))
        bank['entries']['1']['hold_gripper_command'] = None
        missing = self.make(bank=bank)
        with self.assertRaises(ValueError):
            self.apply(missing)

    def test_no_augmentation_original_catalog_remains_unchanged(self):
        entries, instructions = {}, []
        for i in self.groups:
            text = 'synthetic goal ' + str(i)
            key = hashlib.sha256(text.encode()).hexdigest()
            entries[str(i)] = {'instruction': text, 'instruction_id': key, 'group_key': self.groups[i]}
            instructions.append({'canonical_instruction': text, 'instruction_id': key})
        catalog = {'format': 'oxe-training-instruction-catalog-v1', 'scope': 'training_only',
                   'dataset_identity': 'synthetic', 'instructions': instructions, 'window_entries': entries}
        self.path.write_text(json.dumps(catalog))
        control = OriginalInstructionControl(self.path, 42, 'synthetic', list(self.groups), self.groups)
        result = self.apply(control)
        self.assertEqual(result[0], self.texts)
        self.assertIs(result[1], self.targets); self.assertIs(result[2], self.masks)
        self.assertEqual(language_record_counts(result[3]), {'original': 320})

    def test_both_real_training_blocks_apply_before_tokenizer_and_count_exposure(self):
        aug = self.make()
        expected = self.apply(aug)
        for filename, augmentor_name, count_name in (
            ('code/train.py', 'language_augmentor', 'language_condition_counts'),
            ('code/step_training_runner.py', 'augmentor', 'language_counts'),
        ):
            tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                        and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                                and c.func.attr == 'apply' and isinstance(c.func.value, ast.Name)
                                and c.func.value.id == augmentor_name for c in ast.walk(n)))
            # Select the smallest matching block rather than an enclosing train branch.
            matches = [n for n in ast.walk(node) if isinstance(n, ast.If)
                       and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                               and c.func.attr == 'apply' and isinstance(c.func.value, ast.Name)
                               and c.func.value.id == augmentor_name for c in ast.walk(n))]
            node = min(matches, key=lambda n: len(list(ast.walk(n))))
            ns = {augmentor_name: aug, count_name: Counter(), 'instructions': self.texts,
                  'actions': self.targets, 'supervision_masks': self.masks, 'masks': self.masks,
                  'batch': (None,) * 5 + (self.indices,), 'indices': self.indices,
                  'epoch': 1, 'step_count': 11,
                  'plan': type('Plan', (), {'pass_index': 0, 'completed_updates': 11})()}
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), ns)
            self.assertEqual(ns['instructions'], expected[0])
            self.assertTrue(torch.equal(ns['actions'], expected[1]))
            self.assertEqual(ns[count_name], language_record_counts(expected[3]))


if __name__ == '__main__':
    unittest.main()
