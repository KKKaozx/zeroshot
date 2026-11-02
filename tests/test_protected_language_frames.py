"""Semantic counterexamples exercise roles, action constraints and unknown forms."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from protected_language_frames import ProtectedLanguageFrames


class ProtectedFrameTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / 'configs/protected_training_language_frames_20261030.json'
        self.profile = json.loads(self.path.read_bytes())
        self.ids = [row['instruction_id'] for row in self.profile['frames']]
        self.training = {row['instruction_id']: row['original_instruction'] for row in self.profile['frames']}
        self.extra = hashlib.sha256(b'move the orange cup').hexdigest()
        self.training[self.extra] = 'move the orange cup'
        self.checker = ProtectedLanguageFrames(self.path, self.profile['dataset_identity'], self.training)

    def matched(self, index, text):
        result = self.checker.assess(self.ids[index], text)
        self.assertIs(result['usable_for_training'], False)
        self.assertEqual(result['semantic_review_status'], 'pending')
        return result['protected_form_matches']

    def test_role_swaps_cannot_pass_by_retaining_the_same_words(self):
        cases = [
            (0, 'move the burner to the lower right of the silver pot'),
            (1, 'shift the yellow object from the drawer to the top of the table'),
            (2, 'push the yellow hexagon closer to the green circle and blue triangle'),
            (3, 'reposition the blue cube to the left of the blue triangle'),
            (4, 'place the purple bowl inside the grapes'),
            (5, 'wipe the towel with the purple bowl'),
            (6, 'move the red bull can near the apple'),
            (7, 'put the top drawer into the rxbar blueberry'),
        ]
        for index, text in cases:
            with self.subTest(text=text): self.assertFalse(self.matched(index, text))

    def test_negation_extra_actions_and_relation_drift_are_not_recognized(self):
        cases = [
            (0, 'do not place the silver pot to the lower right of the burner'),
            (0, 'place the silver pot to the lower right of the burner then turn it over'),
            (0, 'place the silver pot to the upper right of the burner'),
            (1, 'carry the yellow object from the table to the top of the drawer'),
            (1, 'move the yellow object from the table to the upper part of the drawer'),
            (2, 'move the green circle closer to the yellow hexagon and blue triangle'),
            (2, 'push the green circle closer to the yellow hexagon or blue triangle'),
            (2, 'push the green circle closer to the yellow hexagon'),
            (2, 'push the green circle away from the yellow hexagon and blue triangle'),
            (2, 'gently nudge the green circle closer to the yellow hexagon and blue triangle'),
            (4, 'transfer the grapes to the purple bowl'),
            (4, 'put the grapes on the purple bowl'),
            (4, 'put three grapes inside the purple bowl'),
            (5, 'take a towel and wipe the purple bowl'),
            (5, 'use a towel to clean the purple bowl'),
            (6, 'slide the apple next to the red bull can'),
            (6, 'put the apple near the red can'),
            (6, 'shift the apple nearby the red bull can'),
            (7, 'place the rxbar blueberry into the upper drawer'),
            (7, 'put the rxbar blueberry inside the top shelf'),
        ]
        for index, text in cases:
            with self.subTest(text=text): self.assertFalse(self.matched(index, text))

    def test_compositional_word_order_and_declared_aliases_still_match(self):
        cases = [
            (0, 'To the lower right of the burner, place the silver pot.'),
            (1, 'From the table to the top of the drawer, reposition the yellow object.'),
            (2, 'Push the green circle nearer to the blue triangle and the yellow hexagon.'),
            (3, 'To the left of the blue cube, shift the blue triangle.'),
            (4, 'Inside the purple bowl, put the grapes.'),
            (5, 'Use the towel to wipe the purple bowl.'),
            (6, 'Near the Red Bull can, place the apple.'),
            (7, 'Within the highest drawer, deposit the blueberry rxbar.'),
        ]
        for index, text in cases:
            with self.subTest(text=text): self.assertTrue(self.matched(index, text))

    def test_unknown_forms_and_unprofiled_training_text_remain_pending(self):
        valid_but_unrecognized = self.checker.assess(self.ids[0],
            'Arrange for the silver pot to end up at the lower right of the burner.')
        self.assertEqual(valid_but_unrecognized['status'], 'unrecognized_or_changed_form')
        self.assertEqual(valid_but_unrecognized['semantic_review_status'], 'pending')
        self.assertIsNone(valid_but_unrecognized['frame'])
        unknown = self.checker.assess(self.extra, 'Shift the orange cup.')
        self.assertEqual(unknown['status'], 'not_profiled')
        self.assertFalse(unknown['usable_for_training'])
        with self.assertRaises(ValueError): self.checker.assess('not-a-training-id', 'move the cup')
        original = self.checker.assess(self.ids[0], self.training[self.ids[0]])
        self.assertTrue(original['original_reuse'])
        self.assertFalse(original['usable_for_training'])

    def test_wrong_dataset_original_or_alias_binding_is_rejected(self):
        with self.assertRaises(ValueError):
            ProtectedLanguageFrames(self.path, 'different-data', self.training)
        for change in (
            lambda p: p['frames'][0].update(original_instruction='move the gold pot'),
            lambda p: p['frames'][0]['roles']['object'].update(aliases=['gold pot']),
            lambda p: p.update(automatic_training_approval=True),
        ):
            p = copy.deepcopy(self.profile); change(p)
            with tempfile.TemporaryDirectory(dir=ROOT / 'reports') as temp:
                self.assertEqual(Path(temp).resolve().parent, (ROOT / 'reports').resolve())
                path = Path(temp) / 'wrong_profile.json'
                path.write_text(json.dumps(p), encoding='utf-8')
                with self.assertRaises(ValueError):
                    ProtectedLanguageFrames(path, self.profile['dataset_identity'], self.training)


if __name__ == '__main__': unittest.main()
