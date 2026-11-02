"""Integrity of finite controlled output languages before any native inference."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from protected_language_frames import ProtectedLanguageFrames
from protected_language_generation import build_grammar, finite_strings, validate_generated_group


class ProtectedGenerationTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'configs/protected_training_language_frames_20261030.json'
        self.profile = json.loads(path.read_bytes())
        self.checker = ProtectedLanguageFrames(path, self.profile['dataset_identity'],
            {row['instruction_id']: row['original_instruction'] for row in self.profile['frames']})

    def test_all_finite_choices_preserve_frame_and_distinct_position_languages(self):
        for frame in self.profile['frames']:
            built = build_grammar(frame, self.checker)
            languages = [finite_strings(pattern) for pattern in built['slot_patterns']]
            self.assertEqual(len(languages), 5)
            for i, language in enumerate(languages):
                self.assertTrue(language)
                self.assertNotIn(frame['original_instruction'], language)
                for earlier in languages[:i]: self.assertFalse(earlier & language)
                for text in language:
                    result = self.checker.assess(frame['instruction_id'], text)
                    self.assertTrue(result['protected_form_matches'])
                    self.assertFalse(result['usable_for_training'])
            candidates = [sorted(language)[0] for language in languages]
            self.assertFalse(validate_generated_group(frame, built, candidates, self.checker)['usable_for_training'])
            self.assertIn('root ::=', built['grammar'])
            self.assertIn('abstain_reason', built['grammar'])

    def test_duplicate_wrong_position_and_semantic_drift_are_refused(self):
        frame = self.profile['frames'][2]
        built = build_grammar(frame, self.checker)
        valid = [sorted(finite_strings(pattern))[0] for pattern in built['slot_patterns']]
        for change in (
            lambda xs: xs.__setitem__(1, xs[0]),
            lambda xs: xs.__setitem__(slice(0, 2), list(reversed(xs[:2]))),
            lambda xs: xs.__setitem__(0, xs[0].replace('push ', 'move ', 1)),
            lambda xs: xs.__setitem__(0, xs[0].replace(' and ', ' or ', 1)),
        ):
            invalid = valid[:]; change(invalid)
            with self.assertRaises(ValueError): validate_generated_group(frame, built, invalid, self.checker)

    def test_unbounded_or_unsupported_regex_is_not_compiled_as_a_fallback(self):
        for pattern in ('a+', '.*', '[a-z]'):
            with self.assertRaises(ValueError): finite_strings(pattern)


if __name__ == '__main__': unittest.main()
