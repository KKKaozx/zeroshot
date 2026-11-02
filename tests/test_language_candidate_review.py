"""Candidate preparation and ingestion isolate training IDs and retain failures."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('candidate_prep', ROOT / 'tools/prepare_language_candidate_review_20261030.py')
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)


def queue_fixture():
    rows = []
    for source in ('Bridge', 'Language Table', 'BC-Z', 'Fractal'):
        for i in range(5):
            text = 'synthetic ' + source.casefold() + ' instruction ' + str(i)
            rows.append({'format': 'training-language-review-task-v1',
                'canonical_instruction': text, 'instruction_id': hashlib.sha256(text.encode()).hexdigest(),
                'usable_for_training': False, 'source_window_counts': {source: 1},
                'prior_pilot_text_review': None if i else {'fixture': 'old generation'},
                'representative_training_window': {'source': source, 'window_index': len(rows)}})
    return rows


class CandidateReviewTests(unittest.TestCase):
    def test_saved_outputs_can_receive_protection_checks_without_semantic_approval(self):
        from protected_language_frames import ProtectedLanguageFrames
        run = ROOT / 'reports/local_7b_qualification_20261030_loopback_repair'
        outputs = json.loads((run / 'raw_outputs.json').read_bytes())
        requests_dir = ROOT / 'reports/language_candidate_requests_20261030'
        requests = [json.loads(line) for line in (requests_dir / 'generation_requests.jsonl').read_text(encoding='utf-8').splitlines()]
        bundle = json.loads((requests_dir / 'bundle.json').read_bytes())
        checker = ProtectedLanguageFrames(ROOT / 'configs/protected_training_language_frames_20261030.json',
            bundle['dataset_identity'], {r['instruction_id']: r['original_instruction'] for r in requests})
        tasks = prep.ingest_outputs(bundle, requests, outputs, checker)
        # Historical text review accepted 25; nearby+NP was later corrected.
        self.assertEqual(sum(c['protected_form_matches'] for r in tasks for c in r['protection_checks']), 24)
        self.assertTrue(all(r['usable_for_training'] is False and r['semantic_review_status'] == 'pending' for r in tasks))
        self.assertTrue(all(c['usable_for_training'] is False for r in tasks for c in r['protection_checks']))
        checker.contract['dataset_identity'] = 'different-training-pool'
        with self.assertRaises(ValueError): prep.ingest_outputs(bundle, requests, outputs, checker)

    def test_current_training_selection_skips_existing_pilot_and_rejects_tampering(self):
        queue = queue_fixture()
        requests = prep.select_requests(queue, 'synthetic')
        self.assertEqual(len(requests), 8)
        self.assertTrue(all(r['usable_for_training'] is False for r in requests))
        prior = {r['instruction_id'] for r in queue if r['prior_pilot_text_review']}
        self.assertFalse(prior & {r['instruction_id'] for r in requests})
        self.assertEqual(requests, prep.select_requests(list(reversed(queue)), 'synthetic'))
        prior_ids = {row['instruction_id'] for row in requests}
        subsequent = prep.select_requests(queue, 'synthetic', excluded_ids=prior_ids)
        self.assertFalse(prior_ids & {row['instruction_id'] for row in subsequent})
        with self.assertRaises(ValueError): prep.select_requests(queue, 'synthetic', excluded_ids={'heldout-id'})
        for r in requests:
            self.assertEqual(prep.digest(prep.encode({k: v for k, v in r.items() if k != 'request_sha256'})), r['request_sha256'])
        queue[0]['canonical_instruction'] += ' different'
        with self.assertRaises(ValueError): prep.select_requests(queue, 'synthetic')
        with self.assertRaises(ValueError): prep.select_requests(queue_fixture(), 'synthetic', 3)

    def test_raw_failures_abstentions_and_provenance_are_not_approvals(self):
        requests = prep.select_requests(queue_fixture(), 'synthetic')
        bundle = {'dataset_identity': 'synthetic', 'requests_sha256': 'synthetic-bundle'}
        outputs = {'format': 'local-paraphrase-batch-outputs-v1', 'dataset_identity': 'synthetic',
            'request_bundle_sha256': 'synthetic-bundle',
            'model': {'repository': 'synthetic-only', 'revision': 'test', 'local_only': True},
            'outputs': [{'instruction_id': r['instruction_id'], 'request_sha256': r['request_sha256'],
                'raw_generation': json.dumps({'paraphrases': ['fixture variant ' + str(i) for i in range(5)]})}
                for r in requests]}
        outputs['outputs'][0]['raw_generation'] = '{bad JSON'
        outputs['outputs'][1]['raw_generation'] = '{"abstain_reason":"ambiguous source"}'
        outputs['outputs'][2]['raw_generation'] = '{"paraphrases":["a","a","a","a","a"]}'
        result = prep.ingest_outputs(bundle, requests, outputs)
        self.assertEqual(sum(r['structural_checks_passed'] for r in result), 5)
        self.assertEqual(result[0]['raw_generation'], '{bad JSON')
        self.assertIn('ambiguous source', result[1]['raw_generation'])
        self.assertTrue(all(r['semantic_review_status'] == 'pending' and r['usable_for_training'] is False for r in result))
        for mutation in (
            lambda x: x['outputs'].pop(),
            lambda x: x['outputs'].append(copy.deepcopy(x['outputs'][0])),
            lambda x: x.update(dataset_identity='heldout'),
            lambda x: x['model'].update(local_only=False),
            lambda x: x['outputs'][0].update(request_sha256='different-prompt'),
        ):
            invalid = copy.deepcopy(outputs); mutation(invalid)
            with self.assertRaises(ValueError): prep.ingest_outputs(bundle, requests, invalid)


if __name__ == '__main__': unittest.main()
