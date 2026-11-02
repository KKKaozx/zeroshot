"""Conservative, opt-in checks for manually bound training-language frames.

Only the declared controlled forms are recognized. Matching never approves a
training label or proves equivalence for arbitrary English or actual scenes.
"""
import hashlib
import json
from pathlib import Path
import re


def canonical(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Nonempty instruction text required')
    return ' '.join(text.split()).casefold()


def _np(entity):
    aliases = entity['aliases']
    article = r'(?:the |a )?' if entity.get('allow_indefinite') else r'(?:the )?'
    return article + '(?:' + '|'.join(re.escape(a) for a in aliases) + ')'


def _patterns(frame):
    roles, family = frame['roles'], frame['family']
    obj = _np(roles['object'])
    if family == 'relocate_relation':
        verbs = r'(?:move|shift|transfer|reposition|place|put|set|position)'
        reference = _np(roles['reference'])
        relation = frame['relation']
        if relation == 'lower_right_of':
            goal = r'(?:to|at|in) (?:the )?lower right(?: (?:position|side))? (?:of|relative to) ' + reference
        elif relation == 'left_of':
            goal = r'to (?:the )?left(?: side)? (?:of|relative to) ' + reference
        elif relation == 'near':
            goal = r'(?:near|close to|in proximity to) ' + reference
        else:
            raise ValueError('Unsupported protected relation')
        patterns = [verbs + ' ' + obj + ' ' + goal, goal + ', ' + verbs + ' ' + obj]
        if relation == 'near':
            patterns.append(verbs + ' ' + obj + r' so (?:that )?it is ' + goal)
        return patterns
    if family == 'relocate_from_top_of':
        # Preserve the original top-of phrase without deciding a physical surface.
        source, reference = _np(roles['source']), _np(roles['reference'])
        path = 'from ' + source + r' to (?:the )?(?:very )?top of ' + reference
        verbs = r'(?:move|shift|transfer|reposition)'
        return [verbs + ' ' + obj + ' ' + path, path + ', ' + verbs + ' ' + obj]
    if family == 'push_closer_to_both':
        a, b = _np(roles['reference_a']), _np(roles['reference_b'])
        both = '(?:' + a + ' and ' + b + '|' + b + ' and ' + a + ')'
        goal = r'(?:closer|nearer) to ' + both
        return ['push ' + obj + ' ' + goal, 'push ' + obj + r' so (?:that )?it is ' + goal,
                'by pushing ' + obj + ', move it ' + goal,
                'make ' + obj + ' ' + goal + ' by pushing it']
    if family == 'place_inside':
        destination = _np(roles['destination'])
        verbs = r'(?:place|put|insert|move|deposit)'
        goal = r'(?:in|inside|into|within) ' + destination
        return [verbs + ' ' + obj + ' ' + goal, goal + ', ' + verbs + ' ' + obj]
    if family == 'wipe_with':
        tool = _np(roles['tool'])
        return ['wipe ' + obj + r' (?:with|using|by using) ' + tool,
                'use ' + tool + ' to wipe ' + obj,
                'with ' + tool + ', wipe ' + obj,
                obj + ' should be wiped with ' + tool]
    raise ValueError('Unsupported protected frame family')


class ProtectedLanguageFrames:
    def __init__(self, path, dataset_identity, training_instructions):
        payload = Path(path).read_bytes()
        bank = json.loads(payload)
        if (bank.get('format') != 'protected-training-language-frames-v1'
                or bank.get('dataset_identity') != dataset_identity
                or bank.get('automatic_training_approval') is not False):
            raise ValueError('Wrong frame format, training identity or approval policy')
        self.training = {key: canonical(text) for key, text in training_instructions.items()}
        self.frames, self.compiled = {}, {}
        for row in bank['frames']:
            original = canonical(row['original_instruction'])
            key = hashlib.sha256(original.encode()).hexdigest()
            if (row['instruction_id'] != key or self.training.get(key) != original
                    or key in self.frames):
                raise ValueError('Protected frame is not bound to a unique current training instruction')
            if not isinstance(row.get('review_evidence'), str) or not row['review_evidence'].strip():
                raise ValueError('Manual frame review evidence required')
            for entity in row['roles'].values():
                aliases = entity.get('aliases')
                if (not isinstance(aliases, list) or not aliases
                        or len(aliases) != len(set(aliases))
                        or any(canonical(a) != a for a in aliases)
                        or not isinstance(entity.get('canonical_name'), str)
                        or entity['canonical_name'] not in aliases
                        or type(entity.get('allow_indefinite', False)) is not bool
                        or not any(re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', original) for a in aliases)):
                    raise ValueError('Each protected role needs explicit aliases anchored in the original')
            self.compiled[key] = [re.compile(pattern) for pattern in _patterns(row)]
            self.frames[key] = row
        self.contract = {'format': bank['format'], 'dataset_identity': dataset_identity,
            'checker_version': 2,
            'checker_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'profile_sha256': hashlib.sha256(payload).hexdigest(), 'profiled_instructions': len(self.frames),
            'scope': 'Controlled-form recognition for reviewed training frames; not open-language semantic verification',
            'automatic_training_approval': False}

    def assess(self, instruction_id, candidate):
        if instruction_id not in self.training:
            raise ValueError('Instruction does not belong to the bound training catalog')
        if instruction_id not in self.frames:
            return {'status': 'not_profiled', 'protected_form_matches': False,
                    'semantic_review_status': 'pending', 'usable_for_training': False}
        text = canonical(candidate)
        if text.endswith('.'):
            text = text[:-1].rstrip()
        matched = any(pattern.fullmatch(text) for pattern in self.compiled[instruction_id])
        original_reuse = text == self.training[instruction_id].rstrip('.')
        return {'status': 'protected_form_matches' if matched else 'unrecognized_or_changed_form',
            'protected_form_matches': matched, 'original_reuse': original_reuse,
            'needs_text_review': True, 'semantic_review_status': 'pending',
            'usable_for_training': False,
            'frame': {'family': self.frames[instruction_id]['family'],
                      'relation': self.frames[instruction_id].get('relation'),
                      'roles': {key: value['canonical_name'] for key, value in self.frames[instruction_id]['roles'].items()}}
                     if matched else None}
