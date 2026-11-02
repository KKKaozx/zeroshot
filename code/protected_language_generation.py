"""Finite reviewed surface forms -> five disjoint slots -> native GBNF.

This deliberately restricts language diversity. It is not an arbitrary-English
paraphraser, and successful generation never creates a training bank.
"""
import itertools
import json
import re
import sre_parse


def _np(entity):
    article = r'(?:the |a )' if entity.get('allow_indefinite') else 'the '
    return article + '(?:' + '|'.join(re.escape(a) for a in entity['aliases']) + ')'


def style_patterns(frame):
    roles, family = frame['roles'], frame['family']
    obj = _np(roles['object'])
    if family == 'relocate_relation':
        ref = _np(roles['reference'])
        if frame['relation'] == 'lower_right_of':
            # With movement verbs, "in/at" can mean motion within a region.
            # Keep the destination explicitly directional before any generation.
            goal = r'to the lower right(?: position| side)? (?:of|relative to) ' + ref
        elif frame['relation'] == 'left_of':
            goal = r'to the left(?: side)? (?:of|relative to) ' + ref
        elif frame['relation'] == 'near':
            goal = r'(?:near|close to|in proximity to) ' + ref
        else:
            raise ValueError('Unsupported relation')
        return [verb + ' ' + obj + ' ' + goal for verb in ('move', 'shift', 'reposition', 'position')] + [goal + ', place ' + obj]
    if family == 'relocate_from_top_of':
        path = 'from ' + _np(roles['source']) + r' to the (?:very )?top of ' + _np(roles['reference'])
        return [verb + ' ' + obj + ' ' + path for verb in ('shift', 'transfer', 'reposition')] + [
            path + ', move ' + obj, path + ', shift ' + obj]
    if family == 'push_closer_to_both':
        a, b = _np(roles['reference_a']), _np(roles['reference_b'])
        both = '(?:' + a + ' and ' + b + '|' + b + ' and ' + a + ')'
        return ['push ' + obj + ' nearer to ' + both,
                'push ' + obj + ' so it is closer to ' + both,
                'push ' + obj + ' so that it is nearer to ' + both,
                'by pushing ' + obj + ', move it closer to ' + both,
                'make ' + obj + ' nearer to ' + both + ' by pushing it']
    if family == 'place_inside':
        goal = r'(?:in|inside|into|within) ' + _np(roles['destination'])
        return [verb + ' ' + obj + ' ' + goal for verb in ('put', 'insert', 'move', 'deposit')] + [goal + ', place ' + obj]
    if family == 'wipe_with':
        tool = _np(roles['tool'])
        return ['wipe ' + obj + ' with ' + tool, 'use ' + tool + ' to wipe ' + obj,
                'with ' + tool + ', wipe ' + obj, 'wipe ' + obj + ' by using ' + tool,
                obj + ' should be wiped with ' + tool]
    raise ValueError('Unsupported family')


def finite_strings(pattern, limit=10000):
    """Enumerate only the bounded regex subset for offline integrity checks."""
    def expand(sequence):
        result = ['']
        for op, value in sequence:
            if op is sre_parse.LITERAL:
                pieces = [chr(value)]
            elif op is sre_parse.SUBPATTERN:
                pieces = expand(value[-1])
            elif op is sre_parse.BRANCH:
                pieces = [s for branch in value[1] for s in expand(branch)]
            elif op is sre_parse.MAX_REPEAT and value[0:2] == (0, 1):
                pieces = [''] + expand(value[2])
            else:
                raise ValueError('Unbounded or unsupported protected regex token: ' + str(op))
            if len(result) * len(pieces) > limit:
                raise ValueError('Protected finite language exceeds its size budget')
            result = [a + b for a, b in itertools.product(result, pieces)]
        return result
    return set(expand(sre_parse.parse(pattern)))


def build_grammar(frame, checker):
    patterns = style_patterns(frame)
    languages = [finite_strings(pattern) for pattern in patterns]
    original = frame['original_instruction']
    for i, language in enumerate(languages):
        if not language or original in language:
            raise ValueError('Empty style language or original-instruction reuse')
        if any(language & earlier for earlier in languages[:i]):
            raise ValueError('Output positions are not disjoint')
        for candidate in language:
            result = checker.assess(frame['instruction_id'], candidate)
            if not result['protected_form_matches'] or result['original_reuse']:
                raise ValueError('Generation language escapes the checked source frame')
    rules, counter = [], 0
    def expression(sequence):
        nonlocal counter
        parts, literal = [], []
        def flush():
            if literal:
                text = ''.join(literal)
                if any(c in text for c in ('"', '\\', '\n', '\r')):
                    raise ValueError('JSON escaping for this source alias is not supported')
                parts.append(json.dumps(text)); literal.clear()
        for op, value in sequence:
            if op is sre_parse.LITERAL:
                literal.append(chr(value)); continue
            flush()
            name = 'r-' + str(counter); counter += 1
            if op is sre_parse.SUBPATTERN:
                rhs = expression(value[-1])
            elif op is sre_parse.BRANCH:
                rhs = ' | '.join(expression(branch) for branch in value[1])
            elif op is sre_parse.MAX_REPEAT and value[0:2] == (0, 1):
                rhs = '"" | ' + expression(value[2])
            else:
                raise ValueError('Unsupported native-grammar token')
            rules.append(name + ' ::= ' + rhs); parts.append(name)
        flush()
        return ' '.join(parts) or '""'
    for i, pattern in enumerate(patterns):
        rules.append('slot-' + str(i) + ' ::= ' + expression(sre_parse.parse(pattern)))
    quote = json.dumps('"')
    entries = [quote + ' slot-' + str(i) + ' ' + quote for i in range(5)]
    body = json.dumps('{"paraphrases":[') + ' ' + (' ' + json.dumps(',') + ' ').join(entries) + ' ' + json.dumps(']}')
    abstain = json.dumps('{"abstain_reason":"cannot_preserve_reviewed_frame"}')
    grammar = 'root ::= ' + body + ' | ' + abstain + '\n' + '\n'.join(rules) + '\n'
    return {'grammar': grammar, 'slot_patterns': patterns, 'slot_language_sizes': [len(s) for s in languages],
            'unique_slots_guaranteed_by_disjoint_forms': True,
            'original_excluded_from_all_slots': True, 'automatic_training_approval': False}


def validate_generated_group(frame, bundle, candidates, checker):
    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError('Need five distinct generated strings')
    for index, candidate in enumerate(candidates):
        if not re.fullmatch(bundle['slot_patterns'][index], candidate):
            raise ValueError('Native output violates its declared slot language')
        result = checker.assess(frame['instruction_id'], candidate)
        if not result['protected_form_matches'] or result['original_reuse']:
            raise ValueError('Native output violates the protected source frame')
    return {'five_distinct': True, 'slot_constraints_match': True, 'source_frame_matches': True,
            'semantic_review_status': 'pending', 'usable_for_training': False}
