"""Training-only language interventions with deterministic, independent draws.

Banks must be reviewed separately. Checking review records cannot establish the
semantic truth of a paraphrase or a scene-goal mismatch automatically.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import random

import torch

MODES = ('disabled', 'no_augmentation', 'paraphrase_only', 'counterfactual_only', 'combined')
BANK_V1 = 'reviewed-training-language-bank-v1'
BANK_V2 = 'reviewed-training-language-bank-v2'


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Nonempty instruction text required')
    return ' '.join(value.split()).casefold()


def _known_hold(hold):
    return (isinstance(hold, dict)
            and type(hold.get('open_positive')) in (int, float)
            and hold['open_positive'] in (-1, 1)
            and isinstance(hold.get('source_evidence'), str)
            and bool(hold['source_evidence'].strip()))


def language_record_counts(records):
    """Keep action-kind counts and explicit v2 exposure denominators together."""
    counts = Counter()
    for row in records:
        counts[row['kind']] += 1
        if 'counterfactual_eligible' in row:
            counts['counterfactual_opportunities'] += 1
            counts['counterfactual_eligible' if row['counterfactual_eligible']
                   else 'counterfactual_ineligible'] += 1
            if row['counterfactual_requested']:
                counts['counterfactual_requested'] += 1
                if not row['counterfactual_eligible']:
                    counts['counterfactual_fallback'] += 1
                    counts['counterfactual_fallback:' + row['counterfactual_ineligible_reason']] += 1
    return counts


class LanguageAugmentor:
    def __init__(self, path, mode, seed, dataset_identity, train_indices, group_keys,
                 negative_fraction=None):
        if mode not in MODES[1:]:
            raise ValueError('Choose an explicit four-condition mode')
        self.path = Path(path)
        payload = self.path.read_bytes()
        bank = json.loads(payload)
        if (bank.get('format') not in (BANK_V1, BANK_V2)
                or bank.get('scope') != 'training_only'
                or bank.get('dataset_identity') != dataset_identity):
            raise ValueError('Language bank format, split scope or dataset identity differs')
        self.eligibility_fallback = bank['format'] == BANK_V2
        self.entries = bank['entries']
        if set(self.entries) != {str(i) for i in train_indices}:
            raise ValueError('Bank must cover exactly training indices; no held-out or missing examples')
        self.mode, self.seed = mode, int(seed)
        self.positive_variants = mode in ('paraphrase_only', 'combined')
        self.negatives = mode in ('counterfactual_only', 'combined')
        if self.negatives:
            if negative_fraction is None or not 0 < float(negative_fraction) < 1:
                raise ValueError('Declare a counterfactual fraction strictly between 0 and 1')
        elif negative_fraction is not None:
            raise ValueError('Counterfactual fraction only applies to counterfactual conditions')
        self.negative_fraction = float(negative_fraction) if self.negatives else None
        for key, entry in self.entries.items():
            if entry.get('group_key') != group_keys[int(key)]:
                raise ValueError('Language bank episode identity differs at index ' + key)
            _text(entry['instruction'])
            if self.positive_variants:
                alternatives = entry.get('paraphrases', [])
                if len(alternatives) != 5 or len({_text(x) for x in [entry['instruction']] + alternatives}) != 6:
                    raise ValueError('Exactly five distinct paraphrases per original are required')
                review = entry.get('paraphrase_review', {})
                if review.get('status') != 'verified' or not review.get('evidence'):
                    raise ValueError('Paraphrases need a semantic review record')
            if self.negatives:
                choices = entry.get('counterfactuals', [])
                if self.eligibility_fallback:
                    eligibility = entry.get('counterfactual_review', {})
                    if (not isinstance(choices, list) or not isinstance(eligibility, dict)
                            or eligibility.get('status') not in ('available', 'unavailable')
                            or not isinstance(eligibility.get('evidence'), str)
                            or not eligibility['evidence'].strip()):
                        raise ValueError('V2 requires explicit scene-review availability and evidence')
                    if ((eligibility['status'] == 'available') != bool(choices)
                            or (not choices and (not isinstance(eligibility.get('reason'), str)
                                                 or not eligibility['reason'].strip()))):
                        raise ValueError('V2 scene-review availability differs from reviewed donors')
                    hold = entry.get('hold_gripper_command')
                    if hold is not None and not _known_hold(hold):
                        raise ValueError('Malformed hold command; use null for an unknown command')
                elif not choices:
                    raise ValueError('Reviewed training-only mismatched goal required')
                for choice in choices:
                    donor = self.entries.get(str(choice['donor_index']))
                    if (donor is None or str(choice['donor_index']) == key
                            or not entry.get('suite_id') or entry['suite_id'] != donor.get('suite_id')
                            or not entry.get('goal_id') or not donor.get('goal_id')
                            or entry['goal_id'] == donor['goal_id']
                            or _text(choice['instruction']) != _text(donor['instruction'])):
                        raise ValueError('Counterfactual donor must be another training goal in the same suite')
                    if self.eligibility_fallback and _text(choice['instruction']) == _text(entry['instruction']):
                        raise ValueError('Counterfactual instruction cannot equal the original')
                    review = choice.get('scene_review', {})
                    if (review.get('status') != 'verified_mismatch' or not review.get('evidence')
                            or review.get('another_valid_goal') is not False):
                        raise ValueError('Different goal alone is insufficient; review actual scene mismatch')
        self.contract = {'format': 'language-condition-training-v1', 'mode': mode,
                         'bank_sha256': hashlib.sha256(payload).hexdigest(), 'seed': self.seed,
                         'dataset_identity': dataset_identity, 'negative_fraction': self.negative_fraction,
                         'positive_variant_sampling': 'Uniform original plus five reviewed paraphrases',
                         'draw_plan': 'Independent hash of stream,seed,epoch,step,row,index; no global RNG consumption',
                         'semantic_review_automatically_proven': False}
        if self.eligibility_fallback:
            self.contract.update(format='language-condition-training-v2', bank_format=BANK_V2,
                counterfactual_eligibility='Reviewed scene mismatch and known hold if gripper is supervised',
                ineligible_negative_draw_rule='Keep positive expert target and original/paraphrase rule',
                masked_target_rule='Preserve unsupervised and tail target elements',
                exposure_counts='Report opportunities, eligible, requested, realized and fallback counts')

    def _rng(self, stream, epoch, step, row, index):
        key = json.dumps([stream, self.seed, int(epoch), int(step), int(row), int(index)]).encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))

    def apply(self, instructions, actions, supervision_masks, indices, *, epoch, step):
        if (actions.ndim != 3 or actions.shape[-1] != 8 or supervision_masks.shape != actions.shape
                or len(instructions) != len(actions) or len(indices) != len(actions)
                or not bool(torch.isfinite(actions).all()) or not bool(torch.isfinite(supervision_masks).all())):
            raise ValueError('Expected aligned finite [B,T,8] targets/masks and training indices')
        texts, target, records = list(instructions), actions.clone(), []
        for row, (instruction, index) in enumerate(zip(instructions, indices)):
            entry = self.entries.get(str(int(index)))
            if entry is None or _text(instruction) != _text(entry['instruction']):
                raise ValueError('Decoded instruction differs from bound training index')
            negative = self.negatives and self._rng('negative', epoch, step, row, index).random() < self.negative_fraction
            record = {'index': int(index), 'kind': 'original', 'variant': 0}
            if self.negatives and self.eligibility_fallback:
                reason = None
                if entry['counterfactual_review']['status'] != 'available':
                    reason = 'no_reviewed_scene_mismatch'
                elif bool((supervision_masks[row, :, 7] > 0).any()) and not _known_hold(entry.get('hold_gripper_command')):
                    reason = 'unknown_current_gripper_command'
                record.update(counterfactual_requested=bool(negative),
                              counterfactual_eligible=reason is None,
                              counterfactual_ineligible_reason=reason)
                negative = negative and reason is None
            if negative:
                choices = entry['counterfactuals']
                choice = choices[self._rng('donor', epoch, step, row, index).randrange(len(choices))]
                texts[row] = choice['instruction']
                if self.eligibility_fallback:
                    null = torch.zeros_like(target[row])
                    null[:, 6] = 1.0
                    # Do not synthesize orientation/gripper or alter a masked tail.
                    target[row] = torch.where(supervision_masks[row] > 0, null, target[row])
                else:
                    target[row].zero_()
                    target[row, :, 6] = 1.0  # identity relative xyzw; never an all-zero quaternion
                if bool((supervision_masks[row, :, 7] > 0).any()):
                    hold = entry.get('hold_gripper_command')
                    if (not isinstance(hold, dict) or type(hold.get('open_positive')) not in (int, float)
                            or hold['open_positive'] not in (-1, 1)
                            or not hold.get('source_evidence')):
                        raise ValueError('Known current command required; do not threshold a measured opening')
                    if self.eligibility_fallback:
                        target[row, :, 7] = torch.where(supervision_masks[row, :, 7] > 0,
                            float(hold['open_positive']), target[row, :, 7])
                    else:
                        target[row, :, 7] = float(hold['open_positive'])
                record.update(kind='counterfactual', donor_index=int(choice['donor_index']), variant=None)
            elif self.positive_variants:
                variant = self._rng('positive', epoch, step, row, index).randrange(6)
                if variant:
                    texts[row] = entry['paraphrases'][variant - 1]
                    record.update(kind='paraphrase', variant=variant)
            records.append(record)
        # Masked-out dimensions remain masked; missing modalities are not invented.
        return texts, target, supervision_masks, records


def validate_resume_contract(saved, current):
    if saved != current:
        raise ValueError('Resume cannot change or silently disable language condition/bank/fraction/seed')


class OriginalInstructionControl:
    """Explicit no-augmentation control bound to original training text only.

    This catalogue is not a reviewed augmentation bank. It grants no paraphrase,
    counterfactual, physical-data or benchmark admission.
    """
    def __init__(self,path,seed,dataset_identity,train_indices,group_keys):
        self.path=Path(path)
        payload=self.path.read_bytes()
        catalog=json.loads(payload)
        if (catalog.get('format')!='oxe-training-instruction-catalog-v1'
                or catalog.get('scope') not in ('frozen_engineering_training_only','training_only')
                or catalog.get('dataset_identity')!=dataset_identity):
            raise ValueError('Original instruction catalogue format, training scope or identity differs')
        if len(train_indices)!=len(set(train_indices)) or any(type(i) is not int or i<0 for i in train_indices):
            raise ValueError('Original control requires unique nonnegative training indices')
        self.entries=catalog['window_entries']
        if set(self.entries)!={str(i) for i in train_indices}:
            raise ValueError('Original catalogue must cover exactly training indices')
        vocabulary={}
        for row in catalog['instructions']:
            text=_text(row['canonical_instruction'])
            identity=hashlib.sha256(text.encode('utf-8')).hexdigest()
            if row['instruction_id']!=identity or identity in vocabulary:
                raise ValueError('Original catalogue instruction identity differs or duplicates')
            vocabulary[identity]=text
        for key,entry in self.entries.items():
            text=_text(entry['instruction'])
            if (entry.get('group_key')!=group_keys[int(key)]
                    or vocabulary.get(entry.get('instruction_id'))!=text):
                raise ValueError('Original instruction or episode group differs at '+key)
        self.mode,self.seed='no_augmentation',int(seed)
        self.contract={'format':'original-instruction-control-v1','mode':self.mode,
            'seed':self.seed,'dataset_identity':dataset_identity,
            'original_catalog_sha256':hashlib.sha256(payload).hexdigest(),
            'catalog_scope':catalog['scope'],'negative_fraction':None,
            'positive_variant_sampling':'Original decoded instruction only',
            'semantic_review_or_physical_admission_granted':False}

    def apply(self,instructions,actions,supervision_masks,indices,*,epoch,step):
        if (actions.ndim!=3 or actions.shape[-1]!=8 or actions.shape!=supervision_masks.shape
                or len(actions)!=len(instructions) or len(actions)!=len(indices)
                or not bool(torch.isfinite(actions).all()) or not bool(torch.isfinite(supervision_masks).all())):
            raise ValueError('Expected aligned finite original instructions and [B,T,8] targets/masks')
        records=[]
        for instruction,index in zip(instructions,indices):
            entry=self.entries.get(str(int(index)))
            if entry is None or _text(instruction)!=_text(entry['instruction']):
                raise ValueError('Decoded original instruction differs from bound training index')
            records.append({'index':int(index),'kind':'original','variant':0})
        return list(instructions),actions,supervision_masks,records


def validate_language_inputs(mode,bank_path=None,original_catalog_path=None,negative_fraction=None):
    if mode not in MODES:raise ValueError('Unknown language condition')
    if mode=='disabled':
        if bank_path or original_catalog_path or negative_fraction is not None:
            raise ValueError('Language inputs require an explicit condition')
    elif original_catalog_path:
        if mode!='no_augmentation' or bank_path or negative_fraction is not None:
            raise ValueError('Original catalogue is only for no_augmentation without a bank or negative fraction')
    elif not bank_path:
        raise ValueError('Augmented conditions require a reviewed bank; no_augmentation may use an original catalogue')


def build_language_control(mode,seed,dataset_identity,train_indices,group_keys,*,
                          bank_path=None,original_catalog_path=None,negative_fraction=None):
    validate_language_inputs(mode,bank_path,original_catalog_path,negative_fraction)
    if mode=='disabled':return None
    if original_catalog_path:
        return OriginalInstructionControl(original_catalog_path,seed,dataset_identity,train_indices,group_keys)
    return LanguageAugmentor(bank_path,mode,seed,dataset_identity,train_indices,group_keys,
        negative_fraction=negative_fraction)
