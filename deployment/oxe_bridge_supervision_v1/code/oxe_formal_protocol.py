"""Explicit bounded-source training plan, separate from input preflight.

Admission is to offline learning under disclosed assumptions. It is not a
claim of exact export timing/units, TCP calibration, full OXE or LIBERO success.
"""
import hashlib
import json
from pathlib import Path

FORMAT='bounded-oxe-dino-training-protocol-v1'
BINDING_FORMAT='passed-oxe-cluster-preflight-binding-v1'


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024**2),b''):h.update(data)
    return h.hexdigest()


def load_and_validate(args,dataset=None):
    path=getattr(args,'oxe_formal_protocol',None)
    if not path:raise ValueError('Explicit offline bounded-source training protocol required')
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if (value.get('format')!=FORMAT or value.get('scope')!='bounded_oxe_offline_training_with_disclosed_assumptions'
            or value.get('formal_libero_performance_already_proven') is not False
            or value.get('full_oxe_training_claim') is not False):
        raise ValueError('Bounded offline research scope and unresolved assumptions must remain explicit')
    if args.prepare_only or getattr(args,'oxe_input_preflight',False):
        raise ValueError('Formal bounded-source training is distinct from input preflight or prepare-only')
    for key,expected in value['fixed_cli_values'].items():
        if getattr(args,key,None)!=expected:raise ValueError('Frozen training protocol option differs: '+key)
    if args.seed not in value['training_seeds'] or args.language_condition not in value['currently_admitted_conditions']:
        raise ValueError('Seed/condition has not been admitted by this release')
    stop=args.total_updates if args.stop_after_updates is None else args.stop_after_updates
    if stop not in value['segment_endpoints']:raise ValueError('Segment endpoint differs from the fixed full learning-rate horizon')
    if args.language_bank or args.counterfactual_fraction is not None:
        raise ValueError('This first no-augmentation release cannot silently introduce a language bank/negative exposure')
    for name,expected in value['input_file_sha256'].items():
        option={'split_manifest':'split_manifest','original_instruction_catalog':'original_instruction_catalog'}[name]
        actual=getattr(args,option,None)
        if not actual or file_sha(actual)!=expected:raise ValueError('Frozen input manifest/catalogue changed: '+name)
    binding_path=getattr(args,'oxe_preflight_binding',None)
    if not binding_path:raise ValueError('Actual passed cluster-preflight metadata binding required')
    binding=json.loads(Path(binding_path).read_text(encoding='utf-8'))
    if (binding.get('format')!=BINDING_FORMAT or binding.get('passed') is not True
            or binding.get('dataset_identity')!=value['dataset_identity']
            or binding.get('training_protocol_sha256')!=file_sha(path)
            or binding.get('copied_learned_preflight_weights') is not False):
        raise ValueError('Need passed matching cluster preflight without learned-weight transfer')
    if dataset is not None:
        if (dataset.nominal_target_binding['dataset_identity']!=value['dataset_identity']
                or len(dataset.frozen_nominal_splits['train'])!=value['training_windows']
                or len(dataset.frozen_nominal_splits['validation'])!=value['validation_windows']
                or dataset.frozen_nominal_splits['test']):
            raise ValueError('Formal input roles/targets differ from the frozen bounded pool')
        stores={'train':dataset.training.store,'validation':dataset.validation.store}
        for role,store in stores.items():
            if file_sha(store.root/'manifest.json')!=value['target_manifest_sha256'][role]:
                raise ValueError('Frozen numeric target store changed: '+role)
    return {'format':FORMAT,'sha256':file_sha(path),'protocol':value,
        'cluster_preflight_binding_sha256':file_sha(binding_path),'cluster_preflight':binding}
