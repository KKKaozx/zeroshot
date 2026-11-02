"""Opt-in main-entry fixed-update runner. This is not data/benchmark admission.

Uses train.py's dataset, model, tokenizer, masked loss and validation loss. Only
single-process, workers=0, FP32, one optimizer update per batch is supported.
Legacy epoch experiments and their checkpoint format remain separate.
"""
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from step_training_state import (ResumableBatchPlan, UpdateSchedule, capture_rng,
                                 restore_rng, save_checkpoint, restore_checkpoint)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def frozen_digest(model, saved_keys):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name in saved_keys:
            continue
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tokenizer_identity(tokenizer):
    """Bind both the current Rust backend and older slow CLIP tokenizers."""
    digest_json = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True,
        ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is not None:
        rules = {'backend': json.loads(backend.to_str())}
    else:
        rules = {'bpe_ranks': [[list(k), v] for k, v in sorted(tokenizer.bpe_ranks.items())],
                 'errors': getattr(tokenizer, 'errors', None)}
    return {'class': type(tokenizer).__module__ + '.' + type(tokenizer).__qualname__,
            'vocabulary_sha256': digest_json(tokenizer.get_vocab()),
            'rules_sha256': digest_json(rules), 'special_tokens': tokenizer.special_tokens_map,
            'settings': {k: getattr(tokenizer, k) for k in ('model_max_length',
                'padding_side', 'truncation_side', 'clean_up_tokenization_spaces')}}


def validate_options(args, api):
    if args.total_updates < 1 or args.batch_size < 1 or args.workers != 0:
        raise ValueError('Fixed-update runner requires positive U/B and workers=0')
    if not args.split_manifest or not args.step_diffusion_architecture:
        raise ValueError('Explicit frozen window --split-manifest and --step-diffusion-architecture required')
    if args.decoder_type != 'diffusion' or args.bcz_target != 'reached':
        raise ValueError('Fixed-update main path supports reached-pose diffusion only')
    unsupported = ('resume', 'init_from', 'pool_fit_from', 'gripper_only_fit_from',
                   'overfit_samples', 'overfit_manifest', 'balanced_overfit_targets',
                   'bridge_task_plan', 'bridge_gripper_fit_report', 'prepare_only',
                   'offline_command_experiment', 'native_adapter_comparison',
                   'reset_gripper_readout', 'audit_first_update')
    if any(getattr(args, key, None) for key in unsupported):
        raise ValueError('Do not mix epoch checkpoints/transfer/diagnostic branches with fixed updates')
    defaults = api.parse_args([])
    # Explicit nondefaults here would otherwise look active but have no effect.
    for key in ('epochs', 'max_steps_per_epoch', 'metric_interval', 'metric_batches',
                'metric_samples', 'overfit_trajectories', 'bridge_validation_scope',
                'bridge_window_horizon', 'regression_rotation_weight'):
        if getattr(args, key) != getattr(defaults, key):
            raise ValueError('Epoch/diagnostic option is not used in fixed mode: ' + key)
    if args.gripper_target_mode != 'state' or args.gripper_change_weight != 1:
        raise ValueError('Fixed mode requires absolute state targets without measured-state change weights')
    if args.action_frame != 'tool' and not args.oxe_command_input:
        raise ValueError('Source Cartesian frame requires command/availability input')
    if args.bridge_current_gripper != 'binary' or args.bcz_current_gripper != 'binary':
        raise ValueError('Continuous-measurement diagnostics are separate')
    from language_training_conditions import validate_language_inputs
    validate_language_inputs(args.language_condition,args.language_bank,
        getattr(args,'original_instruction_catalog',None),args.counterfactual_fraction)
    if args.checkpoint_every_updates < 1 or args.validate_every_updates < 1:
        raise ValueError('Positive checkpoint and validation cadence required')
    if args.log_interval < 1 or args.max_validation_batches < 0:
        raise ValueError('Invalid logging/validation limits')
    if not (0 < args.learning_rate < float('inf')) or not (0 <= args.weight_decay < float('inf')):
        raise ValueError('Finite learning rate/weight decay required')
    if args.gripper_learning_rate is not None and not (0 < args.gripper_learning_rate < float('inf')):
        raise ValueError('Finite positive gripper learning rate required')
    end = args.total_updates if args.stop_after_updates is None else args.stop_after_updates
    if not 0 < end <= args.total_updates:
        raise ValueError('Segment stop must be within the frozen total horizon')
    return end


def train_fixed_updates(args, api):
    from oxe_nominal_dataset import validate_options as validate_nominal_options
    validate_nominal_options(args)
    end = validate_options(args, api)
    out = Path(args.output_dir)
    if not args.step_resume and out.exists() and any(out.iterdir()):
        raise ValueError('Fresh fixed-update run requires an empty output directory')
    if (args.step_resume and out.exists() and any(out.iterdir())
            and (out / 'latest_step.pt').resolve() != Path(args.step_resume).resolve()):
        raise ValueError('Resume output must be empty or contain the checkpoint being resumed')
    api.set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_csv = lambda text: tuple(part.strip() for part in text.split(',') if part.strip())
    keys = ('chunk_size', 'stride', 'max_samples', 'max_samples_per_schema',
            'max_tfrecord_episodes', 'max_tfrecord_episodes_per_schema', 'min_trajectory_steps',
            'bcz_target', 'bcz_current_gripper', 'bridge_gripper_policy', 'bridge_current_gripper',
            'rt1_gripper_policy', 'bcz_reached_gripper_policy', 'oxe_command_input',
            'action_frame', 'language_table_groups')
    nominal=bool(getattr(args,'oxe_nominal_targets',None))
    if nominal:
        from oxe_nominal_dataset import from_args
        dataset=from_args(args)
    else:
        dataset = api.UnifiedRobotDataset(data_dir=args.dataset_dir,
            **{key: getattr(args, key) for key in keys},
            **{key: split_csv(getattr(args, key)) for key in
               ('sources', 'exclude_path_parts', 'exclude_schemas', 'tfrecord_splits')})
    if (any(s['source'] == 'tfrecord_rt1_pose' for s in dataset.samples)
            and args.rt1_gripper_policy != 'relative_scan_v2'):
        raise ValueError('New Fractal training requires relative_scan_v2')
    identity = api.dataset_split_identity(dataset)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding='utf-8'))
    if manifest['dataset_identity'] != identity:
        raise ValueError('Frozen window manifest identity differs')
    splits = manifest['split_indices']
    api.validate_split_indices(dataset, splits)
    if nominal and manifest.get('target_binding')!=dataset.nominal_target_binding:
        raise ValueError('Split manifest nominal label/clock binding differs')
    formal_protocol=None
    if getattr(args,'oxe_formal_protocol',None):
        from oxe_formal_protocol import load_and_validate
        formal_protocol=load_and_validate(args,dataset)
    # Identity based on path/mtime alone is insufficient for full-state resume.
    files = sorted({str(Path(s['file_path']).resolve()) for s in dataset.samples})
    # Nominal artifacts bind every expected training payload; the loader verifies
    # the complete used payload before decoding each image. Do not hash17GB of
    # unrelated container bytes on every tiny resume probe or claim such a hash.
    contents = {} if nominal else {path: file_sha(path) for path in files}
    augmentor = None
    if args.language_condition != 'disabled':
        from language_training_conditions import build_language_control
        augmentor = build_language_control(args.language_condition, args.seed,
            identity, splits['train'], {i: dataset.group_key(i) for i in splits['train']},
            bank_path=args.language_bank,
            original_catalog_path=getattr(args,'original_instruction_catalog',None),
            negative_fraction=args.counterfactual_fraction)
    config = api.build_config(args)
    if nominal:
        config['action']['nominal_target_binding']=dataset.nominal_target_binding
        config['action']['source_position_to_meters_verified']=False
    config['model']['diffusion_architecture'] = args.step_diffusion_architecture
    if args.step_diffusion_architecture == 'multiscale':
        config['model'].update(diffusion_down_dims=[256, 512, 1024],
                               diffusion_step_embed_dim=128, diffusion_kernel_size=5)
    if config['action']['representation'] != dataset.action_representation:
        raise ValueError('Model and dataset action representations differ')
    tokenizer = api.CLIPTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir,
        **({'local_files_only': True} if config['model'].get('visual_backbone','clip') == 'dinov2' else {}))
    model = api.RobotAdapterModel(config, cache_dir=args.cache_dir).to(device)
    saved = api.trainable_state_dict(model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any(n not in saved for n, _ in named):
        raise ValueError('Every optimized parameter must be covered by the saved trainable state')
    if any(p.requires_grad for n, p in model.named_parameters() if n not in saved):
        raise ValueError('Backbone must remain frozen')
    prefixes = api.GRIPPER_PARAMETER_PREFIXES
    groups = [[p for n, p in named if not n.startswith(prefixes)],
              [p for n, p in named if n.startswith(prefixes)]]
    rates = [args.learning_rate, args.gripper_learning_rate or args.learning_rate]
    optimizer = torch.optim.AdamW([{'params': g, 'lr': lr} for g, lr in zip(groups, rates)],
                                  weight_decay=args.weight_decay)
    weights = None
    if args.balanced_sampling:
        counts = Counter(dataset.samples[i]['source'] for i in splits['train'])
        weights = [1 / counts[dataset.samples[i]['source']] for i in splits['train']]
    plan = ResumableBatchPlan(splits['train'], batch_size=args.batch_size,
        total_updates=args.total_updates, seed=args.seed, dataset_identity=identity,
        mode='weighted_replacement' if weights is not None else 'shuffle', weights=weights)
    schedule = UpdateSchedule(total_updates=args.total_updates, base_lrs=rates,
                               kind=args.lr_schedule, warmup_updates=args.warmup_updates)
    schedule.apply(optimizer, 0)
    frozen = frozen_digest(model, saved)
    operational = {'output_dir', 'step_resume', 'stop_after_updates', 'checkpoint_every_updates',
                   'log_interval', 'cache_dir', 'dataset_dir', 'split_manifest', 'language_bank',
                   'language_table_groups', 'original_instruction_catalog',
                   'oxe_nominal_targets', 'oxe_nominal_source_map', 'oxe_nominal_validation_targets',
                   'oxe_formal_protocol','oxe_preflight_binding'}
    code_files = ('train.py', 'step_training_runner.py', 'step_training_state.py', 'dataset.py',
                  'models.py', 'adapter.py', 'diffusion_decoder.py', 'cartesian_action_contract.py',
                  'oxe_source_semantics.py', 'language_training_conditions.py')
    if config['model'].get('visual_backbone', 'clip') == 'dinov2':
        code_files += ('frozen_visual_backbone.py',)
    if nominal:code_files+=('oxe_nominal_dataset.py','oxe_nominal_target_store.py','oxe_nominal_timing.py')
    if formal_protocol:code_files+=('oxe_formal_protocol.py','frozen_identity_failure.py')
    contract = {'format': 'main-fixed-update-contract-v1', 'config': config,
        'arguments': {k: v for k, v in vars(args).items() if k not in operational},
        'dataset_identity': identity, 'source_file_sha256': contents,
        'split_indices': splits, 'split_manifest_sha256': file_sha(args.split_manifest),
        'frozen_state_sha256': frozen, 'language': augmentor.contract if augmentor else None,
        'tokenizer': tokenizer_identity(tokenizer),
        'device': str(device), 'optimizer': 'AdamW', 'clip_grad_norm': 1.0,
        'execution': {'intraop_threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_device': ({'index': torch.cuda.current_device(),
                             'name': torch.cuda.get_device_name(),
                             'capability': list(torch.cuda.get_device_capability())}
                            if device.type == 'cuda' else None)},
        'workers': 0, 'amp': False, 'gradient_accumulation': 1,
        'source_code_sha256': {f: file_sha(Path(__file__).parent / f) for f in code_files},
        'packages': {p: importlib.metadata.version(p) for p in ('transformers', 'Pillow')}}
    if nominal:contract['source_record_integrity']=dataset.record_integrity_contract
    if formal_protocol:
        bound=formal_protocol['cluster_preflight']
        print('FORMAL_IDENTITY_CHECK '+json.dumps({
            'actual_frozen_sha256':frozen,'expected_frozen_sha256':bound['frozen_state_sha256'],
            'frozen_matches':frozen==bound['frozen_state_sha256'],
            'tokenizer_matches':contract['tokenizer']==bound['tokenizer']}),flush=True)
        if frozen!=bound['frozen_state_sha256'] or contract['tokenizer']!=bound['tokenizer']:
            try:
                from frozen_identity_failure import write_failure
                write_failure(args,model,saved,frozen,contract,bound)
            except Exception as exc:
                print('FORMAL_IDENTITY_FAILURE_CAPTURE_ERROR '+type(exc).__name__+': '+str(exc),flush=True)
            raise ValueError('Formal frozen encoders/tokenizer differ from the actual passed cluster preflight')
        contract['offline_training_protocol']=formal_protocol
    # HuggingFace configuration is JSON serializable in CLIPTokenizer; reject
    # custom nonserializable inputs before executing any optimizer update.
    json.dumps(contract, sort_keys=True, allow_nan=False)
    history, validation, language_counts = [], [], Counter()

    def load_model(state):
        expected = api.trainable_state_dict(model)
        if (set(state) != set(expected) or any(state[k].shape != expected[k].shape
                or state[k].dtype != expected[k].dtype for k in expected)):
            raise ValueError('Trainable parameter/buffer keys, shapes or dtypes differ')
        model.load_state_dict(state, strict=False)

    if args.step_resume:
        extra = restore_checkpoint(args.step_resume, model_state_loader=load_model,
            optimizer=optimizer, plan=plan, schedule=schedule, contract=contract)
        history, validation = extra['updates'], extra['validation']
        language_counts.update(extra['language_counts'])
        if len(history) != plan.completed_updates:
            raise ValueError('Checkpoint update history/count differs')
    if end < plan.completed_updates:
        raise ValueError('Segment stop precedes restored global update')
    out.mkdir(parents=True, exist_ok=True)
    criterion = nn.MSELoss()
    include_cuda = device.type == 'cuda'
    # Validation does not use the held-out test partition or select a checkpoint.
    loader = DataLoader(Subset(dataset, splits['validation']), batch_size=args.batch_size,
        shuffle=False, num_workers=0, collate_fn=api.collate_batch,
        generator=torch.Generator().manual_seed(args.seed + 10000))

    def checkpoint():
        if frozen_digest(model, saved) != frozen:
            raise RuntimeError('Frozen backbone/buffer identity changed')
        save_checkpoint(out / 'latest_step.pt', model_state=api.trainable_state_dict(model),
            optimizer=optimizer, plan=plan, schedule=schedule, contract=contract,
            include_cuda=include_cuda, extra={'updates': history, 'validation': validation,
                                            'language_counts': dict(language_counts)})

    print('FIXED_UPDATE_MODE: epoch limits inactive; frozen U=', args.total_updates,
          'segment stop=', end, 'restored=', plan.completed_updates)
    training_started=time.monotonic()
    while plan.completed_updates < end:
        model.train()
        indices = plan.begin_batch()
        instructions, images, current, actions, masks = api.collate_batch([dataset[i] for i in indices])
        if augmentor:
            instructions, actions, masks, records = augmentor.apply(instructions, actions, masks,
                indices, epoch=plan.pass_index + 1, step=plan.completed_updates)
            language_counts.update(row['kind'] for row in records)
        text = api.tokenise(tokenizer, instructions, device)
        actions, current, masks = actions.to(device), current.to(device), masks.to(device)
        rates_used = [g['lr'] for g in optimizer.param_groups]
        output = model(images.to(device), text['input_ids'], attention_mask=text.get('attention_mask'),
                       current_gripper=current, actions=actions)
        loss, pose, gripper = api.policy_loss(model, output, actions, criterion,
                                             current_grippers=current, supervision_masks=masks)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite loss; no update committed')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0, error_if_nonfinite=True)
        optimizer.step()
        plan.commit_batch()
        schedule.apply(optimizer, plan.completed_updates)
        history.append({'update': plan.completed_updates, 'pass': plan.pass_index + 1,
            'indices': indices, 'lr': rates_used, 'loss': float(loss.detach()),
            'pose_loss': float(pose.detach()), 'gripper_loss': float(gripper.detach()),
            'gradient_norm': float(norm), 'valid_dimensions': masks.sum(dim=(0, 1)).cpu().tolist()})
        if splits['validation'] and (plan.completed_updates % args.validate_every_updates == 0 or plan.completed_updates == args.total_updates):
            print('FIXED_VALIDATION_START',plan.completed_updates,'windows',len(splits['validation']),
                  'batch_limit',args.max_validation_batches or 'all',flush=True)
            rng = capture_rng(include_cuda=include_cuda)
            try:
                value = api.evaluate_loss(model, loader, tokenizer, device, criterion,
                                          max_batches=args.max_validation_batches or None)
            finally:
                restore_rng(rng)
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite validation loss')
            validation.append({'update': plan.completed_updates, 'loss': value})
            print('FIXED_VALIDATION_END',plan.completed_updates,'loss',value,flush=True)
        if plan.completed_updates % args.log_interval == 0:
            print('FIXED_UPDATE', plan.completed_updates, '/', args.total_updates, 'loss', history[-1]['loss'],
                  'segment_elapsed_seconds',round(time.monotonic()-training_started,1),flush=True)
        if plan.completed_updates % args.checkpoint_every_updates == 0:
            checkpoint()
    checkpoint()
    report = {'format': 'main-fixed-update-segment-report-v1', 'completed_updates': plan.completed_updates,
        'total_updates': args.total_updates, 'complete': plan.completed_updates == args.total_updates,
        'frozen_state_unchanged': True, 'updates': history, 'validation': validation,
        'language_counts': dict(language_counts), 'config': config, 'plan': plan.contract,
        'generalization_verdict': None, 'test_partition_evaluated': False,
        'data_qualification_proven_by_this_runner': False}
    if nominal:report.update(input_preflight_only=not bool(formal_protocol),target_binding=dataset.nominal_target_binding,
        validation_available=bool(splits['validation']),formal_training_admitted=bool(formal_protocol),
        formal_libero_evaluation_completed=False)
    if formal_protocol:report['offline_training_protocol_sha256']=formal_protocol['sha256']
    (out / 'step_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FIXED_UPDATE_SEGMENT_REPORT', out / 'step_report.json')
