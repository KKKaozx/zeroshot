"""Exercise the actual train(parse_args(...)) dispatch with synthetic dependencies.

No robot files, CLIP downloads, benchmark scoring or real policy optimization.
The real collator, masked diffusion/gripper loss, optimizer, LR and resume run.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import train as entry
from step_training_state import capture_rng

CUDA = sys.argv[1:] == ['--cuda']
if sys.argv[1:] and not CUDA:
    raise ValueError('Only --cuda is supported')
OUT = ROOT / ('reports/step_main_runner_cuda_20261020' if CUDA else 'reports/step_main_runner_20261020')
OUT.mkdir(parents=True, exist_ok=True)
ARTIFACTS = Path(tempfile.mkdtemp(prefix='attempt_', dir=OUT))
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
if CUDA:
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
REAL_AVAILABLE = torch.cuda.is_available
MODELS, READS = [], []
POLLUTE = False
TOKEN_SHIFT = False
FROZEN_SHIFT = False
SOURCE = ARTIFACTS / 'synthetic_source.bin'
SOURCE.write_bytes(b'explicitly synthetic input record fixture v1')
MANIFEST = ARTIFACTS / 'synthetic_window_manifest.json'


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


class SyntheticDataset:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.action_representation = entry.representation_for_frame(self.action_frame)
        self.samples = [{'file_path': str(SOURCE), 'source': 'synthetic_a' if i < 5 else 'synthetic_b',
                         'fixture_index': i} for i in range(12)]
        self.language_table_groups_sha256 = None

    def __len__(self):
        return len(self.samples)

    def group_key(self, i):
        return 'synthetic_episode_' + str(i)

    def __getitem__(self, i):
        READS.append(i)
        assert i < 10, 'Held-out test data must never be decoded'
        # Randomized input preprocessing makes Python/NumPy RNG restore necessary.
        jitter = random.random() * .01 + np.random.random() * .01
        image = torch.tensor([i / 12, .2, -.4, jitter], dtype=torch.float32)
        current = torch.tensor([1., 1.])
        action = torch.zeros(self.chunk_size, 8)
        action[:, :3] = i / 50
        action[:, 6] = 1
        action[:, 7] = 1 if i % 2 else -1
        masks = torch.ones_like(action)
        if i % 3 == 0:
            masks[:, 2:8] = 0  # Synthetic planar/pusher partial supervision.
        return 'move synthetic object ' + str(i), image, current, action, masks


class SyntheticTokenizer:
    special_tokens_map = {'bos_token': '<s>'}
    bpe_ranks = {('s', 'y'): 0}
    model_max_length, padding_side, truncation_side = 77, 'right', 'right'
    clean_up_tokenization_spaces, errors = False, 'replace'

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def get_vocab(self):
        return {'synthetic': int(TOKEN_SHIFT)}

    def __call__(self, instructions, **kwargs):
        values = torch.tensor([[int(x.split()[-1]) + 1] for x in instructions])
        return {'input_ids': values, 'attention_mask': torch.ones_like(values)}


class SyntheticModel(nn.Module):
    def __init__(self, config, cache_dir=None):
        super().__init__()
        if POLLUTE:
            random.random(); np.random.rand(11); torch.randn(13)
            if CUDA:
                torch.randn(9, device='cuda')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(713)
            self.encoder = nn.Linear(4, 4).requires_grad_(False)
        if FROZEN_SHIFT:
            with torch.no_grad():
                self.encoder.weight.add_(.1)
        self.adapter = nn.Sequential(nn.Linear(8, 13), nn.Tanh(), nn.Dropout(.25))
        self.diffusion_decoder = nn.Linear(13, config['model']['chunk_size'] * 7)
        self.gripper_head = nn.Linear(13, config['model']['chunk_size'])
        self.chunk_size = config['model']['chunk_size']
        self.decoder_type = 'diffusion'
        self.gripper_target_mode, self.gripper_change_weight = 'state', 1.
        self.balanced_gripper_loss, self.gripper_loss_weight = False, .25
        self.trace = []
        MODELS.append(self)

    def forward(self, images, input_ids, attention_mask=None, current_gripper=None, actions=None):
        timestep = torch.randint(0, 100, (len(images), 1), device=images.device)
        noise = torch.randn(len(images), self.chunk_size, 7, device=images.device)
        features = torch.cat([self.encoder(images), input_ids.float() / 12,
                              current_gripper, timestep.float() / 100], dim=-1)
        context = self.adapter(features)
        prediction = self.diffusion_decoder(context).reshape_as(noise) + .1 * noise
        if self.training:
            self.trace.append({'t': timestep.detach().cpu(), 'noise': noise.detach().cpu(),
                               'images': images.detach().cpu(), 'ids': input_ids.detach().cpu(),
                               'prediction': prediction.detach().cpu()})
        return prediction, noise, self.gripper_head(context)


def arguments(output, *, mode='shuffle', kind='cosine', seed=42, resume=None, stop=None):
    argv = ['--total-updates', '11', '--step-diffusion-architecture', 'compact',
            '--split-manifest', str(MANIFEST), '--output-dir', str(output), '--chunk-size', '2',
            '--batch-size', '3', '--seed', str(seed), '--gripper-target-mode', 'state',
            '--learning-rate', '.01', '--gripper-learning-rate', '.004', '--weight-decay', '.01',
            '--warmup-updates', '2', '--lr-schedule', kind, '--validate-every-updates', '3',
            '--checkpoint-every-updates', '2', '--log-interval', '100', '--max-validation-batches', '0']
    argv += ['--balanced-sampling' if mode == 'weighted' else '--no-balanced-sampling']
    if resume:
        argv += ['--step-resume', str(resume)]
    if stop is not None:
        argv += ['--stop-after-updates', str(stop)]
    return entry.parse_args(argv)


def execute(args):
    global POLLUTE
    POLLUTE = bool(args.step_resume)
    before = len(MODELS)
    entry.train(args)
    assert len(MODELS) == before + 1
    model = MODELS[-1]
    checkpoint = torch.load(Path(args.output_dir) / 'latest_step.pt', map_location='cpu', weights_only=False)
    assert set(checkpoint['model']) == set(entry.trainable_state_dict(model))
    assert not any(k.startswith('encoder.') for k in checkpoint['model'])
    assert not any(p.requires_grad or p.grad is not None for p in model.encoder.parameters())
    report = json.loads((Path(args.output_dir) / 'step_report.json').read_text(encoding='utf-8'))
    assert report['frozen_state_unchanged'] and not report['test_partition_evaluated']
    return checkpoint, copy.deepcopy(model.trace)


def expect_error(args, label):
    reads_before = len(READS)
    try:
        entry.train(args)
    except (ValueError, FileNotFoundError) as exc:
        assert len(READS) == reads_before, ('Invalid contract decoded data', label)
        return {'case': label, 'error': str(exc)}
    raise AssertionError('Expected rejection: ' + label)


with patch.object(entry, 'UnifiedRobotDataset', SyntheticDataset), \
     patch.object(entry, 'CLIPTokenizer', SyntheticTokenizer), \
     patch.object(entry, 'RobotAdapterModel', SyntheticModel), \
     patch.object(torch.cuda, 'is_available', side_effect=lambda: REAL_AVAILABLE() if CUDA else False):
    ds = SyntheticDataset(**{k: getattr(entry.parse_args([]), k) for k in (
        'stride', 'bcz_target', 'bcz_current_gripper', 'bridge_gripper_policy', 'bridge_current_gripper',
        'rt1_gripper_policy', 'bcz_reached_gripper_policy', 'oxe_command_input', 'action_frame')}, chunk_size=2)
    MANIFEST.write_text(json.dumps({'dataset_identity': entry.dataset_split_identity(ds),
        'split_indices': {'train': list(range(7)), 'validation': [7, 8, 9], 'test': [10, 11]}}), encoding='utf-8')
    rows = []
    for mode in ('shuffle', 'weighted'):
        for kind in ('constant', 'cosine'):
            for seed in (42, 43):
                prefix = f'{mode}_{kind}_{seed}'
                full, full_trace = execute(arguments(ARTIFACTS / (prefix + '_full'), mode=mode, kind=kind, seed=seed))
                for cuts in ([1], [2], [3], [4], [8], [2, 3, 7, 10]):
                    path = ARTIFACTS / (prefix + '_cuts_' + '_'.join(map(str, cuts)))
                    traces = []
                    resume = None
                    for stop in list(cuts) + [11]:
                        result, trace = execute(arguments(path, mode=mode, kind=kind, seed=seed, resume=resume, stop=stop))
                        traces.extend(trace)
                        resume = path / 'latest_step.pt'
                    assert identical(full, result), (prefix, cuts, 'full state')
                    assert identical(full_trace, traces), (prefix, cuts, 'actual forward/noise trace')
                    rows.append({'sampling': mode, 'schedule': kind, 'seed': seed, 'cuts': cuts,
                                 'checkpoint_and_forward_trace_exact': True})
    # Synthetic language review records test plumbing only, not real semantic
    # admission. They must never be used as the missing production language bank.
    bank = ARTIFACTS / 'synthetic_language_bank.json'
    entries = {}
    for i in range(7):
        entries[str(i)] = {'group_key': ds.group_key(i), 'suite_id': 'synthetic_only',
            'goal_id': str(i), 'instruction': 'move synthetic object ' + str(i),
            'paraphrases': [f'synthetic variant {v} object {i}' for v in range(1, 6)],
            'paraphrase_review': {'status': 'verified', 'evidence': 'synthetic fixture, no real review'},
            'hold_gripper_command': {'open_positive': 1, 'source_evidence': 'synthetic known command'},
            'counterfactuals': [{'donor_index': (i + 1) % 7,
                'instruction': 'move synthetic object ' + str((i + 1) % 7),
                'scene_review': {'status': 'verified_mismatch', 'another_valid_goal': False,
                                 'evidence': 'synthetic fixture, no real scene admission'}}]}
    bank.write_text(json.dumps({'format': 'reviewed-training-language-bank-v1', 'scope': 'training_only',
        'dataset_identity': entry.dataset_split_identity(ds), 'entries': entries}), encoding='utf-8')
    language_rows = []
    for condition in ('no_augmentation', 'paraphrase_only', 'counterfactual_only', 'combined'):
        def language_args(path, resume=None, stop=None):
            result = arguments(path, resume=resume, stop=stop)
            result.language_condition, result.language_bank = condition, str(bank)
            if condition in ('counterfactual_only', 'combined'):
                result.counterfactual_fraction = .3
            return result
        full, trace = execute(language_args(ARTIFACTS / (condition + '_full')))
        path = ARTIFACTS / (condition + '_resumed')
        resume, combined_trace = None, []
        for cut in (2, 3, 7, 11):
            result, segment_trace = execute(language_args(path, resume, cut))
            combined_trace.extend(segment_trace)
            resume = path / 'latest_step.pt'
        assert identical(full, result) and identical(trace, combined_trace), condition
        language_rows.append({'condition': condition, 'exact': True,
                              'counts': result['extra']['language_counts']})
    errors = []
    args = arguments(ARTIFACTS / 'guard', resume=ARTIFACTS / 'shuffle_cosine_42_full/latest_step.pt')
    for attr, value in [('total_updates', 12), ('batch_size', 2), ('warmup_updates', 1),
                        ('learning_rate', .02), ('language_condition', 'no_augmentation'),
                        ('workers', 1), ('resume', 'old-format.pt'), ('epochs', 31),
                        ('step_diffusion_architecture', 'multiscale')]:
        changed = copy.deepcopy(args)
        setattr(changed, attr, value)
        errors.append(expect_error(changed, attr))
    TOKEN_SHIFT = True
    errors.append(expect_error(args, 'tokenizer_changed'))
    TOKEN_SHIFT = False
    FROZEN_SHIFT = True
    errors.append(expect_error(args, 'frozen_encoder_changed'))
    FROZEN_SHIFT = False
    # Check content binding when path, size and mtime deliberately remain equal.
    old = SOURCE.read_bytes(); stat = SOURCE.stat()
    SOURCE.write_bytes(b'X' + old[1:]); os.utime(SOURCE, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    errors.append(expect_error(args, 'source_bytes_changed_same_metadata'))
    SOURCE.write_bytes(old); os.utime(SOURCE, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    original_manifest = MANIFEST.read_bytes()
    changed_manifest = json.loads(original_manifest)
    changed_manifest['split_indices']['train'][0], changed_manifest['split_indices']['validation'][0] = 7, 0
    MANIFEST.write_text(json.dumps(changed_manifest), encoding='utf-8')
    errors.append(expect_error(args, 'window_roles_changed'))
    MANIFEST.write_bytes(original_manifest)
    for options in (['--step-resume', 'x'], ['--warmup-updates', '2']):
        errors.append(expect_error(entry.parse_args(options), 'step_option_without_mode'))

# Compare legacy function bodies, stripping only the new dispatch/option guards.
before_path = ROOT / 'reports/step_main_runner_20261020/before_train.py'
old_tree, new_tree = [ast.parse(p.read_text(encoding='utf-8-sig')) for p in (before_path, ROOT/'code/train.py')]
old_functions = {n.name: n for n in old_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
new_functions = {n.name: n for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
for name, old_fn in old_functions.items():
    new_fn = copy.deepcopy(new_functions[name])
    if name == 'train':
        new_fn.body = new_fn.body[3:]
    if name == 'parse_args':
        added_flags = {'--total-updates', '--step-resume', '--stop-after-updates',
                       '--checkpoint-every-updates', '--validate-every-updates',
                       '--warmup-updates', '--step-diffusion-architecture'}
        new_fn.body = [n for n in new_fn.body if not (isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == 'add_argument' and n.value.args
            and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value in added_flags)]
    assert ast.dump(old_fn) == ast.dump(new_fn), ('Legacy function changed', name)
assert entry.parse_args([]).total_updates is None
from step_training_runner import tokenizer_identity
real_tokenizer = entry.CLIPTokenizer.from_pretrained(
    str(ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'),
    local_files_only=True)
real_tokenizer_contract = tokenizer_identity(real_tokenizer)
json.dumps(real_tokenizer_contract, sort_keys=True, allow_nan=False)
sources = ['code/train.py', 'code/step_training_runner.py', 'code/step_training_state.py',
           'scripts/check_step_main_runner_20261020.py']
hashes = {}
for relative in sources:
    source = ROOT / relative
    hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copyfile(source, OUT / ('snapshot_' + source.name))
report = {'passed': True, 'scope': 'synthetic dependencies through real train(parse_args) entry',
    'device': 'cuda' if CUDA else 'cpu', 'resumed_paths': rows, 'negative_controls': errors,
    'synthetic_language_resume_paths': language_rows,
    'legacy_functions_and_parser_ast_unchanged_except_explicit_dispatch_flags': True,
    'actual_cached_clip_tokenizer_contract': real_tokenizer_contract,
    'synthetic_artifacts': str(ARTIFACTS),
    'partial_masks_real_policy_loss_used': True, 'heldout_test_indices_read': False,
    'real_clip_or_robot_training': False, 'generalization_verdict': None,
    'sha256': hashes, 'cuda_peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3 if CUDA else None}
(OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('MAIN ENTRY QUALIFIED', len(rows), 'resume paths;', len(errors), 'negative controls;', report['device'])
