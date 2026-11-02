"""Match local HDF5 metadata to pinned official task/assets, no array payloads."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import h5py

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'training_cache/libero_author_8f1084e'
DATA = Path('D:/ntu_related/dissertation/dataset/LIBERO')
OUT = ROOT / 'reports/libero_official_assets_20261020'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mapping_path = AUTHOR / 'libero/libero/benchmark/libero_suite_task_map.py'
    tree = ast.parse(mapping_path.read_text())
    mapping = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                   and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets)))
    records, errors = [], []
    for suite, tasks in mapping.items():
        for task in tasks:
            path = DATA / suite / (task + '_demo.hdf5')
            bddl = AUTHOR / 'libero/libero/bddl_files' / suite / (task + '.bddl')
            init = AUTHOR / 'libero/libero/init_files' / suite / (task + '.pruned_init')
            row = {'suite': suite, 'task': task, 'path': str(path), 'exists': path.is_file(),
                   'official_bddl_exists': bddl.is_file(), 'official_init_states_exist': init.is_file()}
            if not (path.is_file() and bddl.is_file() and init.is_file()):
                errors.append('Missing data or official asset: ' + suite + '/' + task)
                records.append(row)
                continue
            with h5py.File(path, 'r') as file:
                group = file['data']
                env = json.loads(group.attrs['env_args'])
                demos = sorted(k for k in group if k.startswith('demo_'))
                first = group[demos[0]]
                lengths = [int(group[k]['actions'].shape[0]) for k in demos]
                row.update(bytes=path.stat().st_size, demos=len(demos), total_action_steps=sum(lengths),
                           min_demo_length=min(lengths), max_demo_length=max(lengths),
                           first_demo_shapes={k: list(first[k].shape) for k in first if isinstance(first[k], h5py.Dataset)},
                           first_obs_shapes={k: list(v.shape) for k, v in first['obs'].items()},
                           image_convention=str(group.attrs.get('macros_image_convention', 'unspecified')),
                           controller=env['env_kwargs'].get('controller_configs'),
                           control_frequency=env['env_kwargs'].get('control_freq'),
                           robots=env['env_kwargs'].get('robots'),
                           bddl_sha256=sha(bddl), init_file_bytes=init.stat().st_size)
                if any(group[k]['actions'].ndim != 2 or group[k]['actions'].shape[1] != 7 for k in demos):
                    errors.append('Native action dimensions changed: ' + suite + '/' + task)
            records.append(row)
    expected_paths = {r['path'] for r in records}
    unexpected = [str(p) for p in DATA.rglob('*.hdf5') if str(p) not in expected_paths]
    summary = {'complete': not errors and not unexpected, 'author_commit': subprocess.check_output(
        ['git', '-C', str(AUTHOR), 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'Full task-name/official asset matching and HDF5 shapes/attributes only; no numeric states/actions/images or final initial states decoded',
        'suite_task_counts': dict(Counter(r['suite'] for r in records)), 'tasks': len(records),
        'demos': sum(r.get('demos', 0) for r in records),
        'bytes': sum(r.get('bytes', 0) for r in records), 'errors': errors, 'unexpected_files': unexpected,
        'controller_types': dict(Counter(r.get('controller', {}).get('type') for r in records)),
        'control_frequency_counts': dict(Counter(r.get('control_frequency') for r in records)),
        'image_conventions': dict(Counter(r.get('image_convention') for r in records)),
        'metadata_only': True, 'model_loaded': False, 'training_updates': 0, 'task_scoring': False}
    (OUT / 'task_metadata.json').write_text(json.dumps(records, indent=2) + '\n')
    (OUT / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not summary['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
