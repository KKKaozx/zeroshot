"""Read-only Octo cluster inventory; no model, packages, or weights loaded."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    return dict(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new report path')
    plan = json.loads(args.plan.read_text())
    report = dict(stage='inventory_only', model_loaded=False, trained=False,
                  packages_installed=False, weights_downloaded=False,
                  python=sys.version, executable=sys.executable,
                  architecture=platform.machine(), platform=platform.platform(),
                  project=str(args.project.resolve()), plan=plan)
    for name in ['numpy', 'jax', 'jaxlib', 'flax', 'optax', 'orbax-checkpoint',
                 'tensorflow', 'transformers', 'huggingface-hub', 'ml-dtypes']:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        report.setdefault('existing_environment_versions', {})[name] = version
    report['gpu_inventory'] = command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])
    report['project_usage'] = command(['du', '-sb', str(args.project.resolve())])
    used = int(report['project_usage']['stdout'].split()[0]) if report['project_usage']['returncode'] == 0 else None
    report['project_apparent_used_gib'] = None if used is None else used / 2**30
    report['live_quota_verified'] = False
    report['planning_headroom_gib'] = None if used is None else (plan['project_quota_bytes_planning_limit'] - used) / 2**30
    report['inventory_passed'] = (report['architecture'] == 'x86_64'
        and report['gpu_inventory']['returncode'] == 0
        and bool(report['gpu_inventory']['stdout']) and used is not None
        and report['planning_headroom_gib'] >= plan['minimum_project_headroom_gib'])
    report['octo_inference_ready'] = False
    report['next_step'] = 'Review inventory, then prepare a separate pinned Octo environment; no inference executed yet'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['inventory_passed', 'octo_inference_ready', 'project_apparent_used_gib', 'planning_headroom_gib']}, indent=2), flush=True)
    print('Report:', args.output, flush=True)
    if not report['inventory_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
