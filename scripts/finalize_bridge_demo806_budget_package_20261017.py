"""Validate launcher arguments and finalize the small code-only transfer package."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_demo806_budget_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((PACK/'protocol.json').read_text())
    expected = {'--data-root', '--data-reference', '--clip', '--old-cache', '--cache',
                '--control-report', '--preflight-reference', '--stage', '--output'}
    for name in ('preflight.sh', 'fit.sh'):
        text = (PACK/name).read_text()
        assert '\r' not in text and text.startswith('#!/bin/bash\n')
        joined = text.replace('\\\n', '')
        command = next(line for line in joined.splitlines() if 'python -u run_budget.py' in line)
        tokens = shlex.split(command)
        args = tokens[tokens.index('run_budget.py')+1:]
        assert len(args) == 2*len(expected) and set(args[::2]) == expected
        assert '+' not in tokens and dict(zip(args[::2], args[1::2]))['--stage'] == name[:-3].replace('preflight', 'preflight')
    for path in PACK.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    spec = importlib.util.spec_from_file_location('frozen_budget_factory', PACK/'run_budget.py')
    factory = importlib.util.module_from_spec(spec); spec.loader.exec_module(factory)
    source = (ROOT/'reports/bridge-demo806-fit-208207/effective_runner_snapshot.py').read_text()
    effective = factory.adapted_source(source)
    assert hashlib.sha256(effective.encode()).hexdigest() == protocol['budget_effective_runner_sha256']
    # Every changeset is a declared hook or finite endpoint/metadata change.
    assert "advance('extension', 8193, 16170)" in effective
    assert 'update=end, budget_protocol=BUDGET.p' in effective
    assert 'Existing complete806 token cache required; no re-encoding' in effective
    assert effective.index('BUDGET.boundary(locals())') < effective.index("advance('extension', 8193, 16170)")
    probe = json.loads((ROOT/'reports/bridge_budget_optimizer_cpu_20261017/validation.json').read_text())
    assert probe['passed'] and probe['full_adam_reload_continuation_exact'] and probe['weights_only_restart_differs']
    local = json.loads((PACK/'local_validation.json').read_text())
    local.update(launcher_arguments_exact=True, no_stray_plus_argument=True, all_python_sources_compile=True,
                 bash_syntax_checked=True, bash_syntax_evidence='Two separate Git Bash -n calls returned exit0; scripts not executed.',
                 cpu_micro_optimizer_roundtrip_verified=True, cpu_probe_project_model_executed=False,
                 cpu_probe_path='reports/bridge_budget_optimizer_cpu_20261017/validation.json',
                 final_endpoint=16170, actual_project_model_executed=False, actual_gpu_executed=False)
    (PACK/'local_validation.json').write_text(json.dumps(local, indent=2)+'\n', encoding='utf-8', newline='\n')
    members = sorted(p for p in PACK.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    assert not any(p.suffix in ('.pt', '.tfrecord', '.zip') for p in members)
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in members), encoding='utf-8', newline='\n')
    members.append(PACK/'SHA256SUMS')
    archive = ROOT/'cluster_packages/bridge_demo806_budget_v1.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in members:
            z.write(p, 'bridge_demo806_budget_v1/'+p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(members)
        for p in members:
            assert hashlib.sha256(z.read('bridge_demo806_budget_v1/'+p.name)).hexdigest() == sha(p)
    checksum = ROOT/'cluster_packages/bridge_demo806_budget_v1.sha256'
    checksum.write_text(f'{sha(archive)}  {archive.name}\n', encoding='utf-8', newline='\n')
    record = dict(date='2026-10-17', status='code_only_package_prepared_not_submitted',
                  archive=str(archive.relative_to(ROOT)), bytes=archive.stat().st_size, sha256=sha(archive),
                  protocol_sha256=sha(PACK/'protocol.json'), files={p.name: sha(p) for p in members},
                  local_validation=local, cluster_preflight_passed=False, formal_job_submitted=False,
                  data_upload_required=False, new_clip_cache_required=False,
                  purpose='One extra-low-LR-budget experiment on unchanged806 pool; original8192 comparison remains closed.')
    out = ROOT/'reports/bridge_demo806_budget_preparation_20261017.json'
    out.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(prepared=True, archive=str(archive), bytes=archive.stat().st_size,
                         sha256=sha(archive), members=len(members), new_data_or_weights=False,
                         cluster_preflight_passed=False), indent=2))


if __name__ == '__main__':
    main()
