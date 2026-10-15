"""Prepare seed42 replication of the frozen seed43 64-window fitting recipe."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'cluster_packages/bridge_growth64_v1'
PACK = ROOT/'cluster_packages/bridge_growth64_replication_v1'
LOCAL = ROOT/'reports/bridge_growth64_replication_local_preflight_20261013'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    assert not PACK.exists(), 'Preserve existing frozen package'
    manifest = json.loads((BASE/'manifest.json').read_text())
    source = (BASE/'fit_bridge_growth64.py').read_text()
    assert source.count('init = 43; seed(init)') == 1
    source = source.replace('init = 43; seed(init)', 'init = 42; seed(init)')
    # Reversing the one initialization edit must reproduce the original runner.
    assert source.replace('init = 42; seed(init)', 'init = 43; seed(init)') == (BASE/'fit_bridge_growth64.py').read_text()
    ast.parse(source)
    PACK.mkdir()
    for name, expected in manifest['source_sha256'].items():
        assert sha(BASE/name) == expected
        shutil.copyfile(BASE/name, PACK/name)
    shutil.copytree(BASE/'added_reference', PACK/'added_reference')
    (PACK/'fit_bridge_growth64.py').write_text(source, newline='\n')
    manifest['source_sha256']['fit_bridge_growth64.py'] = sha(PACK/'fit_bridge_growth64.py')
    manifest.update(date='2026-10-13', stage='preregistered_growth64_replication', initialization_seeds=[42],
                    changing_variable='Only initialization43 ->42; keep identical64 windows, update order, budget and LR schedule',
                    revision='Seed42 replication of completed and independently verified200600')
    manifest['late_lr']['initialization_seed'] = 42
    manifest['limits'][0] = 'One predetermined replication initialization42; compare to43 without selecting favorable seeds'
    manifest['replication'] = dict(baseline_manifest_sha256=sha(BASE/'manifest.json'),
        baseline_report_sha256=sha(ROOT/'reports/bridge-growth64-fit-200600/report.json'),
        baseline_analysis_sha256=sha(ROOT/'reports/bridge_growth64_analysis_200600_20261013/analysis.json'),
        expected_change='Initialization only; no development-selected weights',
        success='Report all64 and retained/added groups at all four late monitor points; never suppress failed fits')
    (PACK/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', newline='\n')
    for key in ('train', 'development', 'config', 'schedule_sha256', 'batch_size', 'updates_per_arm',
                'draw_slots_per_window', 'sampling_seeds', 'learning_rate_schedule', 'growth'):
        assert manifest[key] == json.loads((BASE/'manifest.json').read_text())[key]
    for stage, limit in [('preflight', '00:25:00'), ('fit', '01:30:00')]:
        script = (BASE/'preflight.sh').read_text() if stage=='preflight' else (ROOT/'cluster_packages/bridge_growth64_launcher_fix_20261013/fit_fixed.sh').read_text()
        script = script.replace('bridge-growth64-', 'bridge-growth64-rep-')
        script = script.replace('/projects/Zeroshot/bridge_growth64_v1', '/projects/Zeroshot/bridge_growth64_replication_v1')
        assert '\n+' not in script and f'#SBATCH --time={limit}' in script
        (PACK/f'{stage}.sh').write_text(script, newline='\n')
    (PACK/'README.md').write_text('Seed42 replication of200600. Same64 training windows,40 development windows,4096 updates, '
        'LR1e-4 ->1e-5 at2048; fullDDPM100, frozenCLIP/gripper. Three sampling draws are not three training runs. '
        'Original production source files unchanged; runner initialization is the only training edit. '
        'Local short preflight is execution/restoration verification, not fitting success.\n')
    print('Prepared:', PACK)

def package():
    report = json.loads((LOCAL/'report.json').read_text())
    verification = json.loads((LOCAL/'verification.json').read_text())
    assert report['stage']=='preflight' and report['initialization_seed']==42
    assert all(report[k] for k in ('execution_checks_passed', 'branch_restore_passed', 'input_contract_passed', 'clip_unchanged', 'other_frozen_unchanged'))
    assert verification['passed'] and verification['report_sha256']==sha(LOCAL/'report.json')
    assert report['manifest_sha256']==sha(PACK/'manifest.json') and not report['checkpoints']
    (PACK/'local_preflight_summary.json').write_text(json.dumps(dict(report_sha256=sha(LOCAL/'report.json'),
        verification_sha256=sha(LOCAL/'verification.json'), execution_passed=True,
        elapsed_seconds=report['elapsed_seconds'], peak_reserved_gib=report['peak_reserved_gib']), indent=2)+'\n')
    files = sorted(p for p in PACK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(PACK).as_posix()}\n' for p in files), newline='\n')
    files.append(PACK/'SHA256SUMS')
    archive = PACK.with_suffix('.zip'); assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p, p.relative_to(PACK.parent).as_posix())
    assert archive.stat().st_size < 100*2**20
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(p.relative_to(PACK.parent).as_posix())==p.read_bytes() for p in files)
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n', newline='\n')
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), files=len(files))))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','package')
    globals()[sys.argv[1]]()
