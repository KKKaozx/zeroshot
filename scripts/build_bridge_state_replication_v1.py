"""Prepare the one predetermined initialization42 replication of state pair202278."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/'cluster_packages/bridge_state_conditioning_v1'
PACK = ROOT/'cluster_packages/bridge_state_replication_v1'
LOCAL = ROOT/'reports/bridge_state_replication_local_preflight_20261013'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))

def prepare():
    assert not PACK.exists(), 'Preserve frozen packages'
    proof = ROOT/'reports/bridge_state_analysis_202278_20261013/analysis.json'
    analysis = read(proof)
    assert analysis['verified'] and analysis['both_training_fit'] and analysis['candidate_screen_passed']
    for line in (ORIGINAL/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha(ORIGINAL/name) == digest
    PACK.mkdir()
    for name in ('baseline_observed.py','fit_state_arm.py'):
        source = (ORIGINAL/name).read_text(encoding='utf-8')
        assert source.count('init = 43; seed(init)') == 1
        changed = source.replace('init = 43; seed(init)', 'init = 42; seed(init)')
        assert changed.replace('init = 42; seed(init)', 'init = 43; seed(init)') == source
        ast.parse(changed)
        (PACK/name).write_text(changed,encoding='utf-8',newline='\n')
    for name in ('state_input_contract.py','current_pose.npz'):
        shutil.copyfile(ORIGINAL/name, PACK/name)
    runner = (ORIGINAL/'run_state_pair.py').read_text(encoding='utf-8')
    line="            report=read(args.output/arm/'report.json');assert report['execution_checks_passed']"
    assert runner.count(line) == 1
    runner=runner.replace(line, line+"\n            assert report['initialization_seed']==protocol['initialization_seed']")
    ast.parse(runner)
    (PACK/'run_state_pair.py').write_text(runner,encoding='utf-8',newline='\n')
    p=read(ORIGINAL/'protocol.json');p['initialization_seed']=42
    p['replication']=dict(original_initialization=43,replication_initialization=42,
        original_protocol_sha256=sha(ORIGINAL/'protocol.json'),
        original_analysis_sha256=sha(proof),
        original_pair_report_sha256=sha(ROOT/'reports/bridge-state-fit-202278/report.json'),
        original_source_sha256={n:sha(ORIGINAL/n) for n in ('baseline_observed.py','fit_state_arm.py','run_state_pair.py','state_input_contract.py')},
        changing_variable='Only shared model initialization43 ->42. Same data, batches, noise seeds, state port, LR and budget.',
        additional_initializations_allowed=1,
        candidate_success='Both training fits; lower paired development path position AND rotation; at least6/10 demonstrations position improved.',
        stop_if_not_replicated=True,no_additional_seed_selection=True,
        development_pass_requires_better_than_static_position_and_rotation=True,
        prepared_before_new_initialization_results=True)
    before=read(ORIGINAL/'protocol.json')
    assert {k:v for k,v in p.items() if k not in ('initialization_seed','replication')} == {k:v for k,v in before.items() if k!='initialization_seed'}
    (PACK/'protocol.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
    for stage in ('preflight','fit'):
        source=(ORIGINAL/f'{stage}.sh').read_text(encoding='utf-8')
        shell=source.replace('bridge_state_conditioning_v1','bridge_state_replication_v1').replace('bridge-state-','bridge-state-rep-').replace('STATE_PREFLIGHT_JOB','STATE_REP_PREFLIGHT_JOB')
        assert '\n+' not in shell and shell.startswith('#!/bin/bash\n')
        (PACK/f'{stage}.sh').write_text(shell,encoding='utf-8',newline='\n')
    (PACK/'README.md').write_text('One fixed initialization42 replication of state pair202278 (initialization43). '
        'Both zero/real arms: same128 train and40 exploratory development windows,8192 updates, nativeDDPM100, '
        'unchanged batches/noise/LR, frozenCLIP/gripper. Zero-vs-real observation pose remains the sole intervention. '
        'No new data, assets or environments. Preflight tests execution/replay, not generalization. '
        'No further seed selection or hyperparameter sweeps if the candidate screen does not replicate.\n',encoding='utf-8')
    print('Prepared fixed initialization42 pair; reused frozen data and production modules.')

def package():
    r=read(LOCAL/'report.json');v=read(LOCAL/'verification.json')
    assert r['passed'] and v['passed'] and v['report_sha256']==sha(LOCAL/'report.json')
    assert r['protocol_sha256']==sha(PACK/'protocol.json')
    assert all(read(LOCAL/a/'report.json')['initialization_seed']==42 for a in ('legacy','zero','real'))
    assert not list(LOCAL.rglob('*.pt'))
    (PACK/'local_preflight_summary.json').write_text(json.dumps(dict(passed=True,initialization_seed=42,
        report_sha256=sha(LOCAL/'report.json'),verification_sha256=sha(LOCAL/'verification.json'),
        elapsed_seconds=r['elapsed_seconds'],zero_replay=r['zero_replay'],state_interface=r['state_interface']),indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in PACK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(PACK).as_posix()}\n' for p in files),encoding='ascii',newline='\n')
    files.append(PACK/'SHA256SUMS');archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(f.relative_to(PACK.parent).as_posix())==f.read_bytes() for f in files)
    assert archive.stat().st_size<100*2**20
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='ascii',newline='\n')
    print(json.dumps(dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(files))))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','package')
    globals()[sys.argv[1]]()
