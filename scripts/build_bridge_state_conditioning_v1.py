"""Prepare a bounded current-pose pair without copying images or checkpoints."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'cluster_packages/bridge_growth128_v1'
PACK = ROOT/'cluster_packages/bridge_state_conditioning_v1'
LOCAL = ROOT/'reports/bridge_state_local_preflight_v2_20261013'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def once(s, a, b):
    assert s.count(a)==1, a
    return s.replace(a,b)

def prepare():
    assert not (PACK/'protocol.json').exists(), 'Preserve already prepared package'
    PACK.mkdir(exist_ok=True)
    m=read(BASE/'manifest.json'); audit=ROOT/'reports/bridge_condition_input_audit_20261013'
    assert read(audit/'raw_audit.json')['passed']
    with np.load(audit/'observed_states.npz',allow_pickle=False) as z:
        # Export current XYZ/RPY only. Future observations never enter this package.
        observed=z['current_state'][:,:6].astype('<f4')
    assert observed.shape==(168,6) and np.isfinite(observed).all()
    np.savez_compressed(PACK/'current_pose.npz',current_pose=observed)
    mean=observed[:128,:3].astype(float).mean(axis=0)
    std=observed[:128,:3].astype(float).std(axis=0)
    assert np.min(std)>1e-6
    p=dict(date='2026-10-13',hypothesis='Does explicit observed current end-effector pose improve same-data prediction?',
        base_directory='bridge_growth128_v1', base_files_sha256={n:sha(BASE/n) for n in m['source_sha256']},
        base_manifest_sha256=sha(BASE/'manifest.json'),
        control_report_sha256=sha(ROOT/'reports/bridge-growth128-fit-201663/report.json'),
        raw_audit_sha256=sha(audit/'raw_audit.json'), current_pose_sha256=sha(PACK/'current_pose.npz'),
        rows=m['train']+m['development'], train_windows=128, development_windows=40,
        state_features=dict(xyz_mean=mean.tolist(),xyz_std=std.tolist(),fit_rows=list(range(128)),
            rotation='Rz(yaw) Ry(pitch) Rx(roll), first two columns concatenated',
            feature_order=['normalized_x','normalized_y','normalized_z','R00','R10','R20','R01','R11','R21'],
            clipping=False, future_state_used=False, indices_used_as_features=False),
        port=dict(input_dim=9,output_dim=1024,bias=False,initial_weight='all_zero',
            placement='Add linear observed-state projection to unchanged Adapter/history context',
            arms=['zero','real'],same_architecture=True),
        initialization_seed=43,batch_size=8,updates_per_arm=8192,learning_rates=[1e-4,1e-5],switch_after_update=4096,
        prediction_type='sample',sampler='native DDPM100',sampling_seeds=m['sampling_seeds'],
        frozen=['CLIP','gripper'],local_full_training=False,local_checkpoints=False,
        screen='Require both arms training fit, lower paired development path position AND rotation, and position improvement in at least6/10 demonstrations; one additional paired initialization only if candidate.',
        limits=['Repeated40-window development set is exploratory.',
            'Observed pose is expressed in source dataset coordinates; cross-episode physical calibration is not independently verified.',
            'Frozen gripper; pose errors cannot establish grasp success. Execution passing is not scientific improvement.'])
    (PACK/'protocol.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
    source=(BASE/'fit_bridge_growth128.py').read_text(encoding='utf-8')
    bootstrap='import sys\nBASE = Path(__file__).resolve().parent.parent/"bridge_growth128_v1"\nsys.path.insert(0,str(BASE))\n'
    source=once(source,'import time\n','import time\n'+bootstrap)
    source=once(source,'    code = Path(__file__).resolve().parent','    code = BASE')
    # A read-only observer exposes final parameter identities for the replay proof.
    source=once(source,'            if args.stage == "preflight":\n                current_hash',
        '            row["final_base_state_sha256"] = state_sha(trainable_copy(model))\n'
        '            if args.stage == "preflight":\n                current_hash')
    (PACK/'baseline_observed.py').write_text(source,encoding='utf-8',newline='\n')
    variant=source
    variant=once(variant,'from trace_bridge_batch_training import PREFIXES, observation_guard, trainable_copy, gradient_norms',
        'from trace_bridge_batch_training import PREFIXES as BASE_PREFIXES, observation_guard, trainable_copy as base_copy, gradient_norms as base_gradients\n'
        'from state_input_contract import load_current_features, add_zero_port, clipping_parameters\n'
        'PREFIXES=BASE_PREFIXES+("state_projection.",)\n'
        'def trainable_copy(model):\n'
        '    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith(PREFIXES)}\n'
        'def gradient_norms(model):\n'
        '    result=base_gradients(model)\n'
        '    result["state_projection."]=float(torch.linalg.vector_norm(model.state_projection.weight.grad.detach()))\n'
        '    assert np.isfinite(result["state_projection."])\n'
        '    return result')
    variant=once(variant,'    args = ap.parse_args()',
        '    ap.add_argument("--state-arm",choices=("zero","real"),required=True)\n'
        '    ap.add_argument("--state-protocol",type=Path,required=True)\n'
        '    args = ap.parse_args()')
    variant=once(variant,'        trainables = [p for p in model.parameters() if p.requires_grad]',
        '        report["base_initial_shared_sha256"] = initial_hash\n'
        '        features, state_checks = load_current_features(args.state_protocol, rows, args.data_root, report["data_checks"])\n'
        '        state_features = torch.from_numpy(features).cuda()\n'
        '        if args.state_arm == "zero": state_features = torch.zeros_like(state_features)\n'
        '        add_zero_port(model)\n'
        '        report.update(state_arm=args.state_arm, state_checks=state_checks, state_features_sha256=tensor_sha(state_features),\n'
        '            state_protocol_sha256=sha(args.state_protocol),\n'
        '            augmented_initial_sha256=state_sha(trainable_copy(model)), state_port_initial_zero=True)\n'
        '        initial["state_projection.weight"] = model.state_projection.weight.detach().cpu().clone()\n'
        '        trainables = [p for p in model.parameters() if p.requires_grad]')
    variant=once(variant,'            return now+model.history_projection(repeated-now)',
        '            return now+model.history_projection(repeated-now)+model.state_projection(state_features[indices])')
    variant=once(variant,'torch.nn.utils.clip_grad_norm_(trainables, 1., error_if_nonfinite=True)',
        'torch.nn.utils.clip_grad_norm_(clipping_parameters(model, trainables, args.state_arm), 1., error_if_nonfinite=True)')
    variant=once(variant,'        report["common_state_sha256"] = warm_weights_hash; report["common_adam_sha256"] = warm_adam_hash',
        '        report["common_base_state_sha256"] = state_sha(base_copy(model))\n'
        '        report["common_state_sha256"] = warm_weights_hash; report["common_adam_sha256"] = warm_adam_hash')
    variant=once(variant,'            row["final_base_state_sha256"] = state_sha(trainable_copy(model))',
        '            row["final_base_state_sha256"] = state_sha(base_copy(model))\n'
        '            row["state_port_weight_norm"] = float(torch.linalg.vector_norm(model.state_projection.weight.detach()))\n'
        '            row["state_port_context_max_abs"] = float(model.state_projection(state_features).detach().abs().max())\n'
        '            row["state_port_sha256"] = tensor_sha(model.state_projection.weight)')
    variant=once(variant,'Recreate initialized frozen model, add zero history_projection, then load named trainable state.',
        'Recreate frozen model; add zero history_projection and bias-free9-to1024 state_projection; load trainable state; use checkpoint state protocol to construct current observed-pose context.')
    variant=once(variant,'            file = args.output/name; torch.save(payload, file)',
        '            payload["state_arm"]=args.state_arm\n'
        '            payload["state_protocol"]=json.loads(args.state_protocol.read_text(encoding="utf-8"))\n'
        '            file = args.output/name; torch.save(payload, file)')
    for name,text in [('baseline_observed.py',source),('fit_state_arm.py',variant)]:
        ast.parse(text);(PACK/name).write_text(text,encoding='utf-8',newline='\n')
    for stage in ('preflight','fit'):
        shell=(BASE/f'{stage}.sh').read_text(encoding='utf-8')
        shell=shell.replace('bridge-growth128-','bridge-state-').replace('bridge_growth128_v1','bridge_state_conditioning_v1')
        shell=shell.replace('GROWTH_PREFLIGHT_JOB','STATE_PREFLIGHT_JOB').replace('fit_bridge_growth128.py','run_state_pair.py')
        shell=shell.replace('--manifest manifest.json','--manifest ../bridge_growth128_v1/manifest.json')
        shell=shell.replace('--stage '+stage,'--control-report /projects/Zeroshot/runs/bridge-growth128-fit-201663/report.json \\\n  --stage '+stage)
        shell=shell.replace('+ 1024*1024*1024','+ 2*1024*1024*1024')
        shell=shell.replace('--time=00:30:00','--time=00:45:00').replace('--time=02:00:00','--time=03:00:00')
        assert '\n+' not in shell
        (PACK/f'{stage}.sh').write_text(shell,encoding='utf-8',newline='\n')
    print('Prepared current-state pair, only6 observed pose components exported.')

def package():
    r=read(LOCAL/'report.json');v=read(LOCAL/'verification.json')
    assert r['passed'] and v['passed'] and v['report_sha256']==sha(LOCAL/'report.json')
    (PACK/'local_preflight_summary.json').write_text(json.dumps(dict(passed=True,
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
