"""Reuse frozen202278 trainer; change only its preceding visual-token slot."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'cluster_packages/bridge_history_generalization_v1'
PREP = ROOT / 'reports/bridge_history_generalization_preparation_20261014'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    assert not (PACK / 'protocol.json').exists()
    p, audit = read(PREP / 'protocol.json'), read(PREP / 'input_audit.json')
    assert audit['passed'] and audit['protocol_sha256'] == sha(PREP / 'protocol.json')
    parent = ROOT / 'cluster_packages/bridge_state_conditioning_v1/fit_state_arm.py'
    original = parent.read_text(encoding='utf-8'); updated = original
    changes = []
    def replace(old, new):
        nonlocal updated
        assert updated.count(old) == 1, old
        updated = updated.replace(old, new); changes.append(dict(old=old, new=new))
    replace('from models import RobotAdapterModel', 'from models import RobotAdapterModel\nfrom history_inputs import load_previous_images')
    replace('    args = ap.parse_args()', '    ap.add_argument("--history-arm", choices=("repeat_current", "real_previous"), required=True)\n    args = ap.parse_args()')
    replace('        targets = torch.stack([v[3] for v in items]).cuda()',
            '        previous_images, history_checks = load_previous_images(rows, items, args.data_root)\n'
            '        report.update(history_arm=args.history_arm, history_input_checks=history_checks)\n'
            '        targets = torch.stack([v[3] for v in items]).cuda()')
    replace('        mask = texts["attention_mask"].cuda()',
            '        with torch.no_grad():\n'
            '            older = (torch.cat([model.vision_encoder(pixel_values=previous_images[i:i+2].cuda()).last_hidden_state\n'
            '                     for i in range(0, len(items), 2)]).detach() if args.history_arm == "real_previous" else visual)\n'
            '        del previous_images\n'
            '        mask = texts["attention_mask"].cuda()')
    replace('            repeated = model.adapter(visual[indices], language[indices], mask[indices])',
            '            repeated = model.adapter(older[indices], language[indices], mask[indices])')
    replace('            payload["state_arm"]=args.state_arm',
            '            payload["history_arm"] = args.history_arm\n'
            '            payload["history_protocol_sha256"] = sha(Path(__file__).resolve().parent/"protocol.json")\n'
            '            payload["reconstruction"] += " History image slot: " + args.history_arm + "; real_previous uses same-record max(0,t-1)."\n'
            '            payload["state_arm"]=args.state_arm')
    # Explicit static proof: the original optimizer/update/loss/sampling/check
    #bodies have not been replaced by a new implementation.
    orig_tree, new_tree = ast.parse(original), ast.parse(updated)
    functions = lambda tree: {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    old_functions, new_functions = functions(orig_tree), functions(new_tree)
    for name in ['advance', 'check', 'gradient_norms', 'cpu_copy', 'state_sha', 'trainable_copy']:
        assert ast.dump(old_functions[name], include_attributes=False) == ast.dump(new_functions[name], include_attributes=False)
    imports = lambda tree: [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                            if isinstance(n, ast.ImportFrom) and n.module == 'trace_bridge_batch_training']
    assert imports(orig_tree) == imports(new_tree)
    (PACK / 'fit_history_arm.py').write_text(updated, encoding='utf-8', newline='\n')
    state = ROOT / 'cluster_packages/bridge_state_conditioning_v1/state_input_contract.py'
    (PACK / state.name).write_bytes(state.read_bytes())
    (PACK / 'history_inputs.json').write_bytes((PREP / 'input_audit.json').read_bytes())
    required = {'bridge_growth128_v1/manifest.json': p['base_manifest_sha256'],
                'bridge_state_conditioning_v1/protocol.json': p['state_protocol_sha256']}
    state_protocol = read(ROOT / 'cluster_packages/bridge_state_conditioning_v1/protocol.json')
    required.update({'bridge_growth128_v1/' + n: h for n, h in state_protocol['base_files_sha256'].items()})
    run = ROOT / 'reports/bridge-state-fit-202278/real'
    for n in ['report.json', 'common_u0000_train.npz', 'common_updates.jsonl']:
        required['runs/bridge-state-fit-202278/real/' + n] = sha(run / n)
    p.update(current_stage='Prepared input/replay/gradient GPU preflight; no job submitted or full training performed',
             history_inputs_sha256=sha(PACK / 'history_inputs.json'), required_file_sha256=required,
             parent_trainer_sha256=sha(parent), parent_state_contract_sha256=sha(state),
             preflight_wall_minutes=25, formal_pair_wall_minutes=150,
             formal_checkpoint_budget_bytes=2*1024**3, minimum_project_headroom_bytes=15*1024**3,
             input_audit_windows=168, input_audit_real_history_train=64, input_audit_real_history_development=30)
    (PACK / 'protocol.json').write_text(json.dumps(p, indent=2) + '\n', encoding='utf-8', newline='\n')
    for stage, minutes in [('preflight', 25), ('fit', 150)]:
        prerequisite = '' if stage == 'preflight' else ': "${HISTORY_PREFLIGHT_JOB:?Supply successful preflight job ID}"\n'
        extra = '' if stage == 'preflight' else ' --preflight-reference "/projects/Zeroshot/runs/bridge-history-preflight-${HISTORY_PREFLIGHT_JOB}"'
        budget = '' if stage == 'preflight' else '''USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
echo "Project bytes (du planning estimate, not live quota): $USED_BYTES"
if (( USED_BYTES + 2*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then exit 1; fi
'''
        shell = f'''#!/bin/bash
#SBATCH --job-name=bridge-history-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time={minutes//60:02d}:{minutes%60:02d}:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-history-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/history-{stage}-${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
{prerequisite}{budget}cd /projects/Zeroshot/bridge_history_generalization_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_history_pair.py --project /projects/Zeroshot --stage {stage} --output "/projects/Zeroshot/runs/bridge-history-{stage}-${{SLURM_JOB_ID}}"{extra}
'''
        (PACK / (stage + '.sh')).write_text(shell, encoding='utf-8', newline='\n')
    proof = dict(stage='static_local_preparation', passed=True, parent_source_sha256=sha(parent),
                 patches=changes, unchanged_function_ast=['advance', 'check', 'gradient_norms', 'cpu_copy', 'state_sha', 'trainable_copy'],
                 original_base_copy_import_unchanged=True,
                 bash_syntax_passed=False, full_gpu_preflight_executed=False, trained=False,
                 scope='Input audit and unchanged trainer-body proof only; numerical GPU replay and gradient tests remain in the cluster preflight')
    (PREP / 'package_preparation.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print('Prepared one bounded pair from frozen trainer; optimization/sampling/check bodies unchanged.')


def package():
    proof = read(PREP / 'package_preparation.json')
    assert proof['passed'] and proof['bash_syntax_passed']
    for n, h in proof['source_sha256'].items(): assert sha(PACK / n) == h
    for f in PACK.glob('*.py'): ast.parse(f.read_text(encoding='utf-8'), feature_version=(3, 10))
    (PACK / 'local_preparation.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    files = sorted(p for p in PACK.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    (PACK / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files), encoding='ascii', newline='\n')
    files.append(PACK / 'SHA256SUMS'); archive = PACK.with_suffix('.zip'); assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, f.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for f in files: assert z.read(f.relative_to(PACK.parent).as_posix()) == f.read_bytes()
    assert archive.stat().st_size < 1024**2
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n', encoding='ascii', newline='\n')
    print(json.dumps(dict(bytes=archive.stat().st_size, files=len(files), sha256=sha(archive))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'package']; globals()[sys.argv[1]]()
