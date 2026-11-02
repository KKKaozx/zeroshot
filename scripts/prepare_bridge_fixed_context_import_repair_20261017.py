"""Freeze an import-only wrapper, keeping all original protocol/source files intact."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_fixed_context_import_repair_v1'
OLD = ROOT/'cluster_packages/bridge_fixed_context_v1'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (PACK/'repair_contract.json').exists(), 'Repair already prepared'
    original = {name: sha(OLD/name) for name in ('run_fixed_context.py','fixed_hooks.py','validate_preflight.py','protocol.json','fit.sh')}
    assert original['protocol.json'] == 'ff909f9eae31c92c11ef1ab5fd3c7167215ed03680d0a8997639879fa2635d6c'
    contract = dict(date='2026-10-17', failed_job='208926', successful_preflight='208897',
        failure='After sys.path prepending inherited input directories, lazy from validate_preflight import validate resolves the expansion directory module, which lacks validate. Failure is before advance8193.',
        repair='Load unchanged fixed_hooks and validator by absolute path; bind that exact validator module only during the unchanged original FixedHooks.fit call. Prevalidate before constructing models. Restore previous module-cache entry after fit.',
        original_files_sha256=original, experiment_protocol_sha256=original['protocol.json'],
        repair_runner_sha256=sha(PACK/'run_repaired.py'),
        preserved=['All old packages/failed output','Original protocol and preflight validator/thresholds',
                   'Successful preflight208897','Original context/check/advance/fit function bodies',
                   '8192 model and Adam','Batches,seeds,Dropout,global clipping,update mask,7978 budget',
                   'Development rows and evaluation gates','All reserved test data'],
        allowed_retry='One new formal job only after unchanged full validator passes and failed208926 report verifies no new updates/checkpoints. Never rerun the12-step GPU preflight or8192 prefix.',
        limitations='Local import regression uses a clearly synthetic preflight fixture and stops at advance entry. It does not replace actual cluster report validation or establish generalization.')
    (PACK/'repair_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    script=(OLD/'fit.sh').read_text()
    old='  --stage fit --output '
    assert script.count(old)==1
    entry='/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_fixed_context.py'
    assert script.count(entry)==1
    script=script.replace(entry,'/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u /projects/Zeroshot/bridge_fixed_context_import_repair_v1/run_repaired.py')
    point='sha256sum --status -c SHA256SUMS\n'
    assert script.count(point)==1
    script=script.replace(point,point+'(cd /projects/Zeroshot/bridge_fixed_context_import_repair_v1 && sha256sum --status -c SHA256SUMS)\n')
    script=script.replace('#SBATCH --job-name=bridge-fixed-fit','#SBATCH --job-name=bridge-fixed-retry')
    (PACK/'fit_repaired.sh').write_bytes(script.encode())
    print('IMPORT-ONLY REPAIR PREPARED; original package unchanged')


if __name__=='__main__': main()
