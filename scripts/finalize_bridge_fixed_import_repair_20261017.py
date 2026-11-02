"""Archive the tested import-only repair; preserve all original package hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_fixed_context_import_repair_v1'
OLD=ROOT/'cluster_packages/bridge_fixed_context_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def main():
    assert not PACK.with_suffix('.zip').exists(),'Repair already archived'
    contract=read(PACK/'repair_contract.json')
    check=read(ROOT/'reports/bridge_fixed_import_repair_validation_20261017/validation.json')
    assert check['passed'] and check['repair_contract_sha256']==sha(PACK/'repair_contract.json')
    assert check['training_updates_executed']==0 and not check['gpu_used']
    for name,expected in contract['original_files_sha256'].items():assert sha(OLD/name)==expected
    assert sha(PACK/'run_repaired.py')==contract['repair_runner_sha256']
    compile((PACK/'run_repaired.py').read_text(),'run_repaired.py','exec')
    old=(OLD/'fit.sh').read_text();new=(PACK/'fit_repaired.sh').read_text()
    expected=old.replace('#SBATCH --job-name=bridge-fixed-fit','#SBATCH --job-name=bridge-fixed-retry')
    expected=expected.replace('sha256sum --status -c SHA256SUMS\n','sha256sum --status -c SHA256SUMS\n(cd /projects/Zeroshot/bridge_fixed_context_import_repair_v1 && sha256sum --status -c SHA256SUMS)\n')
    expected=expected.replace('-u run_fixed_context.py','-u /projects/Zeroshot/bridge_fixed_context_import_repair_v1/run_repaired.py')
    assert new==expected
    def arguments(source):
        line=source[source.index('/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u '):]
        return shlex.split(line.replace('\\\n',' '))[3:]
    assert arguments(old)==arguments(new)
    assert b'\r' not in (PACK/'fit_repaired.sh').read_bytes()
    write(PACK/'cpu_regression.json',check)
    record=dict(verified=True,submitted=False,repair_only=True,
                original_training_arguments_identical=True,original_package_files_unchanged=True,
                bash_syntax_exit_code=0,bash_syntax_script_sha256=sha(PACK/'fit_repaired.sh'),
                existing_preflight_reused='208897',failed_job='208926',
                project_model_executed=False,training_updates_executed=0,gpu_used=False,
                finalizer_sha256=sha(Path(__file__)))
    write(PACK/'local_validation.json',record)
    files=['run_repaired.py','repair_contract.json','fit_repaired.sh','README.md','cpu_regression.json','local_validation.json']
    (PACK/'SHA256SUMS').write_bytes(''.join(f'{sha(PACK/n)}  {n}\n' for n in sorted(files)).encode())
    files+=['SHA256SUMS']
    archive=PACK.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(files):z.write(PACK/name,PACK.name+'/'+name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in files:assert z.read(PACK.name+'/'+name)==(PACK/name).read_bytes()
    archive.with_suffix('.sha256').write_bytes(f'{sha(archive)}  {archive.name}\n'.encode())
    record['archive']=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))
    record['original_archive_sha256']=sha(OLD.with_suffix('.zip'))
    assert record['original_archive_sha256']=='69f2e6cde487f4ce5bece447abc47195531f08e473c5f526048aad665d1e7990'
    write(ROOT/'reports/bridge_fixed_import_repair_preparation_20261017.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
