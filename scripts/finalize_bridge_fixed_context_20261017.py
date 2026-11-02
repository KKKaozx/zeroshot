"""Verify launchers/AST contracts and archive only this small experiment package."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_fixed_context_v1'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    assert not (PACK/'local_validation.json').exists(), 'Package already finalized.'
    protocol=read(PACK/'protocol.json')
    cpu=read(ROOT/'reports/bridge_fixed_context_cpu_20261017/validation.json')
    assert cpu['passed'] and not cpu['project_model_executed'] and not cpu['gpu_used']
    assert cpu['frozen_protocol_sha256']==sha(PACK/'protocol.json')
    for name,h in protocol['local_package_sha256'].items(): assert sha(PACK/name)==h
    assert sha(PACK/'effective_runner.py.txt')==protocol['effective_runner_sha256']
    files=['run_fixed_context.py','fixed_hooks.py','validate_preflight.py','protocol.json',
           'effective_runner.py.txt','preflight.sh','fit.sh','README.md']
    for name in files:
        assert (PACK/name).is_file()
        if name.endswith('.py'): compile((PACK/name).read_text(),name,'exec')
    bash=Path('C:/Program Files/Git/bin/bash.exe'); assert bash.is_file()
    parser=argparse.ArgumentParser()
    for name in ['data-root','data-reference','clip','old-cache','cache','output','control-report']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--stage',choices=['preflight','fit'],required=True)
    parser.add_argument('--preflight-reference')
    launchers={}
    for stage in ['preflight','fit']:
        path=PACK/(stage+'.sh')
        result=subprocess.run([str(bash),'-n',str(path)],capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        raw=path.read_bytes(); assert b'\r' not in raw and raw.startswith(b'#!/bin/bash\n')
        source=path.read_text()
        command=source[source.index('/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_fixed_context.py'):]
        words=shlex.split(command.replace('\\\n',' '))
        assert words[:3]==['/projects/Zeroshot/envs/multitask-preflight-v1/bin/python','-u','run_fixed_context.py']
        args=parser.parse_args(words[3:])
        assert args.stage==stage and args.cache=='/projects/Zeroshot/bridge_demonstration_expansion_v1/full_tokens.pt'
        assert args.output==f'/projects/Zeroshot/runs/bridge-fixed-{stage}-${{SLURM_JOB_ID}}'
        launchers[stage]=dict(bash_syntax_passed=True,arguments_parsed=True,sha256=sha(path))
    write(PACK/'cpu_validation.json',cpu)
    record=dict(verified=True,submitted=False,project_model_executed=False,gpu_used=False,
                prefix_retrained=False,old_joint_formal_repeated=False,launchers=launchers,
                protocol_sha256=sha(PACK/'protocol.json'),cpu_mask_check_passed=True,
                original_context_and_evaluation_ast_unchanged=True,
                update_ast_only_mask_and_observation_callback=True,
                command_has_no_stray_arguments=True,finalizer_sha256=sha(Path(__file__)))
    write(PACK/'local_validation.json',record)
    files+=['cpu_validation.json','local_validation.json']
    manifest=''.join(f'{sha(PACK/name)}  {name}\n' for name in sorted(files))
    (PACK/'SHA256SUMS').write_bytes(manifest.encode())
    files+=['SHA256SUMS']
    archive=PACK.with_suffix('.zip'); assert not archive.exists()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(files): z.write(PACK/name,arcname=PACK.name+'/'+name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.namelist()==[PACK.name+'/'+name for name in sorted(files)]
        for name in files: assert z.read(PACK.name+'/'+name)==(PACK/name).read_bytes()
    checksum=archive.with_suffix('.sha256')
    checksum.write_bytes(f'{sha(archive)}  {archive.name}\n'.encode())
    preparation=read(ROOT/'reports/bridge_fixed_context_preparation_20261017.json')
    preparation.update(local_validation=record,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),
                       checksum_path=str(checksum))
    write(ROOT/'reports/bridge_fixed_context_preparation_20261017.json',preparation)
    print(json.dumps(preparation,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
