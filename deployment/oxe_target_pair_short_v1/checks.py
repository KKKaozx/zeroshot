"""Metadata/space admission only; no encoder/model construction."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from protocols import HERE,read,sha


def check(root):
    pair=read(HERE/'pair_protocol.json');ref=read(HERE/'reference_contract.json')
    logical=__import__('hashlib').sha256(json.dumps(ref,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert logical==pair['source_reference_contract_sha256']
    for name,digest in pair['bundle_inputs'].items():assert sha(HERE/name)==digest,name
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(HERE/name)==digest,name
    for name,digest in ref['source_code_sha256'].items():assert sha(HERE/'code'/name)==digest,name
    assert sha(HERE/'source_protocol.json')==pair['original_source_protocol_sha256']
    assert sha(HERE/'source_binding.json')==ref['offline_training_protocol']['cluster_preflight_binding_sha256']
    source=read(HERE/'source_protocol.json');inputs=root/'oxe_dino_main_preflight_v1/inputs'
    for key,name in [('split_manifest','split_manifest.json'),('original_instruction_catalog','instruction_catalog.json')]:
        assert sha(inputs/name)==source['input_file_sha256'][key]
    for role in ['train','validation']:assert sha(inputs/role/'manifest.json')==source['target_manifest_sha256'][role]
    for target in pair['arms']:
        assert read(HERE/f'protocol_{target}.json')['pair_protocol_sha256']==sha(HERE/'pair_protocol.json')
    used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0]);reserve=8*1024**3
    assert used+reserve<200_000_000_000 and shutil.disk_usage(root).free>reserve,'8GiB reserve required'
    return {'passed':True,'prefix_updates_per_arm':8192,'lr_horizon':65536,'new_training_runs':2,
        'model_constructed':False,'updates_executed':0,'project_allocated_bytes':used,'reserve_bytes':reserve,
        'scope':'Metadata/project du only, not authoritative quota; no automatic submission'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    print(json.dumps(check(ap.parse_args().root.resolve()),indent=2))
