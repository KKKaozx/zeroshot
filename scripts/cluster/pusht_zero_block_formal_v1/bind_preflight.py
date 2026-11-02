"""Bind the successful zero-block resource check, without re-running updates."""
import argparse
import importlib.util
import json
from pathlib import Path
from native_data import sha,save_json


def verify_binding(path,root,bundle):
    contract=json.loads((bundle/'project_contract.json').read_text())
    binding=json.loads(path.read_text())
    if (binding['format']!='pusht-zero-preflight-binding-v1' or binding['preflight_job']!=219260
            or binding['formal_contract_sha256']!=sha(bundle/'project_contract.json')
            or binding['preflight_protocol_sha256']!=contract['preflight_protocol_sha256']
            or not binding['verified']['preflight_verified'] or binding['preflight_weights_used']):
        raise ValueError('Wrong preflight binding')
    directory=root/contract['preflight_directory']
    if sha(directory/'report.json')!=binding['preflight_report_sha256']:
        raise ValueError('Admitted preflight report changed')
    report=json.loads((directory/'report.json').read_text())
    if binding['initialization']!=report['initialization'] or not report['preflight_passed']:
        raise ValueError('Preflight initialization changed')
    return binding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    p.add_argument('--binding',type=Path,required=True);a=p.parse_args()
    root,bundle=a.root.resolve(),Path(__file__).resolve().parent
    if a.binding.exists(): raise ValueError('Binding already exists; do not overwrite or repeat submission')
    c=json.loads((bundle/'project_contract.json').read_text())
    for name,digest in c['bundle_sources_sha256'].items():
        if sha(bundle/name)!=digest: raise ValueError('Changed formal bundle '+name)
    source=root/'pusht_zero_block_preflight_v1'
    if sha(source/'protocol.json')!=c['preflight_protocol_sha256']:
        raise ValueError('Original preflight protocol changed')
    spec=importlib.util.spec_from_file_location('pinned_zero_preflight_validator',source/'verify_preflight.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    directory=root/c['preflight_directory']
    verified=module.validate(directory,source)
    report=json.loads((directory/'report.json').read_text())
    binding=dict(format='pusht-zero-preflight-binding-v1',preflight_job=219260,
        formal_contract_sha256=sha(bundle/'project_contract.json'),preflight_protocol_sha256=c['preflight_protocol_sha256'],
        preflight_report_sha256=sha(directory/'report.json'),initialization=report['initialization'],verified=verified,
        preflight_weights_used=False)
    save_json(a.binding,binding)
    verify_binding(a.binding,root,bundle)
    print('ZERO BLOCK PREFLIGHT BOUND; resource/interface only, no task-performance verdict')
    print(json.dumps(verified,indent=2))


if __name__=='__main__':main()
