"""Bind real cluster metadata only; never transfer learned preflight weights."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/projects/Zeroshot')
    parser.add_argument('--bundle',default=str(Path(__file__).resolve().parent));args=parser.parse_args()
    root=Path(args.root).resolve();bundle=Path(args.bundle).resolve()
    protocol_path=bundle/'training_protocol.json';protocol=json.loads(protocol_path.read_text())
    destination=bundle/'cluster_preflight_binding.json'
    if destination.exists():
        existing=json.loads(destination.read_text())
        if (existing.get('format')!='passed-oxe-cluster-preflight-binding-v1' or existing.get('passed') is not True
                or existing.get('training_protocol_sha256')!=sha(protocol_path)
                or existing.get('dataset_identity')!=protocol['dataset_identity']
                or existing.get('copied_learned_preflight_weights') is not False):
            raise ValueError('Existing cluster binding differs; no overwrite')
        print('Reusing immutable passed-preflight metadata binding',existing['preflight_job_id']);return
    preflight=root/'oxe_dino_main_preflight_v1'
    job=(preflight/'preflight_job_id.txt').read_text().strip()
    if not job.isdigit():raise ValueError('Existing preflight job marker required')
    output=root/'runs'/('oxe-dino-pref-'+job)
    passed=json.loads((output/'preflight_report.json').read_text())
    step=json.loads((output/'step_report.json').read_text())
    if (passed.get('passed') is not True or passed.get('completed_updates')!=2
            or step.get('input_preflight_only') is not True or not step.get('frozen_state_unchanged')
            or step.get('completed_updates')!=2 or len(step.get('validation',[]))!=1
            or step['target_binding']['dataset_identity']!=protocol['dataset_identity']):
        raise ValueError('Matching successful actual main preflight required')
    for name,expected in protocol['input_file_sha256'].items():
        filename={'split_manifest':'split_manifest.json','original_instruction_catalog':'instruction_catalog.json'}[name]
        if sha(preflight/'inputs'/filename)!=expected:raise ValueError('Cluster frozen input file changed')
    for role,expected in protocol['target_manifest_sha256'].items():
        if sha(preflight/'inputs'/role/'manifest.json')!=expected:raise ValueError('Cluster target manifest changed')
    import torch
    checkpoint=torch.load(output/'latest_step.pt',map_location='cpu',weights_only=False,mmap=True)
    contract=checkpoint['contract']
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if checkpoint['format']!='fixed-update-training-state-v1' or digest!=checkpoint['contract_sha256']:
        raise ValueError('Actual preflight checkpoint contract checksum differs')
    if contract['dataset_identity']!=protocol['dataset_identity'] or checkpoint['plan']['completed_updates']!=2:
        raise ValueError('Actual preflight checkpoint dataset/update count differs')
    value={'format':'passed-oxe-cluster-preflight-binding-v1','passed':True,'preflight_job_id':job,
        'dataset_identity':protocol['dataset_identity'],'training_protocol_sha256':sha(protocol_path),
        'preflight_report_sha256':sha(output/'preflight_report.json'),'preflight_step_report_sha256':sha(output/'step_report.json'),
        'preflight_checkpoint_contract_sha256':digest,'frozen_state_sha256':contract['frozen_state_sha256'],
        'tokenizer':contract['tokenizer'],'packages':contract['packages'],
        'gpu':passed['gpu'],'torch_version':passed['torch_version'],'cuda_runtime':passed['cuda_runtime'],
        'preflight_peak_reserved_gib':passed['peak_reserved_gib'],
        'copied_learned_preflight_weights':False,'formal_training_submitted':False}
    if destination.exists():
        if json.loads(destination.read_text())!=value:raise ValueError('A different cluster binding already exists; no overwrite')
    else:destination.write_text(json.dumps(value,indent=2)+'\n')
    print('PASSED CLUSTER PREFLIGHT BOUND; metadata only, fresh formal initialization',job)
    print(json.dumps({'gpu':value['gpu'],'preflight_peak_reserved_gib':value['preflight_peak_reserved_gib'],
        'training_protocol_sha256':value['training_protocol_sha256']},indent=2))


if __name__=='__main__':main()
