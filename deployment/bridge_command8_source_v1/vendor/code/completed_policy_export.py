"""Export a completed fixed-budget policy, then optionally retire its Adam state.

Only the explicitly supplied latest_step.pt from the newly completed protocol
is eligible. No recursive deletion and no changes to any prior experiments.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

FORMAT='main-completed-fixed-policy-v1'


def contract_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def model_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if not isinstance(value,torch.Tensor):raise ValueError('Learned model state must contain tensors only')
        cpu=value.detach().cpu().contiguous()
        h.update(json.dumps([name,str(cpu.dtype),list(cpu.shape)]).encode())
        h.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def export_completed(checkpoint_path,*,discard_completed_optimizer=False):
    source=Path(checkpoint_path).resolve()
    if source.name!='latest_step.pt' or not source.is_file():raise ValueError('An explicit existing latest_step.pt is required')
    checkpoint=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    if checkpoint.get('format')!='fixed-update-training-state-v1':raise ValueError('Not a full main fixed-update checkpoint')
    contract=checkpoint['contract']
    if contract_sha(contract)!=checkpoint['contract_sha256']:raise ValueError('Checkpoint contract checksum differs')
    formal=contract.get('offline_training_protocol')
    if not formal or formal.get('format')!='bounded-oxe-dino-training-protocol-v1' or contract['arguments'].get('oxe_input_preflight'):
        raise ValueError('Only a completed explicit offline training protocol can retire its optimizer state')
    total=contract['arguments']['total_updates'];completed=checkpoint['plan']['completed_updates']
    if type(total) is not int or total<1 or completed!=total or checkpoint['schedule']['total_updates']!=total:
        raise ValueError('Incomplete training must keep its full resume checkpoint')
    if not checkpoint['model']:raise ValueError('Empty learned model state')
    digest=model_sha(checkpoint['model'])
    target=source.parent/'final_policy.pt'
    metadata=source.parent/'completed_policy_export.json'
    if target.exists() or metadata.exists():raise ValueError('Final export already exists; never overwrite or delete another checkpoint')
    payload={'format':FORMAT,'config':contract['config'],'trainable_state_dict':checkpoint['model'],
        'training_contract':contract,'training_contract_sha256':checkpoint['contract_sha256'],
        'completed_updates':completed,'learned_state_sha256':digest,
        'scope':'Completed bounded-source policy weights and provenance; no LIBERO success verdict',
        'optimizer_schedule_rng_resume_state_included':False}
    fd,temporary=tempfile.mkstemp(prefix='final_policy.',suffix='.tmp',dir=source.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        restored=torch.load(temporary,map_location='cpu',weights_only=False,mmap=True)
        if (restored['training_contract_sha256']!=contract_sha(restored['training_contract'])
                or restored['completed_updates']!=total or model_sha(restored['trainable_state_dict'])!=digest):
            raise ValueError('Final policy export verification failed; keep the full checkpoint')
        del restored
        os.replace(temporary,target)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()
    report={'passed':True,'completed_updates':total,'all_learned_tensors_preserved_sha256':digest,
        'policy_file':str(target),'policy_bytes':target.stat().st_size,
        'training_contract_sha256':checkpoint['contract_sha256'],
        'full_checkpoint_before_compaction_bytes':source.stat().st_size,
        'full_optimizer_state_discarded':False,'generalization_verdict':None}
    # Close Windows memory mappings before unlinking the one known source file.
    del payload,checkpoint
    import gc
    gc.collect()
    if discard_completed_optimizer:
        source.unlink();report['full_optimizer_state_discarded']=True
    metadata.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint')
    parser.add_argument('--discard-completed-optimizer',action='store_true')
    args=parser.parse_args()
    print(json.dumps(export_completed(args.checkpoint,discard_completed_optimizer=args.discard_completed_optimizer),indent=2))
