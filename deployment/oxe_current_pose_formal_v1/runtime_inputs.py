"""Reuse the admitted source dataset, schedule, full Adam state and feature statistics."""
import numpy as np
from binding import read


def load(root,pg):
    import torch
    import train as api
    from verify import arm
    from completed_policy_export import contract_sha,model_sha
    from oxe_nominal_dataset import NominalTrainValidationDataset
    from step_training_state import UpdateSchedule
    from pose_features import HERE,validate_stats
    arm(root/'runs/oxe-target-short-231297/sample','sample')
    path=root/'runs/oxe-target-short-231297/sample/latest_step.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    reference=read(HERE/'protocol.json')
    assert payload['contract_sha256']==contract_sha(payload['contract'])==reference['source_contract_sha256']
    assert model_sha(payload['model'])==reference['source_learned_sha256'] and payload['plan']['completed_updates']==8192
    data=NominalTrainValidationDataset(root/'data/oxe_nominal_pool_v1',root/'oxe_dino_main_preflight_v1/inputs/train',root/'oxe_dino_main_preflight_v1/inputs/validation')
    assert data.nominal_target_binding==payload['contract']['config']['action']['nominal_target_binding']
    assert data.frozen_nominal_splits==payload['contract']['split_indices']
    assert api.dataset_split_identity(data)==payload['contract']['dataset_identity']
    assert data.record_integrity_contract==payload['contract']['source_record_integrity']
    stats=read(HERE/'state_stats.json');validate_stats(data.training.store,stats)
    schedule=UpdateSchedule(**payload['schedule'])
    with np.load(HERE/'reference_suffix.npz') as z:
        assert all(schedule.rates(8192+i)==z['lrs'][i].tolist() for i in range(2048))
    return payload,data,reference,stats
