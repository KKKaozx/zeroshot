"""Bind the completed current OXE endpoint, without changing passed bundles."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def attach(root):
    formal=root/'oxe_current_pose_formal_v1';sys.path.insert(0,str(formal))
    import binding
    assert binding.HERE==formal.resolve()
    pg=binding.attach(root)
    return binding,pg


def admit(root,space=True):
    p=read(HERE/'protocol.json')
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(HERE/name)==expected,name
    for name,expected in p['formal_bundle_files'].items():assert sha(root/'oxe_current_pose_formal_v1'/name)==expected,name
    binding,pg=attach(root);binding.admit(root,space=False)
    source=root/'runs/oxe-pose-pair-232587';arm=read(source/'real_pose/arm_receipt.json')
    for key,expected in p['source_identity'].items():assert arm[key]==expected,key
    assert arm['passed'] and arm['training_jobs']==['232587']
    assert read(source/'pair_report.json')['execution_complete']
    evaluation=source/'evaluation';r=read(evaluation/'report.json')
    assert r['execution_complete'] and r['optimizer_updates']==0 and r['generation_steps']==100
    bound={'format':'fixed8-existing-endpoint-binding-v1','protocol_sha256':sha(HERE/'protocol.json'),
        'source_arm_receipt_sha256':sha(source/'real_pose/arm_receipt.json'),
        'source_evaluation_report_sha256':sha(evaluation/'report.json'),
        'source_predictions_sha256':sha(evaluation/'predictions_and_targets.npz')}
    path=HERE/'source_binding.json'
    if path.exists():assert read(path)==bound
    else:path.write_bytes((json.dumps(bound,indent=2)+'\n').encode())
    if space:
        used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0]);reserve=2*1024**3
        assert used+reserve<200_000_000_000 and shutil.disk_usage(root).free>reserve,'Need2GiB for diagnostic policy and temporary write'
    return bound


def load_source(root):
    """Actual original data admission and frozen constructor; restore all three Adam groups."""
    import torch
    binding,pg=attach(root)
    from runtime_inputs import load
    import train as api
    import models
    from completed_policy_export import contract_sha,model_sha
    from step_training_runner import frozen_digest,tokenizer_identity
    from step_training_state import UpdateSchedule
    from resume_constants import training_constant_constructor
    from pose_runtime import zero_port,check_moments,digest
    original,data,reference,stats=load(root,pg)
    path=root/'runs/oxe-pose-pair-232587/real_pose/latest_state.pt'
    source=torch.load(path,map_location='cpu',weights_only=False,mmap=True);p=read(HERE/'protocol.json');events=[]
    c=source['contract'];assert source['format']=='current-pose-bounded-formal-state-v1'
    assert contract_sha(c)==source['contract_sha256']==p['source_identity']['contract_sha256']
    assert source['plan']['completed_updates']==10240 and c['role']=='real_pose'
    config=c['base_config_unchanged'];assert config==original['contract']['config'] and config['model']['diffusion_prediction_type']=='sample'
    tokenizer=api.CLIPTokenizer.from_pretrained(config['model']['name'],local_files_only=True)
    assert tokenizer_identity(tokenizer)==original['contract']['tokenizer']
    with training_constant_constructor(models.RobotAdapterModel,contract=original['contract'],
            profile_path=root/'oxe_target_pair_short_v1/profile_sample.json',learned_state=api.trainable_state_dict,on_restored=events.append):
        model=models.RobotAdapterModel(config).cuda()
    expected=api.trainable_state_dict(model);assert set(expected)==set(source['base'])
    loaded=model.load_state_dict(source['base'],strict=False)
    assert not loaded.unexpected_keys and not set(expected).intersection(loaded.missing_keys)
    port=zero_port('cuda',width=1024);port.load_state_dict(source['pose_port'],strict=True)
    assert model_sha(api.trainable_state_dict(model))==p['source_identity']['base_learned_sha256']
    assert digest(port.state_dict())==p['source_identity']['port_sha256']
    assert frozen_digest(model,expected)==p['source_identity']['frozen_state_sha256']
    named=[(n,t) for n,t in model.named_parameters() if t.requires_grad]
    groups=[[t for n,t in named if not n.startswith(api.GRIPPER_PARAMETER_PREFIXES)],
        [t for n,t in named if n.startswith(api.GRIPPER_PARAMETER_PREFIXES)],list(port.parameters())]
    optimizer=torch.optim.AdamW([{'params':g,'lr':old['lr']} for g,old in zip(groups,source['optimizer']['param_groups'])],
        weight_decay=original['contract']['arguments']['weight_decay'])
    optimizer.load_state_dict(source['optimizer']);check_moments(optimizer)
    assert digest(optimizer.state_dict())==digest(source['optimizer'])
    schedule=UpdateSchedule(**source['schedule']);assert schedule.contract['total_updates']==65536
    assert [g['lr'] for g in optimizer.param_groups]==schedule.rates(10240)
    import numpy as np
    with np.load(HERE/'reference_lrs.npz') as z:
        assert all(schedule.rates(10240+i)==z['lrs'][i].tolist() for i in range(p['updates']))
    from instrument import method_source
    from pose_runtime import NoiseLog
    method_path,method=method_source()
    assert hashlib.sha256(method.encode()).hexdigest()==original['contract']['diagnostic_target_pair']['diffusion_loss_source_sha256']
    noise=NoiseLog();models.__dict__['DIAG_HOOK']=noise
    exec(compile(method,str(method_path),'exec'),models.__dict__);models.RobotAdapterModel.diffusion_loss=models.__dict__['diffusion_loss']
    return {'model':model,'port':port,'optimizer':optimizer,'schedule':schedule,'source':source,'data':data,
        'tokenizer':tokenizer,'api':api,'stats':stats,'keys':expected,'noise':noise,'frozen_restore_events':events,
        'source_path':path,'source_file_size_mtime':[path.stat().st_size,path.stat().st_mtime_ns],
        'dataset_binding':data.nominal_target_binding,'config':config,'tokenizer_identity':original['contract']['tokenizer']}
