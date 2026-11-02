"""Fresh learned policy with source-only standardized command targets."""
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'code'),str(HERE/'vendor/code'),str(HERE/'vendor')]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(4*1024**2),b''):h.update(part)
    return h.hexdigest()


def dependencies():
    import os
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
    import numpy as np
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    from transformers import CLIPModel,CLIPTextModel,CLIPTokenizer,Dinov2Model
    import train,models,bcap_core,bcap_runtime,verify_command8,command8_metrics,command8_inputs,motion_scaling
    from bridge_command8_targets import encode_commands,command_chunk
    assert Path(train.__file__).resolve()==HERE/'vendor/code/train.py'
    assert not torch.cuda.is_initialized()
    return {'numpy':np.__version__,'torch':str(torch.__version__),'tensorflow':tf.__version__,
            'training_imports_passed':True,'cuda_initialized':False,'model_constructed':False,
            'online_scipy_libero_dependencies_required_only_in_local_evaluation':True}


def load_numeric():
    import numpy as np
    p=read(HERE/'protocol.json');receipt=read(HERE/'inputs_receipt.json')
    for name,key in [('inputs.npz','inputs_sha256'),('rows.json','rows_sha256'),('batches.npy','plan_sha256')]:
        if sha(HERE/name)!=receipt[key]:raise ValueError('Command input identity differs: '+name)
    with np.load(HERE/'inputs.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    n=10352+64
    if (a['targets'].shape!=(n,16,8) or a['masks'].shape!=(n,16,8) or a['current'].shape!=(n,2)
            or a['pose_features'].shape!=(n,18) or not np.isfinite(a['targets']).all()
            or not np.isin(a['masks'],[0.,1.]).all() or not np.all(a['masks'][:,0,:7]==1)):
        raise ValueError('Invalid source command labels/masks')
    valid=a['masks'][...,:7].all(-1)
    if (not np.isfinite(a['current']).all() or not np.isfinite(a['pose_features']).all()
            or not np.array_equal(a['masks'][...,:7],np.repeat(valid[...,None],7,axis=-1))
            or not np.array_equal(valid,np.arange(16)[None]<valid.sum(1)[:,None])
            or (a['masks'][...,7]>valid).any()
            or not np.isin(a['targets'][...,7][a['masks'][...,7]>0],[-1.,1.]).all()):
        raise ValueError('Noncausal/invalid source numeric inputs or target masks')
    if not np.allclose(np.linalg.norm(a['targets'][...,3:7][valid],axis=-1),1.,atol=1e-6,rtol=0):raise ValueError('Invalid target quaternion')
    if (np.abs(a['targets'][...,:3][valid])>.500001).any():raise ValueError('Source motion clip profile differs')
    rows=read(HERE/'rows.json')
    for role,count in [('train',10352),('validation',64)]:
        chosen=[r for r in rows if r['role']==role]
        if len(chosen)!=count:raise ValueError('Fixed role changed')
        expected_ids=a['train_window_indices'] if role=='train' else a['validation_window_indices']
        if [r['window_index'] for r in chosen]!=expected_ids.tolist():raise ValueError('Source row order differs')
    if not np.array_equal(a['train_window_indices'],np.load(HERE/'vendor/train_indices.npy',allow_pickle=False)):
        raise ValueError('Original training pool changed')
    plan=np.load(HERE/'batches.npy',allow_pickle=False)
    if plan.shape!=(8192,8) or plan.dtype!=np.int64 or not np.isin(plan,a['train_window_indices']).all():
        raise ValueError('Fixed training-only plan required before submission')
    if {r['group_id'] for r in rows if r['role']=='train'} & {r['group_id'] for r in rows if r['role']=='validation'}:
        raise ValueError('Source role leakage')
    return a,rows


def admit(root):
    import torch
    deps=dependencies()
    for line in (HERE/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if sha(HERE/name)!=digest:raise ValueError('Frozen package differs: '+name)
    a,rows=load_numeric();p=read(HERE/'protocol.json')
    if sha(HERE/'motion_normalizer.json')!=p['normalizer_sha256']:raise ValueError('Source-only motion normalizer identity differs')
    from motion_scaling import MotionStandardizer
    MotionStandardizer(read(HERE/'motion_normalizer.json'))
    if str(torch.__version__)!='2.5.1+cu118' or torch.version.cuda!='11.8':
        raise ValueError('Expected existing project Torch2.5.1+cu118, no environment fallback')
    import shutil,subprocess
    from command8_inputs import SourceInputs
    inputs=SourceInputs(root,a,rows)
    reserve=8*1024**3;used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0])
    if used+reserve>=200_000_000_000 or shutil.disk_usage(root).free<reserve:
        raise RuntimeError('Need8GiB output/atomic-state reserve; du estimate only, no cleanup')
    source={'passed':True,'bridge_input_files_present':len(inputs.files),
            'old_token_or_reached_pose_target_cache_required':False,
            'project_bytes_du_estimate':used,'reserve_bytes':reserve}
    for asset in (Path(root)/'baseline_setup/clip-vit-large-patch14-v1/config.json',Path(root)/'dinov2_large_frozen_assets_20261023/model.safetensors'):
        if not asset.is_file():raise FileNotFoundError('Existing frozen asset required: '+str(asset))
    return {'passed':True,'dependencies':deps,'source_inputs':source,'source_learned_checkpoint_loaded':False,'fresh_initialization_required':True,
            'effective_action_representation':p['action_representation'],
            'source_semantics_status':p['source_semantics_status'],'source_units_status':p['source_units_status'],
            'normalizer_sha256':sha(HERE/'motion_normalizer.json'),'label_rows':len(rows),'target_supervised':False,'new_model_or_training_updates':0}


def replace_command_targets(x,a,slots):
    import torch
    device=x['images'].device
    x=dict(x)
    x['actions']=torch.from_numpy(a['targets'][slots]).to(device)
    x['masks']=torch.from_numpy(a['masks'][slots]).to(device)
    if not torch.equal(x['current'],torch.from_numpy(a['current'][slots]).to(device)):raise ValueError('Causal current command input differs')
    expected_pose=torch.from_numpy(a['pose_features'][slots]).to(device)
    delta=float((x['pose']-expected_pose).abs().max())
    if not torch.isfinite(x['pose']).all() or delta>1e-6:raise ValueError('Current pose phase differs')
    x['pose_feature_abs_delta']=delta
    return x


def attach_command_targets(b,a,rows):
    """Read bound current inputs directly, without the old target-cache reader."""
    import numpy as np
    from command8_inputs import SourceInputs
    inputs=SourceInputs(b['root'],a,rows)
    from motion_scaling import MotionStandardizer,training_coordinates
    norm=MotionStandardizer(read(HERE/'motion_normalizer.json'))
    def pack(role,ids):
        x=inputs.pack(role,ids,b['tokenizer'],device='cuda')
        x=training_coordinates(x,norm)
        b['max_pose_feature_abs_delta']=max(b.get('max_pose_feature_abs_delta',0.),x['pose_feature_abs_delta'])
        return x
    b['batch']=lambda ids:pack('train',ids);b['pack']=pack


def construct_training(root,a,rows):
    import numpy as np
    import torch
    from bcap_runtime import setup,construct
    import bcap_core
    from step_training_state import UpdateSchedule
    setup();b=construct(root,{'plan_sha256':sha(HERE/'vendor/batches.npy')})
    p=read(HERE/'protocol.json')
    b['initial_identity']=bcap_core.identity(b)
    if b['optimizer'].state or b['port'].weight.count_nonzero():raise ValueError('Fresh learned policy and zero state port required')
    from motion_scaling import install_standardized_sampler
    install_standardized_sampler(b['model'],read(HERE/'motion_normalizer.json'))
    b['source_constructor_config']=copy.deepcopy(b['config'])
    b['model'].max_normalized_position=p['sampler_position_limit']
    attach_command_targets(b,a,rows)
    b['plan']=bcap_core.FixedPlan(np.load(HERE/'batches.npy',allow_pickle=False),sha(HERE/'batches.npy'))
    if b['plan'].batches.shape!=(8192,8) or not np.isin(b['plan'].batches,a['train_window_indices']).all():raise ValueError('Training-only plan differs')
    b['schedule']=UpdateSchedule(total_updates=8192,base_lrs=[1e-4]*3,kind='constant',warmup_updates=0)
    b['schedule'].apply(b['optimizer'],0)
    if b['optimizer'].state:raise ValueError('New empty AdamW required')
    b['checkpoint_every']=1024;b['training_cap_seconds']=p['training_cap_seconds']
    b['contract']={'format':'bridge-command8-fresh-scaled-training-v1','protocol_sha256':sha(HERE/'protocol.json'),
        'protocol':p,'inputs_receipt_sha256':sha(HERE/'inputs_receipt.json'),
        'source_constructor_config':b['source_constructor_config'],
        'fresh_initial_identity':b['initial_identity'],'motion_normalizer_sha256':sha(HERE/'motion_normalizer.json'),'motion_normalizer':read(HERE/'motion_normalizer.json'),'source_stats':b['stats'],
        'effective_action_representation':p['action_representation'],'sampler_position_limit':p['sampler_position_limit'],
        'constructor_config_is_architecture_restore_template_not_new_label_definition':True,
        'frozen_state_sha256':read(HERE/'vendor/reference.json')['frozen_state_sha256'],
        'optimizer_initialization':'fresh_seed42_learned_weights_and_new_empty_AdamW'}
    return b
