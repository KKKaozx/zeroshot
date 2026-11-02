"""Isolated frozen-source runtime. No old learned checkpoint is opened."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def fit_stats(store,indices):
    from pose_features import current_rows,raw_features,array_sha
    assert store.partition=='train','State statistics require training role only'
    pose,sources=current_rows(store,indices);assert set(sources)=={'Bridge'}
    values,mask=raw_features(pose,sources);assert (mask==1).all()
    return {'xyz_mean':values[:,:3].mean(0).tolist(),'xyz_std':np.maximum(values[:,:3].std(0),1e-6).tolist(),
        'fit_partition':'train','fit_source':'Bridge','fit_windows':len(indices),'raw_rows_sha256':array_sha(pose)}


def admit(root,*,space=True):
    """CPU qualification: all immutable numeric labels/selection/plan/sources."""
    import shutil
    import subprocess
    from oxe_nominal_target_store import NominalTargetStore
    p=read(HERE/'protocol.json');ref=read(HERE/'reference.json')
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(HERE/name)==expected,name
    for name,expected in ref['source_code_sha256'].items():assert sha(HERE/'code'/name)==expected,name
    assert str(torch.__version__)==p['torch'] and torch.version.cuda==p['cuda']
    assert not torch.cuda.is_initialized(),'check-only must not initialize CUDA'
    stores={role:NominalTargetStore(root/f'oxe_dino_main_preflight_v1/inputs/{role}',partition=role) for role in ['train','validation']}
    for role,s in stores.items():assert s.binding==ref['config']['action']['nominal_target_binding']['partition_bindings'][role]
    train=stores['train'];ids=np.load(HERE/'train_indices.npy',allow_pickle=False)
    actual=np.array([i for i,e in enumerate(train.windows[:,0]) if train.episodes[int(e)]['source']=='Bridge'],dtype=np.int64)
    assert np.array_equal(ids,actual) and len(ids)==10352
    assert len({int(e) for e in train.windows[ids,0]})==1151
    batches=np.load(HERE/'batches.npy',allow_pickle=False,mmap_mode='r')
    assert batches.shape==(65536,8) and batches.dtype==np.int64 and np.isin(batches,ids).all()
    assert fit_stats(train,ids.tolist())==read(HERE/'state_stats.json')
    selection=read(HERE/'selection.json');assert len(selection['rows'])==192
    for role,count in [('train',128),('validation',64)]:
        rows=[r for r in selection['rows'] if r['role']==role];s=stores[role]
        assert len(rows)==count and len({r['group_id'] for r in rows})==count
        for r in rows:
            _,start,_,e,_=s._row(r['window_index'])
            assert r['source']=='Bridge' and int(start)==r['input_row']
            for k in ['source','group_id','episode_key','payload_sha256']:assert e[k]==r[k]
    with np.load(HERE/'evaluation_reference.npz',allow_pickle=False) as z:
        for i,r in enumerate(selection['rows']):
            v=stores[r['role']].get(r['window_index'])
            assert np.array_equal(z['targets'][i],v['actions']) and np.array_equal(z['masks'][i],v['supervision_mask'])
    tg={e['group_id'] for e in train.episodes if e['source']=='Bridge'}
    vg={e['group_id'] for e in stores['validation'].episodes if e['source']=='Bridge'}
    assert not tg&vg
    files={}
    for s in stores.values():
        for e in s.episodes:
            if e['source']=='Bridge':files[e['stage_relative_path']]=root/'data/oxe_nominal_pool_v1'/e['stage_relative_path']
    assert all(f.is_file() for f in files.values()),'Bridge raw image records missing'
    info={'passed':True,'protocol_sha256':sha(HERE/'protocol.json'),'plan_sha256':sha(HERE/'batches.npy'),
        'bridge_train_windows':len(ids),'bridge_train_episodes':1151,'evaluation_windows':192,'model_constructed':False,
        'optimizer_updates':0,'cuda_initialized':False,'source_learned_checkpoints_read':False}
    if space:
        used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0]);reserve=8*1024**3
        assert used+reserve<200_000_000_000 and shutil.disk_usage(root).free>reserve,'Need8GiB state/atomic-save reserve'
        info.update(project_allocated_bytes=used,reserve_bytes=reserve,quota_scope='du estimate; not authoritative user quota')
    return info


def setup():
    import os
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1);torch.set_num_interop_threads(48);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA A40'


def construct(root,admission):
    import train as api
    import models
    from transformers import CLIPTokenizer
    from step_training_runner import frozen_digest,tokenizer_identity
    from step_training_state import UpdateSchedule
    from resume_constants import training_constant_constructor
    from pose_runtime import zero_port
    from bcap_core import FixedPlan
    assert Path(api.__file__).resolve()==(HERE/'code/train.py').resolve()
    assert Path(models.__file__).resolve()==(HERE/'code/models.py').resolve()
    ref=read(HERE/'reference.json');p=read(HERE/'protocol.json');template=copy.deepcopy(ref)
    cfg=template['config'];cfg['model']['diffusion_prediction_type']='sample'
    cfg['model']['name']=str(root/'baseline_setup/clip-vit-large-patch14-v1')
    cfg['model']['dino_assets']=str(root/'dinov2_large_frozen_assets_20261023')
    api.set_seed(42)
    tokenizer=CLIPTokenizer.from_pretrained(cfg['model']['name'],local_files_only=True)
    assert tokenizer_identity(tokenizer)==ref['tokenizer']
    events=[]
    with training_constant_constructor(models.RobotAdapterModel,contract=template,profile_path=HERE/'profile_sample.json',
            learned_state=api.trainable_state_dict,on_restored=events.append):
        model=models.RobotAdapterModel(cfg).cuda()
    assert len(events)==1 and model.diffusion_prediction_type=='sample'
    port=zero_port('cuda');named=[(n,t) for n,t in model.named_parameters() if t.requires_grad]
    groups=[[t for n,t in named if not n.startswith(api.GRIPPER_PARAMETER_PREFIXES)],
        [t for n,t in named if n.startswith(api.GRIPPER_PARAMETER_PREFIXES)],list(port.parameters())]
    assert all(groups)
    opt=torch.optim.AdamW([{'params':g,'lr':2e-4} for g in groups],weight_decay=ref['weight_decay'])
    assert not opt.state and not port.weight.count_nonzero()
    schedule=UpdateSchedule(total_updates=65536,base_lrs=[2e-4]*3,kind='cosine',warmup_updates=1024);schedule.apply(opt,0)
    plan=FixedPlan(np.load(HERE/'batches.npy',allow_pickle=False,mmap_mode='r'),sha(HERE/'batches.npy'))
    stats=read(HERE/'state_stats.json');contract={'format':'fresh-bridge-full-capability-v1','protocol_sha256':sha(HERE/'protocol.json'),
        'plan_sha256':admission['plan_sha256'],'config':cfg,'stats':stats,'source_frozen_sha256':ref['frozen_state_sha256'],
        'tokenizer':ref['tokenizer'],'training_seed':42,'initialization':'fresh_learned_zero_pose_port_empty_adam',
        'weight_decay':ref['weight_decay'],'single_source_supervision':'Bridge','no_augmentation':True}
    keys=api.trainable_state_dict(model)
    def check_frozen():assert frozen_digest(model,keys)==ref['frozen_state_sha256']
    check_frozen()
    b={'root':root,'api':api,'model':model,'port':port,'optimizer':opt,'schedule':schedule,'plan':plan,
        'tokenizer':tokenizer,'stats':stats,'contract':contract,'config':cfg,'cuda':True,
        'checkpoint_every':4096,'training_cap_seconds':p['training_cap_seconds'],'check_frozen':check_frozen,
        'frozen_restore_events':events}
    return b


def attach_data(b):
    from oxe_nominal_dataset import NominalOXEDataset
    from pose_features import batch_features
    root=b['root'];data={r:NominalOXEDataset(root/'data/oxe_nominal_pool_v1',root/f'oxe_dino_main_preflight_v1/inputs/{r}',partition=r) for r in ['train','validation']}
    def pack(role,ids):
        d=data[role]
        assert all(d.store.episodes[int(d.store.windows[i,0])]['source']=='Bridge' for i in ids)
        texts,images,current,actions,masks=b['api'].collate_batch([d[i] for i in ids])
        tokens=b['tokenizer'](texts,padding='max_length',max_length=77,truncation=True,return_tensors='pt')
        assert tokens['input_ids'].shape==(len(ids),77)
        pose=batch_features(d.store,ids,b['stats'],'real_pose')
        return {'images':images.cuda(),'current':current.cuda(),'actions':actions.cuda(),'masks':masks.cuda(),
            'pose':torch.from_numpy(pose).cuda(),'tokens':{k:v.cuda() for k,v in tokens.items()}}
    b['batch']=lambda ids:pack('train',ids);b['pack']=pack;b['data']=data


def evaluate(b):
    from step_training_state import capture_rng,restore_rng
    model=b['model'];port=b['port'];model.eval();port.eval();rows=read(HERE/'selection.json')['rows']
    rng=capture_rng(include_cuda=True);draws=[];targets=[];masks=[]
    try:
        with torch.inference_mode():
            for seed in [1101,1102,1103]:
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);pred=[]
                for lo in range(0,len(rows),8):
                    chunk=rows[lo:lo+8];assert len({r['role'] for r in chunk})==1
                    x=b['pack'](chunk[0]['role'],[r['window_index'] for r in chunk])
                    c=model.get_context_vector(x['images'],x['tokens']['input_ids'],x['tokens']['attention_mask'])+port(x['pose'])
                    pred.append(model.sample(c,x['current']).cpu().numpy())
                    if seed==1101:targets.append(x['actions'].cpu().numpy());masks.append(x['masks'].cpu().numpy())
                draws.append(np.concatenate(pred))
                print('BRIDGE_CAPABILITY_GENERATION',seed,len(rows),flush=True)
    finally:restore_rng(rng)
    b['check_frozen']()
    return np.stack(draws),np.concatenate(targets),np.concatenate(masks)
