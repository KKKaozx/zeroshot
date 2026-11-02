"""Six training-only GPU updates after the frozen physical preparation gate.

No formal fitting, held-out model evaluation, checkpoint writes or downloads.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'cluster_packages/bridge_growth128_v1'
DATA=ROOT/'reports/panda_minimal_generalization_preparation_20261015'
OUT=ROOT/'reports/panda_minimal_gpu_preflight_20261015'
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from transformers import CLIPTokenizer
from dataset import relative_pose_action,prepare_image
from models import RobotAdapterModel
from train import policy_loss


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def module_sha(model,prefixes):
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):
        if n.startswith(prefixes):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def examples(split):
    protocol,collection=read(DATA/'protocol.json'),read(DATA/'collection.json')
    assert split in ['train','development'] and collection['passed']
    assert collection['protocol_sha256']==sha(DATA/'protocol.json')
    receipts={r['episode']:r for r in collection['expert']}
    images,targets,currents,rows=[],[],[],[]
    for row in protocol['episodes']:
        if row['split']!=split:continue
        path=DATA/'episodes'/(row['episode']+'.npz')
        assert sha(path)==receipts[row['episode']]['npz_sha256']
        with np.load(path) as d:
            assert len(d['images'])==len(d['positions'])==len(d['commands'])+1==103
            assert np.allclose(np.diff(d['times_seconds']),.1,atol=1e-12,rtol=0)
            for t in protocol['starts']:
                images.append(prepare_image(d['images'][t]))
                targets.append(np.stack([relative_pose_action(d['positions'][t],d['rotations'][t],
                    d['positions'][j],d['rotations'][j],d['commands'][j-1]) for j in range(t+1,t+17)]))
                currents.append([d['currents'][t]])
                rows.append(dict(episode=row['episode'],start=t,split=split))
    # Object XY, heights and contacts are not returned as policy features.
    return torch.stack(images),torch.from_numpy(np.stack(targets)),torch.tensor(currents,dtype=torch.float32),rows


def main():
    OUT.mkdir(exist_ok=False)
    assert torch.cuda.is_available()
    frozen=read(BASE/'manifest.json')['source_sha256']
    for name in ['models.py','adapter.py','diffusion_decoder.py','dataset.py']:
        assert sha(ROOT/'code'/name)==frozen[name],name+' changed from reviewed model'
    source={n:sha(ROOT/'code'/n) for n in ['models.py','adapter.py','diffusion_decoder.py','dataset.py','train.py']}
    source['scripts/preflight_panda_minimal_generalization_20261015.py']=sha(Path(__file__))
    images,targets,currents,rows=examples('train')
    assert len(rows)==128 and len({r['episode'] for r in rows})==16
    ix=[i*8+i for i in range(8)]
    selected=[rows[i] for i in ix]
    assert len({r['episode'] for r in selected})==8
    assert set(targets[ix,:,7].flatten().tolist())=={-1.,1.}
    config=copy.deepcopy(read(BASE/'manifest.json')['config'])
    config['model']['current_gripper_encoding']='panda_measured_finger_width_affine_v1'
    assert Path(config['model']['name']).is_dir(),'Use existing local CLIP assets; do not download'
    (OUT/'plan.json').write_text(json.dumps(dict(config=config,selection=selected,source_sha256=source,
        protocol_sha256=sha(DATA/'protocol.json'),training_only=True,updates=6),indent=2)+'\n')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.manual_seed(43);torch.cuda.manual_seed_all(43)
    model=RobotAdapterModel(config).cuda().eval()
    tokenizer=CLIPTokenizer.from_pretrained(config['model']['name'],local_files_only=True)
    texts=tokenizer([read(DATA/'protocol.json')['instruction']]*8,padding=True,return_tensors='pt')
    texts={k:v.cuda() for k,v in texts.items()}
    with torch.no_grad():
        visual=torch.cat([model.vision_encoder(pixel_values=images[ix[i:i+2]].cuda()).last_hidden_state for i in range(0,8,2)]).detach()
        language=model.text_encoder(**texts).last_hidden_state.detach()
    model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
    groups=dict(clip=('vision_encoder.','text_encoder.'),adapter=('adapter.',),decoder=('diffusion_decoder.',),
        gripper=('gripper_','current_gripper_projection.'))
    before={k:module_sha(model,p) for k,p in groups.items()}
    trainable=[p for n,p in model.named_parameters() if not n.startswith(groups['clip'])]
    assert all(not p.requires_grad for p in model.vision_encoder.parameters())
    optimizer=torch.optim.Adam(trainable,lr=1e-4)
    actions=targets[ix].cuda();current=currents[ix].cuda();history=[]
    torch.cuda.reset_peak_memory_stats();first_grad={}
    for step in range(1,7):
        model.train();assert not model.vision_encoder.training and not model.text_encoder.training
        torch.manual_seed(430000+step);torch.cuda.manual_seed_all(430000+step)
        torch.cuda.synchronize();start=time.monotonic()
        context=model.adapter(visual,language,texts['attention_mask'])
        torch.manual_seed(420000+step);torch.cuda.manual_seed_all(420000+step)
        output=model.diffusion_loss(actions,context,current)
        loss,pose,grip=policy_loss(model,output,actions,torch.nn.MSELoss(),current_grippers=current)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if step==1:
            first_grad={k:any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in model.named_parameters() if n.startswith(prefix)) for k,prefix in groups.items()}
            assert first_grad==dict(clip=False,adapter=True,decoder=True,gripper=True)
        torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize()
        history.append(dict(step=step,loss=float(loss.detach()),pose=float(pose.detach()),gripper=float(grip.detach()),seconds=time.monotonic()-start))
        print('PANDA PREFLIGHT',history[-1],flush=True)
    after={k:module_sha(model,p) for k,p in groups.items()}
    changed={k:before[k]!=after[k] for k in groups}
    assert changed==dict(clip=False,adapter=True,decoder=True,gripper=True)
    model.eval()
    with torch.no_grad():
        context=model.adapter(visual[:2],language[:2],texts['attention_mask'][:2])
        torch.manual_seed(1101);torch.cuda.manual_seed_all(1101)
        predicted=model.sample(context,current[:2])
    assert predicted.shape==(2,16,8) and torch.isfinite(predicted).all()
    assert torch.allclose(predicted[...,3:7].norm(dim=-1),torch.ones((2,16),device='cuda'),atol=1e-5)
    for name,h in source.items():
        path=ROOT/name if name.startswith('scripts/') else ROOT/'code'/name
        assert sha(path)==h
    report=dict(passed=True,stage='six_update_resource_and_gradient_preflight',formal_fit_executed=False,
        updates=6,source_sha256=source,collection_report_sha256=sha(DATA/'collection.json'),
        train_episodes=16,train_windows=128,selected_training_windows=selected,development_model_evaluated=False,
        test_read=False,clip_unchanged=not changed['clip'],changed=changed,first_step_gradient=first_grad,
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,batch_size=8,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        mean_warm_update_seconds=float(np.mean([r['seconds'] for r in history[1:]])),
        sample_finite=True,checkpoint_written=False,history=history)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PANDA GPU PREFLIGHT PASSED; this is not fitting or generalization success',flush=True)


if __name__=='__main__':main()
