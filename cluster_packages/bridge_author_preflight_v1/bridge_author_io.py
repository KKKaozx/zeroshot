"""Small, dependency-free Bridge-to-author data contract; no future observations."""
import hashlib
import json
from pathlib import Path
import numpy as np

MEAN=np.array([.48145466,.4578275,.40821073],dtype=np.float32)[None,:,None,None]
STD=np.array([.26862954,.26130258,.27577711],dtype=np.float32)[None,:,None,None]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def canonical(s):return ' '.join(s.lower().split())
def rgb_from_clip(images):
    images=np.asarray(images,dtype=np.float32)
    assert images.shape==(168,3,224,224) and np.isfinite(images).all()
    rgb=images*STD+MEAN
    assert rgb.min()>=-1e-6 and rgb.max()<=1+1e-6
    rgb=np.clip(rgb,0,1).astype('<f4') # Preserve floating interpolation and letterbox, no uint8 rounding.
    delta=float(np.abs((rgb-MEAN)/STD-images).max())
    assert delta<=1e-6
    return rgb,delta
def assemble_references(project):
    project=Path(project)
    paths=[project/'bridge_batch_input_audit_v1/reference/arrays.npz',
           project/'bridge_growth64_v1/added_reference/arrays.npz',
           project/'bridge_growth128_v1/new_reference/arrays.npz']
    arrays=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in ('images','targets')})
    return {k:np.concatenate([arrays[0][k][:32],arrays[1][k],arrays[2][k],arrays[0][k][32:]],axis=0) for k in ('images','targets')}
def load_cache(directory,protocol):
    directory=Path(directory);r=read(directory/'export_report.json')
    assert r['passed'] and r['protocol_sha256']==sha(protocol)
    assert r['rows']==read(protocol)['rows'] and r['train_windows']==128
    values={}
    for key,h in r['array_sha256'].items():
        path=directory/(key+'.npy');assert sha(path)==h
        values[key]=np.load(path,mmap_mode='r',allow_pickle=False)
    for key,shape in {'image':(168,3,224,224),'ee_state':(168,9),'language':(168,768),'action':(168,16,7)}.items():
        assert values[key].shape==shape and values[key].dtype==np.float32 and np.isfinite(values[key]).all()
    return values,r
