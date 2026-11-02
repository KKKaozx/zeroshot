"""Read explicitly versioned nominal-clock targets without claiming admission.

This artifact reader never silently substitutes labels in the legacy Dataset.
Its binding must travel with any future consumer/checkpoint using these labels.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from oxe_nominal_timing import chunk, layout, protocol, protocol_sha256

FORMAT='oxe-nominal-training-target-artifacts-v1'
VALIDATION_FORMAT='oxe-nominal-validation-target-artifacts-v1'
BINDING_FORMAT='oxe-nominal-target-binding-v1'
SOURCE_SCHEMAS={'Bridge':'tfrecord_bridge_state_action','Language Table':'tfrecord_language_table_xy',
    'BC-Z':'tfrecord_bc_z_pose','Fractal':'tfrecord_rt1_pose'}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024**2),b''):h.update(part)
    return h.hexdigest()


class NominalTargetStore:
    def __init__(self,directory,*,partition='train'):
        if partition not in ('train','validation'):raise ValueError('Explicit train or validation target role required')
        self.partition=partition
        self.root=Path(directory)
        self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        m=self.manifest
        core={k:v for k,v in m.items() if k!='dataset_identity'}
        identity=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        expected_format=FORMAT if partition=='train' else VALIDATION_FORMAT
        expected_scope='frozen_training_roles_only' if partition=='train' else 'frozen_validation_roles_only'
        if (m.get('format')!=expected_format or m.get('dataset_identity')!=identity
                or m.get('scope')!=expected_scope or m.get('formal_training_ready') is not False):
            raise ValueError('Explicit nominal target identity/role required; cannot exchange train and validation')
        if m['protocol_sha256']!=protocol_sha256() or json.loads((self.root/'protocol.json').read_text())!=protocol():
            raise ValueError('Nominal timing/label semantics changed')
        if file_sha(Path(__file__).with_name('oxe_nominal_timing.py'))!=m['code_sha256']['code/oxe_nominal_timing.py']:
            raise ValueError('Nominal interpolation/command code differs from generated labels')
        for name,info in m['artifacts'].items():
            if Path(name).name!=name:raise ValueError('Artifact name must stay inside the target store')
            path=self.root/name
            if path.stat().st_size!=info['bytes'] or file_sha(path)!=info['sha256']:
                raise ValueError('Nominal target artifact content changed: '+name)
        self.episodes=json.loads((self.root/'episodes.json').read_text(encoding='utf-8'))
        self.offsets=np.load(self.root/'episode_offsets.npy',mmap_mode='r',allow_pickle=False)
        self.windows=np.load(self.root/'window_rows.npy',mmap_mode='r',allow_pickle=False)
        self.pose=np.load(self.root/'recorded_pose.npy',mmap_mode='r',allow_pickle=False)
        self.features=np.load(self.root/'current_features.npy',mmap_mode='r',allow_pickle=False)
        self.commands=np.load(self.root/'interval_commands.npy',mmap_mode='r',allow_pickle=False)
        self.actions=np.load(self.root/'actions.npy',mmap_mode='r',allow_pickle=False)
        self.masks=np.load(self.root/'supervision_masks.npy',mmap_mode='r',allow_pickle=False)
        n=len(self.windows);total=int(self.offsets[-1])
        expected=[(self.offsets,(len(self.episodes)+1,),np.int64),
            (self.windows,(n,3),np.int64),(self.pose,(total,7),np.float64),
            (self.features,(total,2),np.float32),(self.commands,(total,),np.float32),
            (self.actions,(n,16,8),np.float32),(self.masks,(n,16,8),np.uint8)]
        if not n or any(a.shape!=shape or a.dtype!=dtype for a,shape,dtype in expected):
            raise ValueError('Nominal target array shape/dtype differs')
        if self.offsets[0]!=0 or not np.all(np.diff(self.offsets)>0):raise ValueError('Invalid episode offsets')
        groups=set()
        for i,e in enumerate(self.episodes):
            if (e['source'] not in SOURCE_SCHEMAS or e['partition']!=partition
                    or e['steps']!=self.offsets[i+1]-self.offsets[i] or e['row_offset']!=self.offsets[i]):
                raise ValueError('Source observation cache role/offset differs')
            key=(e['source'],e['episode_key'])
            if key in groups:raise ValueError('Duplicate source episode')
            groups.add(key)
        self.binding={'format':BINDING_FORMAT,'dataset_identity':identity,'protocol_sha256':m['protocol_sha256'],
            'label_family':protocol()['label_family'],'clock_basis':protocol()['clock_basis'],
            'target_times_seconds':protocol()['target_times_seconds'],
            'source_units_or_formal_training_admitted':False}
        if partition!='train':self.binding['partition']=partition

    def __len__(self):return len(self.windows)

    def _row(self,index):
        if type(index) is not int or not 0<=index<len(self):raise IndexError('Real nominal training window required')
        ei,start,count=map(int,self.windows[index]);e=self.episodes[ei]
        meta=layout(e['source'],start,e['steps'])
        if count!=meta['valid_targets'] or not count:raise ValueError('Cached row layout differs')
        return ei,start,count,e,meta

    def group_key(self,index):
        _,_,_,e,_=self._row(index)
        return SOURCE_SCHEMAS[e['source']]+':'+e['group_id']

    def get(self,index):
        ei,start,count,e,meta=self._row(index)
        a=self.actions[index].copy();mask=self.masks[index].astype(np.float32)
        width=2 if e['source']=='Language Table' else 8
        expected=np.zeros((16,8),np.float32);expected[:count,:width]=1.
        if not np.isfinite(a).all() or not np.array_equal(mask,expected):
            raise ValueError('Actual target mask differs from source capabilities or nominal tail')
        if count<16 and not np.array_equal(a[count:],np.repeat(a[count-1:count],16-count,axis=0)):
            raise ValueError('Padded storage tail changed')
        return {'actions':a,'supervision_mask':mask,
            'current_features':self.features[int(self.offsets[ei])+start].copy(),
            'layout':meta,'source':e['source'],'episode_key':e['episode_key'],
            'input_row':start,'group_key':self.group_key(index),'binding':dict(self.binding)}

    def recompute(self,index):
        ei,start,_,e,_=self._row(index)
        lo,hi=map(int,self.offsets[ei:ei+2]);planar=e['source']=='Language Table'
        return chunk(e['source'],self.pose[lo:hi,:2 if planar else 3],
            None if planar else self.pose[lo:hi,3:7],None if planar else self.commands[lo:hi-1],
            self.features[lo:hi],start)

    def validate_binding(self,saved):
        if saved!=self.binding:raise ValueError('Cannot resume across raw-row, nominal-clock, or changed source targets')
        return True
