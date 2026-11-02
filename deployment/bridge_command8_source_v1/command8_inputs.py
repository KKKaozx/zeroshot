"""Bound Bridge image/text reader; never reads old reached-pose targets."""
import hashlib
import io
from pathlib import Path,PurePosixPath

import numpy as np
import torch


class SourceInputs:
    def __init__(self,root,a,rows):
        self.a,self.rows=a,rows
        self.lookup={(r['role'],r['window_index']):i for i,r in enumerate(rows)}
        if len(self.lookup)!=len(rows):raise ValueError('Duplicate input row')
        self.files={};base=(Path(root)/'data/oxe_nominal_pool_v1').resolve()
        for r in rows:
            relative=r['stage_relative_path'];logical=PurePosixPath(relative)
            if logical.is_absolute() or '..' in logical.parts or ':' in relative or not relative.startswith('bridge_v2/'):
                raise ValueError('Bridge input must stay within its declared source root')
            path=(base/relative).resolve()
            if not path.is_relative_to(base) or not path.is_file():raise FileNotFoundError('Existing Bridge record file required: '+str(path))
            if not (0<=r['input_row']<r['steps']-1) or r['payload_bytes']>32*1024**2:raise ValueError('Invalid source row/payload limit')
            self.files[relative]=path

    def pack(self,role,ids,tokenizer,*,device):
        import tensorflow as tf
        from PIL import Image
        from dataset import prepare_image
        from command8_runtime import replace_command_targets
        slots=np.array([self.lookup[(role,int(i))] for i in ids],np.int64)
        texts=[];pixels=[]
        for slot in slots:
            row=self.rows[int(slot)]
            with self.files[row['stage_relative_path']].open('rb') as f:
                f.seek(row['payload_offset']);raw=f.read(row['payload_bytes'])
            if hashlib.sha256(raw).hexdigest()!=row['payload_sha256']:raise ValueError('Source image/text record identity changed')
            fields=tf.train.Example.FromString(raw).features.feature
            images=fields['steps/observation/image_0'].bytes_list.value
            language=fields['steps/language_instruction'].bytes_list.value
            if len(images)!=row['steps'] or len(language)!=row['steps']:raise ValueError('Source image/text phase count differs')
            index=row['input_row'];text=language[index].decode('utf-8')
            if not text.strip():raise ValueError('Empty source instruction')
            with Image.open(io.BytesIO(images[index])) as image:rgb=np.asarray(image.convert('RGB'))
            texts.append(text);pixels.append(prepare_image(rgb))
        tokens=tokenizer(texts,padding='max_length',max_length=77,truncation=True,return_tensors='pt')
        x={'images':torch.stack(pixels).to(device),'tokens':{k:v.to(device) for k,v in tokens.items()},
           'current':torch.from_numpy(self.a['current'][slots]).to(device),
           'pose':torch.from_numpy(self.a['pose_features'][slots]).to(device)}
        return replace_command_targets(x,self.a,slots)
