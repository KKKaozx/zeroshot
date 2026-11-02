"""离线检查第2版：A3在Bridge开发集画面上看不看画面？在多个不同LIBERO场景上呢？

相对第1版的改进：
  Bridge  用A3真正的64个开发窗口(E:\\dataset 完整Bridge v2，按rows.json偏移读取并核对sha256)，各自真实指令
  LIBERO  64个不同任务的初始画面(各任务 demo_0 第0帧观测，上下翻转为正立；不读任何动作或后续帧)，各自指令
不仿真、不训练、不拟合任何LIBERO统计。

两边数值输入完全相同(Bridge训练集pose18逐维中位数, current2=[0,0], 历史掩码0)，只有画面和指令不同：
  own     各用自己的指令
  shared  全部用同一句指令，只剩画面在变
指标：首航点在不同画面间的散布(米)、条件向量散布、DINO CLS余弦相似度、首航点平均方向。
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import online_policy as op  # noqa: E402

OUT = ROOT / 'reports/patch_readout_libero_engineering_20261102/image_check_v2.json'
C8 = ROOT / 'deployment/bridge_command8_fresh_scaled_v1'
BRIDGE = Path('E:/dataset/bridge_v2_0.0.1/0.0.1')
LIBERO = ROOT.parent / 'dataset/LIBERO'
CLIP = ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
DINO = ROOT / 'baselines/dinov2_large_frozen_assets_20261023'
CHECKPOINT = ROOT / 'reports/patch_readout_239393/A3_policy/policy.pt'
MAPPING = ROOT / 'configs/libero_bridge_pose_mapping_uncalibrated_engineering_20261030.json'
N_LIBERO = 64


def bridge_dev():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    rows = json.loads((C8 / 'rows.json').read_text(encoding='utf-8'))
    out = []
    for r in rows:
        if r['role'] != 'validation':
            continue
        with (BRIDGE / r['stage_relative_path'].split('/')[-1]).open('rb') as f:
            f.seek(r['payload_offset'])
            raw = f.read(r['payload_bytes'])
        if hashlib.sha256(raw).hexdigest() != r['payload_sha256']:
            raise ValueError('Bridge record identity differs: ' + r['episode_key'])
        fields = tf.train.Example.FromString(raw).features.feature
        t = int(r['input_row'])
        with Image.open(io.BytesIO(fields['steps/observation/image_0'].bytes_list.value[t])) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        out.append({'rgb': rgb, 'text': fields['steps/language_instruction'].bytes_list.value[t].decode('utf-8'),
                    'id': r['episode_key']})
    if len(out) != 64:
        raise ValueError('Expected the 64 A3 development windows')
    return out


def libero_initial():
    files = sorted(LIBERO.glob('*/*_demo.hdf5'))
    pick = [files[i] for i in np.linspace(0, len(files) - 1, N_LIBERO).round().astype(int)]
    out = []
    for p in pick:
        with h5py.File(p, 'r') as f:
            rgb = f['data/demo_0/obs/agentview_rgb'][0][::-1].copy()     # OpenGL存储，翻转为正立
            text = json.loads(f['data'].attrs['problem_info'])['language_instruction']
        out.append({'rgb': rgb, 'text': text, 'id': p.parent.name + '/' + p.stem})
    return out


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; no overwrite')
    bridge, libero = bridge_dev(), libero_initial()
    policy = op.load_policy(CHECKPOINT, clip_dir=CLIP, dino_dir=DINO,
                            mapping=json.loads(MAPPING.read_text(encoding='utf-8')), device='cuda', sampling_seed=1101)
    dev = policy.device
    pose = np.median(np.load(C8 / 'inputs.npz')['pose_features'][:10352], axis=0).astype(np.float32)
    numeric = {'current2': torch.zeros(1, 2, device=dev), 'pose18': torch.from_numpy(pose)[None].to(dev),
               'state_delta': torch.zeros(1, 6, device=dev), 'state_delta_mask': torch.zeros(1, device=dev),
               'prev_command': torch.zeros(1, 7, device=dev), 'prev_command_mask': torch.zeros(1, device=dev)}
    shared = bridge[0]['text']

    def run(samples, fixed_text=None):
        cls, ctx, first = [], [], []
        for s in samples:
            text, mask = policy._text(fixed_text or s['text'])
            vis = policy._visual(s['rgb'])
            batch = dict(numeric, visual=vis, text=text, text_mask=mask)
            with torch.no_grad():
                cls.append(vis[0, 0].cpu().numpy())
                ctx.append(policy.model.context(batch)[0].cpu().numpy())
                g = torch.Generator(device=dev).manual_seed(2024)
                src = policy.model.predict8(batch, g, 'ddim', draws=8, average=True)[0, 0].cpu().numpy()
            first.append(src[0, :3] * op.POSITION_DIVISOR)
        return np.array(cls), np.array(ctx), np.array(first)

    def spread(x):
        return float(np.linalg.norm(x - x.mean(0), axis=1).mean())

    def unit(a):
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    out = {'bridge_dev_windows': len(bridge), 'libero_tasks': [s['id'] for s in libero], 'shared_instruction': shared,
           'numeric_inputs': 'Bridge train median pose18, current2=[0,0], history masked (identical for both sides)'}
    feats = {}
    for mode in ('own', 'shared'):
        res = {}
        for side, samples in (('bridge', bridge), ('libero', libero)):
            cls, ctx, first = run(samples, shared if mode == 'shared' else None)
            feats[side] = cls
            res[side] = {'first_waypoint_spread_m': spread(first), 'context_spread': spread(ctx),
                         'mean_first_waypoint_m': first.mean(0).round(4).tolist(),
                         'mean_step_m': float(np.linalg.norm(first, axis=1).mean()),
                         'fraction_up_and_plus_y': float(np.mean((first[:, 1] > 0) & (first[:, 2] > 0)))}
        out[mode] = res
    bb, ll, bl = (unit(feats['bridge']) @ unit(feats['bridge']).T, unit(feats['libero']) @ unit(feats['libero']).T,
                  unit(feats['bridge']) @ unit(feats['libero']).T)
    off = lambda m: float(m[~np.eye(len(m), dtype=bool)].mean())
    out['dino_cls_cosine'] = {'bridge_bridge_mean': off(bb), 'libero_libero_mean': off(ll),
                              'bridge_libero_mean': float(bl.mean()),
                              'bridge_nearest_other_bridge_mean': float(np.sort(bb, 1)[:, -2].mean()),
                              'libero_nearest_other_libero_mean': float(np.sort(ll, 1)[:, -2].mean()),
                              'libero_nearest_bridge_mean': float(bl.max(0).mean())}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'libero_tasks'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
