"""PLAN.md 第2关：LIBERO离线检查（本机，不仿真、不训练、不拟合任何LIBERO统计）。

画面与指令与 v1 offline_image_check_v2 完全相同：
  Bridge  A3/B 共用的64个开发窗口(E:\\dataset 完整Bridge v2，核对sha256)，各自真实指令
  LIBERO  64个不同任务 demo_0 第0帧(翻转为正立)，各自指令；不读动作或后续帧
第一次调用的情形：历史掩码0。数值输入在 Bridge 与 LIBERO 两边相同：
  neutral   current2=[0,0], pose18=Bridge训练集中位数（v1 image_check_v2 的设定）
  open      current2=[1,1]（run1 第一次调用时的夹爪状态），pose18 同上
B 配置不读 current2/pose18，两种设定结果应相同；A3 作为对照(B0)。

预先固定的判定(PLAN.md 第5节)：
  1. 看画面：LIBERO 64场景首航点散布 >= 同模型 Bridge 开发集散布的 70%
  2. 不漂移：LIBERO 64场景首航点平均向量长度 <= 首航点平均长度的 0.5
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

OUT = ROOT / 'reports/patch_readout_v2_241155/gate2_offline.json'
C8 = ROOT / 'deployment/bridge_command8_fresh_scaled_v1'
BRIDGE = Path('E:/dataset/bridge_v2_0.0.1/0.0.1')
LIBERO = ROOT.parent / 'dataset/LIBERO'
CLIP = ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
DINO = ROOT / 'baselines/dinov2_large_frozen_assets_20261023'
MAPPING = ROOT / 'configs/libero_bridge_pose_mapping_uncalibrated_engineering_20261030.json'
MODELS = {'B0_A3': ROOT / 'reports/patch_readout_239393/A3_policy/policy.pt',
          'B3_vision_delta_aug': ROOT / 'reports/patch_readout_v2_241155/B3_policy/policy.pt'}
N_LIBERO = 64
SPREAD_MIN, DRIFT_MAX = 0.7, 0.5


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
        out.append({'rgb': rgb, 'text': fields['steps/language_instruction'].bytes_list.value[t].decode('utf-8')})
    if len(out) != 64:
        raise ValueError('Expected the 64 development windows')
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


def first_waypoints(policy, samples, numeric):
    dev = policy.device
    first = []
    for s in samples:
        text, mask = policy._text(s['text'])
        batch = dict(numeric, visual=policy._visual(s['rgb']), text=text, text_mask=mask)
        with torch.no_grad():
            g = torch.Generator(device=dev).manual_seed(2024)
            src = policy.model.predict8(batch, g, 'ddim', draws=8, average=True)[0, 0].cpu().numpy()
        first.append(src[0, :3] * op.POSITION_DIVISOR)
    return np.array(first)


def stats(first):
    length = float(np.linalg.norm(first, axis=1).mean())
    mean = first.mean(0)
    return {'spread_m': float(np.linalg.norm(first - mean, axis=1).mean()), 'mean_length_m': length,
            'mean_vector_m': mean.round(4).tolist(), 'mean_vector_length_m': float(np.linalg.norm(mean)),
            'drift_ratio': float(np.linalg.norm(mean) / length)}


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; no overwrite')
    bridge, libero = bridge_dev(), libero_initial()
    mapping = json.loads(MAPPING.read_text(encoding='utf-8'))
    pose = np.median(np.load(C8 / 'inputs.npz')['pose_features'][:10352], axis=0).astype(np.float32)
    out = {'bridge_dev_windows': len(bridge), 'libero_tasks': [s['id'] for s in libero],
           'history': 'masked (first call)', 'pose18': 'Bridge train median',
           'thresholds': {'libero_over_bridge_spread_min': SPREAD_MIN, 'libero_drift_ratio_max': DRIFT_MAX},
           'models': {}}
    for name, ckpt in MODELS.items():
        policy = op.load_policy(ckpt, clip_dir=CLIP, dino_dir=DINO, mapping=mapping, device='cuda', sampling_seed=1101)
        dev = policy.device
        res = {'checkpoint_sha256': policy.checkpoint_metadata['checkpoint_sha256']}
        for cond, current in (('neutral', [0., 0.]), ('open', [1., 1.])):
            numeric = {'current2': torch.tensor([current], device=dev), 'pose18': torch.from_numpy(pose)[None].to(dev),
                       'state_delta': torch.zeros(1, 6, device=dev), 'state_delta_mask': torch.zeros(1, device=dev),
                       'prev_command': torch.zeros(1, 7, device=dev), 'prev_command_mask': torch.zeros(1, device=dev)}
            b, l = stats(first_waypoints(policy, bridge, numeric)), stats(first_waypoints(policy, libero, numeric))
            ratio = l['spread_m'] / b['spread_m']
            res[cond] = {'bridge': b, 'libero': l, 'libero_over_bridge_spread': ratio,
                         'pass_looks_at_image': ratio >= SPREAD_MIN, 'pass_no_drift': l['drift_ratio'] <= DRIFT_MAX}
            res[cond]['gate2_passed'] = res[cond]['pass_looks_at_image'] and res[cond]['pass_no_drift']
            print(name, cond, json.dumps({'bridge_spread': round(b['spread_m'], 4), 'libero_spread': round(l['spread_m'], 4),
                                          'ratio': round(ratio, 3), 'libero_mean': l['mean_vector_m'],
                                          'drift_ratio': round(l['drift_ratio'], 3), 'passed': res[cond]['gate2_passed']}),
                  flush=True)
        out['models'][name] = res
        del policy
        torch.cuda.empty_cache()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print('GATE2_WRITTEN', OUT)


if __name__ == '__main__':
    main()
