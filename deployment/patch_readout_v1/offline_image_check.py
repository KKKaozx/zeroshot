"""离线检查：A3在Bridge画面上看不看画面？LIBERO画面在DINO特征上离Bridge多远？

只读本机已有的Bridge原始记录(重新导出的副本，取画面分布；与训练同一 prepare_image)和20261102试验录下的LIBERO画面。
不仿真、不训练、不拟合任何LIBERO统计。

为了让两边可比，数值输入固定为同一组(取一个Bridge开发窗口的pose18/current2，历史掩码为0；
训练时30%的样本历史同样被丢弃)，只换画面：
  spread  同一组数值输入下，换不同画面时首航点偏离均值的平均距离(米)
  ctx     策略条件向量(读出+本体)随画面变化的幅度
  DINO    CLS余弦相似度：Bridge内部、LIBERO内部、Bridge-LIBERO之间
"""
import hashlib
import json
import sys
from pathlib import Path
import io

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import online_policy as op  # noqa: E402

RUN = ROOT / 'reports/patch_readout_libero_engineering_20261102/run1'
OUT = RUN.parent / 'image_check.json'
C8 = ROOT / 'deployment/bridge_command8_fresh_scaled_v1'
CLIP = ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
DINO = ROOT / 'baselines/dinov2_large_frozen_assets_20261023'
N_BRIDGE = 64


def local_payloads(limit):
    """本机的Bridge文件是重新导出的副本，与训练窗口的字节不一致；这里只需要Bridge画面的分布，
    所以按文件名排序，每个文件取第1条记录，直到limit条。"""
    found = []
    for p in sorted(ROOT.parent.rglob('bridge_data_v2-train.tfrecord-*'), key=lambda x: x.name):
        if len(found) >= limit:
            break
        if not p.is_file():
            continue
        size = p.stat().st_size
        with p.open('rb') as f:
            while True:
                head = f.read(12)
                if len(head) < 12:
                    break
                n = int.from_bytes(head[:8], 'little')
                if n > size - f.tell():         # 不是普通TFRecord(如索引/压缩文件)，跳过
                    break
                raw = f.read(n)
                f.read(4)
                found.append((p.name, hashlib.sha256(raw).hexdigest(), raw))
                break
    return found


def bridge_samples():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    out, seen = [], set()
    for name, digest, raw in local_payloads(4 * N_BRIDGE):
        if digest in seen:
            continue
        seen.add(digest)
        fields = tf.train.Example.FromString(raw).features.feature
        images = fields['steps/observation/image_0'].bytes_list.value
        if not images:
            continue
        t = min(8, len(images) - 1)
        text = fields['steps/language_instruction'].bytes_list.value[t].decode('utf-8')
        with Image.open(io.BytesIO(images[t])) as im:
            out.append({'file': name, 'rgb': np.asarray(im.convert('RGB')).copy(), 'text': text})
        if len(out) == N_BRIDGE:
            break
    return out


def libero_frames():
    with Image.open(RUN / 'execution.gif') as gif:
        frames = []
        for i in range(gif.n_frames - 1):       # 最后一帧之后没有调用
            gif.seek(i)
            frames.append(np.asarray(gif.convert('RGB')).copy())
    return frames


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; no overwrite')
    bridge = bridge_samples()
    libero = libero_frames()
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    policy = op.load_policy(Path(protocol['checkpoint']), clip_dir=CLIP, dino_dir=DINO,
                            mapping=protocol['spatial_mapping'], device='cuda', sampling_seed=1101)
    inputs = np.load(C8 / 'inputs.npz')
    rows = json.loads((C8 / 'rows.json').read_text(encoding='utf-8'))
    slot = next(i for i, r in enumerate(rows) if r['role'] == 'validation')
    dev = policy.device
    numeric = {'current2': torch.tensor(inputs['current'][slot][None], device=dev),
               'pose18': torch.tensor(inputs['pose_features'][slot][None], device=dev),
               'state_delta': torch.zeros(1, 6, device=dev), 'state_delta_mask': torch.zeros(1, device=dev),
               'prev_command': torch.zeros(1, 7, device=dev), 'prev_command_mask': torch.zeros(1, device=dev)}

    def run(images, instruction):
        text, mask = policy._text(instruction)
        cls, ctxs, first = [], [], []
        for rgb in images:
            vis = policy._visual(rgb)
            batch = dict(numeric, visual=vis, text=text, text_mask=mask)
            with torch.no_grad():
                cls.append(vis[0, 0].cpu().numpy())
                ctxs.append(policy.model.context(batch)[0].cpu().numpy())
                g = torch.Generator(device=dev).manual_seed(2024)
                src = policy.model.predict8(batch, g, 'ddim', draws=8, average=True)[0, 0].cpu().numpy()
            first.append(src[0, :3] * op.POSITION_DIVISOR)
        return np.array(cls), np.array(ctxs), np.array(first)

    def spread(x):
        return float(np.linalg.norm(x - x.mean(0), axis=1).mean())

    def cos(a, b):
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return a @ b.T

    # 两边用同一句指令，排除语言差异；再用Bridge自己的指令各跑一次作对照
    instruction = bridge[0]['text']
    b_cls, b_ctx, b_first = run([b['rgb'] for b in bridge], instruction)
    l_cls, l_ctx, l_first = run(libero, instruction)
    _, _, l_first_task = run(libero, 'close the top drawer of the cabinet and put the black bowl on top of it')
    bb, ll, bl = cos(b_cls, b_cls), cos(l_cls, l_cls), cos(b_cls, l_cls)
    off = lambda m: m[~np.eye(len(m), dtype=bool)].mean()
    nearest_b = np.sort(bb, 1)[:, -2]           # 每张Bridge图最像的另一张Bridge图
    report = {
        'bridge_images': len(bridge), 'bridge_files': [b['file'] for b in bridge],
        'libero_images': len(libero), 'numeric_anchor_slot': slot, 'shared_instruction': instruction,
        'dino_cls_cosine': {'bridge_bridge_mean': float(off(bb)), 'libero_libero_mean': float(off(ll)),
                            'bridge_libero_mean': float(bl.mean()),
                            'bridge_nearest_other_bridge_mean': float(nearest_b.mean()),
                            'libero_nearest_bridge_mean': float(bl.max(0).mean())},
        'policy_context_spread': {'bridge': spread(b_ctx), 'libero': spread(l_ctx),
                                  'distance_between_means': float(np.linalg.norm(b_ctx.mean(0) - l_ctx.mean(0)))},
        'first_waypoint_spread_m': {'bridge': spread(b_first), 'libero': spread(l_first),
                                    'libero_task_instruction': spread(l_first_task)},
        'mean_first_waypoint_m': {'bridge': b_first.mean(0).round(4).tolist(), 'libero': l_first.mean(0).round(4).tolist(),
                                  'libero_task_instruction': l_first_task.mean(0).round(4).tolist()},
        'note': 'Numeric inputs fixed to one Bridge dev window, history masked; only the image changes.'}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
