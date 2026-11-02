"""离线检查：在Bridge开发集画面上，"夹爪已知张开"也会让A3往+y走吗？

64个A3开发窗口(E:\\dataset，与训练逐字节一致)，各自真实指令；历史掩码0。
位置输入两种：own=该窗口自己的pose18(分布内) / median=训练集中位数(与LIBERO检查相同)。
夹爪状态 current2 ∈ {[0,0] 未知, [1,1] 已知张开, [-1,1] 已知闭合, true 该窗口真实值}。
对照：同一批窗口command8真实首命令的y方向。不仿真、不训练。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import online_policy as op  # noqa: E402
from offline_image_check_v2 import bridge_dev, CLIP, DINO, CHECKPOINT, MAPPING, C8  # noqa: E402

OUT = ROOT / 'reports/patch_readout_libero_engineering_20261102/bridge_gripper_check.json'
CURRENT = {'unknown': [0., 0.], 'open': [1., 1.], 'closed': [-1., 1.], 'true': None}


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; no overwrite')
    samples = bridge_dev()
    rows = json.loads((C8 / 'rows.json').read_text(encoding='utf-8'))
    slots = [i for i, r in enumerate(rows) if r['role'] == 'validation']
    z = np.load(C8 / 'inputs.npz')
    own_pose, true_current = z['pose_features'][slots], z['current'][slots]
    label_y = z['targets'][slots, 0, 1] * op.POSITION_DIVISOR
    median = np.median(z['pose_features'][:10352], axis=0)
    policy = op.load_policy(CHECKPOINT, clip_dir=CLIP, dino_dir=DINO,
                            mapping=json.loads(MAPPING.read_text(encoding='utf-8')), device='cuda', sampling_seed=1101)
    dev = policy.device
    visual = [policy._visual(s['rgb']) for s in samples]

    def first(i, pose, current):
        text, mask = policy._text(samples[i]['text'])
        batch = {'current2': torch.tensor([current], dtype=torch.float32, device=dev),
                 'pose18': torch.tensor(pose[None], dtype=torch.float32, device=dev),
                 'state_delta': torch.zeros(1, 6, device=dev), 'state_delta_mask': torch.zeros(1, device=dev),
                 'prev_command': torch.zeros(1, 7, device=dev), 'prev_command_mask': torch.zeros(1, device=dev),
                 'visual': visual[i], 'text': text, 'text_mask': mask}
        g = torch.Generator(device=dev).manual_seed(2024)
        with torch.no_grad():
            src = policy.model.predict8(batch, g, 'ddim', draws=8, average=True)[0, 0].cpu().numpy()
        return src[0, :3] * op.POSITION_DIVISOR

    out = {'windows': len(samples), 'history': 'masked',
           'true_current_counts': {str(v): int((true_current == v).all(1).sum()) for v in ([0, 0], [1, 1], [-1, 1])},
           'label_first_command_y': {'mean_m': round(float(label_y.mean()), 4), 'fraction_plus_y': round(float(np.mean(label_y > 0)), 3)},
           'results': {}}
    for pose_name in ('own', 'median'):
        for name, current in CURRENT.items():
            xyz = np.array([first(i, own_pose[i] if pose_name == 'own' else median,
                                  true_current[i].tolist() if current is None else current) for i in range(len(samples))])
            out['results'][f'pose_{pose_name}__{name}'] = {
                'mean_first_waypoint_m': xyz.mean(0).round(4).tolist(),
                'mean_y_m': round(float(xyz[:, 1].mean()), 4),
                'fraction_plus_y': round(float(np.mean(xyz[:, 1] > 0)), 3),
                'corr_y_with_label': round(float(np.corrcoef(xyz[:, 1], label_y)[0, 1]), 3)}
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('true_current_counts', 'label_first_command_y')}))
    for k, v in out['results'].items():
        print(f'{k:24s} y={v["mean_y_m"]:+.4f} +y={v["fraction_plus_y"]:.2f} corr={v["corr_y_with_label"]:+.2f} xyz={v["mean_first_waypoint_m"]}')


if __name__ == '__main__':
    main()
