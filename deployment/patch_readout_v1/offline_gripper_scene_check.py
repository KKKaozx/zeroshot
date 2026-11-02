"""离线检查：往+y漂是"夹爪状态输入"引起的，还是"关抽屉放碗这个场景"引起的？

固定：位置输入=Bridge训练集pose18中位数，历史掩码0，各画面用自己任务的指令。
变化：
  夹爪状态 current2 ∈ {[0,0] 未知, [1,1] 已知张开, [-1,1] 已知闭合}
  画面    64个不同LIBERO任务的初始画面(与image_check_v2同一批) + 工程任务(SCENE10)初始画面
          + 工程试验中已漂移的画面(第10/30/49次调用时)
只读各任务 demo_0 第0帧观测和试验录下的GIF，不读动作、不仿真、不训练。
"""
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
from offline_image_check_v2 import libero_initial, CLIP, DINO, CHECKPOINT, MAPPING, C8, LIBERO  # noqa: E402

RUN = ROOT / 'reports/patch_readout_libero_engineering_20261102/run1'
OUT = RUN.parent / 'gripper_scene_check.json'
TASK = 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it'
INSTRUCTION = 'close the top drawer of the cabinet and put the black bowl on top of it'
DRIFTED = [10, 30, 49]
CURRENT = {'unknown': [0., 0.], 'open': [1., 1.], 'closed': [-1., 1.]}


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; no overwrite')
    others = [s for s in libero_initial() if TASK not in s['id']]
    with h5py.File(LIBERO / 'libero_90' / (TASK + '_demo.hdf5'), 'r') as f:
        scene10 = [{'rgb': f['data/demo_0/obs/agentview_rgb'][0][::-1].copy(), 'text': INSTRUCTION, 'id': 'scene10_initial'}]
    with Image.open(RUN / 'execution.gif') as gif:
        drifted = []
        for k in DRIFTED:
            gif.seek(k)
            drifted.append({'rgb': np.asarray(gif.convert('RGB')).copy(), 'text': INSTRUCTION, 'id': f'scene10_call{k}'})
    policy = op.load_policy(CHECKPOINT, clip_dir=CLIP, dino_dir=DINO,
                            mapping=json.loads(MAPPING.read_text(encoding='utf-8')), device='cuda', sampling_seed=1101)
    dev = policy.device
    pose = torch.from_numpy(np.median(np.load(C8 / 'inputs.npz')['pose_features'][:10352], axis=0).astype(np.float32))[None].to(dev)

    def first(sample, current):
        text, mask = policy._text(sample['text'])
        batch = {'current2': torch.tensor([current], device=dev), 'pose18': pose,
                 'state_delta': torch.zeros(1, 6, device=dev), 'state_delta_mask': torch.zeros(1, device=dev),
                 'prev_command': torch.zeros(1, 7, device=dev), 'prev_command_mask': torch.zeros(1, device=dev),
                 'visual': policy._visual(sample['rgb']), 'text': text, 'text_mask': mask}
        g = torch.Generator(device=dev).manual_seed(2024)
        with torch.no_grad():
            src = policy.model.predict8(batch, g, 'ddim', draws=8, average=True)[0, 0].cpu().numpy()
        return src[0, :3] * op.POSITION_DIVISOR, float(src[0, 7])

    out = {'other_scenes': len(others), 'fixed': 'Bridge median pose18, history masked, own task instruction', 'results': {}}
    for group, samples in (('other_scenes_initial', others), ('scene10_initial', scene10), ('scene10_drifted', drifted)):
        for name, current in CURRENT.items():
            rows = [first(s, current) for s in samples]
            xyz = np.array([r[0] for r in rows])
            out['results'][f'{group}__{name}'] = {
                'mean_first_waypoint_m': xyz.mean(0).round(4).tolist(),
                'mean_plus_y_m': round(float(xyz[:, 1].mean()), 4),
                'fraction_plus_y': round(float(np.mean(xyz[:, 1] > 0)), 3),
                'fraction_up_and_plus_y': round(float(np.mean((xyz[:, 1] > 0) & (xyz[:, 2] > 0))), 3),
                'gripper_open_fraction': round(float(np.mean([r[1] > 0 for r in rows])), 3)}
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out['results'], indent=1))


if __name__ == '__main__':
    main()
