"""离线反事实：把LIBERO的位置输入挪进Bridge训练范围，策略的动作方向会不会变、会不会更看画面。

只读 20261102 试验录下的画面(execution.gif)和每次调用的记录(chunks.json)，不仿真、不训练。
GIF有256色量化，各条件用同一批画面，量化对各条件影响相同。

条件(只改pose18，其余输入与试验时完全相同)：
  original      试验时的实际输入(位置约 z=+21σ, x=-8σ)
  bridge_mean   位置换成Bridge训练均值(标准化后=0)，姿态不变
  bridge_median 整个pose18换成Bridge训练集逐维中位数(位置+姿态都在分布内)
指标：首航点位移(米)；同一条件下换不同画面时输出变化多大(画面敏感度)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import online_policy as op  # noqa: E402

RUN = ROOT / 'reports/patch_readout_libero_engineering_20261102/run1'
OUT = RUN.parent / 'pose_whatif.json'
CLIP = ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
DINO = ROOT / 'baselines/dinov2_large_frozen_assets_20261023'
FRAMES = [1, 5, 10, 20, 30, 40, 49]      # 第0次调用没有历史，从1开始


def frames():
    with Image.open(RUN / 'execution.gif') as gif:
        out = []
        for i in range(gif.n_frames):
            gif.seek(i)
            out.append(np.asarray(gif.convert('RGB')).copy())
    return out


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; no overwrite')
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    chunks = json.loads((RUN / 'chunks.json').read_text(encoding='utf-8'))
    images = frames()
    if len(images) != len(chunks) + 1:
        raise ValueError('Expected one recorded frame per policy call plus the final frame')
    policy = op.load_policy(op.Path(protocol['checkpoint']), clip_dir=CLIP, dino_dir=DINO,
                            mapping=protocol['spatial_mapping'], device='cuda', sampling_seed=1101)
    train_pose = np.load(ROOT / 'deployment/bridge_command8_fresh_scaled_v1/inputs.npz')['pose_features'][:10352]
    median = np.median(train_pose, axis=0)

    def predict(k, image, condition):
        c, prev = chunks[k], chunks[k - 1]
        request = {'reference_position_m': np.array(c['reference_position_m']),
                   'reference_quaternion_xyzw': np.array(c['reference_quaternion_xyzw']),
                   'current_open_command': prev['last_open_command'], 'instruction': op.__dict__.get('INSTRUCTION',
                   'close the top drawer of the cabinet and put the black bowl on top of it')}
        sp, sq = policy.mapping.source_reference(prev['reference_position_m'], prev['reference_quaternion_xyzw'])
        policy.previous = {'source_p': sp, 'source_q': sq, 'native_steps': 4,
                           'source_waypoint': np.array(prev['inference']['source_waypoints'][0])}
        sp, sq = policy.mapping.source_reference(request['reference_position_m'], request['reference_quaternion_xyzw'])
        numeric, _ = policy.numeric_inputs(request, sp, sq)
        if condition == 'bridge_mean':
            numeric['pose18'][0, :3] = 0
        elif condition == 'bridge_median':
            numeric['pose18'][0] = torch.from_numpy(median.astype(np.float32))
        text, mask = policy._text(request['instruction'])
        batch = dict(numeric, visual=policy._visual(image), text=text, text_mask=mask)
        g = torch.Generator(device=policy.device).manual_seed(2024)
        with torch.no_grad():
            src = policy.model.predict8(batch, g, 'ddim', draws=8, average=True)[0, 0].cpu().numpy()
        return src[0, :3] * op.POSITION_DIVISOR, src[:, :3].sum(0) * op.POSITION_DIVISOR, float(src[0, 7])

    # 检查：original条件应复现试验时的首航点(GIF量化会带来少量差异)
    results = {}
    for cond in ('original', 'bridge_mean', 'bridge_median'):
        rows = [predict(k, images[k], cond) for k in FRAMES]
        first = np.array([r[0] for r in rows])
        # 画面敏感度：固定第10次调用的数值输入，换7张不同画面
        swap = np.array([predict(10, images[k], cond)[0] for k in FRAMES])
        results[cond] = {
            'first_waypoint_m': first.round(4).tolist(),
            'mean_first_waypoint_m': first.mean(0).round(4).tolist(),
            'mean_step_m': float(np.linalg.norm(first, axis=1).mean()),
            'gripper_open_fraction': float(np.mean([r[2] > 0 for r in rows])),
            'image_swap_spread_m': float(np.linalg.norm(swap - swap.mean(0), axis=1).mean()),
            'image_swap_first_waypoints_m': swap.round(4).tolist()}
    recorded = np.array([np.array(chunks[k]['inference']['source_waypoints'][0][:3]) * op.POSITION_DIVISOR for k in FRAMES])
    reproduce = float(np.abs(np.array(results['original']['first_waypoint_m']) - recorded).max())
    report = {'frames': FRAMES, 'recorded_first_waypoint_m': recorded.round(4).tolist(),
              'original_vs_recorded_max_abs_m': reproduce, 'results': results,
              'note': 'Offline counterfactual on recorded frames; no simulation, no training, no LIBERO statistics fitted.'}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'reproduce_max_abs_m': reproduce, **{k: {x: v[x] for x in ('mean_first_waypoint_m', 'mean_step_m',
          'gripper_open_fraction', 'image_swap_spread_m')} for k, v in results.items()}}, indent=1))


if __name__ == '__main__':
    main()
