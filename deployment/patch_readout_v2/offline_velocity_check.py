"""离线检查：B3在仿真里步子越走越大、方向固定为-y，是不是"位姿差(速度)输入"自我放大？

只读第3关试验录下的画面(execution.gif)和每次调用记录(chunks.json)，不仿真、不训练。
每个时刻的画面、指令都与试验相同，只改速度输入：
  on        试验时实际的位姿差（采样种子与试验相同 1101+调用序号，用来核对能否复现）
  off       掩码为0（训练时50%样本同样如此；每段第一帧也是如此）
  reversed  位姿差取反（模型若只是"顺着刚才的方向继续走"，输出方向应跟着反过来）
指标：首航点长度、首航点与上一时刻实际位移方向的余弦。
注意：这是在已漂移轨迹的画面上做的开环比较，不能代替闭环仿真。
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

RUN = ROOT / 'reports/patch_readout_v2_241155/libero_gate3/B3_run1'
OUT = RUN.parent / 'velocity_check.json'
CLIP = ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
DINO = ROOT / 'baselines/dinov2_large_frozen_assets_20261023'
INSTRUCTION = 'close the top drawer of the cabinet and put the black bowl on top of it'


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; no overwrite')
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    chunks = json.loads((RUN / 'chunks.json').read_text(encoding='utf-8'))
    with Image.open(RUN / 'execution.gif') as gif:
        images = []
        for i in range(gif.n_frames):
            gif.seek(i)
            images.append(np.asarray(gif.convert('RGB')).copy())
    policy = op.load_policy(Path(protocol['checkpoint']), clip_dir=CLIP, dino_dir=DINO,
                            mapping=protocol['spatial_mapping'], device='cuda', sampling_seed=1101)
    text, mask = policy._text(INSTRUCTION)
    frames = [k for k in range(1, len(chunks)) if chunks[k]['inference']['state_delta_available']]

    def predict(k, mode):
        c, prev = chunks[k], chunks[k - 1]
        request = {'current_open_command': prev['last_open_command']}
        sp0, sq0 = policy.mapping.source_reference(prev['reference_position_m'], prev['reference_quaternion_xyzw'])
        policy.previous = {'source_p': sp0, 'source_q': sq0, 'native_steps': 0 if mode == 'off' else 4,
                           'source_waypoint': np.array(prev['inference']['source_waypoints'][0])}
        sp, sq = policy.mapping.source_reference(c['reference_position_m'], c['reference_quaternion_xyzw'])
        numeric, raw = policy.numeric_inputs(request, sp, sq)
        if mode == 'reversed':
            numeric['state_delta'] = -numeric['state_delta']
        batch = dict(numeric, visual=policy._visual(images[k]), text=text, text_mask=mask)
        g = torch.Generator(device=policy.device).manual_seed(1101 + k)
        with torch.no_grad():
            src = policy.model.predict8(batch, g, 'ddim', draws=8, average=True)[0, 0].cpu().numpy()
        return src[0, :3] * op.POSITION_DIVISOR, np.array(raw['state_delta_raw'][:3]) * op.POSITION_DIVISOR

    recorded = np.array([np.array(chunks[k]['inference']['source_waypoints'][0][:3]) * op.POSITION_DIVISOR
                         for k in frames])
    cos = lambda a, b: float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    out = {'frames': frames, 'note': 'Open-loop counterfactual on frames of the drifted trajectory; not closed-loop.',
           'conditions': {}}
    for mode in ('on', 'off', 'reversed'):
        rows = [predict(k, mode) for k in frames]
        first = np.array([r[0] for r in rows])
        moved = np.array([r[1] for r in rows])          # 上一时刻实际源系位移(米)
        late = np.array(frames) >= 25
        out['conditions'][mode] = {
            'mean_step_m': float(np.linalg.norm(first, axis=1).mean()),
            'mean_step_late_m': float(np.linalg.norm(first[late], axis=1).mean()),
            'mean_first_waypoint_m': first.mean(0).round(4).tolist(),
            'mean_cosine_with_previous_motion': float(np.mean([cos(f, m) for f, m in zip(first, moved)])),
            'first_waypoint_m': first.round(4).tolist()}
        if mode == 'on':
            out['previous_motion_mean_step_m'] = float(np.linalg.norm(moved, axis=1).mean())
            out['on_vs_recorded_max_abs_m'] = float(np.abs(first - recorded).max())
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'reproduce_max_abs_m': out['on_vs_recorded_max_abs_m'],
                      'previous_motion_mean_step_m': out['previous_motion_mean_step_m'],
                      **{k: {x: v[x] for x in ('mean_step_m', 'mean_step_late_m', 'mean_first_waypoint_m',
                                               'mean_cosine_with_previous_motion')}
                         for k, v in out['conditions'].items()}}, indent=1))


if __name__ == '__main__':
    main()
