"""离线2x2：仿真里"往右上漂"来自位置输入超范围，还是来自历史反馈？

只读 20261102 试验录下的画面(execution.gif)和每次调用的记录(chunks.json)，不仿真、不训练。
每个时刻的画面、指令、夹爪命令(current2)都和试验时相同，只改两样：
  位置  original=试验时的实际pose18(z≈+21σ) / median=Bridge训练集pose18逐维中位数
  历史  on=试验时的位姿差和上一条命令 / off=两者掩码为0(训练时30%样本同样如此)
第0次调用本来就没有历史，只取第1次以后的时刻。
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

RUN = ROOT / 'reports/patch_readout_libero_engineering_20261102/run1'
OUT = RUN.parent / 'drift_factorial.json'
C8 = ROOT / 'deployment/bridge_command8_fresh_scaled_v1'
CLIP = ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
DINO = ROOT / 'baselines/dinov2_large_frozen_assets_20261023'
INSTRUCTION = 'close the top drawer of the cabinet and put the black bowl on top of it'
FRAMES = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]


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
    median = np.median(np.load(C8 / 'inputs.npz')['pose_features'][:10352], axis=0).astype(np.float32)
    text, mask = policy._text(INSTRUCTION)

    def predict(k, pose, history):
        c, prev = chunks[k], chunks[k - 1]
        request = {'reference_position_m': np.array(c['reference_position_m']),
                   'reference_quaternion_xyzw': np.array(c['reference_quaternion_xyzw']),
                   'current_open_command': prev['last_open_command'], 'instruction': INSTRUCTION}
        sp, sq = policy.mapping.source_reference(prev['reference_position_m'], prev['reference_quaternion_xyzw'])
        policy.previous = {'source_p': sp, 'source_q': sq, 'native_steps': 4 if history == 'on' else 0,
                           'source_waypoint': np.array(prev['inference']['source_waypoints'][0])}
        sp, sq = policy.mapping.source_reference(request['reference_position_m'], request['reference_quaternion_xyzw'])
        numeric, _ = policy.numeric_inputs(request, sp, sq)
        if pose == 'median':
            numeric['pose18'][0] = torch.from_numpy(median)
        batch = dict(numeric, visual=policy._visual(images[k]), text=text, text_mask=mask)
        g = torch.Generator(device=policy.device).manual_seed(2024)
        with torch.no_grad():
            src = policy.model.predict8(batch, g, 'ddim', draws=8, average=True)[0, 0].cpu().numpy()
        return src[0, :3] * op.POSITION_DIVISOR

    recorded = np.array([np.array(chunks[k]['inference']['source_waypoints'][0][:3]) * op.POSITION_DIVISOR for k in FRAMES])
    out = {'frames': FRAMES, 'recorded_mean_first_waypoint_m': recorded.mean(0).round(4).tolist(), 'conditions': {}}
    for pose in ('original', 'median'):
        for history in ('on', 'off'):
            first = np.array([predict(k, pose, history) for k in FRAMES])
            out['conditions'][f'pose_{pose}__history_{history}'] = {
                'mean_first_waypoint_m': first.mean(0).round(4).tolist(),
                'mean_step_m': round(float(np.linalg.norm(first, axis=1).mean()), 4),
                'fraction_up_and_plus_y': round(float(np.mean((first[:, 1] > 0) & (first[:, 2] > 0))), 3),
                'first_waypoint_m': first.round(4).tolist()}
    reproduce = np.abs(np.array(out['conditions']['pose_original__history_on']['first_waypoint_m']) - recorded).max()
    out['original_on_vs_recorded_max_abs_m'] = float(reproduce)
    out['note'] = 'Open-loop counterfactual on frames of the drifted trajectory; not a closed-loop result.'
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'reproduce_max_abs_m': float(reproduce), 'recorded': out['recorded_mean_first_waypoint_m'],
                      **{k: {x: v[x] for x in ('mean_first_waypoint_m', 'mean_step_m', 'fraction_up_and_plus_y')}
                         for k, v in out['conditions'].items()}}, indent=1))


if __name__ == '__main__':
    main()
