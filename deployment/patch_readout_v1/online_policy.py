"""A3(spatial读出+历史+未来到达位姿)在LIBERO里的在线推理回调。未校准的工程接口，不是正式评分。

接口与 code/libero_waypoint_execution.execute_policy_chunk 一致：policy(request) -> [16,8]
目标系航点(相对位置/0.1 + 左相对四元数 + 开正夹爪)。执行器每次只执行第1个航点(4个20Hz步=0.2秒)。

训练时6种输入在线怎么来（全部不读LIBERO示教，不拟合目标域统计）：
  visual   冻结DINOv2(与build_cache同一 dataset.prepare_image 和 FrozenDinoVision)，CLS+16x16→8x8池化，经fp16往返
  text     冻结CLIP文本，77 token定长填充，last_hidden_state，经fp16往返
  current2 [上一条开正夹爪命令, 1]；未知时 [0, 0]（与command8 inputs.npz同义）
  pose18   旧234100训练统计 + 冻结 pose_features.encode，输入是映射回源系的实测EEF位姿
  state_delta  本次与上次调用的源系位姿差(位置/0.1, 旋转向量 q_now*q_prev^-1)；仅当两次调用恰好间隔
               0.2秒(4个原生步)时可用，否则掩码为0（训练里每段第一帧同样掩码为0）
  prev_command 上次调用实际执行的第1个源系航点，按command8的裁剪规则(位置±0.05m、XYZ欧拉±0.25)编码。
               工程假设：把"上一个0.2秒的目标到达位姿"当作Bridge的"上一条控制命令"，二者在源数据里
               只是近似一致，未经验证。
采样用离线门槛所用的点预测：DDIM 10步×8次在内部运动空间取均值；夹爪取logit符号。
"""
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE / 'code'), str(ROOT / 'code')]
from cached_data import pool_grid  # noqa: E402
from geometry import external8_to_motion6, left_delta, quat_to_rotvec  # noqa: E402
from patch_policy import PatchPolicy  # noqa: E402
from libero_pose_transfer import SourceTargetPoseMapping  # noqa: E402

POSE_STATS = ROOT / 'deployment/bridge_full_capability_v1/state_stats.json'
POSE_CODEC = ROOT / 'deployment/bridge_full_capability_v1/pose_features.py'
POSE_CODEC_SHA = '596a980947c9f8cbbc7b4380fd4585d19960abc36075813684907b2c43e52df5'
POSITION_DIVISOR = 0.1                       # command8 与未来位姿标签相同
COMMAND_CLIP = np.array([.05, .05, .05, .25, .25, .25])   # command8 protocol motion_input_clip_abs
TICKS_PER_CALL = 4                           # 5Hz源 / 20Hz原生


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def pose_codec():
    if sha(POSE_CODEC) != POSE_CODEC_SHA:
        raise ValueError('Frozen training pose encoder changed')
    spec = importlib.util.spec_from_file_location('_a3_pose_features', POSE_CODEC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_delta6(prev_p, prev_q, now_p, now_q):
    """与 build_cache.numeric_features 同一公式。"""
    out = np.zeros(6)
    out[:3] = (np.asarray(now_p) - np.asarray(prev_p)) / POSITION_DIVISOR
    out[3:] = quat_to_rotvec(left_delta(now_q, prev_q))
    return out


def command8_from_waypoint(waypoint):
    """源系航点 -> command8：位置已是/0.1单位，按0.05m裁剪；旋转按XYZ欧拉±0.25裁剪（与encode_commands同序）。"""
    w = np.asarray(waypoint, np.float64)
    out = np.zeros(8)
    out[:3] = np.clip(w[:3], -COMMAND_CLIP[:3] / POSITION_DIVISOR, COMMAND_CLIP[:3] / POSITION_DIVISOR)
    euler = np.clip(Rotation.from_quat(w[3:7]).as_euler('xyz'), -COMMAND_CLIP[3:], COMMAND_CLIP[3:])
    q = Rotation.from_euler('xyz', euler).as_quat()
    out[3:7] = -q if q[3] < 0 else q
    out[7] = w[7]
    return out


def scale(x, stats):
    return (np.asarray(x) - np.asarray(stats['mean'])) / np.asarray(stats['std'])


class A3WaypointPolicy:
    def __init__(self, model, feature_stats, encoders, mapping, *, device, sampling_seed, draws=8, ddim_steps=10):
        self.model = model.to(device).eval().requires_grad_(False)
        self.stats = copy.deepcopy(feature_stats)
        self.tokenizer, self.text_model, self.vision, self.prepare_image = encoders
        self.mapping = SourceTargetPoseMapping(mapping)
        self.pose_stats = __import__('json').loads(POSE_STATS.read_text(encoding='utf-8'))
        self.codec = pose_codec()
        self.device = torch.device(device)
        self.seed, self.draws, self.ddim_steps = int(sampling_seed), draws, ddim_steps
        self.text_cache = {}
        self.reset()

    def reset(self):
        self.calls, self.last_inference = 0, None
        self.previous = None        # 上次调用的源系位姿和实际执行的源系第1航点

    def mark_executed(self, native_steps):
        """执行器跑完一次后告知实际原生步数；不足4步时下一次的历史不可用。"""
        if self.previous is not None:
            self.previous['native_steps'] = int(native_steps)

    @torch.no_grad()
    def _text(self, instruction):
        if instruction not in self.text_cache:
            tok = self.tokenizer([instruction], padding='max_length', max_length=77, truncation=True, return_tensors='pt')
            h = self.text_model(input_ids=tok['input_ids'].to(self.device),
                                attention_mask=tok['attention_mask'].to(self.device)).last_hidden_state
            self.text_cache[instruction] = (h.half().float(), tok['attention_mask'].bool().to(self.device))
        return self.text_cache[instruction]

    @torch.no_grad()
    def _visual(self, rgb_upright):
        px = self.prepare_image(rgb_upright)[None].to(self.device)
        h = self.vision(pixel_values=px).last_hidden_state
        if tuple(h.shape[1:]) != (257, 1024):
            raise ValueError('Expected CLS+256 patch tokens of width 1024')
        h = h.half().float().cpu().numpy()
        grid = self.model.cfg['grid']
        tokens = np.concatenate([h[:, :1], pool_grid(h[:, 1:], grid) if grid != 16 else h[:, 1:]], axis=1)
        return torch.from_numpy(tokens).to(self.device)

    def numeric_inputs(self, request, source_p, source_q):
        """返回 (batch字典的数值部分, 记录用的原始值)。"""
        cmd = request['current_open_command']
        current2 = [0., 0.] if cmd is None else [float(cmd), 1.]
        pose18 = self.codec.encode(np.r_[source_p, source_q][None], ['Bridge'], self.pose_stats, 'real_pose')[0]
        sd, sd_mask, pc8, pc_mask = np.zeros(6), 0., np.zeros(8), 0.
        prev = self.previous
        if prev is not None and prev.get('native_steps') == TICKS_PER_CALL:
            sd, sd_mask = state_delta6(prev['source_p'], prev['source_q'], source_p, source_q), 1.
            pc8, pc_mask = command8_from_waypoint(prev['source_waypoint']), 1.
        pc_in = np.zeros(7)
        if pc_mask:
            pc_in[:6] = scale(external8_to_motion6(pc8[None])[0], self.stats['prev_command'])
            pc_in[6] = pc8[7]
        sd_in = scale(sd, self.stats['state_delta']) * sd_mask
        f = lambda x: torch.tensor(np.asarray(x, np.float32)[None], device=self.device)
        numeric = {'current2': f(current2), 'pose18': f(pose18), 'state_delta': f(sd_in),
                   'state_delta_mask': f(sd_mask)[0], 'prev_command': f(pc_in), 'prev_command_mask': f(pc_mask)[0]}
        raw = {'current2': current2, 'state_delta_raw': sd.tolist(), 'state_delta_available': bool(sd_mask),
               'prev_command8': pc8.tolist(), 'prev_command_available': bool(pc_mask)}
        return numeric, raw

    @torch.no_grad()
    def __call__(self, request):
        source_p, source_q = self.mapping.source_reference(request['reference_position_m'],
                                                           request['reference_quaternion_xyzw'])
        numeric, raw = self.numeric_inputs(request, source_p, source_q)
        text, text_mask = self._text(request['instruction'])
        batch = dict(numeric, visual=self._visual(request['rgb_upright']), text=text, text_mask=text_mask)
        seed = self.seed + self.calls
        g = torch.Generator(device=self.device).manual_seed(seed)
        source = self.model.predict8(batch, g, 'ddim', draws=self.draws, ddim_steps=self.ddim_steps,
                                     average=True)[0, 0].cpu().numpy().astype(np.float64)
        if source.shape != (16, 8) or not np.isfinite(source).all():
            raise ValueError('Invalid decoded A3 sample')
        target = self.mapping.target_waypoints(source, request['reference_position_m'],
                                               request['reference_quaternion_xyzw'], POSITION_DIVISOR, POSITION_DIVISOR)
        self.previous = {'source_p': source_p, 'source_q': source_q, 'source_waypoint': source[0].copy(),
                         'native_steps': None}
        self.last_inference = dict(raw, call=self.calls, sampling_seed=seed, sampling='ddim10_x8_mean',
                                   source_waypoints=source.tolist(),
                                   output_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                                   physical_calibration_verified=False, formal_libero_scoring_admitted=False)
        self.calls += 1
        return target


def load_model(checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') != 'patch-readout-policy-v1' or saved.get('target') != 'future_pose':
        raise ValueError('A3 future-pose patch-readout checkpoint required')
    cfg = saved['config']
    if cfg['readout'] != 'spatial' or set(cfg['history']) != {'state_delta', 'prev_command'}:
        raise ValueError('Checkpoint is not the A3 spatial+history configuration')
    model = PatchPolicy(cfg, n_tokens=1 + cfg['grid'] ** 2)
    model.load_state_dict(saved['ema_state'], strict=True)
    return model.eval(), saved


def load_encoders(clip_dir, dino_dir, device):
    from transformers import CLIPTextModel, CLIPTokenizer
    from dataset import prepare_image
    from frozen_visual_backbone import FrozenDinoVision, dino_config
    tokenizer = CLIPTokenizer.from_pretrained(str(clip_dir), local_files_only=True)
    text = CLIPTextModel.from_pretrained(str(clip_dir), local_files_only=True).to(device).eval()
    vision = FrozenDinoVision(dino_config(dino_dir)).to(device).eval()
    for m in (text, vision):
        m.requires_grad_(False)
    return tokenizer, text, vision, prepare_image


def load_policy(checkpoint, *, clip_dir, dino_dir, mapping, device='cuda', sampling_seed=1101):
    SourceTargetPoseMapping(mapping)     # 映射事实缺失时在加载大模型前就失败
    model, saved = load_model(checkpoint)
    policy = A3WaypointPolicy(model, saved['feature_stats'], load_encoders(clip_dir, dino_dir, device), mapping,
                              device=device, sampling_seed=sampling_seed)
    policy.checkpoint_metadata = {'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_sha256': sha(checkpoint),
                                  'completed_steps': saved['completed_steps'], 'seed': saved['seed'],
                                  'ema_weights': True, 'feature_stats_from_checkpoint': True,
                                  'physical_calibration_verified': False, 'formal_libero_scoring_admitted': False}
    return policy
