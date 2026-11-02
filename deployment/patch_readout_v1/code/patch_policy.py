"""冻结DINOv2(或CLIP)视觉token + CLIP文本token -> Adapter -> 扩散动作策略。

和旧RobotAdapterModel的关键区别：
1. 读出(readout)。旧Adapter里每个视觉token各自做"视觉查询文本"的交叉注意力和FFN，
   token之间没有任何交互；默认pooling='cls'只取CLS，因此256个patch token对输出
   完全没有影响(梯度恒为0)，策略实际只看到一个全局向量。这里保留同样的逐token
   交叉注意力(老师设计：视觉作query、语言作key/value)，但用"文本条件的可学习查询"
   对全部patch做注意力读出，并加入可学习位置编码(每个格子一个向量)，让物体/夹爪位置进入动作条件。
   readout='cls' 保留为对照，复现旧行为。
2. 历史条件：可选上一时刻位姿差(速度)和上一条命令，带可用性掩码和训练时随机丢弃，
   防止策略只抄历史(copycat)。
3. 内部动作：位置3+旋转向量3做标准化，外部仍输出8维(位置3+四元数4+夹爪1)。
4. 扩散后端：Diffusion Policy式三尺度ConditionalUnet1D(每层FiLM)，x0预测，余弦噪声表。
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter import CrossAttentionLayer
from diffusion_decoder import MultiscaleConditionalUnet1D
from geometry import torch_rotvec_to_quat

DEFAULT_CONFIG = {
    'readout': 'patch',          # 'patch' | 'spatial' | 'cls'
    'num_keypoints': 32,         # 仅readout='spatial'
    'keypoint_feat_dim': 32,     # 仅readout='spatial'
    'grid': 8,                   # patch网格 16 或 8 (2x2平均池化)
    'num_adapter_layers': 8,     # 逐token视觉->文本交叉注意力层数 (老师的Adapter深度消融变量)
    'attention_dim': 512,
    'num_heads': 8,
    'dropout': 0.1,
    'num_queries': 8,
    'readout_dim': 512,
    'readout_layers': 2,
    'history': ['state_delta', 'prev_command'],   # 子集；[] 表示无历史
    'history_dropout': 0.3,
    'use_pose18': True,
    'context_dim': 1024,
    'down_dims': [256, 512, 1024],
    'diffusion_steps': 100,
    'clip_x0': 5.0,
    'gripper_loss_weight': 0.25,
}


def cosine_betas(steps, s=0.008):
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    return (1 - ab[1:] / ab[:-1]).clamp(max=0.999).float()


class PatchReadout(nn.Module):
    def __init__(self, n_tokens, vis_dim, text_dim, d, nq, layers, heads, dropout, out_dim):
        super().__init__()
        self.token_proj = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, d))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        nn.init.normal_(self.pos, std=0.02)
        self.queries = nn.Parameter(torch.randn(1, nq, d) * 0.02)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                'qn': nn.LayerNorm(d), 'kn': nn.LayerNorm(d),
                'attn': nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True),
                'fn': nn.LayerNorm(d),
                'ffn': nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d)),
            }))
        self.out = nn.Sequential(nn.LayerNorm(nq * d), nn.Linear(nq * d, out_dim))

    def forward(self, tokens, text, text_mask):
        kv = self.token_proj(tokens) + self.pos
        m = text_mask.unsqueeze(-1).float()
        pooled = (text * m).sum(1) / m.sum(1).clamp_min(1.0)
        q = self.queries.expand(tokens.shape[0], -1, -1) + self.text_proj(pooled).unsqueeze(1)
        for b in self.blocks:
            a, _ = b['attn'](b['qn'](q), b['kn'](kv), b['kn'](kv), need_weights=False)
            q = q + a
            q = q + b['ffn'](b['fn'](q))
        return self.out(q.flatten(1))


class SpatialSoftmaxReadout(nn.Module):
    """空间softmax关键点读出(Levine 2016; robomimic/Diffusion Policy的ResNet读出)。

    每个关键点通道在全部patch上做softmax，输出期望坐标(x,y)和加权特征。坐标来自固定网格，
    不需要学位置编码；只要某通道的打分在关键patch上变高，坐标梯度就直接存在。
    PatchReadout的可学习查询+可学习位置编码在合成定位任务中多个种子都找不到亮块
    (注意力停留在均匀1/64)，这里用固定坐标避免“注意力与位置编码互相等待”。
    """

    def __init__(self, grid, vis_dim, k, feat_dim, out_dim):
        super().__init__()
        self.grid = grid
        self.norm = nn.LayerNorm(vis_dim)
        self.score = nn.Linear(vis_dim, k)
        self.value = nn.Linear(vis_dim, feat_dim)
        self.log_temp = nn.Parameter(torch.zeros(1))
        coords = torch.linspace(-1, 1, grid)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('xy', torch.stack([xx.flatten(), yy.flatten()], -1))   # [grid*grid,2]，行优先
        self.cls_proj = nn.Linear(vis_dim, feat_dim)
        self.out = nn.Sequential(nn.LayerNorm(k * (2 + feat_dim) + feat_dim), nn.Linear(k * (2 + feat_dim) + feat_dim, out_dim))

    def forward(self, tokens):
        cls, patches = tokens[:, 0], self.norm(tokens[:, 1:])
        if patches.shape[1] != self.grid * self.grid:
            raise ValueError('patch count does not match grid')
        w = torch.softmax(self.score(patches) * self.log_temp.exp(), dim=1)        # [B,P,K]
        keypoints = torch.einsum('bpk,pc->bkc', w, self.xy)                          # [B,K,2]
        feats = torch.einsum('bpk,bpd->bkd', w, self.value(patches))                # [B,K,D]
        z = torch.cat([keypoints.flatten(1), feats.flatten(1), self.cls_proj(self.norm(cls))], dim=-1)
        return self.out(z)


class PatchPolicy(nn.Module):
    def __init__(self, config, *, n_tokens, vis_dim=1024, text_dim=768):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        unknown = set(cfg) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f'Unknown policy config keys: {sorted(unknown)}')
        if cfg['readout'] not in ('patch', 'spatial', 'cls'):
            raise ValueError('readout must be patch, spatial or cls')
        if not set(cfg['history']) <= {'state_delta', 'prev_command'}:
            raise ValueError('history entries must be state_delta / prev_command')
        self.cfg = cfg
        self.fusion = nn.ModuleList([
            CrossAttentionLayer(query_dim=vis_dim, kv_dim=text_dim, attention_dim=cfg['attention_dim'],
                                num_heads=cfg['num_heads'], dropout=cfg['dropout'])
            for _ in range(cfg['num_adapter_layers'])])
        C = cfg['context_dim']
        if cfg['readout'] == 'patch':
            self.readout = PatchReadout(n_tokens, vis_dim, text_dim, cfg['readout_dim'], cfg['num_queries'],
                                        cfg['readout_layers'], cfg['num_heads'], cfg['dropout'], C)
        elif cfg['readout'] == 'spatial':
            self.readout = SpatialSoftmaxReadout(cfg['grid'], vis_dim, cfg['num_keypoints'],
                                                 cfg['keypoint_feat_dim'], C)
        else:
            self.readout = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, C))
        pdim = 2 + (18 if cfg['use_pose18'] else 0)
        pdim += 7 if 'state_delta' in cfg['history'] else 0       # 6 + mask
        pdim += 8 if 'prev_command' in cfg['history'] else 0      # 6 + gripper + mask
        self.proprio = nn.Sequential(nn.Linear(pdim, 256), nn.Mish(), nn.Linear(256, C))
        self.decoder = MultiscaleConditionalUnet1D(action_dim=6, chunk_size=16, context_dim=C,
                                                   down_dims=tuple(cfg['down_dims']), num_steps=cfg['diffusion_steps'])
        self.grip_ctx = nn.Linear(C, 256)
        self.grip_time = nn.Embedding(16, 256)
        self.grip_head = nn.Sequential(nn.Mish(), nn.Linear(256, 256), nn.Mish(), nn.Linear(256, 1))
        betas = cosine_betas(cfg['diffusion_steps'])
        alphas = 1 - betas
        ab = torch.cumprod(alphas, 0)
        ab_prev = torch.cat([torch.ones(1), ab[:-1]])
        self.register_buffer('alpha_bars', ab)
        self.register_buffer('post_c0', betas * ab_prev.sqrt() / (1 - ab))
        self.register_buffer('post_ct', (1 - ab_prev) * alphas.sqrt() / (1 - ab))
        self.register_buffer('post_var', betas * (1 - ab_prev) / (1 - ab))
        self.register_buffer('motion_mean', torch.zeros(6))
        self.register_buffer('motion_std', torch.ones(6))

    # ---------- 条件 ----------
    def proprio_input(self, b, training):
        parts = [b['current2']]
        if self.cfg['use_pose18']:
            parts.append(b['pose18'])
        n = b['current2'].shape[0]
        keep = torch.ones(n, 1, device=b['current2'].device)
        if training and self.cfg['history'] and self.cfg['history_dropout'] > 0:
            keep = (torch.rand(n, 1, device=keep.device) >= self.cfg['history_dropout']).float()
        if 'state_delta' in self.cfg['history']:
            m = b['state_delta_mask'].unsqueeze(-1) * keep
            parts += [b['state_delta'] * m, m]
        if 'prev_command' in self.cfg['history']:
            m = b['prev_command_mask'].unsqueeze(-1) * keep
            parts += [b['prev_command'] * m, m]
        return torch.cat(parts, dim=-1)

    def context(self, b, training=False):
        tokens = b['visual']
        if self.cfg['readout'] == 'cls':
            tokens = tokens[:, :1]          # 与旧CLS读出等价：patch本来就不影响CLS
        for layer in self.fusion:
            tokens = layer(tokens, b['text'], b['text_mask'])
        if self.cfg['readout'] == 'patch':
            ctx = self.readout(tokens, b['text'], b['text_mask'])
        elif self.cfg['readout'] == 'spatial':
            ctx = self.readout(tokens)      # 文本已经由上面的逐token交叉注意力融入每个patch
        else:
            ctx = self.readout(tokens[:, 0])
        return ctx + self.proprio(self.proprio_input(b, training))

    def gripper_logits(self, ctx):
        h = self.grip_ctx(ctx).unsqueeze(1) + self.grip_time.weight.unsqueeze(0)
        return self.grip_head(h).squeeze(-1)

    # ---------- 训练 ----------
    def loss(self, b):
        ctx = self.context(b, training=True)
        x0 = b['motion_z']
        valid = b['motion_valid']
        t = torch.randint(0, self.cfg['diffusion_steps'], (x0.shape[0],), device=x0.device)
        ab = self.alpha_bars[t].view(-1, 1, 1)
        noise = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred = self.decoder(xt, t, ctx)
        motion_loss = (((pred - x0) ** 2) * valid.unsqueeze(-1)).sum() / (valid.sum() * 6).clamp_min(1.0)
        gl = self.gripper_logits(ctx)
        gm = b['gripper_mask']
        bce = F.binary_cross_entropy_with_logits(gl, (b['gripper'] > 0).float(), reduction='none')
        grip_loss = (bce * gm).sum() / gm.sum().clamp_min(1.0)
        total = motion_loss + self.cfg['gripper_loss_weight'] * grip_loss
        return total, {'motion': float(motion_loss.detach()), 'gripper': float(grip_loss.detach())}

    @torch.no_grad()
    def denoise_loss_at(self, b, timesteps, generator):
        """开发集固定t、固定噪声的x0误差，用于学习曲线（不参与模型选择）。"""
        ctx = self.context(b)
        x0, valid = b['motion_z'], b['motion_valid']
        out = []
        for step in timesteps:
            t = torch.full((x0.shape[0],), step, device=x0.device, dtype=torch.long)
            ab = self.alpha_bars[t].view(-1, 1, 1)
            noise = torch.randn(x0.shape, generator=generator, device=x0.device)
            pred = self.decoder(ab.sqrt() * x0 + (1 - ab).sqrt() * noise, t, ctx)
            out.append(float((((pred - x0) ** 2) * valid.unsqueeze(-1)).sum() / (valid.sum() * 6)))
        return out

    # ---------- 采样 ----------
    @torch.no_grad()
    def sample_z(self, ctx, generator, mode='ddpm', ddim_steps=10):
        T = self.cfg['diffusion_steps']
        x = torch.randn((ctx.shape[0], 16, 6), generator=generator, device=ctx.device)
        c = self.cfg['clip_x0']
        if mode == 'ddpm':
            for step in reversed(range(T)):
                t = torch.full((ctx.shape[0],), step, device=ctx.device, dtype=torch.long)
                x0 = self.decoder(x, t, ctx).clamp(-c, c)
                mean = self.post_c0[step] * x0 + self.post_ct[step] * x
                if step:
                    x = mean + self.post_var[step].sqrt() * torch.randn(x.shape, generator=generator, device=x.device)
                else:
                    x = mean
            return x
        # DDIM eta=0
        seq = torch.linspace(T - 1, 0, ddim_steps).round().long().tolist()
        for i, step in enumerate(seq):
            t = torch.full((ctx.shape[0],), step, device=ctx.device, dtype=torch.long)
            ab = self.alpha_bars[step]
            x0 = self.decoder(x, t, ctx).clamp(-c, c)
            if i + 1 == len(seq):
                x = x0
                break
            ab_next = self.alpha_bars[seq[i + 1]]
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x

    def decode8(self, z, grip_logits):
        motion = z * self.motion_std + self.motion_mean
        q = torch_rotvec_to_quat(motion[..., 3:6])
        g = torch.where(grip_logits >= 0, torch.ones_like(grip_logits), -torch.ones_like(grip_logits))
        return torch.cat([motion[..., :3], q, g.unsqueeze(-1)], dim=-1)

    @torch.no_grad()
    def predict8(self, b, generator, mode='ddpm', draws=1, ddim_steps=10, average=False):
        """返回 [draws,B,16,8]；average=True 时在内部运动空间取均值后输出 [1,B,16,8]。"""
        ctx = self.context(b)
        gl = self.gripper_logits(ctx)
        zs = torch.stack([self.sample_z(ctx, generator, mode, ddim_steps) for _ in range(draws)])
        if average:
            zs = zs.mean(0, keepdim=True)
        return torch.stack([self.decode8(z, gl) for z in zs])


class EMA:
    """Diffusion Policy常用的权重滑动平均；评估用EMA权重。"""

    def __init__(self, model, decay=0.999, warmup=1000):
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.warmup, self.n = decay, warmup, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (self.warmup + self.n))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)
