"""A compact temporal U-Net denoiser for action diffusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """把离散扩散时间步编码为连续向量。"""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        frequency = math.log(10000) / max(half_dim - 1, 1)
        frequency = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -frequency
        )
        angles = timesteps.float()[:, None] * frequency[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class ConditionalResidualBlock(nn.Module):
    """利用时间与视觉语言条件，对动作特征进行缩放和平移。"""
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.condition = nn.Linear(condition_dim, channels * 2)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = features
        hidden = self.conv1(F.mish(self.norm1(features)))
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        hidden = self.conv2(F.mish(self.norm2(hidden)))
        return residual + hidden


class ConditionalDiffusionDecoder(nn.Module):
    """沿动作序列时间轴运行的一维 U-Net 降噪器。"""

    def __init__(
        self,
        action_dim: int = 8,
        chunk_size: int = 16,
        context_dim: int = 1024,
        hidden_dim: int = 256,
        num_steps: int = 100,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_projection = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)
        self.down_block = ConditionalResidualBlock(hidden_dim, hidden_dim)
        self.downsample = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.mid_block1 = ConditionalResidualBlock(hidden_dim * 2, hidden_dim)
        self.mid_block2 = ConditionalResidualBlock(hidden_dim * 2, hidden_dim)
        self.upsample = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.up_block = ConditionalResidualBlock(hidden_dim, hidden_dim)
        self.output_projection = nn.Sequential(
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1),
        )

    def forward(
        self,
        noisy_action_chunk: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action_chunk.ndim != 3 or noisy_action_chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected [B, T, {self.action_dim}], got {tuple(noisy_action_chunk.shape)}"
            )
        # 将“当前扩散步”和“视觉语言上下文”合并为同一个条件向量。
        condition = self.time_mlp(timestep) + self.context_mlp(context)
        features = self.input_projection(noisy_action_chunk.transpose(1, 2))
        # 下采样捕获长时间依赖；skip connection 保留逐时刻的精细动作信息。
        skip = self.down_block(features, condition)
        hidden = self.downsample(skip)
        hidden = self.mid_block1(hidden, condition)
        hidden = self.mid_block2(hidden, condition)
        hidden = F.interpolate(hidden, size=skip.shape[-1], mode="linear", align_corners=False)
        hidden = self.upsample(hidden) + skip
        hidden = self.up_block(hidden, condition)
        return self.output_projection(hidden).transpose(1, 2)
