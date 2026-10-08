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


class Conv1dBlock(nn.Module):
    """Conv1d, GroupNorm and Mish block used by the multiscale control."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.block(features)


class ConditionalResidualBlock1D(nn.Module):
    """Two convolution blocks with FiLM scale and bias conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            (
                Conv1dBlock(in_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            )
        )
        self.condition = nn.Sequential(
            nn.Mish(), nn.Linear(condition_dim, out_channels * 2)
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks[0](features)
        scale, bias = self.condition(condition).unsqueeze(-1).chunk(2, dim=1)
        hidden = scale * hidden + bias
        hidden = self.blocks[1](hidden)
        return hidden + self.residual(features)


class MultiscaleConditionalUnet1D(nn.Module):
    """Three-scale Conditional U-Net control matching Diffusion Policy's layout.

    This is an independently written, optional control.  The existing compact
    decoder remains unchanged so old checkpoints retain identical parameter
    names and behavior.  The layout follows Stanford Diffusion Policy's
    ConditionalUnet1D (MIT): two conditioned residual blocks per scale,
    symmetric skip connections, kernel size five and FiLM scale/bias.
    """

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        context_dim: int = 1024,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        diffusion_step_embed_dim: int = 128,
        kernel_size: int = 5,
        num_steps: int = 100,
    ) -> None:
        super().__init__()
        if len(down_dims) < 2:
            raise ValueError("down_dims must contain at least two scales")
        if chunk_size % (2 ** (len(down_dims) - 1)):
            raise ValueError("chunk_size must support all downsampling stages")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.down_dims = tuple(int(value) for value in down_dims)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        condition_dim = diffusion_step_embed_dim + context_dim
        dimensions = (action_dim,) + self.down_dims
        pairs = tuple(zip(dimensions[:-1], dimensions[1:]))
        self.down_modules = nn.ModuleList()
        for index, (in_channels, out_channels) in enumerate(pairs):
            self.down_modules.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1D(
                            in_channels, out_channels, condition_dim, kernel_size
                        ),
                        ConditionalResidualBlock1D(
                            out_channels, out_channels, condition_dim, kernel_size
                        ),
                        (
                            nn.Conv1d(
                                out_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                            )
                            if index < len(pairs) - 1
                            else nn.Identity()
                        ),
                    )
                )
            )

        middle_channels = self.down_dims[-1]
        self.mid_modules = nn.ModuleList(
            (
                ConditionalResidualBlock1D(
                    middle_channels, middle_channels, condition_dim, kernel_size
                ),
                ConditionalResidualBlock1D(
                    middle_channels, middle_channels, condition_dim, kernel_size
                ),
            )
        )
        self.up_modules = nn.ModuleList()
        for in_channels, out_channels in reversed(pairs[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1D(
                            out_channels * 2,
                            in_channels,
                            condition_dim,
                            kernel_size,
                        ),
                        ConditionalResidualBlock1D(
                            in_channels, in_channels, condition_dim, kernel_size
                        ),
                        nn.ConvTranspose1d(
                            in_channels,
                            in_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    )
                )
            )
        self.final = nn.Sequential(
            Conv1dBlock(self.down_dims[0], self.down_dims[0], kernel_size),
            nn.Conv1d(self.down_dims[0], action_dim, kernel_size=1),
        )

    def forward(
        self,
        noisy_action_chunk: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action_chunk.shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"Expected [B, {self.chunk_size}, {self.action_dim}], "
                f"got {tuple(noisy_action_chunk.shape)}"
            )
        if timestep.shape != (noisy_action_chunk.shape[0],):
            raise ValueError("timestep must have shape [B]")
        if context.ndim != 2 or context.shape[0] != noisy_action_chunk.shape[0]:
            raise ValueError("context must have shape [B, context_dim]")
        condition = torch.cat((self.time_mlp(timestep), context), dim=-1)
        hidden = noisy_action_chunk.transpose(1, 2)
        skips = []
        for first, second, downsample in self.down_modules:
            hidden = first(hidden, condition)
            hidden = second(hidden, condition)
            skips.append(hidden)
            hidden = downsample(hidden)
        for middle in self.mid_modules:
            hidden = middle(hidden, condition)
        for first, second, upsample in self.up_modules:
            skip = skips.pop()
            if hidden.shape[-1] != skip.shape[-1]:
                raise RuntimeError("U-Net skip length mismatch")
            hidden = first(torch.cat((hidden, skip), dim=1), condition)
            hidden = second(hidden, condition)
            hidden = upsample(hidden)
        return self.final(hidden).transpose(1, 2)
