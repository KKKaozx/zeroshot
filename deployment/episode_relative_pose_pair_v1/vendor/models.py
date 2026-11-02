"""Frozen CLIP backbone, trainable adapter, and DDPM action decoder."""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from adapter import CrossAttentionAdapter
from diffusion_decoder import ConditionalDiffusionDecoder, MultiscaleConditionalUnet1D


def extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = values.gather(0, timesteps)
    return result.reshape(timesteps.shape[0], *((1,) * (target.ndim - 1)))


def diffusion_betas(steps: int, schedule: str) -> torch.Tensor:
    """100 步线性表未接近纯噪声；新训练采用 Diffusion Policy 的余弦表。"""
    if steps < 2:
        raise ValueError("diffusion steps must be at least 2")
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, steps, dtype=torch.float32)
    if schedule != "squaredcos_cap_v2":
        raise ValueError(f"Unknown beta schedule: {schedule}")
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    cumulative = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    return (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999).float()


def build_gripper_readout(hidden_dim: int, head_type: str) -> nn.Sequential:
    """只替换夹爪读出网络；输入投影、时间编码和开合定义保持不变。"""
    if head_type == "legacy":
        return nn.Sequential(nn.LayerNorm(hidden_dim), nn.Mish(), nn.Linear(hidden_dim, 1))
    if head_type == "mlp2":
        return nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
    raise ValueError(f"Unknown gripper head type: {head_type}")


class RobotAdapterModel(nn.Module):
    """冻结 CLIP，只训练跨模态 Adapter 和扩散动作头。"""
    def __init__(self, config: Dict[str, Any], cache_dir: Optional[str] = None) -> None:
        super().__init__()
        model_config = config["model"]
        model_name = str(model_config["name"])
        self.action_dim = int(model_config.get("action_dim", 8))
        self.chunk_size = int(model_config.get("chunk_size", 16))
        self.decoder_type = str(model_config.get("decoder_type", "diffusion"))
        if self.decoder_type not in {"diffusion", "regression"}:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")
        self.num_diffusion_steps = int(model_config.get("num_diffusion_steps", 100))
        # 缺少此配置的旧 checkpoint 继续使用旧表，禁止只改推理噪声表。
        self.beta_schedule = str(model_config.get("beta_schedule", "linear"))
        self.clip_denoised = bool(model_config.get("clip_denoised", False))
        self.diffusion_prediction_type = str(model_config.get("diffusion_prediction_type", "epsilon"))
        if self.diffusion_prediction_type not in {"epsilon", "sample"}:
            raise ValueError("diffusion_prediction_type must be epsilon or sample")
        self.diffusion_architecture = str(
            model_config.get("diffusion_architecture", "compact")
        )
        if self.diffusion_architecture not in {"compact", "multiscale"}:
            raise ValueError(
                "diffusion_architecture must be compact or multiscale"
            )
        self.separate_gripper_head = bool(
            model_config.get("separate_gripper_head", False)
        )
        self.trajectory_conditioned_gripper = bool(
            model_config.get("trajectory_conditioned_gripper", False)
        )
        # 缺省仍是旧头，旧权重的参数名称/形状不变。
        self.gripper_head_type = str(model_config.get("gripper_head_type", "legacy"))
        if self.gripper_head_type not in {"legacy", "mlp2"}:
            raise ValueError("gripper_head_type must be legacy or mlp2")
        if self.gripper_head_type != "legacy" and not self.trajectory_conditioned_gripper:
            raise ValueError("mlp2 requires trajectory-conditioned gripper features")
        self.condition_on_current_gripper = bool(
            model_config.get("condition_on_current_gripper", False)
        )
        self.gripper_target_mode = str(
            model_config.get("gripper_target_mode", "state")
        )
        if self.gripper_target_mode not in {"state", "transition"}:
            raise ValueError(
                "gripper_target_mode must be 'state' or 'transition', got "
                f"{self.gripper_target_mode!r}"
            )
        if (model_config.get("current_gripper_encoding") in {"bridge_measured_open_fraction_v1", "bridge_measured_affine_unbounded_v1"}
                and self.gripper_target_mode != "state"):
            raise ValueError("Bridge连续测量不能作为transition先前命令，必须预测绝对state命令")
        self.gripper_transition_decode = str(
            model_config.get("gripper_transition_decode", "cumulative")
        )
        if self.gripper_transition_decode not in {"cumulative", "single_switch"}:
            raise ValueError(
                "gripper_transition_decode must be 'cumulative' or "
                f"'single_switch', got {self.gripper_transition_decode!r}"
            )
        self.gripper_switch_threshold = float(
            model_config.get("gripper_switch_threshold", 0.0)
        )
        self.gripper_loss_weight = float(model_config.get("gripper_loss_weight", 0.25))
        self.regression_rotation_weight = float(model_config.get("regression_rotation_weight", 1.0))
        if not math.isfinite(self.regression_rotation_weight) or self.regression_rotation_weight <= 0:
            raise ValueError("Regression rotation weight must be finite and positive")
        self.gripper_change_weight = float(
            model_config.get("gripper_change_weight", 1.0)
        )
        self.balanced_gripper_loss = bool(
            model_config.get("balanced_gripper_loss", True)
        )
        action_config = config.get("action", {})
        self.max_normalized_position = action_config.get("max_normalized_position")

        # CLIP 负责提取通用视觉/语言特征，其参数在本项目中保持冻结。
        clip_model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.vision_encoder = clip_model.vision_model
        self.text_encoder = clip_model.text_model
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        vision_dim = int(self.vision_encoder.config.hidden_size)
        text_dim = int(self.text_encoder.config.hidden_size)
        context_dim = vision_dim
        self.fusion_type = str(model_config.get("fusion_type", "cross_attention"))
        if self.fusion_type == "cross_attention":
            self.adapter_pooling = str(model_config.get("adapter_pooling", "cls"))
            self.adapter = CrossAttentionAdapter(
                num_layers=int(model_config.get("num_adapter_layers", 8)),
                embed_dim=vision_dim,
                text_dim=text_dim,
                attention_dim=int(model_config.get("attention_dim", 512)),
                num_heads=int(model_config.get("num_attention_heads", 8)),
                dropout=float(model_config.get("dropout", 0.1)),
                pooling=self.adapter_pooling,
            )
        elif self.fusion_type == "simple_concat":
            # 不做 token 级交叉注意力：只拼接 CLIP 的全局图像/文本向量，
            # 再投影回同一上下文维度，作为 Adapter 的低成本公平基线。
            self.simple_fusion = nn.Sequential(
                nn.LayerNorm(vision_dim + text_dim),
                nn.Linear(vision_dim + text_dim, vision_dim),
                nn.GELU(),
                nn.LayerNorm(vision_dim),
            )
        else:
            raise ValueError(
                "fusion_type must be 'cross_attention' or 'simple_concat', got "
                f"{self.fusion_type!r}"
            )
        diffusion_action_dim = self.action_dim - 1 if self.separate_gripper_head else self.action_dim
        decoder_hidden_dim = int(model_config.get("decoder_hidden_dim", 256))
        if self.decoder_type == "regression":
            if not self.separate_gripper_head:
                raise ValueError("Regression baseline requires a separate gripper head")
            # 诊断对照：同样的 CLIP/Adapter 上下文直接输出整块位姿，没有加噪/采样。
            self.regression_head = nn.Sequential(
                nn.LayerNorm(context_dim), nn.Linear(context_dim, decoder_hidden_dim),
                nn.Mish(), nn.Linear(decoder_hidden_dim, self.chunk_size * 7),
            )
            with torch.no_grad():
                self.regression_head[-1].bias.zero_()
                self.regression_head[-1].bias.reshape(self.chunk_size, 7)[:, 6] = 1
        else:
            if self.diffusion_architecture == "multiscale":
                self.diffusion_decoder = MultiscaleConditionalUnet1D(
                    action_dim=diffusion_action_dim,
                    chunk_size=self.chunk_size,
                    context_dim=context_dim,
                    down_dims=tuple(
                        int(value)
                        for value in model_config.get(
                            "diffusion_down_dims", (256, 512, 1024)
                        )
                    ),
                    diffusion_step_embed_dim=int(
                        model_config.get("diffusion_step_embed_dim", 128)
                    ),
                    kernel_size=int(
                        model_config.get("diffusion_kernel_size", 5)
                    ),
                    num_steps=self.num_diffusion_steps,
                )
            else:
                self.diffusion_decoder = ConditionalDiffusionDecoder(
                    action_dim=diffusion_action_dim,
                    chunk_size=self.chunk_size,
                    context_dim=context_dim,
                    hidden_dim=decoder_hidden_dim,
                    num_steps=self.num_diffusion_steps,
                )
        if self.separate_gripper_head:
            if self.trajectory_conditioned_gripper:
                # 每个夹爪状态同时查看视觉语言上下文、对应的7维位姿目标和
                # 动作块时间位置，从而学习“到达抓取点时闭合”而非数据集多数类。
                self.gripper_context_projection = nn.Linear(
                    context_dim, decoder_hidden_dim
                )
                self.gripper_pose_projection = nn.Linear(7, decoder_hidden_dim)
                if self.condition_on_current_gripper:
                    self.current_gripper_projection = nn.Linear(
                        1, decoder_hidden_dim
                    )
                self.gripper_time_embedding = nn.Embedding(
                    self.chunk_size, decoder_hidden_dim
                )
                self.gripper_head = build_gripper_readout(decoder_hidden_dim, self.gripper_head_type)
            else:
                # 兼容读取 v2/v3 checkpoint 的旧夹爪头。
                self.gripper_head = nn.Sequential(
                    nn.Linear(context_dim, decoder_hidden_dim),
                    nn.Mish(),
                    nn.Linear(decoder_hidden_dim, self.chunk_size),
                )

        betas = diffusion_betas(self.num_diffusion_steps, self.beta_schedule)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer("posterior_mean_x0", betas * alpha_bars_previous.sqrt() / (1 - alpha_bars))
        self.register_buffer("posterior_mean_xt", (1 - alpha_bars_previous) * alphas.sqrt() / (1 - alpha_bars))

    def train(self, mode: bool = True) -> "RobotAdapterModel":
        super().train(mode)
        # Frozen encoders must remain deterministic even while the adapter trains.
        self.vision_encoder.eval()
        self.text_encoder.eval()
        return self

    def get_context_vector(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            visual_output = self.vision_encoder(pixel_values=pixel_values)
            text_output = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        if self.fusion_type == "cross_attention":
            return self.adapter(
                visual_output.last_hidden_state,
                text_output.last_hidden_state,
                attention_mask,
            )
        return self.simple_fusion(
            torch.cat(
                [visual_output.pooler_output, text_output.pooler_output], dim=-1
            )
        )

    def diffusion_loss(
        self,
        actions: torch.Tensor,
        context: torch.Tensor,
        current_gripper: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, ...]:
        if actions.shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"Expected actions [B, {self.chunk_size}, {self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )
        diffusion_targets = actions[..., :7] if self.separate_gripper_head else actions
        if self.decoder_type == "regression":
            poses = self.regression_head(context).reshape(-1, self.chunk_size, 7)
            return poses, diffusion_targets, self.predict_gripper_logits(context, actions[..., :7], current_gripper)
        batch_size = diffusion_targets.shape[0]
        timesteps = torch.randint(
            0, self.num_diffusion_steps, (batch_size,), device=actions.device
        )
        # DDPM 训练目标：随机给真实动作加噪，再让网络预测加入的噪声。
        noise = torch.randn_like(diffusion_targets)
        alpha_bar = extract(self.alpha_bars, timesteps, diffusion_targets)
        noisy_actions = (
            alpha_bar.sqrt() * diffusion_targets
            + (1.0 - alpha_bar).sqrt() * noise
        )
        predicted_noise = self.diffusion_decoder(noisy_actions, timesteps, context)
        training_target = diffusion_targets if self.diffusion_prediction_type == "sample" else noise
        if self.separate_gripper_head:
            return (
                predicted_noise,
                training_target,
                self.predict_gripper_logits(
                    context, actions[..., :7], current_gripper
                ),
            )
        return predicted_noise, training_target

    def predict_gripper_logits(
        self,
        context: torch.Tensor,
        pose_actions: torch.Tensor,
        current_gripper: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict state logits or switch-event logits for an action chunk."""
        if not self.separate_gripper_head:
            raise RuntimeError("This checkpoint has no separate gripper head")
        if self.trajectory_conditioned_gripper:
            if pose_actions.shape[1:] != (self.chunk_size, 7):
                raise ValueError(
                    f"Expected pose actions [B, {self.chunk_size}, 7], "
                    f"got {tuple(pose_actions.shape)}"
                )
            time_indices = torch.arange(
                self.chunk_size, device=pose_actions.device
            )
            features = (
                self.gripper_context_projection(context).unsqueeze(1)
                + self.gripper_pose_projection(pose_actions)
                + self.gripper_time_embedding(time_indices).unsqueeze(0)
            )
            if self.condition_on_current_gripper:
                if current_gripper is None:
                    raise ValueError(
                        "Current gripper state is required by this checkpoint"
                    )
                current_gripper = current_gripper.reshape(context.shape[0], 1)
                features = features + self.current_gripper_projection(
                    current_gripper
                ).unsqueeze(1)
            return self.gripper_head(features).squeeze(-1)
        return self.gripper_head(context)

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        current_gripper: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        diffusion_action_dim = self.action_dim - 1 if self.separate_gripper_head else self.action_dim
        if self.decoder_type == "regression":
            actions = self.regression_head(context).reshape(batch_size, self.chunk_size, 7)
        else:
            actions = torch.randn(
                batch_size, self.chunk_size, diffusion_action_dim, device=context.device,
            )
        # 从纯高斯噪声开始，按时间步反向迭代得到完整动作轨迹。
        for step in reversed(range(self.num_diffusion_steps if self.decoder_type == "diffusion" else 0)):
            timesteps = torch.full(
                (batch_size,), step, device=context.device, dtype=torch.long
            )
            predicted_noise = self.diffusion_decoder(actions, timesteps, context)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            if self.diffusion_prediction_type == "sample":
                # 直接预测x0仍然是扩散模型：输入含噪动作与t，反向过程仍逐步采样。
                predicted_noise = (actions - alpha_bar.sqrt() * predicted_noise) / (1 - alpha_bar).sqrt()
            mean = (
                actions
                - (1.0 - alpha) * predicted_noise / (1.0 - alpha_bar).sqrt()
            ) / alpha.sqrt()
            if self.clip_denoised:
                # 与 DDPM clip_sample 一致：裁剪估计的干净动作，而非只在末尾
                # 裁剪爆炸输出。位置沿用本项目归一化范围，四元数分量在[-1,1]。
                clean = (actions - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
                position_limit = float(self.max_normalized_position or 3.0)
                clean = torch.cat([
                    clean[..., :3].clamp(-position_limit, position_limit),
                    clean[..., 3:].clamp(-1, 1),
                ], dim=-1)
                mean = self.posterior_mean_x0[step] * clean + self.posterior_mean_xt[step] * actions
            if step > 0:
                actions = mean + self.posterior_variance[step].sqrt() * torch.randn_like(actions)
            else:
                actions = mean

        # 扩散输出本身无约束；控制前限制相对位移离群值、重新单位化四元数，
        # 并限制夹爪范围。旧版绝对坐标 checkpoint 没有该配置，因此不裁剪 xyz。
        if self.max_normalized_position is not None:
            position_limit = float(self.max_normalized_position)
            actions[..., :3] = actions[..., :3].clamp(
                -position_limit, position_limit
            )
        quaternion = actions[..., 3:7]
        identity = torch.zeros_like(quaternion)
        identity[..., 3] = 1.0
        quaternion = torch.where(
            quaternion.norm(dim=-1, keepdim=True) > 1e-6,
            F.normalize(quaternion, dim=-1),
            identity,
        )
        actions[..., 3:7] = quaternion
        if self.separate_gripper_head:
            gripper_logits = self.predict_gripper_logits(
                context, actions, current_gripper
            )
            if self.gripper_target_mode == "transition":
                if current_gripper is None:
                    raise ValueError(
                        "Current gripper state is required for transition decoding"
                    )
                # 预测每一步是否翻转状态，再从输入时刻状态累计还原绝对开/合。
                if self.gripper_transition_decode == "single_switch":
                    # 绝大多数16步窗口至多包含一次真实开合。只保留置信度最高
                    # 且超过阈值的事件，避免多个误报不断翻转后续状态。
                    event_logits, event_indices = gripper_logits.max(dim=1)
                    switch_events = torch.zeros_like(
                        gripper_logits, dtype=torch.bool
                    )
                    switch_events.scatter_(
                        1,
                        event_indices.unsqueeze(1),
                        (event_logits >= self.gripper_switch_threshold).unsqueeze(1),
                    )
                else:
                    switch_events = (
                        gripper_logits >= self.gripper_switch_threshold
                    )
                initial_open = current_gripper.reshape(batch_size, 1) >= 0.0
                parity = torch.cumsum(switch_events.to(torch.int64), dim=1) % 2 == 1
                open_states = torch.logical_xor(initial_open, parity)
                gripper = torch.where(
                    open_states,
                    torch.ones_like(gripper_logits),
                    -torch.ones_like(gripper_logits),
                )
            else:
                gripper = torch.where(
                    gripper_logits >= 0.0,
                    torch.ones_like(gripper_logits),
                    -torch.ones_like(gripper_logits),
                )
            actions = torch.cat([actions, gripper.unsqueeze(-1)], dim=-1)
        else:
            actions[..., 7] = actions[..., 7].clamp(-1.0, 1.0)
        return actions

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        current_gripper: torch.Tensor | None = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...] | torch.Tensor:
        context = self.get_context_vector(pixel_values, input_ids, attention_mask)
        if actions is not None:
            return self.diffusion_loss(actions, context, current_gripper)
        return self.sample(context, current_gripper)
