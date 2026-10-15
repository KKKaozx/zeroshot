"""Cross-attention adapter between frozen CLIP vision and text encoders."""

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """让视觉 token 查询文本 token，再通过残差连接保留原视觉信息。"""
    def __init__(
        self,
        query_dim: int = 1024,
        kv_dim: int = 768,
        attention_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(query_dim)
        self.text_norm = nn.LayerNorm(kv_dim)
        self.query_projection = nn.Linear(query_dim, attention_dim)
        self.text_projection = nn.Linear(kv_dim, attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(attention_dim, query_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 2, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Q 来自图像，K/V 来自语言：每个视觉区域都能选择与自己相关的词。
        query = self.query_projection(self.query_norm(visual_tokens))
        key_value = self.text_projection(self.text_norm(text_tokens))
        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = ~text_attention_mask.bool()
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        # 两个残差块分别承担跨模态融合和非线性特征变换。
        visual_tokens = visual_tokens + self.attention_dropout(
            self.output_projection(attended)
        )
        return visual_tokens + self.ffn(self.ffn_norm(visual_tokens))


class CrossAttentionAdapter(nn.Module):
    """串联多层交叉注意力，最后输出一个融合后的上下文向量。"""
    def __init__(
        self,
        num_layers: int = 8,
        embed_dim: int = 1024,
        text_dim: int = 768,
        attention_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "cls_patch_mean"}:
            raise ValueError(f"Unknown adapter pooling: {pooling}")
        self.pooling = pooling
        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    query_dim=embed_dim,
                    kv_dim=text_dim,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused_tokens = visual_tokens
        for layer in self.layers:
            fused_tokens = layer(fused_tokens, text_tokens, text_attention_mask)
        # 旧权重保留 CLS-only 语义。新模式同时读出局部 patch，使局部图文
        # 融合进入动作条件；这里没有新增视觉 self-attention，也不增加参数。
        if self.pooling == "cls_patch_mean":
            if fused_tokens.shape[1] < 2:
                raise ValueError("Patch pooling requires CLS plus at least one patch token")
            pooled = .5 * (fused_tokens[:, 0] + fused_tokens[:, 1:].mean(dim=1))
            return self.output_norm(pooled)
        return self.output_norm(fused_tokens[:, 0])
