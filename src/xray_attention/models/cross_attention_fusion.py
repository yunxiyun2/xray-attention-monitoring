"""三档融合模型（F1 / F2 / F3）。

对应 spec v2.0 §6.2 融合策略：
- F1 早期融合：特征拼接 → MLP（naïve baseline）
- F2 晚期融合：各模态独立子模型 → 学习权重加权（决策级）
- F3 跨模态注意力：双流编码 → Cross-Attention → 分类头（SOTA 主力）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusion(nn.Module):
    """F1：早期融合（特征拼接 + MLP）。"""

    def __init__(self, gaze_dim: int, face_dim: int, hidden: list[int] = [32], dropout: float = 0.3):
        super().__init__()
        in_dim = gaze_dim + face_dim
        dims = [in_dim] + list(hidden) + [2]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, gaze: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        x = torch.cat([gaze, face], dim=-1)
        return self.net(x)


class LateFusion(nn.Module):
    """F2：晚期融合（各模态子模型 → 可学习权重加权）。"""

    def __init__(self, gaze_dim: int, face_dim: int, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.gaze_head = _MLPHead(gaze_dim, hidden, dropout)
        self.face_head = _MLPHead(face_dim, hidden, dropout)
        # 可学习融合权重（softmax 后为概率权重）
        self.w = nn.Parameter(torch.tensor([0.5, 0.5]))

    def fusion_weights(self) -> torch.Tensor:
        return F.softmax(self.w, dim=0)

    def forward(self, gaze: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        p_gaze = self.gaze_head(gaze)        # (B, 2) logits
        p_face = self.face_head(face)
        w = self.fusion_weights()
        return w[0] * p_gaze + w[1] * p_face


class CrossAttentionFusion(nn.Module):
    """F3：跨模态注意力融合（SOTA 主力）。

    双流经线性投影到 d_model → 双向 Cross-Attention → LayerNorm + Linear。
    """

    def __init__(
        self,
        gaze_dim: int,
        face_dim: int,
        d_model: int = 32,
        n_heads: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gaze_proj = nn.Linear(gaze_dim, d_model)
        self.face_proj = nn.Linear(face_dim, d_model)

        # 双向 cross-attention：gaze→face 与 face→gaze
        self.attn_g2f = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_f2g = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(2 * d_model)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * d_model, 2)

    def forward(self, gaze: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        out, _ = self._forward_core(gaze, face)
        return out

    def forward_with_attention(self, gaze: torch.Tensor, face: torch.Tensor):
        """返回 (logits, attention_weights) 用于可视化。"""
        out, attn = self._forward_core(gaze, face, return_attn=True)
        return out, attn

    def _forward_core(self, gaze: torch.Tensor, face: torch.Tensor, return_attn: bool = False):
        # 投影到 d_model，作为单 token 序列 (B, 1, d_model)
        g = self.gaze_proj(gaze).unsqueeze(1)
        f = self.face_proj(face).unsqueeze(1)

        # gaze 查询 face
        g_attn, w_g2f = self.attn_g2f(query=g, key=f, value=f, need_weights=return_attn)
        # face 查询 gaze
        f_attn, w_f2g = self.attn_f2g(query=f, key=g, value=g, need_weights=return_attn)

        # 拼接两流 attended 表示
        merged = torch.cat([g_attn.squeeze(1), f_attn.squeeze(1)], dim=-1)
        merged = self.drop(self.norm(merged))
        logits = self.fc(merged)

        if return_attn:
            return logits, w_g2f  # 返回 gaze→face 注意力矩阵
        return logits, None


class _MLPHead(nn.Module):
    """单模态子网络：MLP 输出 2 类 logits。"""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
