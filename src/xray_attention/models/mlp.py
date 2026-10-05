"""MLP 分类器（小样本友好）。

对应 spec v2.0 §6.1 深度模型。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """多层感知机二分类器。

    Parameters
    ----------
    in_dim : 输入特征维度
    hidden : 隐层宽度列表，如 [32, 16]
    dropout : Dropout 概率（小样本下 0.3-0.5 防过拟合）
    """

    def __init__(self, in_dim: int, hidden: list[int] = [32, 16], dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + list(hidden) + [2]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1).cpu().numpy()
