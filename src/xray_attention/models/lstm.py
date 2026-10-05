"""LSTM 分类器（窗口序列模型）。

对应 spec v2.0 §6.1 深度模型。
输入：(batch, seq_len, in_dim)，seq_len 为窗口内帧数。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, in_dim)
        out, (h_n, _) = self.lstm(x)
        # 取最后时刻隐状态
        last = out[:, -1, :]
        last = self.drop(self.norm(last))
        return self.fc(last)
