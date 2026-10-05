"""PyTorch 模型的 sklearn 兼容包装器。

对应 spec v2.0 §6.1：MLP / LSTM 需在 nested LOSO 中与 LR/SVM/RF 统一接口。
包装为 sklearn BaseEstimator 风格，暴露 fit / predict_proba / predict，
并接受 GridSearchCV 的超参搜索。

- MLPSklearnWrapper：window 级聚合特征 → MLP（in_dim 自动推断）
- LSTMSklearnWrapper：把 (n, feat) 当作长度为 1 的序列 → LSTM 退化运行
  （对 window 级聚合特征，LSTM 语义上不理想，此处仅保证 spec 模型列表全覆盖）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

from xray_attention.models.mlp import MLPClassifier
from xray_attention.models.lstm import LSTMClassifier


def _class_weights(y: np.ndarray) -> torch.Tensor:
    """balanced class weights for CrossEntropyLoss。"""
    classes = np.unique(y)
    counts = np.array([(y == c).sum() for c in classes])
    weights = len(y) / (len(classes) * np.maximum(counts, 1))
    return torch.from_numpy(weights).float()


class _TorchTrainerMixin(BaseEstimator, ClassifierMixin):
    """torch 模型通用训练逻辑。子类提供 _build_model(in_dim) 与 _to_model_input(X)。"""

    # 子类应声明的超参（默认值）
    dropout: float = 0.3
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    random_state: int = 42

    def _build_model(self, in_dim: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def _to_model_input(self, X: np.ndarray) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(np.int64)
        in_dim = X.shape[1]
        torch.manual_seed(self.random_state)
        self._model = self._build_model(in_dim)
        self._model.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y))
        n = len(X)
        bs = min(self.batch_size, n)
        for _ in range(self.epochs):
            idx = torch.randperm(n).numpy()
            for s in range(0, n, bs):
                xb = self._to_model_input(X[idx[s : s + bs]])
                yb = torch.from_numpy(y[idx[s : s + bs]])
                opt.zero_grad()
                logits = self._model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        self._model.eval()
        with torch.no_grad():
            xb = self._to_model_input(X)
            logits = self._model(xb)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class MLPSklearnWrapper(_TorchTrainerMixin):
    """MLP 的 sklearn 包装器。hidden 作为 list 可被 GridSearch 搜索。"""

    def __init__(
        self,
        hidden: list[int] = [32, 16],
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None

    def _build_model(self, in_dim: int) -> nn.Module:
        return MLPClassifier(
            in_dim=in_dim,
            hidden=list(self.hidden),
            dropout=self.dropout,
        )

    def _to_model_input(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(X).float()


class LSTMSklearnWrapper(_TorchTrainerMixin):
    """LSTM 的 sklearn 包装器。

    将 (n, feat) reshape 为 (n, 1, feat) 作为长度 1 的序列。
    对 window 级聚合特征是退化应用，仅保证 spec 模型列表全覆盖。
    """

    def __init__(
        self,
        hidden: int = 32,
        num_layers: int = 1,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None

    def _build_model(self, in_dim: int) -> nn.Module:
        return LSTMClassifier(
            in_dim=in_dim,
            hidden=int(self.hidden),
            num_layers=int(self.num_layers),
            dropout=self.dropout,
        )

    def _to_model_input(self, X: np.ndarray) -> torch.Tensor:
        # (n, feat) -> (n, 1, feat)
        return torch.from_numpy(X).float().unsqueeze(1)
