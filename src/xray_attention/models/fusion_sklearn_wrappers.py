"""F1/F2/F3 融合模型的 sklearn 兼容包装器。

对应 spec v2.0 §6.2 融合策略、§7.1 E3 融合对比实验。
包装 EarlyFusion / LateFusion / CrossAttentionFusion 为 BaseEstimator，
使其能在 nested LOSO 的 GridSearchCV 中统一搜索。

输入约定：X 的前 ``gaze_dim`` 列为注视特征，其后 ``face_dim`` 列为面部特征。
gaze_dim / face_dim 在 fit 时自动推断，需在构建多模态矩阵时保证 gaze 列在前。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

from xray_attention.models.cross_attention_fusion import (
    CrossAttentionFusion,
    EarlyFusion,
    LateFusion,
)


def _class_weights(y: np.ndarray) -> torch.Tensor:
    """balanced class weights for CrossEntropyLoss。"""
    classes = np.unique(y)
    counts = np.array([(y == c).sum() for c in classes])
    weights = len(y) / (len(classes) * np.maximum(counts, 1))
    return torch.from_numpy(weights).float()


class _FusionTrainerMixin(BaseEstimator, ClassifierMixin):
    """双流融合模型通用训练逻辑。

    子类需实现 ``_build_model(gaze_dim, face_dim)`` 返回 nn.Module，
    其 forward 签名为 ``forward(gaze: Tensor, face: Tensor) -> logits``。
    """

    # 超参默认值（子类可覆盖）
    dropout: float = 0.3
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    random_state: int = 42
    # gaze_dim 由 fit 时根据外部设置的属性推断；缺省 None 表示需调用者指定
    gaze_dim: int | None = None

    def _build_model(self, gaze_dim: int, face_dim: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def _split_input(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按 gaze_dim 拆分 X 为 (gaze_X, face_X)。"""
        if self.gaze_dim is None:
            raise ValueError("gaze_dim 未设置：请在构造时指定或由外层设置")
        g = int(self.gaze_dim)
        return X[:, :g], X[:, g:]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(np.int64)
        gaze_X, face_X = self._split_input(X)
        gaze_dim = gaze_X.shape[1]
        face_dim = face_X.shape[1]
        torch.manual_seed(self.random_state)
        self._model = self._build_model(gaze_dim, face_dim)
        self._model.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y))
        n = len(X)
        bs = min(self.batch_size, n)
        for _ in range(self.epochs):
            idx = torch.randperm(n).numpy()
            for s in range(0, n, bs):
                bi = idx[s : s + bs]
                g = torch.from_numpy(gaze_X[bi]).float()
                f = torch.from_numpy(face_X[bi]).float()
                yb = torch.from_numpy(y[bi])
                opt.zero_grad()
                logits = self._model(g, f)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        gaze_X, face_X = self._split_input(X)
        self._model.eval()
        with torch.no_grad():
            g = torch.from_numpy(gaze_X).float()
            f = torch.from_numpy(face_X).float()
            logits = self._model(g, f)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class EarlyFusionWrapper(_FusionTrainerMixin):
    """F1 早期融合（特征拼接 → MLP）的 sklearn 包装器。"""

    def __init__(
        self,
        gaze_dim: int | None = None,
        hidden: list[int] = [32],
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.gaze_dim = gaze_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None

    def _build_model(self, gaze_dim: int, face_dim: int) -> nn.Module:
        return EarlyFusion(
            gaze_dim=gaze_dim,
            face_dim=face_dim,
            hidden=list(self.hidden),
            dropout=self.dropout,
        )


class LateFusionWrapper(_FusionTrainerMixin):
    """F2 晚期融合（各模态子模型 → 可学习权重加权）的 sklearn 包装器。"""

    def __init__(
        self,
        gaze_dim: int | None = None,
        hidden: int = 32,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.gaze_dim = gaze_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None

    def _build_model(self, gaze_dim: int, face_dim: int) -> nn.Module:
        return LateFusion(
            gaze_dim=gaze_dim,
            face_dim=face_dim,
            hidden=int(self.hidden),
            dropout=self.dropout,
        )

    def fusion_weights(self) -> np.ndarray:
        """返回当前模型的融合权重 [w_gaze, w_face]。fit 后可用。"""
        if self._model is None:
            return np.array([0.5, 0.5])
        with torch.no_grad():
            return self._model.fusion_weights().cpu().numpy()


class CrossAttentionFusionWrapper(_FusionTrainerMixin):
    """F3 跨模态注意力融合（SOTA 主力）的 sklearn 包装器。"""

    def __init__(
        self,
        gaze_dim: int | None = None,
        d_model: int = 32,
        n_heads: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.gaze_dim = gaze_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None

    def _build_model(self, gaze_dim: int, face_dim: int) -> nn.Module:
        return CrossAttentionFusion(
            gaze_dim=gaze_dim,
            face_dim=face_dim,
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            dropout=self.dropout,
        )
