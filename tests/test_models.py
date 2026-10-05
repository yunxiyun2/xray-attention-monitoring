from __future__ import annotations

import numpy as np
import pytest
import torch

torch.manual_seed(42)

from xray_attention.models.mlp import MLPClassifier
from xray_attention.models.lstm import LSTMClassifier
from xray_attention.models.cross_attention_fusion import (
    EarlyFusion, LateFusion, CrossAttentionFusion,
)
from xray_attention.models.classical import make_classical


# ---------- MLP ----------

def test_mlp_forward_shape():
    model = MLPClassifier(in_dim=20, hidden=[32, 16])
    x = torch.randn(8, 20)
    out = model(x)
    assert out.shape == (8, 2)


def test_mlp_predict_proba():
    model = MLPClassifier(in_dim=10, hidden=[16])
    x = torch.randn(8, 10)
    proba = model.predict_proba(x)
    assert proba.shape == (8, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------- LSTM ----------

def test_lstm_forward_shape():
    model = LSTMClassifier(in_dim=6, hidden=32, num_layers=1)
    # (batch, seq_len, in_dim)
    x = torch.randn(8, 30, 6)
    out = model(x)
    assert out.shape == (8, 2)


# ---------- F1 早期融合 ----------

def test_early_fusion_forward():
    model = EarlyFusion(gaze_dim=6, face_dim=14, hidden=[32])
    g = torch.randn(8, 6)
    f = torch.randn(8, 14)
    out = model(g, f)
    assert out.shape == (8, 2)


# ---------- F2 晚期融合 ----------

def test_late_fusion_forward():
    model = LateFusion(gaze_dim=6, face_dim=14)
    g = torch.randn(8, 6)
    f = torch.randn(8, 14)
    out = model(g, f)
    assert out.shape == (8, 2)


def test_late_fusion_weights_sum_to_one():
    model = LateFusion(gaze_dim=6, face_dim=14)
    w = model.fusion_weights()
    assert abs(w.sum().item() - 1.0) < 1e-5


# ---------- F3 跨模态注意力融合 ----------

def test_cross_attention_fusion_forward():
    model = CrossAttentionFusion(gaze_dim=6, face_dim=14, d_model=32, n_heads=2)
    g = torch.randn(8, 6)
    f = torch.randn(8, 14)
    out = model(g, f)
    assert out.shape == (8, 2)


def test_cross_attention_returns_attention_weights():
    model = CrossAttentionFusion(gaze_dim=6, face_dim=14, d_model=32, n_heads=2)
    g = torch.randn(4, 6)
    f = torch.randn(4, 14)
    out, attn = model.forward_with_attention(g, f)
    assert out.shape == (4, 2)
    # attention 形状：对 gaze query、face key
    assert attn.dim() >= 2


# ---------- 经典 ML 包装 ----------

@pytest.mark.parametrize("name", ["logreg", "svm", "rf"])
def test_classical_factory_returns_sklearn_model(name):
    model = make_classical(name)
    X = np.random.randn(20, 5)
    y = np.array([0] * 10 + [1] * 10)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape[0] == 20
    # 输出概率应在 [0, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_classical_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        make_classical("unknown_model")


# ---------- sklearn 包装器（MLP / LSTM）----------

def test_mlp_sklearn_wrapper_fit_predict():
    from xray_attention.models.sklearn_wrappers import MLPSklearnWrapper
    X = np.random.randn(40, 6)
    y = np.array([0] * 20 + [1] * 20)
    m = MLPSklearnWrapper(hidden=[16], epochs=10, batch_size=8)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (40, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = m.predict(X)
    assert pred.shape == (40,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_lstm_sklearn_wrapper_fit_predict():
    from xray_attention.models.sklearn_wrappers import LSTMSklearnWrapper
    X = np.random.randn(40, 6)
    y = np.array([0] * 20 + [1] * 20)
    m = LSTMSklearnWrapper(hidden=16, epochs=10, batch_size=8)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (40, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------- F1/F2/F3 融合 sklearn 包装器 ----------

def _make_multimodal_dataset(n=40, gaze_dim=6, face_dim=14, seed=0):
    """构造双流数据集：X 前 gaze_dim 列为 gaze，后面为 face。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, gaze_dim + face_dim)
    # 让 gaze 的第 1 列与标签相关，提供可学习信号
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    X[: n // 2, 0] -= 1.5
    X[n // 2 :, 0] += 1.5
    return X, y, gaze_dim


@pytest.mark.parametrize("wrapper_cls,name", [
    ("EarlyFusionWrapper", "F1"),
    ("LateFusionWrapper", "F2"),
    ("CrossAttentionFusionWrapper", "F3"),
])
def test_fusion_wrapper_fit_predict(wrapper_cls, name):
    from xray_attention.models.fusion_sklearn_wrappers import (
        CrossAttentionFusionWrapper, EarlyFusionWrapper, LateFusionWrapper,
    )
    cls = {
        "EarlyFusionWrapper": EarlyFusionWrapper,
        "LateFusionWrapper": LateFusionWrapper,
        "CrossAttentionFusionWrapper": CrossAttentionFusionWrapper,
    }[wrapper_cls]
    X, y, gaze_dim = _make_multimodal_dataset()
    m = cls(gaze_dim=gaze_dim, epochs=10, batch_size=8)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 2), f"{name} proba shape mismatch"
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = m.predict(X)
    assert pred.shape == (len(y),)
    assert set(np.unique(pred)).issubset({0, 1})


def test_late_fusion_wrapper_weights_after_fit():
    from xray_attention.models.fusion_sklearn_wrappers import LateFusionWrapper
    X, y, gaze_dim = _make_multimodal_dataset()
    m = LateFusionWrapper(gaze_dim=gaze_dim, epochs=5, batch_size=8)
    m.fit(X, y)
    w = m.fusion_weights()
    assert w.shape == (2,)
    assert abs(w.sum() - 1.0) < 1e-5


def test_fusion_wrapper_gaze_dim_required():
    """未设置 gaze_dim 时应报错。"""
    from xray_attention.models.fusion_sklearn_wrappers import EarlyFusionWrapper
    X = np.random.randn(20, 10)
    y = np.array([0] * 10 + [1] * 10)
    m = EarlyFusionWrapper(gaze_dim=None, epochs=3)
    with pytest.raises(ValueError):
        m.fit(X, y)

