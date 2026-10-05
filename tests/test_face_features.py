from __future__ import annotations

import numpy as np
import pytest
from xray_attention.data.face_feature_extractor import (
    FaceFeatureExtractor,
    calculate_ear,
    calculate_mar,
    _au43_proxy,
)

# ---------- EAR / MAR 几何 ----------

def test_ear_open_eye_large():
    # 睁眼：上下眼睑分得开，EAR 大
    # 顺序：p1 左角, p2 上左, p3 上右, p4 右角, p5 下右, p6 下左
    open_eye = np.array([
        [-1.0, 0.0], [ -0.5, 1.0], [0.5, 1.0],
        [1.0, 0.0], [0.5, -1.0], [-0.5, -1.0],
    ])
    ear = calculate_ear(open_eye)
    assert ear > 0.3


def test_ear_closed_eye_near_zero():
    # 闭眼：上下眼睑几乎重合，EAR ≈ 0
    closed_eye = np.array([
        [-1.0, 0.0], [-0.5, 0.01], [0.5, 0.01],
        [1.0, 0.0], [0.5, -0.01], [-0.5, -0.01],
    ])
    ear = calculate_ear(closed_eye)
    assert ear < 0.05


def test_ear_zero_when_h_zero():
    # 退化：水平距离为 0 时不报错，返回 0
    degenerate = np.zeros((6, 2))
    assert calculate_ear(degenerate) == 0.0


def test_mar_open_mouth_large():
    open_mouth = np.array([
        [-1.0, 0.0], [-0.5, 2.0], [0.5, 2.0],
        [1.0, 0.0], [0.5, -2.0], [-0.5, -2.0],
    ])
    assert calculate_mar(open_mouth) > 0.3


def test_mar_closed_mouth_small():
    closed_mouth = np.array([
        [-1.0, 0.0], [-0.5, 0.01], [0.5, 0.01],
        [1.0, 0.0], [0.5, -0.01], [-0.5, -0.01],
    ])
    assert calculate_mar(closed_mouth) < 0.05


# ---------- PERCLOS / micro-sleep / blink ----------

def test_perclos_below_threshold_counts():
    ear_series = np.array([0.3, 0.05, 0.04, 0.3, 0.02])  # 3/5 低于 0.1
    extractor = FaceFeatureExtractor()
    perclos = extractor.perclos(ear_series, ear_threshold=0.1)
    assert abs(perclos - 0.6) < 1e-6


def test_perclos_none_below_returns_zero():
    ear_series = np.array([0.3, 0.4, 0.5])
    extractor = FaceFeatureExtractor()
    assert extractor.perclos(ear_series, ear_threshold=0.1) == 0.0


def test_long_closure_count_microsleep():
    # 1s @ 30fps = 30 帧；构造 35 帧连续闭眼 → 1 次 micro-sleep
    ear_series = np.where(np.arange(100) < 35, 0.02, 0.3)
    extractor = FaceFeatureExtractor()
    cnt = extractor.long_closure_count(
        ear_series, ear_threshold=0.1, min_frames=30, fps=30
    )
    assert cnt == 1


def test_long_closure_count_no_microsleep():
    # 仅短暂闭眼（< min_frames）
    ear_series = np.where(np.arange(100) < 10, 0.02, 0.3)
    extractor = FaceFeatureExtractor()
    cnt = extractor.long_closure_count(
        ear_series, ear_threshold=0.1, min_frames=30, fps=30
    )
    assert cnt == 0


def test_detect_blinks_returns_events():
    # 构造 3 次短暂闭眼
    ear = np.full(300, 0.3)
    for s in [50, 150, 250]:
        ear[s : s + 5] = 0.02
    extractor = FaceFeatureExtractor()
    blinks = extractor.detect_blinks(ear, ear_threshold=0.1)
    assert len(blinks) == 3
    for b in blinks:
        assert b["duration_frames"] == 5
        assert b["amplitude"] > 0


# ---------- AU 近似 ----------

def test_au43_proxy_closed_eye_high():
    closed_eye = np.array([
        [-1.0, 0.0], [-0.5, 0.01], [0.5, 0.01],
        [1.0, 0.0], [0.5, -0.01], [-0.5, -0.01],
    ])
    open_eye = np.array([
        [-1.0, 0.0], [-0.5, 1.0], [0.5, 1.0],
        [1.0, 0.0], [0.5, -1.0], [-0.5, -1.0],
    ])
    au_closed = _au43_proxy(closed_eye)
    au_open = _au43_proxy(open_eye)
    assert au_closed > au_open
    assert 0.0 <= au_open <= 1.0
    assert 0.0 <= au_closed <= 1.0


# ---------- 窗口级聚合 ----------

def test_extract_window_features_basic():
    # 60 帧的 EAR 序列，前 30 帧闭眼
    ear_series = np.where(np.arange(60) < 30, 0.02, 0.3)
    extractor = FaceFeatureExtractor()
    feat = extractor.extract_window_features(ear_series, ear_threshold=0.1, fps=30)
    assert abs(feat["perclos_80"] - 0.5) < 1e-6
    assert feat["long_closure_count"] == 1  # 30 帧 = 1s 持续闭眼
    assert feat["blink_count"] == 0         # 不是眨眼（太长）
    assert feat["blink_rate"] == 0.0
    assert feat["ear_mean"] > 0
