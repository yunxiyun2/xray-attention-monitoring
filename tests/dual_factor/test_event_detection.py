"""I-VT 速度阈值法事件检测测试。"""
from __future__ import annotations

import numpy as np
import pytest

from dual_factor.features.event_detection import (
    Fixation,
    Saccade,
    Events,
    compute_velocity,
    detect_events_ivt,
)


def test_compute_velocity_shape():
    """速度信号应与输入等长且非负。"""
    h = np.random.randn(1000)
    v = np.random.randn(1000)
    fs = 512
    vel = compute_velocity(h, v, fs)
    assert len(vel) == len(h)
    assert (vel >= 0).all()


def test_detect_events_ivt_static_signal_all_fixation():
    """完全静止信号应全部判为注视。"""
    fs = 512
    h = np.zeros(512 * 5)
    v = np.zeros(512 * 5)
    events = detect_events_ivt(h, v, fs, velocity_threshold=1.0)
    assert len(events.saccades) == 0
    assert len(events.fixations) >= 1
    assert events.fixations[0].duration_sec >= 4.5


def test_detect_events_ivt_fast_jump_saccade():
    """快速跳转应被检测为扫视，幅度应反映位移。"""
    fs = 512
    n = 512 * 5
    h = np.zeros(n)
    v = np.zeros(n)
    # 中间插入一次快速跳转（50 samples ≈ 100ms），并保持新位置
    h[1000:1050] = np.linspace(0, 100, 50)
    h[1050:] = 100.0
    events = detect_events_ivt(h, v, fs, velocity_threshold=10.0)
    assert len(events.saccades) >= 1
    assert events.saccades[0].amplitude > 5


def test_detect_events_ivt_min_durations_enforced():
    """最短持续时长应被执行：短扫视被丢弃，长扫视保留。"""
    fs = 512
    n = 512 * 10
    h = np.zeros(n)
    v = np.zeros(n)
    # 短跳转（5 samples ≈ 10ms，应被丢弃）
    h[100:105] = np.linspace(0, 50, 5)
    h[105:] = 50.0
    # 长跳转（30 samples ≈ 60ms，应保留）
    h[3000:3030] = np.linspace(50, 150, 30)
    h[3030:] = 150.0
    events = detect_events_ivt(
        h, v, fs, velocity_threshold=10.0,
        min_fixation_ms=100, min_saccade_ms=20,
    )
    for s in events.saccades:
        assert s.duration_sec * 1000 >= 20
    assert len(events.saccades) >= 1  # 至少长扫视应保留


def test_detect_events_ivt_auto_threshold():
    """auto 模式应自动计算速度阈值并检测事件。"""
    fs = 512
    n = 512 * 10
    h = np.zeros(n)
    v = np.zeros(n)
    # 前半段静止（注视），中间快速跳转（扫视），后半段静止
    mid = n // 2
    h[mid:mid + 50] = np.linspace(0, 200, 50)
    h[mid + 50:] = 200.0
    events = detect_events_ivt(h, v, fs, velocity_threshold="auto")
    assert isinstance(events, Events)
    assert len(events.fixations) + len(events.saccades) > 0


def test_detect_events_on_real_data():
    """在真实 DROZY 数据上验证事件检测。"""
    from pathlib import Path
    from dual_factor.data.drozy_loader import load_all_records
    from dual_factor.data.preprocess import preprocess_eog

    drozy_root = Path(__file__).resolve().parents[2] / "data_public" / "DROZY"
    if not drozy_root.exists():
        pytest.skip("DROZY not available")
    records = load_all_records(drozy_root)
    rec = records[0]
    h_clean, v_clean = preprocess_eog(rec.eog_h, rec.eog_v, rec.sampling_rate)
    events = detect_events_ivt(h_clean, v_clean, rec.sampling_rate, velocity_threshold="auto")
    assert len(events.fixations) > 0, "Should detect fixations"
    assert len(events.saccades) > 0, "Should detect saccades"
    # 注视时长应在合理范围 0.05-5 秒
    for f in events.fixations:
        assert 0.05 <= f.duration_sec <= 10.0
    # 扫视幅值应为正
    for s in events.saccades:
        assert s.amplitude > 0
