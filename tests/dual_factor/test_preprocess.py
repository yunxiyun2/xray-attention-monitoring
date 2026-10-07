"""EOG 信号预处理测试：带通滤波 + Hampel 去伪迹。"""
from __future__ import annotations

import numpy as np
import pytest

from dual_factor.data.preprocess import (
    bandpass_filter,
    hampel_filter,
    preprocess_eog,
)


def test_bandpass_filter_removes_dc():
    """带通滤波应去除直流分量和高频噪声。"""
    fs = 512
    t = np.arange(512 * 10) / fs
    # 直流(100) + 10Hz 正弦 + 100Hz 噪声
    sig = 100 + np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
    out = bandpass_filter(sig, fs, lowcut=0.5, highcut=30.0, order=4)
    assert np.abs(np.mean(out)) < 1.0, "DC should be removed"
    assert np.std(out) > 0.1, "Signal should be preserved"
    # 100Hz 高频应被压制
    fft_in = np.abs(np.fft.rfft(sig))
    fft_out = np.abs(np.fft.rfft(out))
    idx_100hz = int(100 * len(t) // fs)
    assert fft_out[idx_100hz] < fft_in[idx_100hz] * 0.1, "High freq should be attenuated"


def test_bandpass_filter_preserves_passband():
    """通带内信号（10Hz）应被保留。"""
    fs = 512
    t = np.arange(512 * 10) / fs
    sig = np.sin(2 * np.pi * 10 * t)
    out = bandpass_filter(sig, fs, lowcut=0.5, highcut=30.0, order=4)
    # 通带内幅值应接近原始
    assert np.std(out) > 0.5 * np.std(sig)


def test_hampel_filter_removes_spike():
    """Hampel 应检测并替换大幅离群点。"""
    sig = np.sin(2 * np.pi * 5 * np.arange(1000) / 512)
    sig[100] += 50  # 注入伪迹
    out = hampel_filter(sig, window=51, threshold=3.0)
    assert np.abs(out[100]) < np.abs(sig[100]) * 0.5, "Spike should be reduced"


def test_hampel_filter_preserves_clean_signal():
    """无伪迹的干净信号应基本不变。"""
    sig = np.sin(2 * np.pi * 5 * np.arange(1000) / 512)
    out = hampel_filter(sig, window=51, threshold=3.0)
    assert np.allclose(out, sig, atol=1e-10), "Clean signal should be unchanged"


def test_preprocess_eog_pipeline():
    """完整预处理管道：滤波 + Hampel。"""
    fs = 512
    rng = np.random.default_rng(0)
    t = np.arange(512 * 30) / fs
    eog_h = np.sin(2 * np.pi * 5 * t) + 200 + 0.1 * rng.standard_normal(512 * 30)
    eog_v = np.sin(2 * np.pi * 3 * t) + 100
    eog_h[50] += 100  # 伪迹
    h_clean, v_clean = preprocess_eog(eog_h, eog_v, fs, highcut=30.0, lowcut=0.5)
    assert len(h_clean) == len(eog_h), "Length should be preserved"
    assert np.isfinite(h_clean).all(), "Output should be finite"
    # 直流被去除
    assert np.abs(np.mean(h_clean)) < 5.0, "DC should be removed"
    # 伪迹被压制
    assert np.abs(h_clean[50]) < np.abs(eog_h[50]) * 0.5


def test_preprocess_eog_on_real_data():
    """在真实 DROZY 数据上验证预处理。"""
    from pathlib import Path
    from dual_factor.data.drozy_loader import load_all_records

    drozy_root = Path(__file__).resolve().parents[2] / "data_public" / "DROZY"
    if not drozy_root.exists():
        pytest.skip("DROZY not available")
    records = load_all_records(drozy_root)
    rec = records[0]
    h_clean, v_clean = preprocess_eog(rec.eog_h, rec.eog_v, rec.sampling_rate)
    assert len(h_clean) == len(rec.eog_h)
    assert np.isfinite(h_clean).all()
    # EOG 电压幅值应在合理范围
    assert np.std(h_clean) > 1.0, "Signal should have variance"
    assert np.abs(np.mean(h_clean)) < 50.0, "DC offset should be small"
