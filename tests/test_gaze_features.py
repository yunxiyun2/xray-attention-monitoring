from __future__ import annotations

import numpy as np
from pathlib import Path
from xray_attention.data.gaze_feature_extractor import (
    GazeFeatureExtractor, Window, _longest_true_run
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_all_errors_real_data():
    """真实数据：easy/01/alert/all_errors.txt 可加载为非负数组。"""
    test_file = (
        PROJECT_ROOT / "Distan_error（原20）" / "easy" / "01" / "alert" / "all_errors.txt"
    )
    extractor = GazeFeatureExtractor()
    errors = extractor.load_all_errors(test_file)
    assert isinstance(errors, np.ndarray)
    assert len(errors) > 0
    assert np.all(errors >= 0)


def test_longest_true_run_basic():
    assert _longest_true_run(np.array([True, True, False, True, True, True])) == 3
    assert _longest_true_run(np.array([])) == 0
    assert _longest_true_run(np.array([False, False])) == 0


def test_extract_window_features_basic():
    errors = np.array([100, 200, 150, 80, 90, 1200, 1100, 950], dtype=float)
    extractor = GazeFeatureExtractor()
    feat = extractor.extract_window_features(errors, threshold=900)
    assert feat["mean_error"] > 0
    assert 0.0 <= feat["inside_ratio_900"] <= 1.0
    # <= 900 的有 5 个（100,200,150,80,90）→ 5/8 = 0.625
    assert abs(feat["inside_ratio_900"] - 0.625) < 1e-6
    assert feat["longest_run_900"] == 5
    assert feat["p95_error"] >= feat["mean_error"]
    assert feat["median_error"] > 0
    assert feat["std_error"] >= 0


def test_extract_window_features_all_below_threshold():
    errors = np.array([50, 60, 70], dtype=float)
    extractor = GazeFeatureExtractor()
    feat = extractor.extract_window_features(errors, threshold=900)
    assert abs(feat["inside_ratio_900"] - 1.0) < 1e-6
    assert feat["longest_run_900"] == 3


def test_windowing_non_overlap_test():
    """60s 窗口 + 60s 步长 → 测试窗口不重叠。"""
    errors = np.arange(30 * 200)  # 200s @ 30Hz
    extractor = GazeFeatureExtractor()
    windows = extractor.windowing(errors, window_sec=60, stride_sec=60, fps=30)
    assert len(windows) >= 3
    # 相邻窗口起点差应 = window_sec * fps = 1800
    assert (windows[1].start - windows[0].start) == 60 * 30
    # 不重叠：前一窗口 stop <= 后一窗口 start
    assert windows[0].stop <= windows[1].start


def test_windowing_overlap_train():
    """60s 窗口 + 30s 步长 → 训练窗口 50% 重叠。"""
    errors = np.arange(30 * 200)
    extractor = GazeFeatureExtractor()
    windows = extractor.windowing(errors, window_sec=60, stride_sec=30, fps=30)
    assert len(windows) > 3
    assert (windows[1].start - windows[0].start) == 30 * 30


def test_window_data_slice_correct():
    errors = np.arange(1000, dtype=float)
    extractor = GazeFeatureExtractor()
    windows = extractor.windowing(errors, window_sec=10, stride_sec=10, fps=10)
    assert len(windows) >= 1
    np.testing.assert_array_equal(windows[0].data, errors[0:100])
