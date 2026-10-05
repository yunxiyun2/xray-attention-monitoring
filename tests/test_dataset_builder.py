from __future__ import annotations

from pathlib import Path
import pytest
from xray_attention.data.dataset_builder import DatasetBuilder, TaskRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_builder_resolves_project_root():
    b = DatasetBuilder()
    assert b.data_root.exists()
    assert b.dist_error_root.exists()


def test_discover_records_count():
    """应发现 80 条 record = 20 受试者 × 2 难度 × 2 状态。"""
    b = DatasetBuilder()
    records = b.discover_records()
    assert len(records) == 80, f"expected 80, got {len(records)}"


def test_discover_records_fields():
    b = DatasetBuilder()
    records = b.discover_records()
    r = records[0]
    assert isinstance(r, TaskRecord)
    assert r.subject.isdigit()
    assert r.state in ("alert", "sleepy")
    assert r.difficulty in ("easy", "hard")
    assert r.errors_path.exists()


def test_discover_records_covers_all_subjects():
    b = DatasetBuilder()
    records = b.discover_records()
    subjects = {r.subject for r in records}
    # 20 个受试者
    assert len(subjects) == 20


def test_discover_records_state_balance():
    """alert 和 sleepy 各 40 条。"""
    b = DatasetBuilder()
    records = b.discover_records()
    n_alert = sum(1 for r in records if r.state == "alert")
    n_sleepy = sum(1 for r in records if r.state == "sleepy")
    assert n_alert == 40
    assert n_sleepy == 40


def test_build_windows_for_record_real():
    """真实 record 切窗：60s 窗口 60s 步长 → 至少 1 个窗口且不重叠。"""
    b = DatasetBuilder()
    windows = b.build_windows_for_record(
        subject="01", state="alert", difficulty="easy",
        window_sec=60, stride_sec=60, fps=30,
    )
    assert len(windows) >= 1
    # 验证不重叠
    for i in range(1, len(windows)):
        assert windows[i]["window_idx"] >= windows[i - 1]["window_idx"] + 60 * 30


def test_build_windows_respects_fps():
    b = DatasetBuilder()
    # 用 fps=10 测试
    windows = b.build_windows_for_record(
        subject="01", state="alert", difficulty="easy",
        window_sec=60, stride_sec=60, fps=10,
    )
    if windows:
        assert len(windows[0]["data"]) == 60 * 10


def test_write_loso_folds_creates_csv(tmp_path):
    b = DatasetBuilder(output_root=tmp_path)
    b.write_loso_folds(n_subjects=20)
    folds_csv = tmp_path / "splits" / "loso_folds.csv"
    assert folds_csv.exists()
    import pandas as pd
    df = pd.read_csv(folds_csv)
    assert len(df) == 20
    assert set(df.columns) >= {"fold", "test_subject"}


def test_custom_paths_respected(tmp_path):
    b = DatasetBuilder(
        data_root=tmp_path / "data",
        dist_error_root=tmp_path / "errors",
        output_root=tmp_path / "out",
    )
    assert b.data_root == tmp_path / "data"
    assert b.output_root == tmp_path / "out"


def test_build_gaze_feature_matrix_no_nan_threshold_columns():
    """三个固定阈值的 inside_ratio/longest_run 列应在所有行同时存在且无 NaN。

    按 spec §4.1/§5.3：overall/easy/hard 三阈值全部复用，不按难度选单一阈值，
    保证特征空间一致。
    """
    import pandas as pd

    b = DatasetBuilder()
    df = b.build_gaze_feature_matrix(
        window_sec=60, stride_sec=60, fps=30.0,
        thresholds={"overall": 900.0, "easy": 975.0, "hard": 625.0},
    )
    expected_threshold_cols = {
        "inside_ratio_900", "longest_run_900",
        "inside_ratio_975", "longest_run_975",
        "inside_ratio_625", "longest_run_625",
    }
    assert expected_threshold_cols <= set(df.columns), (
        f"missing cols: {expected_threshold_cols - set(df.columns)}"
    )
    # 三阈值列在任何行都不应有 NaN
    assert not df[list(expected_threshold_cols)].isna().any().any(), (
        "threshold columns contain NaN"
    )
    # 基础统计量也应无 NaN
    for base in ("mean_error", "median_error", "std_error", "p95_error"):
        assert not df[base].isna().any(), f"{base} contains NaN"


# ---------- 多模态对齐 ----------

def test_build_multimodal_feature_matrix_aligns_by_window_order(tmp_path):
    """gaze/face 按 (subject,state,difficulty) 分组后按窗口序号 1-1 对齐。"""
    import pandas as pd

    meta = ["subject", "state", "difficulty", "label", "window_idx"]
    # gaze: 2 个 record，每 record 3 个窗口
    gaze_df = pd.DataFrame([
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 0, "gaze_feat_a": 1.0},
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 30, "gaze_feat_a": 2.0},
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 60, "gaze_feat_a": 3.0},
        {"subject": "02", "state": "sleepy", "difficulty": "hard",
         "label": 0, "window_idx": 0, "gaze_feat_a": 4.0},
        {"subject": "02", "state": "sleepy", "difficulty": "hard",
         "label": 0, "window_idx": 30, "gaze_feat_a": 5.0},
        {"subject": "02", "state": "sleepy", "difficulty": "hard",
         "label": 0, "window_idx": 60, "gaze_feat_a": 6.0},
    ])
    # face: 同样 2 个 record 3 窗口，但 window_idx 用 face 采样率的帧索引（不同）
    face_df = pd.DataFrame([
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 0, "feat_b": 10.0},
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 300, "feat_b": 20.0},
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 600, "feat_b": 30.0},
        {"subject": "02", "state": "sleepy", "difficulty": "hard",
         "label": 0, "window_idx": 0, "feat_b": 40.0},
        {"subject": "02", "state": "sleepy", "difficulty": "hard",
         "label": 0, "window_idx": 300, "feat_b": 50.0},
        {"subject": "02", "state": "sleepy", "difficulty": "hard",
         "label": 0, "window_idx": 600, "feat_b": 60.0},
    ])
    g_csv = tmp_path / "gaze.csv"
    f_csv = tmp_path / "face.csv"
    gaze_df.to_csv(g_csv, index=False)
    face_df.to_csv(f_csv, index=False)

    merged = DatasetBuilder.build_multimodal_feature_matrix(g_csv, f_csv)
    # 6 行（inner join，双方都有 6 行）
    assert len(merged) == 6
    # gaze 列在前，face 列在后（face 加 face_ 前缀避免冲突）
    assert "gaze_feat_a" in merged.columns
    assert "face_feat_b" in merged.columns
    # 第 0 行（subject=01, window_order=0）应对齐 gaze_feat_a=1.0, face_feat_b=10.0
    row0 = merged[(merged["subject"] == "01") & (merged["window_order"] == 0)].iloc[0]
    assert row0["gaze_feat_a"] == 1.0
    assert row0["face_feat_b"] == 10.0
    # 第 2 行应对齐 gaze=3.0, face=30.0
    row2 = merged[(merged["subject"] == "01") & (merged["window_order"] == 2)].iloc[0]
    assert row2["gaze_feat_a"] == 3.0
    assert row2["face_feat_b"] == 30.0
    # 无 NaN
    assert not merged[["gaze_feat_a", "face_feat_b"]].isna().any().any()


def test_build_multimodal_feature_matrix_inner_join_drops_missing(tmp_path):
    """face 比 gaze 少一个窗口时，inner join 应丢弃多余 gaze 行。"""
    import pandas as pd

    gaze_df = pd.DataFrame([
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 0, "ga": 1.0},
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 30, "ga": 2.0},
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 60, "ga": 3.0},
    ])
    face_df = pd.DataFrame([
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 0, "fb": 10.0},
        {"subject": "01", "state": "alert", "difficulty": "easy",
         "label": 1, "window_idx": 300, "fb": 20.0},
    ])
    g_csv = tmp_path / "gaze.csv"
    f_csv = tmp_path / "face.csv"
    gaze_df.to_csv(g_csv, index=False)
    face_df.to_csv(f_csv, index=False)

    merged = DatasetBuilder.build_multimodal_feature_matrix(g_csv, f_csv)
    assert len(merged) == 2  # face 只有 2 个窗口
