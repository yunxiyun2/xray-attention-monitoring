"""窗口化切片与被试内标准化。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dual_factor.features.event_detection import detect_events_ivt
from dual_factor.features.indicators import compute_all_indicators


@dataclass
class Window:
    start_idx: int
    end_idx: int
    eog_h: np.ndarray
    eog_v: np.ndarray


def slice_windows(
    eog_h: np.ndarray,
    eog_v: np.ndarray,
    fs: float,
    window_sec: float,
    stride_sec: float,
) -> list[Window]:
    """按窗口长度与步长切片。"""
    win = int(window_sec * fs)
    stride = int(stride_sec * fs)
    n = len(eog_h)
    windows: list[Window] = []
    for s in range(0, n - win + 1, stride):
        windows.append(Window(s, s + win, eog_h[s:s + win], eog_v[s:s + win]))
    return windows


def build_indicator_matrix(
    subject: str,
    test: str,
    kss: int,
    eog_h: np.ndarray,
    eog_v: np.ndarray,
    fs: float,
    window_sec: float = 30.0,
    stride_sec: float = 10.0,
    bins: int = 5,
    min_fixations: int = 5,
    min_saccades: int = 3,
    velocity_threshold: float | str = "auto",
    velocity_percentile: float = 70.0,
    min_fixation_ms: float = 100.0,
    min_saccade_ms: float = 20.0,
) -> pd.DataFrame:
    """对一个 (subject, test) 切窗、检测事件、计算 8 指标。"""
    windows = slice_windows(eog_h, eog_v, fs, window_sec, stride_sec)
    rows: list[dict] = []
    for i, w in enumerate(windows):
        events = detect_events_ivt(
            w.eog_h, w.eog_v, fs,
            velocity_threshold=velocity_threshold,
            velocity_percentile=velocity_percentile,
            min_fixation_ms=min_fixation_ms,
            min_saccade_ms=min_saccade_ms,
        )
        indicator_cols = ["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"]
        if len(events.fixations) < min_fixations or len(events.saccades) < min_saccades:
            row = {
                "subject": subject, "test": test, "kss": kss,
                "window_idx": i, "valid": False,
                **{c: np.nan for c in indicator_cols},
            }
        else:
            indicators = compute_all_indicators(events, window_sec, bins)
            row = {
                "subject": subject, "test": test, "kss": kss,
                "window_idx": i, "valid": True,
                **indicators,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def within_subject_zscore(
    df: pd.DataFrame, indicator_cols: list[str]
) -> pd.DataFrame:
    """被试内 z-score 标准化：消除个体间 EOG 电压基线差异。"""
    out = df.copy()
    for col in indicator_cols:
        if col not in out.columns:
            continue
        z_col = f"{col}_z"
        out[z_col] = np.nan
        for subj, grp in out.groupby("subject"):
            vals = grp[col].astype(float)
            mu = vals.mean()
            sd = vals.std(ddof=0)
            out.loc[grp.index, z_col] = (vals - mu) / sd if sd > 0 else 0.0
    return out
