"""I-VT 速度阈值法：注视/扫视事件检测。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class Fixation:
    """注视事件。"""

    start_idx: int
    end_idx: int
    mean_h: float
    mean_v: float
    duration_sec: float


@dataclass
class Saccade:
    """扫视事件。"""

    start_idx: int
    end_idx: int
    amplitude: float
    peak_velocity: float
    direction_deg: float
    duration_sec: float


@dataclass
class Events:
    """一次检测的全部事件。"""

    fixations: list[Fixation]
    saccades: list[Saccade]


def compute_velocity(eog_h: np.ndarray, eog_v: np.ndarray, fs: float) -> np.ndarray:
    """一阶差分得到二维 EOG 速度信号。"""
    dh = np.diff(eog_h, prepend=eog_h[0])
    dv = np.diff(eog_v, prepend=eog_v[0])
    return np.sqrt(dh ** 2 + dv ** 2) * fs


def _auto_threshold(velocity: np.ndarray, percentile: float = 70.0) -> float:
    """数据驱动速度阈值：基于速度分布百分位。"""
    return float(np.percentile(velocity, percentile))


def _find_runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    """找连续 True/False 片段的 (start, end) 列表，end 为 exclusive。"""
    runs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i] == value:
            j = i
            while j < n and mask[j] == value:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def detect_events_ivt(
    eog_h: np.ndarray,
    eog_v: np.ndarray,
    fs: float,
    velocity_threshold: Union[float, str] = "auto",
    velocity_percentile: float = 70.0,
    min_fixation_ms: float = 100.0,
    min_saccade_ms: float = 20.0,
) -> Events:
    """I-VT 速度阈值法：注视/扫视事件提取。

    Parameters
    ----------
    eog_h, eog_v : 滤波后 EOG 信号
    fs : 采样率 (Hz)
    velocity_threshold : 速度阈值，或 "auto" 自动计算
    velocity_percentile : auto 模式下的百分位
    min_fixation_ms : 最短注视时长 (ms)
    min_saccade_ms : 最短扫视时长 (ms)
    """
    velocity = compute_velocity(eog_h, eog_v, fs)
    if isinstance(velocity_threshold, str) and velocity_threshold == "auto":
        v_th = _auto_threshold(velocity, velocity_percentile)
    else:
        v_th = float(velocity_threshold)

    is_fixation = velocity < v_th
    fixations_raw = _find_runs(is_fixation, True)
    saccades_raw = _find_runs(is_fixation, False)

    min_fix_samples = math.ceil(min_fixation_ms / 1000 * fs)
    min_sac_samples = math.ceil(min_saccade_ms / 1000 * fs)

    fixations: list[Fixation] = []
    for s, e in fixations_raw:
        if e - s < min_fix_samples:
            continue
        fixations.append(Fixation(
            start_idx=s, end_idx=e,
            mean_h=float(np.mean(eog_h[s:e])),
            mean_v=float(np.mean(eog_v[s:e])),
            duration_sec=(e - s) / fs,
        ))

    saccades: list[Saccade] = []
    for s, e in saccades_raw:
        if e - s < min_sac_samples:
            continue
        amp = float(np.sqrt(
            (eog_h[e - 1] - eog_h[s]) ** 2 + (eog_v[e - 1] - eog_v[s]) ** 2
        ))
        peak_v = float(np.max(velocity[s:e]))
        dh = float(eog_h[e - 1] - eog_h[s])
        dv = float(eog_v[e - 1] - eog_v[s])
        direction = float(np.degrees(np.arctan2(dv, dh)))
        saccades.append(Saccade(
            start_idx=s, end_idx=e, amplitude=amp,
            peak_velocity=peak_v, direction_deg=direction,
            duration_sec=(e - s) / fs,
        ))
    return Events(fixations=fixations, saccades=saccades)
