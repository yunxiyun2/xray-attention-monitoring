"""8 个核心眼动指标：呆滞因子 + 发散因子。"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import entropy as scipy_entropy

from dual_factor.features.event_detection import Events, Fixation, Saccade


# ============ 呆滞因子（Hypoactive） ============

def mean_fixation_duration(events: Events) -> float:
    """FD: 平均注视时长（秒）。"""
    if not events.fixations:
        return float("nan")
    return float(np.mean([f.duration_sec for f in events.fixations]))


def peak_saccade_velocity(events: Events) -> float:
    """PSV: 扫视峰值速度均值。"""
    if not events.saccades:
        return float("nan")
    return float(np.mean([s.peak_velocity for s in events.saccades]))


def saccade_amplitude(events: Events) -> float:
    """SA: 扫视幅度均值。"""
    if not events.saccades:
        return float("nan")
    return float(np.mean([s.amplitude for s in events.saccades]))


def fixation_rate(events: Events, window_sec: float) -> float:
    """FN: 单位时间内注视事件次数。"""
    return float(len(events.fixations) / window_sec) if window_sec > 0 else float("nan")


def compute_hypoactive_indicators(events: Events, window_sec: float) -> dict[str, float]:
    """呆滞因子 4 指标：FD, PSV, SA, FN。"""
    return {
        "FD": mean_fixation_duration(events),
        "PSV": peak_saccade_velocity(events),
        "SA": saccade_amplitude(events),
        "FN": fixation_rate(events, window_sec),
    }


# ============ 发散因子（Diffuse） ============

def _fixation_positions(fixations: list[Fixation]) -> np.ndarray:
    """提取注视点 (h, v) 位置数组。"""
    if not fixations:
        return np.empty((0, 2))
    return np.array([[f.mean_h, f.mean_v] for f in fixations])


def _to_grid_indices(positions: np.ndarray, bins: int) -> np.ndarray:
    """将连续位置映射到 bins×bins 网格索引。"""
    if len(positions) == 0:
        return np.empty(0, dtype=int)
    h = positions[:, 0]
    v = positions[:, 1]
    h_min, h_max = h.min(), h.max()
    v_min, v_max = v.min(), v.max()
    h_range = h_max - h_min if h_max > h_min else 1e-9
    v_range = v_max - v_min if v_max > v_min else 1e-9
    hi = np.clip(((h - h_min) / h_range * bins).astype(int), 0, bins - 1)
    vi = np.clip(((v - v_min) / v_range * bins).astype(int), 0, bins - 1)
    return hi * bins + vi


def spatial_entropy(fixations: list[Fixation], bins: int = 5) -> float:
    """SE: 注视点在空间网格中分布的香农熵。"""
    positions = _fixation_positions(fixations)
    if len(positions) == 0:
        return float("nan")
    idx = _to_grid_indices(positions, bins)
    counts = np.bincount(idx, minlength=bins * bins)
    p = counts[counts > 0] / counts.sum()
    return float(scipy_entropy(p, base=2))


def gaze_transition_entropy(fixations: list[Fixation], bins: int = 5) -> float:
    """GTE: 注视点在网格区域间转移模式的信息熵。"""
    positions = _fixation_positions(fixations)
    if len(positions) < 2:
        return float("nan")
    idx = _to_grid_indices(positions, bins)
    transitions = list(zip(idx[:-1], idx[1:]))
    if not transitions:
        return float("nan")
    counter = Counter(transitions)
    total = sum(counter.values())
    p = np.array([c / total for c in counter.values()])
    return float(scipy_entropy(p, base=2))


def spatial_variance(fixations: list[Fixation]) -> float:
    """SV: 注视点坐标的 trace 协方差（简化为马氏距离方差）。"""
    positions = _fixation_positions(fixations)
    if len(positions) < 2:
        return float("nan")
    cov = np.cov(positions.T)
    return float(np.trace(cov))


def saccade_direction_randomness(saccades: list[Saccade]) -> float:
    """SDR: 扫视角度分布与均匀分布的 KL 散度（越小越随机）。"""
    if not saccades:
        return float("nan")
    angles = np.array([s.direction_deg % 360 for s in saccades])
    n_bins = 8
    hist, _ = np.histogram(angles, bins=n_bins, range=(0, 360))
    p = hist / hist.sum() if hist.sum() > 0 else np.ones(n_bins) / n_bins
    q = np.ones(n_bins) / n_bins  # 均匀分布
    return float(np.sum(np.where(p > 0, p * np.log(p / q + 1e-12), 0)))


def compute_diffuse_indicators(events: Events, bins: int = 5) -> dict[str, float]:
    """发散因子 4 指标：SE, GTE, SV, SDR。"""
    return {
        "SE": spatial_entropy(events.fixations, bins),
        "GTE": gaze_transition_entropy(events.fixations, bins),
        "SV": spatial_variance(events.fixations),
        "SDR": saccade_direction_randomness(events.saccades),
    }


def compute_all_indicators(events: Events, window_sec: float, bins: int = 5) -> dict[str, float]:
    """8 个核心指标一次性计算。"""
    out = compute_hypoactive_indicators(events, window_sec)
    out.update(compute_diffuse_indicators(events, bins))
    return out
