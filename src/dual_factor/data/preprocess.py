"""EOG 信号预处理：带通滤波 + Hampel 去伪迹。"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth 带通滤波（零相位 filtfilt）。

    Parameters
    ----------
    signal : 1D 信号数组
    fs : 采样率 (Hz)
    lowcut : 高通截止频率 (Hz)
    highcut : 低通截止频率 (Hz)
    order : 滤波器阶数
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def hampel_filter(
    signal: np.ndarray,
    window: int = 51,
    threshold: float = 3.0,
) -> np.ndarray:
    """Hampel 滤波器：基于滑动中值与 MAD 检测并替换离群点。

    对每个样本，计算其邻域（window 大小）的中值和 MAD，
    若样本偏离中值超过 threshold × MAD，则用中值替换。

    Parameters
    ----------
    signal : 1D 信号数组
    window : 滑动窗口大小（奇数）
    threshold : MAD 倍数阈值
    """
    n = len(signal)
    if window < 1 or window % 2 != 1 or threshold <= 0 or not np.isfinite(signal).all():
        raise ValueError("Hampel requires finite samples, a positive odd window and positive threshold")
    if n < window:
        return signal.copy()
    # Exact local MAD: all deviations in a window use that SAME window's median.
    # Chunked views avoid allocating a full EDF-length N x window temporary.
    neighborhoods = np.lib.stride_tricks.sliding_window_view(
        np.pad(np.asarray(signal, dtype=float), window//2, mode="edge"), window)
    med, mad = np.empty(n), np.empty(n)
    for start in range(0, n, 4096):
        block = neighborhoods[start:start+4096]
        center = np.median(block, axis=1)
        med[start:start+len(block)] = center
        mad[start:start+len(block)] = 1.4826*np.median(np.abs(block-center[:, None]), axis=1)
    out = signal.copy()
    # Zero-MAD plateaus with an isolated deviation are outliers too.
    mask = np.abs(signal-med) > threshold * mad
    out[mask] = med[mask]
    return out


def preprocess_eog(
    eog_h: np.ndarray,
    eog_v: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 30.0,
    filter_order: int = 4,
    hampel_window: int = 51,
    hampel_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """EOG 预处理管道：带通滤波 → Hampel 去伪迹。

    Parameters
    ----------
    eog_h : 水平 EOG 信号
    eog_v : 垂直 EOG 信号
    fs : 采样率 (Hz)
    lowcut : 带通高通截止 (Hz)
    highcut : 带通低通截止 (Hz)
    filter_order : Butterworth 阶数
    hampel_window : Hampel 窗口大小
    hampel_threshold : Hampel MAD 倍数

    Returns
    -------
    (h_clean, v_clean) : 滤波+去伪迹后的 EOG-H, EOG-V
    """
    h_bp = bandpass_filter(eog_h, fs, lowcut, highcut, filter_order)
    v_bp = bandpass_filter(eog_v, fs, lowcut, highcut, filter_order)
    h_clean = hampel_filter(h_bp, hampel_window, hampel_threshold)
    v_clean = hampel_filter(v_bp, hampel_window, hampel_threshold)
    return h_clean, v_clean
