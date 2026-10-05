"""注视特征提取器。

从逐采样注视-目标距离误差序列提取窗口级特征。
对应 spec v2.0 §4.1 注视特征。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Window:
    """一个时间窗口的切片。"""
    start: int          # 起始采样索引
    stop: int           # 结束采样索引（不含）
    data: np.ndarray    # 窗口内的误差序列


def _longest_true_run(mask: np.ndarray) -> int:
    """返回布尔数组中最长连续 True 的长度。"""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


class GazeFeatureExtractor:
    """从逐采样注视-目标距离误差序列提取窗口级特征。

    阈值 T* 来自阈值阶段（overall=900 / easy=975 / hard=625），
    本提取器不重新选阈值，仅复用。
    """

    def load_all_errors(self, filepath: str | Path) -> np.ndarray:
        """加载 all_errors.txt 为一维 float 数组。"""
        return np.loadtxt(Path(filepath), dtype=float).ravel()

    def extract_window_features(
        self, errors: np.ndarray, threshold: float
    ) -> dict[str, float | int]:
        """对一段误差序列提取窗口级特征。

        Parameters
        ----------
        errors : 一段采样误差（通常为一个 Window 的 data）
        threshold : 有效关注判定阈值 T*（px）

        Returns
        -------
        dict 含 mean/median/std/p95 误差、inside_ratio_<T>、longest_run_<T>
        """
        errors = np.asarray(errors, dtype=float)
        inside = errors <= threshold
        t_int = int(threshold)
        return {
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "std_error": float(np.std(errors)),
            "p95_error": float(np.percentile(errors, 95)),
            f"inside_ratio_{t_int}": float(np.mean(inside)),
            f"longest_run_{t_int}": int(_longest_true_run(inside)),
        }

    def windowing(
        self,
        errors: np.ndarray,
        window_sec: int,
        stride_sec: int,
        fps: float,
    ) -> list[Window]:
        """将误差序列按时间窗口切片。

        Parameters
        ----------
        window_sec : 窗口长度（秒）
        stride_sec : 步长（秒）；测试集应 = window_sec 以避免重叠泄漏
        fps : 采样率（Hz）

        Returns
        -------
        list[Window] 按时间顺序
        """
        errors = np.asarray(errors, dtype=float)
        win = int(window_sec * fps)
        stride = int(stride_sec * fps)
        if win <= 0 or stride <= 0:
            raise ValueError("window_sec 与 stride_sec 必须为正")
        if len(errors) < win:
            return []
        windows: list[Window] = []
        for s in range(0, len(errors) - win + 1, stride):
            windows.append(Window(s, s + win, errors[s : s + win]))
        return windows
