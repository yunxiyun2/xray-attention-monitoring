"""DROZY 数据集读取器：EDF / KSS / PVT。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib

# 已知缺失或无效的 (subject, test) 组合
_EXCLUDED_TESTS: set[tuple[str, str]] = {
    ("7", "1"),    # KSS=0
    ("9", "1"),    # EDF 缺失
    ("10", "2"),   # EDF 缺失
    ("12", "2"),   # EDF 缺失
    ("12", "3"),   # EDF 缺失
    ("13", "3"),   # EDF 缺失
}

_TS_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"


@dataclass
class DrozyRecord:
    """单个 DROZY 测试记录。"""

    subject: str
    test: str
    eog_h: np.ndarray
    eog_v: np.ndarray
    cam_sync: np.ndarray
    timestamps: np.ndarray
    kss: int
    pvt_rt: np.ndarray
    sampling_rate: float


def list_available_tests(drozy_root: str | Path) -> list[tuple[str, str]]:
    """枚举 DROZY 中存在的 (subject, test) 对，排除已知缺失测试。"""
    drozy_root = Path(drozy_root)
    tests: list[tuple[str, str]] = []
    for subj in range(1, 15):
        for t in (1, 2, 3):
            if (str(subj), str(t)) in _EXCLUDED_TESTS:
                continue
            edf = drozy_root / "psg" / f"{subj}-{t}.edf"
            if edf.exists():
                tests.append((str(subj), str(t)))
    return tests


def _match_channel(labels: list[str], name: str) -> int | None:
    """在 EDF 通道标签中模糊匹配指定通道名。"""
    name_norm = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    # 精确匹配
    for i, lab in enumerate(labels):
        lab_norm = lab.lower().replace(" ", "").replace("-", "").replace("_", "")
        if lab_norm == name_norm:
            return i
    # 模糊匹配
    for i, lab in enumerate(labels):
        lab_norm = lab.lower().replace(" ", "").replace("-", "").replace("_", "")
        if name_norm in lab_norm:
            return i
    return None


def load_edf_channels(
    edf_path: str | Path, wanted: list[str]
) -> tuple[dict[str, np.ndarray], float, list[str]]:
    """读取 EDF 文件中指定通道。返回 (signals_dict, sampling_freq, channel_labels)。"""
    edf_path = str(edf_path)
    with pyedflib.EdfReader(edf_path) as reader:
        n = reader.signals_in_file
        labels = [reader.getLabel(i) for i in range(n)]
        sfs = reader.getSampleFrequencies()
        fs = float(sfs[0])
        signals: dict[str, np.ndarray] = {}
        for ch_name in wanted:
            idx = _match_channel(labels, ch_name)
            if idx is None:
                raise KeyError(f"Channel {ch_name} not found in {edf_path}: {labels}")
            signals[ch_name] = reader.readSignal(idx).astype(np.float64)
    return signals, fs, labels


def load_kss_matrix(kss_path: str | Path) -> np.ndarray:
    """读取 KSS.txt，返回 14×3 矩阵。"""
    kss_path = Path(kss_path)
    rows: list[list[int]] = []
    with open(kss_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                rows.append([int(float(x)) for x in parts])
    return np.array(rows, dtype=int)


def _parse_ts(ts_str: str) -> datetime:
    """解析 DROZY 时间戳字符串。"""
    return datetime.strptime(ts_str, _TS_FORMAT)


def load_pvt_rt(pvt_path: str | Path) -> np.ndarray:
    """读取 PVT 反应时 CSV，返回反应时数组（毫秒）。

    CSV 格式：
        第1行: 创建时间戳（跳过）
        第2+行: stimulus_ts;response_ts
    反应时 = response_ts - stimulus_ts（毫秒）
    """
    pvt_path = Path(pvt_path)
    rts: list[float] = []
    with open(pvt_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue  # 跳过表头行
            try:
                stim = _parse_ts(parts[0])
                resp = _parse_ts(parts[1])
                rt_ms = (resp - stim).total_seconds() * 1000.0
                if rt_ms >= 0:
                    rts.append(rt_ms)
            except (ValueError, IndexError):
                continue
    return np.array(rts, dtype=float)


def load_all_records(
    drozy_root: str | Path,
    channels: list[str] | None = None,
) -> list[DrozyRecord]:
    """加载全部有效 DROZY 记录。"""
    drozy_root = Path(drozy_root)
    if channels is None:
        channels = ["EOG-H", "EOG-V", "Cam-Sync"]
    kss_matrix = load_kss_matrix(drozy_root / "KSS.txt")
    tests = list_available_tests(drozy_root)
    records: list[DrozyRecord] = []
    for subj, test in tests:
        subj_idx = int(subj) - 1
        test_idx = int(test) - 1
        kss_val = int(kss_matrix[subj_idx, test_idx])
        if kss_val == 0:
            continue
        edf_path = drozy_root / "psg" / f"{subj}-{test}.edf"
        if not edf_path.exists():
            continue
        signals, fs, _ = load_edf_channels(edf_path, channels)
        eog_h = signals.get("EOG-H", np.array([]))
        eog_v = signals.get("EOG-V", np.array([]))
        cam_sync = signals.get("Cam-Sync", np.array([]))
        if len(eog_h) == 0 or len(eog_v) == 0:
            continue
        ts = np.arange(len(eog_h)) / fs
        pvt_path = drozy_root / "pvt-rt" / f"{subj}-{test}.csv"
        pvt_rt = load_pvt_rt(pvt_path) if pvt_path.exists() else np.array([])
        records.append(
            DrozyRecord(
                subject=subj,
                test=test,
                eog_h=eog_h,
                eog_v=eog_v,
                cam_sync=cam_sync,
                timestamps=ts,
                kss=kss_val,
                pvt_rt=pvt_rt,
                sampling_rate=fs,
            )
        )
    return records
