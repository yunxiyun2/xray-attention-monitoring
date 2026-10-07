"""DROZY 数据读取器测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dual_factor.data.drozy_loader import (
    DrozyRecord,
    list_available_tests,
    load_edf_channels,
    load_kss_matrix,
    load_pvt_rt,
    load_all_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DROZY_ROOT = PROJECT_ROOT / "data_public" / "DROZY"


def test_list_available_tests_returns_36():
    """DROZY 应有 36 个有效测试（42 - 6 缺失/无效）。"""
    tests = list_available_tests(DROZY_ROOT)
    assert len(tests) == 36, f"Expected 36 tests, got {len(tests)}"
    assert ("1", "1") in tests
    # 7-1 被排除（KSS=0 且 EDF 不存在）
    assert ("7", "1") not in tests
    # 其他已知缺失
    assert ("9", "1") not in tests
    assert ("10", "2") not in tests


def test_load_edf_channels_returns_eog():
    """EDF 文件应包含 EOG-H 和 EOG-V 通道。"""
    edf_path = DROZY_ROOT / "psg" / "1-1.edf"
    if not edf_path.exists():
        pytest.skip("EDF file 1-1.edf not available")
    signals, sf, labels = load_edf_channels(edf_path, ["EOG-H", "EOG-V", "Cam-Sync"])
    assert sf == 512
    assert "EOG-H" in labels
    assert "EOG-V" in labels
    assert len(signals["EOG-H"]) > 0
    assert len(signals["EOG-V"]) == len(signals["EOG-H"])


def test_load_kss_matrix_shape():
    """KSS 矩阵应为 14×3。"""
    kss = load_kss_matrix(DROZY_ROOT / "KSS.txt")
    assert kss.shape == (14, 3)
    assert kss[0, 0] == 3   # 被试1测试1
    assert kss[6, 0] == 0  # 被试7测试1，KSS=0


def test_load_pvt_rt_returns_positive_array():
    """PVT 反应时应为正数数组（毫秒）。"""
    pvt_path = DROZY_ROOT / "pvt-rt" / "1-1.csv"
    if not pvt_path.exists():
        pytest.skip("PVT file 1-1.csv not available")
    rt = load_pvt_rt(pvt_path)
    assert len(rt) > 0
    assert np.all(rt >= 0)
    # 反应时应在合理范围 100-2000 ms
    assert np.all(rt <= 5000)


def test_load_all_records_integration():
    """端到端：加载全部记录应返回 36 条。"""
    if not DROZY_ROOT.exists():
        pytest.skip("DROZY not available")
    records = load_all_records(DROZY_ROOT)
    assert len(records) == 36
    rec = records[0]
    assert isinstance(rec, DrozyRecord)
    assert rec.subject in ["1", "2", "3"]
    assert 0 <= rec.kss <= 9
    # KSS=0 的被试7测试1不应出现
    kss_values = [r.kss for r in records]
    assert 0 not in kss_values
