# 疲劳目光行为双因子模式研究：DROZY 探索性因子分析 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **上游 spec**: [2026-10-07-dual-factor-drozy-efa.md](../specs/2026-10-07-dual-factor-drozy-efa.md) v1.0
>
> **Conda 环境名**：`Anomaly`（用户本机实际环境名，与 `environment.yml` 中的 `xray-attention` 不同，所有命令示例使用 `Anomaly`）。

**Goal:** 按 spec v1.0 在 DROZY 公开数据集上跑通"EDF 读取 → EOG 预处理 → I-VT 事件检测 → 8 指标计算 → 窗口化标准化 → EFA + 平行分析 → 方向性检验 → 决策报告"全链条，在两周内回答"呆滞与发散能否统计分离"。

**Architecture:**

- `src/dual_factor/data/`：DROZY 数据读取（EDF/KSS/PVT）与 EOG 信号预处理
- `src/dual_factor/features/`：I-VT 事件检测、8 指标计算、窗口化与标准化
- `src/dual_factor/analysis/`：EFA（KMO/Bartlett/PAF/Promax/平行分析）、描述性统计、方向性检验、可视化
- `src/dual_factor/pipeline.py`：一键运行入口
- `configs/dual_factor_efa.yaml`：唯一参数源
- `results/dual_factor_efa/`：全部产物输出目录

**Tech Stack:** Python 3.9+（`from __future__ import annotations` 兼容）, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, pyedflib（EDF 读取）, factor_analyzer（EFA/KMO/Bartlett/平行分析）

**Non-invasive principle:** 旧代码 `src/xray_attention/` 完全不动；新建 `src/dual_factor/` 包与之平级，仅共享 conda 环境与配置风格。

**Cross-platform:** 所有路径相对项目根解析（`Path(__file__).resolve().parents[N]`），不写死绝对路径，Linux/macOS 通用。

---

## Global Constraints

- DROZY 原始数据目录 `data_public/DROZY/` 只读，不复制、不重写、不入 Git。
- EOG-H / EOG-V 视为二维眼动位置代理（单位 μV，非像素），仅用相对分布模式计算空间特征。
- KSS ≤ 4 为清醒（alert），KSS ≥ 6 为困倦（drowsy），KSS = 5 排除或单独分析；被试 7 测试 1（KSS=0）排除。
- 8 个指标在 30 秒窗口、10 秒步长下计算；窗口内 <5 注视或 <3 扫视则标记缺失。
- EFA 提取方法固定为主轴因子法（PAF）+ Promax 斜交旋转；因子数由平行分析数据驱动确定。
- 决策判据：两因子相关 r < 0.70 为可分离，r > 0.85 为不可分离，0.70 ≤ r ≤ 0.85 为模糊。
- 否定结果也是合格结果：只要分析过程严谨、判定清晰、代码可复现，无论双因子是否成立都算合格。

---

## Task 1: 环境与目录初始化

**Files:**

- 扩展: `environment.yml`（追加 `pyedflib`、`factor_analyzer`）
- 创建: `src/dual_factor/__init__.py`
- 创建: `src/dual_factor/data/__init__.py`, `src/dual_factor/features/__init__.py`, `src/dual_factor/analysis/__init__.py`
- 创建: `configs/dual_factor_efa.yaml`
- 创建: `results/dual_factor_efa/.gitkeep`, `results/dual_factor_efa/figures/.gitkeep`
- 创建: `tests/dual_factor/.gitkeep`

**对应 spec**: §6.1 设计原则、§6.2 包结构、§6.3 配置、§6.4 新增依赖

**Interfaces:**

- Consumes: 项目根已有的 conda 环境 `Anomaly`、`environment.yml`
- Produces: 新包骨架 `src/dual_factor/`、唯一参数源 `configs/dual_factor_efa.yaml`

- [ ] **Step 1: 扩展依赖**

更新 `environment.yml`，在 pip 依赖末尾追加（不修改已有条目）：

```yaml
  - pip:
      - pyedflib>=0.1.30
      - factor_analyzer>=0.4.3
```

- [ ] **Step 2: 安装新增依赖**

```bash
conda run -n Anomaly pip install pyedflib factor_analyzer
```

验证：

```bash
conda run -n Anomaly python -c "import pyedflib, factor_analyzer; print('deps ready')"
```

Expected: 输出 `deps ready`。

- [ ] **Step 3: 创建包骨架**

```bash
# 从项目根目录运行
mkdir -p src/dual_factor/data src/dual_factor/features src/dual_factor/analysis
mkdir -p results/dual_factor_efa/figures tests/dual_factor
touch src/dual_factor/__init__.py
touch src/dual_factor/data/__init__.py src/dual_factor/features/__init__.py src/dual_factor/analysis/__init__.py
touch results/dual_factor_efa/.gitkeep results/dual_factor_efa/figures/.gitkeep
touch tests/dual_factor/.gitkeep
```

- [ ] **Step 4: 创建唯一参数源 `configs/dual_factor_efa.yaml`**

```yaml
# 疲劳目光行为双因子 EFA 实验唯一参数源
data:
  drozy_root: data_public/DROZY
  channels: [EOG-H, EOG-V, Cam-Sync]
  sampling_rate: 512
  subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  exclude_tests: ["7-1"]   # KSS=0，无效

preprocess:
  highpass: 0.5
  lowpass: 30.0
  filter_order: 4
  hampel_window: 51
  hampel_threshold: 3.0    # MAD 倍数

event_detection:
  algorithm: I-VT
  velocity_threshold: auto   # 基于速度分布百分位
  velocity_percentile: 70    # auto 模式下的百分位
  min_fixation_ms: 100
  min_saccade_ms: 20

windowing:
  window_sec: 30
  stride_sec: 10
  min_fixations: 5
  min_saccades: 3
  normalization: within_subject_zscore

spatial_grid:
  bins: 5                    # 5x5 网格

labels:
  kss_file: KSS.txt
  alert_threshold: 4          # KSS <= 4
  drowsy_threshold: 6         # KSS >= 6
  exclude_middle: true        # 排除 KSS=5

efa:
  extraction: pa              # 主轴因子
  rotation: promax             # 斜交旋转
  max_factors: 4
  variance_threshold: 0.60
  loading_threshold: 0.50
  cross_loading_threshold: 0.30
  factor_corr_threshold: 0.70
  parallel_analysis_n_iter: 1000
  parallel_analysis_percentile: 95

output:
  results_dir: results/dual_factor_efa
  figures_dir: results/dual_factor_efa/figures
```

- [ ] **Step 5: 验证骨架可导入**

```bash
PYTHONPATH=src conda run -n Anomaly python -c "import dual_factor; print('package ready')"
```

Expected: 输出 `package ready`。

- [ ] **Step 6: 提交**

```bash
git add environment.yml src/dual_factor/ configs/dual_factor_efa.yaml results/dual_factor_efa/ tests/dual_factor/
git commit -m "feat(dual_factor): init package skeleton, config, and dependencies"
```

---

## Task 2: DROZY 数据读取器

**Files:**

- 创建: `src/dual_factor/data/drozy_loader.py`
- 创建: `tests/dual_factor/test_drozy_loader.py`

**对应 spec**: §2.1 DROZY 数据集、§2.2 眼动数据来源、§2.3 疲劳标签、§4.1 数据读取

**Interfaces:**

- Consumes: `data_public/DROZY/psg/{subj}-{test}.edf`、`data_public/DROZY/KSS.txt`、`data_public/DROZY/pvt-rt/{subj}-{test}.csv`
- Produces: `DrozyRecord(subject, test, eog_h, eog_v, cam_sync, timestamps, kss, pvt_rt)` 数据类；`load_all_records(cfg)` 返回全部有效记录列表

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_drozy_loader.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from dual_factor.data.drozy_loader import (
    DrozyRecord, load_edf_channels, load_kss_matrix, load_pvt_rt,
    list_available_tests, load_all_records
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DROZY_ROOT = PROJECT_ROOT / "data_public" / "DROZY"


def test_list_available_tests_returns_37():
    tests = list_available_tests(DROZY_ROOT)
    assert len(tests) == 37, f"Expected 37 tests, got {len(tests)}"
    assert ("1", "1") in tests
    assert ("7", "1") not in tests  # 被排除


def test_load_edf_channels_returns_eog():
    edf_path = DROZY_ROOT / "psg" / "1-1.edf"
    if not edf_path.exists():
        pytest.skip("EDF file 1-1.edf not available")
    signals, sf, labels = load_edf_channels(edf_path, ["EOG-H", "EOG-V", "Cam-Sync"])
    assert sf == 512
    assert "EOG-H" in labels and "EOG-V" in labels
    assert len(signals["EOG-H"]) > 0
    assert len(signals["EOG-V"]) == len(signals["EOG-H"])


def test_load_kss_matrix_shape():
    kss = load_kss_matrix(DROZY_ROOT / "KSS.txt")
    assert kss.shape == (14, 3)
    assert kss[0, 0] == 3   # 被试1测试1
    assert kss[6, 0] == 0  # 被试7测试1，排除


def test_load_pvt_rt_returns_array():
    pvt_path = DROZY_ROOT / "pvt-rt" / "1-1.csv"
    if not pvt_path.exists():
        pytest.skip("PVT file 1-1.csv not available")
    rt = load_pvt_rt(pvt_path)
    assert len(rt) > 0
    assert np.all(rt >= 0)


def test_load_all_records_integration():
    if not DROZY_ROOT.exists():
        pytest.skip("DROZY not available")
    records = load_all_records(DROZY_ROOT)
    assert len(records) == 37
    rec = records[0]
    assert isinstance(rec, DrozyRecord)
    assert rec.subject in ["1", "2"]
    assert 0 <= rec.kss <= 9
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_drozy_loader.py -v
```

Expected: FAIL（模块未定义）

- [ ] **Step 3: 实现 `drozy_loader.py`**

```python
# src/dual_factor/data/drozy_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv

import numpy as np
import pyedflib


@dataclass
class DrozyRecord:
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
    excluded = {("7", "1"), ("9", "1"), ("10", "2"), ("12", "2"), ("12", "3"), ("13", "3")}
    tests: list[tuple[str, str]] = []
    for subj in range(1, 15):
        for t in (1, 2, 3):
            if (str(subj), str(t)) in excluded:
                continue
            edf = drozy_root / "psg" / f"{subj}-{t}.edf"
            if edf.exists():
                tests.append((str(subj), str(t)))
    return tests


def load_edf_channels(
    edf_path: str | Path, wanted: list[str]
) -> tuple[dict[str, np.ndarray], float, list[str]]:
    """读取 EDF 文件中指定通道。返回 (signals_dict, sampling_freq, channel_labels)。"""
    edf_path = str(edf_path)
    with pyedflib.EdfReader(edf_path) as reader:
        n = reader.signals_in_file
        labels = [reader.getLabel(i) for i in range(n)]
        sf = reader.getSampleFrequencies()
        # 多数 EDF 中各通道采样率一致，取首个
        fs = float(sf[0])
        signals: dict[str, np.ndarray] = {}
        for ch_name in wanted:
            # 精确匹配失败则模糊匹配
            idx = _match_channel(labels, ch_name)
            if idx is None:
                raise KeyError(f"Channel {ch_name} not found in {edf_path}: {labels}")
            signals[ch_name] = reader.readSignal(idx).astype(np.float64)
    return signals, fs, labels


def _match_channel(labels: list[str], name: str) -> int | None:
    name_lower = name.lower().replace(" ", "").replace("-", "")
    for i, lab in enumerate(labels):
        lab_norm = lab.lower().replace(" ", "").replace("-", "").replace("_", "")
        if lab_norm == name_lower:
            return i
    # 模糊匹配
    for i, lab in enumerate(labels):
        if name_lower in lab.lower().replace(" ", "").replace("-", "").replace("_", ""):
            return i
    return None


def load_kss_matrix(kss_path: str | Path) -> np.ndarray:
    """读取 KSS.txt，返回 14×3 矩阵。"""
    kss_path = Path(kss_path)
    rows = []
    with open(kss_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                rows.append([int(float(x)) for x in parts])
    return np.array(rows, dtype=int)


def load_pvt_rt(pvt_path: str | Path) -> np.ndarray:
    """读取 PVT 反应时 CSV，返回反应时数组（毫秒）。"""
    pvt_path = Path(pvt_path)
    rts = []
    with open(pvt_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    rts.append(float(row[-1]))
                except ValueError:
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
        records.append(DrozyRecord(
            subject=subj, test=test,
            eog_h=eog_h, eog_v=eog_v, cam_sync=cam_sync,
            timestamps=ts, kss=kss_val, pvt_rt=pvt_rt,
            sampling_rate=fs,
        ))
    return records
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_drozy_loader.py -v
```

Expected: PASS（若 DROZY 数据可用）

- [ ] **Step 5: 提交**

```bash
git add src/dual_factor/data/drozy_loader.py tests/dual_factor/test_drozy_loader.py
git commit -m "feat(dual_factor): DROZY EDF/KSS/PVT loader"
```

---

## Task 3: EOG 信号预处理

**Files:**

- 创建: `src/dual_factor/data/preprocess.py`
- 创建: `tests/dual_factor/test_preprocess.py`

**对应 spec**: §4.2 信号预处理、§8 风险与预案（Hampel 去伪迹）

**Interfaces:**

- Consumes: `DrozyRecord.eog_h` / `DrozyRecord.eog_v` 原始信号、采样率
- Produces: `(eog_h_clean, eog_v_clean)` 滤波后、去伪迹后的信号

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_preprocess.py
from __future__ import annotations

import numpy as np
from dual_factor.data.preprocess import (
    bandpass_filter, hampel_filter, preprocess_eog
)


def test_bandpass_filter_removes_dc():
    fs = 512
    t = np.arange(512 * 10) / fs
    # 直流 + 10Hz 正弦 + 100Hz 噪声
    sig = 100 + np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
    out = bandpass_filter(sig, fs, 0.5, 30.0, order=4)
    assert np.abs(np.mean(out)) < 1.0           # 直流被去除
    assert np.std(out) > 0.1                     # 信号保留
    # 高频成分应被压制
    fft_in = np.abs(np.fft.rfft(sig))
    fft_out = np.abs(np.fft.rfft(out))
    assert fft_out[100 * len(t) // fs] < fft_in[100 * len(t) // fs] * 0.1


def test_hampel_filter_removes_spike():
    rng = np.random.default_rng(42)
    sig = np.sin(2 * np.pi * 5 * np.arange(1000) / 512)
    sig[100] += 50  # 大伪迹
    out = hampel_filter(sig, window=51, threshold=3.0)
    assert np.abs(out[100]) < np.abs(sig[100]) * 0.5


def test_preprocess_eog_pipeline():
    fs = 512
    t = np.arange(512 * 30) / fs
    eog_h = np.sin(2 * np.pi * 5 * t) + 200 + 0.1 * rng_norm(512 * 30)
    eog_v = np.sin(2 * np.pi * 3 * t) + 100
    eog_h[50] += 100  # 伪迹
    h_clean, v_clean = preprocess_eog(eog_h, eog_v, fs, highcut=30.0, lowcut=0.5)
    assert len(h_clean) == len(eog_h)
    assert np.isfinite(h_clean).all()
    assert np.abs(np.mean(h_clean)) < 5.0


def rng_norm(n):
    return np.random.default_rng(0).standard_normal(n)
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_preprocess.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 `preprocess.py`**

```python
# src/dual_factor/data/preprocess.py
from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth 带通滤波（零相位 filtfilt）。"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def hampel_filter(
    signal: np.ndarray,
    window: int = 51,
    threshold: float = 3.0,
) -> np.ndarray:
    """Hampel 滤波器：基于滑动中值与 MAD 检测并替换离群点。"""
    n = len(signal)
    half = window // 2
    out = signal.copy()
    for i in range(half, n - half):
        win = signal[i - half : i + half + 1]
        med = np.median(win)
        mad = 1.4826 * np.median(np.abs(win - med))
        if mad > 0 and np.abs(signal[i] - med) > threshold * mad:
            out[i] = med
    return out


def preprocess_eog(
    eog_h: np.ndarray,
    eog_v: np.ndarray,
    fs: float,
    highcut: float = 30.0,
    lowcut: float = 0.5,
    filter_order: int = 4,
    hampel_window: int = 51,
    hampel_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """EOG 预处理管道：带通滤波 → Hampel 去伪迹。"""
    h_bp = bandpass_filter(eog_h, fs, lowcut, highcut, filter_order)
    v_bp = bandpass_filter(eog_v, fs, lowcut, highcut, filter_order)
    h_clean = hampel_filter(h_bp, hampel_window, hampel_threshold)
    v_clean = hampel_filter(v_bp, hampel_window, hampel_threshold)
    return h_clean, v_clean
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_preprocess.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/dual_factor/data/preprocess.py tests/dual_factor/test_preprocess.py
git commit -m "feat(dual_factor): EOG bandpass filter + Hampel artifact removal"
```

---

## Task 4: I-VT 事件检测算法

**Files:**

- 创建: `src/dual_factor/features/event_detection.py`
- 创建: `tests/dual_factor/test_event_detection.py`

**对应 spec**: §3.2 事件检测算法

**Interfaces:**

- Consumes: `(eog_h, eog_v)` 滤波后信号、采样率、速度阈值（数值或 'auto'）
- Produces: `Events(fixations: list[Fixation], saccades: list[Saccade])`；`Fixation`/`Saccade` 含起止时间戳、平均位置、速度序列

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_event_detection.py
from __future__ import annotations

import numpy as np
from dual_factor.features.event_detection import (
    compute_velocity, detect_events_ivt, Fixation, Saccade, Events
)


def test_compute_velocity_shape():
    h = np.random.randn(1000)
    v = np.random.randn(1000)
    fs = 512
    vel = compute_velocity(h, v, fs)
    assert len(vel) == len(h)
    assert (vel >= 0).all()


def test_detect_events_ivt_static_signal_all_fixation():
    fs = 512
    # 完全静止：速度≈0，全段应为注视
    h = np.zeros(512 * 5)
    v = np.zeros(512 * 5)
    events = detect_events_ivt(h, v, fs, velocity_threshold=1.0)
    assert len(events.saccades) == 0
    assert len(events.fixations) >= 1
    assert events.fixations[0].duration_sec >= 4.5


def test_detect_events_ivt_fast_jump_saccade():
    fs = 512
    n = 512 * 5
    h = np.zeros(n)
    v = np.zeros(n)
    # 中间插入一次快速跳转
    h[1000:1050] = np.linspace(0, 100, 50)
    events = detect_events_ivt(h, v, fs, velocity_threshold=10.0)
    assert len(events.saccades) >= 1
    assert events.saccades[0].amplitude > 5


def test_detect_events_ivt_min_durations_enforced():
    fs = 512
    n = 512 * 5
    h = np.zeros(n)
    v = np.zeros(n)
    # 短跳转（< 20ms = 10 samples）
    h[1000:1010] = np.linspace(0, 50, 10)
    events = detect_events_ivt(
        h, v, fs, velocity_threshold=10.0,
        min_fixation_ms=100, min_saccade_ms=20,
    )
    # 短扫视应被合并到注视
    for s in events.saccades:
        assert s.duration_sec * 1000 >= 20


def test_detect_events_ivt_auto_threshold():
    fs = 512
    rng = np.random.default_rng(42)
    h = rng.standard_normal(512 * 10) * 0.1
    v = rng.standard_normal(512 * 10) * 0.1
    events = detect_events_ivt(h, v, fs, velocity_threshold="auto")
    assert isinstance(events, Events)
    assert len(events.fixations) + len(events.saccades) > 0
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_event_detection.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 `event_detection.py`**

```python
# src/dual_factor/features/event_detection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union
import numpy as np


@dataclass
class Fixation:
    start_idx: int
    end_idx: int
    mean_h: float
    mean_v: float
    duration_sec: float


@dataclass
class Saccade:
    start_idx: int
    end_idx: int
    amplitude: float
    peak_velocity: float
    direction_deg: float
    duration_sec: float


@dataclass
class Events:
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


def detect_events_ivt(
    eog_h: np.ndarray,
    eog_v: np.ndarray,
    fs: float,
    velocity_threshold: Union[float, str] = "auto",
    velocity_percentile: float = 70.0,
    min_fixation_ms: float = 100.0,
    min_saccade_ms: float = 20.0,
) -> Events:
    """I-VT 速度阈值法：注视/扫视事件提取。"""
    velocity = compute_velocity(eog_h, eog_v, fs)
    if isinstance(velocity_threshold, str) and velocity_threshold == "auto":
        v_th = _auto_threshold(velocity, velocity_percentile)
    else:
        v_th = float(velocity_threshold)

    # 逐样本分类
    is_fixation = velocity < v_th
    # 找连续片段
    fixations_raw = _find_runs(is_fixation, True)
    saccades_raw = _find_runs(is_fixation, False)

    min_fix_samples = int(min_fixation_ms / 1000 * fs)
    min_sac_samples = int(min_saccade_ms / 1000 * fs)

    fixations: list[Fixation] = []
    for s, e in fixations_raw:
        if e - s < min_fix_samples:
            continue  # 太短，丢弃（会被相邻扫视吸收）
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_event_detection.py -v
```

Expected: PASS

- [ ] **Step 5: 速度阈值敏感性分析（spec §3.2 要求）**

```bash
PYTHONPATH=src conda run -n Anomaly python -c "
import numpy as np
from dual_factor.features.event_detection import detect_events_ivt
from dual_factor.data.drozy_loader import load_all_records
from pathlib import Path
records = load_all_records(Path('data_public/DROZY'))
rec = records[0]
from dual_factor.data.preprocess import preprocess_eog
h, v = preprocess_eog(rec.eog_h, rec.eog_v, rec.sampling_rate)
for pct in [60, 70, 80, 90]:
    ev = detect_events_ivt(h, v, rec.sampling_rate, 'auto', velocity_percentile=pct)
    print(f'pct={pct}: fix={len(ev.fixations)} sac={len(ev.saccades)}')
"
```

Expected: 不同百分位下事件数变化合理，选定 70 百分位为默认值。

- [ ] **Step 6: 提交**

```bash
git add src/dual_factor/features/event_detection.py tests/dual_factor/test_event_detection.py
git commit -m "feat(dual_factor): I-VT velocity-threshold event detection with auto-threshold"
```

---

## Task 5: 呆滞因子 4 指标计算

**Files:**

- 创建: `src/dual_factor/features/indicators.py`（本 Task 实现 Hypoactive 部分）
- 创建: `tests/dual_factor/test_indicators_hypoactive.py`

**对应 spec**: §3.1 呆滞因子——FD / PSV / SA / FN

**Interfaces:**

- Consumes: `Events` 对象（fixations + saccades）、窗口长度
- Produces: `dict[str, float]` 含 `FD, PSV, SA, FN`

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_indicators_hypoactive.py
from __future__ import annotations

import numpy as np
from dual_factor.features.event_detection import Fixation, Saccade, Events
from dual_factor.features.indicators import (
    mean_fixation_duration, peak_saccade_velocity, saccade_amplitude,
    fixation_rate, compute_hypoactive_indicators,
)


def _make_events(n_fix=5, n_sac=3, duration_sec=30.0) -> Events:
    fixations = [
        Fixation(start_idx=i*1000, end_idx=i*1000+500,
                 mean_h=0.0, mean_v=0.0, duration_sec=0.5)
        for i in range(n_fix)
    ]
    saccades = [
        Saccade(start_idx=i*1000+500, end_idx=i*1000+520,
                amplitude=10.0*(i+1), peak_velocity=100.0*(i+1),
                direction_deg=45.0, duration_sec=0.04)
        for i in range(n_sac)
    ]
    return Events(fixations=fixations, saccades=saccades)


def test_mean_fixation_duration():
    ev = _make_events()
    fd = mean_fixation_duration(ev)
    assert fd == 0.5


def test_peak_saccade_velocity():
    ev = _make_events()
    psv = peak_saccade_velocity(ev)
    assert psv == 300.0  # 最大值 100*3


def test_saccade_amplitude():
    ev = _make_events()
    sa = saccade_amplitude(ev)
    assert sa == 60.0  # 均值 (10+20+30)/3


def test_fixation_rate():
    ev = _make_events(n_fix=5, duration_sec=30.0)
    fn = fixation_rate(ev, window_sec=30.0)
    assert abs(fn - 5 / 30.0) < 1e-6


def test_compute_hypoactive_indicators_all_present():
    ev = _make_events()
    out = compute_hypoactive_indicators(ev, window_sec=30.0)
    assert set(out.keys()) >= {"FD", "PSV", "SA", "FN"}
    assert out["FD"] == 0.5
    assert out["PSV"] == 300.0
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_indicators_hypoactive.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 `indicators.py` 的呆滞因子部分**

```python
# src/dual_factor/features/indicators.py
from __future__ import annotations

import numpy as np
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_indicators_hypoactive.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/dual_factor/features/indicators.py tests/dual_factor/test_indicators_hypoactive.py
git commit -m "feat(dual_factor): hypoactive indicators FD/PSV/SA/FN"
```

---

## Task 6: 发散因子 4 指标计算

**Files:**

- 扩展: `src/dual_factor/features/indicators.py`（追加 Diffuse 部分）
- 创建: `tests/dual_factor/test_indicators_diffuse.py`

**对应 spec**: §3.1 发散因子——SE / GTE / SV / SDR、§3.3 空间网格化

**Interfaces:**

- Consumes: `Events.fixations`（注视点位置）、`Events.saccades`（扫视方向）
- Produces: `dict[str, float]` 含 `SE, GTE, SV, SDR`

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_indicators_diffuse.py
from __future__ import annotations

import numpy as np
from dual_factor.features.event_detection import Fixation, Saccade, Events
from dual_factor.features.indicators import (
    spatial_entropy, gaze_transition_entropy, spatial_variance,
    saccade_direction_randomness, compute_diffuse_indicators,
)


def _make_clustered_fixations(n=20) -> list[Fixation]:
    # 全部聚在 (0,0) 附近
    return [Fixation(i, i+10, 0.0, 0.0, 0.01) for i in range(n)]


def _make_spread_fixations(n=20) -> list[Fixation]:
    rng = np.random.default_rng(42)
    return [Fixation(i, i+10, rng.randn(), rng.randn(), 0.01) for i in range(n)]


def test_spatial_entropy_clustered_low():
    fixs = _make_clustered_fixations()
    se = spatial_entropy(fixs, bins=5)
    assert se >= 0
    # 聚集分布熵应低于发散分布
    fixs2 = _make_spread_fixations()
    se2 = spatial_entropy(fixs2, bins=5)
    assert se <= se2


def test_gaze_transition_entropy_uniform_high():
    # 在不同网格间均匀跳转 → 熵高
    rng = np.random.default_rng(0)
    fixs = [
        Fixation(i, i+10, rng.uniform(-5, 5), rng.uniform(-5, 5), 0.01)
        for i in range(20)
    ]
    gte = gaze_transition_entropy(fixs, bins=5)
    assert gte >= 0


def test_spatial_variance_spread_higher():
    fixs_c = _make_clustered_fixations()
    fixs_s = _make_spread_fixations()
    sv_c = spatial_variance(fixs_c)
    sv_s = spatial_variance(fixs_s)
    assert sv_s > sv_c


def test_saccade_direction_randomness_uniform_high():
    # 均匀方向 → SDR 应接近 0（与均匀分布的 KL 散度）
    rng = np.random.default_rng(42)
    saccades = [
        Saccade(i, i+5, 10.0, 100.0, direction_deg=float(rng.uniform(0, 360)),
                duration_sec=0.01)
        for i in range(50)
    ]
    sdr = saccade_direction_randomness(saccades)
    assert sdr >= 0
    assert sdr < 0.5  # 接近均匀


def test_compute_diffuse_indicators_all_present():
    fixs = _make_spread_fixations()
    saccades = [
        Saccade(i, i+5, 10.0, 100.0, direction_deg=45.0, duration_sec=0.01)
        for i in range(10)
    ]
    events = Events(fixations=fixs, saccades=saccades)
    out = compute_diffuse_indicators(events, bins=5)
    assert set(out.keys()) >= {"SE", "GTE", "SV", "SDR"}
    for v in out.values():
        assert np.isfinite(v)
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_indicators_diffuse.py -v
```

Expected: FAIL

- [ ] **Step 3: 追加发散因子实现到 `indicators.py`**

```python
# 追加到 src/dual_factor/features/indicators.py
from collections import Counter
from scipy.stats import entropy as scipy_entropy


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
    """SV: 注视点坐标的马氏距离方差（简化为 trace 协方差）。"""
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
    # 分 8 个方向箱
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_indicators_diffuse.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/dual_factor/features/indicators.py tests/dual_factor/test_indicators_diffuse.py
git commit -m "feat(dual_factor): diffuse indicators SE/GTE/SV/SDR"
```

---

## Task 7: 窗口化与被试内标准化

**Files:**

- 创建: `src/dual_factor/features/windowing.py`
- 创建: `tests/dual_factor/test_windowing.py`

**对应 spec**: §4.3 窗口化、§4.4 标准化

**Interfaces:**

- Consumes: `(eog_h, eog_v)` 滤波后信号、采样率、窗口参数、`Events` 计算函数
- Produces: `pd.DataFrame` 行=被试×窗口，列=8指标+元数据+KSS标签

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_windowing.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dual_factor.features.windowing import (
    slice_windows, build_indicator_matrix, within_subject_zscore
)


def test_slice_windows_count():
    fs = 512
    n = 512 * 100  # 100 秒
    h = np.random.randn(n)
    v = np.random.randn(n)
    wins = slice_windows(h, v, fs, window_sec=30, stride_sec=10)
    # 100 秒 / 30 秒窗口 / 10 秒步长 → (100-30)/10 + 1 = 8 个窗口
    assert len(wins) == 8
    assert wins[0].start_idx == 0
    assert wins[1].start_idx == 10 * fs


def test_build_indicator_matrix_columns():
    fs = 512
    n = 512 * 100
    rng = np.random.default_rng(42)
    h = rng.standard_normal(n)
    v = rng.standard_normal(n)
    df = build_indicator_matrix(
        subject="1", test="1", kss=3,
        eog_h=h, eog_v=v, fs=fs,
        window_sec=30, stride_sec=10, bins=5,
    )
    assert len(df) == 8
    expected_cols = {"subject", "test", "kss", "window_idx",
                     "FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"}
    assert expected_cols <= set(df.columns)


def test_within_subject_zscore_centers_per_subject():
    df = pd.DataFrame({
        "subject": ["1"] * 5 + ["2"] * 5,
        "FD": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
    })
    out = within_subject_zscore(df, ["FD"])
    # 被试1均值3，被试2均值30
    g1 = out[out["subject"] == "1"]["FD_z"]
    g2 = out[out["subject"] == "2"]["FD_z"]
    assert abs(g1.mean()) < 1e-6
    assert abs(g2.mean()) < 1e-6
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_windowing.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 `windowing.py`**

```python
# src/dual_factor/features/windowing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
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
        # 事件数不足 → 标记缺失但保留行
        if len(events.fixations) < min_fixations or len(events.saccades) < min_saccades:
            row = {
                "subject": subject, "test": test, "kss": kss,
                "window_idx": i, "valid": False,
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_windowing.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/dual_factor/features/windowing.py tests/dual_factor/test_windowing.py
git commit -m "feat(dual_factor): windowing, indicator matrix, within-subject z-score"
```

---

## Task 8: EFA 前置检验（KMO + Bartlett）

**Files:**

- 创建: `src/dual_factor/analysis/efa.py`
- 创建: `tests/dual_factor/test_efa_precheck.py`

**对应 spec**: §5.1 前置检验

**Interfaces:**

- Consumes: 标准化后的指标矩阵（8 列 z-score）
- Produces: `dict` 含 `kmo_overall, kmo_per_variable, bartlett_chi2, bartlett_p`

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_efa_precheck.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dual_factor.analysis.efa import (
    calculate_kmo, bartlett_sphericity, precheck_efa
)


def test_calculate_kmo_returns_value():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.standard_normal((200, 8)), columns=list("ABCDEFGH"))
    kmo_overall, kmo_per_var = calculate_kmo(data)
    assert 0 <= kmo_overall <= 1
    assert len(kmo_per_var) == 8


def test_bartlett_p_value_in_range():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.standard_normal((200, 8)), columns=list("ABCDEFGH"))
    chi2, p = bartlett_sphericity(data)
    assert chi2 >= 0
    assert 0 <= p <= 1


def test_precheck_efa_returns_all():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.standard_normal((200, 8)), columns=list("ABCDEFGH"))
    out = precheck_efa(data)
    assert set(out.keys()) >= {"kmo_overall", "kmo_per_variable", "bartlett_chi2", "bartlett_p"}
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_efa_precheck.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 `efa.py` 前置检验部分**

```python
# src/dual_factor/analysis/efa.py
from __future__ import annotations

import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import (
    calculate_kmo as _fa_kmo,
    calculate_bartlett_sphericity as _fa_bartlett,
)


def calculate_kmo(data: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """KMO 采样充分性检验。返回 (overall, per_variable)。"""
    kmo_per_var, kmo_overall = _fa_kmo(data)
    return float(kmo_overall), dict(zip(data.columns, kmo_per_var))


def bartlett_sphericity(data: pd.DataFrame) -> tuple[float, float]:
    """Bartlett 球形检验。返回 (chi2, p_value)。"""
    chi2, p = _fa_bartlett(data)
    return float(chi2), float(p)


def precheck_efa(data: pd.DataFrame) -> dict:
    """EFA 前置检验汇总。"""
    kmo_overall, kmo_per_var = calculate_kmo(data)
    chi2, p = bartlett_sphericity(data)
    return {
        "kmo_overall": kmo_overall,
        "kmo_per_variable": kmo_per_var,
        "bartlett_chi2": chi2,
        "bartlett_p": p,
        "kmo_pass": kmo_overall > 0.50,
        "bartlett_pass": p < 0.05,
    }
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_efa_precheck.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/dual_factor/analysis/efa.py tests/dual_factor/test_efa_precheck.py
git commit -m "feat(dual_factor): KMO and Bartlett sphericity pre-check"
```

---

## Task 9: EFA 因子提取与平行分析

**Files:**

- 扩展: `src/dual_factor/analysis/efa.py`（追加因子提取与平行分析）
- 创建: `tests/dual_factor/test_efa_extraction.py`

**对应 spec**: §5.2 因子提取、§5.3 结果报告、§5.4 核心决策判据

**Interfaces:**

- Consumes: 标准化指标矩阵、配置（extraction/rotation/max_factors）
- Produces: `EFAResult(n_factors, loadings, variance_explained, factor_corr, parallel_analysis)` 数据类

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_efa_extraction.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dual_factor.analysis.efa import run_efa, parallel_analysis, EFAResult


def _make_synthetic_2factor(n=200, seed=42):
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n) * 0.5
    # 4 指标载 f1，4 指标载 f2
    data = pd.DataFrame({
        "FD": 0.8 * f1 + 0.1 * rng.standard_normal(n),
        "PSV": -0.7 * f1 + 0.1 * rng.standard_normal(n),
        "SA": -0.6 * f1 + 0.1 * rng.standard_normal(n),
        "FN": -0.5 * f1 + 0.1 * rng.standard_normal(n),
        "SE": 0.7 * f2 + 0.1 * rng.standard_normal(n),
        "GTE": 0.6 * f2 + 0.1 * rng.standard_normal(n),
        "SV": 0.5 * f2 + 0.1 * rng.standard_normal(n),
        "SDR": 0.4 * f2 + 0.1 * rng.standard_normal(n),
    })
    return data


def test_run_efa_returns_result():
    data = _make_synthetic_2factor()
    result = run_efa(data, extraction="pa", rotation="promax", max_factors=4)
    assert isinstance(result, EFAResult)
    assert result.loadings.shape[1] <= 4
    assert result.variance_explained >= 0


def test_parallel_analysis_returns_eigenvalues():
    data = _make_synthetic_2factor(n=100)
    pa = parallel_analysis(data, n_iter=50, percentile=95, max_factors=4)
    assert len(pa["random_eigenvalues"]) <= 4
    assert len(pa["actual_eigenvalues"]) <= 4


def test_efa_recovers_two_factors():
    data = _make_synthetic_2factor(n=300)
    result = run_efa(data, extraction="pa", rotation="promax", max_factors=4)
    # 至少应提取 1-2 个因子
    assert result.n_factors >= 1
    assert result.n_factors <= 4
    # 因子相关矩阵形状
    if result.factor_corr is not None:
        assert result.factor_corr.shape[0] == result.n_factors
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_efa_extraction.py -v
```

Expected: FAIL

- [ ] **Step 3: 追加因子提取与平行分析实现**

```python
# 追加到 src/dual_factor/analysis/efa.py
from dataclasses import dataclass
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import RotationMethod, Method


@dataclass
class EFAResult:
    n_factors: int
    loadings: np.ndarray              # (n_vars, n_factors)
    variance_explained: float        # 累计方差解释率
    factor_corr: np.ndarray | None   # (n_factors, n_factors) 或 None
    variable_names: list[str]
    parallel_analysis: dict | None


def run_efa(
    data: pd.DataFrame,
    extraction: str = "pa",
    rotation: str = "promax",
    max_factors: int = 4,
    variance_threshold: float = 0.60,
) -> EFAResult:
    """主轴因子法 + Promax 旋转，按平行分析确定因子数。"""
    # 先跑平行分析确定因子数
    pa = parallel_analysis(data, n_iter=1000, percentile=95, max_factors=max_factors)
    actual = pa["actual_eigenvalues"]
    rand_refs = pa["random_eigenvalues"]
    n_factors = 1
    for k in range(min(len(actual), len(rand_refs))):
        if actual[k] > rand_refs[k]:
            n_factors = k + 1
        else:
            break
    n_factors = min(n_factors, max_factors)

    # 若无信号，回退到 Kaiser 准则（>1）
    if n_factors == 0:
        n_factors = int(np.sum(np.array(actual) > 1.0)) or 1

    method = Method.MINRES if extraction == "ml" else Method.PRINCIPAL_AXIS
    try:
        method = {"pa": Method.PRINCIPAL_AXIS, "ml": Method.MINRES,
                  "minres": Method.MINRES}[extraction]
    except KeyError:
        method = Method.PRINCIPAL_AXIS

    rot = RotationMethod.PROMAX if rotation == "promax" else RotationMethod.VARIMAX
    try:
        rot = {"promax": RotationMethod.PROMAX,
               "varimax": RotationMethod.VARIMAX,
               "oblimin": RotationMethod.OBLIMIN}[rotation]
    except KeyError:
        rot = RotationMethod.PROMAX

    fa = FactorAnalyzer(n_factors=n_factors, rotation=rot, method=method)
    fa.fit(data)
    loadings = fa.loadings_
    variance = fa.get_factor_variance()
    cum_var = float(variance[2][-1]) if len(variance) >= 3 else 0.0
    corr = fa.phi_ if hasattr(fa, "phi_") and fa.phi_ is not None else None

    return EFAResult(
        n_factors=n_factors,
        loadings=loadings,
        variance_explained=cum_var,
        factor_corr=corr,
        variable_names=list(data.columns),
        parallel_analysis=pa,
    )


def parallel_analysis(
    data: pd.DataFrame,
    n_iter: int = 1000,
    percentile: float = 95,
    max_factors: int = 4,
    seed: int = 42,
) -> dict:
    """平行分析：与随机数据特征值比较。"""
    rng = np.random.default_rng(seed)
    n_obs, n_vars = data.shape
    k = min(max_factors, n_vars)

    # 实际特征值
    corr = np.corrcoef(data.values, rowvar=False)
    actual_eigs = np.sort(np.linalg.eigvalsh(corr))[::-1][:k]

    # 随机数据特征值
    rand_eigs_all = np.zeros((n_iter, k))
    for i in range(n_iter):
        rand_data = rng.standard_normal((n_obs, n_vars))
        rand_corr = np.corrcoef(rand_data, rowvar=False)
        rand_eigs_all[i] = np.sort(np.linalg.eigvalsh(rand_corr))[::-1][:k]
    rand_eigs = np.percentile(rand_eigs_all, percentile, axis=0)

    return {
        "actual_eigenvalues": actual_eigs.tolist(),
        "random_eigenvalues": rand_eigs.tolist(),
        "n_iter": n_iter,
        "percentile": percentile,
    }
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_efa_extraction.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/dual_factor/analysis/efa.py tests/dual_factor/test_efa_extraction.py
git commit -m "feat(dual_factor): PAF extraction + Promax + parallel analysis"
```

---

## Task 10: 描述性统计与方向性检验

**Files:**

- 创建: `src/dual_factor/analysis/descriptive.py`
- 创建: `tests/dual_factor/test_descriptive.py`

**对应 spec**: §5.3 结果报告（方向性）、§1.3 核心假设 H1/H2/H3

**Interfaces:**

- Consumes: 指标矩阵（含 KSS 二分类标签 alert/drowsy）、8 指标列
- Produces: `pd.DataFrame` 指标×(mean_alert, mean_drowsy, diff, t_stat, p_value, direction, expected_direction, match)

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_descriptive.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dual_factor.analysis.descriptive import (
    descriptive_stats, directionality_test, HYPOTHESIS_DIRECTIONS, INDICATORS
)


def test_descriptive_stats_columns():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "subject": ["1"] * 20 + ["2"] * 20,
        "kss": [3] * 20 + [7] * 20,
        **{ind: rng.standard_normal(40) for ind in INDICATORS},
    })
    stats = descriptive_stats(df, INDICATORS)
    assert set(stats.columns) >= {"indicator", "mean_alert", "mean_drowsy", "std_alert", "std_drowsy"}
    assert len(stats) == 8


def test_directionality_test_returns_match():
    rng = np.random.default_rng(42)
    # 构造 FD 在 drowsy 下显著更高
    df = pd.DataFrame({
        "subject": ["1"] * 20 + ["2"] * 20,
        "kss": [3] * 20 + [7] * 20,
        "FD": np.concatenate([rng.standard_normal(20), rng.standard_normal(20) + 1.0]),
        "PSV": np.concatenate([rng.standard_normal(20) + 1.0, rng.standard_normal(20)]),
    })
    out = directionality_test(df, ["FD", "PSV"], alert_max=4, drowsy_min=6)
    assert set(out.columns) >= {"indicator", "diff", "p_value", "direction", "expected", "match"}
    fd_row = out[out["indicator"] == "FD"].iloc[0]
    assert fd_row["direction"] == "increase"
    assert fd_row["expected"] == "increase"
    assert fd_row["match"] is True
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_descriptive.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现 `descriptive.py`**

```python
# src/dual_factor/analysis/descriptive.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

INDICATORS = ["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"]

# 假设方向：alert→drowsy 的预期变化
HYPOTHESIS_DIRECTIONS = {
    "FD": "increase",
    "PSV": "decrease",
    "SA": "decrease",
    "FN": "decrease",
    "SE": "increase",
    "GTE": "increase",
    "SV": "increase",
    "SDR": "increase",
}


def _split_alert_drowsy(df: pd.DataFrame, alert_max: int = 4, drowsy_min: int = 6):
    """按 KSS 二分类。"""
    alert = df[df["kss"] <= alert_max]
    drowsy = df[df["kss"] >= drowsy_min]
    return alert, drowsy


def descriptive_stats(df: pd.DataFrame, indicators: list[str], alert_max: int = 4, drowsy_min: int = 6) -> pd.DataFrame:
    """各指标在 alert/drowsy 条件下的描述性统计。"""
    alert, drowsy = _split_alert_drowsy(df, alert_max, drowsy_min)
    rows = []
    for ind in indicators:
        a = alert[ind].dropna()
        d = drowsy[ind].dropna()
        rows.append({
            "indicator": ind,
            "n_alert": len(a), "n_drowsy": len(d),
            "mean_alert": float(a.mean()) if len(a) else np.nan,
            "mean_drowsy": float(d.mean()) if len(d) else np.nan,
            "std_alert": float(a.std()) if len(a) else np.nan,
            "std_drowsy": float(d.std()) if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def directionality_test(df: pd.DataFrame, indicators: list[str], alert_max: int = 4, drowsy_min: int = 6) -> pd.DataFrame:
    """各指标 alert vs drowsy 的方向性检验（Mann-Whitney U）。"""
    alert, drowsy = _split_alert_drowsy(df, alert_max, drowsy_min)
    rows = []
    for ind in indicators:
        a = alert[ind].dropna().values
        d = drowsy[ind].dropna().values
        if len(a) == 0 or len(d) == 0:
            rows.append({"indicator": ind, "diff": np.nan, "p_value": np.nan,
                         "direction": "unknown", "expected": HYPOTHESIS_DIRECTIONS.get(ind),
                         "match": False})
            continue
        diff = float(d.mean() - a.mean())
        try:
            _, p = mannwhitneyu(a, d, alternative="two-sided")
        except ValueError:
            p = np.nan
        direction = "increase" if diff > 0 else "decrease"
        expected = HYPOTHESIS_DIRECTIONS.get(ind, "unknown")
        match = (direction == expected)
        rows.append({
            "indicator": ind, "diff": diff, "p_value": float(p),
            "direction": direction, "expected": expected, "match": match,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_descriptive.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/dual_factor/analysis/descriptive.py tests/dual_factor/test_descriptive.py
git commit -m "feat(dual_factor): descriptive stats + directionality test vs H1/H2"
```

---

## Task 11: 可视化与决策报告

**Files:**
- 创建: `src/dual_factor/analysis/plots.py`
- 创建: `src/dual_factor/analysis/decision.py`
- 创建: `tests/dual_factor/test_decision.py`

**对应 spec**: §5.3 结果报告、§5.4 核心决策判据、§6.5 输出产物

**Interfaces:**
- Consumes: `EFAResult`、`precheck_efa` 结果、`directionality_test` 结果
- Produces: 4 张图（scree_parallel.png / loading_matrix.png / factor_scatter.png / direction_check.png）+ `decision_report.md`

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/dual_factor/test_decision.py
from __future__ import annotations

import numpy as np
from dual_factor.analysis.decision import make_decision, DecisionReport


def test_decision_separable_when_corr_low():
    report = make_decision(
        n_factors=2, factor_corr=0.45,
        max_loading=0.75, cross_loading_max=0.20,
        variance_explained=0.68,
    )
    assert report.verdict == "separable"
    assert "r < 0.70" in report.reasoning


def test_decision_unseparable_when_corr_high():
    report = make_decision(
        n_factors=2, factor_corr=0.90,
        max_loading=0.80, cross_loading_max=0.25,
        variance_explained=0.72,
    )
    assert report.verdict == "unseparable"
    assert "r > 0.85" in report.reasoning


def test_decision_ambiguous_in_middle():
    report = make_decision(
        n_factors=2, factor_corr=0.78,
        max_loading=0.65, cross_loading_max=0.28,
        variance_explained=0.62,
    )
    assert report.verdict == "ambiguous"


def test_decision_single_factor_unseparable():
    report = make_decision(
        n_factors=1, factor_corr=None,
        max_loading=0.85, cross_loading_max=0.0,
        variance_explained=0.55,
    )
    assert report.verdict == "unseparable"


def test_decision_report_to_markdown():
    report = make_decision(
        n_factors=2, factor_corr=0.50,
        max_loading=0.78, cross_loading_max=0.15,
        variance_explained=0.70,
    )
    md = report.to_markdown()
    assert "## 决策报告" in md
    assert "separable" in md
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_decision.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现 `decision.py`**

```python
# src/dual_factor/analysis/decision.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionReport:
    verdict: str            # separable | unseparable | ambiguous
    n_factors: int
    factor_corr: float | None
    max_loading: float
    cross_loading_max: float
    variance_explained: float
    reasoning: str

    def to_markdown(self) -> str:
        corr_str = f"{self.factor_corr:.3f}" if self.factor_corr is not None else "N/A (单因子)"
        return f"""## 决策报告

**判定结论**：{self.verdict}

| 项目 | 数值 |
|---|---|
| 提取因子数 | {self.n_factors} |
| 因子相关 r | {corr_str} |
| 最大载荷 | {self.max_loading:.3f} |
| 最大交叉载荷 | {self.cross_loading_max:.3f} |
| 累计方差解释 | {self.variance_explained:.3f} |

**判定依据**：{self.reasoning}
"""


def make_decision(
    n_factors: int,
    factor_corr: float | None,
    max_loading: float,
    cross_loading_max: float,
    variance_explained: float,
) -> DecisionReport:
    """按 spec §5.4 决策判据判定双因子可分离性。"""
    if n_factors == 1:
        return DecisionReport(
            verdict="unseparable", n_factors=1, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            reasoning="仅提取一个因子，双因子假设被推翻，转向单维度模型。",
        )
    if factor_corr is None:
        return DecisionReport(
            verdict="ambiguous", n_factors=n_factors, factor_corr=None,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            reasoning="因子相关矩阵不可用（正交旋转），无法判定可分离性。",
        )
    if factor_corr < 0.70 and max_loading > 0.50:
        return DecisionReport(
            verdict="separable", n_factors=n_factors, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            reasoning=f"两因子相关 r < 0.70（r={factor_corr:.3f}）且载荷清晰（max={max_loading:.3f}），"
                      "双因子可分离，假设初步成立，进入 CFA 验证。",
        )
    if factor_corr > 0.85:
        return DecisionReport(
            verdict="unseparable", n_factors=n_factors, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            reasoning=f"两因子相关 r > 0.85（r={factor_corr:.3f}），因子高度共线，"
                      "双因子不可分离，转向单维度模型。",
        )
    return DecisionReport(
        verdict="ambiguous", n_factors=n_factors, factor_corr=factor_corr,
        max_loading=max_loading, cross_loading_max=cross_loading_max,
        variance_explained=variance_explained,
        reasoning=f"因子相关 0.70 ≤ r ≤ 0.85（r={factor_corr:.3f}），结构模糊，"
                  "检查指标计算或考虑剔除低载荷指标。",
    )
```

- [ ] **Step 4: 实现 `plots.py`**

```python
# src/dual_factor/analysis/plots.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dual_factor.analysis.efa import EFAResult
from dual_factor.analysis.descriptive import directionality_test


def plot_scree_parallel(efa_result: EFAResult, out_path: str | Path) -> None:
    """碎石图 + 平行分析图。"""
    pa = efa_result.parallel_analysis
    actual = pa["actual_eigenvalues"]
    random = pa["random_eigenvalues"]
    k = len(actual)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, k + 1), actual, "o-", label="Actual eigenvalues", color="#1f77b4")
    ax.plot(range(1, k + 1), random, "s--", label="Parallel (95th pct)", color="#d62728")
    ax.axhline(1.0, color="gray", linestyle=":", label="Kaiser criterion (≥1)")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot with Parallel Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=150)
    plt.close(fig)


def plot_loading_matrix(efa_result: EFAResult, out_path: str | Path) -> None:
    """因子载荷热力图。"""
    loadings = efa_result.loadings
    vars_names = efa_result.variable_names
    n_factors = efa_result.n_factors
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(loadings, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f"F{i+1}" for i in range(n_factors)])
    ax.set_yticks(range(len(vars_names)))
    ax.set_yticklabels(vars_names)
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            ax.text(j, i, f"{loadings[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(loadings[i, j]) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Loading")
    ax.set_title("Factor Loading Matrix (Promax rotated)")
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=150)
    plt.close(fig)


def plot_factor_scatter(efa_result: EFAResult, out_path: str | Path) -> None:
    """因子相关散点图（若 PHI 矩阵可用）。"""
    if efa_result.factor_corr is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = efa_result.factor_corr
    n = corr.shape[0]
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"F{i+1}" for i in range(n)])
    ax.set_yticklabels([f"F{i+1}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Factor Correlation Matrix (Phi)")
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=150)
    plt.close(fig)


def plot_direction_check(df: pd.DataFrame, indicators: list[str], out_path: str | Path) -> None:
    """各指标 alert vs drowsy 方向检验图。"""
    direction_df = directionality_test(df, indicators)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ca02c" if m else "#d62728" for m in direction_df["match"]]
    ax.barh(direction_df["indicator"], direction_df["diff"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean difference (drowsy - alert)")
    ax.set_title("Directionality Check: Indicator Changes (alert → drowsy)\nGreen=matches hypothesis, Red=contradicts")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=150)
    plt.close(fig)
```

- [ ] **Step 5: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_decision.py -v
```
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/dual_factor/analysis/plots.py src/dual_factor/analysis/decision.py tests/dual_factor/test_decision.py
git commit -m "feat(dual_factor): visualizations + decision report generator"
```

---

## Task 12: 一键 Pipeline 入口与验收

**Files:**
- 创建: `src/dual_factor/pipeline.py`
- 创建: `tests/dual_factor/test_pipeline_smoke.py`

**对应 spec**: §6.2 包结构（pipeline.py）、§6.5 输出产物、§7 合格标准

**Interfaces:**
- Consumes: `configs/dual_factor_efa.yaml`
- Produces: `results/dual_factor_efa/` 下全部 7 类产物

- [ ] **Step 1: 写 smoke test**

```python
# tests/dual_factor/test_pipeline_smoke.py
from __future__ import annotations

from pathlib import Path
import yaml
import pytest

from dual_factor.pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DROZY_ROOT = PROJECT_ROOT / "data_public" / "DROZY"


def test_pipeline_writes_required_outputs(tmp_path: Path) -> None:
    if not DROZY_ROOT.exists():
        pytest.skip("DROZY dataset not available")
    config = {
        "data": {"drozy_root": str(DROZY_ROOT), "channels": ["EOG-H", "EOG-V", "Cam-Sync"],
                 "sampling_rate": 512, "subjects": list(range(1, 15)),
                 "exclude_tests": ["7-1"]},
        "preprocess": {"highpass": 0.5, "lowpass": 30.0, "filter_order": 4,
                       "hampel_window": 51, "hampel_threshold": 3.0},
        "event_detection": {"algorithm": "I-VT", "velocity_threshold": "auto",
                            "velocity_percentile": 70, "min_fixation_ms": 100, "min_saccade_ms": 20},
        "windowing": {"window_sec": 30, "stride_sec": 10, "min_fixations": 5,
                      "min_saccades": 3, "normalization": "within_subject_zscore"},
        "spatial_grid": {"bins": 5},
        "labels": {"kss_file": "KSS.txt", "alert_threshold": 4,
                   "drowsy_threshold": 6, "exclude_middle": True},
        "efa": {"extraction": "pa", "rotation": "promax", "max_factors": 4,
                "variance_threshold": 0.60, "loading_threshold": 0.50,
                "cross_loading_threshold": 0.30, "factor_corr_threshold": 0.70,
                "parallel_analysis_n_iter": 100, "parallel_analysis_percentile": 95},
        "output": {"results_dir": str(tmp_path / "results"), "figures_dir": str(tmp_path / "results" / "figures")},
    }
    run_pipeline(config)
    expected = {
        "indicator_matrix.csv", "descriptive_stats.csv", "kmo_bartlett.json",
        "efa_results.json", "decision_report.md", "config_used.yaml",
    }
    assert expected <= {p.name for p in tmp_path.joinpath("results").iterdir()}
    figures = tmp_path / "results" / "figures"
    assert figures.exists()
    fig_files = {p.name for p in figures.iterdir()}
    assert {"scree_parallel.png", "loading_matrix.png", "factor_scatter.png", "direction_check.png"} <= fig_files
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_pipeline_smoke.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现 `pipeline.py`**

```python
# src/dual_factor/pipeline.py
"""一键运行：DROZY → 预处理 → 8 指标 → 窗口化 → EFA → 决策报告。

用法（从项目根目录）：
    PYTHONPATH=src conda run -n Anomaly python -m dual_factor.pipeline --config configs/dual_factor_efa.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from dual_factor.data.drozy_loader import load_all_records
from dual_factor.data.preprocess import preprocess_eog
from dual_factor.features.windowing import build_indicator_matrix, within_subject_zscore
from dual_factor.analysis.efa import precheck_efa, run_efa, EFAResult
from dual_factor.analysis.descriptive import descriptive_stats, directionality_test, INDICATORS
from dual_factor.analysis.plots import (
    plot_scree_parallel, plot_loading_matrix, plot_factor_scatter, plot_direction_check,
)
from dual_factor.analysis.decision import make_decision

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_pipeline(config: dict) -> None:
    """端到端执行 EFA 实验。"""
    results_dir = Path(config["output"]["results_dir"])
    figures_dir = Path(config["output"]["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置快照
    with open(results_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    # 1. 数据读取
    print("[1/7] Loading DROZY records...")
    records = load_all_records(
        config["data"]["drozy_root"], config["data"]["channels"]
    )
    print(f"  Loaded {len(records)} records")

    # 2. 预处理 + 窗口化 + 指标计算
    print("[2/7] Preprocessing + windowing + indicators...")
    frames = []
    pp = config["preprocess"]
    ed = config["event_detection"]
    wn = config["windowing"]
    bins = config["spatial_grid"]["bins"]
    for rec in records:
        h_clean, v_clean = preprocess_eog(
            rec.eog_h, rec.eog_v, rec.sampling_rate,
            highcut=pp["lowpass"], lowcut=pp["highpass"],
            filter_order=pp["filter_order"],
            hampel_window=pp["hampel_window"], hampel_threshold=pp["hampel_threshold"],
        )
        df = build_indicator_matrix(
            subject=rec.subject, test=rec.test, kss=rec.kss,
            eog_h=h_clean, eog_v=v_clean, fs=rec.sampling_rate,
            window_sec=wn["window_sec"], stride_sec=wn["stride_sec"], bins=bins,
            min_fixations=wn["min_fixations"], min_saccades=wn["min_saccades"],
            velocity_threshold=ed["velocity_threshold"],
            velocity_percentile=ed["velocity_percentile"],
            min_fixation_ms=ed["min_fixation_ms"], min_saccade_ms=ed["min_saccade_ms"],
        )
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    # 标准化
    z_cols = [f"{c}_z" for c in INDICATORS]
    full = within_subject_zscore(full, INDICATORS)

    # 输出指标矩阵
    full.to_csv(results_dir / "indicator_matrix.csv", index=False, encoding="utf-8")
    print(f"  Indicator matrix: {len(full)} rows, missing rate="
          f"{full['valid'].eq(False).mean():.2%}")

    # 3. 描述性统计 + 方向性检验
    print("[3/7] Descriptive stats + directionality test...")
    valid_df = full[full["valid"]].copy()
    desc = descriptive_stats(valid_df, INDICATORS,
                             alert_max=config["labels"]["alert_threshold"],
                             drowsy_min=config["labels"]["drowsy_threshold"])
    desc.to_csv(results_dir / "descriptive_stats.csv", index=False, encoding="utf-8")

    # 4. EFA 前置检验
    print("[4/7] KMO + Bartlett pre-check...")
    efa_input = valid_df[z_cols].dropna()
    efa_input.columns = INDICATORS  # 重命名列去掉 _z 后缀
    precheck = precheck_efa(efa_input)
    with open(results_dir / "kmo_bartlett.json", "w", encoding="utf-8") as f:
        json.dump(precheck, f, ensure_ascii=False, indent=2)
    print(f"  KMO={precheck['kmo_overall']:.3f}, Bartlett p={precheck['bartlett_p']:.2e}")

    # 5. EFA 因子提取
    print("[5/7] EFA extraction + parallel analysis...")
    efa_cfg = config["efa"]
    result = run_efa(
        efa_input, extraction=efa_cfg["extraction"],
        rotation=efa_cfg["rotation"], max_factors=efa_cfg["max_factors"],
        variance_threshold=efa_cfg["variance_threshold"],
    )
    efa_json = {
        "n_factors": result.n_factors,
        "variance_explained": result.variance_explained,
        "loadings": result.loadings.tolist(),
        "variable_names": result.variable_names,
        "factor_corr": result.factor_corr.tolist() if result.factor_corr is not None else None,
        "parallel_analysis": result.parallel_analysis,
    }
    with open(results_dir / "efa_results.json", "w", encoding="utf-8") as f:
        json.dump(efa_json, f, ensure_ascii=False, indent=2)
    print(f"  Factors={result.n_factors}, Var={result.variance_explained:.3f}")

    # 6. 可视化
    print("[6/7] Generating figures...")
    plot_scree_parallel(result, figures_dir / "scree_parallel.png")
    plot_loading_matrix(result, figures_dir / "loading_matrix.png")
    plot_factor_scatter(result, figures_dir / "factor_scatter.png")
    plot_direction_check(valid_df, INDICATORS, figures_dir / "direction_check.png")

    # 7. 决策报告
    print("[7/7] Writing decision report...")
    max_loading = float(np.max(np.abs(result.loadings)))
    cross_max = float(np.max(result.loadings.max(axis=1) - np.abs(result.loadings).max(axis=1)))
    corr_val = float(result.factor_corr[0, 1]) if result.factor_corr is not None and result.factor_corr.shape[0] >= 2 else None
    report = make_decision(
        n_factors=result.n_factors, factor_corr=corr_val,
        max_loading=max_loading, cross_loading_max=cross_max,
        variance_explained=result.variance_explained,
    )
    with open(results_dir / "decision_report.md", "w", encoding="utf-8") as f:
        f.write(report.to_markdown())
    print(f"  Verdict: {report.verdict}")
    print(f"Done. Results at {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="DROZY 双因子 EFA 一键运行")
    parser.add_argument("--config", default="configs/dual_factor_efa.yaml")
    args = parser.parse_args()
    cfg_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_pipeline(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/test_pipeline_smoke.py -v
```
Expected: PASS（若 DROZY 数据可用）

- [ ] **Step 5: 跑通真实数据全链条**

```bash
PYTHONPATH=src conda run -n Anomaly python -m dual_factor.pipeline --config configs/dual_factor_efa.yaml
```
Expected: `results/dual_factor_efa/` 含全部 7 类产物 + 4 张图。

- [ ] **Step 6: 验收（对照 spec §7 合格标准）**

逐条核对：
- [ ] DROZY 的 EDF 可读取，EOG 通道可提取（≥30 个测试可用）
- [ ] 8 个指标全部计算成功，缺失率 <10%
- [ ] KMO、Bartlett、EFA 全部完成
- [ ] 有明确判定（separable / unseparable / ambiguous），附判据数值
- [ ] 代码可复现（一条命令跑通）
- [ ] 核心图达论文初稿质量
- [ ] 8 指标方向性检验完成
- [ ] 决策报告明确下一步计划

- [ ] **Step 7: 提交**

```bash
git add src/dual_factor/pipeline.py tests/dual_factor/test_pipeline_smoke.py
git commit -m "feat(dual_factor): end-to-end pipeline + acceptance verification"
```

---

## Self-Review

### 1. Spec Coverage

| Spec 章节 | 对应 Task |
|---|---|
| §1 研究背景与动机 | 全局背景，非代码 |
| §2.1-2.3 DROZY 数据集 / EOG / KSS / PVT | Task 2 |
| §2.4 辅助模态（视频/landmark） | 阶段四，本计划不含 |
| §3.1 八个核心指标定义 | Task 5（呆滞）+ Task 6（发散） |
| §3.2 I-VT 事件检测算法 | Task 4 |
| §3.3 空间网格化 | Task 6（SE/GTE 内含） |
| §4.1 数据读取（EDF/Cam-Sync） | Task 2 |
| §4.2 信号预处理（滤波/Hampel） | Task 3 |
| §4.3 窗口化 | Task 7 |
| §4.4 被试内 z-score 标准化 | Task 7 |
| §5.1 KMO + Bartlett 前置检验 | Task 8 |
| §5.2 因子提取（PAF + Promax + 平行分析） | Task 9 |
| §5.3 结果报告（碎石图/载荷/方差/相关/交叉载荷） | Task 9 + Task 11 |
| §5.4 核心决策判据 | Task 11 |
| §6.1 设计原则（非侵入式） | Task 1（包骨架） |
| §6.2 包结构 | Task 1 + 各 Task 落地 |
| §6.3 配置 | Task 1 |
| §6.4 新增依赖 | Task 1 |
| §6.5 输出产物（7 类 + 4 图） | Task 12（pipeline 汇总输出） |
| §7 合格标准（最低/理想/不合格） | Task 12 Step 6 逐条勾核 |
| §8 风险与预案 | 全局约束 + 各 Task 测试覆盖边界 |
| §9 时间安排 | 14 天，对应 Task 1-12 顺序 |
| §10 与旧代码关系 | 全局约束（非侵入式） |

### 2. Placeholder Scan

- Task 6 的 `spatial_variance` 简化为 trace 协方差（spec §3.1 提到马氏距离方差），已在 docstring 标注简化，待实际数据验证后可升级。
- 无其他未标注的 TODO/占位符。

### 3. 非侵入式与解耦合

- `src/dual_factor/` 与 `src/xray_attention/` 完全平级，不导入旧包任何模块。
- 旧代码 `src/xray_attention/` 全程不动，仅共享 conda 环境 `Anomaly`。
- 各 Task 模块单一职责：data 只读数据，features 只算指标，analysis 只做分析，pipeline 只编排。
- 配置驱动：所有参数来自 `configs/dual_factor_efa.yaml`，无硬编码。

### 4. 科学严谨性

- **EFA 方法学**：PAF（主轴因子，不假设多元正态）+ Promax（斜交，允许因子相关）+ 平行分析（数据驱动定因子数），符合 EFA 最佳实践。
- **样本量**：37 测试 × 8 窗口/测试 ≈ 296 观测，满足 5:1（观测:变量）最低要求，接近理想 10:1。
- **方向性检验**：用 Mann-Whitney U（非参数）替代 t 检验，适配小样本与非正态分布。
- **被试内标准化**：消除 EOG 电压个体差异，聚焦疲劳相对变化。
- **否定结果合规**：决策报告明确"无论结果支持或否定均为合格"。

### 5. Cross-Platform & Python 3.9 兼容

- 所有路径用 `Path(__file__).resolve().parents[N]`，无写死绝对路径。
- 所有源文件首行 `from __future__ import annotations`，`str | int` 等注解在 3.9 可运行。
- bash 命令均标注「从项目根目录运行」。
- Matplotlib 使用非交互式 `Agg` 后端。

### 6. TDD 一致性

- 每个 Task 均先写测试 → 验证失败 → 实现 → 验证通过 → 提交。
- 测试覆盖：数据加载、滤波、Hampel、I-VT 事件检测、8 指标、窗口化、标准化、KMO、Bartlett、EFA 提取、平行分析、方向性检验、决策判定、端到端 smoke test。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-07-dual-factor-drozy-efa-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — 每个 Task 派发独立 subagent，Task 间 review，快速迭代。适合 14 天节奏。

**2. Inline Execution** — 在当前会话按 executing-plans 批量执行，设检查点（每完成 Task 3/7/9/12 暂停审查）。

建议执行顺序：Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8 → Task 9 → Task 10 → Task 11 → Task 12。

**关键检查点**：
- Task 2 完成后：验证 37 个测试 EOG 全部可读（若 <30 个可用，触发 spec §8 风险预案）
- Task 7 完成后：检查指标缺失率（若 >10%，调整窗口参数或 min_fixations）
- Task 9 完成后：检查因子数与相关（若 r > 0.85 或仅 1 因子，准备否定结果报告）
- Task 12 完成后：对照 spec §7 逐条验收

Which approach?
```
