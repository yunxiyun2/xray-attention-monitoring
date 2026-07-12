# 有效关注阈值实验与研究仓库实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可复现研究仓库，并完成基于嵌套 LOSO 的有效关注候选阈值选择实验。

**Architecture:** 使用 `src` 布局的 Python 包读取 `Distan_error（原20）` 中的 80 段误差序列。核心库分别负责数据发现、阈值特征计算和按受试者隔离的嵌套 LOSO 评价；实验入口只加载配置、编排计算和写出表格/图表。

**Tech Stack:** Python 3.11、Conda、NumPy、Pandas、SciPy、scikit-learn、Matplotlib、Pytest；PyTorch 预留给后续模型阶段，不用于本实验。

## Global Constraints

- 原始 `data/` 与 `Distan_error（原20）/` 只读，不复制、不重写、不加入 Git。
- 标签约定固定为 `alert=1`（有效关注）和 `sleepy=0`（无效/不足关注）。
- 所有训练/测试切分均以受试者为单位；测试受试者不得参与候选阈值或判别临界值选择。
- 候选阈值为 `50` 到 `1000` 像素，含端点，步长 `25` 像素。
- 本阶段连续片段长度的单位为采样点，不换算为秒。
- 每次实验输出配置快照、逐任务特征、候选阈值汇总、每折结果和论文图表。

---

### Task 1: 建立研究仓库与可复现实验环境

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `environment.yml`
- Create: `pyproject.toml`
- Create: `configs/threshold_selection.yaml`
- Create: `src/xray_attention/__init__.py`
- Create: `src/xray_attention/data/__init__.py`
- Create: `src/xray_attention/attention/__init__.py`
- Create: `src/xray_attention/models/.gitkeep`
- Create: `experiments/threshold_selection/.gitkeep`
- Create: `manuscript/.gitkeep`
- Create: `results/threshold_selection/.gitkeep`
- Create: `tests/.gitkeep`

**Interfaces:**
- Consumes: 项目根目录中已有的 `Distan_error（原20）/` 原始误差序列。
- Produces: 可由 `conda env create -f environment.yml` 建立的 `xray-attention` 环境，以及阈值实验的唯一参数来源 `configs/threshold_selection.yaml`。

- [ ] **Step 1: 创建 Conda 环境描述和包配置**

```yaml
# environment.yml
name: xray-attention
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy>=1.26
  - pandas>=2.2
  - scipy>=1.13
  - scikit-learn>=1.5
  - matplotlib>=3.9
  - pyyaml>=6.0
  - pytest>=8.0
  - pip
```

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "xray-attention"
version = "0.1.0"
description = "Attention-state analysis for X-ray security inspection"
requires-python = ">=3.11"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: 写入实验配置**

```yaml
# configs/threshold_selection.yaml
data_root: "Distan_error（原20）"
output_dir: "results/threshold_selection"
thresholds_px: {start: 50, stop: 1000, step: 25}
labels: {alert: 1, sleepy: 0}
task_difficulties: [easy, hard]
random_seed: 20260712
```

- [ ] **Step 3: 写入 `.gitignore` 与 README**

`.gitignore` 必须忽略 `.DS_Store`、`.pytest_cache/`、`__pycache__/`、`.venv/`、`data/`、`Distan_error（原20）/`、`*.mp4`、`*.npy`、`*.pt`、`*.ckpt`，同时保留 `results/` 中的 `.gitkeep` 和可追踪的汇总产物。

README 必须说明研究问题、数据不入库原则、环境建立命令、阈值实验命令、输出文件说明及 `alert/sleepy` 标签约定。

- [ ] **Step 4: 建立目录和空模块，并验证打包配置**

Run: `conda env create -f environment.yml`

Expected: 创建名为 `xray-attention` 的环境，包含 NumPy、Pandas、SciPy、scikit-learn、Matplotlib、PyYAML 和 Pytest。

Run: `conda run -n xray-attention python -c "import numpy, pandas, scipy, sklearn, matplotlib, yaml; print('environment ready')"`

Expected: 输出 `environment ready`。

Task 1 不运行 Pytest；测试将在 Task 2 添加首批测试后开始执行。

### Task 2: 实现误差序列读取与关注指标

**Files:**
- Create: `src/xray_attention/data/records.py`
- Create: `src/xray_attention/attention/metrics.py`
- Create: `tests/test_records.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `data_root: pathlib.Path`、候选任务难度与标签。
- Produces: `TaskRecord(subject_id, difficulty, label_name, label, errors)`；`compute_task_features(record, thresholds)` 返回一行基础统计与每阈值 `inside_ratio_<t>`、`longest_run_<t>` 特征。

- [ ] **Step 1: 写读取失败测试**

```python
from pathlib import Path
import pytest
from xray_attention.data.records import discover_records

def test_discover_records_rejects_missing_sequence(tmp_path: Path) -> None:
    (tmp_path / "easy" / "01" / "alert").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="all_errors.txt"):
        discover_records(tmp_path, {"alert": 1, "sleepy": 0}, ["easy"])
```

- [ ] **Step 2: 写读取成功与数值校验测试**

```python
from pathlib import Path
import numpy as np
from xray_attention.data.records import discover_records

def _write_sequence(root: Path, label: str, values: str) -> None:
    target = root / "easy" / "01" / label
    target.mkdir(parents=True)
    (target / "all_errors.txt").write_text(values, encoding="utf-8")

def test_discover_records_loads_all_labelled_sequences(tmp_path: Path) -> None:
    _write_sequence(tmp_path, "alert", "10\\n20\\n")
    _write_sequence(tmp_path, "sleepy", "30\\n40\\n")
    records = discover_records(tmp_path, {"alert": 1, "sleepy": 0}, ["easy"])
    assert [(item.subject_id, item.label) for item in records] == [("01", 1), ("01", 0)]
    assert np.array_equal(records[0].errors, np.array([10.0, 20.0]))
```

- [ ] **Step 3: 写指标测试并确认其失败**

```python
import numpy as np
from xray_attention.attention.metrics import longest_true_run, summarize_errors

def test_longest_true_run_counts_contiguous_samples() -> None:
    assert longest_true_run(np.array([False, True, True, False, True])) == 2

def test_summarize_errors_computes_threshold_ratio_and_run() -> None:
    summary = summarize_errors(np.array([10.0, 40.0, 80.0, 20.0]), [25.0, 50.0])
    assert summary["inside_ratio_25"] == 0.5
    assert summary["longest_run_50"] == 2
    assert summary["mean_error"] == 37.5
```

Run: `conda run -n xray-attention python -m pytest tests/test_records.py tests/test_metrics.py -q`

Expected: FAIL，因为 `records` 和 `metrics` 模块尚不存在。

- [ ] **Step 4: 实现数据读取**

```python
# src/xray_attention/data/records.py
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class TaskRecord:
    subject_id: str
    difficulty: str
    label_name: str
    label: int
    errors: np.ndarray

def discover_records(data_root: Path, labels: dict[str, int], difficulties: list[str]) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for difficulty in difficulties:
        for subject_dir in sorted((data_root / difficulty).iterdir()):
            if not subject_dir.is_dir():
                continue
            for label_name, label in labels.items():
                sequence_path = subject_dir / label_name / "all_errors.txt"
                if not sequence_path.is_file():
                    raise FileNotFoundError(f"Missing required sequence: {sequence_path}")
                errors = np.loadtxt(sequence_path, dtype=float, ndmin=1)
                if errors.size == 0 or not np.isfinite(errors).all():
                    raise ValueError(f"Invalid error sequence: {sequence_path}")
                records.append(TaskRecord(subject_dir.name, difficulty, label_name, label, errors))
    return records
```

- [ ] **Step 5: 实现指标计算**

```python
# src/xray_attention/attention/metrics.py
import numpy as np

def longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best

def summarize_errors(errors: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    summary = {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors, ddof=1)),
        "n_samples": float(errors.size),
    }
    for threshold in thresholds:
        mask = errors <= threshold
        suffix = str(int(threshold))
        summary[f"inside_ratio_{suffix}"] = float(np.mean(mask))
        summary[f"longest_run_{suffix}"] = float(longest_true_run(mask))
    return summary
```

- [ ] **Step 6: 运行单元测试**

Run: `conda run -n xray-attention python -m pytest tests/test_records.py tests/test_metrics.py -q`

Expected: PASS。

### Task 3: 实现嵌套 LOSO 阈值选择与统计汇总

**Files:**
- Create: `src/xray_attention/attention/evaluation.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: 任务级特征表，列 `subject_id`、`difficulty`、`label`、`inside_ratio_<threshold>`。
- Produces: `evaluate_nested_loso(features, thresholds)` 返回候选汇总表、外层折结果表和推荐阈值。

- [ ] **Step 1: 写受试者隔离与阈值选择测试**

```python
import pandas as pd
from xray_attention.attention.evaluation import evaluate_nested_loso

def test_nested_loso_never_uses_test_subject_to_select_threshold() -> None:
    frame = pd.DataFrame({
        "subject_id": ["01", "01", "02", "02", "03", "03"],
        "difficulty": ["easy"] * 6,
        "label": [1, 0, 1, 0, 1, 0],
        "inside_ratio_50": [0.9, 0.1, 0.8, 0.2, 0.2, 0.8],
        "inside_ratio_100": [0.95, 0.05, 0.9, 0.1, 0.85, 0.15],
    })
    _, folds, _ = evaluate_nested_loso(frame, [50, 100])
    assert set(folds["test_subject"]) == {"01", "02", "03"}
    assert all(folds["train_subject_count"] == 2)
```

- [ ] **Step 2: 确认测试失败**

Run: `conda run -n xray-attention python -m pytest tests/test_evaluation.py -q`

Expected: FAIL，因为评价模块尚不存在。

- [ ] **Step 3: 实现候选评价与嵌套 LOSO**

实现必须包含以下规则：

```python
def choose_threshold(training: pd.DataFrame, thresholds: list[int]) -> tuple[int, float]:
    # 对每个 threshold 的 inside_ratio 计算 ROC-AUC；正向分数为 inside_ratio。
    # 以 AUC 降序、平衡准确率降序、threshold 升序排序，返回最佳阈值。
    # 再在该特征的训练数据上枚举唯一中点，选择训练平衡准确率最高的判别临界值。

def evaluate_nested_loso(features: pd.DataFrame, thresholds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    # 外层按 subject_id 留一；训练集调用 choose_threshold。
    # 测试集只能用选中的 threshold 和训练得到的临界值。
    # 报告每折测试的 balanced_accuracy、roc_auc（折内两类存在时）、selected_threshold、decision_cutoff。
    # 对 overall、easy、hard 分别运行；overall 的训练与测试均包含相应受试者的两种难度任务。
```

使用 `sklearn.metrics.roc_auc_score`、`balanced_accuracy_score` 和 `scipy.stats.mannwhitneyu`。效应量使用 rank-biserial correlation：

```python
rank_biserial = 1 - (2 * u_statistic) / (n_positive * n_negative)
```

- [ ] **Step 4: 运行评价测试**

Run: `conda run -n xray-attention python -m pytest tests/test_evaluation.py -q`

Expected: PASS。

### Task 4: 实验入口、结果产物与论文图表

**Files:**
- Create: `experiments/threshold_selection/run.py`
- Create: `src/xray_attention/attention/plots.py`
- Create: `tests/test_threshold_experiment_smoke.py`
- Create: `docs/experiments/threshold-selection.md`

**Interfaces:**
- Consumes: `configs/threshold_selection.yaml` 和 Task 2/3 的公共接口。
- Produces: `results/threshold_selection/config_used.yaml`、`task_features.csv`、`candidate_summary.csv`、`nested_loso_folds.csv`、`recommended_threshold.json`、`threshold_performance.png`、`selected_thresholds.png`。

- [ ] **Step 1: 写端到端 smoke test**

```python
from pathlib import Path
from experiments.threshold_selection.run import run_experiment

def test_run_experiment_writes_required_outputs(sample_config: Path, tmp_path: Path) -> None:
    run_experiment(sample_config, tmp_path)
    expected = {
        "config_used.yaml", "task_features.csv", "candidate_summary.csv",
        "nested_loso_folds.csv", "recommended_threshold.json",
        "threshold_performance.png", "selected_thresholds.png",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
```

- [ ] **Step 2: 实现实验入口**

`run.py` 必须：读取 YAML；生成包含基本元数据与指标的 80 行 `task_features.csv`；依次运行 overall、easy、hard 的嵌套 LOSO；按设计文档规则决定推荐阈值；写入 UTF-8 CSV 与 JSON；将实际使用配置复制到输出目录。所有输出路径由函数参数或配置控制，不允许写死用户绝对路径。

- [ ] **Step 3: 生成两幅论文图**

`plots.py` 必须使用非交互式 Matplotlib 后端，输出：

- `threshold_performance.png`：横轴为像素阈值，纵轴为候选阈值的 ROC-AUC；分别显示 overall、easy、hard。
- `selected_thresholds.png`：显示 20 个外层折中被选阈值的频数，并分别标示 overall、easy、hard。

图中必须标明坐标单位为 `px`，使用英文文件名，标题和图例用能直接进入论文图表清单的中文或英文术语。

- [ ] **Step 4: 写实验记录和运行 smoke test**

`docs/experiments/threshold-selection.md` 必须记录标签定义、候选区间、嵌套 LOSO 逻辑、产物解释和“连续长度单位为采样点”的限制。

Run: `conda run -n xray-attention python -m pytest -q`

Expected: PASS。

- [ ] **Step 5: 运行真实数据实验与边界检查**

Run: `conda run -n xray-attention python experiments/threshold_selection/run.py --config configs/threshold_selection.yaml`

Expected: `results/threshold_selection/` 含全部七类产物，`task_features.csv` 恰有 80 行。

检查 `candidate_summary.csv`：若 best threshold 是 `50` 或 `1000`，将 `configs/threshold_selection.yaml` 的范围向相应方向扩展 250 px 后完整重跑，并在实验记录中说明扩展原因。

### Task 5: 验收与仓库初始化

**Files:**
- Modify: `README.md`
- Modify: `docs/experiments/threshold-selection.md`
- Modify: `results/threshold_selection/*`

**Interfaces:**
- Consumes: 已通过测试的实验实现和真实数据结果。
- Produces: 可交接的研究仓库与第一轮阈值选择结论。

- [ ] **Step 1: 核对数据完整性与结果一致性**

Run: `conda run -n xray-attention python -c "import pandas as pd; frame = pd.read_csv('results/threshold_selection/task_features.csv'); assert len(frame) == 80; assert frame['subject_id'].nunique() == 20; assert set(frame['label']) == {0, 1}; print('80 records / 20 subjects / both labels verified')"`

Expected: 输出 `80 records / 20 subjects / both labels verified`。

- [ ] **Step 2: 完整测试与输出审查**

Run: `conda run -n xray-attention python -m pytest -q`

Expected: 所有测试通过。

检查 README 的运行命令、实验记录中的推荐阈值、JSON 结果和图表是否一致；检查图中不存在截断坐标、空白画布或路径相关文本。

- [ ] **Step 3: 初始化版本控制并提交（若用户授权）**

Run: `git init && git add . && git commit -m "feat: add attention threshold study"`

Expected: 仅源码、配置、文档、测试和小型结果进入首次提交；原始数据未被暂存。
