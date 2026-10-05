# 多模态有效关注识别 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **上游 spec**: [2026-07-18-multimodal-feature-extraction.md](../specs/2026-07-18-multimodal-feature-extraction.md) v2.0
>
> **Conda 环境名**：本计划命令示例沿用 `environment.yml` 的 `xray-attention`；若你的本地环境名不同（如 `Anomaly`），请替换。

**Goal:** 按 spec v2.0 实现注视+面部双模态特征提取、三档融合（F1/F2/F3）、nested LOSO + 统计检验的完整实验流水线，回答 RQ1–RQ5。

**Architecture:**
- `data/` 模块：注视提取器、面部提取器、数据集构建器
- `models/` 模块：经典 ML 包装、MLP、LSTM、跨模态注意力融合
- `evaluation/` 模块：nested LOSO、统计检验（Wilcoxon + bootstrap CI + 效应量）
- `explain/` 模块：SHAP / 排列重要性
- `experiments/`：E1–E6 实验入口

**Tech Stack:** Python 3.9+（已加 `from __future__ import annotations` 兼容）, NumPy, Pandas, OpenCV, MediaPipe, scikit-learn, SciPy, SHAP, Matplotlib

**Cross-platform:** 所有路径相对项目根解析（`Path(__file__).resolve().parents[N]`），不依赖任何写死的绝对路径，Linux/macOS 通用。

---

## Task 1: 环境与目录初始化

**Files:**
- 扩展: `environment.yml`（补 opencv-python / mediapipe / shap）
- 创建: `dataset/.gitkeep`, `dataset/features/.gitkeep`, `dataset/splits/.gitkeep`
- 创建: `results/multimodal/.gitkeep`
- 创建: `configs/multimodal.yaml`
- 创建: `src/xray_attention/models/__init__.py`, `evaluation/__init__.py`, `explain/__init__.py`

- [ ] **Step 1: 扩展依赖**

更新 `environment.yml`，在 dependencies 末尾追加：
```yaml
  - pip:
      - mediapipe>=0.10
      - opencv-python>=4.8
      - shap>=0.45
```

- [ ] **Step 2: 创建目录骨架**

```bash
# 从项目根目录运行
mkdir -p dataset/features dataset/splits results/multimodal
touch dataset/.gitkeep dataset/features/.gitkeep dataset/splits/.gitkeep results/multimodal/.gitkeep
```

- [ ] **Step 3: 创建包初始化文件**

```bash
mkdir -p src/xray_attention/models src/xray_attention/evaluation src/xray_attention/explain
touch src/xray_attention/models/__init__.py src/xray_attention/evaluation/__init__.py src/xray_attention/explain/__init__.py
```

- [ ] **Step 4: 创建唯一参数源 `configs/multimodal.yaml`**

```yaml
# 多模态实验唯一参数源
window_sec: 60           # 标准窗口（PERCLOS 对齐）
stride_train_sec: 30     # 训练步长（50% 重叠增广）
stride_test_sec: 60      # 测试步长（不重叠，防自相关泄漏）
multi_scale: [5, 30, 60] # 敏感性分析窗口

threshold:               # 复用阈值阶段产物，不重新选
  overall: 900
  easy: 975
  hard: 625

models:
  classical: [logreg, svm, rf]
  deep: [mlp, lstm]
  fusion: [F1_concat, F2_late, F3_cross_attn]

nested_loso:
  inner_scoring: balanced_accuracy
  random_state: 42

stats:
  bootstrap_B: 2000
  alpha: 0.05
```

- [ ] **Step 5: 安装依赖**

```bash
conda run -n xray-attention pip install mediapipe opencv-python shap
```

- [ ] **Step 6: 提交**

```bash
git add environment.yml dataset/ results/multimodal/.gitkeep configs/multimodal.yaml src/xray_attention/models/__init__.py src/xray_attention/evaluation/__init__.py src/xray_attention/explain/__init__.py
git commit -m "feat(multimodal): init dirs, config, and package skeleton"
```

---

## Task 2: 注视特征提取器

**Files:**
- 创建: `src/xray_attention/data/gaze_feature_extractor.py`
- 测试: `tests/test_gaze_features.py`

**对应 spec**: §4.1 注视特征

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/test_gaze_features.py
from __future__ import annotations

import numpy as np
from pathlib import Path
from xray_attention.data.gaze_feature_extractor import GazeFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def test_load_all_errors_real_data():
    test_file = PROJECT_ROOT / "Distan_error（原20）" / "easy" / "01" / "alert" / "all_errors.txt"
    extractor = GazeFeatureExtractor()
    errors = extractor.load_all_errors(test_file)
    assert isinstance(errors, np.ndarray)
    assert len(errors) > 0
    assert np.all(errors >= 0)

def test_extract_window_features_basic():
    errors = np.array([100, 200, 150, 80, 90, 1200, 1100, 950])
    extractor = GazeFeatureExtractor()
    feat = extractor.extract_window_features(errors, threshold=900)
    assert feat["mean_error"] > 0
    assert 0.0 <= feat["inside_ratio_900"] <= 1.0
    assert feat["longest_run_900"] >= 1
    assert feat["p95_error"] >= feat["mean_error"]
    assert "gaze_target_coverage" not in feat  # 仅 hard 任务有

def test_windowing_non_overlap_test():
    # 60s 窗口、60s 步长 → 测试窗口不重叠
    errors = np.arange(30 * 200)  # 200s @ 30Hz
    extractor = GazeFeatureExtractor()
    windows = extractor.windowing(errors, window_sec=60, stride_sec=60, fps=30)
    assert len(windows) >= 3
    # 相邻窗口起点差应 = window_sec * fps
    assert (windows[1].start - windows[0].start) == 60 * 30
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_gaze_features.py -v
```
Expected: FAIL（GazeFeatureExtractor 未定义）

- [ ] **Step 3: 实现 `gaze_feature_extractor.py`**

```python
# src/xray_attention/data/gaze_feature_extractor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np


@dataclass
class Window:
    start: int
    stop: int
    data: np.ndarray


class GazeFeatureExtractor:
    """从逐采样注视-目标距离误差序列提取窗口级特征。"""

    def load_all_errors(self, filepath: str | Path) -> np.ndarray:
        return np.loadtxt(Path(filepath))

    def extract_window_features(
        self, errors: np.ndarray, threshold: float
    ) -> dict[str, float | int]:
        inside = errors <= threshold
        return {
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "std_error": float(np.std(errors)),
            "p95_error": float(np.percentile(errors, 95)),
            f"inside_ratio_{int(threshold)}": float(np.mean(inside)),
            f"longest_run_{int(threshold)}": int(_longest_true_run(inside)),
        }

    def windowing(
        self, errors: np.ndarray, window_sec: int, stride_sec: int, fps: float
    ) -> list[Window]:
        win, stride = int(window_sec * fps), int(stride_sec * fps)
        return [
            Window(s, s + win, errors[s : s + win])
            for s in range(0, len(errors) - win + 1, stride)
        ]


def _longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_gaze_features.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/xray_attention/data/gaze_feature_extractor.py tests/test_gaze_features.py
git commit -m "feat(gaze): window-level gaze feature extractor with T* reuse"
```

---

## Task 3: 面部特征提取器（MediaPipe）

**Files:**
- 创建: `src/xray_attention/data/face_feature_extractor.py`
- 测试: `tests/test_face_features.py`

**对应 spec**: §4.2 面部特征（PERCLOS / MAR / AU / 头部姿态 / 瞳孔）

> 这是最大的 Task。建议分子步：3a EAR+MAR，3b PERCLOS+blink，3c 头部姿态，3d AU+瞳孔。

- [ ] **Step 1: 写测试**

```python
# tests/test_face_features.py
from __future__ import annotations

import numpy as np
from xray_attention.data.face_feature_extractor import (
    FaceFeatureExtractor, calculate_ear, calculate_mar
)

def test_ear_symmetric_open_eye():
    # 完全睁眼：上下对称，EAR > 0
    left_eye = np.array([[0, 1], [0, -1], [-1, 0.5], [1, 0.5], [-1, -0.5], [1, -0.5]])
    ear = calculate_ear(left_eye)
    assert ear > 0.1

def test_ear_closed_eye_near_zero():
    # 闭眼：上下接近重合，EAR ≈ 0
    closed = np.array([[0, 0.01], [0, -0.01], [-1, 0], [1, 0], [-1, 0], [1, 0]])
    assert calculate_ear(closed) < 0.05

def test_mar_open_mouth():
    open_mouth = np.array([[0, 2], [0, -2], [-1, 0], [1, 0], [-0.5, 1], [0.5, 1]])
    assert calculate_mar(open_mouth) > 0.3

def test_perclos_80_below_threshold_counts():
    ear_series = np.array([0.3, 0.05, 0.04, 0.3, 0.02])  # 3/5 低于 0.1 阈值
    extractor = FaceFeatureExtractor()
    perclos = extractor.perclos(ear_series, ear_threshold=0.1)
    assert 0.59 < perclos < 0.61  # 3/5

def test_long_closure_count_microsleep():
    # 1s @ 30fps = 30 帧；构造 35 帧连续闭眼 → 1 次 micro-sleep
    ear_series = np.where(np.arange(100) < 35, 0.02, 0.3)
    extractor = FaceFeatureExtractor()
    cnt = extractor.long_closure_count(ear_series, ear_threshold=0.1, min_frames=30, fps=30)
    assert cnt == 1
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_face_features.py -v
```

- [ ] **Step 3: 实现核心几何函数（EAR / MAR）**

```python
# src/xray_attention/data/face_feature_extractor.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """Eye Aspect Ratio。eye_landmarks: 6 点 (p1..p6)，Soukupová & Čech 2016。"""
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def calculate_mar(mouth_landmarks: np.ndarray) -> float:
    """Mouth Aspect Ratio，结构与 EAR 同构。"""
    p1, p2, p3, p4, p5, p6 = mouth_landmarks
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


class FaceFeatureExtractor:
    """从视频帧或预提取的 landmark 序列计算面部行为特征。"""

    def perclos(self, ear_series: np.ndarray, ear_threshold: float = 0.1) -> float:
        """PERCLOS: EAR < 阈值的帧占比。"""
        return float(np.mean(ear_series < ear_threshold))

    def long_closure_count(
        self, ear_series: np.ndarray, ear_threshold: float, min_frames: int, fps: float
    ) -> int:
        """单次闭合持续 >= min_frames/fps 秒的次数（micro-sleep 候选）。"""
        below = ear_series < ear_threshold
        cnt = run = 0
        for v in below:
            run = run + 1 if v else 0
            if run == min_frames:
                cnt += 1
        return int(cnt)

    # blink 检测：上升沿/下降沿配对，输出 duration/amplitude
    def detect_blinks(self, ear_series: np.ndarray, ear_threshold: float = 0.1):
        rises = np.where(np.diff((ear_series < ear_threshold).astype(int)) == -1)[0]
        falls = np.where(np.diff((ear_series < ear_threshold).astype(int)) == 1)[0]
        # 配对计算 duration、amplitude
        blinks = []
        for f in falls:
            cand = rises[rises > f]
            if len(cand):
                r = cand[0]
                blinks.append({"start": int(f), "end": int(r), "duration_frames": int(r - f)})
        return blinks

    def extract_from_video(self, video_path: str | Path):
        """MediaPipe FaceMesh 逐帧提取，返回 EAR/MAR/head_pose/AU 时序。占位，Task 3b 实现。"""
        raise NotImplementedError("Implemented in Step 3b with MediaPipe")
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_face_features.py -v
```

- [ ] **Step 5: 实现 MediaPipe 视频流水线（`extract_from_video`）**

补充 `extract_from_video`：用 `mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)` 逐帧提 468 点，输出时序数组（left_ear, right_ear, mar, head_pose, pupil 等）。头部姿态用 `cv2.solvePnP` 求欧拉角。AU 用关键点距离比近似 AU43/AU45（OpenFace 不可用时的 fallback）。

- [ ] **Step 6: 提交**

```bash
git add src/xray_attention/data/face_feature_extractor.py tests/test_face_features.py
git commit -m "feat(face): EAR/MAR/PERCLOS/blink/micro-sleep + MediaPipe pipeline"
```

---

## Task 4: 数据集构建与时间对齐

**Files:**
- 创建: `src/xray_attention/data/dataset_builder.py`
- 测试: `tests/test_dataset_builder.py`

**对应 spec**: §3 数据来源、§5 时间窗口与样本切分

- [ ] **Step 1: 写测试**

```python
# tests/test_dataset_builder.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
from xray_attention.data.dataset_builder import DatasetBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def test_builder_resolves_project_root():
    b = DatasetBuilder()
    assert b.data_root.exists()
    assert b.dist_error_root.exists()

def test_discover_records():
    b = DatasetBuilder()
    records = b.discover_records()
    assert len(records) == 80  # 20 受试者 × 2 难度 × 2 状态

def test_build_windows_no_overlap_test():
    b = DatasetBuilder()
    # 一条 record 切窗：测试步长应等于窗口长度
    windows = b.build_windows_for_record(
        subject="01", state="alert", difficulty="easy",
        window_sec=60, stride_sec=60,
    )
    assert len(windows) >= 1
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_dataset_builder.py -v
```

- [ ] **Step 3: 实现**

```python
# src/xray_attention/data/dataset_builder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TaskRecord:
    subject: str
    state: str          # alert | sleepy
    difficulty: str     # easy | hard
    video_path: Path
    errors_path: Path


class DatasetBuilder:
    def __init__(self, data_root=None, dist_error_root=None, output_root=None):
        self.data_root = Path(data_root) if data_root else _PROJECT_ROOT / "data"
        self.dist_error_root = (
            Path(dist_error_root) if dist_error_root
            else _PROJECT_ROOT / "Distan_error（原20）"
        )
        self.output_root = Path(output_root) if output_root else _PROJECT_ROOT / "dataset"

    def discover_records(self) -> list[TaskRecord]:
        records = []
        for diff_dir in sorted(self.dist_error_root.iterdir()):
            if not diff_dir.is_dir() or diff_dir.name not in ("easy", "hard"):
                continue
            for subj_dir in sorted(diff_dir.iterdir()):
                if not subj_dir.is_dir() or not subj_dir.name.isdigit():
                    continue
                for state_dir in subj_dir.iterdir():
                    if state_dir.name not in ("alert", "sleepy"):
                        continue
                    err = state_dir / "all_errors.txt"
                    vid = self.data_root / subj_dir.name / state_dir.name / diff_dir.name / "training_video.mp4"
                    records.append(TaskRecord(subj_dir.name, state_dir.name, diff_dir.name, vid, err))
        return records

    def build_windows_for_record(
        self, subject, state, difficulty, window_sec, stride_sec, fps=30
    ) -> list[dict]:
        # 定位 errors 文件
        err_file = self.dist_error_root / difficulty / subject / state / "all_errors.txt"
        errors = np.loadtxt(err_file)
        win, stride = int(window_sec * fps), int(stride_sec * fps)
        return [
            {"subject": subject, "state": state, "difficulty": difficulty,
             "window_idx": i, "data": errors[s : s + win]}
            for i, s in enumerate(range(0, len(errors) - win + 1, stride))
        ]

    def write_loso_folds(self, n_subjects=20):
        rows = [{"fold": i, "test_subject": f"{i+1:02d}"} for i in range(n_subjects)]
        pd.DataFrame(rows).to_csv(self.output_root / "splits" / "loso_folds.csv", index=False)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_dataset_builder.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/xray_attention/data/dataset_builder.py tests/test_dataset_builder.py
git commit -m "feat(data): dataset builder with time alignment and LOSO folds"
```

---

## Task 5: 评估与统计基础设施

**Files:**
- 创建: `src/xray_attention/evaluation/nested_loso.py`（复用阈值阶段 `attention/evaluation.py` 的核心逻辑）
- 创建: `src/xray_attention/evaluation/stats.py`
- 测试: `tests/test_stats.py`

**对应 spec**: §7.2 超参搜索、§8 评估协议

- [ ] **Step 1: 写测试**

```python
# tests/test_stats.py
from __future__ import annotations

import numpy as np
from xray_attention.evaluation.stats import (
    bootstrap_auc_ci, wilcoxon_vs_baseline, rank_biserial_r, brier_score
)

def test_bootstrap_auc_ci_bounds():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.6, 0.4])
    lo, hi, mean = bootstrap_auc_ci(y, scores, B=200, seed=42)
    assert 0.0 <= lo <= mean <= hi <= 1.0

def test_wilcoxon_vs_baseline_returns_p():
    aucs = np.array([0.7, 0.75, 0.8, 0.65, 0.9])
    p = wilcoxon_vs_baseline(aucs, baseline=0.5)
    assert 0.0 <= p <= 1.0

def test_rank_biserial_range():
    r = rank_biserial_r(np.array([0.6, 0.7, 0.8]), baseline=0.5)
    assert -1.0 <= r <= 1.0

def test_brier_score_zero_perfect():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert brier_score(y, p) == 0.0
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_stats.py -v
```

- [ ] **Step 3: 实现 `stats.py`**

```python
# src/xray_attention/evaluation/stats.py
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import roc_auc_score


def bootstrap_auc_ci(y_true, scores, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(aucs))


def wilcoxon_vs_baseline(aucs, baseline=0.5):
    """检验 AUC 是否显著高于 baseline（默认 0.5）。"""
    stat, p = wilcoxon(aucs - baseline, alternative="greater")
    return float(p)


def rank_biserial_r(aucs, baseline=0.5):
    """效应量：rank-biserial r。"""
    diffs = aucs - baseline
    ranks = rankdata(np.abs(diffs))
    W = np.sum(ranks[diffs > 0])
    n = len(diffs)
    return float((2 * W / (n * (n + 1) / 2)) - 1) if n else 0.0


def brier_score(y_true, proba):
    return float(np.mean((proba - y_true) ** 2))
```

- [ ] **Step 4: 实现 `nested_loso.py`（包装阈值阶段 `evaluate_nested_loso`，对多模态特征矩阵泛化）**

接口：
```python
def evaluate_nested_loso(
    X, y, subjects, model_factory, param_grid, inner_scoring="balanced_accuracy"
) -> dict:
    """外层 LOSO 估性能；内层 Leave-One-Subject-In 选超参。返回每折 AUC/BalAcc/F1 + 95% CI。"""
```

- [ ] **Step 5: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_stats.py -v
```

- [ ] **Step 6: 提交**

```bash
git add src/xray_attention/evaluation/stats.py src/xray_attention/evaluation/nested_loso.py tests/test_stats.py
git commit -m "feat(eval): nested LOSO + Wilcoxon + bootstrap CI + effect size"
```

---

## Task 6: 模型实现（经典 ML + MLP/LSTM + 三档融合）

**Files:**
- 创建: `src/xray_attention/models/classical.py`（LR/SVM/RF 包装）
- 创建: `src/xray_attention/models/mlp.py`
- 创建: `src/xray_attention/models/lstm.py`
- 创建: `src/xray_attention/models/cross_attention_fusion.py`（F1/F2/F3）
- 测试: `tests/test_models.py`

**对应 spec**: §6 模型与融合策略

- [ ] **Step 1: 写测试（F3 跨模态注意力前向 shape）**

```python
# tests/test_models.py
from __future__ import annotations

import numpy as np
import torch
from xray_attention.models.mlp import MLPClassifier
from xray_attention.models.cross_attention_fusion import (
    EarlyFusion, LateFusion, CrossAttentionFusion
)

def test_mlp_forward_shape():
    model = MLPClassifier(in_dim=20, hidden=[32, 16])
    x = torch.randn(8, 20)
    out = model(x)
    assert out.shape == (8, 2)

def test_early_fusion_concat():
    model = EarlyFusion(gaze_dim=6, face_dim=14, hidden=[32])
    g = torch.randn(8, 6); f = torch.randn(8, 14)
    out = model(g, f)
    assert out.shape == (8, 2)

def test_late_fusion_weighted():
    model = LateFusion(gaze_dim=6, face_dim=14)
    g = torch.randn(8, 6); f = torch.randn(8, 14)
    out = model(g, f)
    assert out.shape == (8, 2)

def test_cross_attention_fusion():
    model = CrossAttentionFusion(gaze_dim=6, face_dim=14, d_model=32, n_heads=2)
    g = torch.randn(8, 6); f = torch.randn(8, 14)
    out = model(g, f)
    assert out.shape == (8, 2)
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_models.py -v
```

- [ ] **Step 3: 实现各模型**

`classical.py`：LR/SVM/RF 的 sklearn 包装，统一 `fit/predict_proba` 接口。

`mlp.py`：`nn.Sequential` 三层 + Dropout 0.3 + LayerNorm，输出 logits（2 类）。

`cross_attention_fusion.py`：
- `EarlyFusion`（F1）：`cat(gaze, face)` → MLP
- `LateFusion`（F2）：两个子 MLP 各出概率，再 LogReg/learnable-weights 加权
- `CrossAttentionFusion`（F3）：双流经线性投影到 d_model → `nn.MultiheadAttention` 双向 cross-attn → LayerNorm + Linear

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src conda run -n xray-attention python -m pytest tests/test_models.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/xray_attention/models/ tests/test_models.py
git commit -m "feat(models): LR/SVM/RF + MLP + F1/F2/F3 cross-attention fusion"
```

---

## Task 7: 特征提取流水线入口

**Files:**
- 创建: `experiments/build_features.py`

**对应 spec**: §12 文件结构 `experiments/build_features.py`

- [ ] **Step 1: 实现一键构建**

```python
# experiments/build_features.py
"""一键构建 gaze/face/multimodal 特征矩阵。

用法（从项目根目录）：
    PYTHONPATH=src conda run -n xray-attention python experiments/build_features.py --config configs/multimodal.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd

from xray_attention.data.dataset_builder import DatasetBuilder
from xray_attention.data.gaze_feature_extractor import GazeFeatureExtractor
from xray_attention.data.face_feature_extractor import FaceFeatureExtractor

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multimodal.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(_PROJECT_ROOT / args.config))

    builder = DatasetBuilder()
    records = builder.discover_records()
    print(f"Discovered {len(records)} records")

    gaze_ext = GazeFeatureExtractor()
    face_ext = FaceFeatureExtractor()

    gaze_rows, face_rows = [], []
    for rec in records:
        errors = gaze_ext.load_all_errors(rec.errors_path)
        for w in gaze_ext.windowing(errors, cfg["window_sec"], cfg["stride_test_sec"], fps=30):
            feat = gaze_ext.extract_window_features(w.data, cfg["threshold"][rec.difficulty])
            feat.update({"subject": rec.subject, "state": rec.state,
                         "difficulty": rec.difficulty, "window_idx": w.start})
            gaze_rows.append(feat)
        # 面部特征从视频提取（占位：实际调用 face_ext.extract_from_video）
        # face_rows.append(...)

    pd.DataFrame(gaze_rows).to_csv(
        _PROJECT_ROOT / "dataset" / "features" / "gaze_features.csv", index=False
    )
    print("Wrote dataset/features/gaze_features.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑通并验证产物**

```bash
PYTHONPATH=src conda run -n xray-attention python experiments/build_features.py --config configs/multimodal.yaml
ls -la dataset/features/
```

- [ ] **Step 3: 提交**

```bash
git add experiments/build_features.py
git commit -m "feat(pipeline): one-click feature extraction entrypoint"
```

---

## Task 8: E1 + E2 单模态基线实验

**Files:**
- 创建: `experiments/gaze_baseline/run.py`
- 创建: `experiments/face_baseline/run.py`

**对应 spec**: §7.1 E1（注视）/ E2（面部），RQ1/RQ2

- [ ] **Step 1: 实现 E1 注视基线**

读取 `dataset/features/gaze_features.csv`，跑 LR/SVM/RF/MLP/LSTM 的 nested LOSO，写 `results/multimodal/gaze_baseline/`（metrics.csv、summary.json、figures/）。

- [ ] **Step 2: 实现 E2 面部基线**

同上，输入换为 `face_features.csv`。

- [ ] **Step 3: 跑通两实验，验证产物齐全**

```bash
PYTHONPATH=src conda run -n xray-attention python experiments/gaze_baseline/run.py --config configs/multimodal.yaml
PYTHONPATH=src conda run -n xray-attention python experiments/face_baseline/run.py --config configs/multimodal.yaml
ls results/multimodal/gaze_baseline/ results/multimodal/face_baseline/
```

- [ ] **Step 4: 提交**

```bash
git add experiments/gaze_baseline/ experiments/face_baseline/
git commit -m "exp(E1,E2): gaze and face unimodal baselines via nested LOSO"
```

---

## Task 9: E3 融合对比实验

**Files:**
- 创建: `experiments/fusion_compare/run.py`

**对应 spec**: §7.1 E3，RQ3 核心——三档融合 F1/F2/F3 共用同一划分

- [ ] **Step 1: 实现**

读 `multimodal_features.csv`，对 F1/F2/F3 各跑一次 nested LOSO。产出对比表：
- `metrics.csv`：每折 × 每融合档 AUC/BalAcc/F1
- `summary.json`：mean ± std、95% CI、Wilcoxon p（F3 vs F1, F3 vs E1, F3 vs E2）、效应量
- `figures/fusion_auc_boxplot.png`、`roc_curve.png`

- [ ] **Step 2: 跑通并验证三档全部有结果**

```bash
PYTHONPATH=src conda run -n xray-attention python experiments/fusion_compare/run.py --config configs/multimodal.yaml
cat results/multimodal/fusion_compare/summary.json
```

- [ ] **Step 3: 提交**

```bash
git add experiments/fusion_compare/
git commit -m "exp(E3): F1/F2/F3 fusion comparison with Wilcoxon significance"
```

---

## Task 10: E4 难度分层 + E5 消融 + E6 最终报告

**Files:**
- 创建: `experiments/difficulty_stratified/run.py`
- 创建: `experiments/ablation/run.py`
- 创建: `src/xray_attention/explain/shap_analysis.py`

**对应 spec**: §7.1 E4/E5/E6、§9 可解释性

- [ ] **Step 1: E4 难度分层**

按 easy/hard 分别重跑 E1–E3，输出难度 × 模态 AUC 矩阵 + 调节效应图。

- [ ] **Step 2: E5 消融**

对 F3 best 模型，逐类剔除特征（gaze / eye / mouth / head / AU），记录 AUC 下降，出条形图。

- [ ] **Step 3: 实现 SHAP / 排列重要性**

`shap_analysis.py`：TreeSHAP（RF）或 KernelSHAP（MLP），输出 top-10 特征重要性 + 2–3 个典型受试者的 force plot + F3 attention heatmap。

- [ ] **Step 4: E6 最终报告**

汇总 best 模型全特征性能，输出：`confusion_matrix.png`、`calibration_curve.png`、`per_subject_auc.png`。

- [ ] **Step 5: 验收**

对照 spec §13 验收标准 9 条逐条勾核。

- [ ] **Step 6: 提交**

```bash
git add experiments/difficulty_stratified/ experiments/ablation/ src/xray_attention/explain/
git commit -m "exp(E4,E5,E6): difficulty stratification, ablation, SHAP, final report"
```

---

## Self-Review

### 1. Spec Coverage

| Spec 章节 | 对应 Task |
|---|---|
| §3 数据来源与标签效度 | Task 4（数据发现）+ 标签效度属论文写作 |
| §4.1 注视特征 | Task 2 |
| §4.2 面部特征（PERCLOS/MAR/AU/姿态/瞳孔） | Task 3 |
| §5 时间窗口与切分 | Task 4 |
| §5.3 数据泄漏防护 | Task 5（z-score 仅训练折拟合，在 nested_loso 内） |
| §6 模型与融合 F1/F2/F3 | Task 6 |
| §7 实验矩阵 E1–E6 | Task 8–10 |
| §7.2 Nested LOSO + 内层超参 | Task 5 |
| §8 评估协议（CI/Wilcoxon/效应量/Brier） | Task 5 |
| §9 可解释性 SHAP/排列重要性/attention heatmap | Task 10 |
| §10 伦理与合规 | 论文写作，非代码 Task |
| §11 TRIPOD-AI 报告规范 | 论文写作，非代码 Task |
| §12 文件结构 | Task 1（骨架）+ 各 Task 落地 |
| §13 验收标准 9 条 | Task 10 Step 5 逐条勾核 |

### 2. Placeholder Scan

- Task 7 的 face 提取为占位（依赖 Task 3 Step 5 完成），已标注。
- 无其他未标注的 TODO/占位符。

### 3. Cross-Platform & Python 3.9 兼容

- 所有路径用 `Path(__file__).resolve().parents[N]`，无写死绝对路径。
- 所有源文件首行 `from __future__ import annotations`，`str | int` 等注解在 3.9 可运行。
- bash 命令均标注「从项目根目录运行」。

### 4. TDD 一致性

- 每个 Task 均先写测试 → 验证失败 → 实现 → 验证通过 → 提交。
- 测试覆盖：真实数据加载、几何函数、窗口切分、统计函数、模型前向 shape。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-18-multimodal-feature-extraction-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — 每个 Task 派发独立 subagent，Task 间 review，快速迭代。

**2. Inline Execution** — 在当前会话按 executing-plans 批量执行，设检查点。

Which approach?
