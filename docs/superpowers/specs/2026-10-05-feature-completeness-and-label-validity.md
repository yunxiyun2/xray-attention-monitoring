# Phase 2 SPEC：特征体系补全与标签效度验证

> **版本**：v1.0
> **日期**：2026-10-05
> **前序**：Phase 1（多模态特征提取，spec `2026-07-18`）已完成 E1–E6 全部实验，F3 cross-attention AUC=0.929。
> **本阶段定位**：Phase 1 的特征矩阵仅含 22 维（gaze 10 + face 12），缺 AU/瞳孔/头部姿态/gaze_target_coverage；标签效度证据链未落实。本 spec 补全这两条线。

---

## 1. 背景与动机

### 1.1 Phase 1 现状

| 维度 | 现状 | 缺口 |
|------|------|------|
| 特征 | gaze(10) + face(12) = 22 维 | AU、瞳孔、头部姿态、gaze_target_coverage 均缺失 |
| 标签 | `state=alert/sleepy`（目录结构） | 无 KSS/SSS 量表、无 Cohen's κ、无状态诱导协议 |
| 模型 | F3 cross-attn AUC=0.929 | — |
| 可解释性 | SHAP + 排列重要性 | AU 缺失导致面部特征不完整 |

### 1.2 核心问题

1. **特征不完整**：spec §4.2 明确要求 AU04/AU09/AU43/AU45 强度、瞳孔直径、头部姿态 6DoF。Phase 1 仅用 EAR/MAR 几何比近似，未引入 OpenFace AU 强度，面部特征表达力不足。

2. **标签效度断裂**：spec §3.2 要求 4 项效度证据（状态诱导协议 / KSS-SSS / Cohen's κ / 混淆控制），Phase 1 均未落实。标签来自实验目录结构（`state=alert/sleepy`），虽无直接循环依赖（标签不由 gaze 阈值生成），但 gaze 特征与阈值阶段发现的信号同源，需独立效度验证。

3. **gaze_target_coverage 缺失**：spec §4.1 标为"场景特有指标"（hard 任务多目标点覆盖率），是本研究区别于通用疲劳检测的差异化特征。

---

## 2. 目标与研究问题

### 2.1 总目标

引入 OpenFace 全量 AU + 瞳孔 + 头部姿态特征，补全标签效度证据链，探查新特征增量后决定是否重跑 E3–E6。

### 2.2 研究问题

| 编号 | 研究问题 | 对应假设 |
|---|---|---|
| RQ6 | OpenFace AU 特征是否提供独立于 EAR/MAR 的增量信息？ | H6：加入 AU 后 face 单模态 AUC 显著提升 |
| RQ7 | 瞳孔直径与头部姿态是否贡献额外信号？ | H7：消融后 AUC 下降（如贡献显著） |
| RQ8 | gaze_target_coverage 是否为场景特异的有效特征？ | H8：hard 任务下 coverage 显著区分 alert/sleepy |
| RQ9 | 标签效度：独立标注与实验条件标签的一致性如何？ | H9：Cohen's κ ≥ 0.6（标签可用） |

---

## 3. OpenFace 集成

### 3.1 安装与验证

- 编译安装 OpenFace（Ubuntu 22.04，CUDA 可选）
- 用 `data/<subject>/<state>/<difficulty>/training_video.mp4`（80 条视频）逐条提取
- 输出：每视频一个 CSV（逐帧 AU 强度 + 瞳孔 + 头部姿态）

### 3.2 提取特征

**OpenFace 逐帧输出**（共 ~26 列）：

- **AU 强度**（17 个）：`AU01_r` ~ `AU45_r`（连续值 0–5）
  - spec §4.2 重点：AU04（眨眼）、AU09（厌恶）、AU43（闭眼）、AU45（眨眼强度）
- **瞳孔直径**：`pupil_left`、`pupil_right`（像素单位，如分辨率允许）
- **头部姿态 6DoF**：`head_yaw`、`head_pitch`、`head_roll`（旋转）、`head_x`、`head_y`、`head_z`（平移）

### 3.3 窗口级聚合

对 60s 窗口内的逐帧序列，提取以下窗口级统计量：

| 特征类 | 聚合方式 | 列数 |
|--------|---------|------|
| AU 强度 × 17 | mean + std | 34 |
| 瞳孔直径 | mean + std（左/右取均值后统计） | 2 |
| 头部姿态旋转 × 3 | std + range | 6 |
| 头部姿态平移 × 3 | std | 3 |
| **合计** | | **45** |

### 3.4 特征矩阵扩展

Phase 1 特征矩阵：gaze(10) + face(12) = **22 维**
Phase 2 特征矩阵：gaze(10) + face(12) + AU(34) + pupil(2) + head_pose(9) = **67 维**

> 注：face 12 维保留（EAR/MAR/PERCLOS/blink/yawn），AU 为增量。若 AU 与 EAR 高度共线（r > 0.9），在消融中报告。

---

## 4. gaze_target_coverage 实现

### 4.1 定义

spec §4.1 要求的"场景特有指标"：

- **easy 任务**：单目标点，`coverage = 1 - off_target_ratio`（注视落在目标点 10° 视角内的帧比例）
- **hard 任务**：50 帧多目标点序列（`Gaze_hard_centers.npy`），`coverage = hit_targets / total_targets`（注视在每帧对应目标点 10° 视角内计为命中）

### 4.2 实现

从 `all_errors.txt` 和目标点坐标计算：
- easy：已有 `off_screen_rate`，新增 `target_coverage_easy`
- hard：加载 `Gaze_hard_centers.npy`，逐帧匹配，输出 `target_coverage_hard`

---

## 5. 标签效度补全

### 5.1 状态诱导协议文档

撰写 1 份文档说明：
- 受试者招募条件（年龄段、视力、X 光安检经验）
- 状态诱导方法：sleepy 条件如何实现（睡眠剥夺时长 / 时段 / 自评）
- alert 对照条件
- 环境控制（光照、屏幕距离、任务时长）

### 5.2 人工复核标签

请 1 名独立标注者（不参与实验设计）观看 80 条视频的 60s 窗口片段：
- 标注 alert/sleepy 二分类
- 计算 Cohen's κ vs 实验条件标签
- κ ≥ 0.6 → 标签可用；κ < 0.6 → 需讨论

### 5.3 KSS 量表回溯

- 检查原始数据中是否已有 KSS/SSS 记录
- 如有：对齐到窗口级，作为标签协变量
- 如无：在论文 Limitations 说明，并建议后续采集

### 5.4 混淆控制

明确区分「cognitive fatigue」（认知疲劳，任务时长效应）与「drowsiness」（嗜睡，生理驱动）：
- 分析窗口在任务前/后半段的位置 vs 标签
- 如前半段 alert、后半段 sleepy 居多，需控制 window_order 作为协变量

---

## 6. 实验设计（探查优先）

### 6.1 Phase 2a：特征提取与探查

**Task A1**：OpenFace 编译安装 + 80 条视频批量提取
**Task A2**：窗口级聚合 → 扩展特征矩阵（67 维）
**Task A3**：gaze_target_coverage 实现
**Task A4**：标签效度补全（协议文档 + 人工复核 κ）

**Task A5（探查性实验）**：
- E1'：gaze 单模态（含 coverage 新特征）
- E2'：face 单模态（含 AU/瞳孔/头部姿态新特征）
- 对比 Phase 1 vs Phase 2 的 AUC 增量

### 6.2 Phase 2b：条件性重跑（视 A5 结果决定）

**决策规则**：
- 如 E2' AUC 较 Phase 1 提升 ≥ 0.02（或 AU 消融 Δ ≥ 0.02）→ 重跑 E3'–E6'
- 否则 → 在论文中说明 AU 特征在本数据集上增量有限，保留 Phase 1 结论

---

## 7. 评估协议

沿用 Phase 1 的 Nested LOSO + bootstrap CI + Wilcoxon paired + rank-biserial。

新增：
- **共线性诊断**：AU vs EAR 的 Pearson 相关矩阵，r > 0.9 的特征对在消融中报告
- **标签效度报告**：Cohen's κ + KSS 相关性（如有）

---

## 8. 文件结构

```
dataset/features/
├── gaze_features.csv          # Phase 1（不变）
├── face_features.csv          # Phase 1（不变）
├── openface_features.csv      # 【新增】AU + pupil + head_pose 窗口级
├── multimodal_features_v2.csv # 【新增】67 维扩展矩阵
└── splits/loso_folds.csv      # 不变

experiments/
├── openface_extract/          # 【新增】OpenFace 批量提取脚本
├── label_validity/            # 【新增】人工复核 + κ 计算
└── feature_probing/           # 【新增】E1'/E2' 探查实验

results/phase2/
├── openface_raw/              # OpenFace 逐帧 CSV（80 个文件）
├── feature_probing/           # E1'/E2' summary + metrics
└── label_validity/            # κ 报告

docs/
├── superpowers/specs/
│   └── 2026-10-05-feature-completeness-and-label-validity.md  # 本 spec
├── superpowers/plans/
│   └── 2026-10-05-feature-completeness-and-label-validity-plan.md  # 待撰写
└── label_validity/
    ├── state_induction_protocol.md  # 状态诱导协议
    └── annotation_guide.md          # 人工标注指南
```

---

## 9. 验收标准

| # | 标准 | 验证方式 |
|---|------|---------|
| 1 | OpenFace 成功编译，80 条视频全部提取完成 | openface_raw/ 下 80 个 CSV |
| 2 | 扩展特征矩阵 ≥ 60 维，含 AU04/AU09/AU43/AU45 + 瞳孔 + 头部姿态 | multimodal_features_v2.csv 列名检查 |
| 3 | gaze_target_coverage（easy + hard）实现并纳入特征矩阵 | 列名含 target_coverage |
| 4 | E1'/E2' 探查实验完成，输出 AUC 对比表 | feature_probing/summary.json |
| 5 | 共线性诊断报告：AU vs EAR 相关矩阵 | feature_probing/correlation_matrix.png |
| 6 | 状态诱导协议文档完成 | label_validity/state_induction_protocol.md |
| 7 | 人工复核标签完成，Cohen's κ ≥ 0.6 或在 Limitations 说明 | label_validity/kappa_report.json |
| 8 | 所有路径相对项目根解析 | 代码审查 |
| 9 | Phase 2a 结论明确：是否进入 Phase 2b（重跑 E3'–E6'） | feature_probing/decision.json |

---

## 10. 关键参考文献

- Phase 1 spec §14 全部 9 篇继续适用
- **OpenFace**：Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2018). Multimodal Machine Learning Monetising OpenFace. ACM Multimedia.
- **KSS**：Shahid, A. et al. (2012). Karolinska Sleepiness Scale (KSS). Sleep, 35(5), 737-746.
- **Cohen's κ**：Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37-46.
