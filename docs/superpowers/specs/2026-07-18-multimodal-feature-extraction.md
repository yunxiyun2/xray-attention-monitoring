# 多模态有效关注识别研究：注视与面部行为融合的设计规范

> 版本：2.0（全幅重写）
> 日期：2026-07-18
> 上游依赖：[2026-07-12-attention-threshold-study-design.md](./2026-07-12-attention-threshold-study-design.md)（已完成，推荐阈值 overall=900px / easy=975px / hard=625px）

---

## 1. 研究背景与动机

X 光行李安检是典型的**持续警戒（vigilance）任务**：目标稀少、信号微弱、长时间高负荷。Mackworth 以降的警戒研究证明，警戒任务在 20–40 分钟内即可出现显著检出率下降（Molloy & Parasuraman, 1996）。在安检场景下，「警戒衰退」与「有效关注缺失」直接对应漏检风险。

**领域最相关前作**：
- Langhals, Burgoon & Nunamaker (2012) 在模拟行李筛查任务中以眼动+头部姿态+瞳孔+扫视建模警戒变化，证明**眨眼率与扫视率是显著预测因子**。
- Donnelly et al. (2019) 综述指出眼动是理解安检员搜索失败的核心工具。

**通用疲劳检测 SOTA（2024–2026）**：
- 眼部：PERCLOS 是 NHTSA/Dinges 背书的金标准；2025–2026 年所有 SOTA 论文均采用。
- 面部：Action Unit（AU）动力学被 Kaur (2026) 综述列为主流模态之一；MAR（哈欠）与 EAR 互补。
- 融合：Cross-modal Attention（TMU-Net, Sensors 2025）、Gated + Uncertainty-weighted（TMU-Net）、Transformer 编码器融合（Tsinghua 2026、FatigueNet, Sci. Rep. 2025）已成主流，naïve 拼接已被明确列为研究空白（Kaur 2026）。

**本研究的增量**：
1. 首次在**真实 X 光安检任务**（非驾驶）下融合注视偏差序列与面部行为；
2. 注视特征不是裸 gaze 流，而是**已校准的注视-目标距离误差序列**（阈值阶段已确定有效关注阈值），这是场景特异性指标；
3. 严格 nested LOSO + 统计检验 + 置信区间，明确回答「多模态是否**显著**优于单模态」。

---

## 2. 研究问题与假设

| 编号 | 研究问题 | 对应假设 |
|---|---|---|
| RQ1 | 仅注视偏差特征能否识别有效/失效关注？ | H1：AUC 显著高于 0.5 |
| RQ2 | 仅面部行为特征能否识别？ | H2：AUC 显著高于 0.5 |
| RQ3 | 多模态融合是否**显著**优于单模态？ | H3：融合 AUC > max(gaze, face)，Wilcoxon p<0.05 |
| RQ4 | 任务难度（easy/hard）是否调节识别性能？ | H4：hard 任务下各模态 AUC 下降 |
| RQ5 | 哪类特征贡献最大？ | 通过消融与 SHAP 回答，不预设假设 |

**研究增量前提**：阈值阶段已确定 overall=900px / easy=975px / hard=625px 为有效关注判定阈值，本 spec 在此基础上提取窗口级特征。

---

## 3. 数据来源与标签效度

### 3.1 原始数据

- `data/<subject>/<state>/<difficulty>/training_video.mp4`：20 位受试者视频
- `Distan_error（原20）/<difficulty>/<subject>/<state>/all_errors.txt`：已校准逐采样注视-目标距离误差序列
- 阈值阶段产物：`results/threshold_selection/recommended_threshold.json`

### 3.2 标签来源与效度（新增）

**标签**：`alert=1, sleepy=0`，来自实验目录结构。

**效度论证要求**（必须在论文中说明）：
1. 状态诱导协议：说明是睡眠剥夺、时段对比（上午/深夜）还是自评量表诱导。
2. 主观量表：至少采用 **Karolinska Sleepiness Scale (KSS)** 或 **Stanford Sleepiness Scale (SSS)** 在每次任务前后采集，作为标签协变量。
3. 标注复核：如无客观生理金标准（EEG/PSG），需由一名独立标注者对视频进行二分类复核，报告 **Cohen's κ** 信度；κ<0.6 则标签不可用。
4. 混淆控制：明确「cognitive fatigue」与「drowsiness」的区分依据（Kaur 2026 指出此为常见空白）。

> ⚠️ 若上述任一不可得，必须在论文「Limitations」一节明确标签效度局限，避免过度的因果归因。

### 3.3 数据集划分

严格 **Leave-One-Subject-Out（LOSO）**：20 折，每折留 1 名受试者全部数据作测试，其余 19 名作训练。受试者边界是数据泄漏的最强屏障。

---

## 4. 特征体系

### 4.1 注视特征（gaze_features）

从 `all_errors.txt` 提取。阈值阶段已确定阈值 $T^*$（overall/easy/hard 各一），此处复用。

**帧级（每采样点）**：
- `error_distance`：注视-目标欧氏距离（px）

**窗口级（60s 标准窗口，PERCLOS 对齐；可附加 5s/30s 多尺度对比）**：
- `mean_error`、`median_error`、`std_error`
- `inside_ratio_<T*>`：误差 < $T^*$ 的采样占比（有效关注比例）
- `longest_run_<T*>`：最长连续低误差采样数
- `p95_error`：95 分位数误差（抵抗异常峰值）
- **`gaze_target_coverage`**：hard 任务下覆盖的目标点比例（场景特有，源自 brainstorm 的多目标注视要求）

### 4.2 面部特征（face_features）

使用 **MediaPipe Face Mesh（468 点 3D）** 提取，必要时用 **OpenFace** 补 AU 强度。

**眼部（基于 6 点 EAR）**：
- `left_ear`, `right_ear`, `avg_ear`
- **`perclos_80`**：滑动 60s 窗口内 EAR<阈值的帧占比（**金标准特征**）
- `blink_rate`：眨眼频率（次/分钟）
- `blink_duration_mean`、`blink_duration_p95`：眨眼时长均值/95 分位
- `blink_amplitude_mean`：眨眼幅度
- `long_closure_count`：单次闭合 >1s 的次数（**micro-sleep 候选**）

**口部**：
- **`mar`**：Mouth Aspect Ratio
- `yawn_count_60s`：MAR 超阈持续 >2s 的次数

**头部姿态**（PnP 求解欧拉角）：
- `yaw`, `pitch`, `roll`
- `head_nod_count`：pitch 周期性负峰值次数（点头）
- `head_movement_std`：头部平移幅度标准差

**面部动作单元（AU）**（如 OpenFace 可用）：
- `AU04_r`, `AU09_r`, `AU43_r`, `AU45_r`：皱眉/皱鼻/闭眼/眨眼强度
- 若 OpenFace 不可用，则用 FaceMesh 关键点距离比近似 AU43/AU45。

**瞳孔**（如分辨率允许）：
- `pupil_diameter_mean`、`pupil_diameter_std`（Langhals 2012 已证明相关）

> ❌ 删除原 spec 模糊的 `face_stability`，由上述具体特征替代。

### 4.3 标签

- `label`：alert=1, sleepy=0
- `subject_id`：1–20（LOSO 分层用）
- `difficulty`：easy / hard
- `task_id`：受试者×状态×难度组合键

---

## 5. 时间窗口与样本切分

### 5.1 窗口定义

- **标准窗口：60s**，与 PERCLOS 文献对齐；
- **步长**：训练窗口允许 50% 重叠（增广），**测试窗口必须不重叠**（避免 within-subject 自相关膨胀估计）；
- 多尺度对照：附加 5s / 30s 窗口作敏感性分析（RQ4 调节效应用）。

### 5.2 时间对齐

注视序列（webcam 采样）与视频帧（30fps）时间戳对齐到统一时间轴，按 60s 窗口聚合。**对齐误差 < 1 帧**，记录在对齐日志。

### 5.3 数据泄漏防护（强制）

1. **z-score 标准化参数仅在训练折拟合**，再 apply 到测试折；
2. 特征选择（如 SelectKBest）仅在训练折进行；
3. 阈值 $T^*$ 来自阈值阶段的 LOSO 嵌套选择，已无泄漏；多模态阶段不得重新选阈值；
4. 窗口切分在受试者内进行，跨受试者窗口不得跨折。

---

## 6. 模型与融合策略

### 6.1 候选模型（含小样本适配）

**经典 ML 基线**（必做，作为下界与过拟合检测）：
- Logistic Regression（2026 Sensors 实证 LOSO 下反超 LSTM）
- SVM（RBF）
- Random Forest

**深度模型**（在 ML 基线之上才有意义）：
- MLP（小样本友好，主用）
- LSTM（窗口序列）
- 1D-CNN + LSTM（时空混合，参考 2026 NITYMED）

> ⚠️ Transformer 在 n=20 下易过拟合，仅作为**可选上限实验**，且必须配合预训练 backbone 或强正则化（dropout≥0.5, weight decay）。不作为主模型。

**正则化与增广**：
- Dropout 0.3–0.5、L2 weight decay 1e-4
- 训练窗口重叠增广
- 可选：MixUp / 时序抖动

### 6.2 融合策略（三档对比，RQ3 核心）

| 档位 | 名称 | 方法 | 角色 |
|---|---|---|---|
| F1 | 早期融合 | 特征拼接 → MLP | naïve baseline |
| F2 | 晚期融合 | 各模态独立子模型 → 概率加权（学习权重或 LogReg stacking） | 决策级 |
| F3 | 跨模态注意力融合 | 双流编码器 → Cross-Attention → 分类头 | SOTA 主力 |

F3 实现要点：
- 注视流与面部流各经一个轻量编码器（MLP 或 1D-CNN）得到 embedding；
- Cross-Attention 层让两流 token 互相 attend；
- 输出经 LayerNorm + Linear 分类。
- 不确定性加权（可选）：对各模态子网络预测做 Monte Carlo dropout，按熵加权（参考 TMU-Net）。

**所有融合档共用同一 LOSO 划分与同一特征矩阵**，确保差异只来自融合策略。

---

## 7. 实验设计

### 7.1 实验矩阵

| 实验 | 输入 | 模型 | 目的 |
|---|---|---|---|
| E1 | 注视特征 | LR / SVM / RF / MLP / LSTM | RQ1 注视基线 |
| E2 | 面部特征 | 同上 | RQ2 面部基线 |
| E3 | 注视+面部 | F1 / F2 / F3 | RQ3 融合对比 |
| E4 | 按 difficulty 分层重做 E1–E3 | 同上 | RQ4 难度调节 |
| E5 | 特征消融（逐类剔除） | F3 | RQ5 贡献度 |
| E6 | 仅 best 模型 + 全特征 | — | 报告最终性能 |

### 7.2 超参搜索

- **Nested LOSO**：外层 LOSO 估性能，内层 Leave-One-Subject-In 在 19 名训练受试者上做网格/随机搜索；
- 搜索空间文档化（`configs/hyperparams.yaml`）；
- 内层 scoring：balanced accuracy；
- 严格禁止在外层测试折上选超参。

### 7.3 随机性与复现

- 全局 `random_state=42`，记录于 `config_used.yaml`；
- 每个模型 checkpoint 保存于 `results/<experiment>/checkpoints/`；
- 数据划分脚本可复跑出同一划分。

---

## 8. 评估协议

### 8.1 主要指标

- **ROC-AUC**（主指标）
- **Balanced Accuracy**
- **F1-score**（alert 类与 sleepy 类分别报告）
- **混淆矩阵**

### 8.2 统计检验（必做，回答「显著优于」）

1. **单模态 vs 随机**：Wilcoxon signed-rank 检验 AUC 是否显著 >0.5；
2. **模态间比较**：对 20 折 AUC 配对样本做 **Wilcoxon signed-rank**（E3 vs E1, E3 vs E2），报告 p 值与效应量 rank-biserial r；
3. **95% 置信区间**：AUC 用 bootstrap（B=2000）分层重采样得 95% CI；
4. **效应量**：rank-biserial r（阈值阶段已用，延续），或 Cohen's d。

### 8.3 次要指标

- **Brier score** + **calibration curve**（部署场景概率校准）
- **per-subject AUC 分布**（box plot，观察个体差异）

### 8.4 报告模板

每实验产出：
- `metrics.csv`：每折 AUC/BalAcc/F1
- `summary.json`：mean ± std、95% CI、Wilcoxon p、效应量
- `confusion_matrix.png`、`roc_curve.png`、`calibration_curve.png`
- `per_subject_auc.png`

---

## 9. 可解释性分析

### 9.1 全局解释

- **排列重要性**（Permutation Importance）：对最终模型在测试折上打乱每类特征，记录 AUC 下降；
- **SHAP 值**（TreeSHAP 或 KernelSHAP）：报告 top-10 特征重要性条形图。

### 9.2 局部解释

- 选 2–3 个典型受试者（高 AUC / 低 AUC / 困难样本），出 SHAP force plot；
- 跨模态注意力权重可视化（F3）：出 attention heatmap，验证「困倦时段面部 AU43↑ 与注视误差↑ 同步」的生理合理性（参考 TMU-Net 2025）。

---

## 10. 伦理与合规

- **IRB**：论文必须给出伦理审批编号与机构；若无，需在 Limitations 说明；
- **知情同意**：所有受试者视频采集前签署知情同意书，明确同意用于科研；
- **数据合规**：面部数据属生物特征，存储于加密本地盘，不公开；论文中面部图像展示需打码或取得额外授权；
- **用途边界**：本研究用于「状态识别」辅助提醒，**不得**用于绩效惩罚或人格推断（brainstorm.md 已强调 AU 解释需谨慎）。

---

## 11. 报告规范

论文方法学章节遵循 **TRIPOD-AI**（预测模型报告规范）：
- 标题/摘要标注开发/验证；
- 数据来源、参与者、结局、预测因子逐项说明；
- 模型构建含超参与锁定策略；
- 性能含校准与决策曲线；
- 局限性单列。

---

## 12. 文件结构

```
xray-attention-monitoring/
├── data/                                  # 原始（只读）
├── Distan_error（原20）/                  # 注视误差序列（只读）
├── dataset/                               # 新增：特征数据集
│   ├── features/
│   │   ├── gaze_features.csv             # 受试者×窗口×难度
│   │   ├── face_features.csv
│   │   └── multimodal_features.csv       # 对齐后的融合矩阵
│   └── splits/
│       └── loso_folds.csv                # 20 折划分（可复跑）
├── src/xray_attention/
│   ├── data/
│   │   ├── records.py                    # 已有
│   │   ├── gaze_feature_extractor.py     # 新增
│   │   ├── face_feature_extractor.py     # 新增（MediaPipe/OpenFace）
│   │   └── dataset_builder.py            # 新增：对齐与切分
│   ├── models/
│   │   ├── classical.py                  # LR/SVM/RF 包装
│   │   ├── mlp.py
│   │   ├── lstm.py
│   │   └── cross_attention_fusion.py     # F3
│   ├── evaluation/
│   │   ├── nested_loso.py                # 复用阈值阶段
│   │   └── stats.py                      # Wilcoxon + bootstrap CI + 效应量
│   └── explain/
│       └── shap_analysis.py
├── experiments/
│   ├── build_features.py                 # 特征提取入口
│   ├── gaze_baseline/                    # E1
│   ├── face_baseline/                    # E2
│   ├── fusion_compare/                   # E3（F1/F2/F3）
│   ├── difficulty_stratified/            # E4
│   └── ablation/                         # E5
├── configs/
│   └── multimodal.yaml                   # 唯一参数源
└── results/multimodal/
    └── <experiment>/
        ├── metrics.csv
        ├── summary.json
        ├── config_used.yaml
        └── figures/
```

---

## 13. 验收标准

1. ✅ 特征矩阵含 PERCLOS、MAR、AU、blink duration、long_closure、瞳孔（如可用）；
2. ✅ F1/F2/F3 三档融合全部跑通且共享同一数据划分；
3. ✅ Nested LOSO（外层估性能，内层选超参），无泄漏；
4. ✅ 报告 95% CI、Wilcoxon p、效应量，明确回答「融合是否显著优于单模态」；
5. ✅ 含特征消融与 SHAP/排列重要性图；
6. ✅ 含 calibration curve 与 per-subject AUC 分布；
7. ✅ 论文方法学章节符合 TRIPOD-AI；
8. ✅ 伦理与标签效度章节齐备；
9. ✅ 所有路径相对项目根解析（已落实），跨 Linux/macOS 可复现。

---

## 14. 关键参考文献（论文写作时按目标期刊格式补全）

- Langhals, T., Burgoon, J.K., & Nunamaker, J.F. (2012). Using Eye-Based Psychophysiological Cues to Enhance Screener Vigilance. *HICSS*.
- Donnelly, N., Muhl-Richardson, A., Godwin, H.J., & Cave, K.R. (2019). Using Eye Movements to Understand how Security Screeners Search for Threats in X-ray Baggage. *Vision, 3*(2), 24.
- Dinges, D.F., et al. PERCLOS 系列（NHTSA 资助）。
- Zhang, Y., et al. (2025). TMU-Net: A Transformer-Based Multimodal Framework with Uncertainty Quantification for Driver Fatigue Detection. *Sensors, 25*(17), 5364.
- Zendehbad, S.A., et al. (2025). FatigueNet: A hybrid GNN and Transformer framework for real-time multimodal fatigue detection. *Scientific Reports, 15*, 33781.
- 曾琪, 王树祎, 刘奕 (2026). 基于多模态数据融合分析的疲劳驾驶检测方法. *清华大学学报（自然科学版）*, 66(9), 1873–1880.
- Ajayi, O.O., et al. (2026). A Subject-Independent Temporal Framework for Behavioural Eye-Based Driver Drowsiness Detection. *Sensors, 26*(19), 6193.
- Kaur, S. (2026). Explainable Multimodal Intelligence for Cognitive Fatigue Detection: A Systematic Review. *IJCOPE, 2*(7).
- Sun, W., et al. (2024). Exploration of Eye Fatigue Detection Features and Algorithm Based on Eye-Tracking Signal. *Electronics, 13*(10), 1798.
