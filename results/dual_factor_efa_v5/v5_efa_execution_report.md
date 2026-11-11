# v5 EFA 执行汇报

> **项目**：xray-attention-monitoring 双因子疲劳目光行为研究
> **阶段**：v5 EFA 决策（教师反馈闭环修正版）
> **日期**：2026-11-08 至 2026-11-09
> **数据集**：DROZY（14 名被试，36 个有效测试，512 Hz EOG）
> **规范**：SPEC v5.0（2026-11-08-dual-factor-efa-v5.md）
> **运行目录**：`results/dual_factor_efa_v5/20261109T154643Z-aaffec43`

---

## 一、工作目标

对教师反馈（R01-R20）进行全面方法修正后，在真实 DROZY 数据上执行正式 EFA，产出科学决策报告。

v5 是对 v1-v4 历史版本的全面修正：废止零均值聚合 EFA、实现真正迭代 PAF、引入整人 cluster bootstrap、分离可观测信号检测与潜因子接受门控、建立阶段门控状态机。

**候选结构**（固定历史条件性复验）：
- F1 驱动衰减假设成分：FD, PSV, SA, FN
- F2 分布—转移失序假设成分：SE, GTE

---

## 二、v5 核心方法修正

### 2.1 反馈闭环覆盖

v5 对教师 20 个反馈问题（R01-R20）逐一修正，对应 16 个验收标准（AC01-AC16）。关键修正：

| 反馈 ID | 问题 | v5 修正 |
|---|---|---|
| R01 | 3/3 硬错误、分组口径 | 命名常量、模板均按 4/2 映射 |
| R04 | 聚合 z-score 被试均值接近零 | 废止零均值聚合 EFA；用被试内条件差异和整簇重采样 |
| R05 | 非重叠只按 subject，混合 test | 按记录时间区间选择，先锁相位再过滤 valid |
| R06 | `pa` 被映射到 `principal` | 实现真正迭代 PAF；principal 仅历史对照 |
| R11/R12 | CFA 载荷/SRMR 定义错误、两指标因子弱识别 | 阶段 C 公式核对、识别检查、原模型/修正模型分开 |
| R16 | 报告 prompt 允许夸大/编造 | 版本化事实表驱动报告，去除编造指令 |

### 2.2 核心创新：信号检测与潜因子接受分离

v5 将两个独立问题分离检验：

1. **可观测共变信号检测**：按人 bootstrap 的组内相关系数（ICC）和第二相关特征值，检测是否存在两个可观测共变组。信号阳性不能豁免潜因子验证。
2. **潜因子 A 接受**：完整 PAF + C5 协议（SMC 初始化→迭代→收敛→整簇 bootstrap→7 项 C5 门槛），是独立的严格门控。

**决策门控公式**：

```text
decision = "pending" if diagnostic_only or not validated
           else ("A" if ready and latent_power else "B")
```

其中 `latent_power = validated AND latent_power_validated`。

### 2.3 迭代主轴因子法（PAF）

SPEC §4 定义的严格 PAF 算法：

```
SMC 初始化 h_i = 1 - 1/(R^-1)_ii
→ 用 h 替换 R 对角
→ 特征分解取前 k 个正特征值
→ 未旋转载荷 L = V_k sqrt(D_k)
→ 更新共同度 h_i = sum_j L_ij^2
→ 最大绝对更新 < 1e-6 收敛，最多 500 次
```

**严格失败条件**：非正特征值、Heywood 情况（共同度越界 [0,1]）、不收敛、非正定 Phi。**禁止**通过截断共同度、无限迭代或 nearest-PD 修补凑通过。

---

## 三、模拟验证（合成数据）

### 3.1 验证设计

7 类已知真值场景各 200 数据集，每个 200 次 bootstrap：

| 场景 | 真值结构 | 可观测信号检测 | 潜因子 A 接受 |
|---|---|---|---|
| null | 无因子 | 0/200 | 0/200 |
| single | 单因子 | 0/200 | 0/200 |
| two | 双因子 4/2 | 200/200 | 0/200 |
| correlated_two | 高相关双因子 | 0/200 | 0/200 |
| cross_loading | 交叉载荷 | 200/200 | 0/200 |
| heterogeneous | 异质样本 | 130/200 | 0/200 |
| two_nonnormal | 非正态双因子 | 200/200 | 0/200 |

### 3.2 验证结果

- **诊断校准通过**：零维/单维无误报，强双组正确检测。
- **潜因子 A 接受率全部为 0**：4/2 结构的合成数据上，PAF 在 200×7=1400 次尝试中无一通过。
- `latent_power_validated = False`，`passed = False`。
- 证书文件：`docs/dual_factor/v5_validation_final.json`，code_sha256 和 config_sha256 均匹配。

### 3.3 PAF 失败根因诊断

对合成 "two" 场景逐步追踪 PAF 迭代历史，发现：

- **F2 仅有 2 个指标（SE, GTE）**，导致因子不可识别。
- SE 共同度持续下降，GTE 共同度持续上升至超过 1.0（Heywood case）。
- Seed 0：5000 次迭代后误差 1.54e-05，单调下降但极慢，未达 1e-6。
- Seed 1：第 721 次迭代触发 Heywood case（GTE 共同度 = 1.000158）。
- Seed 2：第 1192 次迭代触发 Heywood case（GTE 共同度 = 1.000042）。

**结论**：这是 4/2 结构的固有限制。SPEC §5 明确预期 "PAF 潜因子估计仍不收敛/不适定"，并禁止通过截断共同度等方式修复。低潜因子识别率本身是方法/测量需要修正的结果。

---

## 四、正式 EFA 执行（真实 DROZY 数据）

### 4.1 运行配置

- **入口**：`python scripts/run_efa_v5.py --validation docs/dual_factor/v5_validation_final.json`
- **模式**：正式运行（非 diagnostic-only），因为模拟证书有效
- **证书有效性**：
  - `code_sha256 = 948beacc...`（匹配）
  - `config_sha256 = 0f26a15d...`（匹配）
  - `diagnostic_checks_passed = True`
  - `latent_power_validated = False`
- **数据**：2088 生成窗，2044 有效窗，14 被试，36 记录
- **无效窗率**：0.021 < 0.10（质量线通过）
- **三相位完整窗**：Phase 0: 706 / Phase 1: 668 / Phase 2: 670

### 4.2 门控结果

| 门控项 | 结果 | 说明 |
|---|---|---|
| **科学决策** | **B** | 候选 4/2 结构未被潜因子分析支持 |
| **证据等级** | **L0** | 设计/实现级，不可升级 |
| `implementation_validated` | True | 模拟证书有效 |
| `latent_power_validated` | False | 模拟预期，4/2 结构固有不可识别 |
| `phase_pass` | False | 所有 six_2 模型 PAF 失败 |
| `dimensions_ok` | True | iid 平行分析三相位均建议 n=2 |
| `C5_pass` | False | bootstrap PAF 失败：Heywood case |
| `observable_signal_detected` | False | F1 组内相关 0.303 < 0.50 |
| `method_correction_required` | True | 需修正潜因子估计/测量策略 |
| `invalid_window_rate` | 0.021 | 质量线通过 |
| `n_valid_nonfinite` | 0 | 无异常值 |

### 4.3 PAF 模型结果

所有六指标和八指标模型均失败，两种失败类型：

| 失败类型 | 出现位置 | 原因 |
|---|---|---|
| Heywood case | 多数模型 | SE-GTE 相关 ~0.89-0.91，迭代中 GTE 共同度漂移超过 1.0 |
| 不收敛 | phase_1/2 six_1, six_3 | 500 次迭代后误差未达 1e-6 |

**真实数据相关矩阵特征**：
- SE-GTE 相关极高（0.887-0.905），远超其他指标对
- F1 内部相关中等（FD-PSV: 0.18-0.25, PSV-SA: 0.61-0.65）
- F1-F2 跨组相关接近零（-0.14 到 0.19）
- 六指标 KMO：Phase 0 = 0.48, Phase 1 = 0.49, Phase 2 = 0.50（部分 < 0.50）

### 4.4 平行分析参照

| 相位 | 六指标 iid 建议因子数 | 六指标中心化建议因子数 |
|---|---|---|
| Phase 0 | 2 | 2 |
| Phase 1 | 2 | 2 |
| Phase 2 | 2 | 2 |

平行分析正确建议 2 因子（`dimensions_ok = True`），但 PAF 估计失败导致无法利用此结果进行结构验收。无维度冲突（`center_conflict = False`）。

### 4.5 可观测共变信号

| 统计量 | 值 | 阈值 | 判定 |
|---|---|---|---|
| F1 组内绝对相关均值 | 0.303 | >= 0.50 | FAIL |
| F2 组内绝对相关（SE-GTE） | 0.899 | >= 0.50 | PASS |
| 第二相关特征值 | 1.809 | > 1.0 | PASS |
| **同时满足** | — | — | **FAIL** |

F2 有强可观测共变（SE-GTE 相关 0.90），但 F1 组内相关仅 0.30，未达到 0.50 相干阈值。可观测信号检测未通过。即使通过，也不能豁免 PAF/C5 潜因子验证。

### 4.6 整人 Cluster Bootstrap（C5）

```
C5_pass: False
status: not_evaluable
reason: PAF Heywood case: communalities outside [0, 1]
```

参考 PAF 即失败，bootstrap 无法进行。C5 七项门槛全部不适用。

### 4.7 疲劳方向效度（配对检验）

10 名拥有高低 KSS 两状态的被试，整人 bootstrap 2000 次，符号翻转 + Holm 校正：

| 指标 | 标准化Delta | 95% CI | Holm p | 预期方向 | 结论 |
|---|---|---|---|---|---|
| FD | +0.438 | [-0.06, 0.91] | 0.773 | + | inconclusive |
| PSV | -0.468 | [-1.06, 0.12] | 0.869 | - | inconclusive |
| SA | -0.078 | [-0.72, 0.53] | 1.000 | - | inconclusive |
| FN | +0.154 | [-0.31, 0.62] | 1.000 | - | inconclusive |
| SE | -0.235 | [-0.64, 0.16] | 1.000 | + | inconclusive |
| GTE | -0.243 | [-0.61, 0.14] | 1.000 | + | inconclusive |

全部 inconclusive。14 人规模下区间很宽，FD 和 PSV 方向正确但未达显著性。

---

## 五、交付物清单

### 5.1 源代码（v5 实现）

| 文件 | 说明 |
|---|---|
| `src/dual_factor/v5_pipeline.py` | v5 主 pipeline，决策门控逻辑 |
| `src/dual_factor/analysis/paf.py` | 迭代 PAF 核心实现 |
| `src/dual_factor/analysis/cluster_bootstrap.py` | 整人 cluster bootstrap + C5 判定 |
| `src/dual_factor/analysis/covariance_signal.py` | 可观测信号检测模块 |
| `src/dual_factor/analysis/sampling.py` | 真非重叠相位采样 + 等权标准化 |
| `src/dual_factor/analysis/parallel_analysis_v5.py` | 平行分析（iid + 记录中心化循环位移） |
| `src/dual_factor/analysis/paired_validity.py` | 配对方向效度检验 |
| `src/dual_factor/config.py` | v5 严格配置验证 |
| `src/dual_factor/protocol.py` | 协议常量、指标分组 |
| `src/dual_factor/run_manifest.py` | 运行目录管理 + code_hash |
| `src/dual_factor/data/matrix_io.py` | 矩阵输入审计 |
| `scripts/run_efa_v5.py` | 正式运行入口 |
| `scripts/validate_v5_simulation.py` | 模拟验证脚本 |

### 5.2 运行结果（运行目录 20261109T154643Z-aaffec43）

| 文件 | 说明 |
|---|---|
| `efa_decision_report.md` | 自动生成的 v5 决策报告 |
| `facts.json` / `decision.json` | 结构化决策事实 |
| `manifest.json` | 运行清单（哈希、版本、阶段状态） |
| `data_audit.json` | 数据审计 |
| `efa_models.json` | 全部模型结果（8/6 指标 × 1/2/3 因子 × 3 相位） |
| `parallel_analysis.json` | 平行分析参照 |
| `cluster_summary.json` | C5 bootstrap 汇总 |
| `observable_covariance_signal.json` | 可观测信号检测结果 |
| `validity_tests.csv` | 配对效度检验 |
| `subject_structure.json` | 逐被试 + LOSO 辅助分析 |
| `figures/` | 载荷热图、平行分析图、配对效度图 |

### 5.3 验证与文档

| 文件 | 说明 |
|---|---|
| `docs/dual_factor/v5_validation_final.json` | 模拟验证证书（有效） |
| `docs/dual_factor/v5_validation_results.md` | 模拟验证结果汇总 |
| `docs/dual_factor/v5_execution_log.md` | 实施记录（已更新） |
| `docs/superpowers/specs/2026-11-08-dual-factor-efa-v5.md` | SPEC v5.0 |
| `docs/superpowers/plans/2026-11-08-dual-factor-efa-v5-plan.md` | Plan v5.0 |

---

## 六、测试验证

| 测试范围 | 结果 |
|---|---|
| `pytest tests/dual_factor -q` | 112 通过，8 跳过（原始数据缺失） |
| 模拟验证（7 场景 × 200 数据集 × 200 bootstrap） | 诊断校准通过，潜因子接受率 0 |
| 证书哈希校验 | code_sha256 + config_sha256 均匹配 |
| Dry-run 验证 | 配置、数据、证书均通过 |

---

## 七、关键结论

### 7.1 决策 B 的含义

决策 B 表示：候选 4/2 结构在当前方法下未被潜因子分析支持，需要方法或测量修正。这不是"结构不存在"，而是"当前 PAF 方法在 4/2 结构上不适定"。

### 7.2 失败根因

**直接原因**：F2 仅 2 个指标（SE, GTE），且两者相关极高（~0.90），导致 PAF 迭代中共同度不对称漂移至 Heywood case。

**深层原因**：4/2 结构中 F2 的指标数（2）等于因子数约束下限，统计不可识别。这不是代码 bug，而是结构固有限制。

**SPEC 预期**：SPEC §5 明确写道："固定 4/2 的随机样本可能具有清晰可观测共变信号，而 PAF 潜因子估计仍不收敛/不适定"，"低潜因子识别率本身是方法/测量需要修正的结果"。

### 7.3 平行分析的正向发现

尽管 PAF 失败，平行分析在三个相位上一致建议 2 因子（iid 和记录中心化均如此），且无维度冲突。这表明数据中确实存在两维度的共变结构，只是当前 PAF 方法无法在 4/2 约束下稳定估计。

### 7.4 可观测信号的负向发现

F1 组内相关仅 0.303，低于 0.50 相干阈值。F2 的 SE-GTE 相关很高（0.90），但单独满足不能代表整体信号。这说明 F1 的四个指标虽然被平行分析归为一组，其内部一致性在可观测层面较弱。

---

## 八、后续方向

### 8.1 方法修正（优先级最高）

1. **增加 F2 指标**：当前 SE, GTE 仅 2 个指标导致不可识别。考虑将 SV 或 SDR_KL 纳入 F2，或引入新的分布—转移指标使其达到至少 3 个。
2. **替代提取方法**：探索 MINRES 或 ULS 作为 PAF 的敏感性对照，但 SPEC 要求主方法仍为 PAF。
3. **Bayesian 因子分析**：可通过先验正则化处理弱识别问题，但需独立验证协议。

### 8.2 测量验证

1. **原始 EDF 重提取**：当前 `feature_validation = not_run`。需要在有 EDF 的节点上验证信号质量、事件检测和参数敏感性。
2. **PVT 联合效度**：检查因子结构与反应时间/lapse 的关联。
3. **参数敏感性**：bins 3/7、固定事件数熵、不同速度阈值。

### 8.3 阶段门控状态

| 阶段 | 状态 | 前置条件 |
|---|---|---|
| EFA | executed (决策 B) | 已完成 |
| CFA | not_run | 需 EFA A + 教师验收 |
| 预测 | not_run | 需 EFA A + 教师验收 |
| 外部验证 | not_run | 需 CFA 合格 |
| 不变性 | not_run | 需外部验证 |

**当前状态**：EFA 决策为 B，证据等级 L0。不可自动升级。教师验收尚无来源。所有后续阶段均被门控锁定。
