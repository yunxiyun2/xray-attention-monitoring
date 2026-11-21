# v6 EFA 执行汇报：4/3 结构修正——SV 加入 F2

> 日期：2026-11-15
> 运行目录：`results/dual_factor_efa_v6/20261115T142308Z-fd3e4386/`
> 历史 v6 规范与计划已移除；本报告保留原始运行结论。
> 前序结果：v5 Decision B（运行目录 `20261109T154643Z-aaffec43`）

## 1. 工作目标

v5 在真实 DROZY 数据上产出 Decision B，根因诊断为 F2 仅 2 个指标（SE, GTE）且相关极高（r≈0.90），导致 PAF 迭代中 GTE 共同度越界（Heywood case），4/2 结构统计不可识别。

SPEC v5 §6.1 Decision B 允许"一份下一版本修正建议"。v6 即为该唯一修正：将 SV（Spatial Variance）加入 F2，形成 4/3 结构（F1={FD,PSV,SA,FN}, F2={SE,GTE,SV}），使 F2 达到最小识别要求（3 指标 > 2 因子）。

## 2. 修正方案

### 2.1 SV 选择依据

v5 真实数据相关矩阵中 SV 与 F2 指标的相关：

| 指标对 | 相关系数 | 说明 |
|---|---|---|
| SV–SE | -0.365 | 中等负相关 |
| SV–GTE | -0.293 | 中等负相关 |
| SV–SDR | 0.003 | 近零（SDR 不加入） |

SV 是唯一与 SE/GTE 有实质关联的候选指标。SV 加入 F2 后预期为负载荷（与 SE/GTE 负相关），C2 检查主载荷绝对值≥0.50。

### 2.2 新候选结构

| 成分 | 指标 | 指标数 | 预期疲劳方向 |
|---|---|---|---|
| F1：驱动衰减假设成分 | FD, PSV, SA, FN | 4 | FD↑，PSV↓，SA↓，FN↓ |
| F2：分布—转移共变成分 | SE, GTE, SV | 3 | SE↑，GTE↑，SV↑（负载荷） |

### 2.3 方向处理

SV 预期方向 +1（疲劳增加空间离散度），但在 F2 上预期负载荷（与 SE/GTE 负相关）。主方向由多数指标（SE, GTE 为正）决定。因子使用中性名称"分布—转移共变成分"。

## 3. 模拟验证

### 3.1 合成数据

7 指标 4/3 结构，SV 在 F2 上负载荷 -0.6：

```
loadings[:4, 0] = [.9, -.9, -.9, -.9]   # F1: FD, PSV, SA, FN
loadings[4:, 1] = [.9, .9, -.6]          # F2: SE, GTE (正), SV (负)
```

### 3.2 模拟结果

| 场景 | 潜因子接受 | 信号检测 | 预期 |
|---|---|---|---|
| null | 0/200 | 0/200 | 拒绝 ✓ |
| single | 0/200 | 0/200 | 拒绝 ✓ |
| **two_4_3** | **200/200** | **200/200** | 接受率≥0.80 ✓ |
| correlated_two | 0/200 | 0/200 | 拒绝 ✓ |
| cross_loading | 0/200 | 200/200 | 潜因子拒绝 ✓ |
| heterogeneous | 0/200 | 139/200 | 潜因子拒绝 ✓ |
| two_nonnormal | 200/200 | 200/200 | 接受 ✓ |

**关键结果**：
- `diagnostic_checks_passed = True`
- `latent_power_validated = True`（AC18 通过）
- `two_4_3` 接受率 100%，远超 0.80 阈值

**与 v5 的对比**：v5 的 `two` 场景（6指标4/2）接受率为 0/200，`latent_power_validated=False`。v6 修正后合成 4/3 数据上 PAF 完全收敛且接受，证明修正方向正确——4/3 结构在统计上可识别。

## 4. 正式执行结果

### 4.1 运行配置

- 输入：matrix 模式，`indicator_matrix.csv`（八指标矩阵，v6 选取 7 项）
- 窗口：30s/10s，三相位（0, 1, 2）
- 被试：14 人，36 记录（排除 7-1）
- 有效窗口：2044/2088（无效率 2.1%）
- 配对被试：10 人
- 模拟证书：`simulation_certificate.json`（code/config hash 匹配）

### 4.2 PAF 模型结果

| 模型 | 状态 | 原因 |
|---|---|---|
| seven_1（7指标1因子） | ok | 单因子收敛 |
| **seven_2（7指标2因子）** | **failed** | **PAF Heywood case: communalities outside [0, 1]** |
| seven_3（7指标3因子） | failed | PAF Heywood case |
| eight_2（8指标2因子） | failed | PAF Heywood case |
| four_2（4指标2因子） | ok | F1 子集收敛 |

### 4.3 关键数据

| 指标 | 值 | 说明 |
|---|---|---|
| SE–GTE 相关 | 0.9186 | 极高共线性（Heywood 根因） |
| SV–SE 相关 | -0.183 | 中等负相关（complete case） |
| SV–GTE 相关 | -0.216 | 中等负相关（complete case） |
| within-F2 abs correlation | 0.899 | F2 内部极高相关 |
| 第二特征值 | 1.809 | >1.0 但不足以通过信号检测 |
| dimensions_ok | true | 平行分析建议 2 因子 |
| C5_pass | false | bootstrap Heywood case |
| observable_signal_detected | false | 可观测信号未检测到 |
| phase_pass | false | seven_2 模型失败 |

### 4.4 配对效度（7 项 Holm）

| 指标 | 均值差（标准化） | 95% CI | p_holm | 预期方向 | 结论 |
|---|---|---|---|---|---|
| FD | +0.438 | [-0.06, 0.91] | 0.902 | ↑ | inconclusive |
| PSV | -0.468 | [-1.06, 0.12] | 0.902 | ↓ | inconclusive |
| SA | -0.078 | [-0.72, 0.53] | 1.000 | ↓ | inconclusive |
| FN | +0.154 | [-0.31, 0.62] | 1.000 | ↓ | inconclusive |
| SE | -0.235 | [-0.64, 0.16] | 1.000 | ↑ | inconclusive |
| GTE | -0.243 | [-0.61, 0.14] | 1.000 | ↑ | inconclusive |
| SV | +0.413 | [-0.09, 0.79] | 0.902 | ↑ | inconclusive |

全部 7 项 inconclusive（Holm 校正后无 p < 0.05）。SV 方向正确（+0.413，预期 +1）但不显著。10 人配对样本量下区间很宽。

## 5. 决策

### 5.1 决策结果

**Decision B**。证据等级 L0。

### 5.2 决策依据

| 门控 | 结果 | 说明 |
|---|---|---|
| latent_power_validated | **true** | 模拟验证通过（v5 为 false，v6 改善） |
| phase_pass | false | seven_2 PAF 失败（Heywood） |
| dimensions_ok | true | 平行分析建议 2 因子 |
| C5_pass | false | bootstrap Heywood case |
| observable_signal | false | 信号检测未通过 |
| method_correction_required | **false** | 方法已修正且验证通过 |

决策公式：`decision = "B" if not ready`（phase_pass=false, C5_pass=false → ready=false）。

### 5.3 与 v5 的关键差异

| 项目 | v5 | v6 |
|---|---|---|
| 结构 | 4/2（6指标） | 4/3（7指标） |
| 模拟验证 | 失败（0/200） | **通过（200/200）** |
| latent_power_validated | false | **true** |
| method_correction_required | true | **false** |
| PAF 结果 | Heywood | Heywood |
| 决策 | B | B |
| 根因诊断 | F2 指标不足（2<3） | **SE-GTE 近共线性（r=0.92）** |

v6 的关键进展是**方法有效性已验证**：模拟证明 4/3 结构下 PAF 可以正确收敛和接受。真实数据上的失败不再是方法问题，而是数据特征问题——SE 与 GTE 的极端共线性（r=0.92）导致即使在 3 指标 F2 中，PAF 迭代仍然将共同方差在 SE/GTE 间过度转移，产生 Heywood case。

### 5.4 根因更新

v5 诊断：F2 仅 2 个指标，统计不可识别。
v6 修正后诊断：F2 已有 3 个指标（可识别），但 **SE-GTE 近共线性（r=0.92）** 仍然导致 PAF 数值不稳定。SV 的中等负相关（-0.18, -0.22）不足以打破 SE-GTE 的近共线性主导。

这意味着：
1. 4/2 → 4/3 的结构修正是正确的（模拟验证证明）。
2. 真实数据的固有特征（SE-GTE r=0.92）超出了指标增补能解决的范围。
3. SE 和 GTE 在 DROZY 数据中可能测量的是几乎相同的构念（网格占据熵与转移熵在粗网格 bins=5 下高度重叠）。

## 6. 后续方向

### 6.1 不再继续试删

SPEC v6 §8 明确规定："v6 是 v5 Decision B 允许的唯一修正建议。如果 v6 也产出 Decision B，不再继续试删或换指标，报告当前指标集的固有限制。"

v6 产出 Decision B，修正建议已用尽。

### 6.2 可报告的科学结论

1. **方法有效性已验证**：4/3 结构在合成数据上 100% 接受率，pipeline 统计行为正确。
2. **真实数据固有限制**：SE-GTE 近共线性（r=0.92）是 DROZY 数据的固有特征，非方法缺陷。
3. **平行分析有效**：三相位一致建议 2 因子，但 PAF 无法在该相关结构上稳定估计。
4. **可观测信号未检测到**：F2 内部高相关由 SE-GTE 共线性驱动，非真因子信号。
5. **配对效度全部 inconclusive**：10 人样本下无显著疲劳方向效应，SV 方向正确但不显著。

### 6.3 可能的后续路径（需新 SPEC）

- **SE/GTE 合并**：如果两者测量同一构念，合并为单一指标，F2 变为 2 指标（SE/GTE 合成 + SV），但需新理论依据。
- **网格粒度调整**：bins=5 可能导致 SE 和 GTE 在粗网格上趋同，需敏感性分析。
- **替代估计方法**：MINRES 或 ULS 对 Heywood case 更鲁棒，但改变了统计框架。
- **独立数据集**：DROZY 的 SE-GTE 共线性可能是数据集特异性，需外部验证。
- **CFA 路径**：当前 EFA 无法稳定估计，CFA 依赖 EFA 结果，暂不可行。

以上路径均需新版本 SPEC 设计，不属于 v6 范围。

## 7. 产物清单

| 文件 | 内容 |
|---|---|
| `facts.json` | 结构化事实表（决策依据） |
| `efa_models.json` | 全部 PAF 模型结果（8/7/4 × 1/2/3） |
| `parallel_analysis.json` | 平行分析（iid + record_centered_shift） |
| `cluster_summary.json` | C5 bootstrap 汇总（Heywood 失败） |
| `observable_covariance_signal.json` | 可观测信号检测 |
| `validity_tests.csv` | 7 项配对效度检验 |
| `subject_structure.json` | 逐人/LOSO 辅助分析 |
| `measurement_diagnostics.json` | 测量诊断（含 SV 相关） |
| `efa_decision_report.md` | 自动生成的决策报告 |
| `simulation_certificate.json` | 模拟验证证书 |
| `figures/` | 载荷热图、平行分析图、配对效度图 |
