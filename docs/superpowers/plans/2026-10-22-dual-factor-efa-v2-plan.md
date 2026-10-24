# 疲劳目光行为双因子模式研究：EFA 决策修正与证据链补全 实施计划

> **上游 spec**: [2026-10-22-dual-factor-efa-v2.md](../specs/2026-10-22-dual-factor-efa-v2.md) v2.0
>
> **Conda 环境名**：`Anomaly`
>
> **前版关系**：v1.0 的 12 个 Task 已全部完成，代码保留复用。本版仅修改/扩展必要模块。

**Goal:** 用两周时间完成导师 P0-1 到 P0-4 四个强制决策点，交付一份判据严格执行、窗口独立性已验证、构念效度已论证的 EFA 决策报告，明确回答"呆滞与发散到底能不能分开"。

**Architecture:**

- 复用 v1.0 全部代码（`src/dual_factor/` 包不动）
- 新增 `src/dual_factor/analysis/sensitivity.py`：窗口非独立性敏感性分析
- 新增 `src/dual_factor/analysis/construct_validity.py`：构念效度论证
- 修改 `src/dual_factor/analysis/efa.py`：新增 2vs3 因子对比
- 修改 `src/dual_factor/analysis/decision.py`：严格判据执行
- 修改 `src/dual_factor/pipeline.py`：集成新增模块

**Non-invasive principle:** v1.0 已实现代码不重写，只追加函数或新增模块。

---

## Global Constraints

- 判据 C1-C5 在分析前预注册，结果出来后不可追溯修改阈值
- 交叉载荷 ≥0.3 时禁止判定"可分离成立"，不得使用"待收敛"等模糊表述
- 单主线：EFA 决策报告；辅线：论文提纲（≤20%）、方向校正（≤10%）
- 负结果也是合格结果
- 所有测试用 `conda run -n Anomaly python -m pytest` 执行

---

## Task 1: 方向与数据集锁定（P0-1）

**对应 spec**: §1 方向与数据集锁定

**Files:**
- 无代码改动，产出文档

- [ ] **Step 1: 确认数据集锁定**

编写一页纸文档 `docs/dual_factor/direction_lock.md`，写明：
- 唯一主线 = 双因子模式研究
- 数据集 = DROZY（14 名被试、36 个有效测试、512Hz、EOG-H/V）
- 旧 X 光材料已归档
- 下一决策点 = 第 14 天 EFA 强制决策
- OperatorEYEVP 留待 P0-4 决策后

**验收标准**: 一页纸内能看清唯一主线、数据集、下一决策点，不再出现双线。

---

## Task 2: 数据字典与预处理复现说明（P0-2）

**对应 spec**: §1.2 数据字典、§2 保留基础设施

**Files:**
- `docs/dual_factor/data_dictionary.md`（新建）

- [ ] **Step 1: 编写数据字典**

包含：
- 被试数（14）、有效测试数（36）、缺失/剔除规则（7-1/9-1/10-2/12-2/12-3/13-3 排除）
- 采样率（512 Hz）、通道（EOG-H/EOG-V/Cam-Sync）
- KSS 标签来源与二分类规则（≤4 alert、≥6 drowsy、=5 排除）
- PVT 反应时路径与格式
- EDF 文件路径模式

- [ ] **Step 2: 验证预处理复现**

确认 `configs/dual_factor_efa.yaml` 参数完整，执行一键 pipeline 验证指标矩阵可复现。

**验收标准**: 换一个人能按文档复现指标矩阵；DROZY 完整可读。

---

## Task 3: 标准化指标矩阵与质量报告（P0-3 基础）

**对应 spec**: §6 描述性统计与质量报告

**Files:**
- `src/dual_factor/analysis/descriptive.py`（修改：补充分布报告）
- `tests/dual_factor/test_descriptive.py`（修改）

- [ ] **Step 1: 扩展描述性统计**

在 `descriptive.py` 中新增 `indicator_distribution_report(df, indicators)` 函数：
- 输出每指标均值±标准差、中位数与 IQR、缺失率
- 分 alert/drowsy 组统计
- Shapiro-Wilk 正态性检验

- [ ] **Step 2: 编写/更新测试**

- [ ] **Step 3: 运行验证**

```bash
conda run -n Anomaly python -m pytest tests/dual_factor/test_descriptive.py -v
```

**验收标准**: 8 指标分布报告完整，缺失率 <10%，方向性检验完成。

---

## Task 4: 窗口非独立性敏感性分析（P0-3 核心）

**对应 spec**: §4 窗口非独立性

**Files:**
- `src/dual_factor/features/windowing.py`（修改：新增被试级聚合）
- `src/dual_factor/analysis/sensitivity.py`（新建）
- `tests/dual_factor/test_sensitivity.py`（新建）

- [ ] **Step 1: 被试级聚合函数**

在 `windowing.py` 新增 `aggregate_subject_level(df, subject_col, indicator_cols)`：
- 每被试多窗口 → 1 行均值
- 输出 14 行 × 8 指标

- [ ] **Step 2: 敏感性分析模块**

创建 `sensitivity.py`，实现三种方案：
- `subject_level_efa(df, ...)` → 14 行 EFA
- `loso_efa(df, subject_col, ...)` → 14 折留一 EFA，记录每折因子数/载荷/交叉/相关
- `nonoverlap_efa(df, ...)` → 非重叠窗口子集 EFA

- [ ] **Step 3: 编写测试**

- [ ] **Step 4: 运行验证**

```bash
conda run -n Anomaly python -m pytest tests/dual_factor/test_sensitivity.py -v
```

**验收标准**: 至少一种敏感性分析完成，报告被试级结论是否稳健。

---

## Task 5: 2 因子解 vs 3 因子解对比报告（P0-4 核心）

**对应 spec**: §3.3 对比报告

**Files:**
- `src/dual_factor/analysis/efa.py`（修改：新增对比函数）
- `tests/dual_factor/test_efa_compare.py`（新建）

- [ ] **Step 1: 对比提取函数**

在 `efa.py` 新增 `run_efa_compare(data, n_factors_list=[2, 3])`：
- 并行提取 2 因子解和 3 因子解
- 输出对比表：碎石图位置、平行分析建议、累计方差、最大载荷、最大交叉载荷、因子相关、指标归属

- [ ] **Step 2: 编写测试**

- [ ] **Step 3: 运行验证**

```bash
conda run -n Anomaly python -m pytest tests/dual_factor/test_efa_compare.py -v
```

**验收标准**: 2 因子与 3 因子解对比表完整，能回答"第三因子包含哪些指标、有无实质意义"。

---

## Task 6: 严格判据执行与三选一决策（P0-4 核心）

**对应 spec**: §3.4 预注册判据、§3.5 三选一

**Files:**
- `src/dual_factor/analysis/decision.py`（修改：严格判据）
- `tests/dual_factor/test_decision.py`（修改）

- [ ] **Step 1: 修正决策函数**

修改 `make_decision()` 函数：
- 输入：2 因子解与 3 因子解的完整结果
- 逐项检查 C1-C5 判据
- 输出：每项判据"达标/未达标"及数值
- 若任一未达标，禁止返回"可分离"，必须返回未达标项列表
- 三选一：A(支持双因子) / B(需修正) / C(转结构)

- [ ] **Step 2: 编写测试**
  - 测试全达标 → 判定 A
  - 测试交叉载荷 ≥0.3 → 禁止判定 A，返回 B
  - 测试 r ≥0.70 → 判定 C
  - 测试模糊输入 → 返回未达标项

- [ ] **Step 3: 运行验证**

```bash
conda run -n Anomaly python -m pytest tests/dual_factor/test_decision.py -v
```

**验收标准**: 判据严格执行，交叉 ≥0.3 时禁止"可分离"，三选一决策明确。

---

## Task 7: 构念效度论证与方向偏离校正（P0-3 辅助）

**对应 spec**: §5 构念效度

**Files:**
- `src/dual_factor/analysis/construct_validity.py`（新建）
- `tests/dual_factor/test_construct_validity.py`（新建）

- [ ] **Step 1: 代理效度论证**

创建 `construct_validity.py`：
- `proxy_validity_report(df, indicators)` → 按指标类别（分布形态型/幅度依赖型/时间型）分类报告代理风险
- `amplitude_normalization(df, subject_col, channels)` → 每被试每通道除以全程幅度标准差
- `direction_correction_report(df_before, df_after, indicators)` → 校正前后方向性检验对比

- [ ] **Step 2: 编写测试**

- [ ] **Step 3: 运行验证**

```bash
conda run -n Anomaly python -m pytest tests/dual_factor/test_construct_validity.py -v
```

**验收标准**: 代理效度论证完成，幅度敏感指标方向偏离如实报告。

---

## Task 8: Pipeline 集成与一键复现

**对应 spec**: §9 代码修改清单

**Files:**
- `src/dual_factor/pipeline.py`（修改）
- `tests/dual_factor/test_pipeline_smoke.py`（修改）

- [ ] **Step 1: 集成新模块**

修改 `run_pipeline(config)`：
- Step 5.5: 被试级聚合 EFA + LOSO-EFA（调用 sensitivity.py）
- Step 5.6: 2vs3 因子对比（调用 efa.py 新函数）
- Step 6.5: 构念效度报告（调用 construct_validity.py）
- Step 7: 严格判据决策（调用修正后 decision.py）

- [ ] **Step 2: 更新输出产物**

新增输出：
- `results/dual_factor_efa/sensitivity_analysis.json` → 敏感性分析结果
- `results/dual_factor_efa/efa_compare.csv` → 2vs3 因子对比表
- `results/dual_factor_efa/construct_validity_report.md` → 构念效度报告
- `results/dual_factor_efa/loso_efa_results.csv` → LOSO-EFA 14 折结果

- [ ] **Step 3: 运行测试**

```bash
conda run -n Anomaly python -m pytest tests/dual_factor/test_pipeline_smoke.py -v
```

**验收标准**: 一键 pipeline 含敏感性分析、2vs3 对比、构念效度报告、严格决策。

---

## Task 9: 真实数据端到端运行与决策报告

**对应 spec**: §10 合格标准

- [ ] **Step 1: 全量运行**

```bash
cd /bigdisk/dyx/xray-attention-monitoring
conda run -n Anomaly python -m dual_factor.pipeline --config configs/dual_factor_efa.yaml
```

- [ ] **Step 2: 检查决策报告**

读取 `results/dual_factor_efa/decision_report.md`，确认：
- C1-C5 每项判据达标/未达标及数值
- 三选一决策明确（A/B/C）
- 若有未达标项，原因已写明

- [ ] **Step 3: 检查敏感性分析**

确认 `sensitivity_analysis.json` 和 `loso_efa_results.csv` 存在且内容完整。

**验收标准**: 决策报告明确回答"呆滞与发散能不能分开"，附判据数值与证据等级。

---

## Task 10: 论文结构提纲与文献缺口表（P1-2 辅线）

**对应 spec**: §8 论文提纲

**Files:**
- `docs/dual_factor/paper_outline.md`（新建）
- `docs/dual_factor/literature_gaps.md`（新建）

- [ ] **Step 1: 论文提纲**

按 spec §8.1 编写完整章节结构，每章标注内容要点与当前完成状态。

- [ ] **Step 2: 文献缺口表**

按 spec §8.2 检索关键词，填写现有研究与本研究的填补列。

**验收标准**: 论文提纲完整，文献缺口表初版（辅线，≤20% 时间）。

---

## 验收检查清单

| 检查项 | 对应 Task | 合格标准 |
|---|---|---|
| 方向锁定 | Task 1 | 一页纸，DROZY 锁定 |
| 数据字典 | Task 2 | 可复现 |
| 指标质量报告 | Task 3 | 分布+方向性检验 |
| 窗口独立性 | Task 4 | 至少一种敏感性分析 |
| 2vs3 因子对比 | Task 5 | 对比表完整 |
| 严格判据 | Task 6 | 交叉≥0.3 禁止"可分离" |
| 构念效度 | Task 7 | 代理效度+方向偏离 |
| Pipeline | Task 8 | 一键含全部新增模块 |
| 决策报告 | Task 9 | 三选一明确 |
| 论文提纲 | Task 10 | 章节结构+文献缺口 |
