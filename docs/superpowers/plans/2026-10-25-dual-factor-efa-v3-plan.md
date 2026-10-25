# 疲劳目光行为双因子模式研究：路径 B 执行——修正后 EFA 重做与文献缺口补全 实施计划

> **上游 spec**: [2026-10-25-dual-factor-efa-v3.md](../specs/2026-10-25-dual-factor-efa-v3.md) v3.0
>
> **Conda 环境名**：`Anomaly`
>
> **前版关系**：v2.0 全部 Task 已完成并交付（基线）。本版只新增 `correction.py` 模块、扩展 pipeline 与 config、补全辅线文档，不改写 v2 模块。

**Goal:** 执行 v2 决议的路径 B——按预注册修剪阶梯重做 EFA，把 `needs_correction` 收口为 A（支持双因子，保留指标集）或 C（转结构）；同时完成 P1-2 辅线的文献缺口表 DOI 核实。

**Architecture:**

- 新增 `src/dual_factor/analysis/correction.py`：2 因子解评估、确定性修剪规则、阶梯循环、LOSO 2 因子固定解、最终决策报告格式化
- 修改 `src/dual_factor/pipeline.py`：新增 Step 7.5（阶梯）+ Step 10 改写（修正后决策）；`correction.enabled: false` 时保持 v2 行为
- 修改 `configs/dual_factor_efa.yaml`：新增 `efa.correction` 块
- 新增 `tests/dual_factor/test_correction.py`
- 辅线：`docs/dual_factor/literature_gaps.md`（Crossref 核实 DOI）

**Non-invasive principle:** v2 已实现模块（efa/decision/sensitivity/construct_validity/descriptive/plots）不重写；pipeline 只追加步骤与分支。

---

## Global Constraints

- 修剪规则、阈值、终止条件以 spec v3 §1 预注册为准，代码实现不得引入规则外的人工挑选
- 阶梯每一步落盘：剔除对象、理由码、判据数值、代理效度类别
- 判定 A 时结论必须带"保留指标集"限定；PA≠2 必须带维度警示
- 判定只有 A / C 两个终局，无第四选项；负结果是合格结果
- 辅线 DOI 全部经 Crossref 核实，核实不到标"待核实"，禁止凭记忆填写
- 测试执行：`conda run -n Anomaly python -m pytest tests/dual_factor/ -v`（项目根目录，PYTHONPATH=src）
- 运行：`conda run -n Anomaly python -m dual_factor.pipeline --config configs/dual_factor_efa.yaml`

---

## Task 1: correction.py 修剪阶梯模块

**对应 spec**: §1.2-1.4 阶梯结构、§1.3 修剪规则、§1.4 终止条件、§2 LOSO 2因子

**Files:**
- `src/dual_factor/analysis/correction.py`（新建）

- [ ] **Step 1: 常量与 2 因子解评估**

```python
HYPOACTIVE = ["FD", "PSV", "SA", "FN"]
DIFFUSE = ["SE", "GTE", "SV", "SDR"]
PROXY_CLASS = {"FD": "time", "PSV": "amplitude", "SA": "amplitude", "FN": "time",
               "SE": "distribution", "GTE": "distribution", "SV": "amplitude", "SDR": "distribution"}

def two_factor_solution(data: DataFrame, ...) -> dict
    # run_efa_fixed(data, 2) + 统计：per_var_max / per_var_cross / cum_var / factor_corr / assignments
```

- [ ] **Step 2: 确定性修剪规则**

```python
def prune_rule(solution: dict, thresholds: dict) -> tuple[str | None, str, str]
    # 返回 (剔除指标, 理由码, 明细)
    # 优先级 1: cross ≥ 0.30 → 剔交叉最高者（cross_loading）
    # 优先级 2: 有主载荷 < 0.50 → 剔主载荷最低者（low_loading）
    # 优先级 3: 仅 C1 未达标 → 剔主载荷最低者（variance_driven）
    # 优先级 4: |r| ≥ 0.70 → (None, factor_collinear)
```

- [ ] **Step 3: 下限检查与阶梯循环**

```python
def floor_check(active: list[str], max_prunes: int, min_indicators: int, min_per_factor: int) -> tuple[bool, str]
def run_correction_ladder(efa_input: DataFrame, config: dict) -> dict
    # Rung 0 从全 8 项开始；每 rung 评估 C1/C2'/C3/C4 + 平行分析
    # 通过 → stop；未通过 → prune_rule 选下一个剔除 → 检查 floor
    # 输出: rungs 列表（每 rung 全记录）+ final（active/verdict/pruned 及理由）+ pa_n_factors
```

- [ ] **Step 4: LOSO 2 因子固定解**

```python
def loso_fixed_2factor(df, subject_col, indicator_cols, suffix="_z", **kw) -> DataFrame
    # 14 折，每折 run_efa_fixed(train, 2)，记录 cross_max / factor_corr / per_var_max_min
```

- [ ] **Step 5: 最终决策报告格式化**

```python
def format_final_decision(ladder: dict, sensitivity: dict, loso2: DataFrame) -> str
    # markdown：基线摘要 → 阶梯表（每 rung 指标集/判据/剔除理由）→ 最终判据表
    # → PA 维度警示（若 ≠2）→ 敏感性 → 判定 + 下一步
    # verdict: supported / unsupported_structural，带保留指标集限定语
```

**验收标准**: 规则实现与 spec §1.3 优先级表逐行对应；阶梯输出含每 rung 完整记录。

---

## Task 2: config 扩展与 pipeline 集成

**对应 spec**: §4 代码修改清单、§4.1 阶段划分

**Files:**
- `configs/dual_factor_efa.yaml`（修改）
- `src/dual_factor/pipeline.py`（修改）

- [ ] **Step 1: config 新增 correction 块**

```yaml
efa:
  # ...v2 键全部保留...
  correction:
    enabled: true
    max_prunes: 3
    min_indicators: 5
    min_per_factor: 2
    loading_clarity: 0.50    # C2'：保留指标主载荷下限
    cross_loading_max: 0.30  # C3
    variance_min: 0.60       # C1
    factor_corr_max: 0.70    # C4
```

- [ ] **Step 2: pipeline Step 7.5 阶梯**

在 2vs3 对比（Step 7）之后插入：
- `correction_ladder = run_correction_ladder(efa_input, config["efa"])` → `correction_ladder.json`
- 打印每 rung 摘要（指标数、判据、剔除）

- [ ] **Step 3: 修正后敏感性（最终保留集）**

- 用 `ladder["final"]["active"]` 重跑 subject_level_efa / loso_efa / nonoverlap_efa
- 新跑 `loso_fixed_2factor` → `corrected_loso_2factor.csv`
- 汇总 → `corrected_sensitivity_summary.json`（复用 sensitivity_summary 机制 + LOSO 2 因子反弹统计）

- [ ] **Step 4: Step 10 决策报告改写**

- `correction.enabled: true`：`decision_report.md` = `format_final_decision(...)`（v3 报告，含基线摘要与阶梯全记录）
- `correction.enabled: false`：保持 v2 行为（make_decision 路径），向后兼容

- [ ] **Step 5: 最终 rung 载荷图**

`plot_loading_matrix` 支持传入指定 EFAResult → `figures/loading_matrix_corrected.png`

**验收标准**: `enabled: false` 时输出与 v2 完全一致；`enabled: true` 时新增 4 个产物文件且 v2 基线产物不被覆盖语义（同名基线文件数值不变，因为步骤 1-9 输入未变）。

---

## Task 3: 测试

**对应 spec**: §6 合格标准

**Files:**
- `tests/dual_factor/test_correction.py`（新建）

- [ ] **Step 1: prune_rule 单元测试（合成载荷）**

- 交叉载荷超标 → 剔交叉最高者，理由码 `cross_loading`
- 主载荷过低 → 剔最低者，理由码 `low_loading`
- 仅方差不足 → `variance_driven`
- 因子共线 → `(None, factor_collinear)`
- 优先级：交叉优先于低载荷

- [ ] **Step 2: floor_check 单元测试**

- 指标数 <5 触发；呆滞/发散任一 <2 触发；剔除超限触发

- [ ] **Step 3: 阶梯端到端（合成数据）**

- 合成 8 变量数据：6 个双因子结构变量 + 1 个高交叉噪声变量 + 1 个无载荷噪声变量 → 阶梯应恰好剔除这 2 个噪声变量并在 rung 2 判定 supported
- 合成全噪声数据 → 阶梯应触底判定 unsupported_structural（路径 C）

- [ ] **Step 4: loso_fixed_2factor 冒烟测试**

- 小合成数据跑通，输出列齐全

- [ ] **Step 5: 运行全量测试**

```bash
cd /bigdisk/dyx/xray-attention-monitoring
PYTHONPATH=src conda run -n Anomaly python -m pytest tests/dual_factor/ -v
```

**验收标准**: 新测试全过，v2 既有测试（15 个文件）不回归。

---

## Task 4: 真实数据端到端运行与决策报告核验

**对应 spec**: §5 输出产物、§6 合格标准

- [ ] **Step 1: 全量运行**

```bash
cd /bigdisk/dyx/xray-attention-monitoring
PYTHONPATH=src conda run -n Anomaly python -m dual_factor.pipeline --config configs/dual_factor_efa.yaml
```

- [ ] **Step 2: 核验决策报告**

读取 `decision_report.md`，确认：
- 判定为 A 或 C（无中间态）
- 若 A：带"保留指标集"限定 + PA 维度警示（如适用）+ C5 结论
- 阶梯表完整（每 rung 的指标集、判据数值、剔除理由）

- [ ] **Step 3: 核验阶梯与敏感性产物**

- `correction_ladder.json`：逐 rung 记录完整，剔除对象与规则一致
- `corrected_sensitivity_summary.json` + `corrected_loso_2factor.csv`：内容齐全，LOSO 反弹折数已统计

**验收标准**: 决策报告明确回答"修正后呆滞与发散能不能分开"，且过程可追溯。

---

## Task 5: 辅线——文献缺口表补全（P1-2）

**对应 spec**: §3 辅线

**Files:**
- `docs/dual_factor/literature_gaps.md`（新建）

- [ ] **Step 1: Crossref 逐条核实 R1-R7 + Dorn/Klett**

对每条：`https://api.crossref.org/works?query.bibliographic=...` 检索 → 确认题名/作者/venue/年份/DOI 匹配 → 记录。核实不到 → 标"待核实"。

- [ ] **Step 2: 撰写文献缺口表**

四个缺口行（双因子假设来源 / DROZY 相关 / EOG 代理效度 / 疲劳眼动因子分析）×（支撑文献 + DOI + 一句话贡献 + 本研究的填补）。

**验收标准**: 每个 DOI 可点击验证或如实标"待核实"；总耗时 ≤20%。

---

## Task 6: 文档收口

**Files:**
- `docs/dual_factor/direction_lock.md`（追加）
- `docs/dual_factor/paper_outline_and_literature_gap.md`（更新 §5/§6）

- [ ] **Step 1: direction_lock 追加 v3 决议段**

P0-4 已决议 B（10-24）→ v3 执行 B → 本版判定结果 + 下一决策点（A → CFA 预注册；C → 三因子解释性分析/单维转轨）

- [ ] **Step 2: paper_outline §5 结果摘要更新**

按 v3 实际结果更新"当前实验结果摘要"与"下一步"（判定、保留指标集、剔除清单、维度警示）

**验收标准**: 一页纸与提纲反映最新判定，无过期结论残留。

---

## 验收检查清单

| 检查项 | 对应 Task | 合格标准 |
|---|---|---|
| 修剪规则实现 | Task 1 | 与 spec §1.3 逐行对应，无人工挑选空间 |
| 阶梯透明 | Task 1/4 | 逐 rung 落盘，剔除附理由码+判据数值+代理效度类别 |
| pipeline 兼容 | Task 2 | `enabled: false` 回退 v2；`true` 新增 4 产物 |
| 测试 | Task 3 | 新测试全过，v2 测试零回归 |
| 判定终局 | Task 4 | A / C 明确，表述带限定与警示 |
| 辅线 | Task 5 | DOI 核实或如实标注 |
| 文档收口 | Task 6 | direction_lock / paper_outline 与最新判定一致 |
