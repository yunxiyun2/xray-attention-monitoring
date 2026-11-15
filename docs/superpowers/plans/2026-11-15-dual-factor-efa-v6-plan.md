# Plan v6.0：4/3 结构修正实施计划

> 日期：2026-11-15。
> 状态：计划完成；代码修正、模拟验证和正式执行待实施。
> 唯一规格：[SPEC v6](../specs/2026-11-15-dual-factor-efa-v6.md)。
> 前序计划：[Plan v5](2026-11-08-dual-factor-efa-v5-plan.md)。v5 已执行完成，决策 B。
> 本版是 v5 Decision B 允许的"一份下一版本修正建议"的实施计划。v5 的全部方法论严格性、C0—C5 门控、阶段状态机和诚实报告要求不变；v6 仅修正候选指标集和分组。

## 1. 实施原则

- v5 代码、结果和文档保持原样，不全局替换历史记录。
- v6 新增独立的 protocol/pipeline/config/script 文件，不覆盖 v5 版本。
- v5 已验证的基础设施（sampling、paf、cluster_bootstrap、paired_validity、parallel_analysis 等）直接复用，仅更新协议常量和调用参数。
- 每个任务状态只允许 `pending/implemented/tested/executed/accepted/blocked`。
- 失败也保存；报告不能引用"计划将得到的数字"。

## 2. 依赖图

```text
V6-T01 protocol.py → V6-T02 config.yaml → V6-T03 pipeline.py
                                          → V6-T04 simulation.py → V6-T05 run_efa_v6.py
V6-T03 + V6-T04 + V6-T05 → V6-T06 模拟验证 → V6-T07 正式执行 → V6-T08 决策报告
```

## 3. 实施任务

### V6-T01：更新协议常量

**依赖**：无。**执行**：Mac。

新增 `src/dual_factor/protocol_v6.py`，基于 v5 `protocol.py` 更新：

```python
RETAINED = ("FD", "PSV", "SA", "FN", "SE", "GTE", "SV")  # SV 加入
ALL_INDICATORS = RETAINED + ("SDR",)                       # SDR 仍为基线
GROUPS = {"f1": RETAINED[:4], "f2": RETAINED[4:]}          # F2 = SE, GTE, SV
DIRECTIONS = {"FD": 1, "PSV": -1, "SA": -1, "FN": -1,
              "SE": 1, "GTE": 1, "SV": 1, "SDR": -1}       # SV 保持 +1
```

CRITERIA 和 STABILITY 阈值不变。

验收：AC01 更新为 4/3；`GROUPS["f2"]` 长度为 3。

### V6-T02：创建 v6 配置

**依赖**：V6-T01。**执行**：Mac。

新增 `configs/dual_factor_efa_v6.yaml`，基于 v5 配置更新：

- `protocol_version: v6.0`
- `structure.retained: [FD, PSV, SA, FN, SE, GTE, SV]`（7 项）
- 其余参数（窗、预处理、事件检测、bootstrap、信号检测）不变。
- `output.root: results/dual_factor_efa_v6`

验收：配置 schema 校验通过；retained 长度为 7。

### V6-T03：创建 v6 pipeline

**依赖**：V6-T01、V6-T02。**执行**：Mac/CPU。

新增 `src/dual_factor/v6_pipeline.py`，基于 `v5_pipeline.py` 更新以下关键点：

1. **导入**：`from dual_factor.protocol_v6 import RETAINED, ALL_INDICATORS`
2. **模型对照**（`run_experiment` 内）：从 `eight/six × 2/3` 改为 `eight/seven/four × 2/3`
   - eight: ALL_INDICATORS（8 指标），2/3 因子
   - seven: RETAINED（7 指标），1/2/3 因子
   - four: RETAINED[:4]（F1 子集），2/3 因子（探索性）
3. **`_model()` 中的锚定逻辑**：
   - `expected` 分组：F1 = RETAINED[:4]，F2 = RETAINED[4:]（含 SV）
   - `expected_sign`：SV 的预期方向为 +1（与 SE/GTE 同方向），但实际载荷可能为负
   - `hypothesis_sign_match` 检查：SE/GTE/SV 预期符号均为 +1
4. **决策门控**：`phase_pass` 和 `dimensions_ok` 使用 `seven` 模型（替代 `six`）
5. **报告**：标题更新为 "v6 EFA 决策报告"，结构描述为 "4/3"
6. **配对效度**：7 项 Holm 校正（SV 加入）
7. **图表**：新增 `phase_0_seven_2` 和 `phase_0_seven_3` 载荷热图
8. **facts.json**：`groups.f2` 更新为 `["SE", "GTE", "SV"]`

验收：AC02 更新为 八/七/四 对照；AC17 SV 主载荷检查。

### V6-T04：创建 v6 模拟验证脚本

**依赖**：V6-T01、V6-T03。**执行**：Mac/CPU。

新增 `scripts/validate_v6_simulation.py`，基于 `validate_v5_simulation.py` 更新：

1. **合成数据生成器** `synthetic_matrix()`：
   - 7 指标（FD, PSV, SA, FN, SE, GTE, SV）
   - 载荷矩阵：
     ```python
     loadings = np.zeros((7, 2))
     loadings[:4, 0] = [.9, -.9, -.9, -.9]   # F1
     loadings[4:, 1] = [.9, .9, -.6]          # F2: SE, GTE 正, SV 负
     ```
   - `two` 场景更名为 `two_4_3`
   - `cross_loading` 场景：SV 交叉载荷 0.45（`loadings[6] = [.45, -.6]`）
   - 噪声维度从 6 改为 7

2. **`protocol_support()`**：使用 v6 RETAINED（7 项）

3. **验收检查**：
   - `strong_two_observable_signal_power`：`two_4_3` 信号检测率 ≥ 0.80
   - `latent_power`：`two_4_3` 潜因子接受率 ≥ 0.80（AC18）
   - 其余反例场景（null, single, correlated_two, cross_loading, heterogeneous, two_nonnormal）接受率上界 ≤ 0.10

4. **证书输出**：`code_sha256` 和 `config_sha256` 匹配 v6 代码和配置。

验收：AC06、AC18；`two_4_3` 接受率 ≥ 0.80。

### V6-T05：创建 v6 运行入口

**依赖**：V6-T03。**执行**：Mac。

新增 `scripts/run_efa_v6.py`，基于 `run_efa_v5.py`：
- 导入 `from dual_factor.v6_pipeline import run_experiment`
- 默认配置指向 `configs/dual_factor_efa_v6.yaml`

验收：AC10、AC16；dry-run 路径一致。

### V6-T06：运行模拟验证

**依赖**：V6-T04、V6-T05。**执行**：CPU。

```bash
python scripts/validate_v6_simulation.py \
  --config configs/dual_factor_efa_v6.yaml \
  --datasets 200 --bootstrap-iterations 200 \
  --output results/dual_factor_efa_v6/simulation_certificate.json
```

验证 `diagnostic_checks_passed=True` 且 `latent_power_validated=True`。

如果 `two_4_3` 接受率 < 0.80：停止，报告修正方向问题。

### V6-T07：在真实 DROZY 数据上执行 v6 EFA

**依赖**：V6-T06 有效性通过。**执行**：CPU。

```bash
python scripts/run_efa_v6.py \
  --config configs/dual_factor_efa_v6.yaml \
  --input-mode matrix \
  --validation results/dual_factor_efa_v6/simulation_certificate.json
```

输入为已有 `results/dual_factor_efa/indicator_matrix.csv`（八指标矩阵，v6 从中选取 7 项）。

保存运行目录、manifest、facts.json、efa_models.json 等全部产物。

### V6-T08：生成 v6 决策报告

**依赖**：V6-T07。**执行**：Mac/CPU。

基于 `facts.json` 和运行产物编写完整决策报告：
- 决策 A/B/C 及证据等级
- SV 在 F2 的主载荷绝对值和交叉载荷（AC17）
- 配对效度（7 项，含 SV）
- 与 v5 结果的对比（4/2 → 4/3 的改善或限制）
- 因子命名遵循 §3.3 规则（中性命名，如 SV 方向反转则标注）

## 4. 与 v5 Plan 的差异

| 项目 | v5 Plan | v6 Plan |
|---|---|---|
| 指标集 | 6 项（FD,PSV,SA,FN,SE,GTE） | 7 项（+SV） |
| 分组 | 4/2 | 4/3 |
| 模型对照 | 八/六 × 2/3 | 八/七/四 × 2/3 |
| 模拟场景 | two（6指标4/2） | two_4_3（7指标4/3，SV负载荷） |
| 验收标准 | AC01—AC16 | +AC17（SV主载荷）、+AC18（模拟接受率） |
| 基础设施 | T01—T13 全新实现 | 复用 v5 已验证基础设施，仅更新协议层 |

v5 Plan 的 T01—T13 基础设施任务（sampling、paf、cluster_bootstrap、parallel_analysis 等）已全部实现并测试通过。v6 不重复实现这些模块，仅创建新的协议常量、配置、pipeline 和模拟脚本。

## 5. 风险与终止条件

| 风险 | 处理 |
|---|---|
| SV 交叉载荷 > 0.30 | C3 如实检查；失败则 Decision B |
| SV 主载荷 < 0.50 | C2 如实检查；失败则 Decision B |
| PAF 仍不收敛 | 报告失败；7 指标 3 因子结构理论上可识别 |
| 模拟接受率 < 0.80 | 修正方向有问题；报告并停止 |
| SV 方向反转 | 配对效度如实报告；因子中性命名 |

**终止条件**：v6 是 v5 Decision B 允许的唯一修正建议。如果 v6 也产出 Decision B，不再继续试删或换指标，报告当前指标集的固有限制。

## 6. 完成定义

- V6-T01—T05 代码完成且可运行。
- V6-T06 模拟验证 `diagnostic_checks_passed=True` 且 `latent_power_validated=True`。
- V6-T07 正式执行完成，产出完整运行目录。
- V6-T08 决策报告完成，决策 A/B/C 明确。
- 所有 AC01—AC18 有实证链接或失败原因。
