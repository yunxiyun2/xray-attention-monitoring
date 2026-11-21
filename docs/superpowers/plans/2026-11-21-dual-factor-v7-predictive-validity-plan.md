# Plan v7：增量预测效度实验

> 日期：2026-11-21。  
> 对应规范：[SPEC v7](../specs/2026-11-21-dual-factor-v7-predictive-validity.md)。  
> 目标：在实验机上用现有特征矩阵得到可审计的 AUC 三条件对照。
>
> 执行状态：V7-T01—T06 已完成；正式运行目录
> `results/dual_factor_predictive_v7/20261121T042129Z-70784213`，主判定
> `inconclusive`。

## V7-T01：固定协议与配置

新增 v7 协议、配置校验和 YAML：

- 固定七指标、三个理论分数、五种模型和 PHASE 0；
- 固定 KSS 阈值 4/6、`C=0.1`、bootstrap 2000；
- 校验路径、迭代数、阈值和禁止覆盖规则。

验收：配置缺少字段、类别阈值错误或覆盖开关打开时直接失败。

## V7-T02：构建配对被试数据

读取现有指标矩阵，执行以下检查：

1. 主键与标签一致性；
2. 只保留 phase 0、完整且有限的指标窗口；
3. 对被试内标准化指标进行记录等权聚合；
4. 为每个配对被试构造一条 KSS ≤ 4 和一条 KSS ≥ 6 的被试级记录；
5. 输出每个被试的纳入/排除原因和有效窗口数。

验收：窗口不会作为独立样本；同一被试的清醒与困倦记录始终处于同一交叉验证折。

## V7-T03：实现理论分数与留一模型

按 SPEC 计算 global、F1、F2。对五个固定模型执行留一被试：

- 在训练折拟合 scaler 与 `LogisticRegression(C=0.1)`；
- 只对留出被试产生预测；
- 检查训练折同时包含两类、留出身份不与训练重叠；
- 统一输出预测表、折表和缺失/失败原因。

验收：构造一个打乱标签的测试时，流程仍能完成；若出现训练/留出泄漏测试必须报错。

## V7-T04：计算指标、置信区间和判定

使用外样本预测计算 AUC、AP、Brier 和 log loss。按被试整群 bootstrap 2000 次，分层保留清醒/困倦两类；在同一抽样上计算模型差值的区间。

输出 `primary_delta_auc`，并严格按 SPEC §5 生成四种判定之一。不得在结果生成后再调参或改阈值。

验收：固定随机种子重跑得到一致的结果文件；所有差值区间使用同一次被试重抽样。

## V7-T05：写运行审计与报告

每次运行创建唯一目录，写入：

- `config_used.yaml`、`manifest.json`、`data_audit.json`；
- `subject_summary.csv`、`fold_predictions.csv`、`metrics.csv`、`bootstrap.csv`；
- `decision.json`、`report.md`，以及 ROC 与分层汇总图。

报告中必须区分“双因子结构成立”和“理论分数有增量预测价值”，并附上前置限制。

## V7-T06：实验机执行

从项目根目录安装 CPU 依赖：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[stats]'
```

先运行矩阵审计：

```bash
.venv/bin/python scripts/run_predictive_v7.py --dry-run
```

确认输入路径和 SHA256 后执行：

```bash
.venv/bin/python scripts/run_predictive_v7.py \
  --config configs/dual_factor_predictive_v7.yaml
```

验收：输出目录唯一，并有完整审计、预测、区间和判定。若为 `not_evaluable`，保留失败记录，不补样本。

## 完成定义

- SPEC 中的固定协议全部有代码实现；
- 单元测试覆盖分数方向、折内无泄漏、bootstrap 分层和判定规则；
- 实验机运行后给出 `incremental_supported`、`inconclusive`、`global_better` 或 `not_evaluable`；
- 报告没有用预测结果改写 v6 EFA 结论。
