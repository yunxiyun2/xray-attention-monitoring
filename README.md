# EOG Fatigue State Monitoring

本项目研究如何从 DROZY 的双通道眼电信号识别清醒与困倦状态。当前主线为
**SI-MSET：Subject-Invariant Multi-Scale EOG Transformer（受试者不变多尺度眼电
Transformer）**。早期 X-Ray 多模态分类实验已归档移除，不再维护。

## 当前实验

SI-MSET 直接读取原始 `EOG-H`、`EOG-V` 和速度幅值，采用三分支多尺度时序编码、
Transformer、受试者对抗约束及清醒/困倦配对排序损失。所有评价使用被试级 LOSO，
测试被试不参与训练或早停。

当前 v1.2 协议进一步加入：

- 每个 LOSO 折使用 3 个固定随机种子；
- 融合三个 seed 的窗口 logit；
- 对比 raw、被试内中心化、z-score 和秩百分位四种校准；
- 主方法预注册为“三种子融合 + 被试内秩校准”；
- 校准不读取测试标签，仅适用于包含同一被试样大量未标注窗口的离线评价。

设计与执行计划：

- [SPEC v1.2](docs/superpowers/specs/2026-11-21-si-mset-v1-2.md)
- [Plan v1.2](docs/superpowers/plans/2026-11-21-si-mset-v1-2-plan.md)

## 环境安装

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[stats,raw,deep]'
```

在 `configs/eog_si_mset_v1_2.yaml` 中将 `data.drozy_root` 设置为实验机上的
DROZY 目录。已有 `results/eog_si_mset/cache_v1/` 时不需要重新生成缓存。

## 运行

若缺少 EOG 张量缓存：

```bash
.venv/bin/python scripts/prepare_eog_tensor.py \
  --config configs/eog_si_mset_v1_2.yaml
```

执行当前完整实验：

```bash
.venv/bin/python scripts/train_eog_transformer.py \
  --config configs/eog_si_mset_v1_2.yaml \
  --variant full
```

输出写入新的 `results/eog_si_mset_v1_2/<timestamp>-<id>/`，不会覆盖历史结果。

## 已完成结果

| 实验 | 设置 | 被试状态 AUC | 与 v7 baseline ΔAUC | 判定 |
|---|---|---:|---:|---|
| v1 | 单 seed，96 维，3 层 | 0.750 | +0.080 | inconclusive |
| v1.1 | 单 seed，128 维，5 层，130 epoch | 0.720 | +0.050 | inconclusive |
| 消融 | v1 的 `single_scale` | 0.730 | +0.060 | inconclusive |
| 消融 | v1 的 `no_adversarial` | 0.740 | +0.070 | inconclusive |
| 消融 | v1 的 `binary_only` | 0.730 | +0.060 | inconclusive |

详细记录：

- [v1 执行与消融](docs/dual_factor/si_mset_v1_execution_log.md)
- [v1.1 执行记录](docs/dual_factor/si_mset_v1_1_execution_log.md)
- [v1 消融报告](results/eog_si_mset/ablation_v1/report.md)
- [完整研究任务与结论](docs/project_report/研究任务与结论.md)

## 历史 EFA 与预测基线

在转向原始 EOG 深度表征前，项目完成了 DROZY 眼动代理指标的候选双因子 EFA 和
增量预测效度检验。v6 结构分析为 Decision B，v7 双分数相对单分数模型的
`ΔAUC` 为 −0.150，95% CI `[−0.470, 0.140]`，判定 `inconclusive`。

历史结构化结果保留在：

- `results/dual_factor_efa_v5/`
- `results/dual_factor_efa_v6/`
- `results/dual_factor_predictive_v7/`

v7 可由现有特征矩阵直接复现：

```bash
.venv/bin/python scripts/run_predictive_v7.py --dry-run
.venv/bin/python scripts/run_predictive_v7.py \
  --config configs/dual_factor_predictive_v7.yaml
```

## 测试

```bash
.venv/bin/python -m pytest tests/eog_si_mset -q
.venv/bin/python -m pytest tests/dual_factor -q
```

## 数据政策

原始 DROZY 数据、本地采集数据和视频不进入 Git。训练缓存中的 `windows.npy`
同样被忽略；实验机需从原始 EDF 重建。结果只适用于 DROZY 小样本和当前预处理，
不能外推为临床或生产环境结论。
