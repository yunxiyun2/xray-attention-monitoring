# SI-MSET v1.2 消融实验方案与执行说明

## 实验目的

在 v1.2 的最终训练与评价协议下，分别测量多尺度编码、受试者对抗约束和辅助序数监督对清醒—困倦被试级判别的贡献。v1 的消融运行只作为早期诊断记录，不代替本实验。

## 消融变体

| 变体 | 保留内容 | 移除内容 |
|---|---|---|
| full | SI-MSET v1.2 完整结构与训练目标 | 无 |
| single_scale | 对抗约束与辅助序数监督 | 多尺度编码 |
| no_adversarial | 多尺度编码与辅助序数监督 | 受试者对抗约束 |
| binary_only | 多尺度编码与受试者对抗约束 | 中间状态/序数辅助监督 |

变体开关由 `src/eog_si_mset/protocol.py` 的 `variant_flags` 统一控制，模型构造与损失计算沿用 SI-MSET v1.2 训练实现。

## 固定协议与统计口径

- DROZY 配对被试、缓存、LOSO 测试折及折内训练/验证边界保持一致。
- 每个变体使用同一组三个随机种子；每个种子完成全部 10 折，因此每个变体包含 30 次折内拟合。
- 模型宽度、Transformer 层数、优化器、学习率、训练轮数、早停、采样与损失权重均沿用 `configs/eog_si_mset_v1_2.yaml`。
- 每个变体先做三种子 logit 融合，再按预注册主方案对每名测试被试的未标注窗口分数执行 rank 校准，汇总成被试级清醒/困倦状态分数。
- 主指标为 subject-state AUC。组件效应定义为 `ΔAUC = AUC(full) − AUC(ablation)`。
- 使用相同随机数种子生成的被试级 bootstrap 抽样，对完整模型和各消融模型做配对比较；报告 95% percentile CI。bootstrap 单位是被试，避免将窗口误当作独立样本。
- raw、center、z-score AUC 与原始概率 Brier/log loss作为补充量，主组件结论以 rank 校准 AUC 的配对效应为准。

## 执行

优先复用已完成的 v1.2 full 运行，并在同一 RTX 3090 执行环境运行三个消融变体：

```bash
python scripts/run_si_mset_v12_ablation.py \
  --config configs/eog_si_mset_v1_2.yaml \
  --reuse-full-run results/eog_si_mset_v1_2/20261121T103544Z-2de47193
```

若需将 full 对照也一并重跑：

```bash
python scripts/run_si_mset_v12_ablation.py \
  --config configs/eog_si_mset_v1_2.yaml \
  --train-all
```

执行器在正式训练前核验 full 运行的协议、配置、主校准方法和当前设备；汇总时再次核验代码哈希、缓存路径、GPU、被试列表、随机种子、折划分及评估行。消融运行不保存模型 checkpoint，只保留预测、训练历史和统计产物。任何条件不一致时停止汇总，不生成貌似可比的结论。每次执行均写入唯一目录，不覆盖已有结果。

## 产物

每个模型变体使用现有训练器输出配置、manifest、逐折历史、预测、校准指标和 bootstrap 样本。总汇总目录额外包含：

- `summary.csv`：各变体 AUC、区间及补充指标；
- `paired_deltas.csv`：full 对三个消融的配对 ΔAUC 与置信区间；
- `report.md`：可直接引用的实验方法与实测结果；
- `run_manifest.json`：四个变体运行目录、代码/配置哈希和固定随机种子；
- `rank_auc_intervals.*`、`paired_delta_auc.*`：主结果图。

## 解释规则

若某个 `full − ablation` 的 95% 配对区间完全大于 0，结果支持完整模型中对应组件带来正向贡献；若区间完全小于 0，结果支持消融变体更优；若区间覆盖 0，则报告点估计与区间，不将差异解释为已识别的组件贡献。对组件有效性的叙述只依据本 v1.2 同协议实验，不把 v1 消融或校准敏感性分析当作替代证据。
