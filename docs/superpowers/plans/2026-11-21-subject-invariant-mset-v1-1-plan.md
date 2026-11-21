# Plan SI-MSET v1.1

> 对应规范：[SPEC v1.1](../specs/2026-11-21-subject-invariant-mset-v1-1.md)。  
> 目标：在固定验证集上使用更大模型和更长训练预算重跑同一 LOSO 评价。

## SIM11-T01：修正验证采样

- 新增 `training_subjects` 验证模式；
- 每折开始时固定抽样训练被试的平衡窗口；
- 保持原 `rotating_subject` 模式以复现 v1。

验收：同一折的所有 epoch 使用完全相同的验证索引；验证损失不再来自每轮重新抽样。

## SIM11-T02：扩容配置

- 新增 `configs/eog_si_mset_v1_1.yaml`；
- 模型宽度 128、5 层 Transformer、8 头、128 token；
- 每轮 32 batch，每批 5 名被试，每人每状态 16 窗口；
- 最大 130 epoch，patience 24。

验收：配置通过严格校验；旧 v1 配置和运行记录不被覆盖。

## SIM11-T03：测试

- 模型前向 shape 和反向梯度；
- 固定验证索引跨 epoch 不变；
- `training_subjects` 模式与测试被试不相交；
- 单折 CPU smoke。

验收：专项测试通过，旧 v1 结果不受代码修改影响。

## SIM11-T04：实验机完整运行

```bash
.venv/bin/python scripts/train_eog_transformer.py \
  --config configs/eog_si_mset_v1_1.yaml \
  --variant full
```

数据缓存无需重做，直接复用 `results/eog_si_mset/cache_v1`。

验收：完成 10 折，写出唯一的 `results/eog_si_mset_v1_1/<timestamp>/` 目录和 `result.json`。

## SIM11-T05：结果判断

- 只使用 v1.1 首次完整运行的测试折结果；
- 按原门槛报告 `supported`、`inconclusive`、`below_chance` 或 `not_evaluable`；
- 不根据 v1.1 的测试 AUC 再调 epoch、模型宽度或早停门槛；
- 若需要模块归因，另行运行固定消融变体。
