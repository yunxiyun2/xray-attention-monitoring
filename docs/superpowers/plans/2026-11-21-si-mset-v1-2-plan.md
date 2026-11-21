# Plan SI-MSET v1.2

> 对应规范：[SPEC v1.2](../specs/2026-11-21-si-mset-v1-2.md)。  
> 目标：验证 v1 结果是否跨随机种子稳定，并在无标签被试内分布对齐后重新评价 AUC。

## V12-T01：多种子训练

- 配置中加入固定 seed 列表；
- 每个 LOSO 折对每个 seed 独立初始化、训练和早停；
- 保存单 seed 预测和指标；
- 三个 seed 的 logit 等权融合。

验收：同一折三个 seed 使用完全相同的训练/验证/测试索引；没有 seed 跨折复用 checkpoint。

## V12-T02：固定跨被试验证

- `rotating_subject` 验证模式改为在所有 epoch 复用固定窗口；
- 验证被试不参与当前折训练；
- 不使用测试被试做 early stopping。

验收：同一折的训练、验证、测试被试互斥，验证索引在 fold 内恒定。

## V12-T03：受试者内无标签校准

实现 raw、center、z-score 和平均秩四种逐窗口分数；

- 校准函数不得读取 `label/state`；
- 聚合后才使用标签计算 AUC；
- 每个状态行保存窗口数和校准方法；
- 主结果固定为三 seed 融合 + rank。

验收：打乱测试标签不改变任一校准分数；所有被试少于 20 个评价窗口时拒绝校准结果。

## V12-T04：统计与报告

- 对单 seed 和融合结果计算 AUC、AP、Brier、log loss；
- 2000 次被试整群 bootstrap；
- 在同一 bootstrap 样本上计算 rank−raw 和 rank−v7 的差值；
- 输出四种校准方式、逐被试排序和图形。

验收：主判定严格使用 SPEC §4 四项门槛；不得挑选最有利 seed 或校准方法。

## V12-T05：实验机执行

```bash
.venv/bin/python scripts/train_eog_transformer.py \
  --config configs/eog_si_mset_v1_2.yaml \
  --variant full
```

预期三次 seed 训练共享同一数据缓存和折划分。开始前确认 GPU 可用：

```bash
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 完成定义

- 10 折 × 3 seed 全部成功；
- 四种校准输出和配对 bootstrap 完整；
- `result.json` 明确主方法、AUC、区间和判定；
- 未通过时保留负结果，不继续针对测试集调参。
