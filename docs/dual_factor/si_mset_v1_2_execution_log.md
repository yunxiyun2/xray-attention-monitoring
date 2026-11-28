# SI-MSET v1.2 执行记录

> 运行目录：`results/eog_si_mset_v1_2/20261121T103544Z-2de47193`  
> 设备：NVIDIA GeForce RTX 3090  
> 协议：`si-mset-v1.2`  
> 主判定：**supported**

## 1. 执行完整性

- 10 个 LOSO 折全部完成；
- 每折完成 3 个预注册 seed，共 30 次独立训练；
- 配置哈希：`37fdab1af635b5c29c16d4f9bec623a7068da9b24c1097f821908eed484619da`；
- 代码哈希：`68dfd78273a74e83fa91ef83c7834cbb1cd530bb71f09c6616bc979a392c5c30`；
- 输入继续使用 `results/eog_si_mset/cache_v1`；
- 主方法为 three-seed ensemble + subject rank calibration。

## 2. 结果

| 指标 | 结果 |
|---|---:|
| 单 seed AUC | 0.770 / 0.690 / 0.650 |
| three-seed raw ensemble AUC | 0.710 |
| center calibration AUC | 0.880 |
| z-score calibration AUC | 0.870 |
| rank calibration AUC（主方法） | **0.880** |
| 主方法 bootstrap 95% CI | [0.610, 1.000] |
| v7 global AUC | 0.670 |
| rank − raw | +0.170，95% CI [0.030, 0.300] |
| rank − v7 global | +0.210，95% CI [0.030, 0.420] |
| v1.2 raw Brier | 0.240 |
| v1.2 raw log loss | 0.710 |

10 名配对被试中，主方法保持 9/10 的清醒—困倦排序正确。逐人剔除检验中，rank 校准 AUC 的最低值为 0.864，出现在剔除被试 1 后；没有单个被试完全承载当前点估计。

## 3. 结论拆分

### 3.1 Seed 稳定性

三个 seed 的 raw AUC 分别为 0.770、0.690、0.650，跨度 0.120。三 seed raw 融合 AUC 为 0.710，低于最佳单 seed。因此：

- seed 平均没有证明模型原始概率比最佳单 seed 更稳定；
- 不应把 0.880 描述为“模型在不同初始化下都稳定达到该水平”；
- 任何报告中必须同时给出三个单 seed 结果。

### 3.2 被试内校准

center、z-score、rank 三种校准的 AUC 分别为 0.880、0.870、0.880，方向一致。主方法的配对 bootstrap 相对 raw 和 v7 baseline 的下界均为 0.030，按预注册规则通过。

校准不读取测试标签，只使用同一测试被试全部窗口的模型分数分布。因此它适用于已有整段未标注记录的离线评价，不能直接用于单窗口实时推断，也不能称为跨设备泛化。

### 3.3 总体判定

主协议要求的四项条件均满足：

1. 主 AUC 0.880 ≥ 0.75；
2. bootstrap 下界 0.610 > 0.50；
3. 相对 raw ensemble 提升 0.170，下界 0.030 > 0；
4. 相对 v7 global 提升 0.210，下界 0.030 > 0。

因此主判定为 `supported`。该结论是“在当前 DROZY 小样本、固定预处理和整段未标注被试校准条件下的支持”，不是通用疲劳检测器已经成立。置信区间仍宽，且相对两个基线的下界都只有 0.030。

## 4. 图件

- [ROC、单 seed 与校准比较](../../results/eog_si_mset_v1_2/20261121T103544Z-2de47193/figures/v12_performance.png)
- [四种校准方法的 bootstrap 区间](../../results/eog_si_mset_v1_2/20261121T103544Z-2de47193/figures/calibration_intervals.png)
