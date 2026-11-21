# Plan SI-MSET v1

> 日期：2026-11-21。  
> 对应规范：[SPEC SI-MSET v1](../specs/2026-11-21-subject-invariant-mset-v1.md)。  
> 执行环境：raw EOG 预处理与缓存可在 CPU 完成；模型训练建议使用单张 GPU。

## SIM-T01：配置与协议

新增协议常量和配置校验：

- 固定输入三通道、128 Hz、30 秒窗、10 秒步长；
- 固定 KSS 二分类阈值、LOSO 规则和评价门槛；
- 禁止覆盖历史缓存与运行目录；
- 校验四种模型变体和 CUDA/CPU 设备选择。

验收：错误字段、窗口参数、阈值、折数或变体名均直接报错。

## SIM-T02：原始 EOG 张量缓存

新增一阶数据准备命令：

1. 通过现有 DROZY loader 读取 EOG-H/V；
2. 完成滤波、Hampel、稳健归一化、128 Hz 重采样和速度通道；
3. 逐窗检测最低注视/扫视数；
4. 保存 float32 数组和完整元数据；
5. 输出原始文件、数组和配置 SHA256。

验收：断点路径错误、窗口退化或标签不在 1–9 时失败；缓存不覆盖。

## SIM-T03：模型

新增轻量 PyTorch 模型：

- 三分支多尺度深度可分离卷积；
- 门控融合、Transformer 和注意力池化；
- 二分类头、KSS 序数回归头；
- 梯度反转受试者分类头；
- 四变体均由固定开关控制。

验收：无 torch 的环境只跳过 GPU 模型测试；有 torch 时检查前向 shape、有限输出、变体开关和反向梯度。

## SIM-T04：平衡批次与损失

实现受试者配对 batch sampler：

- 每批含若干训练被试；
- 每人各采清醒/困倦窗口；
- masked BCE、KSS MSE、受试者内排序 hinge、受试者对抗 CE；
- 输出可审计的 loss 分量。

验收：训练 batch 不会混入测试被试；无二分类标签的 KSS=5 只进入序数损失。

## SIM-T05：LOSO 训练与评估

新增训练入口：

- 枚举配对被试形成 LOSO 折；
- 每折固定训练/验证/测试受试者；
- 支持 GPU、AMP、early stopping 和断点记录；
- 聚合窗口预测到被试状态；
- 计算 AUC/AP/Brier/log-loss 和整群 bootstrap；
- 可选加载 v7 `global` 外样本预测完成配对比较。

验收：所有测试受试者在训练和验证折中不出现；无标签泄漏；结果目录唯一。

## SIM-T06：消融与报告

用四个固定变体执行相同划分：

```bash
python scripts/train_eog_transformer.py --config configs/eog_si_mset_v1.yaml --variant full
python scripts/train_eog_transformer.py --config configs/eog_si_mset_v1.yaml --variant single_scale
python scripts/train_eog_transformer.py --config configs/eog_si_mset_v1.yaml --variant no_adversarial
python scripts/train_eog_transformer.py --config configs/eog_si_mset_v1.yaml --variant binary_only
```

报告主模型、配对差、消融方向、失败折和计算环境。不得根据测试集为某个变体调参。

## 实验机执行顺序

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[stats,raw,deep]'

.venv/bin/python scripts/prepare_eog_tensor.py \
  --config configs/eog_si_mset_v1.yaml

.venv/bin/python scripts/train_eog_transformer.py \
  --config configs/eog_si_mset_v1.yaml \
  --variant full
```

先用合成张量运行模型单元测试和单折 smoke，再执行完整 LOSO。完整 GPU 运行会依次训练 10 个测试折；预计主要成本来自 10 折训练，而不是数据缓存。

## 完成定义

- 原始数据缓存可追溯且不泄漏；
- 模型、训练、评估、消融命令可运行；
- 至少完成完整 LOSO 的 `full` 变体；
- 输出被试状态 AUC、区间、与 v7 基线的配对差和明确判定；
- 若未完成全部消融，只能报告主模型结果和“消融未运行”，不得声称各模块有效。
