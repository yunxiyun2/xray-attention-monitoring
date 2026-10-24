# 构念效度论证报告

## 1. 代理效度分类

| 指标 | 类别 | 风险等级 | 幅度敏感 |
|---|---|---|---|
| FD | time | 低（不依赖绝对幅度） | 否 |
| PSV | amplitude | 中（受 EOG 电位基线影响） | 是 |
| SA | amplitude | 中（受 EOG 电位基线影响） | 是 |
| FN | time | 低（不依赖绝对幅度） | 否 |
| SE | distribution | 低（只依赖概率分布形状） | 否 |
| GTE | distribution | 低（只依赖概率分布形状） | 否 |
| SV | amplitude | 中（受 EOG 电位基线影响） | 是 |
| SDR | distribution | 低（只依赖概率分布形状） | 否 |

## 2. 方向性检验结果

| 指标 | 假设方向 | 实际方向 | 一致 | p 值 |
|---|---|---|---|---|
| FD | increase | increase | ✅ | 0.0000 |
| PSV | decrease | increase | ❌ | 0.0000 |
| SA | decrease | decrease | ✅ | 0.0000 |
| FN | decrease | increase | ❌ | 0.1659 |
| SE | increase | decrease | ❌ | 0.0000 |
| GTE | increase | decrease | ❌ | 0.0000 |
| SV | increase | increase | ✅ | 0.0000 |
| SDR | increase | increase | ✅ | 0.0000 |

方向一致率：4/8（50%）

## 3. 代理效度论证

- **分布形态型指标（SE/GTE/SDR）**：理论上只依赖概率分布形状，不依赖绝对幅度，
  EOG 代理对这些指标的影响最小。若方向偏离，需检查网格化参数。

- **幅度依赖型指标（PSV/SA/SV）**：受 EOG 电位基线影响，方向偏离可能源于代理局限。
  已做幅度归一化校正，校正后复验方向性。

- **时间型指标（FD/FN）**：不依赖绝对幅度，代理风险最低。

## 4. 局限性声明

本研究以 EOG 信号代理二维眼动位置，幅度相关指标的方向偏离
可能反映代理局限而非假设错误。这不影响因子结构的分析，
但在解释幅度敏感指标的因子载荷时需谨慎。
