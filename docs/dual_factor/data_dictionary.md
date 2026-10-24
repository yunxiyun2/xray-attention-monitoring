# DROZY 数据字典与预处理复现说明

> 日期：2026-10-22
> 对应：P0-2 强制决策点

## 1. 数据集概述

DROZY 是一个公开嗜睡数据库，用于检测嗜睡/疲劳的算法评估。包含多导睡眠图（PSG）记录的 EOG 信号、PVT 行为反应时、近红外视频和 KSS 主观评分。

## 2. 被试与记录

| 维度 | 数值 |
|---|---|
| 被试总数 | 14 名（编号 1-14） |
| 每被试测试数 | 3 次 PVT 任务 |
| 有效测试数 | 36 个 |
| 排除记录 | 7-1（KSS=0 无效）、9-1（缺失）、10-2（缺失）、12-2（缺失）、12-3（缺失）、13-3（缺失） |
| 采样率 | 512 Hz |

## 3. 信号通道

| 通道 | EDF 标签 | 单位 | 说明 |
|---|---|---|---|
| EOG-H | 取决于 EDF 头 | μV | 水平眼电，编码在 EDF 文件中 |
| EOG-V | 取决于 EDF 头 | μV | 垂直眼电 |
| Cam-Sync | — | — | 帧同步方波（阶段四使用） |

通道识别逻辑：`drozy_loader.py` 中按通道名匹配 `EOG`、`horizontal`、`vertical` 等关键词自动识别。

## 4. 标签

| 标签 | 来源 | 路径 | 格式 |
|---|---|---|---|
| KSS | `KSS.txt` | `{drozy_root}/KSS.txt` | 14 行 × 3 列，整数 1-9 |
| 二分类 | KSS 阈值 | — | ≤4 = alert，≥6 = drowsy，=5 排除 |
| PVT 反应时 | CSV | `{drozy_root}/pvt-rt/{subj}-{test}.csv` | 列：刺激时间;反应时间 |

## 5. 预处理流程

| 步骤 | 方法 | 参数 | 配置键 |
|---|---|---|---|
| 带通滤波 | Butterworth 零相位 | 0.5–30 Hz，4 阶 | `preprocess.highpass`/`lowpass`/`filter_order` |
| 离群抑制 | Hampel 滤波 | 窗口 51，阈值 3.0 σ | `preprocess.hampel_window`/`hampel_threshold` |

## 6. 事件检测

| 参数 | 值 | 配置键 |
|---|---|---|
| 算法 | I-VT 速度阈值法 | — |
| 速度阈值 | 自适应百分位（P70） | `event_detection.velocity_threshold`/`velocity_percentile` |
| 最短注视 | 100 ms | `event_detection.min_fixation_ms` |
| 最短扫视 | 20 ms | `event_detection.min_saccade_ms` |

## 7. 八项指标

| 因子 | 指标 | 符号 | 计算方式 |
|---|---|---|---|
| 呆滞 | 平均注视时长 | FD | 注视段时长均值 |
| 呆滞 | 扫视峰值速度 | PSV | 扫视段速度峰值均值 |
| 呆滞 | 扫视幅度 | SA | 扫视起点到终点位移均值 |
| 呆滞 | 注视频率 | FN | 单位时间内注视次数 |
| 发散 | 空间熵 | SE | 注视点空间分布信息熵 |
| 发散 | 注视转移熵 | GTE | 注视区域转移模式信息熵 |
| 发散 | 空间方差 | SV | 注视点坐标离散程度 |
| 发散 | 方向随机性 | SDR | 扫视角度与均匀分布的 KL 散度 |

## 8. 窗口化

| 参数 | 值 | 配置键 |
|---|---|---|
| 窗口长度 | 30 秒 | `windowing.window_sec` |
| 步长 | 10 秒 | `windowing.stride_sec` |
| 有效门槛 | 注视≥5 且 扫视≥3 | `windowing.min_fixations`/`min_saccades` |
| 标准化 | 被试内 z-score | — |

## 9. 复现方式

从项目根目录执行：

```bash
conda run -n Anomaly python -m dual_factor.pipeline --config configs/dual_factor_efa.yaml
```

配置文件为唯一参数源：`configs/dual_factor_efa.yaml`
