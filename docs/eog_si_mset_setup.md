# SI-MSET 训练环境布置指南

> 供其他 Agent 或开发者复现 EOG 深度学习训练流程。

## 1. 硬件要求

| 项目 | 本机配置 | 最低要求 |
|---|---|---|
| GPU | NVIDIA RTX 3090 (24 GB) | 任何 CUDA ≥ 11.8 的 NVIDIA GPU，≥ 6 GB 显存 |
| CUDA | torch 自带 cu124 runtime | 驱动支持 CUDA ≥ 11.8 |
| 内存 | — | ≥ 16 GB（DROZY 全量加载约 4 GB） |
| 磁盘 | — | ≥ 5 GB（EDF 原始 + 缓存 + checkpoint） |

CPU-only 可运行但极慢（10 折约数小时 vs GPU 约 10 分钟）。设 `training.device: cpu` 即可。

## 2. 系统与 Python

- OS：Ubuntu 20.04 LTS（Linux 任意发行版均可）
- Python：3.11（Anaconda 或 venv）

```bash
# 推荐：用 conda 创建独立环境
conda create -n eog python=3.11 -y
conda activate eog
```

## 3. 安装依赖

### 3.1 核心依赖（必装）

```bash
pip install torch>=2.3          # GPU 版需匹配 CUDA，见 https://pytorch.org
pip install numpy>=1.26 pandas>=2.2 scipy>=1.13
pip install scikit-learn>=1.5,<1.8
pip install matplotlib>=3.9 pyyaml>=6
pip install pyedflib>=0.1.30    # EDF 文件读取
```

### 3.2 实际验证版本

以下版本已在 RTX 3090 上完整跑通 10 折 LOSO：

```
python        3.11.14
torch         2.6.0+cu124
numpy         1.26.4
pandas        2.1.4
scipy         1.11.4
scikit-learn  1.2.2
matplotlib    3.8.0
pyedflib      0.1.42
pyyaml        6.0.1
```

### 3.3 可选依赖（报告/测试用）

```bash
pip install pytest>=8           # 单元测试
pip install python-docx>=1.1    # Word 报告生成
```

### 3.4 项目安装

```bash
cd /bigdisk/dyx/xray-attention-monitoring
pip install -e '.[stats,raw,deep]'
```

## 4. 数据准备

### 4.1 DROZY 数据集

数据位于 `data_public/DROZY/`，结构：

```
data_public/DROZY/
├── KSS.txt              # 每条记录的 KSS 评分
└── psg/
    ├── 1-1.edf          # {subject}-{test}.edf，共 37 个文件
    ├── 1-2.edf
    ├── ...
    └── 14-3.edf
```

EDF 文件包含 EOG-H（水平）和 EOG-V（垂直）通道，采样率 200 Hz。

### 4.2 构建 EOG 张量缓存

训练前必须先运行数据预处理，生成标准化缓存：

```bash
cd /bigdisk/dyx/xray-attention-monitoring
python3 scripts/prepare_eog_tensor.py --config configs/eog_si_mset_v1.yaml
```

输出：
- 缓存目录：`results/eog_si_mset/cache_v1/`
- `windows.npy`：形状 `(N, 3, 3840)` 的 float32 张量（3 通道 × 128Hz × 30s）
- `metadata.csv`：每个窗口的被试/记录/KSS/标签/状态
- `manifest.json`：SHA256 溯源和配对被试列表

预期结果：~2047 个有效窗口，41 个被拒绝，10 名配对被试。

**注意**：缓存目录已存在时会报错（防覆盖）。如需重建，先删除 `results/eog_si_mset/cache_v1/`。

## 5. 配置文件

路径：`configs/eog_si_mset_v1.yaml`

关键参数：

| 段 | 参数 | 值 | 说明 |
|---|---|---|---|
| data | drozy_root | `data_public/DROZY` | DROZY 根目录 |
| data | target_fs | 128 | 重采样频率 (Hz) |
| data | window_sec / stride_sec | 30 / 10 | 窗长 / 步长 (秒) |
| data | alert_max / drowsy_min | 4 / 6 | KSS 清醒/困倦阈值 |
| model | variant | `full` | 4 种消融：full / single_scale / no_adversarial / binary_only |
| model | d_model | 96 | Transformer 隐藏维度 |
| model | transformer_layers | 3 | Transformer 层数 |
| training | epochs | 60 | 最大训练轮数 |
| training | patience | 12 | 早停耐心值 |
| training | learning_rate | 0.0003 | AdamW 学习率 |
| training | adversarial_weight | 0.10 | 对抗损失权重 |
| training | amp | true | 混合精度训练 |
| evaluation | bootstrap_iterations | 2000 | Bootstrap 次数 |
| evaluation | min_auc | 0.75 | 支持判定的 AUC 门槛 |
| output | save_checkpoints | true | 保存每折最佳权重 |

## 6. 启动训练

```bash
cd /bigdisk/dyx/xray-attention-monitoring
python3 scripts/train_eog_transformer.py --config configs/eog_si_mset_v1.yaml
```

### 消融实验（可选）

```bash
python3 scripts/train_eog_transformer.py --variant single_scale
python3 scripts/train_eog_transformer.py --variant no_adversarial
python3 scripts/train_eog_transformer.py --variant binary_only
```

## 7. 训练输出

每次运行创建唯一目录：`results/eog_si_mset/{timestamp}-{hash}/`

| 文件 | 内容 |
|---|---|
| `manifest.json` | 协议版本、设备、SHA256 |
| `config_used.yaml` | 实际使用的配置 |
| `folds.csv` | 10 折分折状态（被试/best epoch/val loss） |
| `fold_NN_history.csv` | 每折训练曲线（epoch/train loss/val loss） |
| `window_predictions.csv` | 窗口级预测概率 |
| `state_predictions.csv` | 被试状态级聚合概率（每被试 alert/drowsy 各一行） |
| `metrics.csv` | AUC/AP/Brier/log loss + bootstrap CI |
| `bootstrap_samples.csv` | 2000 次 bootstrap 原始采样 |
| `result.json` | 决策判定（supported/inconclusive/below_chance/not_evaluable） |
| `report.md` | 中文 Markdown 报告 |
| `figures/loso_performance.png` | ROC 曲线 + 分折训练曲线 |
| `figures/auc_intervals.png` | AUC bootstrap 置信区间图 |
| `fold_NN_subject_X.pt` | 每折最佳 checkpoint（可选） |

## 8. 预期运行时间

| 硬件 | 10 折总时间 |
|---|---|
| RTX 3090 (24GB) | ~10 分钟 |
| RTX 4090 (24GB) | ~6 分钟 |
| CPU only | ~2-3 小时 |

## 9. 验证安装

```bash
# 运行单元测试
cd /bigdisk/dyx/xray-attention-monitoring
python3 -m pytest tests/eog_si_mset/ -v
```

## 10. 常见问题

**Q: `ModuleNotFoundError: No module named 'pyedflib'`**
A: `pip install pyedflib>=0.1.30`

**Q: `FileNotFoundError: DROZY root not found`**
A: 检查 `configs/eog_si_mset_v1.yaml` 中 `data.drozy_root` 路径。默认为 `data_public/DROZY`（项目相对路径）。

**Q: `FileExistsError: Cache directory already exists`**
A: 缓存已存在。如需重建：`rm -rf results/eog_si_mset/cache_v1/` 后重新运行 `prepare_eog_tensor.py`。

**Q: CUDA out of memory**
A: 减小 `training.batch_subjects`（默认 4）或 `training.windows_per_state`（默认 8），或设 `training.amp: true`。

**Q: 训练结果不可复现**
A: 确保固定 `training.seed: 2026112101` 和 `evaluation.seed: 2026112102`，同 GPU 型号和 torch 版本下结果一致。
