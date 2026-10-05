# 多模态特征提取与有效关注识别实验

## 目标

在有效关注阈值确定的基础上，开展面向 X 光安检员的多模态有效关注识别研究。我们将分别提取注视行为特征和面部行为特征，并对比单模态与多模态融合的识别性能。

## 当前状态

### 已完成

1. ✅ 有效关注阈值确定
   - Overall: 900px
   - Easy: 975px
   - Hard: 625px
2. ✅ 已有预处理的误差序列数据 `Distan_error（原20）/`（已误差校准）
3. ✅ 原始视频数据在 `data/` 目录下
4. ✅ 项目结构已搭建完成

### 重要决策

- ❌ 不使用 `same_feature/` 目录下的数据
- ✅ 注视特征从已校准的 `Distan_error/` 提取
- ✅ 面部特征从原始视频重新提取
- ✅ 生成的特征数据集放在 `dataset/` 目录

## 数据来源

### 原始数据

- `data/` 目录：20 位受试者的 `training_video.mp4` 和目标点坐标
- `Distan_error（原20）/` 目录：已校准的逐采样注视-目标距离误差序列

### 数据集结构

```
Distan_error（原20）/
├── easy/
│   └── 01-20/
│       ├── alert/
│       │   ├── all_errors.txt
│       │   └── average_error.txt
│       └── sleepy/
└── hard/
    └── 01-20/
        ├── alert/
        └── sleepy/
```

## 研究问题

1. 仅使用注视行为特征，能否有效识别有效/无效关注？
2. 仅使用面部行为特征，能否有效识别有效/无效关注？
3. 多模态融合（注视+面部）能否提升识别性能？
4. 任务难度（easy vs hard）对识别性能有何影响？

## 系统设计

### 整体文件结构

```
xray-attention-monitoring/
├── data/                    # 原始数据（保持原样）
│   ├── 01/
│   │   ├── alert/
│   │   └── sleepy/
│   └── ... (20 subjects)
├── Distan_error（原20）/    # 已校准的注视误差
├── dataset/                 # 新增：处理后的特征数据集
│   ├── 01/
│   │   ├── alert/
│   │   │   ├── easy/
│   │   │   │   ├── gaze_features.csv
│   │   │   │   └── face_features.csv
│   │   │   └── hard/
│   │   └── sleepy/
│   └── ... (20 subjects)
├── src/xray_attention/
│   └── data/
│       ├── gaze_feature_extractor.py      # 从 Dist_error 提取注视特征
│       ├── face_feature_extractor.py      # 从视频提取面部特征
│       └── dataset_builder.py             # 构建完整数据集
```

### 处理流程

1. **注视特征（gaze_features.csv）**
   - 直接从 `Distan_error/` 加载 `all_errors.txt`（已校准）
   - 计算统计特征
   - 使用我们之前选定的有效关注阈值

2. **面部特征（face_features.csv）**
   - 从 `data/` 加载 `training_video.mp4`
   - 使用 MediaPipe Face Mesh 提取面部特征

3. **对齐与保存**
   - 确保两种特征在时间上对齐
   - 保存为 CSV 格式到 `dataset/` 目录

## 特征设计

### 1. 注视特征（gaze_features）

从 `Distan_error/` 中的 `all_errors.txt` 提取：

**帧级特征：**
- error_distance：注视-目标距离误差（像素）

**窗口级特征（滑动窗口，5秒/10秒可调）：**
- mean_error：平均误差
- std_error：误差标准差
- median_error：误差中位数
- p_in_target：目标区域内采样占比（低于选定阈值）
- max_consecutive_low_error：最长连续低误差时长
- label：有效关注标签（1=alert，0=sleepy）

### 2. 面部特征（face_features）

从视频提取（使用 MediaPipe Face Mesh）：

**眼部特征：**
- left_ear：左眼 EAR（Eye Aspect Ratio）
- right_ear：右眼 EAR
- avg_ear：平均 EAR
- is_blinking：是否眨眼
- blink_rate：眨眼频率

**头部姿态：**
- yaw、pitch、roll：欧拉角
- head_movement：头部微动幅度

**面部稳定性：**
- face_stability：面部关键点运动幅度

### 3. 样本标签

- alert=1，sleepy=0（来自文件夹名）

## 实验方案

### 实验 1：注视特征基线

**输入：** 仅注视行为特征
**输出：** 有效/无效关注（alert=1, sleepy=0）
**模型：** MLP / LSTM / Transformer
**评价指标：** ROC-AUC、平衡准确率、F1-score

### 实验 2：面部特征基线

**输入：** 仅面部行为特征
**输出：** 有效/无效关注（alert=1, sleepy=0）
**模型：** MLP / LSTM / Transformer
**评价指标：** ROC-AUC、平衡准确率、F1-score

### 实验 3：多模态融合

**输入：** 注视特征 + 面部特征
**输出：** 有效/无效关注（alert=1, sleepy=0）
**模型：** 特征级融合（简单拼接 + MLP）
**评价指标：** ROC-AUC、平衡准确率、F1-score

### 验证策略

- 留一受试者交叉验证（LOSO）：与阈值实验保持一致
- 按受试者隔离，确保泛化性

## 验收标准

- 清晰的数据集说明和特征定义文档
- 可复现的特征提取和实验入口
- 完整的实验记录和结果分析
- 包含注视、面部、多模态三种方案的性能对比
