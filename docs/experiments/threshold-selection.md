
# 有效关注阈值选择实验

## 标签定义
- `alert=1`：有效关注状态
- `sleepy=0`：无效/不足关注状态

## 候选阈值范围
- 起始：50 像素
- 终止：1000 像素
- 步长：25 像素

## 嵌套 LOSO 逻辑
1. **外层：留一受试者验证（LOSO）**
   - 在每个外层折，将一名受试者的数据保留为测试集
   - 剩余受试者的数据作为训练集
   
2. **内层：阈值选择**
   - 在训练集上，对每个候选阈值计算：
     - 基于 `inside_ratio_&lt;threshold&gt;` 的 ROC-AUC
     - 平衡准确率（Balanced Accuracy）
     - Mann-Whitney U 检验和秩二列相关系数
   - 选择最佳阈值：
     - 优先按 ROC-AUC 降序排序
     - 其次按平衡准确率降序
     - 最后按阈值升序
   - 对选择的阈值，在训练集上寻找最佳判别临界值（使平衡准确率最大化的中点）

3. **测试**
   - 在测试集上，使用选择的阈值和训练得到的判别临界值
   - 计算测试集上的平衡准确率和 ROC-AUC

## 实验产物说明

1. `config_used.yaml`：本次实验使用的完整配置文件
2. `task_features.csv`：80 个任务的特征表，包含基本统计和每个阈值的指标
3. `candidate_summary.csv`：所有候选阈值的汇总统计
4. `nested_loso_folds.csv`：每个外层折的详细结果
5. `recommended_threshold.json`：推荐的阈值配置
6. `threshold_performance.png`：候选阈值的 ROC-AUC 性能图
7. `selected_thresholds.png`：每个外层折选择的阈值频数图

## 重要限制
- 本阶段中，连续片段长度的单位为采样点，不换算为秒
- 所有切分严格以受试者为单位，确保测试受试者不参与训练阶段

