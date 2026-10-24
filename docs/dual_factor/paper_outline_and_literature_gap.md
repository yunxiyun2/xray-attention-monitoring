# 论文结构提纲与文献缺口表

> 版本：1.0
> 日期：2026-10-24
> 对应 SPEC：2026-10-22-dual-factor-efa-v2.md §8（P1-2 辅线）

---

## 1. 论文章节结构

| 章节 | 内容要点 | 预计字数 |
|---|---|---|
| **第1章 绪论** | 行业需求（疲劳监测/驾驶安全/安检值守）、应用场景、国内外研究动态、研究缺口、科学问题与假设 | 3000 |
| **第2章 数据集与标签效度** | DROZY 数据集描述（14名被试、36个有效测试、512Hz EOG）、EOG 代理决策、KSS 标签、PVT 行为度量 | 2000 |
| **第3章 八指标与构念效度** | 指标定义（FD/PSV/SA/FN + SE/GTE/SV/SDR）、事件检测算法（I-VT）、空间网格化（5×5）、代理效度论证（三类分类） | 2500 |
| **第4章 预处理与事件检测** | 带通滤波（0.5-30Hz）、Hampel 去伪迹、I-VT 算法、自适应百分位阈值 | 1500 |
| **第5章 窗口化与统计模型** | 30s 窗口、10s 步长、被试内 z-score 标准化、窗口非独立性处理与敏感性分析 | 2000 |
| **第6章 EFA/CFA/不变性** | 前置检验（KMO/Bartlett）、PAF+Promax+平行分析、2因子vs3因子对比、严格判据C1-C5、三选一决策、CFA（阶段二）、多群组不变性 | 3000 |
| **第7章 跨数据集验证** | OperatorEYEVP 验证（阶段二） | 1500 |
| **第8章 讨论与局限** | EOG 代理局限、样本量（n=14）、方向偏离、负结果管理、双因子假设的证伪与修正 | 2000 |
| **第9章 结论** | 双因子是否可分离的明确结论（当前：不支持，需修正指标后重做） | 500 |

---

## 2. 论文核心论点链

```
科学问题：疲劳状态下的目光行为是否存在可分离的双因子结构
（呆滞型 vs 发散型）？
    │
    ├── 假设：8项眼动指标可归为2个潜在因子
    │   ├── 呆滞型（Hypoactive）：FD↑ / PSV↓ / SA↓ / FN↓
    │   └── 发散型（Diffuse）：SE↑ / GTE↑ / SV↑ / SDR↑
    │
    ├── 方法：EFA（PAF + Promax + 平行分析）
    │   ├── 预注册判据 C1-C5（不可事后修改）
    │   ├── 2因子 vs 3因子强制对比
    │   └── 窗口非独立性敏感性分析（3种方法）
    │
    ├── 结果（当前）：
    │   ├── 平行分析建议 3 因子（非2因子）
    │   ├── 交叉载荷 0.409 > 0.30（C3 未达标）
    │   ├── 敏感性分析不一致（窗口3/被试4/非重叠2）
    │   ├── 方向一致率 4/8（50%）
    │   └── 决策：needs_correction（剔除/替换指标后重做）
    │
    └── 结论：当前数据不支持双因子可分离，需修正指标或转向单维/三因子模型
```

---

## 3. 文献缺口表

| 缺口 | 检索关键词 | 现有研究 | 本研究的填补 |
|---|---|---|---|
| **双因子假设来源** | fatigue gaze behavior, hypoactive, diffuse, eye movement factor structure | 现有文献将疲劳眼动视为单维结构。Frontiers综述(2024)讨论了头部佩戴传感器的疲劳评估，但未涉及因子结构。PLOS ONE(2014)发现不同眼动参数反映不同疲劳源（时序型vs昼夜型），但未做因子分析。 | 首次提出疲劳目光行为的双因子假设（呆滞型 vs 发散型），用EFA检验其可分离性 |
| **DROZY 相关研究** | DROZY database, drowsiness EOG | Massoz et al.(2016 WACV)发布DROZY数据库。后续研究多用于EAR/PERCLOS/眨眼检测（如Real-time Classification 2020，94.44%准确率）。Nature(2025)发布新数据集时将DROZY列为基线，指出其标注粒度粗（每10分钟1次）。 | 使用DROZY的EOG通道提取8项目光行为指标，而非EAR/PERCLOS，首次在该数据集上做因子结构分析 |
| **EOG 代理效度** | EOG eye position proxy, EOG spatial distribution | MDPI Sensors(2025)证明EOG可在闭眼时追踪眼动轨迹。已知局限：空间分辨率低于视频眼动仪、校准漂移、幅度敏感（Psychophysiology教材）。但无研究系统分类EOG指标的代理风险。 | 提出代理效度三分类（时间型/分布形态型/幅度依赖型），对幅度敏感指标做归一化校正并报告方向偏离 |
| **疲劳眼动因子分析** | fatigue eye movement factor analysis, drowsiness EFA | 检索未发现任何对疲劳眼动指标做EFA的研究。现有研究均将指标单独使用（如IOVS 2017的扫视幅度/速度/注视稳定性，PLOS ONE 2014的注视时长/扫视速度双分离）。 | 首次对8项疲劳眼动指标做探索性因子分析，检验是否存在潜在因子结构 |

---

## 4. 关键参考文献

| 编号 | 文献 | 与本研究关系 |
|---|---|---|
| R1 | Massoz et al. (2016). The ULg Multimodality Drowsiness Database (DROZY). WACV. | 数据集来源 |
| R2 | PLOS ONE (2014). Eye Movements Discriminate Fatigue Due to Chronotypical Factors and Time Spent on Task. | 眼动参数反映不同疲劳源的证据（支持多因子假设动机） |
| R3 | Frontiers (2024). The state of the art in assessing mental fatigue in the cockpit using head-worn sensing technology. | 疲劳评估综述，定位研究空白 |
| R4 | Sleep Advances (2023). PERCLOS-based technologies for detecting drowsiness. | PERCLOS主流方法的局限（单一指标vs因子结构） |
| R5 | MDPI Sensors (2025). Electro-Oculography and Proprioceptive Calibration Enable Gaze Estimation. | EOG代理效度证据 |
| R6 | Nature (2025). A multimodal drowsiness dataset using video, biometric, and behavioral data. | 新数据集，可作为阶段二验证集 |
| R7 | PLOS ONE (2021). Mental fatigue prediction during eye-typing. | 眼动疲劳指标（眨眼频率、瞳孔直径），但不涉及因子结构 |

---

## 5. 当前实验结果摘要（供论文第6章引用）

### 5.1 EFA 决策报告

| 项目 | 数值 |
|---|---|
| 提取因子数 | 3（平行分析驱动） |
| 累计方差解释 | 69.2% |
| 最大载荷 | 0.967 |
| 最大交叉载荷 | **0.409**（C3 未达标，阈值 <0.30） |
| 两因子相关 | -0.121 |

| 判据 | 实测值 | 阈值 | 达标 |
|---|---|---|---|
| C1 累计方差 | 0.692 | >0.60 | ✅ |
| C2 最大载荷 | 0.967 | >0.50 | ✅ |
| C3 交叉载荷 | 0.409 | <0.30 | ❌ |
| C4 因子相关 | -0.121 | <0.70 | ✅ |
| C5 被试级一致 | 不一致 | 一致 | ❌ |

**决策**：needs_correction（剔除高交叉载荷指标后重做EFA）

### 5.2 敏感性分析

| 方法 | 因子数 |
|---|---|
| 窗口级（基线） | 3 |
| 被试级聚合 | 4 |
| 非重叠窗口 | 2 |
| LOSO 众数 | 3（14折中11折=3，3折=2） |

**结论**：因子结构跨方法不稳定，需如实报告。

### 5.3 构念效度

- 方向一致率：4/8（50%）
- 方向偏离指标：PSV（幅度型）、FN（时间型）、SE/GTE（分布型）
- 幅度归一化校正后复验方向性

### 5.4 2因子 vs 3因子对比

| 对比项 | 2因子解 | 3因子解 |
|---|---|---|
| 累计方差 | 54.4% | 69.2% |
| 最大交叉载荷 | 0.469 | 0.409 |
| 因子相关 | -0.048 | -0.121 |

**结论**：3因子解交叉载荷仍 >0.30，2因子解更差。第三因子未显著降低交叉载荷。

---

## 6. 下一步

1. **主线**：识别高交叉载荷指标（SV 载荷 -0.469 在2因子解中交叉严重），剔除后重做EFA
2. **辅线**：补充文献检索（具体DOI和摘要），完善论文第1章绪论
3. **阶段二**：若修正后EFA支持双因子 → CFA验证；若不支持 → 转单维或三因子模型
