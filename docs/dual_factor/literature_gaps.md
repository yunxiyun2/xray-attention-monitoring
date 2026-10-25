# 文献缺口表（DOI 经 Crossref 核实版）

> 版本：1.0
> 日期：2026-10-25
> 对应：SPEC 2026-10-25 v3.0 §3（P1-2 辅线）
> 核实方式：api.crossref.org 检索确认题名/作者/venue/年份/DOI；两次检索仍无法确认的条目标"待核实"，不凭记忆填写
> 上游文档：paper_outline_and_literature_gap.md §3-§4

---

## 1. 已核实文献清单（7/9）

| 编号 | 文献（作者, 年份, venue） | DOI（已核实） | 一句话贡献 | 与本研究关系 |
|---|---|---|---|---|
| R1 | Massoz, Langohr, Francois, Verly et al. (2016). WACV | [10.1109/wacv.2016.7477715](https://doi.org/10.1109/wacv.2016.7477715) | 发布 DROZY 多模态嗜睡数据库（PSG/EOG/近红外视频/KSS/PVT） | 阶段一探索性分析数据集来源 |
| R2 | Cazzoli, Antoniades, Kennard, Nyffeler et al. (2014). PLoS ONE | [10.1371/journal.pone.0087146](https://doi.org/10.1371/journal.pone.0087146) | 双重分离实验：昼夜节律型疲劳↑平均注视时长，时间累积型疲劳↓扫视速度，两效应统计独立 | **双因子假设最直接的文献证据**：不同疲劳源作用于不同眼动参数 → 支持多成分结构 |
| R3 | Hamann, van Klaren, Zon, Dehais (2025). Frontiers in Neuroergonomics | [10.3389/fnrgo.2025.1673268](https://doi.org/10.3389/fnrgo.2025.1673268) | 座舱精神疲劳头戴传感技术综述，梳理现有测量手段但未涉及眼动因子结构 | 绪论定位研究空白（综述级证据：因子结构是未开垦问题） |
| R4 | Abe (2023). SLEEP Advances | [10.1093/sleepadvances/zpad006](https://doi.org/10.1093/sleepadvances/zpad006) | PERCLOS 嗜睡检测技术的证据综述与未来方向 | 单一指标范式代表：论证"从单一指标转向因子结构"的必要性 |
| R5 | Wei, Dollack, Kiyokawa, Perusquía-Hernández et al. (2025). Sensors | [10.3390/s25216754](https://doi.org/10.3390/s25216754) | EOG + 本体感受校准可实现水平/垂直注视估计（含闭眼场景） | EOG 代理效度的正面证据（论文第2/8章引用） |
| R6 | Bodaghi, Hosseini, Gottumukkala, Bhupatiraju et al. (2026). Scientific Data | [10.1038/s41597-025-06540-1](https://doi.org/10.1038/s41597-025-06540-1) | 发布视频+生物特征+行为的多模态嗜睡新数据集，并将 DROZY 列为基线并指出其标注粒度粗 | 阶段二候选验证集；论证 DROZY 局限时的引用来源 |
| R7 | Bafna, Bækgaard, Hansen (2021). PLOS ONE | [10.1371/journal.pone.0246739](https://doi.org/10.1371/journal.pone.0246739) | 眼动打字任务中的精神疲劳预测（眨眼/瞳孔/眼动特征） | 眼动疲劳指标文献，但不涉及因子结构 → 反衬缺口 |

### 与提纲初版的出入（如实记录）

| 项 | 提纲初版表述 | Crossref 核实结果 | 处理 |
|---|---|---|---|
| R3 venue/年份 | "Frontiers (2024)" | Frontiers in **Neuroergonomics**，在线 2025（卷期 2026） | 论文引用时按核实结果更正 |
| R6 venue | "Nature (2025)" | **Scientific Data**（Nature Portfolio），2025 在线/2026 卷期 | 属 Nature 子刊而非 Nature 主刊，引用时更正 |
| R4 作者 | 未记 | Abe, T.（单作者） | 补全 |

---

## 2. 待核实文献（2/9）

| 引用出处 | 提纲中的表述 | 核实状态 | 说明 |
|---|---|---|---|
| Dorn (2017) | 76 名驾驶员 2 小时单调驾驶，20 个眼动变量层次聚类识别 8 个聚类，"驾驶员疲劳眼动存在类型学异质性" | ❌ 待核实 | Crossref 两轮检索（题名/作者组合）未命中。可能为会议论文或预印本（如 VisLab 技术报告）。**在核实到原始出处前，论文中不得作为事实引用**；如需保留该论点，须先找到原始文献或替换为可核实的同类研究 |
| Klett et al. (2025) | 24 小时睡眠剥夺对注视熵的影响存在性别交互：女性转移熵/静止注视熵显著降低，男性无变化 | ❌ 待核实 | Crossref 两轮检索未命中。同上处理 |

> 处理原则（SPEC v3 §3.1）：核实不到标"待核实"，禁止凭记忆填写 DOI。上述两条是双因子假设文献链中"个体异质性/亚类"论点的支撑，缺失时第 1 章该论点需降级表述或暂缓使用。

---

## 3. 文献缺口表（更新版）

| 缺口 | 检索关键词 | 现有研究（已核实） | 本研究的填补 |
|---|---|---|---|
| **双因子假设来源** | fatigue gaze behavior, hypoactive, diffuse, eye movement factor structure | R2（Cazzoli 2014）证明不同疲劳源作用于不同眼动参数（时序型 vs 昼夜型双重分离）；R3（Hamann 2025）综述未涉及因子结构 | 首次提出疲劳目光行为的双因子假设（呆滞 vs 发散）并以 EFA 检验可分离性；R2 的双重分离是"多成分"动机的直接证据，但 R2 未做因子分析 |
| **DROZY 相关研究** | DROZY database, drowsiness EOG | R1（Massoz 2016）发布数据库；后续研究多用于 EAR/PERCLOS/眨眼检测；R6（Bodaghi 2026）将其列为基线并指出标注粒度粗 | 首次在 DROZY 上用 EOG 通道提取 8 项目光行为指标做因子结构分析（而非分类器评测） |
| **EOG 代理效度** | EOG eye position proxy, EOG spatial distribution | R5（Wei 2025）证明 EOG 可（含闭眼时）估计水平/垂直注视；已知局限：空间分辨率低于视频眼动仪、校准漂移、幅度敏感 | 提出代理效度三分类（时间型/分布形态型/幅度依赖型）；v3 实证发现剔除的两项指标（SV=幅度依赖型、SDR=分布形态型但主载荷缺失）与代理风险框架可对话 |
| **疲劳眼动因子分析** | fatigue eye movement factor analysis, drowsiness EFA | 检索未发现任何对疲劳眼动指标做 EFA 的研究；R4（Abe 2023）综述确认 PERCLOS 单一指标范式主导；R7（Bafna 2021）单独使用眼动特征 | 首次对疲劳眼动指标做预注册判据的 EFA；v3 修正后在 6 指标集（FD/PSV/SA/FN + SE/GTE）上支持双因子可分离 |

---

## 4. 论文绪论引用建议（第 1 章落地）

按论证链组织，非按时间罗列：

1. **应用需求段**：R4（PERCLOS 范式及其局限）→ 引出"单一指标不足以刻画疲劳行为异质性"
2. **现象矛盾段**：R2（双重分离）→ 疲劳眼动方向不一致有系统结构，非噪声
3. **测量手段段**：R5（EOG 代理可行性）→ 为什么 EOG 数据可以做目光行为分析
4. **数据基础段**：R1（DROZY）+ R6（新数据集对 DROZY 的定位）→ 数据集选择理由与局限
5. **研究空白段**：R3（综述未涉及因子结构）+ R7（特征单独使用）→ 无人做过疲劳眼动 EFA
6. **个体异质性问题**：Dorn/Klett 若核实到出处则在此引用；否则该论点降级为"有待在阶段二亚类分析中检验"

---

## 5. 下一步（辅线收尾）

- [ ] Dorn 2017 / Klett 2025：在 Web of Science / 出版商官网人工检索原始出处；核实到后补入本表并解除引用限制
- [ ] 论文第 1 章初稿按 §4 论证链展开（列入下期汇报计划，不在本两周内）
