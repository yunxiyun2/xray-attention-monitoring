# v5 实施记录

日期：2026-11-08。当前代码入口为 `scripts/run_efa_v5.py`，配置为 `configs/dual_factor_efa_v5.yaml`。

## 已实现，仍需以测试与运行证据区分状态

- 协议固定六指标4/2、严格配置、输入哈希、不可覆盖run、正式/诊断运行分离。
- 按记录区间与三个相位选非重叠窗；同一分析内记录/被试等权标准化与协方差。
- 真实迭代PAF、SMC初始化、斜交共同方差、合法参数与不收敛错误，独立解析协方差基准测试。
- 八/六指标×两/三因子完整数值、六项单因子参考、独立列PA和记录中心化循环位移参照。
- 整人协方差贡献重采样、仅符号/置换对齐、失败分母、CI/归属概率/C5；辅助逐被试和LOSO。
- 记录均值→被试高低KSS差值、整人CI、完整符号枚举、Holm校正；不能按窗口数量增大人数。
- norm列映射、零均值聚合拒绝、无效行质量分母、事件计数/实际阈值元数据、EOG采样率/单位。
- Hampel改为真正逐窗口局部MAD，零MAD平台的孤立异常也处理；raw重提取和旧matrix必须分run比较。
- CFA指标按名称组装4/2、标准化载荷/真相关、同一样本尺度SRMR、Jacobian/Fisher信息合法性检查。
- 汇报从facts/manifest来源生成，诊断与未执行阶段明示；删除夸大/编造指令和硬编码历史结果。

## 验证边界和待执行任务

完整初次模拟揭示：7类场景各200数据集、每个200次bootstrap，潜因子A接受率均为0，不能把它称为有检验效能的验收。进一步诊断发现4/2随机样本的真PAF常不收敛或出现不适定共同度，延长迭代诊断也不能解决。

实现补充了可观测共变信号检验：按人bootstrap的组内相关及第二相关特征值，使用三个同时描述区间。它只检测是否有两个可观测共变组，不能冒充潜因子验证。模拟分别报告“信号识别率”和“潜因子A接受率”，并保持PAF/C5门槛原样。

`diagnostic_checks_passed`只允许生成完整方法诊断或B报告；必须另有`latent_power_validated`才允许A。不会以可观测信号存在来豁免潜因子估计失败。原模拟失败记录保留，不将新的信号阳性叫作旧潜因子验收成功。

- 小规模合成/回归测试验证实现；正式200数据集/场景、200 bootstrap的校准不能由少量测试替代。
- 4/2结构在随机样本中可能PAF不收敛，失败将阻止稳定性和科学验收，不增加迭代上限凑通过。
- 全套模拟门未通过或未执行时，正式EFA不开放；诊断入口可产出明确pending的数值和报告。
- 原始EDF不在当前Mac：波形盲复核、P60/P80、bins3/bins7、固定事件数熵、PVT联动尚未执行。
- 历史阶梯/新PAF诊断阶梯的完整审计、MINRES/历史principal敏感性、完整原始信号质量清单需补齐；当前主候选绝不自动增删。
- 普通ML CFA的指标计算已修正，但被试cluster稳健SEM推断未实现，CFA脚本明确不升级L3或打开外部阶段。
- 预测三条件/公平校准和外部/不变性是已设计待解锁任务，没有执行或性能结果。
- 教师验收尚无来源，不自动生成accepted=true；历史Word和原结果未改写。

CPU依赖通过隔离 `.venv` 安装，不改变系统Python；精确版本见 `environment_lock_v5.txt`。

## 已完成的实际验证

- `.venv/bin/python -m pytest tests/dual_factor -q`：112通过、8跳过。跳过的是本节点缺少DROZY原始数据的测试；历史factor_analyzer接口有弃用及近奇异矩阵警告，不称这些测试验证了原始信号。
- 原CSV只读审计：2088生成窗、2044有效窗、14人、36记录。真正相位0/1/2分别保留706/668/670完整窗；输入SHA256仍为e20c60040c6ceb7dfa72b8a1567c430cbaedb0d01adac15fa1ee8a27390d4138。
- 已执行7种场景各200数据集、每个200次bootstrap。初次潜因子接受率全部0，正式潜因子强正对照门未通过。
- 分离问题后：零维/单维观测信号各0/200误报，强双组和非正态双组均200/200检测；交叉载荷样本也可有观测信号，不能据此证明简单4/2潜结构；异质样本130/200检测。
- 潜因子A接受率仍为0，`latent_power_validated=false`、整体`passed=false`。仅诊断校准通过；程序允许方法修正B报告，禁止以诊断阳性授予A或启动CFA。
- 合成端到端测试验证CSV、模型、bootstrap、配对、图表、Word来源、诊断pending以及“仅诊断校准不授予A”的门控。

完整数值与最初失败均保留在 `v5_validation_initial.json`、`v5_validation_final.json`；它们是模拟结果，不是DROZY正式研究结论。科学验收没有通过，现阶段仍需修正潜因子估计/测量策略。

## 正式 EFA 执行（DROZY 真实数据）

日期：2026-11-09。运行目录：`results/dual_factor_efa_v5/20261109T154643Z-aaffec43`。

### 运行配置

- 入口：`python scripts/run_efa_v5.py --validation docs/dual_factor/v5_validation_final.json`
- 模拟证书有效：`code_sha256=948beacc...`、`config_sha256=0f26a15d...`、`diagnostic_checks_passed=True`、`latent_power_validated=False`
- `implementation_validated=True`，`latent_power=False`；正式运行（非 diagnostic-only）
- 14被试、36记录、2088生成窗、2044有效窗；invalid_window_rate=0.021 < 0.10

### 关键结果

| 门控项 | 结果 | 说明 |
|--------|------|------|
| **决策** | **B** | 候选4/2结构未被潜因子分析支持 |
| **证据等级** | **L0** | 设计/实现级，不可升级 |
| **phase_pass** | False | 所有 six_2 模型 PAF 失败（Heywood case） |
| **dimensions_ok** | True | iid 平行分析三相位均建议 n=2 |
| **C5_pass** | False | bootstrap PAF 失败：Heywood case |
| **observable_signal_detected** | False | F1 组内相关 0.303 < 0.5 |
| **latent_power_validated** | False | 模拟验证预期 |
| **method_correction_required** | True | 需修正潜因子估计/测量策略 |

### PAF 失败模式

真实数据上所有模型均失败，两种失败类型：
1. **Heywood case**（多数）：SE-GTE 相关 ~0.89-0.91，迭代中 GTE 共同度漂移超过 1.0
2. **不收敛**（少数 six_1/six_3）：500 次迭代后误差未达 1e-6

根因与模拟诊断一致：F2 仅 2 个指标（SE, GTE），4/2 结构固有不可识别。

### 平行分析参照

六指标 iid 平行分析三相位均建议 2 因子（实际特征值 > iid P95 参照）；记录中心化循环位移也建议 2 因子。维度建议正确，但 PAF 估计失败导致无法验收。

### 教师验收状态

`teacher_accepted=False`。决策 B 报告不可自动升级为 A。CFA、预测、外部和不变性阶段均未执行。
