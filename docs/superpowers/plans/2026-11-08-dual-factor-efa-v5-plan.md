# Plan v5.0：反馈闭环与有效实验实施计划

> 日期：2026-11-08。
> 状态：计划完成；下面的代码、实验和汇报生成尚未执行。
> 唯一规格：[SPEC v5](../specs/2026-11-08-dual-factor-efa-v5.md)。
> 来源：[教师最新反馈](../../biweekly_report/fankuiyijian/邓云溪-双周汇报反馈意见-1104.docx)。
> 当前只推进 EFA 决策报告。后续预测、CFA、外部验证和不变性任务是完整设计的待解锁部分，不能因列入计划就自动执行。

## 1. 实施原则和任务状态

- 历史 Word、v1—v4 SPEC/Plan 和 `results/dual_factor_efa/` 保持原样，不能全局替换历史措辞制造“从来没有错误”的记录。
- 当前维护文档、模板及新结果使用候选、4/2、分布—转移失序口径。
- 修代码与运行科学实验分开；Mac 可以完成代码和合成测试，EDF 与正式统计结果在具备输入/依赖的节点执行。
- 每个任务状态只允许 `pending/implemented/tested/executed/accepted/blocked`；`tested`不能代替`executed`，`executed`不能代替教师`accepted`。
- 每项任务有独立证据路径；失败也保存。报告不能引用“计划将得到的数字”。
- 不修改 `src/xray_attention/` 旧安检实现；只按需借鉴其接口/评估思想，不继承旧标签方向、池化CI或深度模型。

实施记录：`docs/dual_factor/v5_execution_log.md`，含任务、依赖、节点、代码hash、测试、run、状态、阻塞和下一步；此文件在开始实施时创建。

## 2. 依赖总图

```text
T01口径/事实 → T02协议/配置 → T03输入审计 → T04非重叠/权重
  → T05真PAF → T06维度参照 → T07完整对照/诊断阶梯
  → T08整人bootstrap → T09逐人/LOSO辅助

T03+T04 → T10配对方向/校正 → T11测量诊断
T05—T11 → T12合成有效性 → T13纯EFA入口 → T14正式统计复验
T14 → T15决策报告 → T16论文辅线 → T17汇报来源/模板 → T18验收交接

真实EFA A被教师验收 → T19—T21 CFA
真实EFA A被教师验收 → T22—T24预测（单独阶段，不与EFA抢跑）
CFA合格 → T25外部验证 → T26不变性
```

T11 的 raw 子任务依赖 EDF；没有 EDF 时可完成 matrix 诊断并明确未验证范围，不能将原始信号质量标为通过。T14 可产出带范围限制的 matrix 分析；是否足够验收由完整证据和教师决定。

## 3. 当前实施任务

### T01：锁定当前科学口径和历史事实

**依赖**：无。**执行**：Mac。

文件：修改 `docs/dual_factor/direction_lock.md`、`data_dictionary.md`、`paper_outline_and_literature_gap.md`；新增 `docs/dual_factor/v5_execution_log.md`。

操作：当前入口指向SPEC v5；明确4/2名单，F2为分布—转移失序，F1/F2是待检验假设；当前待EFA验收。对旧L2/L3及A判定标注历史规则/待修正核验，不改历史结果本体。记录1103/1104文件日期区别。列出2088/2044/14/36及输入hash，说明不是v5结果。

验收：AC01、AC14；当前文档没有3/3或“已验证双因子”表述；历史错误引用有范围和来源。

### T02：协议、分组、配置和运行状态

**依赖**：T01。**执行**：Mac。

新增：`src/dual_factor/protocol.py`、`config.py`、`run_manifest.py`、`configs/dual_factor_efa_v5.yaml`、`tests/dual_factor/test_config_v5.py`、`test_stage_gates.py`。

定义命名分组、指标方向/单位、全部SPEC参数、C0—C5及后续CFA/预测阈值。配置schema拒绝未知键、非法范围、缺关键字段；每键有消费点。所有输入/输出路径相对显式project_root解析。run_id和目录已存在即失败，不覆盖；随机数流按stage/replicate派生。

状态包含 `analysis_status, scientific_decision, evidence_level, hypothesis_validity, teacher_acceptance`；默认CFA/预测/external/invariance关闭。教师accepted必须与具体报告hash对应，不根据supported自动设置。

修改 `pyproject.toml` 的可选依赖组：统计、raw、CFA、report分开。创建独立CPU统计环境说明，安装后输出精确版本lock；环境名只是建议，不假设任何节点已有Conda。raw模块/CFA依赖延迟导入，matrix审计不强制需要pyedflib/semopy/torch。

验收：AC10、AC16；从不同cwd dry-run的路径一致；仅变阈值配置结果判定随之变化；缺验收来源时CFA入口拒绝；只配置未实现的字段不能静默忽略。

### T03：输入迁移、时间及质量审计

**依赖**：T02。**执行**：Mac；raw部分需数据节点。

新增 `src/dual_factor/data/matrix_io.py`、`audit.py`、`tests/dual_factor/test_matrix_io.py`、`test_data_audit_v5.py`；修改 `drozy_loader.py`、`features/windowing.py`。

迁移历史subject/test/KSS/时间；重建时间必须有原配置证明；记录和窗主键强校验。保持八项raw值；当前CSV无事件数则NA。新增时间区间、每窗事件数、实际P70、invalid_reason、record metadata。有效行nonfinite、零方差等显式失败。

loader从配置读取subjects/exclusions/KSS路径，逐通道校验单位/fs/长度；非同步任务允许缺Cam-Sync。输出文件存在/排除原因和KSS来源清单。质量报告的分母是全部生成窗口，分别列每指标、记录、人、状态。

验收：AC10；重复主键、不同KSS、错误配置时间、采样率不一致均被识别；2088/2044计数只作为当前历史输入一致性检查，raw新版本不以该数强行约束。

### T04：真非重叠样本、标准化与公平权重

**依赖**：T03。**执行**：Mac/CPU。

新增 `src/dual_factor/analysis/sampling.py`、`weighting.py`；修改 `sensitivity.py`；新增 `test_nonoverlap_v5.py`、`test_weighting_v5.py`。

先按recording_id和相位选区间，再过滤valid；每记录断言区间不重叠。三相位分别输出。统一八/六对照完整案例；六项额外样本作为敏感性。标准化只用于结构，被试内均值/SD也用记录等权权重；每人/每记录等权，协方差/相关矩阵需正定。

禁用全程z-score均值聚合EFA主入口；历史接口显式标deprecated且返回输入退化原因。不能改成“均值加1e-8噪声”。

关键测试：3条记录编号均从0开始、乱序、删除中间窗、无效窗不回填、窗口与步长非整倍数、三相位不能拼接。不同记录窗数扩大但每人权重总和不变；构造条件变化保留Delta而全程均值为零。

验收：AC03、AC05；所有record interval audit零重叠，source_subject权重检查可复算。

### T05：实现真正PAF和斜交统计量

**依赖**：T04。**执行**：Mac/CPU。

新增 `src/dual_factor/analysis/paf.py`；修改 `efa.py`；新增 `test_paf.py`、`test_efa_metrics_v5.py`。

按SPEC E2.1实现SMC迭代PAF，明确tol/max_iter和失败；Promax旋转使用经过校验的库接口。保留principal/minres显式方法，`pa`历史别名只用于历史replay，v5使用`paf`。method字段不得误标。

保存pattern/structure/Phi、共同度、独特方差、重建R、残差和收敛轨迹；C1用 `trace(L Phi L.T)/p`，历史平方和另命名。因子交换时同时重排Phi；所有准则按配置。

验证：已知载荷矩阵、零/单因子、非正定/退化、Heywood与收敛失败；与独立PAF实现/psych参考比较重建矩阵。PAF与SVD在含独特方差的合成输入上应有方法差异，避免“测试只验证两者都返回数组”。

验收：AC06；独立参考差异≤SPEC容差，有明确方法记录；未通过则不运行真实EFA。

### T06：可解释的维度参照及全部PA谱

**依赖**：T05。**执行**：CPU。

修改 `efa.py`；新增 `src/dual_factor/analysis/parallel_analysis.py`、`test_parallel_analysis_v5.py`。

传统PA固定随机/权重/完整谱，允许0维；图注说明iid null和PA/PAF不同。依赖保留参照在record内连续有效段去均值、各指标独立循环位移；真实和null使用同一中心化、区段与权重。

保存每段长度、偏移、有效人数及谱；≥8窗且≥10名贡献人为可评估门；传统/中心化参照不可混用，不自动继承传统结论。增加同一权重尺度下的记录内/记录均值协方差分解，数值验证两者之和等于总协方差；去均值不支持不自动推导“结构只由记录均值造成”。零信号、同一因子、AR依赖合成对照测试；随机段不能跨test或丢失缺窗边界。

验收：AC02、AC03；图表可追溯null、S/R/N；不可评估不是PASS；不能生成预填“PA已解决伪重复”的结论。

### T07：完整模型对照与有限诊断阶梯

**依赖**：T05、T06。**执行**：CPU。

修改 `correction.py`、`efa.py`；新增 `src/dual_factor/analysis/model_comparison.py`、`test_model_comparison_v5.py`。

生成八/六×2/3、六×1，三相位和重叠参考；每model即使失败也有status、原因、输入hash。全载荷而不是只报告最大值；pattern/structure/Phi和共同度并报。

历史阶梯按原方法回放；真PAF规则只诊断，不把新删选替代六项。因子共线优先终止、并列按固定指标顺序、下限检查实际模型归属及预定组数量，两套全rung/PA结果独立落盘。

验收：AC02；不会把第三因子预命名为SV/SDR；不会为了通过C1再自动改主候选。

### T08：真正被试cluster bootstrap

**依赖**：T04、T05、T07。**执行**：CPU。

新增 `src/dual_factor/analysis/cluster_bootstrap.py`、`factor_alignment.py`；新增 `test_cluster_bootstrap.py`、`test_factor_alignment.py`。

整人抽样B=2000，保存source_subject和cluster副本；按副本权重重算相关和固定六项PAF。仅带符号置换对齐；记录失败、每重复载荷/相关/共同方差/C0—C4。实现SPEC C5全部概率/CI/匹配门槛和MC误差。

输出每重复表及汇总；失败计分母；不能补抽到2000次成功。bootstrap选择条件性范围在报告字段中强制显示。

关键测试：重复抽人不合并、记录/时间/多变量一起带入、失败不消失、factor交换/符号后不误判归属、某指标交叉CI越线导致C5失败、零均值聚合不参与。

验收：AC03；可从重复表独立计算比例/CI；没有整人输入时不能改用逐窗bootstrap。

### T09：诚实的LOSO及逐被试辅助证据

**依赖**：T07、T08。**执行**：CPU。

修改 `sensitivity.py`、`scripts/run_per_subject_efa.py`；新增 `test_subject_structure_v5.py`。

分离 `requested_n_factors, pa_suggested_n_factors, fit_success, assignment_match, hypothesis_sign_match`。记录每人的实际载荷、相关、交叉、样本及失败。报告三相位点估计越线及LOSO冲突，不把固定k=2解释成发现。

归属一致≠疲劳方向一致；如8/14归属一致、4人交叉超标必须显示，不能只生成“14人全部稳定”。

验收：AC07；合成单因子也可请求2，但报告不能称其发现两因子。

### T10：配对效度和正确的校正列

**依赖**：T03、T04、T07。**执行**：CPU。

新增 `src/dual_factor/analysis/paired_validity.py`；修改 `descriptive.py`、`construct_validity.py`；新增 `test_paired_validity.py`、`test_correction_columns_v5.py`。

原指标按人/记录/状态等权，只有同时拥有高低KSS的人做Delta；保存10人名单和未配对原因。2000次整人CI、全部符号翻转、六项Holm、supported/opposite/inconclusive。KSS5只从二分类排除。

column_map显式区分raw/z/norm；校正后仍z不变应如实报告，不称“方向被修复”。原窗口U检验保持历史模式但不产生主效度p。PVT缺失/单位/刺激分母有契约，不从KSS造PVT。

关键测试：增加同一人窗数不改变投票权；两类缺一被试不补成两类；全程均值0但Delta可恢复；标签5有EFA样本但无二分类标签；SDR_KL定义与方向不冲突。

验收：AC04、AC08；结果不符预期时中性命名，不自动翻号。

### T11：测量定义、事件算法与共变诊断

**依赖**：T03、T10。**执行**：matrix诊断CPU；信号/计数部分需EDF。

新增 `src/dual_factor/analysis/measurement_diagnostics.py`、`scripts/run_measurement_diagnostics.py`、`test_measurement_diagnostics.py`；修改 `features/indicators.py`、`event_detection.py`、`data/preprocess.py`的说明和版本记录。

明确GTE是转移对联合熵、SDR是KL、SV是协方差trace而非马氏距离；短事件是剔除，不称合并；P70窗内阈值与窗内minmax网格的局限。

matrix可跑FD/FN关系、PSV/SA关系、norm/z不变量；raw跑计数控制、5注视固定计数熵、条件转移熵附录、P60/P80和bins3/bins7单项敏感性、波形叠图与分层盲复核。

Hampel与朴素局部MAD参考不一致时区分legacy/v5信号算法，重提取到新目录；不能在matrix统计复算中悄改原始特征。新特征名称/定义变更不覆盖旧CSV。

验收：AC09；有定义、样本、偏倚及不可执行范围。参数扰动推翻归属时进入B/边界，不取最好版本。

### T12：已知真值模拟和反例回归

实施补充：完整首轮模拟在全部7类场景均不产生潜因子A，说明不能以低误报替代检验效能。实现另测“可观测共变信号检测”，与潜因子合法性/接受率分开。完整诊断校准通过只允许B方法修正报告；潜因子强正对照≥0.80仍是A的必要门，PAF/C5阈值不放宽。补充诊断并非原始预注册，失败记录和修订原因必须保留。

**依赖**：T05—T11。**执行**：CPU；Mac先准备，具备依赖后运行。

新增 `tests/dual_factor/test_v5_regressions.py`、`tests/simulation/test_structural_protocol.py`、`scripts/validate_v5_simulation.py`。

按SPEC §5确定性反例和200数据集/场景评估；生成10秒基础过程并构成30秒重叠观测，提供三个真实可选相位；强单/两因子、高相关、交叉、异质人、AR和非正态输入，B=200验证工作判据。真实数据B=2000。固定配置、种子、Monte Carlo/Wilson区间落盘；统计生成器不替代真实信号提取效度。

有效性门：误判A上界、强两因子识别率达到SPEC线；增加窗口不冒充增加人数。若不通过，先定位实现问题或前瞻修改假设，禁止先看真实v5结果再优化模拟。

验收：AC03、AC06；测试证明实际统计行为，不只是接口形状。没有执行模拟时manifest标not_verified。

### T13：纯EFA入口、依赖隔离和可复现运行

**依赖**：T02—T12。**执行**：Mac/CPU。

修改 `src/dual_factor/pipeline.py`；新增 `scripts/run_efa_v5.py`、`test_pipeline_v5.py`；更新 `README.md`当前入口说明。

入口读取严格配置，先audit、dry-run、方法有效性，再执行E0—E5；保存manifest/阶段checkpoint/输入输出hash。无raw时显示哪些诊断缺失；CFA模块延迟导入。所有阶段运行结果保存，不覆盖旧目录。

计划提供的命令（本文件编写时尚不可用）：

```bash
python scripts/run_efa_v5.py --config configs/dual_factor_efa_v5.yaml --dry-run
python scripts/validate_v5_simulation.py --config configs/dual_factor_efa_v5.yaml
python scripts/run_efa_v5.py --config configs/dual_factor_efa_v5.yaml --input-mode matrix
python scripts/run_efa_v5.py --config configs/dual_factor_efa_v5.yaml --input-mode raw --drozy-root /absolute/path/DROZY
```

相同inputs/config/code/seed的统计输出在容差内一致；恢复run只能沿用匹配manifest，不能在旧run换参数。输入或配置hash变化创建新run。采集操作不在本计划范围。

验收：AC10、AC16；绝对/相对路径一致，无semopy也能audit；已有run不能覆盖；不满足门时输出pending/B而不是直接跑CFA。

### T14：正式统计复验和数据来源复核

**依赖**：T12有效性通过、T13测试通过。**执行**：CPU统计环境；raw诊断需数据节点。

先matrix完整run，复算真非重叠/真PAF/cluster/配对/计数可用诊断，不复制旧结果。保存确切环境lock和运行命令。

具备EDF后raw复验信号特征和计数敏感性；matrix/raw输入版本不同分别出run，不合并为一个报告无差别结论。核查新窗口数变化、无效率、排除原因和KSS分布；无法解释差异时不验收。

证据：全部SPEC §10.2产物；逐模型/重复失败记录；旧/新数字对照，说明差异来自抽样、权重、提取法、方差定义还是特征重提取。

验收：AC02—10；完成运行不保证A。无数据/依赖阻塞必须写清具体输入与任务状态。

### T15：三选一决策报告和可核对事实表

**依赖**：T14。**执行**：Mac/CPU。

新增 `src/dual_factor/analysis/decision_v5.py`、`reporting.py`、`test_decision_v5.py`、`test_report_facts_v5.py`；更新旧 `decision.py`只作legacy入口说明。

按C0—C5、维度冲突和方法/质量给A/B/C或pending；H-V另判定；禁止C5不通过但写“支持且稳定”。生成九节决策报告、facts.json、decision.json，每事实来源字段可追溯。

边界测试：只C1—C4过、bootstrap失败多、相位不一致、宽CI、PA一维与数值质量错误应走不同分支；负结果仍报告完整；缺任一必需产物为pending；未执行CFA/预测不被格式化为0。

验收：AC01、AC02、AC03、AC15；文档完整与科学结果通过分别有状态。

### T16：论文结构与文献缺口辅线

**依赖**：T01；结论段依赖T15。**执行**：Mac。

修改 `paper_outline_and_literature_gap.md`、`literature_gaps.md`。主张—假设—实验—判据—证据级别逐一对应；保留失败的疲劳方向和预测增量作为可报告结果。

不把有DOI当已读原文；核实未确认的Dorn/Klett和教师文档1—3出处，没有原文就标未核实。删除无系统检索依据的“首次/无人”。不凭猜测填写OperatorEYEVP字段。

验收：AC14；辅线不启动新算法、亚类聚类或论文未证结果。

### T17：汇报模板、错误源和Word生成

**依赖**：T15、T16。**执行**：Mac；report依赖按需安装。

修改 `docs/biweekly_report/prompt.md`、`scripts/gen_biweekly_report.py`；新增 `test_biweekly_report_v5.py`。

去除编造、夸大或隐瞒问题的指令。facts/run为唯一结果源；日期、4/2、Hampel51、P70、完整载荷、无效率分母、配对人数、cluster CI、明确阶段状态进入正文。新报告文件名使用实际生成日期，不覆盖旧Word。

审计生成Word XML：不能出现3/3、窗口7、14人“发现两因子”、无依据“校正有效”或未执行CFA/预测通过。需要引用旧数则标历史版本。数值与事实表一致、NA明确、表格来源可查。

验收：AC01、AC10、AC15；模板修好但没有新run时只生成设计/待执行汇报，不生成实验成功报告。

### T18：EFA验收包与恢复交接

**依赖**：T15—T17。**执行**：Mac。

产出 `review_checklist.md`、当前run的 `handoff.md`。逐条AC01—10、14—16核对，指出raw/独立效度未验证范围；列全部失败与科学决策。

如果A，提交具体报告供教师验收；只有实际验收来源才写对应run/hash的approval.json。B/C则给有限修正/转结构建议，保持后续门关闭。无需为提交本地验收材料创建外部消息或自动发送。

验收：每个AC有实证链接或not_run原因；交接包含当前节点、环境、数据、下一可执行任务、避免重跑的已有run信息。报告生成器不能替教师设置accepted。

## 4. 后续待解锁任务

### T19：修正CFA参数、SRMR和模型识别

**依赖**：真实EFA A且被教师验收。**执行**：CPU。

修改 `analysis/cfa.py`、`scripts/run_cfa.py`、`tests/dual_factor/test_cfa.py`；新增 `test_cfa_identification.py`。

按名组装4/2；标准载荷/真相关按潜变量方差计算。SRMR同一S尺度含对角下三角，与参考SEM核对。M0无残差为主，M1为历史事后修正敏感性。Jacobian秩、信息条件数、SE、残差/潜变量方差和Heywood先验证。

验收：AC11；近零Phi的两指标因子未识别时失败，不用强加等载荷/残差消除失败。

### T20：CFA依赖处理和推断能力核查

**依赖**：T19。**执行**：CPU。

新增 `analysis/sem_backend.py`、`test_sem_backend.py`；保存backend能力清单（non-normal、cluster、稳健chi2）。常规ML与人cluster参数CI分别命名；不能把semopy基础fit说成MLR。

若接入支持聚类稳健推断的backend，以同一模型/数据/统计定义作跨实现测试；否则明确常规拟合限制，不创造稳健RMSEA。

验收：AC11；所有指标阈值由配置生效，Mardia不豁免RMSEA；M1改善不替代M0。

### T21：内部CFA执行和证据分级

**依赖**：T19、T20。**执行**：CPU。

独立run输出M0/M1、标准载荷、Phi、识别诊断、bootstrap、逐人结果和失败；同DROZY分析明确内部一致性。只有主模型及可用推断满足SPEC才记L3条件，独立确认仍待外部数据；失败不升级或开启外部阶段。

验收：AC11；默认开关不自动打开后续；新相关与旧协方差字段不可混用。

### T22：无泄漏的三条件预测与强基线

**依赖**：真实EFA A被教师验收；当前只设计。**执行**：CPU。

新增 `evaluation/subject_loso.py`、`models/factor_transformer.py`、`experiments/factor_prediction/run.py`、`configs/factor_prediction_v5.yaml`、`test_factor_prediction_leakage.py`。

固定L2/C=1逻辑回归。P0总分、P1内层选单因子、P2双因子、P3六项、P4八项；inner-LOSO每折重新拟合尺度/PAF/回归得分权重，不能复用外层或全数据得分。总分/因子分数再用训练自身尺度标准化，以公平应用固定C。所有source_subject副本同折；训练失败明确记录，无全数据fallback。

外层每人留出，窗概率汇聚成记录概率；KSS5排除；返回每人/记录预测和训练来源清单。未能完成的外层折不静默删除；所有条件需共同完成全14外层预测才能声明完整LOSO比较，无法完成报告not_evaluable。

验收：AC12；改变测试标签/未来值不影响训练或过去校准；单类被试不造个体AUROC。

### T23：公平前缀基线校准和共同样本对照

**依赖**：T22。**执行**：CPU。

新增 `data/prefix_calibration.py`、`test_prefix_calibration.py`；按最早可用记录120秒/4非重叠窗锁定，无效不延长；仅减均值，SD训练人拟合；所有P0—P4共享校准和相同评价样本。

保存calibration窗口与评价窗口，断言时间无交叠；无校准共同样本再评价，比较双因子和校准各自贡献。没有校准资格不从KSS挑“更清醒”记录替代。

验收：AC12；无测试全程z、无校准标签选择、P2专享校准不能作为有效三条件结果。

### T24：匹配estimand的AUC/PR/校准与增量推断

**依赖**：T22、T23。**执行**：CPU。

新增 `evaluation/cluster_metrics.py`、`test_cluster_metrics.py`；输出10人的记录宏AUROC、14人的池化AUROC/AP/Brier/log loss/3箱校准、每人/记录标签概率。

2000次共享被试bootstrap，三项主比较98.33%配对CI和0.02增量线；内层最佳单因子选择不能用测试人；P4相同八项完整案例另对照。固定OOF CI范围与500次重拟合敏感性分开报告，不能把fold重叠当独立重复。

验收：AC12；宏点估计配宏CI，池化点配池化CI；两项压缩参考通过但六项未通过时正确限制表述；CI含0不叫等效/失败证明。

### T25：OperatorEYEVP适配和独立确认

**依赖**：CFA合格；当前仅整理待核实清单。**执行**：有授权数据的节点。

新增 `docs/dual_factor/external_validation_protocol.md`、对应loader/config及schema测试。核实字段、许可、任务、标签、坐标/速度单位、实际人数；六项不匹配则在看结果前提出替代候选，记录原因。

冻结六项模型、阈值、预处理同义规则；外部自身CFA/模式一致性，DROZY冻结模型预测迁移另报。抽样仍以人，模拟精度后执行，不合并EOG电位与角速度当同一raw量。

验收：AC13；不能先看结果再选数据集/指标；外部低功效和结构失败有诚实边界。

### T26：重复测量适用的不变性检验

**依赖**：T25结构重复通过。**执行**：支持对应SEM模型的CPU节点。

新增 `analysis/invariance.py`、`test_invariance.py`。跨数据集先核实测量可比；KSS两组包含同一人时用重复测量/cluster-aware模型，不能普通独立多群组。按configural/metric/scalar顺序，固定残差、ΔCFI/ΔRMSEA参考线及参数合法性。

验收：AC13；不变性没过不比较潜均值；backend不支持时not_evaluable，不强行在窗级生成显著结果。

## 5. 问题—任务—验收映射

| 问题 | 主任务 | 必需验收 |
|---|---|---|
| R01 4/2与候选口径 | T01、T02、T17 | AC01 |
| R02 完整对照 | T06、T07、T15 | AC02 |
| R03 被试稳健性 | T08、T09、T12 | AC03 |
| R04 零均值聚合 | T04、T10 | AC03、AC04 |
| R05 非重叠错误 | T04 | AC05 |
| R06 PAF方法错位 | T05、T12 | AC06 |
| R07 固定k误读 | T09、T17 | AC07 |
| R08 校正列错误 | T10 | AC08 |
| R09 构念/方向边界 | T10、T11 | AC04、AC09 |
| R10 复现/质量 | T02、T03、T13 | AC10 |
| R11 CFA统计量 | T19、T20 | AC11 |
| R12 CFA识别/事后修改 | T19—T21 | AC11 |
| R13 分类门控冲突 | T02、T18、T22 | AC12、AC16 |
| R14 强基线/外部 | T22—T26 | AC12、AC13 |
| R15 论文/文献 | T16 | AC14 |
| R16 编造源/数字硬编码 | T15、T17 | AC15 |
| R17 参数/路径/覆盖 | T02、T13 | AC10、AC16 |
| R18 伪重复/CI错位 | T06、T08、T10、T24 | AC03、AC04、AC12 |
| R19 GTE定义 | T11 | AC09 |
| R20 算法共变 | T11、T12 | AC09 |

## 6. 完成定义和风险处理

本次文档完成：SPEC全部R01—R20有方法和失败分支，Plan全部AC01—AC16有文件、任务和检查；两份链接、参数和阶段门一致。

实施完成：T01—T13代码与对应测试完成，版本和依赖可复现。不能由“写了函数”升级为实验完成。

本周期科学交付完成：T14—T18有真实run、完整决策和范围限制，负结果也是完成；教师验收是否通过独立记录。

关键风险及行动：

- 14人/10配对人低功效：保留区间和B，不改人数或阈值；bootstrap次数不增加证据人数。
- 真PAF/真共同方差改变结论：新估计全部重算，不保持旧0.666或强说方法等价。
- 六指标方向相反：H-V如实否定/不确定，成分中性命名；H-S单独报告。
- 两指标CFA弱识别：停止L3升级；观察合成分数不是潜因子通过。
- raw缺失：matrix能解决统计问题但不能证明测量有效，checkpoint记录等待EDF任务。
- 外部字段/许可未知：先适配协议，不提前承诺泛化结果或取数权限。
- 报告模板再引入旧错误：fact来源检查、Word XML检查和阶段状态回归一起验收。

当前唯一下一执行入口是T01，而不是训练模型或启动CFA。所有后续任务以SPEC的真实阶段门为准。
