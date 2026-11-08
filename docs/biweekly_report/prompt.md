# 双周汇报生成规则

仅使用已完成run目录的facts.json、manifest.json及对应结果文件。每个数字注明来源；不预估、夸大或编造实验结果。

必须呈现：当前候选结构及4/2名单、数据与质量分母、方法及实际运行状态、完整载荷和区间的来源、被试级稳健性、方向效度、局限、后续阶段门与需协调事项。

六指标为FD/PSV/SA/FN与SE/GTE，使用分布—转移失序假设成分。原始测量未验证、实验未执行、检验未通过均如实写明。固定提取两因子不等于发现两个因子。

诊断运行必须注明不能作为科学验收。CFA、预测和外部验证没有结果就写not_run，不填0或PASS。既有Word和旧结果不得覆盖。

生成命令：.venv/bin/python scripts/gen_biweekly_report.py --run-dir /absolute/path/to/run --output /absolute/path/to/new-report.docx

正文宋体小四、标题黑体、1.5倍行距、三线表。只可改善文字表达，不能改善或隐藏真实数据。
