"""决策报告生成器：严格判据执行与三选一决策。

预注册判据（不可事后修改）：
  C1: 累计方差解释率 > 60%
  C2: 最大因子载荷 > 0.50
  C3: 最大交叉载荷 < 0.30  ← 禁止判定"可分离"的关键判据
  C4: 两因子相关 < 0.70
  C5: 被试级聚合结论一致（由 sensitivity 模块传入）

强制规则：C1-C4 全部达标方可判定"双因子可分离"。
         任一未达标，必须如实报告，不得使用"待收敛"等模糊表述。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CriterionCheck:
    """单条判据检查结果。"""
    code: str          # C1-C5
    name: str          # 判据名称
    value: float       # 实测值
    threshold: str     # 阈值描述
    passed: bool       # 是否达标
    note: str = ""     # 说明


@dataclass
class DecisionReport:
    verdict: str            # supported | unsupported | needs_correction | ambiguous
    verdict_label: str      # 中文标签
    n_factors: int
    factor_corr: float | None
    max_loading: float
    cross_loading_max: float
    variance_explained: float
    criteria: list[CriterionCheck] = field(default_factory=list)
    reasoning: str = ""
    next_step: str = ""

    def to_markdown(self) -> str:
        corr_str = f"{self.factor_corr:.3f}" if self.factor_corr is not None else "N/A (单因子)"
        lines = ["## EFA 决策报告\n"]
        lines.append(f"**判定结论**：{self.verdict_label}（{self.verdict}）\n")
        lines.append("| 项目 | 数值 |")
        lines.append("|---|---|")
        lines.append(f"| 提取因子数 | {self.n_factors} |")
        lines.append(f"| 因子相关 r | {corr_str} |")
        lines.append(f"| 最大载荷 | {self.max_loading:.3f} |")
        lines.append(f"| 最大交叉载荷 | {self.cross_loading_max:.3f} |")
        lines.append(f"| 累计方差解释 | {self.variance_explained:.1%} |")
        lines.append("")
        lines.append("### 预注册判据逐项检查\n")
        lines.append("| 判据 | 名称 | 实测值 | 阈值 | 达标 |")
        lines.append("|---|---|---|---|---|")
        for c in self.criteria:
            mark = "✅" if c.passed else "❌"
            lines.append(f"| {c.code} | {c.name} | {c.value:.3f} | {c.threshold} | {mark} |")
        lines.append("")
        all_pass = all(c.passed for c in self.criteria)
        lines.append(f"**全部达标**：{'是' if all_pass else '否'}\n")
        lines.append(f"**判定依据**：{self.reasoning}\n")
        lines.append(f"**下一步**：{self.next_step}\n")
        return "\n".join(lines)


def make_decision(
    n_factors: int,
    factor_corr: float | None,
    max_loading: float,
    cross_loading_max: float,
    variance_explained: float,
    subject_level_consistent: bool | None = None,
) -> DecisionReport:
    """按预注册判据严格判定双因子可分离性。

    参数：
        n_factors: 平行分析/自动确定的因子数
        factor_corr: 两因子相关（单因子时为 None）
        max_loading: 最大因子载荷
        cross_loading_max: 最大交叉载荷
        variance_explained: 累计方差解释率（0-1）
        subject_level_consistent: 被试级聚合结论是否与窗口级一致（C5）

    返回 DecisionReport，包含每条判据的达标/未达标状态。
    """
    criteria: list[CriterionCheck] = []

    # C1: 累计方差 > 60%
    c1 = CriterionCheck(
        code="C1", name="累计方差解释率",
        value=variance_explained, threshold="> 0.60",
        passed=variance_explained > 0.60,
        note=f"实测 {variance_explained:.1%}",
    )
    criteria.append(c1)

    # C2: 最大载荷 > 0.50
    c2 = CriterionCheck(
        code="C2", name="最大因子载荷",
        value=max_loading, threshold="> 0.50",
        passed=max_loading > 0.50,
        note=f"实测 {max_loading:.3f}",
    )
    criteria.append(c2)

    # C3: 最大交叉载荷 < 0.30（关键判据）
    c3 = CriterionCheck(
        code="C3", name="最大交叉载荷",
        value=cross_loading_max, threshold="< 0.30",
        passed=cross_loading_max < 0.30,
        note=f"实测 {cross_loading_max:.3f}",
    )
    criteria.append(c3)

    # C4: 两因子相关 < 0.70
    if factor_corr is not None:
        c4 = CriterionCheck(
            code="C4", name="两因子相关",
            value=factor_corr, threshold="< 0.70",
            passed=abs(factor_corr) < 0.70,
            note=f"实测 r={factor_corr:.3f}",
        )
    else:
        c4 = CriterionCheck(
            code="C4", name="两因子相关",
            value=0.0, threshold="< 0.70",
            passed=False,
            note="因子相关矩阵不可用（正交旋转或单因子）",
        )
    criteria.append(c4)

    # C5: 被试级聚合结论一致
    if subject_level_consistent is not None:
        c5 = CriterionCheck(
            code="C5", name="被试级聚合一致性",
            value=1.0 if subject_level_consistent else 0.0,
            threshold="一致",
            passed=subject_level_consistent,
            note="被试级 EFA 因子数与窗口级一致" if subject_level_consistent
                 else "被试级 EFA 因子数与窗口级不一致",
        )
    else:
        c5 = CriterionCheck(
            code="C5", name="被试级聚合一致性",
            value=0.0, threshold="一致",
            passed=False,
            note="未执行敏感性分析",
        )
    criteria.append(c5)

    # === 三选一决策 ===
    all_c1_c4_pass = c1.passed and c2.passed and c3.passed and c4.passed

    # 情况 1：仅提取 1 个因子 → 转结构
    if n_factors == 1:
        return DecisionReport(
            verdict="unsupported", verdict_label="不支持双因子（转单维）",
            n_factors=n_factors, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            criteria=criteria,
            reasoning="平行分析仅支持 1 个因子，双因子假设被推翻。",
            next_step="转向单维度疲劳模型，不再做 CFA。",
        )

    # 情况 2：C4 不达标（r ≥ 0.70）→ 转结构
    if not c4.passed and factor_corr is not None and abs(factor_corr) >= 0.70:
        return DecisionReport(
            verdict="unsupported", verdict_label="不支持双因子（因子共线）",
            n_factors=n_factors, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            criteria=criteria,
            reasoning=f"两因子相关 r={factor_corr:.3f} ≥ 0.70，因子高度共线，不可分离。",
            next_step="转向单维度模型或合并因子后重新分析。",
        )

    # 情况 3：C1-C4 全部达标 → 支持双因子
    if all_c1_c4_pass:
        reasoning_parts = []
        reasoning_parts.append(f"两因子相关 r={factor_corr:.3f} < 0.70")
        reasoning_parts.append(f"最大载荷 {max_loading:.3f} > 0.50")
        reasoning_parts.append(f"交叉载荷 {cross_loading_max:.3f} < 0.30")
        reasoning_parts.append(f"累计方差 {variance_explained:.1%} > 60%")
        if c5.passed:
            reasoning_parts.append("被试级聚合结论一致")
            next_step = "进入 CFA 验证，建立验证性因子分析模型。"
        else:
            reasoning_parts.append("被试级聚合结论不一致，需在 CFA 中重点考察")
            next_step = "进入 CFA 验证，但需重点关注被试级结构稳定性。"
        return DecisionReport(
            verdict="supported", verdict_label="支持双因子可分离",
            n_factors=n_factors, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            criteria=criteria,
            reasoning="；".join(reasoning_parts) + "。预注册判据 C1-C4 全部达标，双因子可分离。",
            next_step=next_step,
        )

    # 情况 4：C3 未达标（交叉载荷 ≥ 0.30）→ 需修正
    if not c3.passed:
        return DecisionReport(
            verdict="needs_correction", verdict_label="不支持双因子（交叉载荷超标，需修正）",
            n_factors=n_factors, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            criteria=criteria,
            reasoning=f"最大交叉载荷 {cross_loading_max:.3f} ≥ 0.30，判据 C3 未达标。"
                      "禁止判定「可分离」。需检查哪些指标交叉载荷高，"
                      "考虑剔除或替换后重做 EFA。",
            next_step="识别高交叉载荷指标，剔除后重做 EFA；或检查 3 因子解是否更优。",
        )

    # 情况 5：其他未达标 → 需修正
    failed = [c.code for c in criteria[:4] if not c.passed]
    return DecisionReport(
        verdict="needs_correction", verdict_label=f"不支持双因子（判据 {','.join(failed)} 未达标）",
        n_factors=n_factors, factor_corr=factor_corr,
        max_loading=max_loading, cross_loading_max=cross_loading_max,
        variance_explained=variance_explained,
        criteria=criteria,
        reasoning=f"判据 {','.join(failed)} 未达标，双因子可分离性无法确认。"
                  "需修正指标或因子结构后重做 EFA。",
        next_step=f"针对 {','.join(failed)} 修正：剔除低载荷指标/检查因子相关/调整因子数。",
    )
