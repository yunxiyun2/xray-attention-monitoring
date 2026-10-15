"""决策报告生成器。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionReport:
    verdict: str            # separable | unseparable | ambiguous
    n_factors: int
    factor_corr: float | None
    max_loading: float
    cross_loading_max: float
    variance_explained: float
    reasoning: str

    def to_markdown(self) -> str:
        corr_str = f"{self.factor_corr:.3f}" if self.factor_corr is not None else "N/A (单因子)"
        return f"""## 决策报告

**判定结论**：{self.verdict}

| 项目 | 数值 |
|---|---|
| 提取因子数 | {self.n_factors} |
| 因子相关 r | {corr_str} |
| 最大载荷 | {self.max_loading:.3f} |
| 最大交叉载荷 | {self.cross_loading_max:.3f} |
| 累计方差解释 | {self.variance_explained:.3f} |

**判定依据**：{self.reasoning}
"""


def make_decision(
    n_factors: int,
    factor_corr: float | None,
    max_loading: float,
    cross_loading_max: float,
    variance_explained: float,
) -> DecisionReport:
    """按 spec §5.4 决策判据判定双因子可分离性。"""
    if n_factors == 1:
        return DecisionReport(
            verdict="unseparable", n_factors=1, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            reasoning="仅提取一个因子，双因子假设被推翻，转向单维度模型。",
        )
    if factor_corr is None:
        return DecisionReport(
            verdict="ambiguous", n_factors=n_factors, factor_corr=None,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            reasoning="因子相关矩阵不可用（正交旋转），无法判定可分离性。",
        )
    if factor_corr < 0.70 and max_loading > 0.50:
        return DecisionReport(
            verdict="separable", n_factors=n_factors, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            reasoning=f"两因子相关 r < 0.70（r={factor_corr:.3f}）且载荷清晰（max={max_loading:.3f}），"
                      "双因子可分离，假设初步成立，进入 CFA 验证。",
        )
    if factor_corr > 0.85:
        return DecisionReport(
            verdict="unseparable", n_factors=n_factors, factor_corr=factor_corr,
            max_loading=max_loading, cross_loading_max=cross_loading_max,
            variance_explained=variance_explained,
            reasoning=f"两因子相关 r > 0.85（r={factor_corr:.3f}），因子高度共线，"
                      "双因子不可分离，转向单维度模型。",
        )
    return DecisionReport(
        verdict="ambiguous", n_factors=n_factors, factor_corr=factor_corr,
        max_loading=max_loading, cross_loading_max=cross_loading_max,
        variance_explained=variance_explained,
        reasoning=f"因子相关 0.70 ≤ r ≤ 0.85（r={factor_corr:.3f}），结构模糊，"
                  "检查指标计算或考虑剔除低载荷指标。",
    )
