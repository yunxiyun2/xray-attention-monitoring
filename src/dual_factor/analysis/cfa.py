"""CFA：验证性因子分析（L3 证据等级）。

基于 EFA 保留的六指标两因子结构做验证性因子分析，检验拟合优度。

因子结构（来自 EFA 决策报告）：
  F1 驱动衰减（DriveDecay）: FD, PSV, SA, FN
  F2 空间失序（SpatialDisorg）: SE, GTE

判据（预注册）：
  CFI   > 0.90
  TLI   > 0.90（报告，参考阈值）
  RMSEA < 0.08
  SRMR  < 0.08
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import semopy

# 保留六指标分组（与 EFA 决策报告一致）
DRIVE_DECAY = ["FD", "PSV", "SA", "FN"]
SPATIAL_DISORG = ["SE", "GTE"]
RETAINED = DRIVE_DECAY + SPATIAL_DISORG

# CFA 判据阈值
CFI_THRESHOLD = 0.90
TLI_THRESHOLD = 0.90
RMSEA_THRESHOLD = 0.08
SRMR_THRESHOLD = 0.08


def _build_model_spec(
    indicator_cols: list[str],
    residual_corr: list[tuple[str, str]] | None = None,
) -> str:
    """构建 semopy 模型语法。

    Parameters
    ----------
    indicator_cols : 指标列名列表（含 _z 后缀），前4项属 F1，后2项属 F2。
    residual_corr : 额外残差相关对列表，如 [("PSV_z", "SA_z")]。
        用于添加理论上可 justified 的残差协方差（如眼动主序列关系）。

    Returns
    -------
    semopy 模型语法字符串。
    """
    f1_indicators = " + ".join(indicator_cols[:4])
    f2_indicators = " + ".join(indicator_cols[4:])
    lines = [
        f"DriveDecay =~ {f1_indicators}",
        f"SpatialDisorg =~ {f2_indicators}",
        "DriveDecay ~~ SpatialDisorg",
    ]
    if residual_corr:
        for a, b in residual_corr:
            lines.append(f"{a} ~~ {b}")
    return "\n".join(lines) + "\n"


@dataclass
class CFAResult:
    """CFA 拟合结果。"""
    model_spec: str
    n_obs: int
    # 拟合指数
    chi2: float
    dof: int
    chi2_p: float
    cfi: float
    tli: float
    rmsea: float
    srmr: float
    # 判据达标
    cfi_pass: bool
    tli_pass: bool
    rmsea_pass: bool
    srmr_pass: bool
    all_pass: bool
    # 参数估计
    loadings: dict[str, dict[str, float]]  # {因子: {指标: 载荷}}
    factor_corr: float
    # 逐指标标准化载荷（主载荷）
    main_loadings: dict[str, float]


def run_cfa(
    data: pd.DataFrame,
    indicator_cols: list[str] | None = None,
    residual_corr: list[tuple[str, str]] | None = None,
) -> CFAResult:
    """对标准化指标数据运行 CFA。

    Parameters
    ----------
    data : 包含指标列的 DataFrame。
    indicator_cols : 指标列名列表。前4项属驱动衰减因子，后2项属空间失序因子。
        默认为 RETAINED + "_z"。
    residual_corr : 额外残差相关对列表（如眼动主序列 PSV~~SA）。

    Returns
    -------
    CFAResult
    """
    if indicator_cols is None:
        indicator_cols = [f"{c}_z" for c in RETAINED]

    if len(indicator_cols) != 6:
        raise ValueError(f"需要恰好6项指标列，得到 {len(indicator_cols)}")

    # 检查列存在
    missing = [c for c in indicator_cols if c not in data.columns]
    if missing:
        raise ValueError(f"数据缺少列: {missing}")

    # 去除缺失行
    df = data[indicator_cols].dropna()
    n_obs = len(df)
    if n_obs < 20:
        raise ValueError(f"有效观测数过少: {n_obs}，CFA 需至少 20")

    # 构建模型
    spec = _build_model_spec(indicator_cols, residual_corr=residual_corr)
    model = semopy.Model(spec)
    model.fit(df)

    # 拟合指数（semopy calc_stats 返回单行 DataFrame，统计量为列名）
    stats = semopy.calc_stats(model)

    def _get_stat(name: str) -> float:
        if name in stats.columns:
            return float(stats.loc["Value", name])
        return float("nan")

    chi2 = _get_stat("chi2")
    dof = _get_stat("DoF")
    chi2_p = _get_stat("chi2 p-value")
    cfi = _get_stat("CFI")
    tli = _get_stat("TLI")
    rmsea = _get_stat("RMSEA")

    # SRMR：semopy 不内置，手动计算
    # 模型隐含协方差矩阵（使用模型矩阵，变量顺序与 model.mx_data 一致）
    implied_cov = model.mx_lambda @ model.mx_psi @ model.mx_lambda.T + model.mx_theta

    # 样本协方差矩阵（使用 model.mx_data 确保变量顺序与模型矩阵一致）
    samp_cov = np.cov(model.mx_data.T, ddof=1)

    # 标准化为相关矩阵后取残差
    def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)

    samp_corr = _corr_from_cov(samp_cov)
    impl_corr = _corr_from_cov(implied_cov)
    residual = samp_corr - impl_corr
    n_var = residual.shape[0]
    lower_idx = np.tril_indices(n_var, k=-1)
    srmr = float(np.sqrt(np.mean(residual[lower_idx] ** 2)))

    # 参数估计
    params = model.inspect()

    # 提取载荷
    loadings: dict[str, dict[str, float]] = {"DriveDecay": {}, "SpatialDisorg": {}}
    main_loadings: dict[str, float] = {}
    factor_corr = float("nan")

    for _, row in params.iterrows():
        if row["op"] == "~":
            # 载荷行: lval (指标) ~ rval (因子)
            indicator = str(row["lval"])
            factor = str(row["rval"])
            val = float(row["Estimate"])
            # semopy 标准化需要 semopy.standardize，这里取非标准化估计
            loadings[factor][indicator] = val
            main_loadings[indicator] = val
        elif row["op"] == "~~":
            # 协方差行
            if str(row["lval"]) == "DriveDecay" and str(row["rval"]) == "SpatialDisorg":
                factor_corr = float(row["Estimate"])

    # 判据检查
    cfi_pass = cfi > CFI_THRESHOLD if not np.isnan(cfi) else False
    tli_pass = tli > TLI_THRESHOLD if not np.isnan(tli) else False
    rmsea_pass = rmsea < RMSEA_THRESHOLD if not np.isnan(rmsea) else False
    srmr_pass = srmr < SRMR_THRESHOLD if not np.isnan(srmr) else False
    all_pass = cfi_pass and rmsea_pass and srmr_pass

    return CFAResult(
        model_spec=spec,
        n_obs=n_obs,
        chi2=chi2,
        dof=dof,
        chi2_p=chi2_p,
        cfi=cfi,
        tli=tli,
        rmsea=rmsea,
        srmr=srmr,
        cfi_pass=cfi_pass,
        tli_pass=tli_pass,
        rmsea_pass=rmsea_pass,
        srmr_pass=srmr_pass,
        all_pass=all_pass,
        loadings=loadings,
        factor_corr=factor_corr,
        main_loadings=main_loadings,
    )


def per_subject_cfa(
    df: pd.DataFrame,
    subject_col: str = "subject",
    indicator_cols: list[str] | None = None,
    residual_corr: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """逐被试 CFA：对每名被试的窗口数据单独做 CFA。

    Parameters
    ----------
    df : 包含被试列和指标列的 DataFrame。
    subject_col : 被试编号列名。
    indicator_cols : 指标列名列表。
    residual_corr : 额外残差相关对列表。

    Returns
    -------
    每名被试一行的 CFA 汇总 DataFrame。
    """
    if indicator_cols is None:
        indicator_cols = [f"{c}_z" for c in RETAINED]

    records = []
    for sid, sub_df in df.groupby(subject_col):
        sub_data = sub_df[indicator_cols].dropna()
        n = len(sub_data)
        if n < 20:
            records.append({
                "subject": sid,
                "n_windows": n,
                "status": "insufficient_data",
                "chi2": None, "dof": None, "chi2_p": None,
                "cfi": None, "tli": None, "rmsea": None, "srmr": None,
                "cfi_pass": False, "tli_pass": False,
                "rmsea_pass": False, "srmr_pass": False, "all_pass": False,
                "factor_corr": None,
            })
            continue
        try:
            res = run_cfa(sub_data, indicator_cols, residual_corr=residual_corr)
            records.append({
                "subject": sid,
                "n_windows": n,
                "status": "ok",
                "chi2": res.chi2,
                "dof": res.dof,
                "chi2_p": res.chi2_p,
                "cfi": res.cfi,
                "tli": res.tli,
                "rmsea": res.rmsea,
                "srmr": res.srmr,
                "cfi_pass": res.cfi_pass,
                "tli_pass": res.tli_pass,
                "rmsea_pass": res.rmsea_pass,
                "srmr_pass": res.srmr_pass,
                "all_pass": res.all_pass,
                "factor_corr": res.factor_corr,
            })
        except Exception as e:
            records.append({
                "subject": sid,
                "n_windows": n,
                "status": f"failed: {e}",
                "chi2": None, "dof": None, "chi2_p": None,
                "cfi": None, "tli": None, "rmsea": None, "srmr": None,
                "cfi_pass": False, "tli_pass": False,
                "rmsea_pass": False, "srmr_pass": False, "all_pass": False,
                "factor_corr": None,
            })
    return pd.DataFrame(records)


def cfa_to_dict(result: CFAResult) -> dict:
    """将 CFAResult 转为可序列化 dict。"""
    d = asdict(result)
    return d


def save_cfa_result(result: CFAResult, path: str | Path) -> None:
    """保存 CFA 结果为 JSON。"""
    d = cfa_to_dict(result)
    Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
