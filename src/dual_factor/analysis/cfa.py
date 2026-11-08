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
from dataclasses import dataclass, asdict, field
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
    indicator_groups: dict[str, list[str]] | None = None,
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
    if len(indicator_cols) != 6 or len(set(indicator_cols)) != 6:
        raise ValueError("需要恰好6项唯一指标")
    if indicator_groups is None:
        lookup = {c.removesuffix("_z"): c for c in indicator_cols}
        if set(lookup) != set(RETAINED):
            raise ValueError("Unknown indicators require explicit named indicator_groups")
        indicator_groups = {"DriveDecay": [lookup[c] for c in DRIVE_DECAY],
                            "SpatialDisorg": [lookup[c] for c in SPATIAL_DISORG]}
    if (set(indicator_groups) != {"DriveDecay", "SpatialDisorg"}
            or len(indicator_groups["DriveDecay"]) != 4 or len(indicator_groups["SpatialDisorg"]) != 2
            or sorted(indicator_groups["DriveDecay"] + indicator_groups["SpatialDisorg"]) != sorted(indicator_cols)):
        raise ValueError("Indicator groups must partition the named six indicators as 4/2")
    f1_indicators = " + ".join(indicator_groups["DriveDecay"])
    f2_indicators = " + ".join(indicator_groups["SpatialDisorg"])
    lines = [
        f"DriveDecay =~ {f1_indicators}",
        f"SpatialDisorg =~ {f2_indicators}",
        "DriveDecay ~~ SpatialDisorg",
    ]
    if residual_corr:
        for a, b in residual_corr:
            if a not in indicator_cols or b not in indicator_cols or a == b:
                raise ValueError("Invalid residual covariance pair")
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
    factor_covariance: float = float("nan")
    identification_valid: bool = False
    identification: dict = field(default_factory=dict)
    model_role: str = "internal_consistency"


def run_cfa(
    data: pd.DataFrame,
    indicator_cols: list[str] | None = None,
    residual_corr: list[tuple[str, str]] | None = None,
    thresholds: dict[str, float] | None = None,
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

    thresholds = thresholds or {"cfi": CFI_THRESHOLD, "tli": TLI_THRESHOLD,
                               "rmsea": RMSEA_THRESHOLD, "srmr": SRMR_THRESHOLD}
    if set(thresholds) != {"cfi", "tli", "rmsea", "srmr"} or any(not 0 < v < 1 for v in thresholds.values()):
        raise ValueError("Invalid CFA thresholds")
    # SRMR: standardize sample/model covariance by the SAME sample variances.
    # 模型隐含协方差矩阵（使用模型矩阵，变量顺序与 model.mx_data 一致）
    implied_cov = model.mx_lambda @ model.mx_psi @ model.mx_lambda.T + model.mx_theta

    # 样本协方差矩阵（使用 model.mx_data 确保变量顺序与模型矩阵一致）
    samp_cov = np.cov(model.mx_data.T, ddof=1)

    # 标准化为相关矩阵后取残差
    def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)

    sample_scale = np.sqrt(np.outer(np.diag(samp_cov), np.diag(samp_cov)))
    residual = (samp_cov - implied_cov) / sample_scale
    n_var = residual.shape[0]
    lower_idx = np.tril_indices(n_var, k=0)
    srmr = float(np.sqrt(np.mean(residual[lower_idx] ** 2)))

    # 参数估计
    params = model.inspect()

    # 提取载荷
    loadings: dict[str, dict[str, float]] = {"DriveDecay": {}, "SpatialDisorg": {}}
    main_loadings: dict[str, float] = {}
    factor_corr = float("nan")
    factor_covariance = float("nan")
    latent_variances = {}
    observed = list(model.names_lambda[0])

    for _, row in params.iterrows():
        if row["op"] == "~":
            # 载荷行: lval (指标) ~ rval (因子)
            indicator = str(row["lval"])
            factor = str(row["rval"])
            val = float(row["Estimate"])
            loadings[factor][indicator] = val
        elif row["op"] == "~~":
            # 协方差行
            left, right = str(row["lval"]), str(row["rval"])
            if {left, right} == {"DriveDecay", "SpatialDisorg"}:
                factor_covariance = float(row["Estimate"])
            if left == right and left in loadings:
                latent_variances[left] = float(row["Estimate"])

    if set(latent_variances) == set(loadings) and min(latent_variances.values()) > 0:
        factor_corr = factor_covariance / np.sqrt(latent_variances["DriveDecay"] * latent_variances["SpatialDisorg"])
        for factor, indicators in loadings.items():
            for indicator, estimate in indicators.items():
                main_loadings[indicator] = estimate * np.sqrt(latent_variances[factor]/implied_cov[observed.index(indicator), observed.index(indicator)])
    identification = {}
    try:
        _, (m, c) = model.calc_sigma()
        gradients = model.calc_sigma_grad(m, c)
        jacobian = np.column_stack([(g / sample_scale)[lower_idx] for g in gradients])
        norms = np.linalg.norm(jacobian, axis=0)
        normalized = jacobian / np.where(norms > 0, norms, 1.)
        singular = np.linalg.svd(normalized, compute_uv=False)
        ratio = float(singular[-1]/singular[0]) if len(singular) and singular[0] > 0 else 0.
        condition = float(np.linalg.cond(model.calc_fim()))
        standard_errors = pd.to_numeric(params["Std. Err"], errors="coerce")
        free_rows = params["Std. Err"].ne("-")
        valid = (jacobian.shape[0] >= jacobian.shape[1] and ratio >= 1e-8 and condition <= 1e8
                 and np.isfinite(standard_errors[free_rows]).all()
                 and len(main_loadings) == 6 and all(abs(v) <= 1+1e-6 for v in main_loadings.values())
                 and np.linalg.eigvalsh(model.mx_psi).min() > 0 and np.diag(model.mx_theta).min() >= 0)
        identification = {"jacobian_singular_ratio": ratio, "information_condition": condition,
                          "valid": bool(valid), "scope": "Numerical identification only; ordinary ML does not correct subject clustering"}
    except (ValueError, np.linalg.LinAlgError) as error:
        identification = {"valid": False, "reason": str(error)}

    # 判据检查
    cfi_pass = bool(np.isfinite(cfi) and cfi > thresholds["cfi"])
    tli_pass = bool(np.isfinite(tli) and tli > thresholds["tli"])
    rmsea_pass = bool(np.isfinite(rmsea) and rmsea < thresholds["rmsea"])
    srmr_pass = bool(np.isfinite(srmr) and srmr < thresholds["srmr"])
    all_pass = cfi_pass and rmsea_pass and srmr_pass and identification["valid"]

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
        factor_covariance=factor_covariance,
        identification_valid=identification["valid"],
        identification=identification,
        model_role="post_hoc_residual_sensitivity" if residual_corr else "internal_consistency",
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
    from dual_factor.run_manifest import save_json
    save_json(path, d)
