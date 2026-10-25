"""路径 B 执行：预注册指标修剪阶梯（SPEC v3.0 §1）。

判据（预注册，不可事后修改）：
  C1  累计方差解释率（2 因子解）  > 0.60
  C2' 所有保留指标的主载荷        >= 0.50（载荷清晰，逐指标检查）
  C3  最大交叉载荷                < 0.30
  C4  两因子相关                  |r| < 0.70
  C5  修正后指标集上敏感性分析结论一致（由 pipeline 汇总传入）

修剪规则（确定性，优先级从高到低，SPEC §1.3）：
  1. C3 未达标   → 剔除交叉载荷最高的指标（cross_loading）
  2. C2' 未达标  → 剔除主载荷最低的指标（low_loading）
  3. 仅 C1 未达标 → 剔除主载荷最低的指标（variance_driven）
  4. C4 未达标   → 不剔除，立即终止（factor_collinear）

终止条件（SPEC §1.4）：C1-C4 全过即停；保留指标 <5、任一假设因子 <2、
剔除次数超限、或触发规则 4 → 路径 C（unsupported_structural）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dual_factor.analysis.efa import run_efa_fixed, parallel_analysis

# 假设因子分组（固定，SPEC §1.4）
HYPOACTIVE = ["FD", "PSV", "SA", "FN"]
DIFFUSE = ["SE", "GTE", "SV", "SDR"]
ALL_INDICATORS = HYPOACTIVE + DIFFUSE

# 代理效度类别（来自 v2 构念效度报告）
PROXY_CLASS = {
    "FD": "time", "PSV": "amplitude", "SA": "amplitude", "FN": "time",
    "SE": "distribution", "GTE": "distribution", "SV": "amplitude", "SDR": "distribution",
}
PROXY_CLASS_LABEL = {"time": "时间型", "amplitude": "幅度依赖型", "distribution": "分布形态型"}

# 预注册默认阈值（SPEC §1.5；config 可显式覆盖但不得在结果出来后修改）
DEFAULT_THRESHOLDS = {
    "variance_min": 0.60,        # C1
    "loading_clarity": 0.50,     # C2'（逐指标主载荷下限）
    "cross_loading_max": 0.30,   # C3
    "factor_corr_max": 0.70,     # C4
}
DEFAULT_CORRECTION_CONFIG = {
    "enabled": True,
    "max_prunes": 3,
    "min_indicators": 5,
    "min_per_factor": 2,
    **DEFAULT_THRESHOLDS,
}


def _abs_stats(loadings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (每指标主载荷, 每指标交叉载荷)。"""
    abs_l = np.abs(loadings)
    if abs_l.ndim == 1:
        return abs_l, np.zeros_like(abs_l)
    order = np.argsort(abs_l, axis=1)
    primary = order[:, -1]
    per_var_max = abs_l[np.arange(abs_l.shape[0]), primary]
    # 次大载荷为交叉载荷
    per_var_cross = np.zeros(abs_l.shape[0])
    if abs_l.shape[1] > 1:
        for i in range(abs_l.shape[0]):
            row = np.delete(abs_l[i], primary[i])
            per_var_cross[i] = row.max()
    return per_var_max, per_var_cross


def two_factor_solution(
    data: pd.DataFrame,
    extraction: str = "pa",
    rotation: str = "promax",
) -> dict:
    """提取 2 因子固定解并计算判据所需统计量。"""
    res = run_efa_fixed(data, 2, extraction=extraction, rotation=rotation)
    per_var_max, per_var_cross = _abs_stats(res.loadings)
    corr_val = float(res.factor_corr[0, 1]) if (
        res.factor_corr is not None and res.factor_corr.shape[0] >= 2
    ) else None
    return {
        "n_factors": 2,
        "variable_names": res.variable_names,
        "loadings": res.loadings.tolist(),
        "variance_explained": res.variance_explained,
        "factor_corr": corr_val,
        "per_var_max": per_var_max.tolist(),
        "per_var_cross": per_var_cross.tolist(),
        "max_primary_loading": float(np.max(per_var_max)),
        "min_primary_loading": float(np.min(per_var_max)),
        "max_cross_loading": float(np.max(per_var_cross)),
        "efa_result": res,  # 供绘图使用，不落盘
    }


def evaluate_criteria(solution: dict, thresholds: dict) -> dict:
    """按预注册判据 C1-C4 逐项检查（C2' 为逐指标主载荷，从严版）。"""
    c1 = solution["variance_explained"] > thresholds["variance_min"]
    c2 = solution["min_primary_loading"] >= thresholds["loading_clarity"]
    c3 = solution["max_cross_loading"] < thresholds["cross_loading_max"]
    r = solution["factor_corr"]
    c4 = (r is not None) and (abs(r) < thresholds["factor_corr_max"])
    return {
        "C1_variance": {"value": solution["variance_explained"],
                        "threshold": f"> {thresholds['variance_min']}", "passed": bool(c1)},
        "C2_loading_clarity": {"value": solution["min_primary_loading"],
                               "threshold": f">= {thresholds['loading_clarity']}", "passed": bool(c2)},
        "C3_cross_loading": {"value": solution["max_cross_loading"],
                             "threshold": f"< {thresholds['cross_loading_max']}", "passed": bool(c3)},
        "C4_factor_corr": {"value": r if r is not None else 0.0,
                           "threshold": f"|r| < {thresholds['factor_corr_max']}", "passed": bool(c4)},
        "all_pass": bool(c1 and c2 and c3 and c4),
    }


def prune_rule(solution: dict, thresholds: dict) -> tuple[str | None, str, str]:
    """确定性修剪规则（SPEC §1.3 优先级表）。

    返回 (剔除指标 | None, 理由码, 明细)。返回 None 表示不可修复（规则 4）。
    """
    names = solution["variable_names"]
    cross = solution["per_var_cross"]
    primary = solution["per_var_max"]

    # 规则 4：因子共线，剔除无法修复
    r = solution["factor_corr"]
    if r is not None and abs(r) >= thresholds["factor_corr_max"]:
        return None, "factor_collinear", f"|r|={abs(r):.3f} ≥ {thresholds['factor_corr_max']}，剔除指标无法修复因子共线"

    # 规则 1：交叉载荷最高者
    if solution["max_cross_loading"] >= thresholds["cross_loading_max"]:
        idx = int(np.argmax(cross))
        return names[idx], "cross_loading", (
            f"{names[idx]} 交叉载荷 {cross[idx]:.3f} ≥ {thresholds['cross_loading_max']}"
            f"（全指标最高）"
        )

    # 规则 2：主载荷最低者（载荷不清）
    if solution["min_primary_loading"] < thresholds["loading_clarity"]:
        idx = int(np.argmin(primary))
        return names[idx], "low_loading", (
            f"{names[idx]} 主载荷 {primary[idx]:.3f} < {thresholds['loading_clarity']}"
            f"（全指标最低）"
        )

    # 规则 3：仅方差不足，剔主载荷最低者
    if solution["variance_explained"] <= thresholds["variance_min"]:
        idx = int(np.argmin(primary))
        return names[idx], "variance_driven", (
            f"仅 C1 未达标（方差 {solution['variance_explained']:.3f} ≤ "
            f"{thresholds['variance_min']}），剔除最弱指标 {names[idx]}"
            f"（主载荷 {primary[idx]:.3f}）"
        )

    return None, "pass", "全部判据通过，无需修剪"


def floor_violation(active: list[str], cfg: dict) -> tuple[bool, str]:
    """下限检查：指标总数与各假设因子剩余数（SPEC §1.4）。"""
    if len(active) < cfg["min_indicators"]:
        return True, f"保留指标 {len(active)} < {cfg['min_indicators']}"
    h_left = [c for c in HYPOACTIVE if c in active]
    d_left = [c for c in DIFFUSE if c in active]
    if len(h_left) < cfg["min_per_factor"]:
        return True, f"呆滞簇仅剩 {h_left}（< {cfg['min_per_factor']}）"
    if len(d_left) < cfg["min_per_factor"]:
        return True, f"发散簇仅剩 {d_left}（< {cfg['min_per_factor']}）"
    return False, ""


def run_correction_ladder(
    efa_input: pd.DataFrame,
    correction_cfg: dict | None = None,
    extraction: str = "pa",
    rotation: str = "promax",
    n_iter: int = 1000,
    percentile: float = 95.0,
) -> dict:
    """执行预注册修剪阶梯（SPEC §1.2-1.4）。

    efa_input: 列名为指标名的标准化矩阵（窗口级，已 dropna）。
    返回 dict：rungs（逐 rung 全记录）+ final（判定与保留集）+ final_efa_result（不落盘）。
    """
    cfg = dict(DEFAULT_CORRECTION_CONFIG)
    if correction_cfg:
        cfg.update(correction_cfg)
    th = {k: cfg[k] for k in DEFAULT_THRESHOLDS}

    active = [c for c in ALL_INDICATORS if c in efa_input.columns]
    if len(active) < len(ALL_INDICATORS):
        missing = [c for c in ALL_INDICATORS if c not in efa_input.columns]
        raise ValueError(f"指标矩阵缺少列: {missing}")

    rungs: list[dict] = []
    pruned: list[dict] = []
    verdict = "unsupported_structural"
    stop_reason = ""
    final_solution = None
    final_criteria = None
    final_pa = None
    final_efa = None

    for rung_idx in range(cfg["max_prunes"] + 1):
        data = efa_input[active]
        sol = two_factor_solution(data, extraction=extraction, rotation=rotation)
        criteria = evaluate_criteria(sol, th)

        # 平行分析（context，不作为 A/C 否决项，SPEC §1.6）
        try:
            pa = parallel_analysis(
                data, n_iter=n_iter, percentile=percentile,
                max_factors=min(4, len(active)),
            )
            pa_k = 0
            for k in range(min(len(pa["actual_eigenvalues"]), len(pa["random_eigenvalues"]))):
                if pa["actual_eigenvalues"][k] > pa["random_eigenvalues"][k]:
                    pa_k = k + 1
                else:
                    break
            if pa_k == 0:
                pa_k = int(np.sum(np.array(pa["actual_eigenvalues"]) > 1.0)) or 1
        except Exception:
            pa_k = None

        rung_record = {
            "rung": rung_idx,
            "active_indicators": list(active),
            "n_indicators": len(active),
            "variance_explained": sol["variance_explained"],
            "min_primary_loading": sol["min_primary_loading"],
            "max_cross_loading": sol["max_cross_loading"],
            "factor_corr": sol["factor_corr"],
            "loadings": sol["loadings"],
            "variable_names": sol["variable_names"],
            "criteria": criteria,
            "parallel_analysis_n_factors": pa_k,
            "pruned_this_rung": None,
        }
        rungs.append(rung_record)
        final_solution, final_criteria, final_pa, final_efa = sol, criteria, pa_k, sol["efa_result"]

        if criteria["all_pass"]:
            verdict = "supported"
            stop_reason = f"rung {rung_idx} 判据 C1-C4 全部通过"
            break

        if rung_idx == cfg["max_prunes"]:
            stop_reason = f"剔除预算耗尽（max_prunes={cfg['max_prunes']}），仍未能通过全部判据"
            break

        target, reason, detail = prune_rule(sol, th)
        if target is None:
            stop_reason = f"触发规则 4（{reason}）：{detail}"
            break

        remaining = [c for c in active if c != target]
        hit, why = floor_violation(remaining, cfg)
        if hit:
            stop_reason = f"继续剔除将触犯下限：{why}"
            break

        rung_record["pruned_this_rung"] = {
            "indicator": target,
            "reason_code": reason,
            "detail": detail,
            "proxy_class": PROXY_CLASS.get(target, "unknown"),
            "proxy_class_label": PROXY_CLASS_LABEL.get(PROXY_CLASS.get(target, ""), "未知"),
        }
        pruned.append(rung_record["pruned_this_rung"])
        active = remaining

    return {
        "config": cfg,
        "rungs": rungs,
        "pruned": pruned,
        "stop_reason": stop_reason,
        "final": {
            "verdict": verdict,
            "active_indicators": list(active),
            "n_indicators": len(active),
            "n_prunes": len(pruned),
            "variance_explained": final_solution["variance_explained"] if final_solution else None,
            "min_primary_loading": final_solution["min_primary_loading"] if final_solution else None,
            "max_cross_loading": final_solution["max_cross_loading"] if final_solution else None,
            "factor_corr": final_solution["factor_corr"] if final_solution else None,
            "criteria": final_criteria,
            "parallel_analysis_n_factors": final_pa,
        },
        "final_efa_result": final_efa,  # EFAResult，供绘图；勿写入 JSON
    }


def loso_fixed_2factor(
    df: pd.DataFrame,
    subject_col: str,
    indicator_cols: list[str],
    suffix: str = "_z",
    extraction: str = "pa",
    rotation: str = "promax",
) -> pd.DataFrame:
    """留一被试者的 2 因子固定解：回答"2 因子结构跨被试稳不稳"（SPEC §2）。"""
    z_cols = [f"{c}{suffix}" for c in indicator_cols]
    available = [c for c in z_cols if c in df.columns]
    if not available:
        available = [c for c in indicator_cols if c in df.columns]
    if not available:
        raise ValueError(f"指标列 {indicator_cols} 在 DataFrame 中不存在")

    valid = df[df["valid"]] if "valid" in df.columns else df
    subjects = sorted(valid[subject_col].unique())
    results = []

    for held_out in subjects:
        train = valid[valid[subject_col] != held_out][available].dropna()
        if train.shape[0] < len(indicator_cols):
            results.append({
                "held_out_subject": held_out, "n_train": int(train.shape[0]),
                "variance_explained": None, "min_primary_loading": None,
                "max_cross_loading": None, "factor_corr": None, "status": "insufficient",
            })
            continue
        try:
            sol = two_factor_solution(train, extraction=extraction, rotation=rotation)
            results.append({
                "held_out_subject": held_out, "n_train": int(train.shape[0]),
                "variance_explained": sol["variance_explained"],
                "min_primary_loading": sol["min_primary_loading"],
                "max_cross_loading": sol["max_cross_loading"],
                "factor_corr": sol["factor_corr"], "status": "ok",
            })
        except Exception as e:
            results.append({
                "held_out_subject": held_out, "n_train": int(train.shape[0]),
                "variance_explained": None, "min_primary_loading": None,
                "max_cross_loading": None, "factor_corr": None, "status": f"error: {e}",
            })
    return pd.DataFrame(results)


def summarize_loso_2factor(loso2_df: pd.DataFrame, cross_threshold: float = 0.30) -> dict:
    """LOSO 2 因子解的反弹统计：多少折交叉载荷回到阈值以上。"""
    if loso2_df is None or loso2_df.empty:
        return {"n_folds": 0, "n_ok": 0, "n_rebound": None, "rebound_ratio": None}
    ok = loso2_df[loso2_df["status"] == "ok"]
    n_ok = len(ok)
    if n_ok == 0:
        return {"n_folds": len(loso2_df), "n_ok": 0, "n_rebound": None, "rebound_ratio": None}
    n_rebound = int((ok["max_cross_loading"] >= cross_threshold).sum())
    corr_vals = ok["factor_corr"].dropna()
    return {
        "n_folds": len(loso2_df),
        "n_ok": n_ok,
        "n_rebound": n_rebound,
        "rebound_ratio": n_rebound / n_ok,
        "rebound_subjects": ok.loc[ok["max_cross_loading"] >= cross_threshold,
                                   "held_out_subject"].tolist(),
        "cross_max_median": float(ok["max_cross_loading"].median()),
        "factor_corr_median": float(corr_vals.median()) if len(corr_vals) else None,
    }


def format_final_decision(
    ladder: dict,
    corrected_sensitivity: dict | None = None,
    loso2_summary: dict | None = None,
) -> str:
    """生成 v3 决策报告 markdown（SPEC §5）。"""
    final = ladder["final"]
    rungs = ladder["rungs"]
    pruned = ladder["pruned"]
    active_str = "、".join(final["active_indicators"])

    # === 判定与标签 ===
    if final["verdict"] == "supported":
        verdict_label = f"支持双因子可分离（保留 {final['n_indicators']} 项指标：{active_str}）"
    else:
        verdict_label = "转结构（三因子或单维）——双因子假设在 DROZY 上不成立"

    # 不稳定降级判定（SPEC §7：LOSO 反弹 ≥1/3 或 C5 不一致 → 附不稳警示）
    instability_notes = []
    if corrected_sensitivity is not None:
        if not corrected_sensitivity.get("consistent", False):
            instability_notes.append(
                f"C5 未达标：修正集上因子数跨方法不一致"
                f"（窗口{corrected_sensitivity.get('window_level_n_factors')}/"
                f"被试{corrected_sensitivity.get('subject_level_n_factors')}/"
                f"非重叠{corrected_sensitivity.get('nonoverlap_n_factors')}/"
                f"LOSO众数{corrected_sensitivity.get('loso_n_factors_mode')}）"
            )
    if loso2_summary and loso2_summary.get("rebound_ratio") is not None:
        if loso2_summary["rebound_ratio"] >= 1 / 3:
            instability_notes.append(
                f"LOSO 2 因子解反弹：{loso2_summary['n_rebound']}/{loso2_summary['n_ok']} 折"
                f"交叉载荷 ≥0.30（反弹被试：{loso2_summary.get('rebound_subjects', [])}）"
            )

    lines = ["## EFA 决策报告（v3：路径 B 执行——修正后 EFA）\n"]
    lines.append(f"**判定结论**：{verdict_label}\n")
    lines.append(f"**阶梯终止原因**：{ladder['stop_reason']}\n")

    # === 基线对照 ===
    r0 = rungs[0] if rungs else {}
    lines.append("### 基线对照（v2：全 8 指标 2 因子解）\n")
    lines.append("| 项目 | 基线（8 指标） | 修正后（%d 指标） |" % final["n_indicators"])
    lines.append("|---|---|---|")
    lines.append(f"| 累计方差 | {r0.get('variance_explained', float('nan')):.3f} | "
                 f"{final['variance_explained']:.3f} |")
    lines.append(f"| 最低主载荷 | {r0.get('min_primary_loading', float('nan')):.3f} | "
                 f"{final['min_primary_loading']:.3f} |")
    lines.append(f"| 最大交叉载荷 | {r0.get('max_cross_loading', float('nan')):.3f} | "
                 f"{final['max_cross_loading']:.3f} |")
    r0c = r0.get("factor_corr")
    lines.append(f"| 两因子相关 | {r0c:.3f} | {final['factor_corr']:.3f} |" if r0c is not None
                 else f"| 两因子相关 | N/A | {final['factor_corr']:.3f} |")
    lines.append(f"| 平行分析建议因子数 | {r0.get('parallel_analysis_n_factors')} | "
                 f"{final['parallel_analysis_n_factors']} |")
    lines.append("")

    # === 阶梯全记录 ===
    lines.append("### 修剪阶梯全记录（预注册规则，SPEC v3 §1.3）\n")
    lines.append("| Rung | 指标数 | 保留指标 | 方差 | 最低主载荷 | 最大交叉 | r | C1-C4 | PA | 本轮剔除 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for rg in rungs:
        crit = rg["criteria"]
        marks = "".join(
            "✅" if crit[k]["passed"] else "❌"
            for k in ["C1_variance", "C2_loading_clarity", "C3_cross_loading", "C4_factor_corr"]
        )
        pr = rg.get("pruned_this_rung")
        pr_str = f"{pr['indicator']}（{pr['reason_code']}）" if pr else "—"
        r_val = rg.get("factor_corr")
        r_str = f"{r_val:.3f}" if r_val is not None else "N/A"
        lines.append(
            f"| {rg['rung']} | {rg['n_indicators']} | {'、'.join(rg['active_indicators'])} "
            f"| {rg['variance_explained']:.3f} | {rg['min_primary_loading']:.3f} "
            f"| {rg['max_cross_loading']:.3f} | {r_str} | {marks} "
            f"| {rg.get('parallel_analysis_n_factors')} | {pr_str} |"
        )
    lines.append("")

    # === 剔除的实质性论证 ===
    if pruned:
        lines.append("### 剔除指标及实质性理由\n")
        lines.append("| 指标 | 理由码 | 判据明细 | 代理效度类别 |")
        lines.append("|---|---|---|---|")
        for p in pruned:
            lines.append(f"| {p['indicator']} | {p['reason_code']} | {p['detail']} "
                         f"| {p['proxy_class_label']} |")
        lines.append("")

    # === 最终判据表 ===
    lines.append("### 最终保留集判据逐项检查\n")
    lines.append("| 判据 | 名称 | 实测值 | 阈值 | 达标 |")
    lines.append("|---|---|---|---|---|")
    name_map = {
        "C1_variance": "累计方差（2因子解）",
        "C2_loading_clarity": "载荷清晰（最低主载荷）",
        "C3_cross_loading": "最大交叉载荷",
        "C4_factor_corr": "两因子相关",
    }
    for code, name in name_map.items():
        c = final["criteria"][code]
        val = c["value"]
        val_str = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
        lines.append(f"| {code.split('_')[0]} | {name} | {val_str} | {c['threshold']} "
                     f"| {'✅' if c['passed'] else '❌'} |")

    c5_val = "一致" if (corrected_sensitivity and corrected_sensitivity.get("consistent")) else "不一致/不可评估"
    c5_pass = bool(corrected_sensitivity and corrected_sensitivity.get("consistent"))
    lines.append(f"| C5 | 修正集敏感性一致性 | {c5_val} | 一致 | {'✅' if c5_pass else '❌'} |")
    lines.append("")

    # === 平行分析维度警示 ===
    pa_k = final.get("parallel_analysis_n_factors")
    if final["verdict"] == "supported" and pa_k is not None and pa_k != 2:
        lines.append(f"### ⚠ 维度警示（不可省略）\n")
        lines.append(
            f"2 因子解在保留指标上满足全部判据，但平行分析建议 **{pa_k} 维**结构。"
            f"双因子结论应理解为\"保留指标子集上两因子可分离\"，而非\"数据整体为二维\"。\n"
        )

    # === 敏感性 ===
    if corrected_sensitivity is not None:
        lines.append("### 修正后敏感性分析（最终保留集）\n")
        lines.append("| 方法 | 因子数 |")
        lines.append("|---|---|")
        lines.append(f"| 窗口级（基线） | {corrected_sensitivity.get('window_level_n_factors')} |")
        lines.append(f"| 被试级聚合 | {corrected_sensitivity.get('subject_level_n_factors')} |")
        lines.append(f"| 非重叠窗口 | {corrected_sensitivity.get('nonoverlap_n_factors')} |")
        lines.append(f"| LOSO 众数 | {corrected_sensitivity.get('loso_n_factors_mode')} |")
        lines.append("")
    if loso2_summary and loso2_summary.get("n_ok", 0) > 0:
        lines.append(
            f"LOSO 2 因子固定解：{loso2_summary['n_ok']} 折中 "
            f"{loso2_summary['n_rebound']} 折交叉载荷反弹（≥0.30），"
            f"交叉载荷中位数 {loso2_summary['cross_max_median']:.3f}，"
            f"因子相关中位数 {loso2_summary['factor_corr_median']:.3f}。\n"
        )

    # === 不稳定警示 ===
    if instability_notes and final["verdict"] == "supported":
        lines.append("### ⚠ 不稳定警示\n")
        for note in instability_notes:
            lines.append(f"- {note}")
        lines.append("")
        verdict_label += "（附被试级不稳警示）"

    # === 结论限定与下一步 ===
    lines.append("### 结论表述限定（强制）\n")
    if final["verdict"] == "supported":
        lines.append(
            f"- 对外表述必须为：**\"在保留的 {final['n_indicators']} 项指标"
            f"（{active_str}）上，呆滞与发散两因子可分离\"**。\n"
            f"- 禁止省略\"保留指标集\"限定的任何简写；禁止写\"八项指标双因子可分离\"。\n"
        )
        next_step = "启动 CFA 预注册（P1-1），因子结构以本保留指标集为准。"
    else:
        lines.append(
            "- 双因子假设在 DROZY 上不成立（路径 C）。对最终保留集做 3 因子解解释性分析"
            "（检验 SDR 是否独立维度），或转单维模型。\n"
        )
        next_step = ("对最终保留集做 3 因子解解释性分析并尝试命名；"
                     "如无实质意义则转单维模型，如实报告假设被推翻。")

    lines.append(f"**判定依据**：{ladder['stop_reason']}。预注册修剪规则与判据见 "
                 f"docs/superpowers/specs/2026-10-25-dual-factor-efa-v3.md §1。\n")
    lines.append(f"**下一步**：{next_step}\n")
    lines.append(f"**最终判定**：{verdict_label}\n")
    return "\n".join(lines)
