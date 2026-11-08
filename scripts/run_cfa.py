"""CFA internal-consistency diagnostic, gated by real EFA acceptance."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--efa-report", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    from dual_factor.run_manifest import require_teacher_acceptance, file_hash, create_run, save_json
    approval = require_teacher_acceptance(args.approval, args.efa_report)
    if approval.get("matrix_sha256") != file_hash(args.matrix):
        parser.error("Acceptance must also bind the exact matrix SHA256")
    import pandas as pd
    import numpy as np
    from dual_factor.analysis.sampling import select_nonoverlap
    from dual_factor.analysis.cfa import run_cfa, cfa_to_dict
    from dual_factor.protocol import RETAINED
    data = select_nonoverlap(pd.read_csv(args.matrix))
    data = data.loc[np.isfinite(data[list(RETAINED)].to_numpy(float)).all(axis=1)].copy()
    for _, person in data.groupby("subject"):
        counts = person.groupby("test")["test"].transform("size").to_numpy(float)
        w = 1/(person.test.nunique()*counts)
        values = person[list(RETAINED)].to_numpy(float)
        mu = w @ values
        sd = np.sqrt(w @ (values-mu)**2)
        if np.any(sd <= 0):
            parser.error("Degenerate within-person indicators")
        for index, column in enumerate(RETAINED):
            data.loc[person.index, column+"_z"] = (values[:, index]-mu[index])/sd[index]
    output = create_run(args.output_root)
    results = {}
    for name, residual in [("M0", None), ("M1_historical_posthoc", [("PSV_z", "SA_z")])]:
        try:
            results[name] = cfa_to_dict(run_cfa(data, residual_corr=residual))
        except (ValueError, np.linalg.LinAlgError) as error:
            results[name] = {"status": "failed", "reason": str(error)}
    save_json(output/"cfa_models.json", results)
    save_json(output/"manifest.json", {"matrix_sha256": file_hash(args.matrix),
        "accepted_efa_report_sha256": file_hash(args.efa_report),
        "scope": "Ordinary row-weighted ML internal fit; cluster-robust CFA validation is not implemented",
        "evidence_upgrade": False, "external_validation_enabled": False})
    (output/"cfa_report.md").write_text("# CFA内部拟合诊断\n\n主模型M0与历史事后修改M1分别报告。\n\n本运行未提供被试聚类稳健推断，不升级L3或开启外部验证。非正态不豁免RMSEA。\n\n参数、合法性和全部达标字段见cfa_models.json。\n")
    print(output)


if __name__ == "__main__":
    main()
