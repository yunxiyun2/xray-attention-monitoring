"""E2 面部单模态基线实验入口。

对应 spec v2.0 §7.1 E2，回答 RQ2（面部行为特征能否区分 alert/sleepy）。

用法（从项目根目录）：
    PYTHONPATH=src conda run -n Anomaly python experiments/face_baseline/run.py \
        --config configs/multimodal.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

# 把项目根加入 sys.path 以便 import experiments._unimodal_baseline
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments._unimodal_baseline import run_unimodal_baseline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 面部单模态基线")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument(
        "--features", default="dataset/features/face_features.csv",
        help="面部特征 CSV 路径（相对项目根）",
    )
    parser.add_argument(
        "--output", default="results/multimodal/face_baseline",
        help="输出目录（相对项目根）",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(_PROJECT_ROOT / args.config))
    feature_csv = _PROJECT_ROOT / args.features
    output_dir = _PROJECT_ROOT / args.output

    if not feature_csv.exists():
        raise FileNotFoundError(
            f"面部特征矩阵不存在: {feature_csv}\n"
            f"请先运行: PYTHONPATH=src python experiments/build_features.py --with-face"
        )

    run_unimodal_baseline(
        feature_csv=feature_csv,
        output_dir=output_dir,
        modality="face",
        config_path=_PROJECT_ROOT / args.config,
        inner_cv_folds=cfg.get("nested_loso", {}).get("inner_cv_folds", 3),
        random_state=cfg.get("nested_loso", {}).get("random_state", 42),
    )


if __name__ == "__main__":
    main()
