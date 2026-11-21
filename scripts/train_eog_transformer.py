"""Train SI-MSET with leave-one-subject-out validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eog_si_mset.config import load_config
from eog_si_mset.protocol import VARIANTS
from eog_si_mset.train import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/eog_si_mset_v1.yaml"),
    )
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    args = parser.parse_args()
    result = run_training(load_config(args.config), ROOT, variant=args.variant)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
