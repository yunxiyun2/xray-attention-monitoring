"""Prepare a validated raw-EOG tensor cache for SI-MSET."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eog_si_mset.config import load_config
from eog_si_mset.data import prepare_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/eog_si_mset_v1.yaml"),
    )
    args = parser.parse_args()
    result = prepare_cache(load_config(args.config), ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
