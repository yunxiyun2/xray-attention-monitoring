"""CPU-only EFA entry point; --dry-run never writes experiment outputs."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from dual_factor.config import load_config
from dual_factor.v5_pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT/"configs/dual_factor_efa_v5.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--validation", help="Full synthetic-validation certificate matching code/config hashes")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT/config_path
    result = run_experiment(load_config(config_path), ROOT, dry_run=args.dry_run,
                            diagnostic_only=args.diagnostic_only, validation=args.validation)
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
