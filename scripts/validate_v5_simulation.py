"""Known-truth clustered scenarios; quick runs cannot issue a formal certificate."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from dual_factor.config import load_config, config_hash
from dual_factor.protocol import RETAINED
from dual_factor.analysis.sampling import select_nonoverlap
from dual_factor.analysis.cluster_bootstrap import bootstrap_structure
from dual_factor.run_manifest import save_json, code_hash
from dual_factor.v5_pipeline import _model
from dual_factor.analysis.covariance_signal import covariance_signal

SCENARIOS = ("null", "single", "two", "correlated_two", "cross_loading", "heterogeneous", "two_nonnormal")
EXCLUDED = {(7, 1), (9, 1), (10, 2), (12, 2), (12, 3), (13, 3)}


def synthetic_matrix(seed=0, scenario="two"):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    rng = np.random.default_rng(seed)
    parts = []
    for subject in range(1, 15):
        for test in range(1, 4):
            if (subject, test) in EXCLUDED:
                continue
            rho = .85 if scenario == "correlated_two" else .0
            latent = rng.multivariate_normal([0., 0.], [[1., rho], [rho, 1.]], size=60)
            for t in range(1, 60):
                latent[t] = .4 * latent[t-1] + np.sqrt(1-.4**2) * latent[t]
            loadings = np.zeros((6, 2))
            loadings[:4, 0] = [.9, -.9, -.9, -.9]
            loadings[4:, 1] = .9
            if scenario == "single":
                loadings[4:, 0], loadings[4:, 1] = .9, 0
            if scenario == "null":
                loadings[:] = 0
            if scenario == "cross_loading":
                loadings[4] = [.45, .8]
            if scenario == "heterogeneous" and subject >= 8:
                loadings[2] = [0, -.9]
            noise = rng.standard_t(5, (60, 6))/np.sqrt(5/3) if scenario == "two_nonnormal" else rng.normal(size=(60, 6))
            frame_values = latent @ loadings.T + noise*np.sqrt(1-(loadings**2).sum(axis=1))
            frame_values += rng.normal(0, .10, 6)  # independent record effects
            window_values = np.array([frame_values[i:i+3].mean(axis=0) for i in range(58)])
            data = pd.DataFrame(window_values, columns=RETAINED)
            data["SV"] = rng.normal(size=58)
            data["SDR"] = rng.normal(size=58)
            data["subject"], data["test"] = str(subject), str(test)
            data["window_idx"], data["valid"] = np.arange(58), True
            data["kss"] = 3 if test == 1 else 7
            parts.append(data)
    return pd.concat(parts, ignore_index=True)


def protocol_support(df, config, iterations, seed):
    primary = select_nonoverlap(df)
    for phase in (0, 1, 2):
        model = _model(select_nonoverlap(df, phase=phase), list(RETAINED), 2, config)
        if (model.get("status") != "ok" or not model.get("assignment_match")
                or not model.get("precheck", {}).get("kmo_pass") or not all(model.get("criteria", {}).values())):
            return False
    try:
        _, summary = bootstrap_structure(primary, iterations=iterations, seed=seed,
            thresholds=config["structure"]["criteria"], stability=config["structure"]["stability"],
            tol=config["structure"]["paf_tol"], max_iter=config["structure"]["paf_max_iter"])
        return bool(summary["C5_pass"] and all(summary["reference_criteria"].values()))
    except (ValueError, np.linalg.LinAlgError):
        return False


def wilson_upper(successes, n):
    z = 1.959963984540054
    p = successes/n
    return (p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def validate(config, datasets=200, bootstrap_iterations=200):
    outcomes = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        flags, signals = [], []
        for i in range(datasets):
            data = synthetic_matrix(100000*scenario_index+i, scenario)
            flags.append(protocol_support(data, config, bootstrap_iterations, 2026110804+i))
            signal_parameters = config["signal_detection"] | {"iterations": bootstrap_iterations, "seed": 2026110806+i}
            signals.append(covariance_signal(select_nonoverlap(data), **signal_parameters)["observable_signal_detected"])
        count, signal_count = sum(flags), sum(signals)
        outcomes[scenario] = {"latent_supported": count, "n": datasets, "latent_fraction": count/datasets,
                              "latent_wilson_upper": wilson_upper(count, datasets),
                              "observable_signal": signal_count, "signal_fraction": signal_count/datasets,
                              "signal_wilson_upper": wilson_upper(signal_count, datasets)}
        print(f"{scenario}: latent={count}/{datasets}, observable_signal={signal_count}/{datasets}", flush=True)
    checks = {"null_false_latent_support": outcomes["null"]["latent_wilson_upper"] <= .10,
              "single_false_latent_support": outcomes["single"]["latent_wilson_upper"] <= .10,
              "null_false_signal": outcomes["null"]["signal_wilson_upper"] <= .10,
              "single_false_signal": outcomes["single"]["signal_wilson_upper"] <= .10,
              "strong_two_observable_signal_power": outcomes["two"]["signal_fraction"] >= .80,
              "high_correlation_latent_rejected": outcomes["correlated_two"]["latent_wilson_upper"] <= .10,
              "cross_loading_latent_rejected": outcomes["cross_loading"]["latent_wilson_upper"] <= .10,
              "heterogeneity_latent_rejected": outcomes["heterogeneous"]["latent_wilson_upper"] <= .10}
    diagnostic_passed = datasets >= 200 and bootstrap_iterations >= 200 and all(checks.values())
    latent_power = outcomes["two"]["latent_fraction"] >= .80
    return {"datasets_per_scenario": datasets, "bootstrap_iterations": bootstrap_iterations,
            "outcomes": outcomes, "checks": checks,
            "diagnostic_checks_passed": diagnostic_passed,
            "latent_power_validated": latent_power,
            "passed": diagnostic_passed and latent_power,
            "code_sha256": code_hash(ROOT), "config_sha256": config_hash(config),
            "scope": "Separately calibrates observable signal detection and unsafe latent acceptance; not raw measurement validation or teacher acceptance",
            "amendment": "Finite 4/2 latent estimation may be inadmissible despite real covariance signal. Signal detection never grants A or waives PAF/C5."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT/"configs/dual_factor_efa_v5.yaml"))
    parser.add_argument("--datasets", type=int, default=200)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.datasets < 1 or args.bootstrap_iterations < 1:
        parser.error("Positive simulation counts required")
    path = Path(args.output)
    if path.exists():
        parser.error("Refusing to overwrite an existing certificate")
    result = validate(load_config(args.config), args.datasets, args.bootstrap_iterations)
    save_json(path, result)
    print(result)


if __name__ == "__main__":
    main()
