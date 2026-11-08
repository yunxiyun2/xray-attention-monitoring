"""Observable covariance signal, explicitly NOT a latent-factor acceptance test.

Two-indicator latent factors can be weakly identified even when their covariance
group is obvious. Report the signal separately; never use it to waive PAF/C5.
"""
import numpy as np
from dual_factor.protocol import RETAINED
from dual_factor.analysis.sampling import structural_correlation


def covariance_signal(df, iterations=2000, seed=2026110806, coherence_min=.5, eigen_floor=1.):
    correlation, matrices, identities = structural_correlation(df, list(RETAINED))
    pairs = np.triu_indices(4, 1)
    def statistics(r):
        return np.array([np.mean(np.abs(r[:4, :4][pairs])), abs(r[4, 5]),
                         np.linalg.eigvalsh(r)[-2]])
    rng = np.random.default_rng(seed)
    draws = [statistics(matrices[rng.integers(0, len(identities), len(identities))].mean(axis=0))
             for _ in range(iterations)]
    # Three simultaneous descriptive intervals; calibrated by known-truth cases.
    ci = np.quantile(draws, [.05/6, 1-.05/6], axis=0)
    signal = bool(ci[0, 0] >= coherence_min and ci[0, 1] >= coherence_min and ci[0, 2] > eigen_floor)
    return {"observable_signal_detected": signal, "n_subjects": len(identities), "iterations": iterations,
            "statistics": dict(zip(["within_f1_abs_correlation", "within_f2_abs_correlation", "second_correlation_eigenvalue"], statistics(correlation).tolist())),
            "simultaneous_descriptive_ci": ci.T.tolist(), "eigen_floor": eigen_floor, "coherence_min": coherence_min,
            "scope": "Observable covariance groups only; eigen_floor is descriptive, not a cluster-corrected PA p-value",
            "latent_acceptance": False, "can_unlock_cfa": False}
