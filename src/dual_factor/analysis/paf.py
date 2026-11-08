"""Iterative principal-axis factoring, distinct from principal/SVD extraction."""
from dataclasses import dataclass
import numpy as np
from factor_analyzer.rotator import Rotator


@dataclass
class PAFFit:
    loadings: np.ndarray
    phi: np.ndarray
    communalities: np.ndarray
    uniquenesses: np.ndarray
    reproduced: np.ndarray
    structure: np.ndarray
    iterations: int
    history: list[float]


def validate_correlation(corr):
    r = np.asarray(corr, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not np.isfinite(r).all():
        raise ValueError("Correlation must be a finite square matrix")
    if not np.allclose(r, r.T, atol=1e-10) or not np.allclose(np.diag(r), 1, atol=1e-10):
        raise ValueError("Correlation must be symmetric with unit diagonal")
    if np.linalg.eigvalsh(r).min() <= 1e-10:
        raise ValueError("Correlation is singular or not positive definite")
    return r


def fit_paf(corr, n_factors=2, rotation="promax", tol=1e-6, max_iter=500):
    r = validate_correlation(corr)
    p = len(r)
    if not 1 <= n_factors < p or tol <= 0 or max_iter < 1:
        raise ValueError("Invalid factor count or convergence settings")
    if rotation not in (None, "promax", "varimax", "oblimin"):
        raise ValueError(f"Unknown rotation: {rotation}")
    h = 1 - 1 / np.diag(np.linalg.inv(r))
    history = []
    for iteration in range(1, max_iter + 1):
        reduced = r.copy()
        np.fill_diagonal(reduced, h)
        values, vectors = np.linalg.eigh(reduced)
        indices = np.argsort(values)[::-1][:n_factors]
        if np.any(values[indices] <= 0):
            raise ValueError("PAF selected a non-positive common eigenvalue")
        loading = vectors[:, indices] * np.sqrt(values[indices])
        updated = (loading ** 2).sum(axis=1)
        if np.any(updated > 1 + 1e-10) or np.any(updated < -1e-10):
            raise ValueError("PAF Heywood case: communalities outside [0, 1]")
        error = float(np.max(np.abs(updated - h)))
        history.append(error)
        h = updated
        if error < tol:
            break
    else:
        raise ValueError(f"PAF did not converge in {max_iter} iterations")
    phi = np.eye(n_factors)
    if rotation and n_factors > 1:
        rotator = Rotator(method=rotation)
        loading = rotator.fit_transform(loading)
        if rotator.phi_ is not None:
            phi = rotator.phi_
    reproduced = loading @ phi @ loading.T
    if not np.allclose(np.diag(reproduced), h, atol=1e-6):
        raise ValueError("Rotation failed to preserve reconstructed common variance")
    if np.linalg.eigvalsh(phi).min() <= 0:
        raise ValueError("Factor correlation is not positive definite")
    return PAFFit(loading, phi, np.diag(reproduced), 1-np.diag(reproduced),
                  reproduced, loading @ phi, iteration, history)
