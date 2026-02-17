from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Callable

import numpy as np


FloatVec = np.ndarray


def project_to_simplex(v: FloatVec) -> FloatVec:
    """Project vector onto probability simplex {x >= 0, sum x = 1}."""
    x = np.asarray(v, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * (np.arange(1, x.size + 1)) > (cssv - 1.0))[0]
    if rho_idx.size == 0:
        return np.full_like(x, 1.0 / x.size)
    rho = int(rho_idx[-1] + 1)
    theta = (cssv[rho - 1] - 1.0) / float(rho)
    w = np.maximum(x - theta, 0.0)
    s = float(np.sum(w))
    if s <= 0.0:
        return np.full_like(x, 1.0 / x.size)
    return w / s


def l1_dist(p: FloatVec, q: FloatVec) -> float:
    return float(np.sum(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64))) / 2.0)


def l2_sq_dist(p: FloatVec, q: FloatVec) -> float:
    d = np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)
    return float(np.dot(d, d))


def gini_continuous(values: FloatVec) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = int(arr.size)
    if n <= 1:
        return 0.0
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    s = np.sort(arr)
    i = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * float(np.sum(i * s))) / (n * total) - (n + 1.0) / n
    return float(min(1.0, max(0.0, g)))


def utilitarian_ref_l2sq(personal_dists: FloatVec) -> FloatVec:
    d = np.asarray(personal_dists, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    p = np.mean(d, axis=0)
    return project_to_simplex(p)


def nash_ref_kl(personal_dists: FloatVec, *, eps: float = 1e-12) -> FloatVec:
    """Compute p* = argmin_p sum_i KL(p || d_i), closed-form geometric mean."""
    d = np.asarray(personal_dists, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    d = np.clip(d, eps, None)
    log_mean = np.mean(np.log(d), axis=0)
    p = np.exp(log_mean)
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0.0:
        return np.full(d.shape[1], 1.0 / float(d.shape[1]), dtype=np.float64)
    return (p / s).astype(np.float64)


def rawlsian_ref_minimax_l2sq(
    personal_dists: FloatVec,
    *,
    max_iter: int = 800,
    step0: float = 0.2,
    tol: float = 1e-9,
) -> FloatVec:
    """Minimize max_i ||p - d_i||^2 on simplex with deterministic projected subgradient."""
    d = np.asarray(personal_dists, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    p = utilitarian_ref_l2sq(d)
    if p.size == 0:
        return p
    best = p.copy()
    best_val = _max_l2sq(best, d)
    for t in range(1, max_iter + 1):
        l2sq = np.sum((p[None, :] - d) ** 2, axis=1)
        i_star = int(np.argmax(l2sq))
        grad = 2.0 * (p - d[i_star])
        alpha = step0 / np.sqrt(float(t))
        p_new = project_to_simplex(p - alpha * grad)
        val = _max_l2sq(p_new, d)
        if val < best_val - tol:
            best_val = val
            best = p_new.copy()
        if np.linalg.norm(p_new - p) <= tol:
            p = p_new
            break
        p = p_new
    return best


@dataclass(frozen=True)
class EgalitarianRefs:
    lam_low: FloatVec
    lam_mid: FloatVec
    lam_high: FloatVec


def egalitarian_refs_mean_plus_lambda_gini(
    personal_dists: FloatVec,
    *,
    lambdas: tuple[float, float, float] = (0.25, 1.0, 4.0),
    max_grid_points: int = 4000,
) -> EgalitarianRefs:
    d = np.asarray(personal_dists, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] == 0:
        empty = np.asarray([], dtype=np.float64)
        return EgalitarianRefs(empty, empty, empty)

    grid = simplex_grid_points(num_dims=int(d.shape[1]), max_points=max_grid_points)
    refs: list[FloatVec] = []
    for lam in lambdas:
        def obj(p: FloatVec) -> float:
            z = np.asarray([l1_dist(p, row) for row in d], dtype=np.float64)
            return float(np.mean(z) + float(lam) * gini_continuous(z))

        idx_best = int(np.argmin([obj(p) for p in grid]))
        p0 = grid[idx_best]
        p_best = _refine_deterministic(p0, obj)
        refs.append(p_best)

    return EgalitarianRefs(refs[0], refs[1], refs[2])


def _max_l2sq(p: FloatVec, d: FloatVec) -> float:
    return float(np.max(np.sum((p[None, :] - d) ** 2, axis=1)))


def _refine_deterministic(p0: FloatVec, obj: Callable[[FloatVec], float]) -> FloatVec:
    p = np.asarray(p0, dtype=np.float64).copy()
    best = p.copy()
    best_v = float(obj(best))
    dim = int(p.size)
    for h in (0.1, 0.05, 0.02, 0.01, 0.005):
        improved = True
        while improved:
            improved = False
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        continue
                    cand = p.copy()
                    cand[i] += h
                    cand[j] -= h
                    cand = project_to_simplex(cand)
                    v = float(obj(cand))
                    if v < best_v - 1e-12:
                        best_v = v
                        best = cand
                        p = cand
                        improved = True
    return best


@lru_cache(maxsize=32)
def simplex_grid_points(*, num_dims: int, max_points: int) -> tuple[FloatVec, ...]:
    if num_dims <= 1:
        return (np.asarray([1.0], dtype=np.float64),)
    m = 1
    while comb(m + num_dims - 1, num_dims - 1) <= max_points:
        m += 1
    m = max(1, m - 1)

    out: list[FloatVec] = []

    def _rec(remaining: int, k: int, prefix: list[int]) -> None:
        if k == num_dims - 1:
            vals = prefix + [remaining]
            arr = np.asarray(vals, dtype=np.float64) / float(m)
            out.append(arr)
            return
        for x in range(remaining + 1):
            _rec(remaining - x, k + 1, prefix + [x])

    _rec(m, 0, [])
    # Deterministic order preserved by recursion.
    return tuple(out)
