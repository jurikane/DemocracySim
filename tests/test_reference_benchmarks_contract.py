from __future__ import annotations

import numpy as np

from src.analysis.reference_benchmarks import (
    project_to_simplex,
    utilitarian_ref_l2sq,
    nash_ref_kl,
    rawlsian_ref_minimax_l2sq,
    egalitarian_refs_mean_plus_lambda_gini,
)


def test_project_to_simplex_returns_valid_distribution() -> None:
    v = np.asarray([2.0, -1.0, 0.5], dtype=np.float64)
    p = project_to_simplex(v)
    assert p.shape == (3,)
    assert np.all(p >= -1e-12)
    np.testing.assert_allclose(np.sum(p), 1.0, rtol=0.0, atol=1e-12)


def test_utilitarian_l2sq_is_arithmetic_mean() -> None:
    d = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.4, 0.4, 0.2],
        ],
        dtype=np.float64,
    )
    p = utilitarian_ref_l2sq(d)
    expected = np.mean(d, axis=0)
    np.testing.assert_allclose(p, expected, rtol=0.0, atol=1e-12)


def test_nash_kl_matches_normalized_geometric_mean() -> None:
    d = np.asarray(
        [
            [0.64, 0.16, 0.20],
            [0.36, 0.24, 0.40],
        ],
        dtype=np.float64,
    )
    p = nash_ref_kl(d)
    gm = np.exp(np.mean(np.log(d), axis=0))
    gm = gm / float(np.sum(gm))
    np.testing.assert_allclose(p, gm, rtol=0.0, atol=1e-12)


def test_nash_kl_with_zeros_uses_eps_clipped_geometric_mean() -> None:
    eps = 1e-12
    d = np.asarray(
        [
            [0.7, 0.3, 0.0],
            [0.6, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    p = nash_ref_kl(d, eps=eps)
    clipped = np.clip(d, eps, None)
    gm = np.exp(np.mean(np.log(clipped), axis=0))
    gm = gm / float(np.sum(gm))
    np.testing.assert_allclose(p, gm, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(p), 1.0, rtol=0.0, atol=1e-12)
    assert np.all(p > 0.0)


def test_nash_kl_is_better_than_random_simplex_candidates() -> None:
    d = np.asarray(
        [
            [0.64, 0.16, 0.20],
            [0.36, 0.24, 0.40],
            [0.28, 0.32, 0.40],
        ],
        dtype=np.float64,
    )
    p = nash_ref_kl(d)

    def _objective(x: np.ndarray) -> float:
        # Sum_i KL(x || d_i)
        x = np.asarray(x, dtype=np.float64)
        return float(np.sum(x[None, :] * (np.log(x[None, :]) - np.log(d)), axis=(0, 1)))

    rng = np.random.default_rng(12345)
    candidates = rng.dirichlet(alpha=np.ones(d.shape[1]), size=500)
    best_random = float(min(_objective(c) for c in candidates))
    got = float(_objective(p))
    assert got <= best_random + 1e-12


def test_rawlsian_minimax_l2sq_in_symmetric_two_color_case_is_midpoint() -> None:
    d = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    p = rawlsian_ref_minimax_l2sq(d)
    np.testing.assert_allclose(p, np.asarray([0.5, 0.5], dtype=np.float64), rtol=0.0, atol=1e-2)


def test_egalitarian_refs_return_valid_simplex_points() -> None:
    d = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ],
        dtype=np.float64,
    )
    refs = egalitarian_refs_mean_plus_lambda_gini(d)
    for p in (refs.lam_low, refs.lam_mid, refs.lam_high):
        assert p.shape == (3,)
        assert np.all(p >= -1e-9)
        np.testing.assert_allclose(np.sum(p), 1.0, rtol=0.0, atol=1e-9)
