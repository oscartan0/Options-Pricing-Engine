"""Functional utilities for simulating geometric Brownian motion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _validate_inputs(S0: float, sigma: float, T: float) -> None:
    """Validate common GBM parameters."""
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if T < 0:
        raise ValueError("T must be non-negative.")


def generate_standard_normal_shocks(
    n_simulations: int,
    n_steps: int,
    *,
    rng: np.random.Generator | None = None,
    antithetic: bool = False,
) -> FloatArray:
    """Generate a matrix of independent standard-normal shocks.

    Antithetic sampling pairs every simulated shock path with its negative,
    which is useful for reducing Monte Carlo variance.
    """
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    rng = np.random.default_rng() if rng is None else rng

    if not antithetic:
        return rng.standard_normal((n_simulations, n_steps))

    n_base = (n_simulations + 1) // 2
    base_shocks = rng.standard_normal((n_base, n_steps))
    return np.concatenate((base_shocks, -base_shocks), axis=0)[:n_simulations]


def paths_from_standard_normal_shocks(
    S0: float,
    drift: float,
    sigma: float,
    T: float,
    shocks: NDArray[np.floating],
) -> FloatArray:
    """Construct exact-step GBM paths from pre-generated normal shocks.

    Supplying the shocks explicitly makes common-random-number comparisons
    possible when parameters are bumped for numerical Greeks.
    """
    _validate_inputs(S0, sigma, T)
    shock_array = np.asarray(shocks, dtype=float)
    if shock_array.ndim != 2 or shock_array.shape[1] == 0:
        raise ValueError("shocks must be a two-dimensional non-empty array.")

    n_simulations, n_steps = shock_array.shape
    if T == 0:
        return np.full((n_simulations, n_steps + 1), float(S0), dtype=float)

    dt = T / n_steps
    log_increments = (
        (drift - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * shock_array
    )

    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_simulations, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def simulate_terminal_prices(
    S0: float,
    drift: float,
    sigma: float,
    T: float,
    n_simulations: int,
    *,
    rng: np.random.Generator | None = None,
    antithetic: bool = False,
) -> FloatArray:
    """Simulate terminal prices under the exact GBM transition."""
    _validate_inputs(S0, sigma, T)
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    if T == 0:
        return np.full(n_simulations, float(S0), dtype=float)

    rng = np.random.default_rng() if rng is None else rng

    if antithetic:
        n_base = (n_simulations + 1) // 2
        shocks = rng.standard_normal(n_base)
        shocks = np.concatenate((shocks, -shocks))[:n_simulations]
    else:
        shocks = rng.standard_normal(n_simulations)

    log_returns = (
        (drift - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * shocks
    )
    return S0 * np.exp(log_returns)


def simulate_paths(
    S0: float,
    drift: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_simulations: int,
    *,
    rng: np.random.Generator | None = None,
    antithetic: bool = False,
) -> FloatArray:
    """Simulate complete GBM paths using exact stepwise transitions.

    Returns an array with shape ``(n_simulations, n_steps + 1)``. The first
    column contains the initial asset price.
    """
    _validate_inputs(S0, sigma, T)
    shocks = generate_standard_normal_shocks(
        n_simulations=n_simulations,
        n_steps=n_steps,
        rng=rng,
        antithetic=antithetic,
    )
    return paths_from_standard_normal_shocks(
        S0=S0,
        drift=drift,
        sigma=sigma,
        T=T,
        shocks=shocks,
    )
