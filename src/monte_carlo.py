"""Functional Monte Carlo pricing utilities for option payoffs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .gbm import (
    generate_standard_normal_shocks,
    paths_from_standard_normal_shocks,
    simulate_terminal_prices,
)


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MonteCarloEstimate:
    """Monte Carlo price estimate and sampling uncertainty."""

    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    n_simulations: int


def european_call_payoff(terminal_prices: ArrayLike, strike: float) -> np.ndarray:
    """Return European call payoffs at maturity."""
    return np.maximum(np.asarray(terminal_prices, dtype=float) - strike, 0.0)


def european_put_payoff(terminal_prices: ArrayLike, strike: float) -> np.ndarray:
    """Return European put payoffs at maturity."""
    return np.maximum(strike - np.asarray(terminal_prices, dtype=float), 0.0)


def arithmetic_asian_payoff(
    paths: NDArray[np.floating],
    strike: float,
    option_type: OptionType,
) -> np.ndarray:
    """Return fixed-strike arithmetic-average Asian option payoffs.

    The averaging window includes observations after time zero and excludes
    the initial price.
    """
    path_array = np.asarray(paths, dtype=float)
    if path_array.ndim != 2 or path_array.shape[1] < 2:
        raise ValueError("paths must have shape (n_simulations, n_steps + 1).")

    arithmetic_average = path_array[:, 1:].mean(axis=1)
    if option_type == "call":
        return np.maximum(arithmetic_average - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - arithmetic_average, 0.0)
    raise ValueError("option_type must be either 'call' or 'put'.")


def discounted_estimate(
    payoffs: ArrayLike,
    r: float,
    T: float,
    *,
    confidence_level: float = 0.95,
) -> MonteCarloEstimate:
    """Discount simulated payoffs and return an estimate with a confidence interval."""
    payoff_array = np.asarray(payoffs, dtype=float)
    if payoff_array.ndim != 1 or payoff_array.size == 0:
        raise ValueError("payoffs must be a non-empty one-dimensional array.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")

    discounted_payoffs = np.exp(-r * T) * payoff_array
    price = float(discounted_payoffs.mean())

    if discounted_payoffs.size == 1:
        standard_error = 0.0
    else:
        standard_error = float(
            discounted_payoffs.std(ddof=1) / np.sqrt(discounted_payoffs.size)
        )

    critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    margin = critical_value * standard_error

    return MonteCarloEstimate(
        price=price,
        standard_error=standard_error,
        confidence_interval=(price - margin, price + margin),
        n_simulations=int(discounted_payoffs.size),
    )


def price_european_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_simulations: int,
    payoff: Callable[[np.ndarray, float], np.ndarray],
    *,
    rng: np.random.Generator | None = None,
    antithetic: bool = True,
    confidence_level: float = 0.95,
) -> MonteCarloEstimate:
    """Price a European option under risk-neutral GBM."""
    terminal_prices = simulate_terminal_prices(
        S0=S0,
        drift=r,
        sigma=sigma,
        T=T,
        n_simulations=n_simulations,
        rng=rng,
        antithetic=antithetic,
    )
    payoffs = payoff(terminal_prices, K)
    return discounted_estimate(
        payoffs,
        r=r,
        T=T,
        confidence_level=confidence_level,
    )


def price_arithmetic_asian_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_simulations: int,
    option_type: OptionType,
    *,
    rng: np.random.Generator | None = None,
    shocks: NDArray[np.floating] | None = None,
    antithetic: bool = True,
    confidence_level: float = 0.95,
) -> MonteCarloEstimate:
    """Price a fixed-strike arithmetic-average Asian option under GBM.

    Passing a shock matrix allows the same random numbers to be reused across
    bumped valuations, materially stabilising finite-difference Greeks.
    """
    if shocks is None:
        shocks = generate_standard_normal_shocks(
            n_simulations=n_simulations,
            n_steps=n_steps,
            rng=rng,
            antithetic=antithetic,
        )
    else:
        shocks = np.asarray(shocks, dtype=float)
        if shocks.shape != (n_simulations, n_steps):
            raise ValueError(
                "shocks must have shape (n_simulations, n_steps)."
            )

    paths = paths_from_standard_normal_shocks(
        S0=S0,
        drift=r,
        sigma=sigma,
        T=T,
        shocks=shocks,
    )
    payoffs = arithmetic_asian_payoff(paths, K, option_type)
    return discounted_estimate(
        payoffs,
        r=r,
        T=T,
        confidence_level=confidence_level,
    )


def price_arithmetic_asian_call(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_simulations: int,
    *,
    rng: np.random.Generator | None = None,
    shocks: NDArray[np.floating] | None = None,
    antithetic: bool = True,
    confidence_level: float = 0.95,
) -> MonteCarloEstimate:
    """Price a fixed-strike arithmetic-average Asian call under GBM."""
    return price_arithmetic_asian_option(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_simulations=n_simulations,
        option_type="call",
        rng=rng,
        shocks=shocks,
        antithetic=antithetic,
        confidence_level=confidence_level,
    )


def price_arithmetic_asian_put(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_simulations: int,
    *,
    rng: np.random.Generator | None = None,
    shocks: NDArray[np.floating] | None = None,
    antithetic: bool = True,
    confidence_level: float = 0.95,
) -> MonteCarloEstimate:
    """Price a fixed-strike arithmetic-average Asian put under GBM."""
    return price_arithmetic_asian_option(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_simulations=n_simulations,
        option_type="put",
        rng=rng,
        shocks=shocks,
        antithetic=antithetic,
        confidence_level=confidence_level,
    )
