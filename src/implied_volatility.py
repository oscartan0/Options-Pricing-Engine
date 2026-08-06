"""Numerical inversion of Black--Scholes prices to implied volatility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .derivatives import EuropeanCall, EuropeanPut

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class ImpliedVolatilityResult:
    """Result returned by the safeguarded implied-volatility solver."""

    volatility: float
    converged: bool
    iterations: int
    method: str
    residual: float


def european_price_bounds(
    *,
    S0: float,
    K: float,
    T: float,
    yield_curve,
    option_type: OptionType,
) -> tuple[float, float]:
    """Return no-arbitrage bounds for a non-dividend European option."""
    S0 = float(S0)
    K = float(K)
    T = float(T)
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K and T must be positive.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    discount_factor = float(yield_curve.get_discount_factor(T))
    discounted_strike = K * discount_factor

    if option_type == "call":
        return max(S0 - discounted_strike, 0.0), S0
    return max(discounted_strike - S0, 0.0), discounted_strike


def _option_value_and_vega(
    *,
    volatility: float,
    option_type: OptionType,
    S0: float,
    K: float,
    T: float,
    yield_curve,
) -> tuple[float, float]:
    option_class = EuropeanCall if option_type == "call" else EuropeanPut
    option = option_class(S0, K, T, volatility, yield_curve)
    return option.price(), option.vega()


def solve_implied_volatility(
    market_price: float,
    *,
    option_type: OptionType,
    S0: float,
    K: float,
    T: float,
    yield_curve,
    initial_guess: float = 0.20,
    lower_volatility: float = 1e-8,
    upper_volatility: float = 5.0,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-10,
    max_newton_iterations: int = 20,
    max_bisection_iterations: int = 200,
    min_vega: float = 1e-10,
) -> ImpliedVolatilityResult:
    """Recover Black--Scholes implied volatility from an observed option price.

    Newton--Raphson is attempted first because it is fast near the root. The
    iteration is safeguarded by a volatility bracket. If vega becomes too small
    or the Newton step leaves the bracket, the solver falls back to bisection.

    Parameters use decimal units: ``0.20`` represents 20% volatility.
    """
    market_price = float(market_price)
    if not np.isfinite(market_price):
        raise ValueError("market_price must be finite.")
    if market_price < 0:
        raise ValueError("market_price cannot be negative.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if lower_volatility <= 0:
        raise ValueError("lower_volatility must be positive.")
    if upper_volatility <= lower_volatility:
        raise ValueError("upper_volatility must exceed lower_volatility.")
    if price_tolerance <= 0 or volatility_tolerance <= 0:
        raise ValueError("solver tolerances must be positive.")
    if max_newton_iterations < 0 or max_bisection_iterations <= 0:
        raise ValueError("iteration limits are invalid.")

    lower_price_bound, upper_price_bound = european_price_bounds(
        S0=S0,
        K=K,
        T=T,
        yield_curve=yield_curve,
        option_type=option_type,
    )
    if market_price < lower_price_bound - price_tolerance:
        raise ValueError(
            f"market_price={market_price:.10g} is below the no-arbitrage "
            f"lower bound {lower_price_bound:.10g}."
        )
    if market_price > upper_price_bound + price_tolerance:
        raise ValueError(
            f"market_price={market_price:.10g} is above the no-arbitrage "
            f"upper bound {upper_price_bound:.10g}."
        )

    low = float(lower_volatility)
    high = float(upper_volatility)
    low_price, _ = _option_value_and_vega(
        volatility=low,
        option_type=option_type,
        S0=S0,
        K=K,
        T=T,
        yield_curve=yield_curve,
    )
    high_price, _ = _option_value_and_vega(
        volatility=high,
        option_type=option_type,
        S0=S0,
        K=K,
        T=T,
        yield_curve=yield_curve,
    )
    low_residual = low_price - market_price
    high_residual = high_price - market_price

    if abs(low_residual) <= price_tolerance:
        return ImpliedVolatilityResult(low, True, 0, "lower-bound", low_residual)
    if abs(high_residual) <= price_tolerance:
        return ImpliedVolatilityResult(high, True, 0, "upper-bound", high_residual)
    if low_residual > 0 or high_residual < 0:
        raise ValueError(
            "The selected volatility interval does not bracket a solution. "
            "Increase upper_volatility or inspect the market price."
        )

    volatility = float(np.clip(initial_guess, low, high))
    total_iterations = 0

    for _ in range(max_newton_iterations):
        total_iterations += 1
        model_price, vega = _option_value_and_vega(
            volatility=volatility,
            option_type=option_type,
            S0=S0,
            K=K,
            T=T,
            yield_curve=yield_curve,
        )
        residual = model_price - market_price
        if abs(residual) <= price_tolerance:
            return ImpliedVolatilityResult(
                volatility, True, total_iterations, "newton", residual
            )

        if residual < 0:
            low = volatility
        else:
            high = volatility

        if not np.isfinite(vega) or abs(vega) < min_vega:
            break

        candidate = volatility - residual / vega
        if not np.isfinite(candidate) or candidate <= low or candidate >= high:
            break
        volatility = candidate

    # Guaranteed convergence for a continuous monotonic Black--Scholes price.
    for _ in range(max_bisection_iterations):
        total_iterations += 1
        midpoint = 0.5 * (low + high)
        model_price, _ = _option_value_and_vega(
            volatility=midpoint,
            option_type=option_type,
            S0=S0,
            K=K,
            T=T,
            yield_curve=yield_curve,
        )
        residual = model_price - market_price

        if (
            abs(residual) <= price_tolerance
            or 0.5 * (high - low) <= volatility_tolerance
        ):
            return ImpliedVolatilityResult(
                midpoint, True, total_iterations, "bisection", residual
            )

        if residual < 0:
            low = midpoint
        else:
            high = midpoint

    final_volatility = 0.5 * (low + high)
    final_price, _ = _option_value_and_vega(
        volatility=final_volatility,
        option_type=option_type,
        S0=S0,
        K=K,
        T=T,
        yield_curve=yield_curve,
    )
    return ImpliedVolatilityResult(
        final_volatility,
        False,
        total_iterations,
        "bisection",
        final_price - market_price,
    )


def implied_volatility(market_price: float, **kwargs) -> float:
    """Convenience wrapper returning only the recovered volatility."""
    result = solve_implied_volatility(market_price, **kwargs)
    if not result.converged:
        raise RuntimeError(
            "Implied-volatility solver did not converge within the iteration limit."
        )
    return result.volatility
