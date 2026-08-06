"""Cox-Ross-Rubinstein binomial-tree pricing utilities.

The functions in this module are deliberately independent of the derivative
classes.  A contract supplies only its option type and exercise style, while
the tree handles risk-neutral valuation and early-exercise decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BinomialTreeResult:
    """Price and, when relevant, the estimated early-exercise boundary.

    ``exercise_boundary`` contains ``NaN`` at dates where early exercise is
    not optimal at any node. For an American put, the boundary is the highest
    stock price at which exercise is optimal. For an American call, it is the
    lowest such stock price.
    """

    price: float
    time_grid: FloatArray
    exercise_boundary: FloatArray


def _validate_inputs(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
) -> None:
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be either 'call' or 'put'.")
    if exercise_style not in ("european", "american"):
        raise ValueError("exercise_style must be either 'european' or 'american'.")
    if not np.isfinite(r):
        raise ValueError("r must be finite.")


def _payoff(stock_prices: FloatArray, K: float, option_type: OptionType) -> FloatArray:
    if option_type == "call":
        return np.maximum(stock_prices - K, 0.0)
    return np.maximum(K - stock_prices, 0.0)


def _crr_parameters(r: float, sigma: float, T: float, steps: int) -> tuple[float, float, float]:
    """Return the CRR up multiplier, risk-neutral probability and discount."""
    dt = T / steps
    up = float(np.exp(sigma * np.sqrt(dt)))
    down = 1.0 / up
    growth = float(np.exp(r * dt))
    probability = (growth - down) / (up - down)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "CRR risk-neutral probability lies outside [0, 1]. "
            "Increase the number of steps or review the model inputs."
        )

    discount = float(np.exp(-r * dt))
    return up, probability, discount


def analyse_binomial_tree(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    *,
    option_type: OptionType,
    exercise_style: ExerciseStyle = "american",
    exercise_tolerance: float = 1e-12,
) -> BinomialTreeResult:
    """Price an option and estimate its early-exercise boundary with a CRR tree.

    The implementation stores only one layer of option values at a time, so
    memory usage grows linearly with the number of time steps.  The boundary is
    recorded during backward induction without retaining the full lattice.
    """
    _validate_inputs(S0, K, r, sigma, T, steps, option_type, exercise_style)
    if exercise_tolerance < 0:
        raise ValueError("exercise_tolerance must be non-negative.")

    up, probability, discount = _crr_parameters(r, sigma, T, steps)
    log_up = np.log(up)
    dt = T / steps

    terminal_up_moves = np.arange(steps + 1, dtype=float)
    terminal_prices = S0 * np.exp((2.0 * terminal_up_moves - steps) * log_up)
    values = _payoff(terminal_prices, K, option_type)

    time_grid = np.arange(steps, dtype=float) * dt
    boundary = np.full(steps, np.nan, dtype=float)

    for time_index in range(steps - 1, -1, -1):
        continuation = discount * (
            (1.0 - probability) * values[:-1]
            + probability * values[1:]
        )

        if exercise_style == "american":
            up_moves = np.arange(time_index + 1, dtype=float)
            stock_prices = S0 * np.exp((2.0 * up_moves - time_index) * log_up)
            exercise = _payoff(stock_prices, K, option_type)
            exercise_nodes = (exercise > 0.0) & (
                exercise > continuation + exercise_tolerance
            )

            if np.any(exercise_nodes):
                exercise_prices = stock_prices[exercise_nodes]
                if option_type == "put":
                    boundary[time_index] = float(exercise_prices.max())
                else:
                    boundary[time_index] = float(exercise_prices.min())

            values = np.maximum(exercise, continuation)
        else:
            values = continuation

    return BinomialTreeResult(
        price=float(values[0]),
        time_grid=time_grid,
        exercise_boundary=boundary,
    )


def price_binomial_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    *,
    option_type: OptionType,
    exercise_style: ExerciseStyle = "american",
) -> float:
    """Return a European or American option price from a CRR tree."""
    return analyse_binomial_tree(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        steps=steps,
        option_type=option_type,
        exercise_style=exercise_style,
    ).price
