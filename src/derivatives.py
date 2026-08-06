"""Object-oriented derivative contracts and their risk sensitivities."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm

from .binomial_tree import BinomialTreeResult, analyse_binomial_tree, price_binomial_option
from .gbm import generate_standard_normal_shocks
from .monte_carlo import MonteCarloEstimate, price_arithmetic_asian_option


class Derivative:
    """Base class defining a common valuation and Greeks interface."""

    def __init__(self, S0, K, T, sigma, yield_curve):
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.sigma = float(sigma)
        self.yield_curve = yield_curve
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.S0 <= 0:
            raise ValueError("S0 must be positive.")
        if self.K <= 0:
            raise ValueError("K must be positive.")
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

    @property
    def discount_factor(self) -> float:
        """Discount factor to maturity supplied by the yield curve."""
        return float(self.yield_curve.get_discount_factor(self.T))

    @property
    def r(self) -> float:
        """Equivalent continuously compounded zero rate to maturity."""
        return -np.log(self.discount_factor) / self.T

    def price(self) -> float:
        raise NotImplementedError("Pricing logic must be implemented by subclasses.")

    def delta(self) -> float:
        raise NotImplementedError

    def gamma(self) -> float:
        raise NotImplementedError

    def vega(self) -> float:
        raise NotImplementedError

    def theta(self) -> float:
        raise NotImplementedError

    def rho(self) -> float:
        raise NotImplementedError

    def greeks(self) -> dict[str, float]:
        """Return delta, gamma, vega, theta and rho in one dictionary.

        Vega and rho are derivatives with respect to decimal volatility and
        decimal interest rates. A one-percentage-point move therefore changes
        value by approximately ``0.01 * vega`` or ``0.01 * rho``. Theta is the
        value change per year of calendar time, holding other inputs fixed.
        """
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
            "rho": self.rho(),
        }


class EuropeanOption(Derivative):
    """Shared Black-Scholes calculations for European calls and puts."""

    @property
    def d1(self) -> float:
        return (
            np.log(self.S0 / self.K)
            + (self.r + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

    @property
    def d2(self) -> float:
        return self.d1 - self.sigma * np.sqrt(self.T)

    def gamma(self) -> float:
        return float(norm.pdf(self.d1) / (self.S0 * self.sigma * np.sqrt(self.T)))

    def vega(self) -> float:
        return float(self.S0 * norm.pdf(self.d1) * np.sqrt(self.T))

    def implied_volatility(self, market_price: float, **solver_kwargs):
        """Recover volatility from a market price for this contract.

        Returns an ``ImpliedVolatilityResult`` containing the recovered
        volatility, convergence method, iteration count and final residual.
        """
        from .implied_volatility import solve_implied_volatility

        option_type = "call" if isinstance(self, EuropeanCall) else "put"
        initial_guess = solver_kwargs.pop("initial_guess", self.sigma)
        return solve_implied_volatility(
            market_price,
            option_type=option_type,
            S0=self.S0,
            K=self.K,
            T=self.T,
            yield_curve=self.yield_curve,
            initial_guess=initial_guess,
            **solver_kwargs,
        )


class EuropeanCall(EuropeanOption):
    """European call option priced analytically under Black-Scholes."""

    def price(self) -> float:
        return float(
            self.S0 * norm.cdf(self.d1)
            - self.K * self.discount_factor * norm.cdf(self.d2)
        )

    def delta(self) -> float:
        return float(norm.cdf(self.d1))

    def theta(self) -> float:
        diffusion_decay = -(
            self.S0 * norm.pdf(self.d1) * self.sigma
        ) / (2.0 * np.sqrt(self.T))
        financing_decay = -self.r * self.K * self.discount_factor * norm.cdf(self.d2)
        return float(diffusion_decay + financing_decay)

    def rho(self) -> float:
        return float(self.K * self.T * self.discount_factor * norm.cdf(self.d2))


class EuropeanPut(EuropeanOption):
    """European put option priced analytically under Black-Scholes."""

    def price(self) -> float:
        return float(
            self.K * self.discount_factor * norm.cdf(-self.d2)
            - self.S0 * norm.cdf(-self.d1)
        )

    def delta(self) -> float:
        return float(norm.cdf(self.d1) - 1.0)

    def theta(self) -> float:
        diffusion_decay = -(
            self.S0 * norm.pdf(self.d1) * self.sigma
        ) / (2.0 * np.sqrt(self.T))
        financing_effect = self.r * self.K * self.discount_factor * norm.cdf(-self.d2)
        return float(diffusion_decay + financing_effect)

    def rho(self) -> float:
        return float(-self.K * self.T * self.discount_factor * norm.cdf(-self.d2))


class ArithmeticAsianOption(Derivative):
    """Fixed-strike arithmetic-average Asian option priced by Monte Carlo.

    Greeks are calculated by central finite differences using common random
    numbers. Reusing the same shocks across bumped valuations sharply reduces
    simulation noise relative to independent revaluation.
    """

    option_type: Literal["call", "put"]

    def __init__(
        self,
        S0,
        K,
        T,
        sigma,
        yield_curve,
        *,
        n_steps: int = 52,
        n_simulations: int = 100_000,
        seed: int = 0,
        antithetic: bool = True,
        confidence_level: float = 0.95,
    ):
        super().__init__(S0, K, T, sigma, yield_curve)
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1.")

        self.n_steps = int(n_steps)
        self.n_simulations = int(n_simulations)
        self.seed = int(seed)
        self.antithetic = bool(antithetic)
        self.confidence_level = float(confidence_level)
        self._shocks = generate_standard_normal_shocks(
            n_simulations=self.n_simulations,
            n_steps=self.n_steps,
            rng=np.random.default_rng(self.seed),
            antithetic=self.antithetic,
        )

    def _estimate_with_parameters(
        self,
        *,
        S0: float | None = None,
        sigma: float | None = None,
        T: float | None = None,
        r: float | None = None,
    ) -> MonteCarloEstimate:
        return price_arithmetic_asian_option(
            S0=self.S0 if S0 is None else float(S0),
            K=self.K,
            r=self.r if r is None else float(r),
            sigma=self.sigma if sigma is None else float(sigma),
            T=self.T if T is None else float(T),
            n_steps=self.n_steps,
            n_simulations=self.n_simulations,
            option_type=self.option_type,
            shocks=self._shocks,
            antithetic=self.antithetic,
            confidence_level=self.confidence_level,
        )

    def estimate(self) -> MonteCarloEstimate:
        """Return price, standard error and confidence interval."""
        return self._estimate_with_parameters()

    def price(self) -> float:
        return self.estimate().price

    def delta(self, relative_bump: float = 1e-3) -> float:
        if relative_bump <= 0:
            raise ValueError("relative_bump must be positive.")
        h = self.S0 * relative_bump
        up = self._estimate_with_parameters(S0=self.S0 + h).price
        down = self._estimate_with_parameters(S0=self.S0 - h).price
        return (up - down) / (2.0 * h)

    def gamma(self, relative_bump: float = 1e-3) -> float:
        if relative_bump <= 0:
            raise ValueError("relative_bump must be positive.")
        h = self.S0 * relative_bump
        up = self._estimate_with_parameters(S0=self.S0 + h).price
        base = self._estimate_with_parameters().price
        down = self._estimate_with_parameters(S0=self.S0 - h).price
        return (up - 2.0 * base + down) / h**2

    def vega(self, volatility_bump: float = 1e-3) -> float:
        if volatility_bump <= 0:
            raise ValueError("volatility_bump must be positive.")
        h = min(volatility_bump, 0.5 * self.sigma)
        up = self._estimate_with_parameters(sigma=self.sigma + h).price
        down = self._estimate_with_parameters(sigma=self.sigma - h).price
        return (up - down) / (2.0 * h)

    def theta(self, time_bump: float = 1.0 / 365.0) -> float:
        if time_bump <= 0:
            raise ValueError("time_bump must be positive.")
        h = min(time_bump, 0.5 * self.T)
        shorter = self._estimate_with_parameters(T=self.T - h).price
        longer = self._estimate_with_parameters(T=self.T + h).price
        return (shorter - longer) / (2.0 * h)

    def rho(self, rate_bump: float = 1e-4) -> float:
        if rate_bump <= 0:
            raise ValueError("rate_bump must be positive.")
        up = self._estimate_with_parameters(r=self.r + rate_bump).price
        down = self._estimate_with_parameters(r=self.r - rate_bump).price
        return (up - down) / (2.0 * rate_bump)


class AsianCall(ArithmeticAsianOption):
    """Fixed-strike arithmetic-average Asian call."""

    option_type: Literal["call"] = "call"


class AsianPut(ArithmeticAsianOption):
    """Fixed-strike arithmetic-average Asian put."""

    option_type: Literal["put"] = "put"


# Explicit aliases for users who prefer the longer product name.
ArithmeticAsianCall = AsianCall
ArithmeticAsianPut = AsianPut


class AmericanOption(Derivative):
    """American option valued by a Cox-Ross-Rubinstein binomial tree.

    The complete Greek set is obtained by bump-and-reprice finite differences.
    This is slower than the analytical European formulas, but it preserves the
    early-exercise feature at every bumped valuation.
    """

    option_type: Literal["call", "put"]

    def __init__(
        self,
        S0,
        K,
        T,
        sigma,
        yield_curve,
        *,
        steps: int = 500,
    ):
        super().__init__(S0, K, T, sigma, yield_curve)
        if steps <= 0:
            raise ValueError("steps must be positive.")
        self.steps = int(steps)

    def _price_with_parameters(
        self,
        *,
        S0: float | None = None,
        sigma: float | None = None,
        T: float | None = None,
        r: float | None = None,
    ) -> float:
        return price_binomial_option(
            S0=self.S0 if S0 is None else float(S0),
            K=self.K,
            r=self.r if r is None else float(r),
            sigma=self.sigma if sigma is None else float(sigma),
            T=self.T if T is None else float(T),
            steps=self.steps,
            option_type=self.option_type,
            exercise_style="american",
        )

    def price(self) -> float:
        return self._price_with_parameters()

    def tree_analysis(self) -> BinomialTreeResult:
        """Return the price and estimated early-exercise boundary."""
        return analyse_binomial_tree(
            S0=self.S0,
            K=self.K,
            r=self.r,
            sigma=self.sigma,
            T=self.T,
            steps=self.steps,
            option_type=self.option_type,
            exercise_style="american",
        )

    def delta(self, relative_bump: float = 1e-2) -> float:
        if relative_bump <= 0:
            raise ValueError("relative_bump must be positive.")
        h = self.S0 * relative_bump
        up = self._price_with_parameters(S0=self.S0 + h)
        down = self._price_with_parameters(S0=self.S0 - h)
        return (up - down) / (2.0 * h)

    def gamma(self, relative_bump: float = 2e-2) -> float:
        if relative_bump <= 0:
            raise ValueError("relative_bump must be positive.")
        h = self.S0 * relative_bump
        up = self._price_with_parameters(S0=self.S0 + h)
        base = self._price_with_parameters()
        down = self._price_with_parameters(S0=self.S0 - h)
        return (up - 2.0 * base + down) / h**2

    def vega(self, volatility_bump: float = 1e-3) -> float:
        if volatility_bump <= 0:
            raise ValueError("volatility_bump must be positive.")
        h = min(volatility_bump, 0.5 * self.sigma)
        up = self._price_with_parameters(sigma=self.sigma + h)
        down = self._price_with_parameters(sigma=self.sigma - h)
        return (up - down) / (2.0 * h)

    def theta(self, time_bump: float = 1.0 / 365.0) -> float:
        if time_bump <= 0:
            raise ValueError("time_bump must be positive.")
        h = min(time_bump, 0.5 * self.T)
        shorter = self._price_with_parameters(T=self.T - h)
        longer = self._price_with_parameters(T=self.T + h)
        return (shorter - longer) / (2.0 * h)

    def rho(self, rate_bump: float = 1e-4) -> float:
        if rate_bump <= 0:
            raise ValueError("rate_bump must be positive.")
        up = self._price_with_parameters(r=self.r + rate_bump)
        down = self._price_with_parameters(r=self.r - rate_bump)
        return (up - down) / (2.0 * rate_bump)


class AmericanCall(AmericanOption):
    """American call option valued by CRR backward induction."""

    option_type: Literal["call"] = "call"


class AmericanPut(AmericanOption):
    """American put option valued by CRR backward induction."""

    option_type: Literal["put"] = "put"
