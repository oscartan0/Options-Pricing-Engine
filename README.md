# Option Pricing Engine

A Python pricing framework that compares three core approaches to option valuation:

- analytical Black–Scholes pricing for European options;
- risk-neutral geometric Brownian motion and Monte Carlo pricing for path-dependent Asian options;
- Cox–Ross–Rubinstein binomial trees for American early-exercise options.

## Features

- interpolated zero-rate yield curve and discount factors;
- analytical European call and put prices under Black–Scholes;
- exact-step GBM terminal-price and full-path simulation;
- reusable Monte Carlo estimators with antithetic sampling, standard errors and confidence intervals;
- fixed-strike arithmetic-average Asian call and put classes;
- American call and put classes valued by backward induction on a CRR tree;
- estimated American put early-exercise boundary;
- delta, gamma, vega, theta and rho for every option class;
- analytical European Greeks and numerical Greeks for Asian and American options;
- common random numbers for more stable Asian finite-difference Greeks;
- safeguarded implied-volatility inversion using Newton–Raphson with bisection fallback;
- no-arbitrage price-bound checks and synthetic skew analysis across strikes and maturities;
- automated pricing, convergence and consistency tests.

## Example

```python
from src.derivatives import EuropeanCall, AsianCall, AmericanPut
from src.implied_volatility import solve_implied_volatility
from src.yieldcurve import YieldCurve

yield_curve = YieldCurve(
    maturities=[0.5, 1.0, 2.0],
    zero_rates=[0.03, 0.032, 0.035],
)

european_call = EuropeanCall(
    S0=100,
    K=100,
    T=1.0,
    sigma=0.20,
    yield_curve=yield_curve,
)

asian_call = AsianCall(
    S0=100,
    K=100,
    T=1.0,
    sigma=0.20,
    yield_curve=yield_curve,
    n_steps=52,
    n_simulations=100_000,
    seed=42,
)

american_put = AmericanPut(
    S0=100,
    K=100,
    T=1.0,
    sigma=0.20,
    yield_curve=yield_curve,
    steps=500,
)

market_price = european_call.price()
iv_result = solve_implied_volatility(
    market_price,
    option_type="call",
    S0=100,
    K=100,
    T=1.0,
    yield_curve=yield_curve,
)

print(european_call.price(), european_call.greeks())
print(iv_result)
print(asian_call.estimate(), asian_call.greeks())
print(american_put.price(), american_put.greeks())
print(american_put.tree_analysis().exercise_boundary)
```

## Pricing methods

### European options

European calls and puts use the closed-form Black–Scholes solution. Their Greeks are analytical and provide a benchmark for validating the numerical methods.

### Arithmetic-average Asian options

Asian calls and puts are priced by simulating risk-neutral GBM paths and applying the payoff to the arithmetic average of the monitored prices. Their Greeks are central finite differences evaluated with the same random shocks across bumped valuations.

### Implied volatility

European call and put prices can be inverted to recover the Black–Scholes volatility consistent with an observed market price. The solver attempts Newton–Raphson first and falls back to bisection if vega becomes too small or a Newton step leaves the valid bracket. It rejects prices outside no-arbitrage bounds rather than returning a misleading result.

The accompanying notebook generates a transparent synthetic volatility skew across strikes and maturities, converts those volatilities into option prices and verifies that the solver recovers the original inputs.

### American options

American calls and puts use a Cox–Ross–Rubinstein tree. At every node, backward induction compares the discounted continuation value with the immediate-exercise payoff. The implementation records the early-exercise boundary without storing the entire tree.

For a non-dividend-paying stock, the American call should match the corresponding European tree value. An American put can be more valuable than a European put because early exercise may be optimal.

## Greek conventions

- `delta` and `gamma` are with respect to the spot price.
- `vega` is the derivative with respect to decimal volatility. Multiply by `0.01` for a one-volatility-point move.
- `theta` is the annual value change as calendar time passes. Divide by `365` for an approximate one-day value.
- `rho` is the derivative with respect to a decimal continuously compounded rate. Multiply by `0.01` for a one-percentage-point move.

European Greeks are analytical. Asian and American Greeks are numerical and therefore depend on the finite-difference bump size and numerical pricing resolution.

## Assumptions

- no dividends or other carrying costs;
- continuously compounded zero rates;
- constant volatility;
- risk-neutral GBM for Monte Carlo valuation;
- fixed-strike arithmetic Asian options with equally spaced observations excluding the initial spot;
- CRR recombining trees for American options.

## Structure

```text
src/
    binomial_tree.py   CRR pricing and exercise-boundary analysis
    derivatives.py     European, Asian and American contracts and Greeks
    gbm.py             functional GBM simulation utilities
    implied_volatility.py  safeguarded Black–Scholes inversion
    monte_carlo.py     payoff functions and Monte Carlo estimators
    yieldcurve.py      zero-rate interpolation and discount factors

notebooks/             analysis and validation
tests/                 automated checks
```
