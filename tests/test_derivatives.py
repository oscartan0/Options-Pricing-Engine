import numpy as np

from src.derivatives import AsianCall, AsianPut, EuropeanCall, EuropeanPut
from src.yieldcurve import YieldCurve


def curve() -> YieldCurve:
    return YieldCurve([0.5, 1.0, 2.0], [0.03, 0.032, 0.035])


def test_european_greeks_are_consistent_with_put_call_relationships():
    yc = curve()
    call = EuropeanCall(100, 100, 1.0, 0.20, yc)
    put = EuropeanPut(100, 100, 1.0, 0.20, yc)

    assert np.isclose(call.delta() - put.delta(), 1.0)
    assert np.isclose(call.gamma(), put.gamma())
    assert np.isclose(call.vega(), put.vega())
    assert call.rho() > 0
    assert put.rho() < 0


def test_european_analytic_delta_matches_finite_difference():
    yc = curve()
    option = EuropeanCall(100, 100, 1.0, 0.20, yc)
    h = 1e-3 * option.S0
    up = EuropeanCall(option.S0 + h, option.K, option.T, option.sigma, yc).price()
    down = EuropeanCall(option.S0 - h, option.K, option.T, option.sigma, yc).price()
    finite_difference_delta = (up - down) / (2 * h)

    assert np.isclose(option.delta(), finite_difference_delta, rtol=1e-5)


def test_asian_call_and_put_have_prices_and_full_greeks():
    yc = curve()
    kwargs = dict(
        S0=100,
        K=100,
        T=1.0,
        sigma=0.20,
        yield_curve=yc,
        n_steps=24,
        n_simulations=10_000,
        seed=12,
    )
    call = AsianCall(**kwargs)
    put = AsianPut(**kwargs)

    assert call.price() > 0
    assert put.price() > 0
    assert call.delta() > 0
    assert put.delta() < 0
    assert call.gamma() >= 0
    assert put.gamma() >= 0
    assert set(call.greeks()) == {"delta", "gamma", "vega", "theta", "rho"}


def test_binomial_european_prices_converge_toward_black_scholes():
    from src.binomial_tree import price_binomial_option

    yc = curve()
    european_call = EuropeanCall(100, 100, 1.0, 0.20, yc)
    tree_price = price_binomial_option(
        S0=100,
        K=100,
        r=european_call.r,
        sigma=0.20,
        T=1.0,
        steps=750,
        option_type="call",
        exercise_style="european",
    )

    assert np.isclose(tree_price, european_call.price(), atol=5e-3)


def test_american_put_has_non_negative_early_exercise_premium():
    from src.derivatives import AmericanPut

    yc = curve()
    european_put = EuropeanPut(100, 100, 1.0, 0.20, yc)
    american_put = AmericanPut(100, 100, 1.0, 0.20, yc, steps=500)

    assert american_put.price() >= european_put.price()
    assert np.any(np.isfinite(american_put.tree_analysis().exercise_boundary))


def test_non_dividend_american_call_matches_european_tree_value():
    from src.binomial_tree import price_binomial_option

    inputs = dict(S0=100, K=100, r=0.032, sigma=0.20, T=1.0, steps=400)
    european_tree = price_binomial_option(
        **inputs,
        option_type="call",
        exercise_style="european",
    )
    american_tree = price_binomial_option(
        **inputs,
        option_type="call",
        exercise_style="american",
    )

    assert np.isclose(american_tree, european_tree, atol=1e-12)


def test_american_options_expose_full_greek_interface():
    from src.derivatives import AmericanCall, AmericanPut

    yc = curve()
    call = AmericanCall(100, 100, 1.0, 0.20, yc, steps=250)
    put = AmericanPut(100, 100, 1.0, 0.20, yc, steps=250)

    call_greeks = call.greeks()
    put_greeks = put.greeks()

    assert set(call_greeks) == {"delta", "gamma", "vega", "theta", "rho"}
    assert set(put_greeks) == {"delta", "gamma", "vega", "theta", "rho"}
    assert call_greeks["delta"] > 0
    assert put_greeks["delta"] < 0
    assert call_greeks["vega"] > 0
    assert put_greeks["vega"] > 0


def test_implied_volatility_recovers_known_call_and_put_volatility():
    from src.implied_volatility import solve_implied_volatility

    yc = curve()
    true_volatility = 0.27
    call = EuropeanCall(100, 105, 1.25, true_volatility, yc)
    put = EuropeanPut(100, 95, 0.75, true_volatility, yc)

    call_result = solve_implied_volatility(
        call.price(),
        option_type="call",
        S0=call.S0,
        K=call.K,
        T=call.T,
        yield_curve=yc,
        initial_guess=0.15,
    )
    put_result = put.implied_volatility(put.price(), initial_guess=0.40)

    assert call_result.converged
    assert put_result.converged
    assert np.isclose(call_result.volatility, true_volatility, atol=1e-9)
    assert np.isclose(put_result.volatility, true_volatility, atol=1e-9)
    assert abs(call_result.residual) < 1e-8
    assert abs(put_result.residual) < 1e-8


def test_implied_volatility_rejects_prices_outside_no_arbitrage_bounds():
    import pytest

    from src.implied_volatility import solve_implied_volatility

    yc = curve()
    with pytest.raises(ValueError, match="above the no-arbitrage upper bound"):
        solve_implied_volatility(
            101.0,
            option_type="call",
            S0=100,
            K=100,
            T=1.0,
            yield_curve=yc,
        )


def test_implied_volatility_recovers_synthetic_skew_across_strikes():
    from src.implied_volatility import implied_volatility

    yc = curve()
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    synthetic_volatilities = 0.20 - 0.001 * (strikes - 100.0) + 0.00001 * (strikes - 100.0) ** 2

    recovered = []
    for strike, volatility in zip(strikes, synthetic_volatilities):
        market_price = EuropeanCall(100, strike, 1.0, volatility, yc).price()
        recovered.append(
            implied_volatility(
                market_price,
                option_type="call",
                S0=100,
                K=strike,
                T=1.0,
                yield_curve=yc,
            )
        )

    assert np.allclose(recovered, synthetic_volatilities, atol=1e-8)
