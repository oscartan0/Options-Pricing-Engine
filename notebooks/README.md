# Notebooks

- `yield_curve_validation.ipynb` validates interpolation and discount factors.
- `european_option_pricing_analysis.ipynb` validates Black–Scholes European option pricing, put–call parity and volatility sensitivity.
- `gbm_monte_carlo_pricing.ipynb` validates risk-neutral GBM simulation, Monte Carlo convergence and arithmetic-average Asian pricing.
- `option_greeks_and_asian_options.ipynb` compares the complete Greek profiles of European and arithmetic-average Asian calls and puts.
- `american_options_binomial_tree.ipynb` validates CRR convergence, quantifies the American put early-exercise premium and visualises the exercise boundary.
- `implied_volatility_analysis.ipynb` validates safeguarded Black–Scholes inversion and recovers a synthetic volatility skew across strikes and maturities.

Core pricing and simulation logic is kept under `src/` so the notebooks remain focused on analysis and interpretation.
