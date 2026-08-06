import numpy as np
import matplotlib.pyplot as plt


class YieldCurve:
    """
    YieldCurve represents a term structure of zero rates and
    provides discount factors for valuation.

    This class is infrastructure: all interest-rate logic
    should live here and be reused by other models.
    """

    def __init__(self, maturities, zero_rates, compounding="continuous"):
        """
        Parameters
        ----------
        maturities : array-like
            Maturities in years (e.g. [0.25, 0.5, 1, 2, 5, 10])
        zero_rates : array-like
            Annualised zero rates as decimals (e.g. 0.045)
        compounding : str
            'continuous' or 'annual'
        """
        self.maturities = np.array(maturities, dtype=float)
        self.zero_rates = np.array(zero_rates, dtype=float)
        self.compounding = compounding

        if len(self.maturities) != len(self.zero_rates):
            raise ValueError("Maturities and zero rates must have the same length")

        # Ensure data is sorted by maturity (important for interpolation)
        order = np.argsort(self.maturities)
        self.maturities = self.maturities[order]
        self.zero_rates = self.zero_rates[order]

    def get_zero_rate(self, T):
        """
        Return the interpolated zero rate for maturity T (in years).
        """
        T = float(T)
        return float(np.interp(T, self.maturities, self.zero_rates))

    def get_discount_factor(self, T):
        """
        Return the discount factor D(T) using the yield curve.
        """
        z = self.get_zero_rate(T)

        if self.compounding == "continuous":
            return np.exp(-z * T)

        elif self.compounding == "annual":
            return 1.0 / (1.0 + z) ** T

        else:
            raise ValueError("Unsupported compounding type")

    def plot(self, max_maturity=None):
        """
        Plot the zero-rate yield curve.
        """
        if max_maturity is None:
            T_grid = self.maturities
        else:
            T_grid = np.linspace(
                self.maturities.min(),
                max_maturity,
                100
            )

        z_grid = [self.get_zero_rate(T) for T in T_grid]

        plt.figure()
        plt.plot(T_grid, z_grid)
        plt.xlabel("Maturity (years)")
        plt.ylabel("Zero rate")
        plt.title("Yield Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.show()