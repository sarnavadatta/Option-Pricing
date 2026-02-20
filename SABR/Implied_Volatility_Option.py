import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# SABR Model using Hagan (2002) Implied Vol Approximation

class SABRModel:
    """
    SABR Model for European Option Pricing
    Uses Hagan et al. (2002) closed-form implied volatility approximation.
    """

    def __init__(self, S0, K, T_days, r, sigma0, alpha, beta, rho, q=0.0):
        """
        Parameters
        ----------
        S0 : float
            Spot price
        K : float
            Strike price
        T_days : int
            Time to maturity in days
        r : float
            Risk-free rate
        sigma0 : float
            Initial volatility level
        alpha : float
            Vol of vol
        beta : float
            CEV elasticity parameter (0 ≤ beta ≤ 1)
        rho : float
            Correlation between asset and volatility
        q : float
            Dividend yield
        """

        self.S0 = S0
        self.K = K
        self.T = T_days / 365
        self.r = r
        self.q = q
        self.sigma0 = sigma0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        # Forward price under dividend yield
        self.F = S0 * np.exp((r - q) * self.T)

    # SABR Implied Volatility (Hagan Formula)

    def sabr_implied_vol(self):

        F = self.F
        K = self.K
        T = self.T
        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        sigma0 = self.sigma0

        if F == K:
            # ATM formula
            term1 = sigma0 / (F ** (1 - beta))
            term2 = (
                ((1 - beta) ** 2 / 24) * (sigma0 ** 2 / F ** (2 - 2 * beta))
                + (rho * beta * alpha * sigma0) / (4 * F ** (1 - beta))
                + (2 - 3 * rho ** 2) * alpha ** 2 / 24
            )
            return term1 * (1 + term2 * T)

        # Non-ATM case
        z = (alpha / sigma0) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)

        x_z = np.log(
            (np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho)
        )

        A = sigma0 / (
            (F * K) ** ((1 - beta) / 2)
            * (1 + ((1 - beta) ** 2 / 24) * np.log(F / K) ** 2
               + ((1 - beta) ** 4 / 1920) * np.log(F / K) ** 4)
        )

        B = 1 + (
            ((1 - beta) ** 2 / 24) * (sigma0 ** 2 / (F * K) ** (1 - beta))
            + (rho * beta * alpha * sigma0) / (4 * (F * K) ** ((1 - beta) / 2))
            + (2 - 3 * rho ** 2) * alpha ** 2 / 24
        ) * T

        return A * (z / x_z) * B

    # Black-Scholes Price using SABR Implied Volatility
    def price_option(self, option_type="call"):

        sigma_sabr = self.sabr_implied_vol()
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * sigma_sabr ** 2) * self.T) / (
            sigma_sabr * np.sqrt(self.T)
        )
        d2 = d1 - sigma_sabr * np.sqrt(self.T)

        if option_type.lower() == "call":
            price = (
                self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1)
                - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            )
        else:
            price = (
                self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
                - self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1)
            )

        return price

    def call_price(self):
        return self.price_option("call")

    def put_price(self):
        return self.price_option("put")

    # Sweep Analysis
 
    def sweep_strikes(self, strike_range):
        results = []
        for K in strike_range:
            model = SABRModel(
                self.S0, K, self.T * 365, self.r,
                self.sigma0, self.alpha, self.beta, self.rho, self.q
            )
            results.append({
                "Strike": K,
                "Call Price": model.call_price(),
                "Put Price": model.put_price(),
                "Implied Vol": model.sabr_implied_vol()
            })

        return pd.DataFrame(results)



# Example Usage

if __name__ == "__main__":

    S0 = 100
    K = 105
    T_days = 180
    r = 0.05
    q = 0.02          # dividend yield
    sigma0 = 0.2      # initial vol
    alpha = 0.4       # vol of vol
    beta = 0.7
    rho = -0.3

    model = SABRModel(S0, K, T_days, r, sigma0, alpha, beta, rho, q)

    print("SABR Option Pricing and Implied Volatility")
    print("Call Price:", model.call_price())
    print("Put Price :", model.put_price())
    print("Implied Vol:", model.sabr_implied_vol())

    # Strike Sweep
    strikes = np.linspace(60, 140, 40)
    df = model.sweep_strikes(strikes)

    print("\nFirst few rows:")
    print(df.head())

    # Plot Call and Put separately
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df["Strike"], df["Call Price"])
    axes[0].set_title("SABR Call Prices")
    axes[0].set_xlabel("Strike")
    axes[0].set_ylabel("Call Price")
    axes[0].grid(True)

    axes[1].plot(df["Strike"], df["Put Price"])
    axes[1].set_title("SABR Put Prices")
    axes[1].set_xlabel("Strike")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
