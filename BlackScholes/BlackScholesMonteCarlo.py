"""
Monte Carlo Simulation for Option Pricing under the Black-Scholes model
including continuous dividend yield.

- EuropeanOptionMonteCarlo: Standard European option pricing via Monte Carlo
- AmericanOptionMonteCarlo: Least-Squares Monte Carlo (LSM) for early exercise options
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# Base Class for Monte Carlo Simulation 

class BaseMonteCarloModel:
    """
    Base class for simulating underlying price paths under Black-Scholes assumptions.
    """

    def __init__(self, S0, K, T_days, r, sigma, q=0.0, N_paths=10000, seed=42):
        """
        Initialize parameters for Monte Carlo simulation.

        Parameters
        ----------
        S0 : float
            Initial underlying price
        K : float
            Strike price
        T_days : int
            Time to maturity in days
        r : float
            Annualized risk-free rate
        sigma : float
            Annualized volatility of underlying
        q : float
            Continuous dividend yield
        N_paths : int
            Number of simulated paths
        seed : int
            Random seed for reproducibility
        """
        self.S0 = S0
        self.K = K
        self.T = T_days / 365
        self.r = r
        self.q = q
        self.sigma = sigma
        self.N_paths = N_paths
        self.steps = T_days
        self.dt = self.T / self.steps
        self.seed = seed
        self.simulated_paths = None

    def simulate_paths(self):
        """
        Simulate price paths following geometric Brownian motion:
        S_{t+1} = S_t * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        """
        np.random.seed(self.seed)
        S = np.zeros((self.steps + 1, self.N_paths))
        S[0] = self.S0

        for t in range(1, self.steps + 1):
            Z = np.random.standard_normal(self.N_paths)
            S[t] = S[t - 1] * np.exp(
                (self.r - self.q - 0.5 * self.sigma**2) * self.dt
                + self.sigma * np.sqrt(self.dt) * Z
            )

        self.simulated_paths = S
        return S

    def plot_paths(self, n_paths_to_plot=20):
        """Plot simulated price paths."""
        if self.simulated_paths is None:
            raise ValueError("Run simulate_paths() first.")

        plt.figure(figsize=(12, 7))
        plt.plot(self.simulated_paths[:, :n_paths_to_plot])
        plt.axhline(self.K, color="red", linestyle="--", label="Strike Price")
        plt.xlabel("Days to Maturity")
        plt.ylabel("Underlying Price")
        plt.title(f"Monte Carlo Simulated Paths (q={self.q})")
        plt.legend()
        plt.grid(True)
        plt.show()



# European Option Monte Carlo 

class EuropeanOptionMonteCarlo(BaseMonteCarloModel):
    """Monte Carlo pricing for European-style options."""

    def price_call(self):
        """Compute call price: E[max(S_T - K, 0)] discounted by e^{-rT}."""
        if self.simulated_paths is None:
            self.simulate_paths()

        ST = self.simulated_paths[-1]
        payoff = np.maximum(ST - self.K, 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)

    def price_put(self):
        """Compute put price: E[max(K - S_T, 0)] discounted by e^{-rT}."""
        if self.simulated_paths is None:
            self.simulate_paths()

        ST = self.simulated_paths[-1]
        payoff = np.maximum(self.K - ST, 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)



# American Option Monte Carlo (LSM) 

class AmericanOptionMonteCarlo(BaseMonteCarloModel):
    """Least Squares Monte Carlo (Longstaff-Schwartz) for American options."""

    def price_option(self, option_type="put"):
        """
        Compute price for American call or put using LSM.

        Parameters
        ----------
        option_type : str
            'call' or 'put'
        """
        if self.simulated_paths is None:
            self.simulate_paths()

        S = self.simulated_paths
        M, N = S.shape  # steps, paths
        dt = self.dt

        # Payoffs
        if option_type == "call":
            payoff = np.maximum(S - self.K, 0)
        elif option_type == "put":
            payoff = np.maximum(self.K - S, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        cashflows = payoff[-1]

        # Backward induction for early exercise
        for t in range(M - 2, 0, -1):
            itm = payoff[t] > 0
            X = S[t, itm]
            Y = cashflows[itm] * np.exp(-self.r * dt)

            if len(X) == 0:
                continue

            # Regression (basis: [1, S, S^2])
            A = np.vstack([np.ones_like(X), X, X**2]).T
            model = LinearRegression().fit(A, Y)
            continuation = model.predict(np.vstack([np.ones_like(S[t]), S[t], S[t]**2]).T)

            exercise = payoff[t] > continuation
            cashflows[exercise] = payoff[t, exercise]
            cashflows[~exercise] = cashflows[~exercise] * np.exp(-self.r * dt)

        price = np.exp(-self.r * dt) * np.mean(cashflows)
        return price

# def compare_dividend_effect(S0, K, T_days, r, sigma, q, N_paths=20000):
#     """
#     Compare option prices for European and American options
#     with and without dividends (q = 0 vs q > 0).

#     Returns a Pandas DataFrame.
#     """
#     results = []

#     for q_val in [0.0, q]:
#         # European
#         euro = EuropeanOptionMonteCarlo(S0, K, T_days, r, sigma, q_val, N_paths)
#         euro.simulate_paths()
#         euro_call = euro.price_call()
#         euro_put = euro.price_put()

#         # American
#         amer = AmericanOptionMonteCarlo(S0, K, T_days, r, sigma, q_val, N_paths)
#         amer.simulate_paths()
#         am_call = amer.price_option("call")
#         am_put = amer.price_option("put")

#         results.append({
#             "Dividend Yield (q)": q_val,
#             "European Call": euro_call,
#             "European Put": euro_put,
#             "American Call": am_call,
#             "American Put": am_put
#         })

#     df = pd.DataFrame(results)
#     df["Dividend Scenario"] = ["No Dividend", "With Dividend"]
#     df = df[
#         ["Dividend Scenario", "Dividend Yield (q)",
#          "European Call", "European Put", "American Call", "American Put"]
#     ]
#     return df


# Sensitivity Analysis: Dividend Yield Sweep 

def dividend_sensitivity_analysis(S0, K, T_days, r, sigma, N_paths=20000):
    """
    Computes option prices for q = 0.00 to 0.15 with 0.01 increments.
    """
    q_values = np.round(np.arange(0.00, 0.16, 0.01), 2)
    results = []

    for q in q_values:
        # European
        euro = EuropeanOptionMonteCarlo(S0, K, T_days, r, sigma, q, N_paths)
        euro.simulate_paths()
        euro_call = euro.price_call()
        euro_put = euro.price_put()

        # American
        amer = AmericanOptionMonteCarlo(S0, K, T_days, r, sigma, q, N_paths)
        amer.simulate_paths()
        am_call = amer.price_option("call")
        am_put = amer.price_option("put")

        results.append({
            "Dividend Yield (q)": q,
            "European Call": euro_call,
            "European Put": euro_put,
            "American Call": am_call,
            "American Put": am_put
        })

    df = pd.DataFrame(results)
    return df



# Visualization

def plot_dividend_sensitivity(df):
    """Plot the sensitivity of option prices to dividend yield."""
    plt.figure(figsize=(12, 7))
    plt.plot(df["Dividend Yield (q)"], df["European Call"], label="European Call", lw=2)
    plt.plot(df["Dividend Yield (q)"], df["American Call"], label="American Call", lw=2)
    plt.plot(df["Dividend Yield (q)"], df["European Put"], label="European Put", lw=2)
    plt.plot(df["Dividend Yield (q)"], df["American Put"], label="American Put", lw=2)
    plt.xlabel("Dividend Yield (q)")
    plt.ylabel("Option Price")
    plt.title("Dividend Yield Sensitivity: European vs American Options")
    plt.legend()
    plt.grid(True)
    plt.show()



# Usage 

if __name__ == "__main__":
    # Parameters
    S0 = 100
    K = 100
    T_days = 365
    r = 0.05
    sigma = 0.2
    q = 0.03  # continuous dividend yield
    N = 20000

    # print("\n=== EUROPEAN OPTION MONTE CARLO (with Dividends) ===")
    # euro = EuropeanOptionMonteCarlo(S0, K, T_days, r, sigma, q, N)
    # euro.simulate_paths()
    # euro.plot_paths(15)

    # euro_call = euro.price_call()
    # euro_put = euro.price_put()
    # print(f"European Call Price: {euro_call:.4f}")
    # print(f"European Put  Price: {euro_put:.4f}")

    # print("\n=== AMERICAN OPTION MONTE CARLO (with Dividends) ===")
    # amer = AmericanOptionMonteCarlo(S0, K, T_days, r, sigma, q, N)
    # amer.simulate_paths()

    # am_call = amer.price_option("call")
    # am_put = amer.price_option("put")

    # print(f"American Call Price (LSM): {am_call:.4f}")
    # print(f"American Put  Price (LSM): {am_put:.4f}")

    # Comparison
    df_div = dividend_sensitivity_analysis(S0, K, T_days, r, sigma, N)
    print("\n OPTION PRICE SENSITIVITY TO DIVIDEND YIELD (q)")
    print(df_div.round(4).to_string(index=False))

    # Plot
    plot_dividend_sensitivity(df_div)