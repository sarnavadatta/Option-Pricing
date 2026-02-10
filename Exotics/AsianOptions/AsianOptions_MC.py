"""
Systematic pricing analysis to study sensitivity to strikes, maturities, or dividend yields under Monte Carlo Asian option pricing (or any pricing model).
Python implementation for Asian option Monte Carlo (with control variates).

It adds a flexible function that:
- Sweeps strike, maturity (T), or dividend yield (q) values
- Prices both call and put, European-style Asian options
- Returns a tidy pandas.DataFrame comparing:
    - Plain Monte Carlo vs Control-Variate estimates
    - Dividend vs No-Dividend cases

Monte Carlo pricing of Asian options (call & put) with dividends and control variates.
Includes a function to sweep strikes, maturities, or dividend yields
and return a pandas.DataFrame of results.

"""

import numpy as np
import pandas as pd
from scipy.stats import norm


    
# Utility functions
def geometric_asian_price(S0, K, T, r, q, sigma, option_type="call"):
    # Analytical geometric-average Asian option price (control variate reference).
    sigma_g = sigma * np.sqrt((2 * T + 1) / (6 * (T + 1)))
    mu_g = (r - q - 0.5 * sigma ** 2) * ((T + 1) / 2) + 0.5 * sigma_g ** 2
    d1 = (np.log(S0 / K) + (mu_g + 0.5 * sigma_g ** 2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)

    if option_type == "call":
        price = np.exp(-r * T) * (S0 * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - S0 * np.exp(mu_g * T) * norm.cdf(-d1))
    return price


    
# Monte Carlo Asian option pricing class
class AsianOptionMonteCarlo:
    def __init__(self, S0, K, T, r, sigma, q=0.0, n_paths=100000, n_steps=252):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = T / n_steps

    def simulate_paths(self):
        # Simulate geometric Brownian motion under risk-neutral dynamics with dividends.
        np.random.seed(42)
        Z = np.random.standard_normal((self.n_steps, self.n_paths))
        S = np.zeros_like(Z)
        S[0, :] = self.S0

        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)

        for t in range(1, self.n_steps):
            S[t, :] = S[t - 1, :] * np.exp(drift + diffusion * Z[t, :])
        return S

    def price(self, option_type="call"):
        # Monte Carlo pricing with control variate (geometric Asian as control).
        S = self.simulate_paths()

        # Arithmetic and geometric averages
        S_arith_mean = np.mean(S, axis=0)
        S_geom_mean = np.exp(np.mean(np.log(S), axis=0))

        if option_type == "call":
            payoff_arith = np.exp(-self.r * self.T) * np.maximum(S_arith_mean - self.K, 0)
            payoff_geom = np.exp(-self.r * self.T) * np.maximum(S_geom_mean - self.K, 0)
        else:
            payoff_arith = np.exp(-self.r * self.T) * np.maximum(self.K - S_arith_mean, 0)
            payoff_geom = np.exp(-self.r * self.T) * np.maximum(self.K - S_geom_mean, 0)

        # --- Monte Carlo Estimate ---
        mc_price = np.mean(payoff_arith)
        mc_std_error = np.std(payoff_arith) / np.sqrt(self.n_paths)

        # --- Control Variate Estimate ---
        geo_price_analytical = geometric_asian_price(self.S0, self.K, self.T, self.r, self.q, self.sigma, option_type)
        cov_xy = np.cov(payoff_arith, payoff_geom)[0, 1]
        var_y = np.var(payoff_geom)
        b_opt = cov_xy / var_y
        cv_price = mc_price - b_opt * (np.mean(payoff_geom) - geo_price_analytical)
        cv_std_error = np.std(payoff_arith - b_opt * payoff_geom) / np.sqrt(self.n_paths)

        return {
            "mc_price": mc_price,
            "mc_std_error": mc_std_error,
            "cv_price": cv_price,
            "cv_std_error": cv_std_error,
            "geo_ref": geo_price_analytical,
            "b_opt": b_opt,
        }


    
# Sweep Function
# Sweep over K (strike), T (maturity), or q (dividend yield) and return a DataFrame
# comparing Monte Carlo and Control Variate prices.
      
def asian_option_sweep(
    S0=100, K_list=None, T_list=None, q_list=None, r=0.05, sigma=0.2,
    option_type="call", n_paths=100000, n_steps=252
):    
    results = []

    if K_list is None:
        K_list = [80, 90, 100, 110, 120]
    if T_list is None:
        T_list = [0.25, 0.5, 1.0]
    if q_list is None:
        q_list = [0.0, 0.05, 0.10]

    for q in q_list:
        for T in T_list:
            for K in K_list:
                model = AsianOptionMonteCarlo(S0, K, T, r, sigma, q, n_paths=n_paths, n_steps=n_steps)
                res = model.price(option_type=option_type)

                results.append({
                    "Dividend Yield (q)": q,
                    "Maturity (T)": T,
                    "Strike (K)": K,
                    "Option Type": option_type,
                    "MC Price": res["mc_price"],
                    "MC StdErr": res["mc_std_error"],
                    "CV Price": res["cv_price"],
                    "CV StdErr": res["cv_std_error"],
                    "Analytic Geometric": res["geo_ref"],
                })

    df = pd.DataFrame(results)
    return df


    
# Usage
if __name__ == "__main__":
    # Sweep dividend yields from 0 to 0.15
    q_range = np.linspace(0, 0.15, 16)
    
    # Calculate call and put prices
    df_call = asian_option_sweep(option_type="call", q_list=q_range)
    df_put = asian_option_sweep(option_type="put", q_list=q_range)

    print("\n=== Asian CALL Options (Dividend Sweep) ===")
    print(df_call.head(5))
    print("\n=== Asian PUT Options (Dividend Sweep) ===")
    print(df_put.head(5))

    # Visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Asian Option Prices vs Dividend Yield (Control Variate Estimates)", fontsize=13, fontweight="bold")

    # Left: CALL prices 
    axes[0].plot(df_call["Dividend Yield (q)"], df_call["CV Price"], "b-o", label="Call (CV)")
    axes[0].set_title("Asian CALL Option")
    axes[0].set_xlabel("Dividend Yield (q)")
    axes[0].set_ylabel("Option Price")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Right: PUT prices 
    axes[1].plot(df_put["Dividend Yield (q)"], df_put["CV Price"], "r-o", label="Put (CV)")
    axes[1].set_title("Asian PUT Option")
    axes[1].set_xlabel("Dividend Yield (q)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
