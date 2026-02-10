"""
Merton Jump Diffusion Model for Option Pricing
This script implements the analytical solution for the Merton Jump Diffusion model,
including the calculation of Greeks and visualizations for sensitivity analysis.
"""

# Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Black-Scholes formula
# Standard Black-Scholes option price

def black_scholes_price(S, K, T, r, q, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price



# Merton Jump Diffusion

"""
Analytical Merton Jump Diffusion option price.

Parameters:
    S : float  - Spot price
    K : float  - Strike price
    T : float  - Time to maturity (in years)
    r : float  - Risk-free rate
    q : float  - Dividend yield
    sigma : float - Volatility of diffusion part
    lam : float - Jump intensity (expected number of jumps per year)
    mu_j : float - Mean of jump size (in log space)
    delta_j : float - Std dev of jump size (in log space)
    option_type : str - "call" or "put"
    N : int - Number of Poisson terms to sum (higher = more accurate)

Returns:
    Option price under Merton Jump Diffusion model
"""

def merton_jump_diffusion_price(S, K, T, r, q, sigma, lam, mu_j, delta_j, option_type="call", N=50):

    # Mean jump adjustment
    k = np.exp(mu_j + 0.5 * delta_j ** 2) - 1
    price = 0.0

    for n in range(N):
        # Poisson probability of n jumps
        poisson_prob = np.exp(-lam * T) * ((lam * T) ** n) / np.math.factorial(n)

        # Adjusted parameters for each term
        r_n = r - lam * k + (n * (mu_j + 0.5 * delta_j ** 2)) / T
        sigma_n = np.sqrt(sigma ** 2 + (n * delta_j ** 2) / T)

        # Black-Scholes price for this adjusted state
        bs_price = black_scholes_price(S, K, T, r_n, q, sigma_n, option_type)
        price += poisson_prob * bs_price

    return price



# Greeks (Finite Difference)
def merton_greeks(S, K, T, r, q, sigma, lam, mu_j, delta_j, option_type="call"):

    dS = 0.01 * S
    dr = 0.0001
    dsig = 0.001
    dT = 1 / 365  # 1 day

    # Base price
    base = merton_jump_diffusion_price(S, K, T, r, q, sigma, lam, mu_j, delta_j, option_type)

    # Delta and Gamma
    price_up = merton_jump_diffusion_price(S + dS, K, T, r, q, sigma, lam, mu_j, delta_j, option_type)
    price_down = merton_jump_diffusion_price(S - dS, K, T, r, q, sigma, lam, mu_j, delta_j, option_type)
    delta = (price_up - price_down) / (2 * dS)
    gamma = (price_up - 2 * base + price_down) / (dS ** 2)

    # Vega
    price_sigma_up = merton_jump_diffusion_price(S, K, T, r, q, sigma + dsig, lam, mu_j, delta_j, option_type)
    vega = (price_sigma_up - base) / dsig

    # Theta
    price_T_down = merton_jump_diffusion_price(S, K, T - dT, r, q, sigma, lam, mu_j, delta_j, option_type)
    theta = (price_T_down - base) / dT

    # Rho
    price_r_up = merton_jump_diffusion_price(S, K, T, r + dr, q, sigma, lam, mu_j, delta_j, option_type)
    rho = (price_r_up - base) / dr

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}



# Plotting Sensitivity
"""
Generate plots for:
1. Price vs Strike
2. Price vs Dividend yield q
3. Price vs Jump intensity λ
"""
def plot_sensitivity(S, K, T, r, q, sigma, lam, mu_j, delta_j, option_type="call"):

    # Price vs Strike 
    Ks = np.linspace(60, 140, 40)
    prices_K = [merton_jump_diffusion_price(S, k, T, r, q, sigma, lam, mu_j, delta_j, option_type) for k in Ks]

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.plot(Ks, prices_K)
    plt.title("Price vs Strike (K)")
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Option Price")

    # Price vs Dividend yield 
    qs = np.linspace(0, 0.1, 30)
    prices_q = [merton_jump_diffusion_price(S, K, T, r, qv, sigma, lam, mu_j, delta_j, option_type) for qv in qs]
    plt.subplot(1, 3, 2)
    plt.plot(qs, prices_q, color='orange')
    plt.title("Price vs Dividend Yield (q)")
    plt.xlabel("Dividend Yield")
    plt.ylabel("Option Price")

    # Price vs Jump Intensity 
    lambdas = np.linspace(0, 1, 30)
    prices_lam = [merton_jump_diffusion_price(S, K, T, r, q, sigma, lv, mu_j, delta_j, option_type) for lv in lambdas]
    plt.subplot(1, 3, 3)
    plt.plot(lambdas, prices_lam, color='green')
    plt.title("Price vs Jump Intensity (λ)")
    plt.xlabel("Jump Intensity λ")
    plt.ylabel("Option Price")

    plt.tight_layout()
    plt.show()



# Example Usage

if __name__ == "__main__":
    # Parameters
    S0 = 100        # Spot price
    K = 100         # Strike
    T = 1.0         # 1 year
    r = 0.05        # Risk-free rate
    q = 0.02        # Dividend yield
    sigma = 0.2     # Volatility
    lam = 0.3       # Jump intensity
    mu_j = -0.1     # Mean jump size
    delta_j = 0.25  # Jump volatility

    option_type = "call"

    # Calculate price and Greeks
    price = merton_jump_diffusion_price(S0, K, T, r, q, sigma, lam, mu_j, delta_j, option_type)
    greeks = merton_greeks(S0, K, T, r, q, sigma, lam, mu_j, delta_j, option_type)

    # Print results
    print("=== Merton Jump Diffusion Model ===")
    print(f"Option Type: {option_type.capitalize()}")
    print(f"Option Price: {price:.4f}")
    print("Greeks:")
    for g, v in greeks.items():
        print(f"  {g}: {v:.4f}")

    # Generate plots
    plot_sensitivity(S0, K, T, r, q, sigma, lam, mu_j, delta_j, option_type)
