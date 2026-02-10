"""
Merton Jump Diffusion Model vs Black-Scholes Comparison
Author: [Your Name]
Date: 2025-10-06

This script implements the analytical Merton Jump Diffusion option pricing model,
computes Greeks, and compares it against the standard Black-Scholes model.

It also generates visual plots showing how the jump intensity (λ) changes option prices
relative to Black-Scholes.
"""

  
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


  
# Black-Scholes Formula
def black_scholes_price(S, K, T, r, q, sigma, option_type="call"):
    """Standard Black-Scholes formula for European options."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price


  
# Merton Jump Diffusion Formula
def merton_jump_diffusion_price(S, K, T, r, q, sigma, lam, mu_j, delta_j, option_type="call", N=50):
    """
    Analytical Merton Jump Diffusion option price.

    S : Spot price
    K : Strike price
    T : Time to maturity (years)
    r : Risk-free rate
    q : Dividend yield
    sigma : Diffusion volatility
    lam : Jump intensity (expected jumps per year)
    mu_j : Mean jump size (in log)
    delta_j : Std. dev. of jump size (in log)
    option_type : "call" or "put"
    N : Number of Poisson terms to sum (default 50)
    """

    # Expected relative jump size
    k = np.exp(mu_j + 0.5 * delta_j ** 2) - 1
    price = 0.0

    # Weighted Poisson sum of Black–Scholes prices
    for n in range(N):
        # Poisson probability term
        poisson_prob = np.exp(-lam * T) * ((lam * T) ** n) / np.math.factorial(n)

        # Adjust drift and volatility
        r_n = r - lam * k + (n * (mu_j + 0.5 * delta_j ** 2)) / T
        sigma_n = np.sqrt(sigma ** 2 + (n * delta_j ** 2) / T)

        # Compute Black–Scholes price for adjusted params
        bs_price = black_scholes_price(S, K, T, r_n, q, sigma_n, option_type)
        price += poisson_prob * bs_price

    return price


  
# Greeks via Finite Difference
def merton_greeks(S, K, T, r, q, sigma, lam, mu_j, delta_j, option_type="call"):
    """Compute Greeks numerically using central differences."""
    dS = 0.01 * S
    dr = 0.0001
    dsig = 0.001
    dT = 1 / 365

    base = merton_jump_diffusion_price(S, K, T, r, q, sigma, lam, mu_j, delta_j, option_type)

    # Delta and Gamma
    up = merton_jump_diffusion_price(S + dS, K, T, r, q, sigma, lam, mu_j, delta_j, option_type)
    down = merton_jump_diffusion_price(S - dS, K, T, r, q, sigma, lam, mu_j, delta_j, option_type)
    delta = (up - down) / (2 * dS)
    gamma = (up - 2 * base + down) / (dS ** 2)

    # Vega
    vega = (merton_jump_diffusion_price(S, K, T, r, q, sigma + dsig, lam, mu_j, delta_j, option_type) - base) / dsig

    # Theta
    theta = (merton_jump_diffusion_price(S, K, T - dT, r, q, sigma, lam, mu_j, delta_j, option_type) - base) / dT

    # Rho
    rho = (merton_jump_diffusion_price(S, K, T, r + dr, q, sigma, lam, mu_j, delta_j, option_type) - base) / dr

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}


  
# Comparison Function
def compare_models(S, K, T, r, q, sigma, lam, mu_j, delta_j):
    """
    Compare Merton and Black-Scholes model for call and put options.
    Returns a pandas DataFrame with side-by-side prices.
    """
    bs_call = black_scholes_price(S, K, T, r, q, sigma, "call")
    bs_put = black_scholes_price(S, K, T, r, q, sigma, "put")

    merton_call = merton_jump_diffusion_price(S, K, T, r, q, sigma, lam, mu_j, delta_j, "call")
    merton_put = merton_jump_diffusion_price(S, K, T, r, q, sigma, lam, mu_j, delta_j, "put")

    df = pd.DataFrame({
        "Black-Scholes": {"Call": bs_call, "Put": bs_put},
        "Merton Jump Diffusion": {"Call": merton_call, "Put": merton_put},
        "Difference": {
            "Call": merton_call - bs_call,
            "Put": merton_put - bs_put
        }
    })
    return df


  
# Visualization
def plot_comparison(S, K, T, r, q, sigma, mu_j, delta_j):
    """
    Plot how option price changes with λ (jump intensity)
    compared to the standard Black-Scholes model.
    """
    lambdas = np.linspace(0, 1.0, 40)
    merton_prices = [merton_jump_diffusion_price(S, K, T, r, q, sigma, l, mu_j, delta_j, "call") for l in lambdas]
    bs_price = black_scholes_price(S, K, T, r, q, sigma, "call")

    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, merton_prices, label="Merton Jump Diffusion")
    plt.axhline(bs_price, color="red", linestyle="--", label="Black-Scholes (No Jumps)")
    plt.title("Call Option Price vs Jump Intensity (λ)")
    plt.xlabel("Jump Intensity (λ)")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid(True)
    plt.show()


  
# Usage  
if __name__ == "__main__":
    # Parameters
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.2
    lam = 0.3       # Jump intensity
    mu_j = -0.1     # Mean jump size (in log)
    delta_j = 0.25  # Jump volatility

    # Compare models
    print("=== Merton Jump Diffusion vs Black-Scholes ===")
    df = compare_models(S0, K, T, r, q, sigma, lam, mu_j, delta_j)
    print(df)

    # Calculate Greeks for Merton model
    greeks = merton_greeks(S0, K, T, r, q, sigma, lam, mu_j, delta_j, "call")
    print("\nMerton Model Greeks (Call):")
    for g, v in greeks.items():
        print(f"  {g}: {v:.4f}")

    # Generate plot comparison
    plot_comparison(S0, K, T, r, q, sigma, mu_j, delta_j)
