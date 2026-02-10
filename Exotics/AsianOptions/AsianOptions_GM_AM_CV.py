'''
ASIAN OPTION PRICING MODULE:
This module implements pricing models for Asian options under the 
Black-Scholes framework, including analytical, Monte Carlo, and 
control variate approaches. It supports both call and put options 
and allows inclusion of continuous dividend yields.


FUNCTIONS OVERVIEW:
1. geometric_asian_price(S0, K, T, r, sigma, q=0.0, option_type="call")
   - Analytical (closed-form) pricing of a geometric Asian option.

2. monte_carlo_asian(S0, K, T, r, sigma, q=0.0, option_type="call",
                     n_paths=10000, n_steps=252)
   - Monte Carlo simulation for arithmetic Asian option pricing.
     Simulates log-normal price paths and estimates option value by 
     averaging discounted payoffs.

3. control_variate_asian(S0, K, T, r, sigma, q=0.0, option_type="call",
                         n_paths=10000, n_steps=252)
   - Variance reduction method using geometric Asian option as a 
     control variate. Produces more stable and accurate results 
     compared to the standard Monte Carlo approach.

4. asian_option_sweep(S0, K_list, T_list, q_list, r, sigma, option_type="call")
   - Loops through combinations of strike prices, maturities, and 
     dividend yields. Returns a clean pandas.DataFrame containing 
     Monte Carlo, control variate, and analytical geometric prices.


MATHEMATICAL BACKGROUND

Asian options are path-dependent derivatives whose payoff depends on 
the average price of the underlying asset over a certain period.

- Arithmetic average (A_arith) = (1/n) * Σ S_i  → no closed form
- Geometric average (A_geom)  = (Π S_i)^(1/n) → has analytical solution

The geometric Asian option serves as a control variate because it 
shares a strong correlation with the arithmetic Asian payoff.

- The control variate method is significantly more efficient for 
  pricing arithmetic Asian options.
- The sweep function allows for comparative studies across parameters 
  such as strike price, maturity, and dividend yield.
'''

import numpy as np
import pandas as pd
from scipy.stats import norm

   
# Geometric Asian option price (analytical)
def geometric_asian_price(S0, K, T, r, sigma, q=0.0, option_type="call"):
    # Analytical price of a geometric Asian option under Black-Scholes (dividend optional).
    sigma_hat = sigma * np.sqrt((2*T + 1) / (6*(T+1)))
    mu_hat = 0.5 * sigma_hat**2 + (r - q - 0.5 * sigma**2) * (T + 1) / (2*(T))
    d1 = (np.log(S0 / K) + (mu_hat + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)
    disc_factor = np.exp(-r*T)
    fwd_factor = np.exp(-q*T)

    if option_type == "call":
        price = fwd_factor * S0 * np.exp(mu_hat * T) * norm.cdf(d1) - K * disc_factor * norm.cdf(d2)
    else:
        price = K * disc_factor * norm.cdf(-d2) - fwd_factor * S0 * np.exp(mu_hat * T) * norm.cdf(-d1)
    return price


   
# Monte Carlo simulation for arithmetic Asian option   
def monte_carlo_asian(S0, K, T, r, sigma, q=0.0, option_type="call", n_paths=10000, n_steps=252):
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # simulate log-normal paths
    Z = np.random.normal(size=(n_paths, n_steps))
    S = np.zeros((n_paths, n_steps+1))
    S[:, 0] = S0
    for t in range(1, n_steps+1):
        S[:, t] = S[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])

    # arithmetic average
    A = S[:, 1:].mean(axis=1)

    # payoff
    if option_type == "call":
        payoff = np.maximum(A - K, 0)
    else:
        payoff = np.maximum(K - A, 0)

    price = np.exp(-r * T) * np.mean(payoff)
    return price


   
# Control variate estimator   
def control_variate_asian(S0, K, T, r, sigma, q=0.0, option_type="call", n_paths=10000, n_steps=252):
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = np.random.normal(size=(n_paths, n_steps))
    S = np.zeros((n_paths, n_steps+1))
    S[:, 0] = S0
    for t in range(1, n_steps+1):
        S[:, t] = S[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])

    A_arith = S[:, 1:].mean(axis=1)
    A_geom = np.exp(np.mean(np.log(S[:, 1:]), axis=1))

    if option_type == "call":
        payoff_arith = np.maximum(A_arith - K, 0)
        payoff_geom = np.maximum(A_geom - K, 0)
    else:
        payoff_arith = np.maximum(K - A_arith, 0)
        payoff_geom = np.maximum(K - A_geom, 0)

    # control variate coefficient
    cov_xy = np.cov(payoff_arith, payoff_geom)[0, 1]
    var_y = np.var(payoff_geom)
    b = cov_xy / var_y

    # analytical price of geometric asian
    geo_price = geometric_asian_price(S0, K, T, r, sigma, q, option_type)
    control_payoff = payoff_arith - b * (payoff_geom - geo_price)

    price = np.exp(-r * T) * np.mean(control_payoff)
    return price

   
# Sweep function: strikes / T / dividend yield   
def asian_option_sweep(S0, K_list, T_list, q_list, r, sigma, option_type="call"):
    results = []

    for K in K_list:
        for T in T_list:
            for q in q_list:
                mc_price = monte_carlo_asian(S0, K, T, r, sigma, q, option_type)
                cv_price = control_variate_asian(S0, K, T, r, sigma, q, option_type)
                geo_price = geometric_asian_price(S0, K, T, r, sigma, q, option_type)

                results.append({
                    "K": K,
                    "T": T,
                    "q": q,
                    "Option": option_type,
                    "MC_Price": mc_price,
                    "CV_Price": cv_price,
                    "Geometric_Control": geo_price
                })

    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    S0 = 100
    r = 0.05
    sigma = 0.2

    K_list = np.arange(80, 121, 10)          # sweep strikes
    T_list = [0.5, 1.0, 2.0]                 # sweep maturities
    q_list = np.arange(0.0, 0.16, 0.05)      # sweep dividend yields

    df_call = asian_option_sweep(S0, K_list, T_list, q_list, r, sigma, option_type="call")
    df_put = asian_option_sweep(S0, K_list, T_list, q_list, r, sigma, option_type="put")

    df = pd.concat([df_call, df_put], ignore_index=True)
    print(df)
