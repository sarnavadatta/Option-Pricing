"""
Merton Jump-Diffusion option pricing using Carr-Madan FFT, extended to produce PUT prices
in two ways:
  - via Call-Put parity (recommended)
  - via a direct FFT inversion for puts (optional; uses a negative dampening alpha)

Notes:
- The main function carr_madan_merton_fft returns arrays of strikes and CALL prices.
- put_prices_via_parity computes puts from those calls using parity.
- carr_madan_merton_fft_put demonstrates how to adapt Carr–Madan for put pricing by choosing
  a negative dampening factor (alpha_put < -1).
- Example at bottom runs sanity checks comparing:
    * FFT call price (interpolated)
    * Series call price (direct summation)
    * Put via parity (from FFT calls)
    * Put via direct-FFT-put (optional)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
from math import log, exp, sqrt, pi
from scipy.special import factorial



# Characteristic function (Merton)
def phi_merton(u, S0, T, r, q, sigma, lam, mu_j, sigma_j):
    """
    Characteristic function of ln(S_T) under Merton jump-diffusion model
    under the risk-neutral measure.

    Parameters
    ----------
    u : complex or ndarray of complex
        Argument of characteristic function (can be vectorized).
    S0 : float
        Spot price.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free rate (annual, continuous compounding).
    q : float
        Continuous dividend yield.
    sigma : float
        Diffusion volatility (annual).
    lam : float
        Jump intensity (expected jumps per year).
    mu_j : float
        Mean of log-jump (ln Y)
    sigma_j : float
        Std dev of log-jump

    Returns
    -------
    phi : complex or np.ndarray[complex]
        Characteristic function value(s).
    """
    
    # kappa = E[Y - 1] where Y = jump multiplier
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    # drift of log-price under risk-neutral measure
    # note: we use (r - q - lam * kappa - 0.5*sigma^2) * T in mu part
    iu = 1j * u
    # diffusion part:
    diffusion_cf = -0.5 * (sigma**2) * (u**2) * T + 1j * u * ( -0.5 * sigma**2 * T )
    # more simply combine below

    # full characteristic function:
    # phi(u) = exp( i u ( ln S0 + (r - q - lam*kappa - 0.5*sigma^2) T )
    #               - 0.5 sigma^2 u^2 T
    #               + lam T ( e^{i u mu_j - 0.5 sigma_j^2 u^2} - 1 ) )
    mu_log = np.log(S0) + (r - q - lam * kappa - 0.5 * sigma**2) * T
    cf = np.exp(1j * u * mu_log - 0.5 * (sigma**2) * (u**2) * T
                + lam * T * (np.exp(1j * u * mu_j - 0.5 * (sigma_j**2) * (u**2)) - 1.0))
    return cf


"""
Carr-Madan FFT for Merton model.

Parameters
----------
S0, T, r, q, sigma, lam, mu_j, sigma_j : model params (see phi_merton)
alpha : float
    Dampening factor (alpha > 0 for calls). Typical 1.0 - 2.5.
N : int
    Number of FFT points (power of 2 recommended). Larger N -> finer strikes.
eta : float
    Spacing in integration domain (v grid). Smaller eta -> larger strike domain.

Returns
-------
strikes : ndarray (N,)
    Strike prices (K grid) corresponding to returned prices.
call_prices : ndarray (N,)
    Call option prices for strikes array.
"""

# Carr-Madan FFT for CALL prices
def carr_madan_merton_fft(
    S0,
    T,
    r,
    q,
    sigma,
    lam,
    mu_j,
    sigma_j,
    alpha=1.5,
    N=2**12,
    eta=0.25,
):

    # Fourier grid
    v = np.arange(N) * eta
    u = v - 1j * (alpha + 1.0)

    # characteristic function evaluated at u
    phi_u = phi_merton(u, S0, T, r, q, sigma, lam, mu_j, sigma_j)

    # Carr-Madan integrand factor
    denom = (alpha**2 + alpha) - (v**2) + 1j * v * (2.0 * alpha + 1.0)
    fft_func = np.exp(-r * T) * phi_u / denom

    # log-strike grid spacing and center
    lambda_ = (2.0 * np.pi) / (N * eta)
    b = 0.5 * N * lambda_

    # integration weights (trapezoid)
    SimpsonW = np.ones(N)
    SimpsonW[0] = 0.5
    SimpsonW[-1] = 0.5

    # integrand (note the factor eta)
    y = np.exp(1j * v * b) * fft_func * eta * SimpsonW

    # FFT
    Y = np.fft.fft(y)

    # log-strike grid and strikes
    k = -b + lambda_ * np.arange(N)
    strikes = np.exp(k)

    # recover call prices
    call_prices = np.exp(-alpha * k) / np.pi * np.real(Y)
    # clip tiny negatives
    call_prices = np.maximum(call_prices, 0.0)
    return strikes, call_prices


 
# Compute puts via Call–Put parity
 
def put_prices_via_parity(call_strikes, call_prices, S0, T, r, q):
    """
    Compute PUT prices from CALL prices on the same strike grid using parity:

      P(K) = C(K) - S0 * e^{-qT} + K * e^{-rT}

    Returns array of put prices aligned with call_strikes.
    """
    call_prices = np.asarray(call_prices)
    K = np.asarray(call_strikes)
    puts = call_prices - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    # numerical safety: puts >= 0
    puts = np.maximum(puts, 0.0)
    return puts

 
# Carr-Madan FFT adapted to produce PUT prices directly
# (uses a negative dampening alpha_put < -1)
def carr_madan_merton_fft_put(
    S0,
    T,
    r,
    q,
    sigma,
    lam,
    mu_j,
    sigma_j,
    alpha_put=-1.5,
    N=2**12,
    eta=0.25,
):
    """
    Carr-Madan FFT returning PUT prices on a grid of strikes by using a negative
    dampening factor alpha_put (< -1). This follows the same pipeline as the
    call FFT but with a different alpha and slightly different denom.

    Note: Using parity is preferred in production; this function is included
    for completeness and educational purposes.
    """
    # alpha_put must satisfy alpha_put < -1 for puts damping (ensures integrability)
    if alpha_put >= -1.0:
        raise ValueError("alpha_put must be < -1 for put dampening.")

    v = np.arange(N) * eta
    u = v - 1j * (alpha_put + 1.0)

    phi_u = phi_merton(u, S0, T, r, q, sigma, lam, mu_j, sigma_j)

    denom = (alpha_put**2 + alpha_put) - (v**2) + 1j * v * (2.0 * alpha_put + 1.0)
    fft_func = np.exp(-r * T) * phi_u / denom

    lambda_ = (2.0 * np.pi) / (N * eta)
    b = 0.5 * N * lambda_
    SimpsonW = np.ones(N)
    SimpsonW[0] = 0.5
    SimpsonW[-1] = 0.5

    y = np.exp(1j * v * b) * fft_func * eta * SimpsonW
    Y = np.fft.fft(y)

    k = -b + lambda_ * np.arange(N)
    strikes = np.exp(k)

    # recover put prices from transformed domain: adjust sign because damping is different
    put_prices = np.exp(-alpha_put * k) / np.pi * np.real(Y)
    put_prices = np.maximum(put_prices, 0.0)
    return strikes, put_prices


 
# Direct Merton series (sanity)
def merton_price_series(S0, K, T, r, q, sigma, lam, mu_j, sigma_j, option_type="call", Nterms=400):
    """
    Direct (series) Merton price for sanity checking.
    """
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    lamT = lam * T
    price = 0.0
    for n in range(Nterms):
        # Poisson term
        pk = np.exp(-lamT) * (lamT ** n) / factorial(n)
        var_ln = sigma**2 * T + n * sigma_j**2
        mu_ln = np.log(S0) + (r - q - lam * kappa - 0.5 * sigma**2) * T + n * mu_j
        if var_ln <= 0:
            ST = exp(mu_ln)
            payoff = max(ST - K, 0) if option_type == "call" else max(K - ST, 0)
            price_k = exp(-r * T) * payoff
        else:
            s_k = sqrt(var_ln)
            d1 = (mu_ln + s_k**2 - np.log(K)) / s_k
            d2 = (mu_ln - np.log(K)) / s_k
            if option_type == "call":
                term1 = exp(mu_ln + 0.5 * s_k**2) * norm.cdf(d1)
                term2 = K * norm.cdf(d2)
                price_k = exp(-r * T) * max(term1 - term2, 0.0)
            else:
                term1 = K * norm.cdf(-d2)
                term2 = exp(mu_ln + 0.5 * s_k**2) * norm.cdf(-d1)
                price_k = exp(-r * T) * max(term1 - term2, 0.0)
        price += pk * price_k
        if pk < 1e-14:
            break
    return float(price)


 
# Interpolate price at single strike K
def price_at_strike_fft(K_target, strikes, prices, method="linear"):
    f = interp1d(strikes, prices, kind=method, bounds_error=False, fill_value="extrapolate")
    return float(f(K_target))


 
# Example usage & comparison
if __name__ == "__main__":
    # model params
    S0 = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.2
    lam = 0.3
    mu_j = -0.1
    sigma_j = 0.25

    # FFT parameters
    alpha = 1.5       # dampening for call FFT
    N = 2**12
    eta = 0.25

    print("Running call-FFT (Carr-Madan)...")
    strikes_call, call_prices = carr_madan_merton_fft(
        S0, T, r, q, sigma, lam, mu_j, sigma_j, alpha=alpha, N=N, eta=eta
    )

    # Interpolate ATM call price from FFT
    K_atm = S0
    call_fft_atm = price_at_strike_fft(K_atm, strikes_call, call_prices)
    call_series_atm = merton_price_series(S0, K_atm, T, r, q, sigma, lam, mu_j, sigma_j, "call", Nterms=500)

    print(f"Call FFT (K={K_atm}): {call_fft_atm:.6f}")
    print(f"Call Series(K={K_atm}): {call_series_atm:.6f}")
    print(f"Abs diff: {abs(call_fft_atm - call_series_atm):.6e}")

    # Compute puts via parity
    put_prices_parity = put_prices_via_parity(strikes_call, call_prices, S0, T, r, q)
    put_parity_atm = price_at_strike_fft(K_atm, strikes_call, put_prices_parity)
    put_series_atm = merton_price_series(S0, K_atm, T, r, q, sigma, lam, mu_j, sigma_j, "put", Nterms=500)

    print(f"\nPut via parity (K={K_atm}): {put_parity_atm:.6f}")
    print(f"Put Series (K={K_atm}): {put_series_atm:.6f}")
    print(f"Abs diff: {abs(put_parity_atm - put_series_atm):.6e}")

    # Optional: direct FFT for puts (may require tuning alpha_put)
    try:
        alpha_put = -1.5
        strikes_put, put_prices_fft_direct = carr_madan_merton_fft_put(
            S0, T, r, q, sigma, lam, mu_j, sigma_j, alpha_put=alpha_put, N=N, eta=eta
        )
        put_fft_direct_atm = price_at_strike_fft(K_atm, strikes_put, put_prices_fft_direct)
        print(f"\nPut direct-FFT (alpha_put={alpha_put}) (K={K_atm}): {put_fft_direct_atm:.6f}")
        print(f"Abs diff (direct-FFT put vs series): {abs(put_fft_direct_atm - put_series_atm):.6e}")
    except Exception as e:
        print("\nDirect FFT put failed or unstable (this can happen for bad alpha_put).")
        print("Error:", e)

    # Plot calls and puts (FFT + parity) around ATM
    idx = (strikes_call > 0.2 * S0) & (strikes_call < 3.0 * S0)
    plt.figure(figsize=(10, 5))
    plt.plot(strikes_call[idx], call_prices[idx], label="Call (FFT)")
    plt.plot(strikes_call[idx], put_prices_parity[idx], label="Put (via parity from FFT)")
    plt.axvline(K_atm, color="k", linestyle="--", label=f"K={K_atm}")
    plt.xscale("log")
    plt.xlabel("Strike K (log scale)")
    plt.ylabel("Option price")
    plt.title("Merton FFT: Call and Put prices (puts via parity)")
    plt.legend()
    plt.grid(True)
    plt.show()
