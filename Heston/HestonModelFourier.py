"""
Heston model option pricing via Carr-Madan FFT.

Produces CALL prices on a grid of strikes using the characteristic function of log S_T under Heston,
and recovers PUT prices via call-put parity.

"""

import numpy as np
from numpy import exp, log, sqrt, pi
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Tuple


    
# Heston characteristic function (vectorized)
def phi_heston(u, S0, T, r, q, v0, kappa, theta, sigma, rho):
    """
    Characteristic function of ln(S_T) under Heston model, vectorized over u.

    Parameters:
      u       : array_like (real or complex) -- Fourier variable
      S0      : spot price
      T       : time to maturity (years)
      r       : risk-free rate
      q       : continuous dividend yield
      v0      : initial variance (sigma_0^2)
      kappa   : mean-reversion speed of variance
      theta   : long-run variance
      sigma   : vol-of-vol
      rho     : correlation between Brownian motions

    Returns:
      phi(u) : complex ndarray
    """
    # ensure complex numpy array
    u = np.asarray(u, dtype=complex)

    # common parameters
    i = 1j
    a = kappa * theta

    # intermediate terms: (kappa - rho*sigma*i*u)
    ku = kappa - rho * sigma * i * u

    # d = sqrt( (kappa - rho*sigma*i*u)^2 + sigma^2 * (i*u + u^2) )
    disc = ku * ku + (sigma * sigma) * (i * u + u * u)
    d = np.sqrt(disc)

    # g = (ku - d) / (ku + d)
    # avoid division by zero (handled by numpy complex)
    g = (ku - d) / (ku + d)

    # exponent terms
    exp_minus_dT = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_minus_dT
    one_minus_g = 1.0 - g

    # C and D coefficients (Gatheral / Heston common representation)
    # C = r*i*u*T + a/sigma^2 * ((ku - d)*T - 2*log( (1 - g*e^{-dT})/(1 - g) ))
    # D = (ku - d)/sigma^2 * (1 - e^{-dT})/(1 - g*e^{-dT})
    # phi = exp(C + D*v0 + i u log S0)
    # note: include -q*T in linear drift for forward price (we put ln S0 factor separately as ln S0)
    # many consistent variants exist; this is standard.
    log_term = np.log(one_minus_g_exp / one_minus_g)
    C = (r - q) * i * u * T + (a / (sigma * sigma)) * ((ku - d) * T - 2.0 * log_term)
    D = (ku - d) / (sigma * sigma) * (1.0 - exp_minus_dT) / one_minus_g_exp

    phi = np.exp(C + D * v0 + 1j * u * np.log(S0))
    return phi



# Carr-Madan FFT for Heston calls
def carr_madan_heston_fft(
    S0: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carr-Madan FFT implementation for Heston model.

    Returns:
      strikes (K grid), call_prices (for those strikes)
    Notes:
      - alpha: dampening factor > 0 for calls (typical 1.0 - 2.5)
      - N: number of FFT points (power of two recommended)
      - eta: spacing in integration variable v
    """
    # integration grid in Fourier domain: v_j = j * eta
    v = np.arange(N) * eta

    # u argument for characteristic function
    # Carr-Madan uses u = v - i*(alpha+1)
    u = v - 1j * (alpha + 1.0)

    # characteristic function evaluated at u
    phi_u = phi_heston(u, S0, T, r, q, v0, kappa, theta, sigma, rho)

    # Carr-Madan integrand factor:
    # psi(v) = e^{-rT} * phi(u) / (alpha^2 + alpha - v^2 + i v (2 alpha + 1))
    denom = (alpha**2 + alpha) - (v**2) + 1j * v * (2.0 * alpha + 1.0)
    fft_func = np.exp(-r * T) * phi_u / denom

    # log-strike grid spacing and center
    lambda_ = (2.0 * pi) / (N * eta)   # spacing in log-strike domain
    b = 0.5 * N * lambda_              # center for log-strike grid

    # integration weights (trapezoidal)
    weights = np.ones(N)
    weights[0] = 0.5
    weights[-1] = 0.5

    # integrand y_j = e^{i v b} * fft_func * eta * weights
    y = np.exp(1j * v * b) * fft_func * eta * weights

    # apply FFT
    Y = np.fft.fft(y)

    # log-strike grid k_m = -b + lambda_ * m
    k = -b + lambda_ * np.arange(N)
    strikes = np.exp(k)   # strikes K_m

    # recover call prices: C(K_m) = e^{-alpha k} / pi * Re(Y_m)
    call_prices = np.exp(-alpha * k) / pi * np.real(Y)

    # clip small negative numerical artifacts
    call_prices = np.maximum(call_prices, 0.0)

    return strikes, call_prices


    
# Interpolate FFT grid to price a single strike
    
def price_at_strike_fft(K_target: float, strikes: np.ndarray, call_prices: np.ndarray, method: str = "linear") -> float:
    """
    Interpolate call_price at a single strike K_target from the FFT grid.
    """
    f = interp1d(strikes, call_prices, kind=method, bounds_error=False, fill_value="extrapolate")
    return float(f(K_target))


    
# Put via call-put parity
def put_via_parity(call_price: float, S0: float, K: float, T: float, r: float, q: float) -> float:
    """
    Compute PUT price from call via parity:
      P = C - S0 e^{-qT} + K e^{-rT}
    """
    return call_price - S0 * exp(-q * T) + K * exp(-r * T)


    
# Example usage and sanity checks
if __name__ == "__main__":
    # Model parameters example
    S0 = 100.0
    K = 100.0
    T = 1.0          # 1 year
    r = 0.05
    # try both q=0 and q>0
    for q in (0.0, 0.03):
        v0 = 0.04      # initial variance (0.2^2)
        kappa = 1.5
        theta = 0.04
        sigma = 0.3
        rho = -0.7

        print("\n--- Heston Carr-Madan FFT (q = {:.2%}) ---".format(q))
        # FFT parameters (tune to balance speed & accuracy)
        alpha = 1.5
        N = 2**12      # 4096 points
        eta = 0.25

        # compute FFT grid
        strikes, call_prices = carr_madan_heston_fft(
            S0=S0, T=T, r=r, q=q, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            alpha=alpha, N=N, eta=eta
        )

        # Interpolate ATM call price
        call_atm = price_at_strike_fft(K, strikes, call_prices, method="linear")
        put_atm = put_via_parity(call_atm, S0, K, T, r, q)

        print(f"Call (FFT)  at K={K:.2f}: {call_atm:.6f}")
        print(f"Put  (parity) at K={K:.2f}: {put_atm:.6f}")

        # Plot price vs strike around ATM
        idx = (strikes > 0.2 * S0) & (strikes < 3.0 * S0)
        plt.figure(figsize=(8, 4))
        plt.plot(strikes[idx], call_prices[idx], label=f"Heston-FFT (q={q:.2f})")
        plt.axvline(K, color="k", linestyle="--", label=f"K={K}")
        plt.xscale("log")
        plt.xlabel("Strike K (log scale)")
        plt.ylabel("Call price")
        plt.title(f"Heston Carr-Madan FFT: Call Price vs Strike (q={q:.2f})")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Note: For validation you can compare to a direct Heston integrator (semi-analytical P1/P2) or QuantLib.
    print("\nDone.")
