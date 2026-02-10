"""
Heston (1993) semi-analytical option pricing implementation (call & put).
Supports continuous dividend yield q (q=0 for no-dividend).

- This uses Heston's P1 and P2 integrals:
    P_j = 1/2 + (1/pi) * \int_0^\infty Re( e^{-i u ln K} * phi_j(u) / (i u) ) du
  with phi_j being characteristic functions with two slightly different parameterizations
  (j = 1, 2). The call price is:
    C = S0 * exp(-qT) * P1 - K * exp(-rT) * P2
  and put via put-call parity.
- Integration uses scipy.integrate.quad on a finite interval [0, U], with a chosen cutoff.
- Numeric tolerances and cutoff may need tuning for extreme parameters.
"""

import numpy as np
from scipy.integrate import quad
from math import log, exp, sqrt, pi
import warnings



# Characteristic function
def _heston_characteristic(u, params):
    """
    Returns the characteristic function phi(u) = E[e^{i u ln S_T}] under Heston model.

    Parameters
    ----------
    u : complex or numpy array of complex
        Fourier variable (can be vectorized).
    params : dict containing model parameters:
        S0   : spot price
        v0   : initial variance
        kappa: mean reversion rate of variance
        theta: long-run variance
        sigma: vol-of-vol (xi)
        rho  : correlation between asset and variance increments
        r    : risk-free rate
        q    : continuous dividend yield
        T    : time to maturity (years)

    Returns
    -------
    phi : complex or numpy array of complex
        Characteristic function value(s).
    """
    # unpack parameters
    S0 = params["S0"]
    v0 = params["v0"]
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    rho = params["rho"]
    r = params["r"]
    q = params["q"]
    T = params["T"]

    # ensure numpy array handling
    u = np.array(u, dtype=complex)

    # i*u
    iu = 1j * u

    # parameters used in formula (Gatheral / Heston common form)
    a = kappa * theta

    # b = kappa - rho * sigma * i u
    b = kappa - rho * sigma * iu

    # d = sqrt( b^2 + sigma^2 * (i u + u^2) )
    # ensure sqrt chooses principal branch (numpy does)
    disc = b * b + (sigma * sigma) * (iu + u * u)
    d = np.sqrt(disc)

    # g = (b - d) / (b + d)
    # avoid division by zero
    g = (b - d) / (b + d)

    # Avoid log of zero in denominator later; where g ~ 1 handle numerically
    # C(t,u) term
    # C = r * i u * T + (a / sigma^2) * ((b - d) * T - 2 * log( (1 - g * exp(-dT)) / (1 - g) ))
    # D = (b - d) / sigma^2 * (1 - exp(-dT)) / (1 - g * exp(-dT))
    # phi = exp(C + D * v0 + i u ln S0 * exp(-q T)?)
    # For drift term, we include (r - q) in the linear factor with ln(S0) included separately

    # To keep consistent with common derivations, put the factor exp(iu*(ln S0 + (r - q) T))
    # inside phi via multiplication.
    # Note: many references put i u (ln S0) in phi and include r * i u * T in C. Both are equivalent.

    # Compute terms carefully
    exp_minus_dT = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_minus_dT
    one_minus_g = 1.0 - g

    # Log term: log( (1 - g e^{-dT}) / (1 - g) )
    # Use numpy.log which works with complex
    log_term = np.log(one_minus_g_exp / one_minus_g)

    C = (r - q) * iu * T + (a / (sigma * sigma)) * ((b - d) * T - 2.0 * log_term)
    D = (b - d) / (sigma * sigma) * (1.0 - exp_minus_dT) / one_minus_g_exp

    # characteristic function
    phi = np.exp(C + D * v0 + iu * np.log(S0))

    return phi



# Two variant characteristic functions phi1, phi2
# (they differ by sign convention in parameter b)

def _phi_j(u, j, params):
    """
    Compute phi_j(u) used in Heston P_j integrals.

    Heston's semi-analytic solution uses two characteristic functions:
    phi_1 uses kappa - rho*sigma*i*u (as implemented in _heston_characteristic)
    phi_2 can be obtained by adjusting u in the same formula (standard approach:
    shift u by -i for phi1 or keep u for phi2 depending on derivation).

    Here we follow a standard, practical approach:
      - For P1 compute phi with argument u - i (i.e. u_shift = u - i)
      - For P2 compute phi with argument u (i.e. u_shift = u)

    Many implementations differ slightly in how they build phi_j; this approach
    matches standard numerical recipes used in practice.

    Parameters
    ----------
    u : float (real) integration variable
    j : 1 or 2
    params : same dict passed to _heston_characteristic

    Returns
    -------
    phi_j(u) : complex
    """
    # convert u (real) to complex argument for phi
    if j == 1:
        u_shift = u - 1j  # u - i
    elif j == 2:
        u_shift = u
    else:
        raise ValueError("j must be 1 or 2")
    return _heston_characteristic(u_shift, params)


   
# Integrand for P_j integral
def _integrand_P(u, j, params, logK):
    """
    Integrand for P_j: Re( e^{-i u ln K} * phi_j(u) / (i u) )
    where u is the real integration variable (>= 0).

    Returns real-valued integrand for integration.
    """
    if u == 0.0:
        return 0.0  # removable singularity handled as limit
    phi = _phi_j(u, j, params)
    numer = np.exp(-1j * u * logK) * phi
    denom = 1j * u
    val = numer / denom
    return np.real(val)


   
# Compute P_j via numerical integration
def compute_Pj(j, params, K, integration_limit=200.0, epsabs=1e-8, epsrel=1e-6):
    """
    Compute probability P_j via numerical integration.

    P_j = 1/2 + (1/pi) * integral_0^∞ integrand du

    Parameters
    ----------
    j : 1 or 2
    params : dict of Heston parameters
    K : strike
    integration_limit : float
        Upper limit for integral (finite truncation)
    epsabs, epsrel : integration tolerances

    Returns
    -------
    P_j : float
    """

    logK = log(K)

    # integrand function for quad (wrap to accept scalar u)
    integrand = lambda u: _integrand_P(u, j, params, logK)

    # integrate from 0 to integration_limit
    # quad may have difficulties at 0, but integrand defined to return 0 there
    integral_value, integral_err = quad(integrand, 0.0, integration_limit, epsabs=epsabs, epsrel=epsrel, limit=200)

    Pj = 0.5 + (1.0 / pi) * integral_value
    return Pj


   
# Heston price function
def heston_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, integration_limit=200.0):
    """
    Compute European CALL and PUT prices under Heston model.

    Parameters
    ----------
    S0 : float - spot price
    K  : float - strike
    T  : float - time to maturity (years)
    r  : float - risk-free rate
    q  : float - continuous dividend yield
    v0 : float - initial variance
    kappa, theta, sigma, rho : Heston parameters
    integration_limit : float - upper limit for u integration (tune as needed)

    Returns
    -------
    call_price, put_price : floats
    """
    # pack params for CF
    params = {
        "S0": S0,
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho,
        "r": r,
        "q": q,
        "T": T,
    }

    # Compute P1 and P2
    # Use moderate tolerances; integration_limit may need to be increased for very small T or extreme params
    try:
        P1 = compute_Pj(1, params, K, integration_limit=integration_limit)
        P2 = compute_Pj(2, params, K, integration_limit=integration_limit)
    except Exception as e:
        warnings.warn(f"Integration failed or is unstable: {e}. Trying relaxed tolerances.")
        P1 = compute_Pj(1, params, K, integration_limit=integration_limit * 2, epsabs=1e-7, epsrel=1e-5)
        P2 = compute_Pj(2, params, K, integration_limit=integration_limit * 2, epsabs=1e-7, epsrel=1e-5)

    # Call price
    call = S0 * exp(-q * T) * P1 - K * exp(-r * T) * P2

    # Put by parity
    put = call - S0 * exp(-q * T) + K * exp(-r * T)

    return float(call), float(put)


   
# usage 
if __name__ == "__main__":
    # Model parameters (example)
    S0 = 100.0
    K = 100.0
    T = 1.0         # 1 year
    r = 0.05
    q = 0.02        # dividend yield (set to 0.0 to test no-dividend case)
    v0 = 0.04       # initial variance (sigma^2), e.g. 0.2^2 = 0.04
    kappa = 1.5     # mean reversion
    theta = 0.04    # long-run variance
    sigma = 0.3     # vol-of-vol
    rho = -0.7      # correlation

    print("Heston parameters example:")
    print(f"S0={S0}, K={K}, T={T}, r={r}, q={q}")
    print(f"v0={v0}, kappa={kappa}, theta={theta}, sigma={sigma}, rho={rho}")

    # compute prices
    call_price, put_price = heston_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, integration_limit=200.0)

    print(f"\nComputed prices (Heston):")
    print(f"Call: {call_price:.6f}")
    print(f"Put : {put_price:.6f}")

