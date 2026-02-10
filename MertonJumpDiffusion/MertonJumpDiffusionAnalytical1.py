"""
merton_jump_diffusion.py

Merton (1976) Jump-Diffusion Model - option pricing (analytic series) + Greeks (finite differences).

Usage:
    python merton_jump_diffusion.py
Notes:
- This implementation prices European options under Merton's jump-diffusion:
    dS_t / S_{t-} = (r - lambda*kappa) dt + sigma dW_t + (Y - 1) dN_t
  where ln Y ~ N(mu_j, sigma_j^2), N_t is Poisson(lambda).
- The option price is:
    C = sum_{k=0..inf} PoissonPMF(k; lambda T) * Price_conditional_on_k
  where Price_conditional_on_k is the Black-type price computed from the conditional
  lognormal distribution of S_T given k jumps.
- Greeks are computed using central finite differences by default.
"""

from math import exp, log, sqrt
import numpy as np
from scipy.stats import norm, poisson
from typing import Dict


class MertonJumpDiffusion:
    """
    Merton jump-diffusion model for European option pricing (analytical series).

    Parameters
    ----------
    S0 : float
        Current underlying spot price.
    K : float
        Strike price.
    days_to_maturity : int
        Time to maturity in days.
    r : float
        Annual risk-free rate (continuous compounding).
    sigma : float
        Diffusion volatility (annual).
    lam : float
        Jump intensity (lambda) — expected number of jumps per year.
    mu_j : float
        Mean of ln(jump multiplier) (i.e. mean of ln(Y)).
    sigma_j : float
        Std. dev. of ln(jump multiplier).
    tol : float, optional
        Truncation tolerance for Poisson tail when summing series (default 1e-12).
    max_k : int or None, optional
        Hard cap on k summation. If None, computed automatically from lam*T and tol.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        days_to_maturity: int,
        r: float,
        sigma: float,
        lam: float,
        mu_j: float,
        sigma_j: float,
        tol: float = 1e-12,
        max_k: int = None,
    ):
        # store params
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(days_to_maturity) / 365.0
        self.r = float(r)
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.mu_j = float(mu_j)
        self.sigma_j = float(sigma_j)
        self.tol = float(tol)

        # Poisson sum truncation cap
        self.max_k_user = max_k

        # Precompute kappa = E[Y - 1] where Y is multiplicative jump size
        # With ln Y ~ N(mu_j, sigma_j^2), E[Y] = exp(mu_j + 0.5 sigma_j^2)
        self.kappa = exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0

        # Safety checks
        if self.T <= 0:
            raise ValueError("T must be > 0 (days_to_maturity > 0).")
        if self.sigma < 0 or self.sigma_j < 0:
            raise ValueError("Volatilities must be non-negative.")
        if self.lam < 0:
            raise ValueError("Lambda (jump intensity) must be non-negative.")

     
    # Internal helpers
     
    def _truncate_k(self):
        """
        Return truncation index k_max for Poisson sum.

        Strategy:
        - If user-provided max_k, use it.
        - Otherwise compute using mean = lam*T and tail until cumulative Poisson
          probability >= (1 - tol). Also add a small safety margin.
        """
        if self.max_k_user is not None:
            return int(self.max_k_user)

        mean = self.lam * self.T
        # Start around mean, extend until cumulative probability >= 1 - tol
        k = max(0, int(np.floor(mean)))
        cdf = poisson.cdf(k, mean)
        # Expand to the right until tail small enough
        upper = k
        while cdf < 1.0 - self.tol:
            upper += 1
            cdf = poisson.cdf(upper, mean)
            # safety cap in case lam*T extremely large
            if upper > mean + 50 + 10 * sqrt(max(mean, 1.0)):
                break
        # add small margin
        return int(upper + 5)

    def _price_conditional_k(self, k: int) -> float:
        """
        Price the European option conditional on exactly k jumps.

        Derivation:
        - Conditional on k jumps, ln S_T is normal with:
            mean = ln(S0) + (r - lambda*kappa - 0.5*sigma^2)*T + k * mu_j
            variance = sigma^2 * T + k * sigma_j^2
        - So S_T | k ~ LogNormal with parameters (mu_k, var_k).
        - Option price = e^{-rT} * E[(S_T - K)^+ | k] which can be computed similarly
          to Black-style closed-form using normal cdf.
        """
        # if extremely large k, variance increases; handle numerically
        T = self.T
        sigma2_T = self.sigma**2 * T + k * self.sigma_j**2  # variance of ln S_T
        if sigma2_T <= 0:
            # degenerate case (zero variance) -> option payoff equals intrinsic on expected S_T
            # fallback: price = discounted intrinsic if S deterministic
            mu_k = (
                log(self.S0)
                + (self.r - self.lam * self.kappa - 0.5 * self.sigma**2) * T
                + k * self.mu_j
            )
            S_T_exp = exp(mu_k)  # deterministic since variance zero
            return exp(-self.r * T) * max(S_T_exp - self.K, 0.0)

        s_k = sqrt(sigma2_T)  # std dev of ln S_T
        mu_k = (
            log(self.S0)
            + (self.r - self.lam * self.kappa - 0.5 * self.sigma**2) * T
            + k * self.mu_j
        )  # mean of ln S_T

        # d1, d2 using lognormal-moment form
        # E[S_T | k] = exp(mu_k + 0.5 * s_k^2)
        # price_k = e^{-rT} * (E[S_T * 1_{S_T>K}] - K * P(S_T > K))
        # Let d = (mu_k - ln K) / s_k
        # Then P(S_T > K) = 1 - Phi((ln K - mu_k)/s_k) = Phi((mu_k - ln K)/s_k)
        # And E[S_T * 1_{S_T>K}] = exp(mu_k + 0.5 s_k^2) * Phi( (mu_k + s_k^2 - ln K)/s_k )
        lnK = log(self.K)
        d1 = (mu_k + s_k**2 - lnK) / s_k
        d2 = (mu_k - lnK) / s_k

        term1 = exp(mu_k + 0.5 * s_k**2) * norm.cdf(d1)
        term2 = self.K * norm.cdf(d2)
        price_k = exp(-self.r * T) * max(term1 - term2, 0.0)
        return float(price_k)

     
    # Public price method
    def price(self, option_type: str = "call") -> float:
        """
        Compute the European option price under Merton's model by truncating the
        Poisson-weighted series.

        Parameters
        ----------
        option_type : "call" or "put"
            Option type. (Currently only European options are supported in closed form.)

        Returns
        -------
        price : float
            Merton model price.
        """
        if option_type.lower() not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")

        # For put, use put-call parity: P = C - S0*e^{-qT} + K*e^{-rT}
        # However in Merton's model q is not explicitly stored (we used continuous jumps),
        # so we price call directly; for put, we can compute call then parity if needed.
        # Simpler: price the call distribution for (S_T) above, and for put use same integral:
        # We'll implement both via conditional integration (price formula above yields call price).
        # For puts, use put-call parity with forward adjustment under Merton model:
        # but since we have a direct closed-form for call given conditional distribution, we compute both by
        # using call price and parity:
        # P = C - S0 * e^{-lambda*T * ???} + K e^{-rT} ???  <-- That becomes messy.
        # Instead, compute put via the same conditional method by replacing payoff with (K - S_T)+:
        # We implement both by direct integration (same machinery) by switching payoff inside _price_conditional_k.
        # To keep the code simple and robust, we'll implement call and put separately by adapting the formula.

        # For calls we use _price_conditional_k which returns call price conditional on k jumps.
        # For puts we can use a related closed-form via:
        # price_put_k = exp(-rT) * ( K * Phi(-d2) - exp(mu_k + 0.5 s_k^2) * Phi(-d1) )
        # We'll implement both in one loop.

        k_max = self._truncate_k()
        lamT = self.lam * self.T

        price = 0.0
        # sum Poisson weight * conditional price
        for k in range(0, k_max + 1):
            # Poisson probability for k jumps
            pk = poisson.pmf(k, lamT)
            if pk == 0.0:
                continue

            # compute conditional lognormal parameters
            sigma2_T = self.sigma**2 * self.T + k * self.sigma_j**2
            if sigma2_T <= 0:
                # degenerate case
                mu_k = (
                    log(self.S0)
                    + (self.r - self.lam * self.kappa - 0.5 * self.sigma**2) * self.T
                    + k * self.mu_j
                )
                S_T_exp = exp(mu_k)
                if option_type.lower() == "call":
                    price_k = max(S_T_exp - self.K, 0.0) * exp(-self.r * self.T)
                else:
                    price_k = max(self.K - S_T_exp, 0.0) * exp(-self.r * self.T)
            else:
                s_k = sqrt(sigma2_T)
                mu_k = (
                    log(self.S0)
                    + (self.r - self.lam * self.kappa - 0.5 * self.sigma**2) * self.T
                    + k * self.mu_j
                )
                lnK = log(self.K)
                d1 = (mu_k + s_k**2 - lnK) / s_k
                d2 = (mu_k - lnK) / s_k

                if option_type.lower() == "call":
                    term1 = exp(mu_k + 0.5 * s_k**2) * norm.cdf(d1)
                    term2 = self.K * norm.cdf(d2)
                    price_k = exp(-self.r * self.T) * max(term1 - term2, 0.0)
                else:  # put
                    # put formula via lognormal integrals:
                    term1 = self.K * norm.cdf(-d2)
                    term2 = exp(mu_k + 0.5 * s_k**2) * norm.cdf(-d1)
                    price_k = exp(-self.r * self.T) * max(term1 - term2, 0.0)

            price += pk * price_k

        # Consider tail mass beyond k_max (rare) — add approximation using last term * tail mass (optional)
        tail = 1.0 - poisson.cdf(k_max, lamT)
        if tail > 0:
            # crude upper-bound correction: assume price_k for k=k_max is representative
            # This is conservative if price_k decreases with k (often true).
            # Compute price_k at k_max (again)
            pk_last = poisson.pmf(k_max, lamT)
            if pk_last > 0:
                # reuse last computed price_k if available, else compute fast
                # compute last price_k exactly:
                sigma2_T = self.sigma**2 * self.T + k_max * self.sigma_j**2
                if sigma2_T <= 0:
                    mu_k = (
                        log(self.S0)
                        + (self.r - self.lam * self.kappa - 0.5 * self.sigma**2) * self.T
                        + k_max * self.mu_j
                    )
                    price_k_last = exp(-self.r * self.T) * max(exp(mu_k) - self.K, 0.0) if option_type.lower() == "call" else exp(-self.r * self.T) * max(self.K - exp(mu_k), 0.0)
                else:
                    s_k = sqrt(sigma2_T)
                    mu_k = (
                        log(self.S0)
                        + (self.r - self.lam * self.kappa - 0.5 * self.sigma**2) * self.T
                        + k_max * self.mu_j
                    )
                    lnK = log(self.K)
                    d1 = (mu_k + s_k**2 - lnK) / s_k
                    d2 = (mu_k - lnK) / s_k
                    if option_type.lower() == "call":
                        price_k_last = exp(-self.r * self.T) * (exp(mu_k + 0.5 * s_k**2) * norm.cdf(d1) - self.K * norm.cdf(d2))
                    else:
                        price_k_last = exp(-self.r * self.T) * (self.K * norm.cdf(-d2) - exp(mu_k + 0.5 * s_k**2) * norm.cdf(-d1))
                # add tail approximation
                price += tail * price_k_last

        return float(price)

     
    # Greeks via finite differences
    def greeks(
        self,
        option_type: str = "call",
        eps: Dict[str, float] = None,
        method: str = "central",
    ) -> Dict[str, float]:
        """
        Compute Greeks (Delta, Gamma, Vega, Theta, Rho) via finite differences.

        Parameters
        ----------
        option_type : "call" or "put"
            Option type.
        eps : dict or None
            Step sizes for finite differences. Default values are used if None:
            - eps_s: relative step for S0 (1e-4)
            - eps_sigma: relative step for sigma (1e-4)
            - eps_t: absolute days step for theta (1 day)
            - eps_r: absolute step for interest rate (1e-4)
        method : "central" or "forward"
            FD scheme. Central is recommended.

        Returns
        -------
        dict containing Delta, Gamma, Vega, Theta (per year), Rho.
        """
        if eps is None:
            eps = {}
        s0 = self.S0
        sigma = self.sigma
        r = self.r

        eps_s = eps.get("eps_s", max(1e-4 * s0, 1e-6))
        eps_sigma = eps.get("eps_sigma", max(1e-4 * sigma, 1e-6))
        eps_t_days = eps.get("eps_t_days", 1.0)  # 1 day
        eps_r = eps.get("eps_r", 1e-4)

        # Central difference helper
        def fd(func, x, h):
            return (func(x + h) - func(x - h)) / (2.0 * h)

        def fd_second(func, x, h):
            return (func(x + h) - 2.0 * func(x) + func(x - h)) / (h * h)

        # Delta: dC/dS
        # use central FD in S0
        def price_with_S(S_val):
            copy = MertonJumpDiffusion(
                S_val, self.K, int(self.T * 365), self.r, self.sigma, self.lam, self.mu_j, self.sigma_j, self.tol, self.max_k_user
            )
            return copy.price(option_type=option_type)

        Delta = fd(price_with_S, s0, eps_s)

        # Gamma: d^2 C / dS^2
        Gamma = fd_second(price_with_S, s0, eps_s)

        # Vega: dC/dsigma
        def price_with_sigma(sig):
            copy = MertonJumpDiffusion(
                self.S0, self.K, int(self.T * 365), self.r, sig, self.lam, self.mu_j, self.sigma_j, self.tol, self.max_k_user
            )
            return copy.price(option_type=option_type)

        Vega = fd(price_with_sigma, sigma, eps_sigma)

        # Theta: approximate by reducing time by eps_t_days (in days)
        # Theta returned as per-year derivative (so scale by 365)
        eps_T_days = eps_t_days
        T_days = max(1.0, self.T * 365.0)
        # We create a small-copy with days_to_maturity +/- eps_T_days
        def price_with_T_days(T_days_val):
            # safe guard: must be at least 1 day
            T_days_val_int = max(1, int(round(T_days_val)))
            copy = MertonJumpDiffusion(
                self.S0, self.K, T_days_val_int, self.r, self.sigma, self.lam, self.mu_j, self.sigma_j, self.tol, self.max_k_user
            )
            return copy.price(option_type=option_type)

        # use central approx in days, then convert to per-year
        if method == "central":
            C_plus = price_with_T_days(T_days - eps_T_days)
            C_minus = price_with_T_days(T_days + eps_T_days)
            Theta_days = (C_plus - C_minus) / (2.0 * eps_T_days)
        else:
            # forward difference: (C(T) - C(T + h)) / h
            C_now = self.price(option_type=option_type)
            C_forward = price_with_T_days(T_days + eps_T_days)
            Theta_days = (C_forward - C_now) / eps_T_days

        # convert per-day to per-year (multiply by 365) and negative sign for time decay
        Theta = Theta_days * 365.0

        # Rho: dC/dr
        def price_with_r(r_val):
            copy = MertonJumpDiffusion(
                self.S0, self.K, int(self.T * 365), r_val, self.sigma, self.lam, self.mu_j, self.sigma_j, self.tol, self.max_k_user
            )
            return copy.price(option_type=option_type)

        Rho = fd(price_with_r, r, eps_r)

        return {
            "Delta": float(Delta),
            "Gamma": float(Gamma),
            "Vega": float(Vega),
            "Theta": float(Theta),
            "Rho": float(Rho),
        }


 
# Example usage / quick test
 
if __name__ == "__main__":
    # Example parameters (reasonable market-like numbers)
    S0 = 100.0
    K = 100.0
    days_to_maturity = 365  # 1 year
    r = 0.05
    sigma = 0.2
    lam = 0.75  # 0.75 jumps per year
    mu_j = -0.1  # mean jump size in log-space (negative -> downward mean jump)
    sigma_j = 0.25  # jump volatility

    model = MertonJumpDiffusion(S0, K, days_to_maturity, r, sigma, lam, mu_j, sigma_j)

    price_call = model.price("call")
    price_put = model.price("put")
    greeks_call = model.greeks("call")
    greeks_put = model.greeks("put")

    print("Merton Jump-Diffusion (example):")
    print(f"Call Price: {price_call:.6f}")
    print(f"Put Price : {price_put:.6f}")
    print("\nCall Greeks:")
    for k, v in greeks_call.items():
        print(f"  {k}: {v:.6f}")
    print("\nPut Greeks:")
    for k, v in greeks_put.items():
        print(f"  {k}: {v:.6f}")

