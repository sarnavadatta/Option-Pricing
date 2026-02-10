"""
asian_option_pricer.py

Price arithmetic Asian options (call & put) by Monte Carlo under Black-Scholes,
supporting continuous dividend yield q (pass q=0 for no-dividend).

Variance reduction:
 - Antithetic variates (optional)
 - Control variate using closed-form geometric Asian option (continuous monitoring formula)

Notes:
 - The geometric closed-form used here corresponds to the continuous-time geometric average;
   when you simulate a discrete-time average the control variate is not exact but still
   provides substantial variance reduction. For exact control variate one can use the discrete
   geometric-average analytic formula (more algebra), but continuous formula works well in practice.
"""

import numpy as np
from math import log, exp, sqrt
from scipy.stats import norm
from typing import Tuple, Dict


     
# Geometric Asian closed-form (continuous monitoring)
     
def geometric_asian_price_continuous(S0: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str = "call") -> float:
    """
    Closed-form price of a continuously-monitored geometric-average Asian option under Black-Scholes
    with continuous dividend yield q.

    This uses the fact that G = exp( (1/T) integral_0^T ln S_t dt ) is lognormal with:
       E[ln G] = ln S0 + (r - q - 0.5*sigma^2) * T/2
       Var(ln G) = sigma^2 * T / 3

    Then price is exp(-rT) * E[(G - K)^+] (for call) which can be computed analytically.

    Parameters
    ----------
    S0, K, T, r, q, sigma : floats
    option_type : "call" or "put"

    Returns
    -------
    analytic geometric Asian price (float)
    """
    if T <= 0:
        # immediate maturity -> payoff
        if option_type == "call":
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)

    # parameters of ln G
    mu_lnG = log(S0) + (r - q - 0.5 * sigma * sigma) * (T / 2.0)
    var_lnG = (sigma * sigma) * T / 3.0
    std_lnG = sqrt(var_lnG)

    # compute terms analogous to Black-Scholes formula
    # d1 = (mu + var - ln K) / std
    # d2 = (mu - ln K) / std
    lnK = log(K)
    d1 = (mu_lnG + var_lnG - lnK) / std_lnG
    d2 = (mu_lnG - lnK) / std_lnG

    # E[G * 1_{G>K}] = exp(mu + 0.5 var) * Phi(d1)
    term1 = exp(mu_lnG + 0.5 * var_lnG) * norm.cdf(d1)
    # P(G > K) = Phi(d2)
    term2 = norm.cdf(d2)

    if option_type.lower() == "call":
        price = exp(-r * T) * max(term1 - K * term2, 0.0)
    else:
        # Put: price = e^{-rT} * E[(K - G)^+] = e^{-rT} * (K*P(G<K) - E[G 1_{G<K}])
        # P(G < K) = 1 - Phi(d2) = Phi(-d2)
        term1_put = K * norm.cdf(-d2)
        term2_put = exp(mu_lnG + 0.5 * var_lnG) * norm.cdf(-d1)
        price = exp(-r * T) * max(term1_put - term2_put, 0.0)

    return float(price)


     
# Monte Carlo arithmetic Asian pricer with control variate
class AsianOptionPricer:
    def __init__(
        self,
        S0: float,
        K: float,
        days_to_maturity: int,
        r: float,
        sigma: float,
        q: float = 0.0,
        n_steps: int = None,
        n_paths: int = 200_000,
        antithetic: bool = True,
        control_variate: bool = True,
        seed: int = 12345,
    ):
        """
        Parameters:
        - S0: spot price
        - K: strike
        - days_to_maturity: integer days until expiry
        - r: risk-free rate (annual, continuous)
        - sigma: volatility (annual)
        - q: continuous dividend yield (set 0.0 for no-dividend)
        - n_steps: number of averaging points (if None, use days_to_maturity)
        - n_paths: number of Monte Carlo simulation paths (before antithetic doubling)
        - antithetic: use antithetic variates (halves random draws variance)
        - control_variate: use geometric Asian analytic formula as control variate
        - seed: RNG seed
        """
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(days_to_maturity) / 365.0
        if self.T <= 0:
            raise ValueError("days_to_maturity must be positive")
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.n_steps = int(n_steps) if n_steps is not None else int(days_to_maturity)
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.n_paths = int(n_paths)
        self.antithetic = bool(antithetic)
        self.control_variate = bool(control_variate)
        self.seed = int(seed)

        # time increment per observation
        self.dt = self.T / self.n_steps

    def _simulate_arithmetic_paths(self) -> np.ndarray:
        """
        Simulate asset paths and compute arithmetic averages for each path.

        Returns:
            A numpy array of shape (n_paths_effective,) with arithmetic averages.
            If antithetic=True then n_paths_effective = n_paths or 2*n_paths? Implementation:
            - If antithetic True: generate n_paths / 2 random draws and mirror; but we keep simpler:
              We generate n_paths/2 draws and produce 2 averages per draw (Z and -Z) so final # paths = n_paths (if n_paths even).
        """
        rng = np.random.default_rng(self.seed)

        # If antithetic, we will generate n_paths/2 and reflect; ensure even
        if self.antithetic:
            if self.n_paths % 2 != 0:
                self.n_paths += 1  # make even
        # number of normal draws to generate
        n_draws = self.n_paths // 2 if self.antithetic else self.n_paths

        # simulate increments: shape (n_steps, n_draws)
        Z = rng.standard_normal(size=(self.n_steps, n_draws))

        # precompute drift and diffusion scale
        drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * self.dt
        vol = self.sigma * np.sqrt(self.dt)

        # compute log-price increments and build log-paths
        lnS_paths = np.zeros((self.n_steps + 1, n_draws))
        lnS_paths[0, :] = log(self.S0)

        for t in range(1, self.n_steps + 1):
            lnS_paths[t, :] = lnS_paths[t - 1, :] + drift + vol * Z[t - 1, :]

        # compute arithmetic average over observation times (exclude S0? typical Asian uses observed prices at times 1..n)
        S_paths = np.exp(lnS_paths)  # shape (n_steps+1, n_draws)
        # For averaging, use the observation times S_1 ... S_n (exclude initial)
        A_arith = S_paths[1:, :].mean(axis=0)  # shape (n_draws,)

        if self.antithetic:
            # build antithetic paths by flipping signs of Z
            Z_a = -Z
            lnS_a = np.zeros_like(lnS_paths)
            lnS_a[0, :] = log(self.S0)
            for t in range(1, self.n_steps + 1):
                lnS_a[t, :] = lnS_a[t - 1, :] + drift + vol * Z_a[t - 1, :]
            A_arith_a = np.exp(lnS_a[1:, :]).mean(axis=0)

            # concatenate original and antithetic averages to get final sample of size n_draws*2 = n_paths (or n_paths+1 earlier adjustment)
            A_all = np.concatenate([A_arith, A_arith_a], axis=0)
        else:
            A_all = A_arith

        return A_all  # length = effective number of simulated averages

    def _simulate_geometric_averages(self) -> np.ndarray:
        """
        Simulate geometric averages G = (Π S_i)^{1/n} for the same paths used above.
        We can compute this easily from lnS_paths: ln G = (1/n) Σ ln S_i (i=1..n)
        This convenience function is not used externally but helpful for control variate covariance.
        """
        rng = np.random.default_rng(self.seed)
        if self.antithetic:
            if self.n_paths % 2 != 0:
                self.n_paths += 1
        n_draws = self.n_paths // 2 if self.antithetic else self.n_paths
        Z = rng.standard_normal(size=(self.n_steps, n_draws))

        drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * self.dt
        vol = self.sigma * np.sqrt(self.dt)

        lnS_paths = np.zeros((self.n_steps + 1, n_draws))
        lnS_paths[0, :] = log(self.S0)
        for t in range(1, self.n_steps + 1):
            lnS_paths[t, :] = lnS_paths[t - 1, :] + drift + vol * Z[t - 1, :]

        lnG = lnS_paths[1:, :].mean(axis=0)  # discrete geometric mean uses average of logs
        G = np.exp(lnG)

        if self.antithetic:
            Z_a = -Z
            lnS_a = np.zeros_like(lnS_paths)
            lnS_a[0, :] = log(self.S0)
            for t in range(1, self.n_steps + 1):
                lnS_a[t, :] = lnS_a[t - 1, :] + drift + vol * Z_a[t - 1, :]
            lnG_a = lnS_a[1:, :].mean(axis=0)
            G_a = np.exp(lnG_a)
            G_all = np.concatenate([G, G_a], axis=0)
        else:
            G_all = G

        return G_all

    def price_mc(self, option_type: str = "call") -> Dict[str, float]:
        """
        Price arithmetic Asian option (call or put) using Monte Carlo.

        Returns a dictionary:
          {
            'mc_price': raw Monte Carlo discounted price,
            'mc_std_error': Monte Carlo standard error,
            'cv_price': control-variate adjusted price (if control_variate True),
            'cv_std_error': std error of control-variate estimator (if used),
            'geometric_price': analytic geometric Asian price used as control variate (if used)
          }
        """
        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        # simulate arithmetic averages
        A = self._simulate_arithmetic_paths()  # shape (n_effective,)
        m = A.shape[0]

        # Monte Carlo payoffs (arithmetic)
        if option_type == "call":
            payoffs = np.maximum(A - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - A, 0.0)

        # discount
        discounted = np.exp(-self.r * self.T) * payoffs
        mc_price = np.mean(discounted)
        mc_std_error = np.std(discounted, ddof=1) / sqrt(m)

        result = {
            "mc_price": float(mc_price),
            "mc_std_error": float(mc_std_error),
        }

        # control variate
        if self.control_variate:
            # analytic geometric Asian price (continuous closed-form)
            geo_price = geometric_asian_price_continuous(self.S0, self.K, self.T, self.r, self.q, self.sigma, option_type)
            # simulate geometric averages (discrete) to compute sample covariance with arithmetic payoff
            # BUT easier: we can compute discrete geometric averages alongside arithmetic from same paths;
            # here we re-simulate but use same seed: to ensure same draws we'd incorporate both in one sim.
            # For simplicity and because control variate only needs a strongly correlated variable, we compute G_all similarly:
            G = self._simulate_geometric_averages()  # shape (m,)
            # geometric payoff
            if option_type == "call":
                payoff_geo = np.maximum(G - self.K, 0.0)
            else:
                payoff_geo = np.maximum(self.K - G, 0.0)
            discounted_geo = np.exp(-self.r * self.T) * payoff_geo

            # sample covariance and compute optimal b
            cov = np.cov(discounted, discounted_geo, ddof=1)
            cov_xy = cov[0, 1]
            var_y = cov[1, 1]
            if var_y == 0 or np.isnan(var_y):
                b_opt = 0.0
            else:
                b_opt = cov_xy / var_y

            # control variate estimator: estimator = X_bar - b (Y_bar - E[Y])
            X_bar = np.mean(discounted)
            Y_bar = np.mean(discounted_geo)
            adjusted = X_bar - b_opt * (Y_bar - geo_price)

            # standard error of control-variate estimator: use variance of adjusted sample
            adjusted_sample = discounted - b_opt * discounted_geo
            adj_std_error = np.std(adjusted_sample, ddof=1) / sqrt(m)

            result.update({
                "control_variate": True,
                "geometric_price": float(geo_price),
                "b_opt": float(b_opt),
                "cv_price": float(adjusted),
                "cv_std_error": float(adj_std_error),
            })

        return result


     
# Usage
if __name__ == "__main__":
    # Parameters
    S0 = 100.0
    K = 100.0
    days = 90
    r = 0.05
    sigma = 0.25
    q_values = [0.0, 0.03]  # test both no-dividend and with dividend
    n_paths = 100_000
    n_steps = 90  # daily averaging

    for q in q_values:
        print("\n=== Asian (arithmetic) MC pricing (q = {:.2%}) ===".format(q))
        pricer_call = AsianOptionPricer(
            S0=S0,
            K=K,
            days_to_maturity=days,
            r=r,
            sigma=sigma,
            q=q,
            n_steps=n_steps,
            n_paths=n_paths,
            antithetic=True,
            control_variate=True,
            seed=20251007,
        )
        res_call = pricer_call.price_mc("call")

        pricer_put = AsianOptionPricer(
            S0=S0,
            K=K,
            days_to_maturity=days,
            r=r,
            sigma=sigma,
            q=q,
            n_steps=n_steps,
            n_paths=n_paths,
            antithetic=True,
            control_variate=True,
            seed=20251007,
        )
        res_put = pricer_put.price_mc("put")

        print("CALL:  MC price = {:.6f} ± {:.6f}".format(res_call["mc_price"], res_call["mc_std_error"]))
        if res_call.get("control_variate", False):
            print("       CV price = {:.6f} ± {:.6f} (geo analytic = {:.6f}, b = {:.6f})".format(
                res_call["cv_price"], res_call["cv_std_error"], res_call["geometric_price"], res_call["b_opt"]
            ))

        print("PUT:   MC price = {:.6f} ± {:.6f}".format(res_put["mc_price"], res_put["mc_std_error"]))
        if res_put.get("control_variate", False):
            print("       CV price = {:.6f} ± {:.6f} (geo analytic = {:.6f}, b = {:.6f})".format(
                res_put["cv_price"], res_put["cv_std_error"], res_put["geometric_price"], res_put["b_opt"]
            ))
