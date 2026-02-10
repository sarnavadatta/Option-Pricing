import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import solve_banded

# -------------------------
# Analytical Black-Scholes
# -------------------------
class BlackScholesAnalytical:
    def __init__(self, S, K, days, r, sigma):
        self.S = S
        self.K = K
        self.T = days / 365.0
        self.r = r
        self.sigma = sigma
        self._compute_d1d2()

    def _compute_d1d2(self):
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def call_option_price(self):
        return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)

    def put_option_price(self):
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

    def delta(self, option_type='call'):
        if option_type == 'call':
            return norm.cdf(self.d1)
        else:
            return norm.cdf(self.d1) - 1

    def gamma(self):
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T)

    def theta(self, option_type='call'):
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            return term1 - term2
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return term1 + term2

    def rho(self, option_type='call'):
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)


# -------------------------
# Finite Difference Crank-Nicolson solver
# -------------------------
class BlackScholesFD:
    """
    Crank-Nicolson FD solver on S-grid for European Call and Put.
    """

    def __init__(self, S, K, days, r, sigma,
                 S_max_factor=4.0, M=400, N_time=400):
        """
        S : underlying spot
        K : strike
        days : days to maturity
        r : rate
        sigma : volatility
        S_max_factor : S_max = S_max_factor * max(S,K)
        M : number of spatial steps (S-grid points = M+1)
        N_time : number of time steps
        """
        self.S0 = S
        self.K = K
        self.T = days / 365.0
        self.r = r
        self.sigma = sigma

        # spatial grid
        self.S_max = S_max_factor * max(S, K)  # large enough S_max
        self.M = M
        self.dS = self.S_max / M
        self.S_grid = np.linspace(0, self.S_max, M + 1)

        # time grid
        self.N = N_time
        self.dt = self.T / self.N
        self.time_grid = np.linspace(0, self.T, self.N + 1)  # forward time

    def _setup_CN_matrices(self, option_type='call'):
        """
        Build banded matrices for Crank-Nicolson:
        We use i = 1..M-1 interior nodes.
        Follow standard coefficients (Hull style):
           a_i = 0.25*dt*(sigma^2 * i^2 - r*i)
           b_i = -0.5*dt*(sigma^2 * i^2 + r)
           c_i = 0.25*dt*(sigma^2 * i^2 + r*i)
        Matrices: M1 * V^{n+1} = M2 * V^n + boundary_terms
        We'll create banded form suitable for solve_banded (with (3,3) shape).
        """
        M = self.M
        dt = self.dt
        sigma = self.sigma
        r = self.r

        # interior indices i = 1..M-1
        i = np.arange(1, M)
        # using S_i = i * dS
        # But the coefficient form using i is standard (S_i dependent through i).
        a = 0.25 * dt * (sigma**2 * i**2 - r * i)
        b = -0.5 * dt * (sigma**2 * i**2 + r)
        c = 0.25 * dt * (sigma**2 * i**2 + r * i)

        # M1: implicit (left) coefficients (size (M-1)x(M-1))
        # diag lower = -a, diag main = 1 - b, diag upper = -c
        lower_M1 = -a[1:]              # length M-2
        main_M1 = 1.0 - b              # length M-1
        upper_M1 = -c[:-1]             # length M-2

        # M2: explicit (right) coefficients
        # diag lower = a, diag main = 1 + b, diag upper = c
        lower_M2 = a[1:]
        main_M2 = 1.0 + b
        upper_M2 = c[:-1]

        # For solve_banded, banded matrix `ab` for (l,u)=(1,1) has shape (3, n)
        # ab[0,1:] = upper diag
        # ab[1,:]  = main diag
        # ab[2,:-1]= lower diag
        def make_ab_banded(lower, main, upper):
            n = len(main)
            ab = np.zeros((3, n))
            ab[1, :] = main
            ab[0, 1:] = upper
            ab[2, :-1] = lower
            return ab

        A_banded = make_ab_banded(lower_M1, main_M1, upper_M1)  # left matrix banded
        B_banded = make_ab_banded(lower_M2, main_M2, upper_M2)  # right matrix banded

        return A_banded, B_banded, a, b, c

    def _terminal_payoff(self, option_type='call'):
        if option_type == 'call':
            return np.maximum(self.S_grid - self.K, 0.0)
        else:
            return np.maximum(self.K - self.S_grid, 0.0)

    def _boundary_at_time(self, t, option_type='call'):
        """
        Returns boundary values V(0,t) and V(S_max,t)
        For call: V(0,t)=0, V(Smax,t)=Smax - K*exp(-r*(T-t))
        For put:  V(0,t)=K*exp(-r*(T-t)), V(Smax,t)=0
        """
        if option_type == 'call':
            V0 = 0.0
            Vmax = self.S_max - self.K * np.exp(-self.r * (self.T - t))
        else:
            V0 = self.K * np.exp(-self.r * (self.T - t))
            Vmax = 0.0
        return V0, Vmax

    def solve(self, option_type='call'):
        """
        Solve backward in time from maturity to t=0 using Crank-Nicolson.
        Returns the grid of option values at t=0 (array length M+1).
        """
        M = self.M
        V = self._terminal_payoff(option_type=option_type).copy()
        # interior nodes vector (i=1..M-1)
        V_int = V[1:-1].copy()

        A_banded, B_banded, a, b, c = self._setup_CN_matrices(option_type=option_type)

        # We will compute for n = N-1 down to 0 (backwards)
        for n in range(self.N - 1, -1, -1):
            t_n = n * self.dt
            # right-hand side = B * V_int + boundary adjustments
            # compute B * V_int as banded product: but easiest is to assemble tri-diagonal multiplication
            # we'll use explicit form:
            # size interior = M-1
            rhs = np.zeros_like(V_int)
            # main terms
            rhs += B_banded[1, :] * V_int
            # upper
            rhs[0:-1] += B_banded[0, 1:] * V_int[1:]
            # lower
            rhs[1:] += B_banded[2, :-1] * V_int[:-1]

            # boundary contributions from V_0 and V_M at times n and n+1 (Crank-Nicolson)
            V0_n, Vmax_n = self._boundary_at_time(t_n + 0.0, option_type)
            V0_np1, Vmax_np1 = self._boundary_at_time(t_n + self.dt, option_type)

            # leftmost interior equation (i=1) has contributions from V_0 terms:
            # For B side the lower term corresponds to index 0 receiving a * V0_n  (because B lower = a)
            rhs[0] += a[0] * V0_n  # a[0] corresponds to interior i=1 lower coefficient on B
            # For A side (implicit) the left boundary appears on RHS as well:
            # Move A * V0_np1 to RHS: since A lower = -a, contribution = -(-a)*V0_np1 = a*V0_np1 ??? simpler: incorporate after solving
            # For consistency we include boundary contribution from implicit side by adding (-A_upper*Vmax_np1 etc)
            # But easier and standard: incorporate combined boundary term:
            rhs[0] += a[0] * V0_np1  # approximate CN boundary treatment
            # rightmost interior equation (i=M-1) receives Vmax terms
            rhs[-1] += c[-1] * Vmax_n
            rhs[-1] += c[-1] * Vmax_np1

            # Solve A * V_next_int = rhs
            # A_banded is banded for solve_banded with (1,1)
            V_int_next = solve_banded((1, 1), A_banded, rhs)
            V_int = V_int_next

        # rebuild full vector with boundaries at t=0
        V_full = np.zeros(M + 1)
        V_full[0], V_full[-1] = self._boundary_at_time(0.0, option_type)
        V_full[1:-1] = V_int
        return V_full

    def interpolate_price(self, V_grid, S_target):
        """
        Linear interpolation on S_grid for price at S_target.
        """
        if S_target <= 0:
            return V_grid[0]
        if S_target >= self.S_max:
            return V_grid[-1]
        i = int(S_target // self.dS)
        if i >= self.M:
            i = self.M - 1
        S_i = self.S_grid[i]
        S_ip1 = self.S_grid[i + 1]
        V_i = V_grid[i]
        V_ip1 = V_grid[i + 1]
        # linear interp
        return V_i + (V_ip1 - V_i) * (S_target - S_i) / (S_ip1 - S_i)

    # Wrapper to compute price and Greeks using FD
    def price_and_greeks(self, option_type='call',
                         h_S_factor=1e-2,  # relative to S0 for delta/gamma bump
                         eps_param=1e-4):
        """
        Returns price, delta, gamma, vega, theta, rho (all annualized units consistent with analytical)
        vega: per unit volatility (same units as analytical)
        theta: per year (analytical uses per year)
        rho: per unit rate
        """
        # baseline price grid
        V0_grid = self.solve(option_type=option_type)
        price = self.interpolate_price(V0_grid, self.S0)

        # Delta, Gamma via S perturbations using interpolation (no re-solve)
        hS = max(self.S0 * h_S_factor, self.dS * 0.5)  # ensure not smaller than grid spacing fraction
        V_plus = self.interpolate_price(V0_grid, self.S0 + hS)
        V_minus = self.interpolate_price(V0_grid, self.S0 - hS)
        delta = (V_plus - V_minus) / (2 * hS)
        gamma = (V_plus - 2 * price + V_minus) / (hS**2)

        # Vega: bump sigma and re-solve
        sigma_up = self.sigma + eps_param
        sigma_down = max(1e-8, self.sigma - eps_param)

        solver_up = BlackScholesFD(self.S0, self.K, self.T * 365.0, self.r, sigma_up,
                                   S_max_factor=self.S_max / max(self.S0, self.K),
                                   M=self.M, N_time=self.N)
        solver_down = BlackScholesFD(self.S0, self.K, self.T * 365.0, self.r, sigma_down,
                                     S_max_factor=self.S_max / max(self.S0, self.K),
                                     M=self.M, N_time=self.N)
        V_up = solver_up.solve(option_type=option_type)
        V_down = solver_down.solve(option_type=option_type)
        price_up = solver_up.interpolate_price(V_up, self.S0)
        price_down = solver_down.interpolate_price(V_down, self.S0)
        vega = (price_up - price_down) / (2 * eps_param)

        # Theta: bump T by eps_param (in years). We'll bump days accordingly and re-solve.
        # eps_param provided in years; convert to days approx for constructor
        eps_days = eps_param * 365.0
        solver_T_up = BlackScholesFD(self.S0, self.K, (self.T + eps_param) * 365.0, self.r, self.sigma,
                                     S_max_factor=self.S_max / max(self.S0, self.K),
                                     M=self.M, N_time=self.N)
        solver_T_down = BlackScholesFD(self.S0, self.K, max((self.T - eps_param) * 365.0, 1e-6 * 365.0), self.r, self.sigma,
                                       S_max_factor=self.S_max / max(self.S0, self.K),
                                       M=self.M, N_time=self.N)
        V_T_up = solver_T_up.solve(option_type=option_type)
        V_T_down = solver_T_down.solve(option_type=option_type)
        price_T_up = solver_T_up.interpolate_price(V_T_up, self.S0)
        price_T_down = solver_T_down.interpolate_price(V_T_down, self.S0)
        # theta = -dV/dT (per year)
        theta = - (price_T_up - price_T_down) / (2 * eps_param)

        # Rho: bump r
        r_up = self.r + eps_param
        r_down = self.r - eps_param
        solver_r_up = BlackScholesFD(self.S0, self.K, self.T * 365.0, r_up, self.sigma,
                                     S_max_factor=self.S_max / max(self.S0, self.K),
                                     M=self.M, N_time=self.N)
        solver_r_down = BlackScholesFD(self.S0, self.K, self.T * 365.0, r_down, self.sigma,
                                       S_max_factor=self.S_max / max(self.S0, self.K),
                                       M=self.M, N_time=self.N)
        V_r_up = solver_r_up.solve(option_type=option_type)
        V_r_down = solver_r_down.solve(option_type=option_type)
        price_r_up = solver_r_up.interpolate_price(V_r_up, self.S0)
        price_r_down = solver_r_down.interpolate_price(V_r_down, self.S0)
        rho = (price_r_up - price_r_down) / (2 * eps_param)

        return {
            "Price": price,
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta,
            "Rho": rho
        }


# -------------------------
# Run comparison & print table
# -------------------------
if __name__ == "__main__":
    # Parameters (example)
    S = 100.0
    K = 105.0
    days = 30.0
    r = 0.01
    sigma = 0.2

    # Analytical
    ana = BlackScholesAnalytical(S, K, days, r, sigma)

    # FD solver (tune M,N for accuracy vs runtime)
    M = 400        # spatial steps (increase for more accuracy)
    N_time = 400   # time steps
    fd_solver = BlackScholesFD(S, K, days, r, sigma, S_max_factor=4.0, M=M, N_time=N_time)

    # Compute FD results for call and put
    fd_call = fd_solver.price_and_greeks(option_type='call', h_S_factor=1e-3, eps_param=1e-4)
    fd_put  = fd_solver.price_and_greeks(option_type='put',  h_S_factor=1e-3, eps_param=1e-4)

    # Analytical results
    ana_call = {
        "Price": ana.call_option_price(),
        "Delta": ana.delta('call'),
        "Gamma": ana.gamma(),
        "Vega": ana.vega(),
        "Theta": ana.theta('call'),
        "Rho": ana.rho('call')
    }
    ana_put = {
        "Price": ana.put_option_price(),
        "Delta": ana.delta('put'),
        "Gamma": ana.gamma(),
        "Vega": ana.vega(),
        "Theta": ana.theta('put'),
        "Rho": ana.rho('put')
    }

    # Build dataframe
    rows = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]
    df = pd.DataFrame({
        "Analytical Call": [ana_call[rn] for rn in rows],
        "FD Call":         [fd_call[rn] for rn in rows],
        "RelErr Call (%)": [100.0 * (fd_call[rn] - ana_call[rn]) / (ana_call[rn] if abs(ana_call[rn])>1e-12 else 1.0) for rn in rows],
        "Analytical Put":  [ana_put[rn] for rn in rows],
        "FD Put":          [fd_put[rn] for rn in rows],
        "RelErr Put (%)":  [100.0 * (fd_put[rn] - ana_put[rn]) / (ana_put[rn] if abs(ana_put[rn])>1e-12 else 1.0) for rn in rows],
    }, index=rows)

    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
    print("\nBlack–Scholes comparison (Analytical vs Crank–Nicolson FD)\n")
    print(df)
