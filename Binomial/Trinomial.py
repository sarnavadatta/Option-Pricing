import numpy as np
import pandas as pd

# ======================================================
# Base Class: Trinomial Tree Model (No Dividends)
# ======================================================
class TrinomialTreeModel:
    def __init__(self, S0, K, days_to_maturity, r, sigma, N):
        self.S0 = S0
        self.K = K
        self.T = days_to_maturity / 365.0
        self.r = r
        self.sigma = sigma
        self.N = N
        self.q = 0.0

        self.dt = self.T / self.N
        self.nu = self.r - 0.5 * self.sigma ** 2
        self.dx = self.sigma * np.sqrt(3 * self.dt)

        self.u = np.exp(self.dx)
        self.d = np.exp(-self.dx)

        # Kamrad-Ritchken probabilities
        self.p_u = 1 / 6 + (self.nu * np.sqrt(self.dt / (12 * self.sigma ** 2)))
        self.p_m = 2 / 3
        self.p_d = 1 / 6 - (self.nu * np.sqrt(self.dt / (12 * self.sigma ** 2)))

        # Normalize probabilities to avoid numerical issues
        self.p_u = max(min(self.p_u, 1), 0)
        self.p_d = max(min(self.p_d, 1), 0)
        self.p_m = 1 - self.p_u - self.p_d

        self.disc = np.exp(-self.r * self.dt)

    # Option pricing 
    def price_option(self, option_type="call", style="european"):
        # terminal payoffs
        V = np.array([max((self.S0 * (self.u ** j) * (self.d ** (self.N - j))) - self.K, 0)
                      if option_type == "call"
                      else max(self.K - (self.S0 * (self.u ** j) * (self.d ** (self.N - j))), 0)
                      for j in range(2 * self.N + 1)])

        # backward induction
        for i in range(self.N - 1, -1, -1):
            V_new = np.zeros(2 * i + 1)
            for j in range(2 * i + 1):
                Vu = V[j + 2]
                Vm = V[j + 1]
                Vd = V[j]
                cont_value = self.disc * (self.p_u * Vu + self.p_m * Vm + self.p_d * Vd)

                if style.lower() == "american":
                    S = self.S0 * (self.u ** j) * (self.d ** (i * 2 - j))
                    intrinsic = (S - self.K) if option_type == "call" else (self.K - S)
                    V_new[j] = max(cont_value, intrinsic)
                else:
                    V_new[j] = cont_value
            V = V_new

        return V[0]

    def call_option_price(self, style="european"):
        return self.price_option("call", style)

    def put_option_price(self, style="european"):
        return self.price_option("put", style)


    # --------------------------
    # Greeks Calculation
    # --------------------------
    def greeks(self, option_type="call", style="european"):
        dt = self.dt
        S0 = self.S0
        u, d, m = self.u, self.d, 1.0

        # first-step node prices
        S_up, S_mid, S_down = S0 * u, S0 * m, S0 * d

        # option values one step ahead
        V_up = max(S_up - self.K, 0) if option_type == "call" else max(self.K - S_up, 0)
        V_mid = max(S_mid - self.K, 0) if option_type == "call" else max(self.K - S_mid, 0)
        V_down = max(S_down - self.K, 0) if option_type == "call" else max(self.K - S_down, 0)

        # discount to time 0
        V0 = self.price_option(option_type, style)

        # Greeks
        delta = (V_up - V_down) / (S_up - S_down)
        gamma = ((V_up - V_mid) / (S_up - S_mid) - (V_mid - V_down) / (S_mid - S_down)) / ((S_up - S_down) / 2)
        theta = (V_mid - V0) / (2 * dt)

        return {"Delta": delta, "Gamma": gamma, "Theta": theta}

    # Comparison table for European and American options
    def comparison_table(self):
        euro_call = self.call_option_price("european")
        euro_put = self.put_option_price("european")
        amer_call = self.call_option_price("american")
        amer_put = self.put_option_price("american")
        return pd.DataFrame({
            "European": {"Call": euro_call, "Put": euro_put},
            "American": {"Call": amer_call, "Put": amer_put}
        })


# ======================================================
# Extended Class: Trinomial Tree Model WITH Dividends
# ======================================================
class TrinomialTreeModelWithDividend:
    def __init__(self, S0, K, days_to_maturity, r, sigma, N, q):
        self.S0 = S0
        self.K = K
        self.T = days_to_maturity / 365.0
        self.r = r
        self.sigma = sigma
        self.N = N
        self.q = q

        self.dt = self.T / self.N
        self.nu = (self.r - self.q) - 0.5 * self.sigma ** 2
        self.dx = self.sigma * np.sqrt(3 * self.dt)

        self.u = np.exp(self.dx)
        self.d = np.exp(-self.dx)

        # Kamrad-Ritchken probabilities
        self.p_u = 1 / 6 + (self.nu * np.sqrt(self.dt / (12 * self.sigma ** 2)))
        self.p_m = 2 / 3
        self.p_d = 1 / 6 - (self.nu * np.sqrt(self.dt / (12 * self.sigma ** 2)))

        self.p_u = max(min(self.p_u, 1), 0)
        self.p_d = max(min(self.p_d, 1), 0)
        self.p_m = 1 - self.p_u - self.p_d

        self.disc = np.exp(-self.r * self.dt)

    def price_option(self, option_type="call", style="european"):
        V = np.array([max((self.S0 * (self.u ** j) * (self.d ** (self.N - j))) - self.K, 0)
                      if option_type == "call"
                      else max(self.K - (self.S0 * (self.u ** j) * (self.d ** (self.N - j))), 0)
                      for j in range(2 * self.N + 1)])

        for i in range(self.N - 1, -1, -1):
            V_new = np.zeros(2 * i + 1)
            for j in range(2 * i + 1):
                Vu = V[j + 2]
                Vm = V[j + 1]
                Vd = V[j]
                cont_value = self.disc * (self.p_u * Vu + self.p_m * Vm + self.p_d * Vd)

                if style.lower() == "american":
                    S = self.S0 * (self.u ** j) * (self.d ** (i * 2 - j))
                    intrinsic = (S - self.K) if option_type == "call" else (self.K - S)
                    V_new[j] = max(cont_value, intrinsic)
                else:
                    V_new[j] = cont_value
            V = V_new

        return V[0]

    def call_option_price(self, style="european"):
        return self.price_option("call", style)

    def put_option_price(self, style="european"):
        return self.price_option("put", style)

    def comparison_table(self):
        euro_call = self.call_option_price("european")
        euro_put = self.put_option_price("european")
        amer_call = self.call_option_price("american")
        amer_put = self.put_option_price("american")
        return pd.DataFrame({
            "European": {"Call": euro_call, "Put": euro_put},
            "American": {"Call": amer_call, "Put": amer_put}
        })


# ======================================================
# Class WITH Dividends
# ======================================================
class TrinomialTreeModelWithDividend(TrinomialTreeModel):
    def __init__(self, S0, K, days_to_maturity, r, sigma, N, q):
        super().__init__(S0, K, days_to_maturity, r, sigma, N)
        self.q = q
        self.nu = (self.r - self.q) - 0.5 * self.sigma ** 2
        self.dx = self.sigma * np.sqrt(3 * self.dt)
        self.u = np.exp(self.dx)
        self.d = np.exp(-self.dx)

        # Recalculate probabilities
        self.p_u = 1 / 6 + (self.nu * np.sqrt(self.dt / (12 * self.sigma ** 2)))
        self.p_m = 2 / 3
        self.p_d = 1 / 6 - (self.nu * np.sqrt(self.dt / (12 * self.sigma ** 2)))

        self.p_u = max(min(self.p_u, 1), 0)
        self.p_d = max(min(self.p_d, 1), 0)
        self.p_m = 1 - self.p_u - self.p_d

        self.disc = np.exp(-self.r * self.dt)


# ======================================================
# Example Usage
# ======================================================
if __name__ == "__main__":
    S0 = 100
    K = 100
    days_to_maturity = 180
    r = 0.05
    sigma = 0.2
    N = 100

    print("=== Model WITHOUT Dividends ===")
    model_no_div = TrinomialTreeModel(S0, K, days_to_maturity, r, sigma, N)
    print(model_no_div.comparison_table())
    print("Greeks (European Call):", model_no_div.greeks("call", "european"))   
    print("Greeks (European Put):", model_no_div.greeks("put", "european"))   


    print("\n=== Model WITH Dividends (q = 6%) ===")
    q = 0.06  # Dividend yield
    model_div = TrinomialTreeModelWithDividend(S0, K, days_to_maturity, r, sigma, N, q)
    print(model_div.comparison_table())
    print("Greeks (European Call):", model_div.greeks("call", "european"))   
    print("Greeks (European Put):", model_div.greeks("put", "european"))   
