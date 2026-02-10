import numpy as np
import pandas as pd

class BinomialTreeModel:
    def __init__(self, S0, K, days_to_maturity, r, sigma, N, q=0.0):
        """
        Unified CRR Binomial Tree Model (European & American) with continuous dividend yield q.
        """
        if N < 1:
            raise ValueError("N must be >= 1.")
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(days_to_maturity) / 365.0
        self.r = float(r)
        self.sigma = float(sigma)
        self.N = int(N)
        self.q = float(q)

        # Tree parameters
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)
        self.p = min(max(self.p, 0.0), 1.0)

    def _build_stock_tree(self):
        S = []
        for j in range(self.N + 1):
            row = [self.S0 * (self.u ** i) * (self.d ** (j - i)) for i in range(j + 1)]
            S.append(row)
        return S

    def _build_value_tree(self, option_type="call", style="european"):
        S = self._build_stock_tree()
        if option_type == "call":
            V_last = [max(s - self.K, 0.0) for s in S[self.N]]
        elif option_type == "put":
            V_last = [max(self.K - s, 0.0) for s in S[self.N]]
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")
        V = [None] * (self.N + 1)
        V[self.N] = V_last

        disc = np.exp(-self.r * self.dt)

        for j in range(self.N - 1, -1, -1):
            row = []
            for i in range(j + 1):
                cont = disc * (self.p * V[j + 1][i + 1] + (1.0 - self.p) * V[j + 1][i])
                if style.lower() == "american":
                    intrinsic = (S[j][i] - self.K) if option_type == "call" else (self.K - S[j][i])
                    intrinsic = max(intrinsic, 0.0)
                    val = max(cont, intrinsic)
                elif style.lower() == "european":
                    val = cont
                else:
                    raise ValueError("Invalid style. Use 'european' or 'american'.")
                row.append(val)
            V[j] = row
        return V, S

    def option_price(self, option_type="call", style="european"):
        V, _ = self._build_value_tree(option_type=option_type, style=style)
        return float(V[0][0])

    def call_option_price(self, style="european"):
        return self.option_price("call", style=style)

    def put_option_price(self, style="european"):
        return self.option_price("put", style=style)

    def greeks(self, option_type="call", style="european"):
        if self.N < 2:
            raise ValueError("At least N >= 2 required to compute Greeks via tree.")
        V, S = self._build_value_tree(option_type=option_type, style=style)

        V0 = V[0][0]
        V_d, V_u = V[1][0], V[1][1]
        S_d, S_u = S[1][0], S[1][1]
        Delta = (V_u - V_d) / (S_u - S_d)

        V_dd, V_ud, V_uu = V[2][0], V[2][1], V[2][2]
        S_dd, S_ud, S_uu = S[2][0], S[2][1], S[2][2]
        delta_up = (V_uu - V_ud) / (S_uu - S_ud)
        delta_down = (V_ud - V_dd) / (S_ud - S_dd)
        price_span = (S_uu - S_dd) / 2.0
        Gamma = (delta_up - delta_down) / price_span
        Theta = (V_ud - V0) / (2.0 * self.dt)
        return {"Delta": float(Delta), "Gamma": float(Gamma), "Theta": float(Theta)}

    def comparison_table(self):
        euro_call = self.call_option_price("european")
        euro_put = self.put_option_price("european")
        amer_call = self.call_option_price("american")
        amer_put = self.put_option_price("american")
        data = {
            "European": {"Call": euro_call, "Put": euro_put},
            "American": {"Call": amer_call, "Put": amer_put},
        }
        return pd.DataFrame(data)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    S0 = 100
    K = 95
    days_to_maturity = 90
    r = 0.05
    sigma = 0.2
    N = 200
    q = 0.06  # Dividend yield

    print("\n=== Example 1: Dividend-paying asset (q=6%) ===")
    model_dividend = BinomialTreeModel(S0, K, days_to_maturity, r, sigma, N, q)
    print("European Call:", model_dividend.call_option_price("european"))
    print("American Call:", model_dividend.call_option_price("american"))
    print("European Put :", model_dividend.put_option_price("european"))
    print("American Put :", model_dividend.put_option_price("american"))
    print("\nComparison Table:\n", model_dividend.comparison_table())

    print("\n=== Example 2: Non-dividend asset (q=0%) ===")
    model_no_div = BinomialTreeModel(S0, K, days_to_maturity, r, sigma, N, q=0.0)
    print("European Call:", model_no_div.call_option_price("european"))
    print("American Call:", model_no_div.call_option_price("american"))
    print("European Put :", model_no_div.put_option_price("european"))
    print("American Put :", model_no_div.put_option_price("american"))
    print("\nComparison Table:\n", model_no_div.comparison_table())
