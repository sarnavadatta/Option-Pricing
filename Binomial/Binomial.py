import numpy as np
import pandas as pd
from scipy.stats import binom

class BinomialTreeModel:
    def __init__(self, S0, K, days_to_maturity, r, sigma, N, q=0.0):
        """
        Cox-Ross-Rubinstein (CRR) Binomial Tree Model with Greeks.

        S0 : float - current spot price
        K : float - strike price
        days_to_maturity : int - maturity in days
        r : float - risk-free rate
        sigma : float - volatility
        N : int - number of time steps
        q : float - continuous dividend yield (default 0)
        """
        self.S0 = S0
        self.K = K
        self.T = days_to_maturity / 365
        self.r = r
        self.sigma = sigma
        self.N = N
        self.q = q

        # Parameters
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

    def _european_option_price(self, option_type="call"):
        """European option pricing using binomial distribution closed form."""
        V = 0.0
        for k in range(self.N + 1):
            p_k = binom.pmf(k, self.N, self.p)
            S_k = self.S0 * (self.u ** k) * (self.d ** (self.N - k))
            if option_type == "call":
                payoff = max(S_k - self.K, 0)
            elif option_type == "put":
                payoff = max(self.K - S_k, 0)
            else:
                raise ValueError("Invalid option type. Use 'call' or 'put'.")
            V += payoff * p_k
        return V * np.exp(-self.r * self.T)

    def call_option_price(self):
        return self._european_option_price(option_type="call")

    def put_option_price(self):
        return self._european_option_price(option_type="put")

    def american_option_price(self, option_type="call"):
        """Backward induction method for American options."""
        ST = np.array([self.S0 * (self.u**j) * (self.d**(self.N-j)) for j in range(self.N+1)])
        if option_type == "call":
            V = np.maximum(ST - self.K, 0)
        elif option_type == "put":
            V = np.maximum(self.K - ST, 0)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        disc = np.exp(-self.r * self.dt)
        for i in range(self.N-1, -1, -1):
            ST = ST[:i+1] / self.u
            V = disc * (self.p * V[1:i+2] + (1-self.p) * V[0:i+1])
            if option_type == "call":
                V = np.maximum(V, ST - self.K)  # Early exercise check
            else:
                V = np.maximum(V, self.K - ST)
        return V[0]

    def greeks(self, option_type="call"):
        """Compute Delta, Gamma, Theta using the binomial tree (2-step method)."""
        S_u = self.S0 * self.u
        S_d = self.S0 * self.d
        S_uu = self.S0 * (self.u**2)
        S_ud = self.S0 * self.u * self.d
        S_dd = self.S0 * (self.d**2)

        if option_type == "call":
            V_uu = max(S_uu - self.K, 0)
            V_ud = max(S_ud - self.K, 0)
            V_dd = max(S_dd - self.K, 0)
        elif option_type == "put":
            V_uu = max(self.K - S_uu, 0)
            V_ud = max(self.K - S_ud, 0)
            V_dd = max(self.K - S_dd, 0)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        disc = np.exp(-self.r * self.dt)

        V_u = disc * (self.p * V_uu + (1-self.p) * V_ud)
        V_d = disc * (self.p * V_ud + (1-self.p) * V_dd)

        V0 = disc * (self.p * V_u + (1-self.p) * V_d)

        Delta = (V_u - V_d) / (S_u - S_d)
        Gamma = ((V_uu - V_ud) / (S_uu - S_ud) - (V_ud - V_dd) / (S_ud - S_dd)) / ((S_uu - S_dd) / 2)
        Theta = (V_ud - V0) / (2 * self.dt)

        return {"Delta": Delta, "Gamma": Gamma, "Theta": Theta}

    def comparison_table(self):
        """Return a comparison table for European vs American option prices."""
        data = {
            "European": {
                "Call": self.call_option_price(),
                "Put": self.put_option_price(),
            },
            "American": {
                "Call": self.american_option_price("call"),
                "Put": self.american_option_price("put"),
            },
        }
        return pd.DataFrame(data)

if __name__ == "__main__":
    
    # Example parameters
    S0 = 100          # Current stock price
    K = 100           # Strike price
    days_to_maturity = 30  # Maturity in days
    r = 0.01          # Risk-free rate
    sigma = 0.2       # Volatility
    N = 200           # Number of steps
    q = 0.03          # Dividend yield

    model = BinomialTreeModel(S0, K, days_to_maturity, r, sigma, N, q)

    print("Binomial Tree Model:")
    print("European Call Price:", model.call_option_price())
    print("European Put Price:", model.put_option_price())
    print("American Call Price:", model.american_option_price("call"))
    print("American Put Price:", model.american_option_price("put"))
    
    greeks_call = model.greeks("call")
    greeks_put = model.greeks("put")
    
    print("\nGreeks (Call):", greeks_call)
    print("Greeks (Put):", greeks_put)
    
    print("\nComparison Table:")
    print(model.comparison_table())