import numpy as np
import pandas as pd
from scipy.stats import binom

class BaseBinomialModel:
    def __init__(self, S0, K, days_to_maturity, r, sigma, N, q=0.0):
        """
        Base class for CRR Binomial Tree Models (European/American, with dividends).
        """
        self.S0 = S0
        self.K = K
        self.T = days_to_maturity / 365
        self.r = r
        self.sigma = sigma
        self.N = N
        self.q = q

        # Tree parameters
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)


class EuropeanBinomialWithDividends(BaseBinomialModel):
    def option_price(self, option_type="call"):
        """European option price (closed form, binomial distribution)."""
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

    def call_price(self):
        return self.option_price("call")

    def put_price(self):
        return self.option_price("put")


class AmericanBinomialWithDividends(BaseBinomialModel):
    def option_price(self, option_type="call"):
        """American option price (backward induction)."""
        # Step 1: stock prices at maturity
        ST = np.array([self.S0 * (self.u**j) * (self.d**(self.N-j)) for j in range(self.N+1)])

        # Step 2: option payoffs at maturity
        if option_type == "call":
            V = np.maximum(ST - self.K, 0)
        elif option_type == "put":
            V = np.maximum(self.K - ST, 0)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        # Step 3: backward induction with early exercise check
        disc = np.exp(-self.r * self.dt)
        for i in range(self.N-1, -1, -1):
            ST = ST[:i+1] / self.u
            V = disc * (self.p * V[1:i+2] + (1 - self.p) * V[0:i+1])
            if option_type == "call":
                V = np.maximum(V, ST - self.K)
            else:
                V = np.maximum(V, self.K - ST)
        return V[0]

    def call_price(self):
        return self.option_price("call")

    def put_price(self):
        return self.option_price("put")


# Comparison function
def comparison_table_with_dividends(S0, K, days_to_maturity, r, sigma, N, q):
    euro = EuropeanBinomialWithDividends(S0, K, days_to_maturity, r, sigma, N, q)
    amer = AmericanBinomialWithDividends(S0, K, days_to_maturity, r, sigma, N, q)

    data = {
        "European": {"Call": euro.call_price(), "Put": euro.put_price()},
        "American": {"Call": amer.call_price(), "Put": amer.put_price()},
    }
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    S0 = 100
    K = 100
    days_to_maturity = 90
    r = 0.05
    sigma = 0.2
    N = 200
    q = 0.03  # Dividend yield

    df = comparison_table_with_dividends(S0, K, days_to_maturity, r, sigma, N, q)
    print(df)