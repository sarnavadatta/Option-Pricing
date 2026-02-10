import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.fft import fft

# -------------------------
# Analytical Black-Scholes
# -------------------------
class BlackScholes_Analytical:
    def __init__(self, S, K, days, r, sigma):
        self.S = S
        self.K = K
        self.T = days / 365
        self.r = r
        self.sigma = sigma
        self.d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
        self.d2 = self.d1 - sigma * np.sqrt(self.T)

    def call_option_price(self):
        return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)

    def put_option_price(self):
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

    def delta(self, option_type='call'):
        return norm.cdf(self.d1) if option_type == 'call' else norm.cdf(self.d1) - 1

    def gamma(self):
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T)

    def theta(self, option_type='call'):
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == 'call':
            return term1 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return term1 + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)

    def rho(self, option_type='call'):
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)


# -------------------------
# Fourier (Carr-Madan)
# -------------------------
class BlackScholes_Fourier:
    def __init__(self, S, K, days, r, sigma, alpha=1, N=4096, B=200):
        self.S, self.K, self.T, self.r, self.sigma = S, K, days/365, r, sigma
        self.alpha, self.N, self.B = alpha, N, B
        self.eta = B / N
        self.lambd = 2 * np.pi / (N * self.eta)
        self.k = -N * self.lambd / 2 + self.lambd * np.arange(N)

    def characteristic_function(self, u, S=None, r=None, sigma=None, T=None):
        S, r, sigma, T = S or self.S, r or self.r, sigma or self.sigma, T or self.T
        i = 1j
        mu = np.log(S) + (r - 0.5 * sigma**2) * T
        var = sigma**2 * T
        return np.exp(i * u * mu - 0.5 * var * u**2)

    def call_option_price(self, S=None, r=None, sigma=None, T=None):
        S, r, sigma, T = S or self.S, r or self.r, sigma or self.sigma, T or self.T
        i = 1j
        j = np.arange(self.N)
        vj = j * self.eta
        psi = np.exp(-r*T) * self.characteristic_function(vj - (self.alpha+1)*i, S, r, sigma, T) / \
              (self.alpha**2 + self.alpha - vj**2 + i*(2*self.alpha+1)*vj)
        wj = 3 + (-1)**j; wj[0]=wj[-1]=1
        fft_input = np.exp(i*vj*self.B/self.N) * psi * self.eta * wj / 3.0
        fft_output = fft(fft_input).real
        k_index = int(np.floor((np.log(self.K) - self.k[0]) / self.lambd))
        return np.exp(-self.alpha*self.k[k_index]) / np.pi * fft_output[k_index]

    def put_option_price(self):
        return self.call_option_price() - self.S + self.K * np.exp(-self.r*self.T)

    # Greeks via finite diff
    def delta(self, option_type='call', h=1e-4):
        up, down = self.call_option_price(S=self.S+h), self.call_option_price(S=self.S-h)
        d = (up - down) / (2*h)
        return d if option_type=='call' else d - 1
    def gamma(self, h=1e-3):
        up = self.call_option_price(S=self.S+h); mid = self.call_option_price(S=self.S)
        down = self.call_option_price(S=self.S-h)
        return (up - 2*mid + down)/(h**2)
    def vega(self, h=1e-4):
        up, down = self.call_option_price(sigma=self.sigma+h), self.call_option_price(sigma=self.sigma-h)
        return (up - down)/(2*h)
    def theta(self, option_type='call', h=1e-4):
        up, down = self.call_option_price(T=self.T+h), self.call_option_price(T=self.T-h)
        return -(up - down)/(2*h)  # same for put
    def rho(self, option_type='call', h=1e-4):
        up, down = self.call_option_price(r=self.r+h), self.call_option_price(r=self.r-h)
        d = (up - down)/(2*h)
        return d if option_type=='call' else d - self.K*self.T*np.exp(-self.r*self.T)


# -------------------------
# Comparison
# -------------------------
if __name__ == "__main__":
    S, K, days, r, sigma = 100, 105, 30, 0.01, 0.2
    ana, FFT = BlackScholes_Analytical(S,K,days,r,sigma), BlackScholes_Fourier(S,K,days,r,sigma)

    data = {
        "Analytical Call": [
            ana.call_option_price(), ana.delta("call"), ana.gamma(), ana.vega(), ana.theta("call"), ana.rho("call")
        ],
        "Fourier Call": [
            FFT.call_option_price(), FFT.delta("call"), FFT.gamma(), FFT.vega(), FFT.theta("call"), FFT.rho("call")
        ],
        "Analytical Put": [
            ana.put_option_price(), ana.delta("put"), ana.gamma(), ana.vega(), ana.theta("put"), ana.rho("put")
        ],
        "Fourier Put": [
            FFT.put_option_price(), FFT.delta("put"), FFT.gamma(), FFT.vega(), FFT.theta("put"), FFT.rho("put")
        ],
    }

    df = pd.DataFrame(data, index=["Price","Delta","Gamma","Vega","Theta","Rho"])
    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
    print(df)
