import numpy as np
from numpy.fft import fft
from scipy.stats import norm
import pandas as pd
from BlackScholes import BlackScholesAnalytical


class BlackScholesFourier:
    
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, alpha=1.01, N=16384, B=200):
        """
        Carr-Madan Fourier transform method for Black-Scholes option pricing.
        """
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        self.alpha = alpha
        self.N = N
        self.B = B

        # Integration grid spacing
        self.eta = B / N  
        self.lambd = 2 * np.pi / (N * self.eta)  # log-strike spacing
        self.k = -N * self.lambd / 2 + self.lambd * np.arange(N)  # log-strike grid
        self.discount_factor = np.exp(-self.r * self.T)

    def characteristic_function(self, u, S=None, r=None, sigma=None, T=None):
        """
        Black-Scholes characteristic function of log-price ln(S_T).
        Allows overriding parameters for sensitivity calculations.
        """
        if S is None: S = self.S
        if r is None: r = self.r
        if sigma is None: sigma = self.sigma
        if T is None: T = self.T
        
        i = 1j
        mu = np.log(S) + (r - 0.5 * sigma**2) * T
        var = sigma**2 * T
        return np.exp(i * u * mu - 0.5 * var * u**2)

    def call_option_price(self, S=None, r=None, sigma=None, T=None):
        """
        Computes European call option price using Carr-Madan Fourier transform.
        """
        if S is None: S = self.S
        if r is None: r = self.r
        if sigma is None: sigma = self.sigma
        if T is None: T = self.T

        i = 1j
        j = np.arange(self.N)
        vj = j * self.eta  # frequency grid

        # Characteristic function with damping
        psi = np.exp(-r * T) * self.characteristic_function(vj - (self.alpha + 1) * i, S, r, sigma, T) \
              / (self.alpha**2 + self.alpha - vj**2 + i * (2 * self.alpha + 1) * vj)

        # Simpson’s rule weights
        wj = 3 + (-1)**j
        wj[0] = 1
        wj[-1] = 1

        # FFT input
        fft_input = np.exp(i * vj * self.B / self.N) * psi * self.eta * wj / 3.0
        fft_output = fft(fft_input).real

        # Find index closest to log-strike
        k_index = int(np.floor((np.log(self.K) - self.k[0]) / self.lambd))

        # Recover option price
        call_price = np.exp(-self.alpha * self.k[k_index]) / np.pi * fft_output[k_index]
        return call_price

    def put_option_price(self):
        """
        Put price via put-call parity.
        """
        call_price = self.call_option_price()
        return call_price - self.S + self.K * np.exp(-self.r * self.T)

    # --------------------
    # Greeks (via finite differences)
    # --------------------
    def delta(self, option_type='call', h=1e-4):
        """
        Sensitivity to underlying price S.
        """
        price_up = self.call_option_price(S=self.S + h)
        price_down = self.call_option_price(S=self.S - h)
        delta = (price_up - price_down) / (2 * h)
        if option_type == 'call':
            return delta
        else:
            # put-call parity: Δ_put = Δ_call - 1
            return delta - 1

    def gamma(self, h=1e-3):
        """
        Second derivative wrt underlying price S.
        """
        price_up = self.call_option_price(S=self.S + h)
        price_mid = self.call_option_price(S=self.S)
        price_down = self.call_option_price(S=self.S - h)
        return (price_up - 2 * price_mid + price_down) / (h**2)

    def vega(self, h=1e-4):
        """
        Sensitivity to volatility σ.
        """
        price_up = self.call_option_price(sigma=self.sigma + h)
        price_down = self.call_option_price(sigma=self.sigma - h)
        return (price_up - price_down) / (2 * h)

    def theta(self, option_type='call', h=1e-4):
        """
        Sensitivity to time (T). Note: Theta is negative of dPrice/dT.
        """
        price_up = self.call_option_price(T=self.T + h)
        price_down = self.call_option_price(T=self.T - h)
        dVdT = (price_up - price_down) / (2 * h)
        theta_val = -dVdT
        if option_type == 'call':
            return theta_val
        else:
            # use parity for puts
            return theta_val

    def rho(self, option_type='call', h=1e-4):
        """
        Sensitivity to risk-free rate r.
        """
        price_up = self.call_option_price(r=self.r + h)
        price_down = self.call_option_price(r=self.r - h)
        rho_val = (price_up - price_down) / (2 * h)
        if option_type == 'call':
            return rho_val
        else:
            return rho_val - self.K * self.T * np.exp(-self.r * self.T)


# -----------------------
# Example usage:
# -----------------------
bs_fft = BlackScholesFourier(
    underlying_spot_price=100,
    strike_price=105,
    days_to_maturity=30,
    risk_free_rate=0.01,
    sigma=0.2
)

bs_analytical = BlackScholesAnalytical(
    underlying_spot_price=100,  
    strike_price=105,
    days_to_maturity=30,
    risk_free_rate=0.01,
    sigma=0.2
)

# print("Black-Scholes Model Results (Analytical):")
# print(f"Call Option Price : {bs_analytical.call_option_price():.4f}")
# print(f"Put Option Price: {bs_analytical.put_option_price():.4f}")
# print(f"Call Delta: {bs_analytical.delta('call'):.4f}")             
# print(f"Put Delta: {bs_analytical.delta('put'):.4f}")
# print(f"Gamma: {bs_analytical.gamma():.4f}")    
# print(f"Vega: {bs_analytical.vega():.4f}")
# print(f"Call Theta: {bs_analytical.theta('call'):.4f}")
# print(f"Put Theta: {bs_analytical.theta('put'):.4f}")
# print(f"Call Rho: {bs_analytical.rho('call'):.4f}")
# print(f"Put Rho: {bs_analytical.rho('put'):.4f}")
# print('----------------------------------------')

# Option prices from FFT
# Note: The Greeks are computed using finite differences, which may introduce numerical errors.
print("Note: Greeks computed via finite differences; may have numerical errors.")
call_price = bs_fft.call_option_price()
put_price = bs_fft.put_option_price()
print(f"Call Price (FFT): {call_price:.4f}")
print(f"Put Price (FFT): {put_price:.4f}")
print('----------------------------------------')
print(f"Delta (Call): {bs_fft.delta('call'):.4f}")
print(f"Delta (Put): {bs_fft.delta('put'):.4f}")
print(f"Gamma: {bs_fft.gamma():.4f}")
print(f"Vega: {bs_fft.vega():.4f}")
print(f"Theta (Call): {bs_fft.theta('call'):.4f}")
print(f"Theta (Put): {bs_fft.theta('put'):.4f}")
print(f"Rho (Call): {bs_fft.rho('call'):.4f}")
print(f"Rho (Put): {bs_fft.rho('put'):.4f}")
print('----------------------------------------')



# Comparison DataFrame
df = pd.DataFrame({
    "Method": ["Analytical", "FFT"],
    "Call Price": [bs_analytical.call_option_price(), call_price],
    "Put Price": [bs_analytical.put_option_price(), put_price], 
    "Call Delta": [bs_analytical.delta('call'), bs_fft.delta('call')],
    "Put Delta": [bs_analytical.delta('put'), bs_fft.delta('put')],
    "Gamma": [bs_analytical.gamma(), bs_fft.gamma()],   
    "Vega": [bs_analytical.vega(), bs_fft.vega()],
    "Call Theta": [bs_analytical.theta('call'), bs_fft.theta('call')],
    "Put Theta": [bs_analytical.theta('put'), bs_fft.theta('put')],
    "Call Rho": [bs_analytical.rho('call'), bs_fft.rho('call')],
    "Put Rho": [bs_analytical.rho('put'), bs_fft.rho('put')]
})  

print(df)
