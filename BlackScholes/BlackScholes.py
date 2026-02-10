import numpy as np  
from scipy.stats import norm
import pandas as pd

class BlackScholesAnalytical:
    
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma):
        
        """
        Initializes variables used in Black-Scholes formula.

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option cotract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        
        """
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        
        
        # Calculate d1 and d2 for reuse in option price methods
        self.d1 = (np.log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
        
    
    def call_option_price(self):
        return self.S * norm.cdf(self.d1, 0, 1) - self.K * np.exp(-self.r * self.T)*norm.cdf(self.d2, 0, 1)
        
    def put_option_price(self):
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2, 0.0, 1.0) - self.S * norm.cdf(-self.d1, 0.0, 1.0)
    
    # Greeks calculations
    def delta(self, option_type='call'):
        '''
        Measures the sensitivity of the option price to a small change in the spot price S. 
        Delta for a call is N(d1), and for a put, it is N(d1) - 1.
        '''
        if option_type == 'call':
            return norm.cdf(self.d1)
        elif option_type == 'put':
            return norm.cdf(self.d1) - 1
    
    def gamma(self):
        '''
        Measures the rate of change of Delta with respect to the underlying price. It’s the same for both call and put options.
        '''
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        """
        Measures sensitivity to volatility changes. It’s also the same for both call and put options.
        """
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T)
    
    def theta(self, option_type='call'):
        '''
        Measures the time decay of the option’s value. Call and put options have different expressions for Theta.
        '''
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            return term1 - term2
        elif option_type == 'put':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return term1 + term2
    
    def rho(self, option_type='call'):
        '''
        Measures the sensitivity to changes in the risk-free interest rate. Rho is different for call and put options.
        '''
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        elif option_type == 'put':
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
    


# Black-Scholes with dividend yield

class BlackScholesAnalyticalDividend:
    
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, 
                 risk_free_rate, sigma, dividend_yield):
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        self.q = dividend_yield
        
        self.d1 = (np.log(self.S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*np.sqrt(self.T)
    
    def call_option_price(self):
        return (self.S*np.exp(-self.q*self.T)*norm.cdf(self.d1) 
                - self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2))
    
    def put_option_price(self):
        return (self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2) 
                - self.S*np.exp(-self.q*self.T)*norm.cdf(-self.d1))
    
    def delta(self, option_type='call'):
        if option_type == 'call':
            return np.exp(-self.q*self.T)*norm.cdf(self.d1)
        else:
            return np.exp(-self.q*self.T)*(norm.cdf(self.d1) - 1)
    
    def gamma(self):
        return (np.exp(-self.q*self.T)*norm.pdf(self.d1)) / (self.S*self.sigma*np.sqrt(self.T))
    
    def vega(self):
        return self.S*np.exp(-self.q*self.T)*norm.pdf(self.d1)*np.sqrt(self.T)
    
    def theta(self, option_type='call'):
        term1 = -(self.S*np.exp(-self.q*self.T)*norm.pdf(self.d1)*self.sigma) / (2*np.sqrt(self.T))
        if option_type == 'call':
            term2 = self.q*self.S*np.exp(-self.q*self.T)*norm.cdf(self.d1)
            term3 = self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2)
            return term1 - term2 - term3
        else:
            term2 = self.q*self.S*np.exp(-self.q*self.T)*norm.cdf(-self.d1)
            term3 = self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2)
            return term1 + term2 + term3
    
    def rho(self, option_type='call'):
        if option_type == 'call':
            return self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(self.d2)
        else:
            return -self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(-self.d2)



# Comparison Example

if __name__ == "__main__":
    S = 100
    K = 105
    T_days = 30
    r = 0.01
    sigma = 0.2
    q = 0.03   # dividend yield 3%

    bs = BlackScholesAnalytical(S, K, T_days, r, sigma)
    bs_div = BlackScholesAnalyticalDividend(S, K, T_days, r, sigma, q)

    df = pd.DataFrame({
        "Method": ["No Dividend", "With Dividend"],
        "Call Price": [bs.call_option_price(), bs_div.call_option_price()],
        "Put Price": [bs.put_option_price(), bs_div.put_option_price()],
        "Call Delta": [bs.delta("call"), bs_div.delta("call")],
        "Put Delta": [bs.delta("put"), bs_div.delta("put")],
        "Gamma": [bs.gamma(), bs_div.gamma()],
        "Vega": [bs.vega(), bs_div.vega()],
        "Call Theta": [bs.theta("call"), bs_div.theta("call")],
        "Put Theta": [bs.theta("put"), bs_div.theta("put")],
        "Call Rho": [bs.rho("call"), bs_div.rho("call")],
        "Put Rho": [bs.rho("put"), bs_div.rho("put")]
    })

    print(df)
    
# Parameters

S = 100       # Spot price
K = 100       # Strike price
T_days = 180  # Maturity (6 months)
r = 0.02      # Risk-free rate (2%)
sigma = 0.25  # Volatility (25%)

# Dividend yields and strike prices to explore
q_values = np.linspace(0, 0.1, 50)       # dividend yield from 0% to 10%
K_values = np.linspace(80, 120, 50)      # strikes from 80 to 120

# Create grid
Q, K = np.meshgrid(q_values, K_values)

call_prices = []
put_prices = []

# Price storage
call_prices = np.zeros_like(Q)
put_prices = np.zeros_like(Q)

# Fill in prices
for i in range(Q.shape[0]):
    for j in range(Q.shape[1]):
        model = BlackScholesAnalyticalDividend(S, K[i,j], T_days, r, sigma, Q[i,j])
        call_prices[i,j] = model.call_option_price()
        put_prices[i,j] = model.put_option_price()
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D Surface plots
fig = plt.figure(figsize=(14,6))

# Call surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(Q*100, K, call_prices, cmap="viridis", edgecolor="none", alpha=0.9)
ax1.set_title("Call Option Prices vs Dividend Yield & Strike")
ax1.set_xlabel("Dividend Yield (%)")
ax1.set_ylabel("Strike Price (K)")
ax1.set_zlabel("Call Price")

# Put surface
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(Q*100, K, put_prices, cmap="plasma", edgecolor="none", alpha=0.9)
ax2.set_title("Put Option Prices vs Dividend Yield & Strike")
ax2.set_xlabel("Dividend Yield (%)")
ax2.set_ylabel("Strike Price (K)")
ax2.set_zlabel("Put Price")
plt.tight_layout()
plt.show()

'''
Call Prices:
Decrease as dividend yield increases (less benefit from holding stock).
Decrease when strike price goes up (as usual).

Put Prices:
Increase as dividend yield increases (stock less attractive → higher put value).
Increase when strike price goes up.
'''