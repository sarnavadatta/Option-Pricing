import numpy as np  
from scipy.stats import norm

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
    


# # Black-Scholes model testing
# bs = BlackScholesAnalytical(underlying_spot_price=100, strike_price=105, days_to_maturity=30, risk_free_rate=0.01, sigma=0.2)

# # Option prices
# call_price = bs.call_option_price()
# put_price = bs.put_option_price()

# # Greeks
# call_delta = bs.delta(option_type='call')
# put_delta = bs.delta(option_type='put')
# gamma = bs.gamma()
# vega = bs.vega()
# call_theta = bs.theta(option_type='call')
# put_theta = bs.theta(option_type='put')
# call_rho = bs.rho(option_type='call')
# put_rho = bs.rho(option_type='put')

# print(f"Call Price: {call_price}")
# print(f"Put Price: {put_price}")
# print(f"Call Delta: {call_delta}")
# print(f"Put Delta: {put_delta}")
# print(f"Gamma: {gamma}")
# print(f"Vega: {vega}")
# print(f"Call Theta: {call_theta}")
# print(f"Put Theta: {put_theta}")
# print(f"Call Rho: {call_rho}")
# print(f"Put Rho: {put_rho}")