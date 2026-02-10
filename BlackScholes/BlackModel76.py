import numpy as np  
from scipy.stats import norm

# Black 76 model
# An extension of the Black-Scholes formula specifically designed for pricing options on futures contracts

class Black76:
    def __init__(self, F, K, T_days, r, sigma):
        self.F = F
        self.K = K
        self.T = T_days / 365
        self.r = r
        self.sigma = sigma
        
        self.d1 = (np.log(self.F/self.K) + 0.5*self.sigma**2*self.T) / (self.sigma*np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*np.sqrt(self.T)
    
    def call_option_price(self):
        return np.exp(-self.r*self.T) * (self.F*norm.cdf(self.d1) - self.K*norm.cdf(self.d2))
    
    def put_option_price(self):
        return np.exp(-self.r*self.T) * (self.K*norm.cdf(-self.d2) - self.F*norm.cdf(-self.d1))
    
    # Greeks
    def delta(self, option_type='call'):
        if option_type == 'call':
            return np.exp(-self.r*self.T) * norm.cdf(self.d1)
        else:
            return -np.exp(-self.r*self.T) * norm.cdf(-self.d1)
    
    def gamma(self):
        return (np.exp(-self.r*self.T) * norm.pdf(self.d1)) / (self.F*self.sigma*np.sqrt(self.T))
    
    def vega(self):
        return self.F * np.exp(-self.r*self.T) * norm.pdf(self.d1) * np.sqrt(self.T)
    
    def theta(self, option_type='call'):
        term1 = -(self.F*np.exp(-self.r*self.T) * norm.pdf(self.d1) * self.sigma) / (2*np.sqrt(self.T))
        if option_type == 'call':
            return term1 + self.r*np.exp(-self.r*self.T)*(self.K*norm.cdf(self.d2) - self.F*norm.cdf(self.d1))
        else:
            return term1 + self.r*np.exp(-self.r*self.T)*(-self.K*norm.cdf(-self.d2) + self.F*norm.cdf(-self.d1))
    
    def rho(self, option_type='call'):
        price = self.call_option_price() if option_type=='call' else self.put_option_price()
        return -self.T * price

if __name__ == "__main__":
    # Black-76 example
    black = Black76(F=100, K=105, T_days=90, r=0.02, sigma=0.25)
    print("Black-76 Model:")
    print("Call Price:", black.call_option_price())
    print("Put Price:", black.put_option_price())
    print("Delta (Call):", black.delta('call'))
    print("Delta (Put):", black.delta('put'))
    print("Gamma:", black.gamma())
    print("Vega:", black.vega())
    print("Theta (Call):", black.theta('call'))
    print("Rho (Call):", black.rho('call'))