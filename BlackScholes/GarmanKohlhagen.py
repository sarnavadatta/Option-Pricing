import numpy as np  
from scipy.stats import norm

# Garman-Kohlhagen model
# An extension of the Black-Scholes formula for pricing European options on foreign exchange (FX)

class GarmanKohlhagen:
    def __init__(self, S, K, T_days, r_d, r_f, sigma):
        self.S = S
        self.K = K
        self.T = T_days / 365
        self.r_d = r_d
        self.r_f = r_f
        self.sigma = sigma
        
        self.d1 = (np.log(self.S/self.K) + (self.r_d - self.r_f + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*np.sqrt(self.T)
    
    def call_option_price(self):
        return self.S*np.exp(-self.r_f*self.T)*norm.cdf(self.d1) - self.K*np.exp(-self.r_d*self.T)*norm.cdf(self.d2)
    
    def put_option_price(self):
        return self.K*np.exp(-self.r_d*self.T)*norm.cdf(-self.d2) - self.S*np.exp(-self.r_f*self.T)*norm.cdf(-self.d1)
    
    # Greeks
    def delta(self, option_type='call'):
        if option_type == 'call':
            return np.exp(-self.r_f*self.T) * norm.cdf(self.d1)
        else:
            return -np.exp(-self.r_f*self.T) * norm.cdf(-self.d1)
    
    def gamma(self):
        return (np.exp(-self.r_f*self.T) * norm.pdf(self.d1)) / (self.S*self.sigma*np.sqrt(self.T))
    
    def vega(self):
        return self.S*np.exp(-self.r_f*self.T) * norm.pdf(self.d1) * np.sqrt(self.T)
    
    def theta(self, option_type='call'):
        term1 = -(self.S*np.exp(-self.r_f*self.T)*norm.pdf(self.d1)*self.sigma)/(2*np.sqrt(self.T))
        if option_type == 'call':
            return term1 - self.r_f*self.S*np.exp(-self.r_f*self.T)*norm.cdf(self.d1) + self.r_d*self.K*np.exp(-self.r_d*self.T)*norm.cdf(self.d2)
        else:
            return term1 + self.r_f*self.S*np.exp(-self.r_f*self.T)*norm.cdf(-self.d1) - self.r_d*self.K*np.exp(-self.r_d*self.T)*norm.cdf(-self.d2)
    
    def rho_domestic(self, option_type='call'):
        if option_type == 'call':
            return self.T*self.K*np.exp(-self.r_d*self.T)*norm.cdf(self.d2)
        else:
            return -self.T*self.K*np.exp(-self.r_d*self.T)*norm.cdf(-self.d2)
    
    def rho_foreign(self, option_type='call'):
        if option_type == 'call':
            return -self.T*self.S*np.exp(-self.r_f*self.T)*norm.cdf(self.d1)
        else:
            return self.T*self.S*np.exp(-self.r_f*self.T)*norm.cdf(-self.d1)


if __name__ == "__main__":   
    # Garman-Kohlhagen example
    gk = GarmanKohlhagen(S=1.10, K=1.12, T_days=90, r_d=0.03, r_f=0.01, sigma=0.20)
    print("\nGarman-Kohlhagen Model:")
    print("Call Price:", gk.call_option_price())
    print("Put Price:", gk.put_option_price())
    print("Delta (Call):", gk.delta('call'))
    print("Gamma:", gk.gamma())
    print("Vega:", gk.vega())
    print("Theta (Call):", gk.theta('call'))
    print("Rho Domestic (Call):", gk.rho_domestic('call'))
    print("Rho Foreign (Call):", gk.rho_foreign('call'))