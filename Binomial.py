import numpy as np  
from scipy.special import comb

class BinomialTreeModel:
    
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, num_of_steps):
        
        """
        Initializes variables used in Black-Scholes formula.

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option cotract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        num_of_steps: number of steps in the binomial tree
        
        """
        
        
        self.S_0 = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        self.N = num_of_steps 
        
        # Binomial tree parameters
        
        
        self.dt= self.T/self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = np.exp(-self.sigma * np.sqrt(self.dt))
        self.p = (np.exp(self.r*self.dt) - self.d)  /  (self.u - self.d)
        
    def calculate_option_price(self, option_type="call"):
            """
            Calculates the option price using the Binomial Tree model.

            option_type: "call" for a call option or "put" for a put option
            """
            V = 0
            for k in range(self.N + 1):
                p_k = comb(self.N, k) * self.p**k * (1 - self.p)**(self.N - k)
                S_k = self.S_0 * (self.u**k) * (self.d**(self.N - k))
                if option_type == "call":
                    payoff = max(S_k - self.K, 0)  # Call option payoff
                elif option_type == "put":
                    payoff = max(self.K - S_k, 0)  # Put option payoff
                else:
                    raise ValueError("Invalid option type. Use 'call' or 'put'.")
                V += payoff * p_k
            return V * np.exp(-self.r * self.T)

    def call_option_price(self):
        return self.calculate_option_price(option_type="call")
    
    def put_option_price(self):
        return self.calculate_option_price(option_type="put")  
        
            