import numpy as np  
from scipy.stats import norm
import matplotlib.pyplot as plt

class BlackScholesMonteCarlo:
    
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, num_simulation):
        
        """
        Initializes variables used in Black-Scholes formula.

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option cotract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        
        """
        
        # Browninan process parameter
        self.S_0 = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        
        # Simulation parameter
        self.N = num_simulation
        self.steps = days_to_maturity
        self.dt = self.T / self.steps
        self.simulation_results = None
        
        
        '''
        Simulating price movement of underlying prices using Brownian random process.
        Saving random results.
        '''

    def simulate_prices(self):
        np.random.seed(42)
        
        # Initializing price movements for simulation: rows as time index and columns as different random price movements.
        S = np.zeros((self.steps, self.N))
        # Starting value for all price movements is the current spot price
        S[0] = self.S_0
        
            
        for t in range(1, self.steps):
            # Random values to simulate Brownian motion (Gaussian distibution)
            Z = np.random.standard_normal(self.N)
            X = np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + (self.sigma * np.sqrt(self.dt) * Z))
            # Updating prices for next point in time 
            S[t] = S[t - 1] * X
            
        self.simulation_results = S      
                
                
                
    def calculate_call_option_price(self): 
        """
        Call option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Call option payoff (it's exercised only if the price at expiry date is higher than a strike price): max(S_t - K, 0)
        """
        if self.simulation_results is None:
            return -1
        return np.exp(-self.r * self.T) * np.mean(np.maximum(self.simulation_results[-1] - self.K, 0))
    

    def calculate_put_option_price(self): 
        """
        Put option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Put option payoff (it's exercised only if the price at expiry date is lower than a strike price): max(K - S_t, 0)
        """
        if self.simulation_results is None:
            return -1
        return np.exp(-self.r * self.T) * np.mean(np.maximum(self.K - self.simulation_results[-1], 0))
        

    def plot_simulation_results(self, num_of_movements):
        
        """
        Plots specified number of simulated price movements.
        """
        if self.simulation_results is None:
            print("No simulation results to plot")
            return
        
        
        plt.figure(figsize=(12,8))
        plt.plot(self.simulation_results[:,0:num_of_movements])
        plt.axhline(self.K, color='red', linestyle='--', label='Strike Price')
        plt.xlim([0, self.steps])
        plt.ylabel('Simulated Price movements')
        plt.xlabel('Days in future')
        plt.title(f'First {num_of_movements} Simulated Price Movements')
        plt.legend(loc='best')
        plt.show()
        
        
        
# Black Scholes Monte Carlo Method Testing 
# mc = BlackScholesMonteCarlo(underlying_spot_price=100, strike_price=105, days_to_maturity=30, risk_free_rate=0.01, sigma=0.2, num_simulation=1000)
# mc.simulate_prices()

# # Calculate option prices
# call_price = mc.calculate_call_option_price()
# put_price = mc.calculate_put_option_price()        

# print(f"Call Option Price (Monte Carlo): {call_price}")
# print(f"Put Option Price (Monte Carlo): {put_price}")

# # Plot simulation results
# mc.plot_simulation_results(num_of_movements=100)