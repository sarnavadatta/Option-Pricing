from BlackScholes import BlackScholesAnalytical
from MonteCarlo import BlackScholesMonteCarlo
from Binomial import BinomialTreeModel

underlying_spot_price=100 
strike_price=105 
days_to_maturity=30
risk_free_rate=0.01 
sigma=0.2



# Black-Scholes model testing
bs = BlackScholesAnalytical(underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma)

# Option prices
call_price = bs.call_option_price()
put_price = bs.put_option_price()

# Greeks
call_delta = bs.delta(option_type='call')
put_delta = bs.delta(option_type='put')
gamma = bs.gamma()
vega = bs.vega()
call_theta = bs.theta(option_type='call')
put_theta = bs.theta(option_type='put')
call_rho = bs.rho(option_type='call')
put_rho = bs.rho(option_type='put')

# Print results
print("Black-Scholes Model Results:")
print(f"BS Model - Call Option Price : {call_price}")
print(f"BS Model - Put Option Price: {put_price}")
print(f"BS Model - Call Delta: {call_delta}")
print(f"BS Model - Put Delta: {put_delta}")
print(f"BS Model - Gamma: {gamma}")
print(f"BS Model - Vega: {vega}")
print(f"BS Model - Call Theta: {call_theta}")
print(f"BS Model - Put Theta: {put_theta}")
print(f"BS Model - Call Rho: {call_rho}")
print(f"BS Model - Put Rho: {put_rho}")
print("----------------------------------------")




# Monte Carlo Model Testing
num_simulation = 5000  # Number of simulations for Monte Carlo
# Create and simulate the Monte Carlo model
mc = BlackScholesMonteCarlo(underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, num_simulation)
mc.simulate_prices()

# Calculate option prices
call_price = mc.calculate_call_option_price()
put_price = mc.calculate_put_option_price()

# Print results
print("\nMonte Carlo Model Results:")
print(f"Monte Carlo - Call Option Price: {call_price}")
print(f"Monte Carlo - Put Option Price: {put_price}")
print('----------------------------------------')
# Plot the simulation results
mc.plot_simulation_results(num_of_movements=100)

# Binomial Tree Model Testing
num_of_steps = 100
# Binomial Tree Model
bin_model = BinomialTreeModel(underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, num_of_steps)

# Calculate prices
call_price = bin_model.call_option_price()
put_price = bin_model.put_option_price()
print("\nBinomial Model Results:")
print(f"Binomial Model - Call Option Price : {call_price}")
print(f"Binomial Model - Put Option Price: {put_price}")

print('----------------------------------------')

