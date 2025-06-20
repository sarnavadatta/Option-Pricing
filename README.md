This project demonstrates the implementation of three widely used option pricing models:

- **Black-Scholes Analytical Model**
- **Monte Carlo Simulation**
- **Binomial Tree Model**

Each model is used to compute the prices of European-style call and put options, along with various option Greeks.
### ✅ Black-Scholes Analytical Model
Implemented in `BlackScholes.py`, this model calculates:

- European call and put option prices using the Black-Scholes formula
- Option Greeks:  
  - Delta (for call and put)  
  - Gamma  
  - Vega  
  - Theta (for call and put)  
  - Rho (for call and put)

### ✅ Monte Carlo Simulation
Implemented in `MonteCarlo.py`, this model:

- Simulates stock price paths using Geometric Brownian Motion
- Estimates option prices via Monte Carlo averaging
- Visualizes the simulated price paths

### ✅ Binomial Tree Model
Implemented in `Binomial.py`, this model:

- Builds a recombining binomial tree for the underlying price
- Uses backward induction to compute European option prices




