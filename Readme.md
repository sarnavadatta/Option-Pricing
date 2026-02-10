# Option Pricing Library

A Python-based option pricing library implementing classical and numerical
methods used in quantitative finance.

This repository is designed for:
- learning and teaching derivatives pricing
- research and experimentation
- clean, modular financial engineering code

---

## 📌 Models Implemented

### 1. Black–Scholes Model
Analytical pricing for European options under the assumptions of:
- lognormal asset dynamics
- constant volatility
- frictionless markets

Supports:
- Call and put options
- Continuous dividend yield
- Analytical Greeks (planned)

---

### 2. Binomial Tree Model
Discrete-time lattice model for option pricing.

Features:
- Cox–Ross–Rubinstein (CRR) tree
- European options
- Convergence to Black–Scholes as time steps increase

Planned extensions:
- American options
- Early exercise premium
- Dividend handling

---

### 3. Monte Carlo Simulation
Simulation-based pricing using risk-neutral dynamics.

Features:
- European call and put options
- Large-scale path simulation
- Discounted payoff estimation



