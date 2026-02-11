# Option Pricing Models

This repository provides a comprehensive implementation of classical and advanced option pricing models, including lattice methods, continuous-time stochastic models, stochastic volatility models, and jump-diffusion frameworks. The project emphasizes mathematical rigor, numerical stability, and reproducibility.

---

# 1. Vanilla Option Pricing Models

## 1.1 Black–Scholes–Merton Model

Under the risk-neutral measure, the asset follows:

$$
dS_t = r S_t dt + \sigma S_t dW_t
$$

The European call price:

$$
C = S_0 N(d_1) - K e^{-rT} N(d_2)
$$

where

$$
d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, 
\quad
d_2 = d_1 - \sigma \sqrt{T}
$$

Implemented variants:
- Standard Black-Scholes
- Fourier Transform approach
- Finite Difference Method
- Monte Carlo Simulation
- Black 76 Model
- Garman-Kohlhagen (FX options)

---

## 1.2 Binomial & Trinomial Models

Discrete-time approximation:

$$
S_{t+\Delta t} =
\begin{cases}
S_t u & \text{with probability } p \\
S_t d & \text{with probability } 1-p
\end{cases}
$$

Risk-neutral probability:

$$
p = \frac{e^{r\Delta t} - d}{u - d}
$$

Supports:
- Dividend and non-dividend cases
- Trinomial tree extension
- American exercise

---

# 2. Stochastic Volatility & Jump Models

## 2.1 Heston Model

Stochastic variance process:

$$
dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^S
$$

$$
dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_t^v
$$

with correlation:

$$
dW_t^S dW_t^v = \rho dt
$$

Captures:
- Volatility smile
- Mean reversion in variance
- Leverage effect

---

## 2.2 Merton Jump Diffusion

Adds Poisson jump component:

$$
dS_t = r S_t dt + \sigma S_t dW_t + S_t (J - 1) dN_t
$$

Where:
- $N_t$ is Poisson process
- $J$ is jump size

Captures:
- Fat tails
- Market crashes
- Discontinuous price moves

---

# 3. Exotic Options

---

# 3.1 Asian Options

Asian options depend on the average asset price over time.

Let monitoring times be $t_1, \dots, t_n$.

---

## Arithmetic Asian Option

Average:

$$
A_{arith} = \frac{1}{n} \sum_{i=1}^{n} S_{t_i}
$$

Payoff (Call):

$$
\max(A_{arith} - K, 0)
$$

No closed-form solution under Black-Scholes → Monte Carlo required.

---

## Geometric Asian Option

Average:

$$
A_{geo} = \left( \prod_{i=1}^{n} S_{t_i} \right)^{1/n}
$$

Closed-form solution exists under lognormal assumption.
Used as control variate due to analytical tractability.

---

## Control Variate Method

Variance reduction technique:

Let:

$$
\hat{X}_{CV} = \hat{X} + \beta (Y - E[Y])
$$

Where:
- $X$ = arithmetic payoff
- $Y$ = geometric payoff (known expectation)
- $\beta$ chosen to minimize variance

Significantly improves Monte Carlo efficiency.

---

# 3.2 Barrier Options

Path-dependent options activated or extinguished when asset crosses a barrier $B$.

---

## Types

- Up-and-In
- Up-and-Out
- Down-and-In
- Down-and-Out

---

### Example: Up-and-Out Call

Payoff:

$$
\max(S_T - K, 0) \cdot \mathbf{1}_{\{ \max_{t \le T} S_t < B \}}
$$

Barrier monitoring introduces path-dependency.

Pricing approaches:
- Reflection principle (closed-form for some cases)
- Monte Carlo simulation
- Finite difference methods

---

# 3.3 Basket Options

Options written on multiple assets.

Let $S_1, \dots, S_m$ be correlated assets.

Weighted basket:

$$
B_T = \sum_{i=1}^{m} w_i S_{i,T}
$$

Call payoff:

$$
\max(B_T - K, 0)
$$

Key considerations:
- Correlation matrix
- Covariance structure
- Cholesky decomposition for simulation

If assets follow correlated GBM:

$$
dS_i = r S_i dt + \sigma_i S_i dW_i
$$

with:

$$
dW_i dW_j = \rho_{ij} dt
$$

---

# Numerical Methods Included

- Monte Carlo Simulation
- Variance Reduction (Control Variate)
- Finite Difference Methods
- Lattice Methods (Binomial / Trinomial)
- Fourier Transform Techniques

---

# Academic Focus

This repository emphasizes:

- Mathematical derivations
- Risk-neutral valuation
- Stochastic calculus foundations
- Numerical stability
- Reproducible computational finance research

---

# Author

Sarnava Datta  
Computational Finance & Quantitative Research
