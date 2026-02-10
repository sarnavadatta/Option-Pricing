# Option Pricing Library

A Python-based option pricing library implementing classical and numerical
methods used in quantitative finance.

This repository is designed for:
- learning and teaching derivatives pricing
- research and experimentation
- clean, modular financial engineering code

---

## 📌 Models Implemented

Perfect — below is a **READY-TO-PASTE, GitHub-safe README section**.
You can **copy this entire block as-is** and paste it directly into `README.md` in VS Code.

✔ Academic tone
✔ GitHub-compatible math
✔ Uses only `$$ ... $$` and `\( ... \)`
✔ No unsupported LaTeX environments

---

## 📘 Option Pricing Models Implemented

This repository implements a comprehensive set of **analytical, numerical, and stochastic option pricing models** under the **risk-neutral valuation framework**.
The models are suitable for academic research, quantitative finance applications, and pedagogical use.

---

## 1. Binomial and Trinomial Tree Models

Discrete-time lattice-based option pricing models.

### Implemented variants

* Binomial tree (Cox–Ross–Rubinstein)
* Binomial tree with continuous dividend yield
* Binomial tree without dividends
* Trinomial tree extensions

### Asset price dynamics

At each time step ( \Delta t ), the underlying evolves as:

$$
S_{t+\Delta t} =
\begin{cases}
S_t u & \text{with probability } p \
S_t d & \text{with probability } 1 - p
\end{cases}
$$

with

$$
u = e^{\sigma \sqrt{\Delta t}}, \quad
d = e^{-\sigma \sqrt{\Delta t}}
$$

and risk-neutral probability

$$
p = \frac{e^{(r - q)\Delta t} - d}{u - d}
$$

where:

* ( r ) is the risk-free rate
* ( q ) is the continuous dividend yield

---

## 2. Black–Scholes Family of Models

Continuous-time diffusion-based models assuming lognormal asset price dynamics.

---

### 2.1 Classical Black–Scholes Model

The underlying asset follows a geometric Brownian motion:

$$
dS_t = (r - q) S_t , dt + \sigma S_t , dW_t
$$

The European call option price is given by:

$$
C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$

where

$$
d_1 = \frac{\ln(S_0 / K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}},
\quad
d_2 = d_1 - \sigma \sqrt{T}
$$

and ( N(\cdot) ) denotes the standard normal cumulative distribution function.

---

### 2.2 Black–Scholes via Fourier Methods

Option prices are computed using **characteristic functions** and Fourier inversion techniques.
This approach improves numerical stability and allows extension to more complex stochastic models.

---

### 2.3 Black–Scholes Finite Difference Methods

Numerical solution of the Black–Scholes partial differential equation:

$$
\frac{\partial V}{\partial t}

* \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
* (r - q) S \frac{\partial V}{\partial S}

- rV = 0
  $$

Implemented schemes include:

* Explicit finite difference
* Implicit finite difference

---

### 2.4 Black Model (Black 1976)

Used for pricing options on **forwards and futures**, where the forward price follows:

$$
dF_t = \sigma F_t , dW_t
$$

Pricing is performed under the forward measure.

---

### 2.5 Black–Scholes Monte Carlo

Monte Carlo simulation of terminal asset prices:

$$
S_T = S_0 \exp\left(
(r - q - \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z
\right)
$$

where ( Z \sim \mathcal{N}(0,1) ).

---

### 2.6 Garman–Kohlhagen Model

Extension of Black–Scholes for **foreign exchange options**, incorporating domestic and foreign interest rates:

$$
dS_t = (r_d - r_f) S_t , dt + \sigma S_t , dW_t
$$

---

## 3. Heston Stochastic Volatility Model

A two-factor stochastic volatility model where variance is itself stochastic:

$$
dS_t = r S_t , dt + \sqrt{v_t} S_t , dW_t^{(1)}
$$

$$
dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^{(2)}
$$

with correlation:

$$
\mathbb{E}[dW_t^{(1)} dW_t^{(2)}] = \rho , dt
$$

Option pricing is performed using characteristic functions and Fourier inversion.

---

## 4. Merton Jump Diffusion Model

Extends Black–Scholes by allowing **discontinuous jumps** in asset prices:

$$
dS_t = (r - \lambda k) S_t , dt + \sigma S_t , dW_t + S_t dJ_t
$$

where:

* ( \lambda ) is the jump intensity
* ( J_t ) is a compound Poisson jump process

This model captures skewness, excess kurtosis, and volatility smile effects.

---

## Research and Extensions

The framework supports:

* Cross-model price comparisons
* Convergence analysis
* Sensitivity and parameter studies
* Extensions toward risk metrics such as VaR and CVaR

---
