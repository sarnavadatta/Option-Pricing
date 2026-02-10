'''
Barrier options are a type of path-dependent exotic option, meaning their payoff depends not only on the price of the underlying asset at maturity but also on whether the asset price breaches a predefined barrier level during the option’s life.
They are often cheaper than standard (vanilla) options because the barrier condition reduces the probability of payoff.


| Type          | Condition           | Option Activated/Extinguished When                                  | Description                                                               |
| ------------- | ------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Knock-In**  | Barrier is breached | The option **comes into existence** only if the barrier is touched. | You start with *no* option, and it becomes active if the barrier is hit.  |
| **Knock-Out** | Barrier is breached | The option **ceases to exist** if the barrier is touched.           | You start with an active option, and it disappears if the barrier is hit. |

## Up-and-Out Option:
An Up-and-Out option is a knock-out type barrier option where the barrier is set above the current underlying price.
- If the underlying price rises and hits the barrier, the option immediately expires worthless.
- If the barrier is never reached, the option behaves like a standard European option at maturity.

Intuition
- Often used by investors who expect moderate price increases, but not large spikes.
- Since the option can be “knocked out” if prices rise too high, it's cheaper than a regular call.

## Down-and-In Option:
A Down-and-In option is a knock-in type barrier option where the barrier is set below the current underlying price.
- The option only comes into existence if the price falls below the barrier at any point before maturity.
- If the barrier is never touched, the option never activates, and the payoff is zero.

Intuition:
- Used by investors who expect significant downside movement, but not too early.
- It's cheaper than a vanilla put or call, because activation requires the barrier to be breached.

| Feature          | Up-and-Out                                | Down-and-In                              |
| ---------------- | ----------------------------------------- | ---------------------------------------- |
| Barrier position | Above spot price                          | Below spot price                         |
| Type             | Knock-out                                 | Knock-in                                 |
| Activation logic | Option dies if barrier is hit             | Option activates if barrier is hit       |
| Typical buyer    | Bullish investor expecting limited upside | Bearish investor expecting temporary dip |
| Premium          | Lower than vanilla option                 | Lower than vanilla option                |
| Sensitivity      | Strongly path-dependent                   | Strongly path-dependent                  |


Up-and-Out Calls get cheaper as dividend yield increases — higher dividends reduce expected upward drift.
Down-and-In Puts become more valuable with higher dividends, as they are more likely to be "knocked in".

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Monte Carlo Barrier Option Pricing (with & without dividends)
class BarrierOptionMonteCarlo:
    def __init__(self, S0, K, T_days, r, sigma, N_sim, steps, q=0.0, barrier=120, barrier_type="up-and-out"):
        """
        Monte Carlo simulation for barrier option pricing.

        Parameters:
        ----------
        S0 : float
            Spot price of underlying
        K : float
            Strike price
        T_days : int
            Time to maturity in days
        r : float
            Risk-free rate
        sigma : float
            Volatility of underlying
        N_sim : int
            Number of Monte Carlo simulations
        steps : int
            Number of time steps
        q : float
            Continuous dividend yield (default 0)
        barrier : float
            Barrier level
        barrier_type : str
            'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
        """
        self.S0 = S0
        self.K = K
        self.T = T_days / 365
        self.r = r
        self.sigma = sigma
        self.q = q
        self.N = N_sim
        self.steps = steps
        self.dt = self.T / self.steps
        self.barrier = barrier
        self.barrier_type = barrier_type.lower()

        self.paths = None

    # -----------------------------------------
    def simulate_paths(self, seed=42):
        """Simulate underlying asset price paths under geometric Brownian motion."""
        np.random.seed(seed)
        S = np.zeros((self.steps + 1, self.N))
        S[0] = self.S0

        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)

        for t in range(1, self.steps + 1):
            Z = np.random.standard_normal(self.N)
            S[t] = S[t - 1] * np.exp(drift + diffusion * Z)

        self.paths = S
        return S

    # -----------------------------------------
    def payoff_barrier(self, option_type="call"):
        """Compute barrier option payoff."""
        if self.paths is None:
            self.simulate_paths()

        S_T = self.paths[-1]
        barrier_crossed = np.zeros(self.N, dtype=bool)

        # Determine if barrier was crossed
        if "up" in self.barrier_type:
            barrier_crossed = (self.paths >= self.barrier).any(axis=0)
        elif "down" in self.barrier_type:
            barrier_crossed = (self.paths <= self.barrier).any(axis=0)

        # Plain vanilla payoff
        if option_type.lower() == "call":
            vanilla_payoff = np.maximum(S_T - self.K, 0)
        else:
            vanilla_payoff = np.maximum(self.K - S_T, 0)

        # Apply barrier condition
        if "out" in self.barrier_type:
            payoff = np.where(barrier_crossed, 0, vanilla_payoff)
        else:  # "in"
            payoff = np.where(barrier_crossed, vanilla_payoff, 0)

        return np.exp(-self.r * self.T) * payoff.mean()

    # -----------------------------------------
    def call_option_price(self):
        return self.payoff_barrier(option_type="call")

    def put_option_price(self):
        return self.payoff_barrier(option_type="put")

    # -----------------------------------------
    def summary(self):
        """Return formatted summary as dictionary."""
        return {
            "Barrier Type": self.barrier_type,
            "Dividend Yield (q)": round(self.q, 4),
            "Call Price": round(self.call_option_price(), 4),
            "Put Price": round(self.put_option_price(), 4),
        }


      
# Utility: Sweep over dividend yields and return comparison table
      
def barrier_option_sweep(q_list, option_type="up-and-out"):
    S0, K, T_days, r, sigma, N_sim, steps, barrier = 100, 100, 180, 0.05, 0.2, 20000, 180, 120

    results = []
    for q in q_list:
        model = BarrierOptionMonteCarlo(S0, K, T_days, r, sigma, N_sim, steps, q=q, barrier=barrier, barrier_type=option_type)
        results.append(model.summary())

    df = pd.DataFrame(results)
    return df


      
# Usage and Visualization    
if __name__ == "__main__":
    q_values = np.linspace(0, 0.15, 16)
    df_out = barrier_option_sweep(q_values, option_type="up-and-out")
    df_in = barrier_option_sweep(q_values, option_type="down-and-in")

    print("\n=== Up-and-Out Barrier Option Pricing (Dividend Sweep) ===")
    print(df_out.head())
    print("\n=== Down-and-In Barrier Option Pricing (Dividend Sweep) ===")
    print(df_in.head())

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Barrier Option Prices vs Dividend Yield", fontsize=13, fontweight="bold")

    # Left: Up-and-Out
    axes[0].plot(df_out["Dividend Yield (q)"], df_out["Call Price"], "b-o", label="Call (Up-and-Out)")
    axes[0].plot(df_out["Dividend Yield (q)"], df_out["Put Price"], "r--o", label="Put (Up-and-Out)")
    axes[0].set_title("Up-and-Out Option")
    axes[0].set_xlabel("Dividend Yield (q)")
    axes[0].set_ylabel("Option Price")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Right: Down-and-In
    axes[1].plot(df_in["Dividend Yield (q)"], df_in["Call Price"], "b-o", label="Call (Down-and-In)")
    axes[1].plot(df_in["Dividend Yield (q)"], df_in["Put Price"], "r--o", label="Put (Down-and-In)")
    axes[1].set_title("Down-and-In Option")
    axes[1].set_xlabel("Dividend Yield (q)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
