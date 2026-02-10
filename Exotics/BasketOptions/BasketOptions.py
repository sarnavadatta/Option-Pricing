import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

       
# Monte Carlo Basket Option Pricing (4 assets)
class BasketOptionMonteCarlo:
    def __init__(self, S0_list, K, T_days, r, sigma_list, rho, N_sim, steps, q_list=None, weights=None):
        """
        Monte Carlo simulation for Basket Option Pricing with 4 correlated assets.

        Parameters
        ----------
        S0_list : list
            List of spot prices for assets [S1, S2, S3, S4]
        K : float
            Strike price
        T_days : int
            Time to maturity in days
        r : float
            Risk-free interest rate
        sigma_list : list
            List of volatilities for each asset
        rho : float
            Correlation coefficient between assets
        N_sim : int
            Number of Monte Carlo simulations
        steps : int
            Number of time steps
        q_list : list
            Dividend yields for each asset
        weights : list
            Weights of each asset in basket (should sum to 1)
        """
        self.S0_list = np.array(S0_list)
        self.K = K
        self.T = T_days / 365
        self.r = r
        self.sigma_list = np.array(sigma_list)
        self.rho = rho
        self.N = N_sim
        self.steps = steps
        self.dt = self.T / steps
        self.q_list = np.zeros_like(self.S0_list) if q_list is None else np.array(q_list)
        self.weights = np.ones_like(self.S0_list) / len(self.S0_list) if weights is None else np.array(weights)
        self.paths = None

    def simulate_paths(self, seed=42):
        """Simulate correlated GBM paths for all assets."""
        np.random.seed(seed)
        n_assets = len(self.S0_list)

        # Correlation matrix and Cholesky decomposition
        corr_matrix = np.full((n_assets, n_assets), self.rho)
        np.fill_diagonal(corr_matrix, 1.0)
        L = np.linalg.cholesky(corr_matrix)

        # Initialize price paths
        S = np.zeros((self.steps + 1, n_assets, self.N))
        S[0] = np.repeat(self.S0_list[:, np.newaxis], self.N, axis=1)

        for t in range(1, self.steps + 1):
            Z = np.random.standard_normal((n_assets, self.N))
            Z_corr = L @ Z  # Correlated shocks
            for i in range(n_assets):
                drift = (self.r - self.q_list[i] - 0.5 * self.sigma_list[i] ** 2) * self.dt
                diffusion = self.sigma_list[i] * np.sqrt(self.dt) * Z_corr[i]
                S[t, i] = S[t - 1, i] * np.exp(drift + diffusion)
        self.paths = S
        return S

    def price_option(self, option_type="call"):
        """Compute the price of a call or put basket option."""
        if self.paths is None:
            self.simulate_paths()
        S_T = self.paths[-1]
        basket_T = np.dot(self.weights, S_T)
        if option_type.lower() == "call":
            payoff = np.maximum(basket_T - self.K, 0)
        else:
            payoff = np.maximum(self.K - basket_T, 0)
        return np.exp(-self.r * self.T) * payoff.mean()

    def summary(self, label=""):
        """Return results dictionary for DataFrame collection."""
        return {
            "q1": round(self.q_list[0], 3),
            "q2": round(self.q_list[1], 3),
            "q3": round(self.q_list[2], 3),
            "q4": round(self.q_list[3], 3),
            "Call Price": round(self.price_option("call"), 4),
            "Put Price": round(self.price_option("put"), 4),
            "Scenario": label
        }

       
# Generate all dividend combinations and evaluate
def all_dividend_combinations_analysis():
    # --- Model Parameters ---
    S0_list = [100, 95, 105, 90]
    sigma_list = [0.2, 0.25, 0.22, 0.18]
    weights = [0.25, 0.25, 0.25, 0.25]
    K = 100
    T_days = 180
    r = 0.05
    rho = 0.4
    N_sim = 10000
    steps = 180

    # Dividend grid
    q_values = np.round(np.arange(0.0, 0.09, 0.01), 2)

    # All 4D combinations (q1,q2,q3,q4)
    combos = list(itertools.product(q_values, repeat=4))

    results = []
    total = len(combos)
    print(f"Running {total} dividend combinations...")

    for idx, q_tuple in enumerate(combos):
        q_list = list(q_tuple)
        model = BasketOptionMonteCarlo(S0_list, K, T_days, r, sigma_list, rho, N_sim, steps, q_list=q_list, weights=weights)
        result = model.summary(label=f"q={q_tuple}")
        results.append(result)

        # Progress feedback
        if (idx + 1) % 500 == 0:
            print(f"Completed {idx+1}/{total} combinations")

    df = pd.DataFrame(results)
    return df

       
# Run the analysis and visualize
if __name__ == "__main__":
    df_results = all_dividend_combinations_analysis()

    print("\n=== Basket Option Prices for All Dividend Combinations ===")
    print(df_results.head())

    # Save results
    df_results.to_csv("basket_option_dividend_combinations.csv", index=False)
    print("\nSaved results to 'basket_option_dividend_combinations.csv'")

    # Simple aggregated visualization
    avg_df = df_results.groupby(["q1", "q2", "q3", "q4"])[["Call Price", "Put Price"]].mean().reset_index()

    # Example: Plot mean call & put vs average dividend across assets
    avg_df["avg_q"] = avg_df[["q1", "q2", "q3", "q4"]].mean(axis=1)
    grouped = avg_df.groupby("avg_q")[["Call Price", "Put Price"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Basket Option Prices vs Average Dividend Yield (4 Assets)", fontsize=13, fontweight="bold")

    axes[0].plot(grouped["avg_q"], grouped["Call Price"], "b-o")
    axes[0].set_title("Call Option Prices")
    axes[0].set_xlabel("Average Dividend Yield")
    axes[0].set_ylabel("Price")
    axes[0].grid(True)

    axes[1].plot(grouped["avg_q"], grouped["Put Price"], "r-o")
    axes[1].set_title("Put Option Prices")
    axes[1].set_xlabel("Average Dividend Yield")
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
