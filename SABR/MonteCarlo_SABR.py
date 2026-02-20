import numpy as np
import matplotlib.pyplot as plt


class MonteCarloSABR:
    """
    Production-grade Monte Carlo implementation of the SABR model.

    Model:
        dF_t = sigma_t * F_t^beta * dW1
        d sigma_t = nu * sigma_t * dW2
        corr(dW1, dW2) = rho

    Simulates under forward measure.
    """

    def __init__(self, S0, T, r, q,
                 alpha, beta, rho, nu,
                 n_paths=10000, n_steps=200,
                 seed=None):

        self.S0 = S0
        self.T = T
        self.r = r
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = T / n_steps

        if seed is not None:
            np.random.seed(seed)

        # Forward price
        self.F0 = self.S0 * np.exp((self.r - self.q) * self.T)

        self._terminal_paths = None

     
    # Monte Carlo Simulation
    def simulate_paths(self):

        F = np.full(self.n_paths, self.F0)
        sigma = np.full(self.n_paths, self.alpha)

        for _ in range(self.n_steps):

            Z1 = np.random.normal(size=self.n_paths)
            Z2 = np.random.normal(size=self.n_paths)

            # Correlate shocks
            Z_sigma = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

            # Volatility evolution (exact log solution)
            sigma = sigma * np.exp(
                -0.5 * self.nu**2 * self.dt
                + self.nu * np.sqrt(self.dt) * Z_sigma
            )

            # Full truncation to prevent instability
            F = np.maximum(F, 1e-10)

            # Euler step
            F = F + sigma * (F ** self.beta) * np.sqrt(self.dt) * Z1

        self._terminal_paths = F
        return F

     
    # Pricing
    def price(self, K, option_type="call"):

        if self._terminal_paths is None:
            F_T = self.simulate_paths()
        else:
            F_T = self._terminal_paths

        if option_type == "call":
            payoff = np.maximum(F_T - K, 0)
        else:
            payoff = np.maximum(K - F_T, 0)

        return np.exp(-self.r * self.T) * np.mean(payoff)

     
    # Vectorized Strike Sweep
    def price_sweep_strikes(self, strike_range, option_type="call"):

        if self._terminal_paths is None:
            F_T = self.simulate_paths()
        else:
            F_T = self._terminal_paths

        prices = []

        for K in strike_range:
            if option_type == "call":
                payoff = np.maximum(F_T - K, 0)
            else:
                payoff = np.maximum(K - F_T, 0)

            price = np.exp(-self.r * self.T) * np.mean(payoff)
            prices.append(price)

        return np.array(prices)

     
    # Path Visualization
     
    def plot_simulation(self, num_paths_to_plot=50):

        F = np.full((num_paths_to_plot, self.n_steps + 1), self.F0)
        sigma = np.full(num_paths_to_plot, self.alpha)

        for t in range(1, self.n_steps + 1):

            Z1 = np.random.normal(size=num_paths_to_plot)
            Z2 = np.random.normal(size=num_paths_to_plot)

            Z_sigma = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

            sigma = sigma * np.exp(
                -0.5 * self.nu**2 * self.dt
                + self.nu * np.sqrt(self.dt) * Z_sigma
            )

            F[:, t] = np.maximum(
                F[:, t-1] +
                sigma * (F[:, t-1] ** self.beta) *
                np.sqrt(self.dt) * Z1,
                1e-10
            )

        plt.figure(figsize=(12, 6))
        for i in range(num_paths_to_plot):
            plt.plot(np.linspace(0, self.T, self.n_steps + 1), F[i], lw=0.7)

        plt.title("Monte Carlo Simulation of SABR Paths")
        plt.xlabel("Time")
        plt.ylabel("Forward Price")
        plt.grid(True)
        plt.show()



# Example Usage
if __name__ == "__main__":

    mc = MonteCarloSABR(
        S0=100,
        T=1,
        r=0.01,
        q=0.02,
        alpha=0.2,
        beta=0.5,
        rho=-0.3,
        nu=0.4,
        n_paths=10000,
        n_steps=200,
        seed=42
    )

    call_price = mc.price(K=100, option_type="call")
    put_price = mc.price(K=100, option_type="put")

    print(f"Call Price: {call_price:.4f}")
    print(f"Put Price: {put_price:.4f}")

    strikes = np.linspace(80, 120, 20)
    call_prices = mc.price_sweep_strikes(strikes, "call")

    mc.plot_simulation(40)
