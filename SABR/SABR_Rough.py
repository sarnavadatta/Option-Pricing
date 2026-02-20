import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


class RoughSABR:
    """
    ===========================================================
    ROUGH SABR MODEL
    ===========================================================

    Classical SABR:
        dF_t = σ_t F_t^β dW_t
        dσ_t = ν σ_t dB_t

    Rough SABR replaces volatility with:

        σ_t = σ0 * exp( ν W_t^H - 0.5 ν² t^{2H} )

    where:

        W_t^H = ∫_0^t (t-s)^{H-1/2} dW_s

    H ∈ (0, 0.5)

    H < 0.5 → rough volatility
    H ≈ 0.1 in equity markets
    """

    def __init__(self, F0, sigma0, beta,
                 nu, H, T,
                 N=200, paths=1000):

        self.F0 = F0
        self.sigma0 = sigma0
        self.beta = beta
        self.nu = nu
        self.H = H
        self.T = T
        self.N = N
        self.paths = paths

        self.dt = T / N
        self.time = np.linspace(0, T, N)


    def simulate(self):

        F_paths = np.zeros((self.paths, self.N))
        sigma_paths = np.zeros((self.paths, self.N))

        F_paths[:,0] = self.F0
        sigma_paths[:,0] = self.sigma0

        for p in range(self.paths):

            dW = np.random.normal(0, np.sqrt(self.dt), self.N)

            # Construct fractional Brownian motion
            W_H = np.zeros(self.N)

            for i in range(1, self.N):
                kernel = (self.time[i] - self.time[:i])**(self.H - 0.5)
                W_H[i] = np.sum(kernel * dW[:i])

            for i in range(1, self.N):

                t = self.time[i]

                sigma = self.sigma0 * np.exp(
                    self.nu * W_H[i]
                    - 0.5 * self.nu**2 * t**(2*self.H)
                )

                sigma_paths[p,i] = sigma

                # Log Euler step for F
                drift = -0.5 * sigma**2 \
                        * F_paths[p,i-1]**(2*self.beta - 2) \
                        * self.dt

                diffusion = sigma \
                            * F_paths[p,i-1]**(self.beta - 1) \
                            * dW[i]

                F_paths[p,i] = F_paths[p,i-1] \
                               * np.exp(drift + diffusion)

        return F_paths, sigma_paths


if __name__ == "__main__":

    model = RoughSABR(
        F0=100,
        sigma0=0.2,
        beta=0.5,
        nu=0.4,
        H=0.1,      # Rough volatility
        T=1,
        N=250,
        paths=20
    )

    F_paths, sigma_paths = model.simulate()

    plt.figure(figsize=(12,5))

    # Forward paths
    plt.subplot(1,2,1)
    for i in range(10):
        plt.plot(F_paths[i])
    plt.title("Rough SABR Forward Paths")

    # Volatility paths
    plt.subplot(1,2,2)
    for i in range(10):
        plt.plot(sigma_paths[i])
    plt.title("Rough SABR Volatility Paths (Roughness Visible)")

    plt.tight_layout()
    plt.show()
