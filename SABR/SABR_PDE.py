import numpy as np
import matplotlib.pyplot as plt


class SABRPDE_ADI:
    """
    ===========================================================
    SABR PDE Solver using ADI (Douglas Scheme)
    ===========================================================

    SABR dynamics:

        dF = σ F^β dW1
        dσ = ν σ dW2
        corr(dW1,dW2) = ρ

    Risk-neutral pricing PDE:

        ∂V/∂t
        + 1/2 σ² F^{2β} ∂²V/∂F²
        + ρ ν σ² F^β ∂²V/∂F∂σ
        + 1/2 ν² σ² ∂²V/∂σ²
        = 0

    Terminal condition:

        V(F,σ,T) = max(F-K,0)

    We solve backward in time using ADI splitting.
    """

    def __init__(self, F_max, sigma_max,
                 K, T,
                 beta, nu, rho,
                 NF=100, NS=50, NT=100):

        self.F_max = F_max
        self.sigma_max = sigma_max
        self.K = K
        self.T = T
        self.beta = beta
        self.nu = nu
        self.rho = rho

        self.NF = NF
        self.NS = NS
        self.NT = NT

        self.dF = F_max / NF
        self.dS = sigma_max / NS
        self.dt = T / NT

        self.F = np.linspace(0, F_max, NF)
        self.S = np.linspace(0, sigma_max, NS)

        self.grid = np.zeros((NF, NS))


    # Terminal payoff
    def initialize_payoff(self):

        for i in range(self.NF):
            self.grid[i, :] = np.maximum(self.F[i] - self.K, 0)


    # Finite Difference Coefficients
    def step(self):

        V = self.grid.copy()

        beta = self.beta
        nu = self.nu
        rho = self.rho

        new_V = V.copy()

        # Backward time stepping
        for n in range(self.NT):

            for i in range(1, self.NF-1):
                for j in range(1, self.NS-1):

                    F = self.F[i]
                    sigma = self.S[j]

                    # Second derivatives
                    V_FF = (V[i+1,j] - 2*V[i,j] + V[i-1,j]) / self.dF**2
                    V_SS = (V[i,j+1] - 2*V[i,j] + V[i,j-1]) / self.dS**2
                    V_FS = (V[i+1,j+1] - V[i+1,j-1]
                            - V[i-1,j+1] + V[i-1,j-1]) \
                           / (4*self.dF*self.dS)

                    # SABR PDE operator
                    diffusion_F = 0.5 * sigma**2 * F**(2*beta) * V_FF
                    diffusion_S = 0.5 * nu**2 * sigma**2 * V_SS
                    cross = rho * nu * sigma**2 * F**beta * V_FS

                    new_V[i,j] = V[i,j] - self.dt * (
                        diffusion_F + diffusion_S + cross
                    )

            V = new_V.copy()

        self.grid = V

   
    # Solve PDE
    def solve(self):

        self.initialize_payoff()
        self.step()
        return self.grid


# Example Usage
if __name__ == "__main__":

    solver = SABRPDE_ADI(
        F_max=200,
        sigma_max=1.0,
        K=100,
        T=1,
        beta=0.5,
        nu=0.4,
        rho=-0.3,
        NF=150,
        NS=50,
        NT=250
    )

    solution = solver.solve()

    F_grid, S_grid = np.meshgrid(
        solver.F,
        solver.S,
        indexing="ij"
    )

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        F_grid,
        S_grid,
        solution,
        cmap='viridis',
        linewidth=0,
        antialiased=True
    )

    # # ZOOMED RANGES FOR BETTER VISUALIZATION
    # ax.set_xlim(50, 150)        # Focus around ATM
    # ax.set_ylim(0.1, 0.9)       # Realistic volatility range
    # ax.set_zlim(0, 60)          # Adjust based on payoff scale

    ax.view_init(elev=30, azim=135)

    ax.set_xlabel("Forward F")
    ax.set_ylabel("Volatility σ")
    ax.set_zlabel("Option Value")

    plt.title("SABR PDE Solution Surface (Zoomed)")
    plt.tight_layout()
    plt.show()
