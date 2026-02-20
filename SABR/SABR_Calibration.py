from scipy.optimize import minimize
import numpy as np


class SABRCalibration:
    
    def __init__(self, F, T, beta, strikes, market_vols):
        self.F = F
        self.T = T
        self.beta = beta
        self.strikes = np.asarray(strikes)
        self.market_vols = np.asarray(market_vols)
        

    def hagan_vol(self, alpha, rho, nu):

        F = self.F
        T = self.T
        beta = self.beta
        K = self.strikes

        eps = 1e-12

        FK = (F * K) ** ((1 - beta) / 2)
        logFK = np.log(F / K)

        z = (nu / alpha) * FK * logFK

        sqrt_term = np.sqrt(1 - 2*rho*z + z**2)
        x_z = np.log((sqrt_term + z - rho) / (1 - rho))

        # Stability when z → 0
        z_over_x = np.where(np.abs(z) < 1e-8, 1.0, z / x_z)

        leading = (alpha / FK) * z_over_x

        correction = (
            ((1 - beta)**2 / 24) * (alpha**2 / (FK**2))
            + (1/4) * (rho * beta * nu * alpha / FK)
            + ((2 - 3*rho**2) / 24) * nu**2
        ) * T

        sigma = leading * (1 + correction)

        # ATM override (more accurate)
        atm_mask = np.abs(F - K) < eps
        if np.any(atm_mask):

            term1 = alpha / (F ** (1 - beta))

            correction_atm = (
                ((1 - beta)**2 / 24) * (alpha**2 / (F**(2 - 2*beta)))
                + (1/4) * (rho * beta * nu * alpha / (F**(1 - beta)))
                + ((2 - 3*rho**2) / 24) * nu**2
            ) * T

            sigma[atm_mask] = term1 * (1 + correction_atm)

        return sigma
    

    def objective(self, params):

        alpha, rho, nu = params

        # Parameter safety
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10

        model_vols = self.hagan_vol(alpha, rho, nu)

        error = model_vols - self.market_vols

        return np.sum(error**2)


    def calibrate(self):

        initial_guess = [0.2, 0.0, 0.5]
        bounds = [(0.001, 2.0), (-0.999, 0.999), (0.001, 5.0)]

        result = minimize(
            self.objective,
            initial_guess,
            bounds=bounds,
            method="L-BFGS-B"
        )

        return result.x


# Example Usage
if __name__ == "__main__":

    F = 100
    T = 1
    beta = 0.5

    strikes = np.array([80, 90, 100, 110, 120])
    market_vols = np.array([0.25, 0.22, 0.20, 0.21, 0.23])

    calibrator = SABRCalibration(F, T, beta, strikes, market_vols)

    alpha, rho, nu = calibrator.calibrate()

    print(f"Calibrated Parameters: alpha={alpha:.4f}, rho={rho:.4f}, nu={nu:.4f}")
