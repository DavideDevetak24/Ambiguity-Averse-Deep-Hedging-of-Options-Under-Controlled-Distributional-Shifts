"""
Vectorized GARCH(1,1) price-path generator.

The generator returns dataframes with the same orientation used by the Heston
generator: rows are time steps and columns are simulated paths.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _garch_step_intercept(omega, alpha, beta, long_run_variance, dt):
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be non-negative.")
    if alpha + beta >= 1.0:
        raise ValueError("GARCH stationarity requires alpha + beta < 1.")

    if omega is not None:
        return float(omega)

    return float(long_run_variance) * dt * (1.0 - float(alpha) - float(beta))


def garch_data_generator(S0=1.0,
                         T=30/252,
                         N=30,
                         mu=0.0,
                         omega=None,
                         alpha=0.05,
                         beta=0.90,
                         long_run_variance=0.04,
                         initial_variance=0.04,
                         n_simulations=1000,
                         random=True,
                         seed=42):
    """
    Generate price and conditional variance paths from a GARCH(1,1) process.

    Parameters
    ----------
    S0:
        Initial asset price.
    T, N:
        Time horizon and number of steps.
    mu:
        Annualized drift. Set to 0.0 for the same convention used in the main
        notebooks.
    omega:
        Optional per-step GARCH intercept. If None, it is derived from the
        annualized long-run variance.
    alpha, beta:
        GARCH(1,1) parameters. They must satisfy alpha + beta < 1.
    long_run_variance, initial_variance:
        Annualized variances. The recursion is done on per-step variance and
        returned as annualized conditional variance for readability.
    n_simulations:
        Number of price paths.
    random, seed:
        If random is True, use the provided seed. If random is False, use seed 42.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Price paths and annualized conditional variance paths.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    dt = T / N
    if dt <= 0:
        raise ValueError("T / N must be positive.")

    omega_step = _garch_step_intercept(omega, alpha, beta, long_run_variance, dt)

    data_S = np.empty((N + 1, n_simulations))
    data_var_step = np.empty((N + 1, n_simulations))
    data_S[0, :] = S0
    data_var_step[0, :] = float(initial_variance) * dt

    rng = np.random.default_rng(seed if random else 42)
    innovations = rng.standard_normal(size=(N, n_simulations))

    for j in range(1, N + 1):
        var_step = np.maximum(data_var_step[j - 1, :], 1e-12)
        shock = np.sqrt(var_step) * innovations[j - 1, :]
        log_return = float(mu) * dt - 0.5 * var_step + shock

        data_S[j, :] = data_S[j - 1, :] * np.exp(log_return)
        data_var_step[j, :] = omega_step + float(alpha) * shock**2 + float(beta) * var_step

    annualized_variance = data_var_step / dt

    return pd.DataFrame(data_S), pd.DataFrame(annualized_variance)


def garch_option_price_mc(data_S, K, mu=0.0, T=30/252):
    S_T = data_S.iloc[-1, :].values
    n_simulations = data_S.shape[1]
    payoffs = np.maximum(S_T - K, 0.0)
    price_mc = np.exp(-float(mu) * T) * np.mean(payoffs)
    std_error = np.exp(-float(mu) * T) * np.std(payoffs) / np.sqrt(n_simulations)

    print(f"Monte Carlo option price {price_mc:.4f} +- {1.96*std_error:.4f} (95% CI)")

    return price_mc, std_error


def garch_chart(data_S, data_variance, n_sim=20):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(data_S.iloc[:, :n_sim])
    plt.title("Asset Price Process - GARCH")
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(data_variance.iloc[:, :n_sim])
    plt.title("Annualized Conditional Variance Process - GARCH")
    plt.xlabel("t")
    plt.ylabel("variance")
    plt.grid()

    plt.tight_layout()
    plt.show()
