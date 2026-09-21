"""
Vectorized CEV price-path generator.

Model:
    dS_t = sigma * sqrt(S_t) dW_t

The generator returns dataframes with the same orientation used by the Heston
and GARCH generators: rows are time steps and columns are simulated paths.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cev_data_generator(S0=1.0,
                       T=30/252,
                       N=30,
                       sigma=0.2,
                       n_simulations=1000,
                       random=True,
                       seed=42,
                       absorb_zero=True):
    """
    Generate price paths from dS_t = sigma * sqrt(S_t) dW_t.

    Euler discretization:
        S_{t+1} = S_t + sigma * sqrt(max(S_t, 0)) * sqrt(dt) * Z_t

    If absorb_zero is True, paths are floored at zero after each step, which
    matches the non-negative state space of the square-root diffusion.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Price paths and instantaneous variance paths sigma^2 * S_t.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")
    if S0 < 0:
        raise ValueError("S0 must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    dt = T / N
    if dt <= 0:
        raise ValueError("T / N must be positive.")

    data_S = np.empty((N + 1, n_simulations))
    data_S[0, :] = S0

    rng = np.random.default_rng(seed if random else 42)
    shocks = rng.standard_normal(size=(N, n_simulations))

    for j in range(1, N + 1):
        S_prev = data_S[j - 1, :]
        diffusion_scale = sigma * np.sqrt(np.maximum(S_prev, 0.0))
        S_next = S_prev + diffusion_scale * np.sqrt(dt) * shocks[j - 1, :]

        if absorb_zero:
            S_next = np.maximum(S_next, 0.0)

        data_S[j, :] = S_next

    instantaneous_variance = sigma**2 * np.maximum(data_S, 0.0)

    return pd.DataFrame(data_S), pd.DataFrame(instantaneous_variance)


def cev_option_price_mc(data_S, K, r=0.0, T=30/252):
    S_T = data_S.iloc[-1, :].values
    n_simulations = data_S.shape[1]
    payoffs = np.maximum(S_T - K, 0.0)
    price_mc = np.exp(-float(r) * T) * np.mean(payoffs)
    std_error = np.exp(-float(r) * T) * np.std(payoffs) / np.sqrt(n_simulations)

    print(f"Monte Carlo option price {price_mc:.4f} +- {1.96*std_error:.4f} (95% CI)")

    return price_mc, std_error


def cev_chart(data_S, data_variance, n_sim=20):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(data_S.iloc[:, :n_sim])
    plt.title("Asset Price Process - CEV")
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(data_variance.iloc[:, :n_sim])
    plt.title("Instantaneous Variance Process - CEV")
    plt.xlabel("t")
    plt.ylabel("sigma^2 S(t)")
    plt.grid()

    plt.tight_layout()
    plt.show()
