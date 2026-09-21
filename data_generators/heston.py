"""
The script provides three functions to generate Heston model data and related utilities:

1. heston_data_generator: Simulates stock price and variance paths using the Heston stochastic
    volatility model via Euler-Maruyama discretization.
2. heston_option_price_mc: Computes European call option prices using Monte Carlo simulation
    based on the generated Heston model paths.
3. heston_chart: Visualizes sample paths of the asset price and variance processes.

"""

# Create a Heston model to simulate stock prices and volatility paths
# The will be imported into another script for feature clustering so it will function as a module

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define Heston parameters: these can be modified when calling the functions
S0 = 100          # initial stock price
v0 = 0.04         # initial variance
T = 1             # time horizon
N = 252           # number of yearly steps
r = 0.03         # risk-free rate
mu = r            # drift under risk neutral measure
kappa = 1         # rate of mean reversion
theta = 0.04      # long-term variance
sigma_v = 2    # volatility of volatility
rho = -0.7         # correlation between Brownian motions

# Define strike price for option price calculation
K = 100

# Number of Monte Carlo simulations
n_simulations = 100

# Function to generate Heston model data
def heston_data_generator(S0, 
                          v0, 
                          T, 
                          N, 
                          r, 
                          mu, 
                          kappa, 
                          theta, 
                          sigma_v, 
                          rho, 
                          n_simulations=1000, 
                          risk_neutrality=True, 
                          random=True,
                          seed=42):
    
    if risk_neutrality == True:
        mu = r
    else:
        mu = mu  # keep original mu if not risk neutral
    dt = T / N

    data_S = np.empty((N+1, n_simulations))
    data_v = np.empty((N+1, n_simulations))
    data_S[0, :] = S0
    data_v[0, :] = v0

    rng = np.random.default_rng(seed if random else 42)

    # Proper correlated Brownian increments for all paths at once.
    Z = rng.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[1.0, rho], [rho, 1.0]],
        size=(N, n_simulations)
    )
    dW_s = np.sqrt(dt) * Z[:, :, 0]
    dW_v = np.sqrt(dt) * Z[:, :, 1]

    for j in range(1, N+1):
        # Reflection method from "The Volatility Surface" by Jim Gatheral.
        vt = np.abs(data_v[j-1, :])
        sqrt_vt = np.sqrt(vt)
        data_S[j, :] = data_S[j-1, :] * np.exp((mu - 0.5 * vt) * dt + sqrt_vt * dW_s[j-1, :])
        v_new = data_v[j-1, :] + kappa * (theta - vt) * dt + sigma_v * sqrt_vt * dW_v[j-1, :]
        data_v[j, :] = np.abs(v_new)

    return pd.DataFrame(data_S), pd.DataFrame(data_v)


# Calculate MC option price
def heston_option_price_mc(data_S, 
                           K, 
                           mu, 
                           r, 
                           T, 
                           risk_neutrality=True):
    # Use Heston generated data
    S_T = data_S.iloc[-1, :].values  # S_T shape: (n_simulations,)
    # Consider risk neutrality feature
    if risk_neutrality == True:
        mu = r
    else:
        mu = mu  # keep original mu if not risk neutral
    # Count the number of simulations
    n_simulations = data_S.shape[1]
    # compute discounted payoff
    payoffs = np.maximum(S_T - K, 0.0)
    price_mc = np.exp(-mu * T) * np.mean(payoffs)
    std_error = np.exp(-mu * T) * np.std(payoffs) / np.sqrt(n_simulations)

    print(f"Monte Carlo option price {price_mc:.4f} +- {1.96*std_error:.4f} (95% CI)")
    
    return price_mc, std_error


# Create Heston chart sampling the first n simulations
def heston_chart(data_S, data_v, n_sim=20):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(data_S.iloc[:,:n_sim])
    plt.title('Asset Price Process - Heston')
    plt.xlabel('t')
    plt.ylabel('S(t)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(data_v.iloc[:,:n_sim])
    plt.title('Variance Process - Heston')
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.grid()

    plt.tight_layout()
    plt.show()

