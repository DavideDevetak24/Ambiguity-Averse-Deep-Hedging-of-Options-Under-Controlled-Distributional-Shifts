"""
In this script, we create various charts to visualize the performance of the hedging strategy.

"""

# Adjust path for imports
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
# Import modules from other scripts
from risk_measures.test_metrics import PL_summary
from utils.charts_utils import build_comparison_datasets

# plot P&L distributions as overlayed histograms
def plot_pnl_distributions(pl_1, pl_2, bins_range=(-10, 10), num_bins=50, 
                           labels=("SDH", "AADH"), 
                           title="P&L Distributions", xlabel="P&L", ylabel="Density"):
    # print summary statistics
    summary_1 = PL_summary(pl_1)
    summary_2 = PL_summary(pl_2)
    print(f"{labels[0]} P&L Summary: Mean = {summary_1['mean']:.8f}, Std = {summary_1['std']:.8f}")
    print(f"{labels[1]} P&L Summary: Mean = {summary_2['mean']:.8f}, Std = {summary_2['std']:.8f}")
    # Define bins
    bins = np.linspace(bins_range[0], bins_range[1], num_bins)

    # Compute normalized histogram counts
    hist_1, _ = np.histogram(pl_1, bins=bins, density=False)
    hist_2, _ = np.histogram(pl_2, bins=bins, density=False)
    
    # Center positions for bars
    width = (bins[1] - bins[0]) * 0.4          # half bin width per strategy
    centers = 0.5 * (bins[1:] + bins[:-1])

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.bar(centers - width/2, hist_1, width=width, color="blue", alpha=0.7, label=labels[0])
    plt.bar(centers + width/2, hist_2, width=width, color="orange", alpha=0.7, label=labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Plot_deltas
"""
This version might be good for the MLP model in NN where we calculate delta_n+1
Uncomment to use the MLP module
"""
# def plot_deltas(test_loader, model, n_paths=50, device='cpu'):
#     # Switch model to evaluation mode
#     model.eval()

#     # Collect all paths and deltas
#     paths_list = []
#     deltas_list = []

#     with torch.no_grad():
#         for paths, labels in test_loader:
#             paths = paths.to(device)
#             delta = model(paths)
#             paths_list.append(paths.cpu().numpy())
#             deltas_list.append(delta.cpu().numpy())

#     # Combine batches into one array
#     paths_all = np.concatenate(paths_list, axis=0)     # (N, T)
#     deltas_all = np.concatenate(deltas_list, axis=0)   # (N, T-1)

#     T = paths_all.shape[1]

#     # Define x-axes
#     time_steps_paths = np.arange(T)        # 0, 1, ..., T-1
#     time_steps_deltas = np.arange(T)       # include 0 for alignment

#     plt.figure(figsize=(12, 12))

#     # First plot: Deltas with NaN prepended
#     # plot starts at zero but shows zero at the initial time step on the axis for alignment
#     plt.subplot(2, 1, 1)
#     for i in range(min(n_paths, deltas_all.shape[0])):
#         # Prepend NaN so nothing is plotted at initial time step
#         deltas_with_nan = np.insert(deltas_all[i], 0, np.nan)
#         plt.plot(time_steps_deltas, deltas_with_nan)
    
#     plt.title('Deltas Over Time')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Delta')
#     plt.grid()
#     plt.xlim(0, T - 1)
#     # Second plot: Underlying Price Paths
#     plt.subplot(2, 1, 2)
#     for i in range(min(n_paths, paths_all.shape[0])):
#         plt.plot(time_steps_paths, paths_all[i])

#     plt.title('Underlying Price Paths')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Price')
#     plt.grid()
#     plt.xlim(0, T - 1)

#     plt.tight_layout()
#     plt.show()


def plot_deltas(test_loader, model, n_paths=50, device='cpu'):
    model.eval()

    paths_list = []
    deltas_list = []

    device = torch.device(device)
    with torch.inference_mode():
        for paths, _ in test_loader:
            paths = paths.to(device, non_blocking=True)
            delta = model(paths)  # expected shape: (batch_size, T-1)
            paths_list.append(paths.cpu().numpy())
            deltas_list.append(delta.cpu().numpy())

    paths_all = np.concatenate(paths_list, axis=0)     # (N, T)
    deltas_all = np.concatenate(deltas_list, axis=0)   # (N, T-1)

    T = paths_all.shape[1]

    time_steps_paths = np.arange(T)
    time_steps_deltas = np.arange(T)

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    for i in range(min(n_paths, deltas_all.shape[0])):
        # Append NaN so there is no hedge at the final time step T-1
        deltas_with_nan = np.append(deltas_all[i], np.nan)
        plt.plot(time_steps_deltas, deltas_with_nan)

    plt.title('Deltas Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Delta')
    plt.grid()
    plt.xlim(0, T - 1)

    plt.subplot(2, 1, 2)
    for i in range(min(n_paths, paths_all.shape[0])):
        plt.plot(time_steps_paths, paths_all[i])

    plt.title('Underlying Price Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.grid()
    plt.xlim(0, T - 1)

    plt.tight_layout()
    plt.show()





# create function to build scatter plots of deltas vs S/K for the two models
def plot_models_deltas_comparison(model_1, model_2, data_loader, time_step, K, device='cpu'):
    comparison_df = build_comparison_datasets(
        model_1, model_2, data_loader, time_step, K, device
    )

    # sort by underlying for cleaner plots
    comparison_df = comparison_df.sort_values(by='S/K_SDH')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # plot 1: Deltas of both models
    axes[0].scatter(
        comparison_df['S/K_SDH'],
        comparison_df['Delta_SDH'],
        alpha=0.4,
        s=10,
        label='SDH'
    )

    axes[0].scatter(
        comparison_df['S/K_SDH'],
        comparison_df['Delta_AADH'],
        alpha=0.4,
        s=10,
        label='AADH'
    )

    axes[0].set_xlabel('S / K')
    axes[0].set_ylabel('Delta')
    axes[0].set_title(f'DH Holdings at Time Step {time_step}')
    axes[0].legend()
    axes[0].grid(True)

    # plot 2: Difference in holdings
    axes[1].scatter(
        comparison_df['S/K_SDH'],
        comparison_df['Delta_Difference'],
        alpha=0.4,
        s=10
    )

    axes[1].set_xlabel('S / K')
    axes[1].set_ylabel('Delta Difference')
    axes[1].set_title(f'Delta Difference (AADH - SDH) at Time Step {time_step}')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()




# Test functions diagnostics

# Create dummy dataset and dataloader
# from AADH_GAN_Scripts.dataset.dataset import AADeepHedgingDataset, loader
# from AADH_GAN_Scripts.models.neural_network import HedgingNN
# from torch.optim import Adam
# from AADH_GAN_Scripts.training.train_loop_nn import main
# from AADH_GAN_Scripts.risk_measures.test_metrics import *
# import pandas as pd
# import numpy as np

# n_paths = 50
# n_steps = 10

# cost_rate = 0.002
# lam = 1/100
# strike = 0.5
# # Create random paths
# data_S = pd.DataFrame(np.random.rand(n_steps, n_paths))
# # Create random labels (3 clusters)
# labels = np.random.randint(0, 3, size=n_paths)
# dataset = AADeepHedgingDataset(data_S, labels)
# train_loader, val_loader, test_loader = loader(dataset, batch_size=5)

# # Create model
# model_1 = HedgingNN()
# optimizer_1 = Adam(model_1.parameters(), lr=0.001)

# # test main function
# model_1 = main(model_1, train_loader, val_loader, epochs=5, 
#              AADH=True, adaptive_learning=True, device='cpu')

# model_2 = HedgingNN()
# optimizer_2 = Adam(model_2.parameters(), lr=0.001)
# model_2 = main(model_2, train_loader, val_loader, epochs=5, 
#              AADH=False, adaptive_learning=True, device='cpu')
    
# # print(test_result_metrics(model, test_loader, strike=strike, cost_rate=cost_rate, lam=lam))

# # print(compute_PL_array(model, test_loader, strike=strike, cost_rate=cost_rate, p0=0.0))
# # print(PL_summary(compute_PL_array(model, test_loader, strike=strike, cost_rate=cost_rate, p0=0.0)))

# # Generate P&L arrays using dummy data
# PL_array_1 = np.random.normal(0, 1, size=1000)
# PL_array_2 = np.random.normal(0, 1, size=1000)
# # Plot P&L distributions
# PL_array_1 = np.random.normal(0, 1, size=1000)
# PL_array_2 = np.random.normal(0, 1, size=1000)
    
# # plot_pnl_distributions(PL_array_1, PL_array_2, bins_range=(-5, 5), num_bins=50, 
# #                        labels=("Strategy 1", "Strategy 2"), title="P&L Distributions Comparison", 
# #                        xlabel="P&L", ylabel="Density")

# plot_deltas(test_loader=test_loader, model=model_1, n_paths=10, device='cpu')

# # print(calculate_deltas_chart(model, test_loader, device='cpu'))
# # df_delta, df_paths = calculate_deltas_chart(model, test_loader, device='cpu')
# # print(slice_deltas_chart(df_delta, df_paths, time_step=1))
# # build_df_sliced_chart(model, test_loader, time_step=1, device='cpu')
# # print(build_comparison_datasets(model_1, model_2, test_loader, time_step=1, K=0.5, device='cpu'))

# plot_models_deltas_comparison(model_1, model_2, test_loader, time_step=1, K=0.5, device='cpu')



