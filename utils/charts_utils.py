"""
Utils for charts.

"""

# import libraries
import numpy as np
import pandas as pd
import torch

# We create the utils for a chart that represent the delta over S/K
# We first find the deltas of given the model and the patsh from a dataloader
def calculate_deltas_chart(model, data_loader, device='cpu'):
    model.eval()
    deltas = []
    paths = []
    device = torch.device(device)
    with torch.inference_mode():
        for batch_paths, _ in data_loader:
            batch_paths = batch_paths.to(device, non_blocking=True)
            batch_deltas = model(batch_paths)
            deltas.append(batch_deltas.cpu())
            paths.append(batch_paths.cpu())

    # transform deltas and paths in pandas dataframes
    deltas = torch.cat(deltas, dim=0).numpy().squeeze(-1)
    paths = torch.cat(paths, dim=0).numpy().squeeze(-1)
    deltas_df = pd.DataFrame(deltas) # shape (n_paths, n_steps)
    paths_df = pd.DataFrame(paths) # shape (n_paths, n_steps+1)
    return deltas_df, paths_df


# Old version of plot_deltas
"""
The version might be good for second model in NN where we calculate delta_n+1
"""
# # now we slice the dataframes for a specific time to maturity
# # the function returns two columns dataframes: one for deltas and one for paths
# def slice_deltas_chart(deltas_df, paths_df, time_step):
#     # check that time_step is valid
#     if time_step < 0 or time_step > deltas_df.shape[1]:
#         raise ValueError("Invalid time_step value.")
#     # The deltas start at time step 1, so we slice accordingly
#     deltas_slice = deltas_df.iloc[:, time_step-1] # shape (n_paths,)
#     paths_slice = paths_df.iloc[:, time_step] # shape (n_paths,)
#     return deltas_slice, paths_slice

# now we slice the dataframes for a specific time to maturity
# the function returns two columns dataframes: one for deltas and one for paths
def slice_deltas_chart(deltas_df, paths_df, time_step):
    """
    We slice the deltas and paths dataframes at a specific time step.
    Indexes are matched so that deltas at time step t correspond to paths at time step t.

    Old version assumed deltas started at time step 1, but now they start at time step 0.
    The old version might be useful for models predicting delta at next time step. eg.model proposed in
    "Ambiguity-Averse Deep Hedging with Feature Clustering" by Jones et al., 2025.
    
    """

    # check that time_step is valid
    if time_step < 0 or time_step > deltas_df.shape[1]:
        raise ValueError("Invalid time_step value.")
    # The deltas start at time step 1, so we slice accordingly
    deltas_slice = deltas_df.iloc[:, time_step] # shape (n_paths,)
    paths_slice = paths_df.iloc[:, time_step] # shape (n_paths,)
    return deltas_slice, paths_slice
    

# put together the two functions in a single one
def build_df_sliced_chart(model, data_loader, time_step, device='cpu'):
    deltas_df, paths_df = calculate_deltas_chart(model, data_loader, device)
    deltas_slice, paths_slice = slice_deltas_chart(deltas_df, paths_df, time_step)
    return deltas_slice, paths_slice

# function to create the datasets for the two models
def build_comparison_datasets(model_1, model_2, data_loader, time_step, K, device='cpu'):
    delta_sdh, S_sdh = build_df_sliced_chart(model_1, data_loader, time_step, device)
    delta_aadh, S_aadh = build_df_sliced_chart(model_2, data_loader, time_step, device)
    # transform S to S/K
    S_sdh = S_sdh / K
    # Create dataframe
    comparison_df = pd.DataFrame({
        'S/K_SDH': S_sdh,
        'Delta_SDH': delta_sdh,
        'Delta_AADH': delta_aadh
    })

    # compute difference of the two models' deltas
    comparison_df['Delta_Difference'] = comparison_df['Delta_AADH'] - comparison_df['Delta_SDH']

    return comparison_df

