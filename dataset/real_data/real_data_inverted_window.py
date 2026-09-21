"""
We create a script to create a dataset suitable for testing the AADH and SDH models,
using the reversed window convention.

(Verify with the original paper)

"""

# Import libraries
import os
import numpy as np
import pandas as pd

def create_real_data_dataset(input_file_name, output_file_name, n_steps):
    """
    Create a csv file using a sliding window approach to generate paths
    from an existing empirical time series.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(current_dir, input_file_name)
    output_path = os.path.join(current_dir, output_file_name)

    data = pd.read_excel(input_path)
    S_values = data['Price'].values

    # drop NaN values if any
    S_values = S_values[~np.isnan(S_values)]

    # Generate paths using sliding window
    paths = []
    n_data = len(S_values)
    for start_idx in range(n_data - n_steps):
        end_idx = start_idx + n_steps + 1
        path = S_values[start_idx:end_idx]
        paths.append(path)
    paths_array = np.array(paths)
    # Save to CSV
    df = pd.DataFrame(paths_array).T

    # normalize each path to start at 1
    for col in df.columns:
        df[col] = df[col] / df[col].iloc[0]

    df = df.T
    df.to_csv(output_path, index=False, header=False)
    return f"Dataset saved to {output_path} with shape {df.shape}"


# Apply script
print(create_real_data_dataset("ES_50_Covid.xlsx", "ES_50_Covid_samples_2.csv", 30))
# print(create_real_data_dataset("Euro_Stoxx_50.xlsx", "Euro_Stoxx_50_samples.csv", 30))
# print(create_real_data_dataset("FTSE_100.xlsx", "FTSE_100_samples.csv", 30))
# print(create_real_data_dataset("Nasdaq_100.xlsx", "Nasdaq_100_samples.csv", 30))
# print(create_real_data_dataset("Nikkei_225.xlsx", "Nikkei_225_samples.csv", 30))

