"""
This script implements KMeans clustering on features extracted from generated paths.
There are 6 main functions that can be used in the main file

1. compute_feature_matrix: computes a feature matrix from the generated paths
2. kmeans_clustering: applies KMeans clustering to the feature matrix
3. calculate_cluster_probabilities: calculate the probabilities of each cluster and the labels of the clusters
4. plot_feature_distributions: plots histograms of the feature distributions
5. plot_clusters_3d: visualizes the clusters in a 3D feature space
6. plot_paths_by_clusters: plots the paths grouped by their assigned clusters

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

"""
Define features for KMeans clustering
"""
# We define the features
# For each feature we create a def that will be used in the for cycle for every path

# Max drawdown: takes a sample path and return the max drawdown value
def max_drawdown(path):
    max = path[0]
    drawdown = 0.0
    for i in range(1, len(path)):
        if path[i] > max:
            max = path[i]
        dd = (max - path[i]) / max
        if dd > drawdown:
            drawdown = dd
    return drawdown

# Autocorrelation: takes a sample path and return the autocorrelation value
def autocorrelation(path, lag=1):
    a = acf(path, nlags=lag)
    return a[lag]
    

# Realized volatility: takes a sample path and return the realized volatility
def realized_volatility(path):
    path = np.asarray(path)
    log_returns = np.log(path[1:] / path[:-1])
    realized_vol = np.sqrt(np.sum(log_returns**2))
    return realized_vol
    # data can be logged to reduce skewness 
    # (it produces better centroids but loss on test is similar)
    # return np.log(realized_vol + 1e-9)

# Create a min-max normalization function that maps from R to [0,1]
# The function takes a column vector and returns the normalized column vector
def min_max_normalize(column):
    min_val = np.min(column)
    max_val = np.max(column)
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column
    # return np.log(normalized_column + 1e-9)


def _lag1_autocorrelation_matrix(paths):
    centered = paths - paths.mean(axis=0, keepdims=True)
    denominator = np.sum(centered * centered, axis=0)
    numerator = np.sum(centered[1:, :] * centered[:-1, :], axis=0)
    return numerator / denominator

"""
Create feature matrix
"""
# Define function to compute features for all paths storing them into a matrix
def compute_feature_matrix(data_S):
    paths = data_S.to_numpy(dtype=float, copy=False)

    running_max = np.maximum.accumulate(paths, axis=0)
    max_drawdowns = np.max((running_max - paths) / running_max, axis=0)
    autocorrelations = _lag1_autocorrelation_matrix(paths)
    log_returns = np.diff(np.log(paths), axis=0)
    realized_volatilities = np.sqrt(np.sum(log_returns**2, axis=0))

    feature_matrix = np.column_stack([
        max_drawdowns,
        autocorrelations,
        realized_volatilities
    ])

    # Apply min-max normalization to each feature column
    for j in range(feature_matrix.shape[1]):
        feature_matrix[:, j] = min_max_normalize(feature_matrix[:, j])

    # Transform fature matrix to pandas DataFrame
    feature_matrix = pd.DataFrame(feature_matrix, columns=["Max_Drawdown", "Autocorrelation", "Realized_Volatility"])

    return feature_matrix   # (n_paths x 3) matrix

"""
KMeans clustering
"""
n_clusters = 5

# Apply KMeans clustering to the feature matrix
def kmeans_clustering(feature_matrix, n_clusters=n_clusters):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_model.fit(feature_matrix)
    labels = kmeans_model.labels_
    return labels, kmeans_model


"""
Probability calculation for each cluster
"""
def calculate_cluster_probabilities(labels):
    labels = np.asarray(labels)
    n_paths = len(labels)
    cluster_counts = np.bincount(labels)

    cluster_info = {}

    for cluster_label, count in enumerate(cluster_counts):
        probability = count / n_paths
        indices = np.where(labels == cluster_label)[0]

        cluster_info[cluster_label] = {
            "probability": probability,
            "path_indices": indices.tolist()
        }

    return cluster_info


"""
Useful functions for visualization
"""
# Function to plot feature distributions using histograms
def plot_feature_distributions(feature_matrix, bins=50):
    feature_matrix.hist(bins=bins, figsize=(12, 6))
    plt.suptitle("Feature Distributions")
    plt.show()

# Function to visualize clusters in 3D feature space
# The plot shows each feature as a point in 3D space colored by cluster label
def plot_clusters_3d(feature_matrix, labels):

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(feature_matrix['Max_Drawdown'], 
                         feature_matrix['Autocorrelation'], 
                         feature_matrix['Realized_Volatility'], 
                         c=labels, cmap='viridis', s=50)
    ax.set_xlabel('Max Drawdown')
    ax.set_ylabel('Autocorrelation')
    ax.set_zlabel('Realized Volatility')
    plt.title('KMeans Clusters in Feature Space')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

# Create function that takes tensor dataset_kmeans and computes a chart with the paths divided by clusters
def plot_paths_by_clusters(dataset_kmeans, n_clusters):
    cluster_paths = {i: [] for i in range(n_clusters)}

    for i in range(len(dataset_kmeans)):
        path, label = dataset_kmeans[i]
        label = int(label)   # Convert tensor to int
        cluster_paths[label].append(path.squeeze().numpy())  # optional: convert to 1D array

    # plot on the same figure the paths. The color of the path depends on the cluster
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    plt.figure(figsize=(10, 6))
    
    for cluster, paths in cluster_paths.items():
        for path in paths:
            plt.plot(path, color=colors[cluster], alpha=0.3)

    plt.title('Paths Colored by Cluster')
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.tight_layout()
    plt.grid()
    plt.show()


