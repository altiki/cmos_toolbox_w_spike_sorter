import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def cluster_neural_responses(data, n_clusters=3):
    """
    Perform K-means clustering on neural response data, focusing on latency bands.
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing 'spike time', 'latency', and 'amplitude' columns
    n_clusters (int): Number of clusters to identify (default=3)
    
    Returns:
    tuple: (DataFrame with cluster labels, KMeans model)
    """
    # Extract latency for clustering - reshape to 2D array for sklearn
    X = data['latency'].values.reshape(-1, 1)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Add cluster labels to the original data
    result_df = data.copy()
    result_df['cluster'] = cluster_labels
    
    return result_df, kmeans

def plot_clusters(data, kmeans):
    """
    Plot the clustering results.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with cluster labels
    kmeans (KMeans): Fitted KMeans model
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with points colored by cluster
    scatter = plt.scatter(data['latency'], data['spike time'], 
                         c=data['cluster'], cmap='viridis',
                         alpha=0.6, s=100)
    
    # Plot cluster centers as vertical lines
    for i, center in enumerate(kmeans.cluster_centers_):
        plt.axvline(x=center[0], color='red', linestyle='--', alpha=0.5,
                   label=f'Cluster {i} center')
    
    plt.xlabel('Latency (ms)')
    plt.ylabel('Experiment Time (s)')
    plt.title('K-means Clustering of Neural Response Latency Bands')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Focus the x-axis on the relevant region
    plt.xlim(0, 10.)
    
    return plt.gcf()

def analyze_clusters(data):
    """
    Analyze the characteristics of each cluster.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with cluster labels
    """
    # Calculate statistics for each cluster
    cluster_stats = data.groupby('cluster').agg({
        'latency': ['count', 'mean', 'std'],
        'amplitude': ['mean', 'std']
    }).round(3)
    
    # Calculate the average distance between cluster centers
    cluster_centers = np.sort(data.groupby('cluster')['latency'].mean().values)
    distances = np.diff(cluster_centers)
    
    return cluster_stats, distances
