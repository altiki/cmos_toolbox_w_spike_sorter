import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def perform_hierarchical_clustering(data, n_clusters=3, method='ward'):
    """
    Perform hierarchical clustering on neural response data, focusing on latency bands.
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing 'spike time', 'latency', and 'amplitude' columns
    n_clusters (int): Number of clusters to identify (default=3)
    method (str): Linkage method ('ward', 'complete', 'average', or 'single')
    
    Returns:
    tuple: (DataFrame with cluster labels, linkage matrix)
    """
    # Extract latency for clustering - reshape to 2D array
    X = data['latency'].values.reshape(-1, 1)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(X, method=method)
    
    # Get cluster assignments
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
    
    # Add cluster labels to the original data
    result_df = data.copy()
    result_df['cluster'] = cluster_labels
    
    return result_df, linkage_matrix

def plot_clusters_with_dendrogram(data, linkage_matrix):
    """
    Plot the clustering results with a dendrogram.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with cluster labels
    linkage_matrix: Linkage matrix from hierarchical clustering
    """
    # Create figure with subplot grid
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
    
    # Dendrogram
    ax_dendrogram = fig.add_subplot(gs[0, 0])
    dendrogram(linkage_matrix, ax=ax_dendrogram)
    ax_dendrogram.set_xticklabels([])
    ax_dendrogram.set_ylabel('Distance')
    ax_dendrogram.set_title('Dendrogram')
    
    # Scatter plot
    ax_scatter = fig.add_subplot(gs[1, 0])
    scatter = ax_scatter.scatter(data['latency'], data['spike time'], 
                               c=data['cluster'], cmap='viridis',
                               alpha=0.6, s=100)
    
    # Add vertical lines for cluster centers
    for cluster in np.unique(data['cluster']):
        center = data[data['cluster'] == cluster]['latency'].mean()
        ax_scatter.axvline(x=center, color='red', linestyle='--', alpha=0.5,
                          label=f'Cluster {cluster} center')
    
    ax_scatter.set_xlabel('Latency (ms)')
    ax_scatter.set_ylabel('Experiment Time (s)')
    ax_scatter.set_title('Hierarchical Clustering of Neural Response Latency Bands')
    plt.colorbar(scatter, ax=ax_scatter, label='Cluster')
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_xlim(2.5, 4.5)
    
    # Adjust layout
    plt.tight_layout()
    return fig

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

def plot_cluster_distribution(data):
    """
    Plot the distribution of latencies for each cluster.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with cluster labels
    """
    plt.figure(figsize=(12, 6))
    
    for cluster in sorted(data['cluster'].unique()):
        cluster_data = data[data['cluster'] == cluster]['latency']
        plt.hist(cluster_data, bins=30, alpha=0.5, 
                label=f'Cluster {cluster}', density=True)
    
    plt.xlabel('Latency (ms)')
    plt.ylabel('Density')
    plt.title('Distribution of Latencies by Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()