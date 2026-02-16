import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import matplotlib.pyplot as plt

def second_derivative_method(x, y):
    """
    Find the elbow point using the second derivative method.
    """
    # Calculate first differences
    dx = np.diff(x)
    dy = np.diff(y)
    first_derivative = dy / dx

    # Calculate second differences
    dx2 = np.diff(x[1:])
    dy2 = np.diff(first_derivative)
    second_derivative = dy2 / dx2

    # Find the point of maximum second derivative
    elbow_idx = np.argmax(np.abs(second_derivative)) + 2
    return elbow_idx

def find_optimal_clusters(data, max_clusters=10):
    """
    Find the optimal number of clusters using multiple methods:
    1. Elbow Method (with automatic knee detection)
    2. Second Derivative Method
    3. Silhouette Analysis
    """
    X = data.reshape(-1, 1)
    n_clusters_range = range(2, max_clusters + 1)  # Start from 2 clusters
    
    # Store metrics
    inertias = []
    silhouette_scores = []
    
    # Calculate metrics for each number of clusters
    for n_clusters in n_clusters_range:
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        
        # Calculate inertia (for elbow method)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Find optimal number using elbow method (knee detection)
    # Normalize inertias to make the elbow detection more sensitive
    inertias_normalized = np.array(inertias) / max(inertias)
    
    knee_locator = KneeLocator(
        list(n_clusters_range),
        inertias_normalized,
        curve='convex',
        direction='decreasing',
        S=1.0,  # Increase sensitivity
        online=True,
        interp_method='polynomial'
    )
    elbow_optimal = knee_locator.elbow
    
    # Find optimal number using silhouette score
    silhouette_optimal = n_clusters_range[np.argmax(silhouette_scores)]
    
    # Second derivative method
    second_derivative_optimal = n_clusters_range[
        second_derivative_method(
            np.array(list(n_clusters_range)), 
            np.array(inertias_normalized)
        )
    ]

    return {
        'elbow_optimal': elbow_optimal,
        'second_derivative_optimal': second_derivative_optimal,
        'silhouette_optimal': silhouette_optimal,
        'metrics': {
            'n_clusters_range': list(n_clusters_range),
            'inertias': inertias,
            'inertias_normalized': inertias_normalized.tolist(),
            'silhouette_scores': silhouette_scores
        }
    }

def plot_cluster_metrics(metrics):
    """
    Plot the metrics used for determining optimal cluster number.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    n_clusters = metrics['n_clusters_range']
    
    # Plot Elbow curve
    ax1.plot(n_clusters, metrics['inertias'], 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_clusters)
    
    # Plot Silhouette scores
    ax2.plot(n_clusters, metrics['silhouette_scores'], 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(n_clusters)
    
    plt.tight_layout()
    return fig

def cluster_neural_responses(data):
    """
    Perform K-means clustering with automatically determined number of clusters.
    """
    # Extract latency for clustering
    X = data['latency'].values.reshape(-1, 1)
    
    # Find optimal number of clusters
    optimal_results = find_optimal_clusters(X)
    
    # Use the silhouette optimal number
    n_clusters = optimal_results['second_derivative_optimal']   
    
    # Perform clustering with optimal number
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Add cluster labels to the original data
    result_df = data.copy()
    result_df['cluster'] = cluster_labels
    
    return result_df, n_clusters, kmeans, optimal_results

def plot_clusters(data, kmeans):
    """
    Plot the clustering results.
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
    plt.title(f'K-means Clustering with {len(kmeans.cluster_centers_)} clusters')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)
    
    return plt.gcf()
