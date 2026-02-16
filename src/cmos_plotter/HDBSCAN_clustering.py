import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
from scipy import stats

def perform_hdbscan_clustering(data, min_cluster_size=15, min_samples=2, epsilon=0.05):
    """
    Perform HDBSCAN clustering on neural response data with parameters tuned for fine separation.
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing 'spike time' and 'latency' columns
    min_cluster_size (int): The minimum size of clusters (smaller value for finer clustering)
    min_samples (int): The number of samples in a neighborhood (smaller value for sensitivity)
    epsilon (float): The epsilon value for cluster selection (smaller value for distinct clusters)
    """
    X = data['latency'].values.reshape(-1, 1)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='leaf',  # Changed to leaf for finer cluster detection
        metric='euclidean',
        cluster_selection_epsilon=float(epsilon),
        allow_single_cluster=True,
        prediction_data=True
    )
    
    cluster_labels = clusterer.fit_predict(X)
    
    result_df = data.copy()
    result_df['cluster'] = cluster_labels
    
    return result_df, clusterer

def find_density_peaks(latencies, bandwidth_factor=0.1):
    """
    Find peaks in the density distribution of latencies using a narrow bandwidth.
    """
    # Calculate appropriate bandwidth based on data range
    data_range = np.ptp(latencies)
    bandwidth = bandwidth_factor * data_range
    
    kde = stats.gaussian_kde(latencies, bw_method=bandwidth)
    x_eval = np.linspace(min(latencies), max(latencies), 1000)
    density = kde(x_eval)
    
    # Find peaks with minimum prominence
    peak_indices = []
    for i in range(1, len(density)-1):
        if density[i] > density[i-1] and density[i] > density[i+1]:
            peak_indices.append(i)
    
    peaks = x_eval[peak_indices]
    peak_densities = density[peak_indices]
    
    # Sort peaks by density
    sorted_idx = np.argsort(peak_densities)[::-1]
    return peaks[sorted_idx]

def optimize_clustering(data, min_clusters=2, max_clusters=6):
    """
    Optimize HDBSCAN parameters with focus on detecting distinct vertical bands.
    """
    X = data['latency'].values.reshape(-1, 1)
    
    # Find density peaks to guide parameter selection
    density_peaks = find_density_peaks(X.ravel())
    expected_clusters = len(density_peaks)
    
    # Adjusted parameter ranges for finer clustering
    min_cluster_sizes = range(5, 30, 2)  # Smaller sizes for finer clusters
    epsilon_range = [float(x) for x in np.arange(0.02, 0.15, 0.01)]  # Finer epsilon values
    min_samples_range = [2, 3, 4]  # Small min_samples for sensitivity
    
    best_score = -np.inf
    best_params = None
    best_n_clusters = 0
    results = []
    
    for min_cluster_size in min_cluster_sizes:
        for epsilon in epsilon_range:
            for min_samples in min_samples_range:
                if min_samples > min_cluster_size:
                    continue
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method='leaf',
                    metric='euclidean',
                    cluster_selection_epsilon=float(epsilon),
                    allow_single_cluster=True
                )
                
                clusterer.fit(X)
                n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
                
                if min_clusters <= n_clusters <= max_clusters:
                    # Calculate cluster centers
                    cluster_centers = []
                    for label in set(clusterer.labels_) - {-1}:
                        center = np.mean(X[clusterer.labels_ == label])
                        cluster_centers.append(center)
                    
                    # Score based on matching density peaks
                    if cluster_centers:
                        # Calculate minimum distances between cluster centers and density peaks
                        center_peak_distances = []
                        for peak in density_peaks:
                            min_dist = min(abs(peak - center) for center in cluster_centers)
                            center_peak_distances.append(min_dist)
                        peak_matching_score = np.mean(1 / (1 + np.array(center_peak_distances)))
                    else:
                        peak_matching_score = 0
                    
                    # Calculate noise ratio
                    noise_ratio = np.sum(clusterer.labels_ == -1) / len(X)
                    
                    # Combined score favoring solutions that match density peaks
                    combined_score = (peak_matching_score * (1 - noise_ratio) * 
                                   (1 - 0.5 * abs(n_clusters - expected_clusters) / max_clusters))
                    
                    results.append({
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'epsilon': epsilon,
                        'n_clusters': n_clusters,
                        'score': combined_score,
                        'noise_ratio': noise_ratio
                    })
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = {
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'epsilon': epsilon
                        }
                        best_n_clusters = n_clusters
    
    return {
        'best_params': best_params,
        'n_clusters_found': best_n_clusters,
        'best_score': best_score,
        'all_results': pd.DataFrame(results) if results else None
    }

def plot_clusters(data, clusterer):
    """
    Plot the clustering results.
    """
    plt.figure(figsize=(12, 8))
    
    unique_clusters = sorted(set(data['cluster'].unique()) - {-1})
    
    # Plot noise points first
    noise_mask = data['cluster'] == -1
    if noise_mask.any():
        plt.scatter(data.loc[noise_mask, 'latency'], 
                   data.loc[noise_mask, 'spike time'],
                   c='lightgray', alpha=0.5, s=100, label='Noise')
    
    # Plot clustered points
    scatter = plt.scatter(data.loc[~noise_mask, 'latency'],
                         data.loc[~noise_mask, 'spike time'],
                         c=data.loc[~noise_mask, 'cluster'],
                         cmap='viridis', alpha=0.6, s=100)
    
    # Plot cluster centers as vertical lines
    for cluster in unique_clusters:
        center = data[data['cluster'] == cluster]['latency'].mean()
        plt.axvline(x=center, color='red', linestyle='--', alpha=0.5,
                   label=f'Cluster {cluster} center')
    
    plt.xlabel('Latency (ms)')
    plt.ylabel('Experiment Time (s)')
    plt.title(f'HDBSCAN Clustering (Found {len(unique_clusters)} clusters)')
    if len(unique_clusters) > 0:
        plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)
    
    return plt.gcf()