import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the pickle file
def load_data(pickle_file_path):
    """Load waveform data from pickle file"""
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Prepare data for clustering (separate classes)
def prepare_waveforms_by_class(data):
    """Extract correctly classified waveforms for each class separately"""
    correct_waveforms_0 = data['correct_waveforms_0']
    correct_waveforms_1 = data['correct_wavefoms_1']  # Note: typo in original key name
    
    return correct_waveforms_0, correct_waveforms_1

# Perform UMAP embedding for a single class
def perform_umap_single_class(waveforms, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    """Apply UMAP dimensionality reduction to single class"""
    # Standardize the data
    scaler = StandardScaler()
    waveforms_scaled = scaler.fit_transform(waveforms)
    
    # Apply UMAP
    reducer = UMAP(n_neighbors=n_neighbors, 
                   min_dist=min_dist, 
                   n_components=n_components,
                   random_state=random_state)
    embedding = reducer.fit_transform(waveforms_scaled)
    
    return embedding, scaler, reducer

# Perform clustering for a single class
def perform_clustering_single_class(embedding, n_clusters=3, random_state=42):
    """Apply K-means clustering to UMAP embedding for single class"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding)
    return cluster_labels, kmeans

# Create visualization for a single class
def create_single_class_plot(embedding, cluster_labels, waveforms, class_num, n_clusters=3):
    """Create UMAP plot with cluster colors and waveform subplots for single class"""
    
    # Set up the figure
    fig = plt.figure(figsize=(16, 10))
    
    # Create color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Main UMAP plot
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    
    # Plot points colored by cluster
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.sum(mask) > 0:
            ax_main.scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[colors[i]], label=f'Cluster {i+1}', 
                           alpha=0.7, s=30)
    
    ax_main.set_xlabel('UMAP 1')
    ax_main.set_ylabel('UMAP 2')
    ax_main.set_title(f'UMAP Clustering of Class {class_num} Waveforms')
    ax_main.legend()
    
    # Add numbered annotations for cluster centers
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.sum(mask) > 0:
            center_x = np.mean(embedding[mask, 0])
            center_y = np.mean(embedding[mask, 1])
            ax_main.annotate(str(i+1), (center_x, center_y), 
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle="circle,pad=0.3", 
                                   facecolor='white', edgecolor='black'))
    
    # Create waveform plots for each cluster
    # Arrange clusters in a grid
    subplot_positions = []
    rows = 3
    cols = 4
    for r in range(rows):
        for c in range(2, cols):  # Start from column 2 (columns 0,1 are for main plot)
            subplot_positions.append((r, c))
    
    # Show waveforms for all clusters
    for i in range(min(n_clusters, len(subplot_positions))):
        row, col = subplot_positions[i]
        ax_wave = plt.subplot2grid((rows, cols), (row, col))
        
        # Get waveforms from this cluster
        mask = cluster_labels == i
        cluster_waveforms = waveforms[mask]
        
        if len(cluster_waveforms) > 0:
            # Plot mean waveform with std
            mean_waveform = np.mean(cluster_waveforms, axis=0)
            std_waveform = np.std(cluster_waveforms, axis=0)
            
            time_points = np.arange(len(mean_waveform))
            
            ax_wave.plot(time_points, mean_waveform, color=colors[i], linewidth=2)
            ax_wave.fill_between(time_points, 
                               mean_waveform - std_waveform,
                               mean_waveform + std_waveform,
                               alpha=0.3, color=colors[i])
            
            ax_wave.set_title(f'Cluster {i+1} (n={len(cluster_waveforms)})')
            ax_wave.set_xlabel('Time (samples)')
            ax_wave.set_ylabel('Amplitude')
            ax_wave.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Create combined comparison plot
def create_combined_comparison_plot(results_0, results_1, waveforms_0, waveforms_1):
    """Create side-by-side comparison of both classes"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Class 0 UMAP
    embedding_0, cluster_labels_0, n_clusters_0 = results_0['embedding'], results_0['cluster_labels'], results_0['n_clusters']
    colors_0 = plt.cm.Set1(np.linspace(0, 1, n_clusters_0))
    
    ax = axes[0, 0]
    for i in range(n_clusters_0):
        mask = cluster_labels_0 == i
        if np.sum(mask) > 0:
            ax.scatter(embedding_0[mask, 0], embedding_0[mask, 1], 
                      c=[colors_0[i]], label=f'Cluster {i+1}', alpha=0.7, s=20)
    
    # Add cluster numbers
    for i in range(n_clusters_0):
        mask = cluster_labels_0 == i
        if np.sum(mask) > 0:
            center_x = np.mean(embedding_0[mask, 0])
            center_y = np.mean(embedding_0[mask, 1])
            ax.annotate(str(i+1), (center_x, center_y), 
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="circle,pad=0.3", 
                               facecolor='white', edgecolor='black'))
    
    ax.set_title('Class 0 - UMAP Clusters')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()
    
    # Class 1 UMAP
    embedding_1, cluster_labels_1, n_clusters_1 = results_1['embedding'], results_1['cluster_labels'], results_1['n_clusters']
    colors_1 = plt.cm.Set2(np.linspace(0, 1, n_clusters_1))
    
    ax = axes[0, 1]
    for i in range(n_clusters_1):
        mask = cluster_labels_1 == i
        if np.sum(mask) > 0:
            ax.scatter(embedding_1[mask, 0], embedding_1[mask, 1], 
                      c=[colors_1[i]], label=f'Cluster {i+1}', alpha=0.7, s=20)
    
    # Add cluster numbers
    for i in range(n_clusters_1):
        mask = cluster_labels_1 == i
        if np.sum(mask) > 0:
            center_x = np.mean(embedding_1[mask, 0])
            center_y = np.mean(embedding_1[mask, 1])
            ax.annotate(str(i+1), (center_x, center_y), 
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="circle,pad=0.3", 
                               facecolor='white', edgecolor='black'))
    
    ax.set_title('Class 1 - UMAP Clusters')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()
    
    # Class 0 average waveforms
    ax = axes[1, 0]
    for i in range(n_clusters_0):
        mask = cluster_labels_0 == i
        cluster_waveforms = waveforms_0[mask]
        
        if len(cluster_waveforms) > 0:
            mean_waveform = np.mean(cluster_waveforms, axis=0)
            time_points = np.arange(len(mean_waveform))
            ax.plot(time_points, mean_waveform, color=colors_0[i], 
                   linewidth=2, label=f'Cluster {i+1} (n={len(cluster_waveforms)})')
    
    ax.set_title('Class 0 - Average Waveforms by Cluster')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Class 1 average waveforms
    ax = axes[1, 1]
    for i in range(n_clusters_1):
        mask = cluster_labels_1 == i
        cluster_waveforms = waveforms_1[mask]
        
        if len(cluster_waveforms) > 0:
            mean_waveform = np.mean(cluster_waveforms, axis=0)
            time_points = np.arange(len(mean_waveform))
            ax.plot(time_points, mean_waveform, color=colors_1[i], 
                   linewidth=2, label=f'Cluster {i+1} (n={len(cluster_waveforms)})')
    
    ax.set_title('Class 1 - Average Waveforms by Cluster')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Print cluster statistics for single class
def print_single_class_stats(cluster_labels, class_num, n_clusters):
    """Print statistics about clusters for single class"""
    print(f"Class {class_num} Cluster Statistics:")
    print("-" * 40)
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_size = np.sum(mask)
        
        if cluster_size > 0:
            percentage = cluster_size / len(cluster_labels) * 100
            print(f"Cluster {i+1}: {cluster_size} samples ({percentage:.1f}%)")
    
    print()

# Analyze single class
def analyze_single_class(waveforms, class_num, n_clusters=3, umap_params=None):
    """Complete analysis for a single class"""
    
    if umap_params is None:
        umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 42}
    
    print(f"Analyzing Class {class_num}...")
    print(f"Number of waveforms: {len(waveforms)}")
    
    # Perform UMAP
    embedding, scaler, reducer = perform_umap_single_class(waveforms, **umap_params)
    
    # Perform clustering
    cluster_labels, kmeans = perform_clustering_single_class(embedding, n_clusters=n_clusters)
    
    # Print statistics
    print_single_class_stats(cluster_labels, class_num, n_clusters)
    
    # Create detailed plot
    fig = create_single_class_plot(embedding, cluster_labels, waveforms, class_num, n_clusters)
    
    return {
        'embedding': embedding,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'scaler': scaler,
        'reducer': reducer,
        'kmeans': kmeans,
        'fig': fig
    }

# Main execution function
def main(pickle_file_path, n_clusters_0=3, n_clusters_1=3, umap_params=None):
    """Main function to run separate clustering analysis for each class"""
    
    if umap_params is None:
        umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 42}
    
    # Load data
    print("Loading data...")
    data = load_data(pickle_file_path)
    
    # Prepare waveforms by class
    print("Preparing waveforms...")
    waveforms_0, waveforms_1 = prepare_waveforms_by_class(data)
    
    print(f"Class 0 waveforms: {len(waveforms_0)}")
    print(f"Class 1 waveforms: {len(waveforms_1)}")
    print(f"Waveform shape: {waveforms_0.shape[1]} samples")
    print()
    
    # Analyze Class 0
    results_0 = analyze_single_class(waveforms_0, 0, n_clusters_0, umap_params)
    
    # Analyze Class 1
    results_1 = analyze_single_class(waveforms_1, 1, n_clusters_1, umap_params)
    
    # Create combined comparison plot
    print("Creating combined comparison plot...")
    fig_combined = create_combined_comparison_plot(results_0, results_1, waveforms_0, waveforms_1)
    
    # Show all plots
    plt.show()
    
    # Return results for further analysis
    return {
        'class_0': results_0,
        'class_1': results_1,
        'waveforms_0': waveforms_0,
        'waveforms_1': waveforms_1,
        'combined_fig': fig_combined
    }


