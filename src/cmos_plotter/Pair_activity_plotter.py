import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def convert_elno_to_xy(elno):
    """
    Convert electrode number to x,y coordinates on the chip
    
    Parameters:
    elno (int): Electrode number
    
    Returns:
    tuple: (x, y) coordinates
    """
    chipWidth = 220
    x = int(elno/chipWidth)
    y = elno % chipWidth
    return x, y

def parse_filename(filename):
    """
    Extract metadata from the filename
    
    Parameters:
    filename (str): Filename to parse
    
    Returns:
    dict: Dictionary containing extracted metadata
    """
    # First try the pattern with time component
    pattern1 = r'ID(\d+)_N(\d+)_DIV(\d+)_DATE(\d+)_(\d+)_.*\.pkl'
    match = re.match(pattern1, filename)
    
    if match:
        chip_id = match.group(1)
        n = match.group(2)
        div = match.group(3)
        date = match.group(4)
        time = match.group(5)
        
        return {
            'chip_id': chip_id,
            'n': n,
            'div': div,
            'date': date,
            'time': time,
            'filename': filename
        }
    
    # Try alternative pattern without time component
    pattern2 = r'ID(\d+)_N(\d+)_DIV(\d+)_DATE(\d+)_.*\.pkl'
    match = re.match(pattern2, filename)
    
    if match:
        chip_id = match.group(1)
        n = match.group(2)
        div = match.group(3)
        date = match.group(4)
        
        return {
            'chip_id': chip_id,
            'n': n,
            'div': div,
            'date': date,
            'time': 'unknown',
            'filename': filename
        }
    
    # Try a more general pattern if still no match
    pattern3 = r'ID(\d+)_N(\d+)_DIV(\d+)_.*\.pkl'
    match = re.match(pattern3, filename)
    
    if match:
        chip_id = match.group(1)
        n = match.group(2)
        div = match.group(3)
        
        return {
            'chip_id': chip_id,
            'n': n,
            'div': div,
            'date': 'unknown',
            'time': 'unknown',
            'filename': filename
        }
    
    # If none of the patterns match
    print(f"Warning: Couldn't parse filename format: {filename}")
    return None

def calculate_distance(source_electrode, target_electrode):
    """
    Calculate the Euclidean distance between two electrodes
    
    Parameters:
    source_electrode (int): Source electrode number
    target_electrode (int): Target electrode number
    
    Returns:
    float: Euclidean distance between the electrodes
    """
    source_x, source_y = convert_elno_to_xy(source_electrode)
    target_x, target_y = convert_elno_to_xy(target_electrode)
    
    return np.sqrt((target_x - source_x)**2 + (target_y - source_y)**2)

def analyze_neural_data(directory='.'):
    """
    Analyze neural data from pickle files
    
    Parameters:
    directory (str): Directory containing pickle files
    
    Returns:
    dict: Analysis results
    """
    # Find all pickle files in the directory
    pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    print(f"Found {len(pickle_files)} pickle files")
    
    # Initialize data structures
    files_metadata = []
    pairs_per_chip_div = defaultdict(int)
    all_pairs_data = []
    
    # Process each file
    for filename in pickle_files:
        metadata = parse_filename(filename)
        if not metadata:
            print(f"Skipping file with unrecognized format: {filename}")
            continue
            
        files_metadata.append(metadata)
        
        # Load the pickle file
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                data = data['validated_results']
                print("Successfully loaded", filename) 
            # Count rows (pairs) in this file
            if isinstance(data, list):
                num_rows = len(data)
            elif isinstance(data, dict) and 'data' in data:
                num_rows = len(data['data'])
            elif isinstance(data, pd.DataFrame):
                num_rows = len(data)
            else:
                print(f"Unknown data format in {filename}, trying to get length directly")
                try:
                    num_rows = len(data)
                except:
                    print(f"Could not determine number of rows in {filename}")
                    num_rows = 0
            
            # Add to the count for this chip/DIV
            chip_div_key = f"{metadata['chip_id']}_{metadata['div']}"
            pairs_per_chip_div[chip_div_key] += num_rows
            
            # Process each row to extract lag and electrode position information
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and 'data' in data:
                rows = data['data']
            elif isinstance(data, pd.DataFrame):
                rows = data.to_dict('records')
            else:
                rows = []
                print(f"Could not extract rows from {filename}")
            
            for row in rows:
                # Extract the fields we need
                try:
                    source_electrode = row['source_electrode']
                    target_electrode = row['target_electrode']
                    lag = row['lag']
                    
                    # Calculate distance between electrodes
                    distance = calculate_distance(source_electrode, target_electrode)
                    
                    # Calculate electrode positions
                    source_x, source_y = convert_elno_to_xy(source_electrode)
                    target_x, target_y = convert_elno_to_xy(target_electrode)
                    
                    pair_data = {
                        'chip_id': metadata['chip_id'],
                        'n': metadata['n'],
                        'div': metadata['div'],
                        'source_electrode': source_electrode,
                        'target_electrode': target_electrode,
                        'source_x': source_x,
                        'source_y': source_y,
                        'target_x': target_x,
                        'target_y': target_y,
                        'distance': distance,
                        'lag': lag
                    }
                    
                    all_pairs_data.append(pair_data)
                except KeyError as e:
                    print(f"Missing key in data: {e} for file {filename}")
                except Exception as e:
                    print(f"Error processing row in {filename}: {e}")
                
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
    
    # Convert to DataFrame for easier analysis
    all_pairs_df = pd.DataFrame(all_pairs_data)
    
    # Prepare data for "pairs per DIV per chip" plot
    pairs_plot_data = []
    for key, count in pairs_per_chip_div.items():
        chip_id, div = key.split('_')
        pairs_plot_data.append({
            'chip_id': chip_id,
            'div': div,
            'count': count
        })
    
    pairs_plot_df = pd.DataFrame(pairs_plot_data)
    
    return {
        'files_metadata': files_metadata,
        'pairs_per_chip_div': pairs_per_chip_div,
        'pairs_plot_df': pairs_plot_df,
        'all_pairs_df': all_pairs_df
    }

def plot_pairs_per_div(save_path, pairs_plot_df):
    """
    Plot the mean number of rows (pairs) per DIV across all chips
    
    Parameters:
    pairs_plot_df (pd.DataFrame): DataFrame with pairs count data
    save_path (str): Directory path where the plot should be saved
    """
    plt.figure(figsize=(12, 8))
    
    # Convert DIV to numeric for proper ordering
    pairs_plot_df['div'] = pd.to_numeric(pairs_plot_df['div'])
    
    # Aggregate data by DIV
    div_summary = pairs_plot_df.groupby('div').agg(
        total_pairs=('count', 'sum'),
        mean_pairs=('count', 'mean'),
        std_pairs=('count', 'std'),
        num_chips=('chip_id', 'nunique')
    ).reset_index()
    
    # Plot the mean pairs per DIV
    ax = plt.subplot(111)
    bars = ax.bar(
        div_summary['div'], 
        div_summary['mean_pairs'],
        yerr=div_summary['std_pairs'],
        capsize=5,
        color='steelblue',
        width=0.7
    )
    
    # Add total pairs and number of chips as text above each bar
    for i, bar in enumerate(bars):
        total = div_summary.iloc[i]['total_pairs']
        num_chips = div_summary.iloc[i]['num_chips']
        
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + div_summary.iloc[i]['std_pairs'] + 5,
            f'Total: {total}',
            ha='center', va='bottom',
            fontsize=10
        )
        
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + div_summary.iloc[i]['std_pairs'] + 30,
            f'Chips: {num_chips}',
            ha='center', va='bottom',
            fontsize=10
        )
    
    plt.title('Mean Number of Source-Target Pairs per DIV Across All Chips', fontsize=16)
    plt.xlabel('DIV (Days In Vitro)', fontsize=14)
    plt.ylabel('Mean Number of Pairs (± Std Dev)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height()/2,
            f'{div_summary.iloc[i]["mean_pairs"]:.1f}',
            ha='center',
            fontsize=10,
            fontweight='bold',
            color='white'
        )
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_path, 'mean_pairs_per_div.png'), dpi=300)
    plt.savefig(os.path.join(save_path, 'mean_pairs_per_div.pdf'), format = 'pdf', dpi=300)

    plt.close()

def plot_lag_vs_position(save_path, all_pairs_df):
    """
    Plot lag vs electrode position, aggregated across all chips
    
    Parameters:
    all_pairs_df (pd.DataFrame): DataFrame with all pairs data
    save_path (str): Directory path where the plots should be saved
    """
    # Create a custom colormap from light to dark blue
    colors = [(0.8, 0.8, 1), (0, 0, 0.8)]  # Light blue to dark blue
    cmap = LinearSegmentedColormap.from_list('lag_cmap', colors, N=100)
    
    fig = plt.figure(figsize=(15, 16))
    
    # 1. Lag vs Distance Plot (aggregated across all chips)
    plt.subplot(3, 1, 1)
    
    # Group by distance and calculate mean lag
    dist_bins = np.arange(0, all_pairs_df['distance'].max() + 1, 1)
    all_pairs_df['distance_bin'] = pd.cut(all_pairs_df['distance'], bins=dist_bins, right=False)
    lag_by_distance = all_pairs_df.groupby('distance_bin')['lag'].agg(['mean', 'count', 'std']).reset_index()
    lag_by_distance['distance'] = lag_by_distance['distance_bin'].apply(lambda x: x.left)
    
    # Plot with scatter size based on count and color based on lag
    scatter = plt.scatter(
        lag_by_distance['distance'], 
        lag_by_distance['mean'], 
        s=lag_by_distance['count']/5,  # Adjust size based on count
        c=lag_by_distance['mean'],     # Color based on lag value
        cmap=cmap,
        alpha=0.7
    )
    
    # Add error bars
    plt.errorbar(
        lag_by_distance['distance'], 
        lag_by_distance['mean'], 
        yerr=lag_by_distance['std'], 
        fmt='none', 
        ecolor='gray', 
        alpha=0.3
    )
    
    plt.title('Average Lag vs Electrode Distance (All Chips)', fontsize=14)
    plt.xlabel('Distance Between Electrodes', fontsize=12)
    plt.ylabel('Average Lag (ms)', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.colorbar(scatter, label='Average Lag (ms)')
    
    # 2. Mean Lag by DIV (aggregated across all chips)
    plt.subplot(3, 1, 2)
    
    # Convert DIV to numeric and sort
    all_pairs_df['div'] = pd.to_numeric(all_pairs_df['div'])
    
    # Calculate mean lag per DIV
    div_lag_summary = all_pairs_df.groupby('div')['lag'].agg(['mean', 'std', 'count']).reset_index()
    
    # Bar plot with error bars
    bars = plt.bar(
        div_lag_summary['div'],
        div_lag_summary['mean'],
        yerr=div_lag_summary['std'],
        capsize=5,
        color='steelblue',
        alpha=0.7
    )
    
    # Add count labels
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + div_lag_summary.iloc[i]['std'] + 0.1,
            f'n={div_lag_summary.iloc[i]["count"]}',
            ha='center',
            fontsize=10
        )
    
    plt.title('Mean Lag by DIV (All Chips)', fontsize=14)
    plt.xlabel('DIV (Days In Vitro)', fontsize=12)
    plt.ylabel('Mean Lag (ms) ± Std Dev', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Lag vs DIV Box Plot (all chips but color-coded by DIV)
    plt.subplot(3, 1, 3)
    
    # Create a box plot by DIV only (not separated by chip)
    sns.boxplot(
        x='div', 
        y='lag', 
        data=all_pairs_df, 
        palette='Blues'
    )
    
    # Add swarm plot with small points
    sns.swarmplot(
        x='div', 
        y='lag', 
        data=all_pairs_df.sample(min(1000, len(all_pairs_df))),  # Sample to avoid overcrowding
        color='black',
        size=3,
        alpha=0.5
    )
    
    plt.title('Lag Distribution by DIV (All Chips)', fontsize=14)
    plt.xlabel('DIV (Days In Vitro)', fontsize=12)
    plt.ylabel('Lag (ms)', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_path, 'lag_vs_position.png'), dpi=300)
    plt.savefig(os.path.join(save_path, 'lag_vs_position.pdf'), format = 'pdf', dpi=300)

    plt.close()
    
    # Create additional plot showing lag trends over DIV
    plt.figure(figsize=(12, 6))
    
    # Group by DIV and distance bin
    all_pairs_df['distance_category'] = pd.cut(
        all_pairs_df['distance'],
        bins=[0, 5, 10, 20, 50, 100, float('inf')],
        labels=['0-5', '5-10', '10-20', '20-50', '50-100', '>100']
    )
    
    # Calculate mean lag for each DIV and distance category
    lag_by_div_dist = all_pairs_df.groupby(['div', 'distance_category'])['lag'].mean().reset_index()
    
    # Pivot for plotting
    lag_pivot = lag_by_div_dist.pivot(index='div', columns='distance_category', values='lag')
    
    # Plot
    ax = lag_pivot.plot(marker='o', linewidth=2)
    
    plt.title('Lag Trends by DIV for Different Electrode Distances', fontsize=14)
    plt.xlabel('DIV (Days In Vitro)', fontsize=12)
    plt.ylabel('Mean Lag (ms)', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(title='Distance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'lag_trends_by_div.png'), dpi=300)
    plt.savefig(os.path.join(save_path, 'lag_trends_by_div.pdf'), format = 'pdf', dpi=300)

    plt.close()

def plot_spatial_lag_map(save_path, all_pairs_df):
    """
    Create a spatial map of lag values with arrows showing connectivity direction from source to target
    
    Parameters:
    all_pairs_df (pd.DataFrame): DataFrame with all pairs data
    """
    # Get unique DIVs
    divs = sorted(all_pairs_df['div'].unique())
    n_divs = len(divs)
    
    # Create subplots for each DIV
    fig, axes = plt.subplots(1, n_divs, figsize=(6 * n_divs, 6))
    
    # Ensure axes is always an array
    if n_divs == 1:
        axes = np.array([axes])
    
    # Create a custom colormap for lag values
    cmap = plt.cm.Blues
    
    # Calculate the global chip dimensions based on all data
    max_x = max(all_pairs_df['source_x'].max(), all_pairs_df['target_x'].max())
    max_y = max(all_pairs_df['source_y'].max(), all_pairs_df['target_y'].max())
    
    # Find global min and max lag for consistent color scaling across subplots
    max_lag = all_pairs_df['lag'].max()
    min_lag = all_pairs_df['lag'].min()
    
    for i, div in enumerate(divs):
        ax = axes[i]
        
        # Filter data for this DIV (include all chips)
        filtered_df = all_pairs_df[all_pairs_df['div'] == div]
        
        if len(filtered_df) > 0:
            # Sample a subset of connections to display arrows
            # (too many arrows would clutter the visualization)
            max_arrows = 100  # Maximum number of arrows to display
            arrow_sample = filtered_df.sample(min(max_arrows, len(filtered_df)))
            
            # Create a background heatmap showing activity density
            activity_map = np.zeros((max_x + 1, max_y + 1))
            
            # Count activity at each position
            for _, row in filtered_df.iterrows():
                source_x, source_y = int(row['source_x']), int(row['source_y'])
                if source_x <= max_x and source_y <= max_y:
                    activity_map[source_x, source_y] += 1
            
            # Apply log scale for better visualization
            activity_map = np.log1p(activity_map)
            
            # Plot activity heatmap
            img = ax.imshow(
                activity_map.T,  # Transpose for correct orientation
                origin='lower',
                aspect='auto',
                cmap='YlOrRd',
                interpolation='nearest',
                alpha=0.6  # Semi-transparent to see arrows
            )
            
            # Add colorbar for activity
            cbar = plt.colorbar(img, ax=ax)
            cbar.set_label('Log(1+Activity)')
            
            # Plot arrows from source to target
            for _, row in arrow_sample.iterrows():
                source_x, source_y = row['source_x'], row['source_y']
                target_x, target_y = row['target_x'], row['target_y']
                lag = row['lag']
                
                # Calculate arrow properties based on lag
                arrow_width = 0.3
                arrow_head_width = 3 * arrow_width
                arrow_head_length = 3 * arrow_width
                
                # Normalize lag to determine color intensity (dark blue for high lag)
                color_intensity = (lag - min_lag) / (max_lag - min_lag) if max_lag > min_lag else 0.5
                arrow_color = plt.cm.Blues(0.5 + 0.5 * color_intensity)  # Map to second half of Blues colormap
                
                # Draw the arrow from source to target
                ax.arrow(
                    source_x, source_y,  # Start point (source)
                    target_x - source_x, target_y - source_y,  # Direction vector
                    width=arrow_width,
                    head_width=arrow_head_width,
                    head_length=arrow_head_length,
                    fc=arrow_color,  # Fill color based on lag
                    ec=arrow_color,  # Edge color
                    length_includes_head=True,
                    alpha=0.7
                )
            
            # Create a custom legend for lag values
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min_lag, vmax=max_lag))
            sm.set_array([])
            lag_cbar = plt.colorbar(sm, ax=ax, location='right', pad=0.1)
            lag_cbar.set_label('Lag (ms) - Arrow Color')
            
        # Set title and labels
        ax.set_title(f'DIV {div} (All Chips Combined)', fontsize=12)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
    
    plt.suptitle('Neural Connectivity: Arrows from Source to Target', fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_path,'connectivity_map_with_arrows.png'), dpi=300)
    plt.savefig(os.path.join(save_path,'connectivity_map_with_arrows.pdf'), format = 'pdf', dpi=300)

    plt.close()
    
    # Create an alternative visualization with quiver plot
    fig, axes = plt.subplots(1, n_divs, figsize=(6 * n_divs, 6))
    
    # Ensure axes is always an array
    if n_divs == 1:
        axes = np.array([axes])
    
    for i, div in enumerate(divs):
        ax = axes[i]
        
        # Filter data for this DIV
        filtered_df = all_pairs_df[all_pairs_df['div'] == div]
        
        if len(filtered_df) > 0:
            # Sample connections if too many
            max_arrows = 200  # Maximum number of arrows to display
            arrow_sample = filtered_df.sample(min(max_arrows, len(filtered_df)))
            
            # Extract positions and calculate direction vectors
            X = arrow_sample['source_x'].values
            Y = arrow_sample['source_y'].values
            U = arrow_sample['target_x'].values - X  # X direction
            V = arrow_sample['target_y'].values - Y  # Y direction
            C = arrow_sample['lag'].values  # Color by lag
            
            # Create quiver plot (arrows)
            quiver = ax.quiver(
                X, Y, U, V,
                C,
                cmap='Blues',
                scale=30,  # Adjust scale to make arrows visible
                width=0.003,
                headwidth=3,
                headlength=4,
                alpha=0.7
            )
            
            # Add colorbar
            cbar = plt.colorbar(quiver, ax=ax)
            cbar.set_label('Lag (ms)')
            
            # Create a background density map
            activity_map = np.zeros((max_x + 1, max_y + 1))
            for _, row in filtered_df.iterrows():
                source_x, source_y = int(row['source_x']), int(row['source_y'])
                if source_x <= max_x and source_y <= max_y:
                    activity_map[source_x, source_y] += 1
            
            # Apply log scale and normalize
            activity_map = np.log1p(activity_map)
            
            # Plot as a contour
            contour_levels = np.linspace(0, activity_map.max(), 10)
            contour = ax.contourf(
                np.arange(activity_map.shape[0]),
                np.arange(activity_map.shape[1]),
                activity_map.T,
                levels=contour_levels,
                cmap='YlOrRd',
                alpha=0.3
            )
        
        # Set title and labels
        ax.set_title(f'DIV {div} Signal Flow Direction', fontsize=12)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_aspect('equal')
    
    plt.suptitle('Neural Signal Flow: Source to Target', fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_path,'signal_flow_quiver.png'), dpi=300)
    plt.savefig(os.path.join(save_path,'signal_flow_quiver.pdf'), format = 'pdf', dpi=300)
    
    plt.close()

def plot_spatial_lag_map_per_chip(save_path, all_pairs_df):
    """
    Create a spatial map of lag values with arrows showing connectivity direction
    from source to target, separated by chip and DIV
    
    Parameters:
    save_path (str): Path to save the output figures
    all_pairs_df (pd.DataFrame): DataFrame with all pairs data including 'chip_id' column
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Get unique DIVs and chips
    divs = sorted(all_pairs_df['div'].unique())
    chips = sorted(all_pairs_df['chip_id'].unique())
    
    # Calculate the global chip dimensions based on all data
    max_x = max(all_pairs_df['source_x'].max(), all_pairs_df['target_x'].max())
    max_y = max(all_pairs_df['source_y'].max(), all_pairs_df['target_y'].max())
    
    # Find global min and max lag for consistent color scaling across all plots
    max_lag = all_pairs_df['lag'].max()
    min_lag = all_pairs_df['lag'].min()
    
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Create a subdirectory for the per-chip figures
    per_chip_dir = os.path.join(save_path, 'per_chip_per_div')
    os.makedirs(per_chip_dir, exist_ok=True)
    
    # Loop through each DIV and chip combination
    for div in divs:
        for chip in chips:
            # Filter data for this DIV and chip
            filtered_df = all_pairs_df[(all_pairs_df['div'] == div) & 
                                       (all_pairs_df['chip_id'] == chip)]
            
            if len(filtered_df) > 0:
                # Create a figure for this chip and DIV
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Sample a subset of connections to display arrows
                max_arrows = 200  # Maximum number of arrows to display
                arrow_sample = filtered_df.sample(min(max_arrows, len(filtered_df)))
                
                # Create a background heatmap showing activity density
                activity_map = np.zeros((max_x + 1, max_y + 1))
                
                # Count activity at each position
                for _, row in filtered_df.iterrows():
                    source_x, source_y = int(row['source_x']), int(row['source_y'])
                    if source_x <= max_x and source_y <= max_y:
                        activity_map[source_x, source_y] += 1
                
                # Apply log scale for better visualization
                activity_map = np.log1p(activity_map)
                
                # Plot activity heatmap
                img = ax.imshow(
                    activity_map.T,  # Transpose for correct orientation
                    origin='lower',
                    aspect='auto',
                    cmap='YlOrRd',
                    interpolation='nearest',
                    alpha=0.6  # Semi-transparent to see arrows
                )
                
                # Add colorbar for activity
                cbar = plt.colorbar(img, ax=ax)
                cbar.set_label('Log(1+Activity)')
                
                # Plot arrows from source to target
                for _, row in arrow_sample.iterrows():
                    source_x, source_y = row['source_x'], row['source_y']
                    target_x, target_y = row['target_x'], row['target_y']
                    lag = row['lag']
                    
                    # Calculate arrow properties based on lag
                    arrow_width = 0.3
                    arrow_head_width = 3 * arrow_width
                    arrow_head_length = 3 * arrow_width
                    
                    # Normalize lag to determine color intensity (dark blue for high lag)
                    color_intensity = (lag - min_lag) / (max_lag - min_lag) if max_lag > min_lag else 0.5
                    arrow_color = plt.cm.Blues(0.5 + 0.5 * color_intensity)  # Map to second half of Blues colormap
                    
                    # Draw the arrow from source to target
                    ax.arrow(
                        source_x, source_y,  # Start point (source)
                        target_x - source_x, target_y - source_y,  # Direction vector
                        width=arrow_width,
                        head_width=arrow_head_width,
                        head_length=arrow_head_length,
                        fc=arrow_color,  # Fill color based on lag
                        ec=arrow_color,  # Edge color
                        length_includes_head=True,
                        alpha=0.7
                    )
                
                # Create a custom legend for lag values
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min_lag, vmax=max_lag))
                sm.set_array([])
                lag_cbar = plt.colorbar(sm, ax=ax, location='right', pad=0.1)
                lag_cbar.set_label('Lag (ms) - Arrow Color')
                
                # Set title and labels
                ax.set_title(f'DIV {div}, Chip {chip} - Neural Connectivity', fontsize=14)
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                
                # Save figure with detailed naming
                filename = f'connectivity_map_div{div}_chip{chip}'
                plt.tight_layout()
                plt.savefig(os.path.join(per_chip_dir, f'{filename}.png'), dpi=300)
                plt.savefig(os.path.join(per_chip_dir, f'{filename}.pdf'), format='pdf', dpi=300)
                plt.close()
                
                # Create an alternative visualization with quiver plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Extract positions and calculate direction vectors
                X = arrow_sample['source_x'].values
                Y = arrow_sample['source_y'].values
                U = arrow_sample['target_x'].values - X  # X direction
                V = arrow_sample['target_y'].values - Y  # Y direction
                C = arrow_sample['lag'].values  # Color by lag
                
                # Create quiver plot (arrows)
                quiver = ax.quiver(
                    X, Y, U, V,
                    C,
                    cmap='Blues',
                    scale=30,  # Adjust scale to make arrows visible
                    width=0.003,
                    headwidth=3,
                    headlength=4,
                    alpha=0.7
                )
                
                # Add colorbar
                cbar = plt.colorbar(quiver, ax=ax)
                cbar.set_label('Lag (ms)')
                
                # Create a background density map
                contour_levels = np.linspace(0, activity_map.max(), 10)
                contour = ax.contourf(
                    np.arange(activity_map.shape[0]),
                    np.arange(activity_map.shape[1]),
                    activity_map.T,
                    levels=contour_levels,
                    cmap='YlOrRd',
                    alpha=0.3
                )
                
                # Set title and labels
                ax.set_title(f'DIV {div}, Chip {chip} - Signal Flow Direction', fontsize=14)
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_aspect('equal')
                
                # Save quiver plot
                filename = f'quiver_flow_div{div}_chip{chip}'
                plt.tight_layout()
                plt.savefig(os.path.join(per_chip_dir, f'{filename}.png'), dpi=300)
                plt.savefig(os.path.join(per_chip_dir, f'{filename}.pdf'), format='pdf', dpi=300)
                plt.close()
    
    # Also create the original combined plots for reference
    
    # Function to create combined plots per DIV (same as original, but moved inside to reuse variables)
    def create_combined_div_plots():
        # Create subplots for each DIV (arrow plot)
        n_divs = len(divs)
        fig, axes = plt.subplots(1, n_divs, figsize=(6 * n_divs, 6))
        
        # Ensure axes is always an array
        if n_divs == 1:
            axes = np.array([axes])
        
        for i, div in enumerate(divs):
            ax = axes[i]
            
            # Filter data for this DIV (include all chips)
            filtered_df = all_pairs_df[all_pairs_df['div'] == div]
            
            if len(filtered_df) > 0:
                # Same plotting code as before...
                max_arrows = 100
                arrow_sample = filtered_df.sample(min(max_arrows, len(filtered_df)))
                
                activity_map = np.zeros((max_x + 1, max_y + 1))
                for _, row in filtered_df.iterrows():
                    source_x, source_y = int(row['source_x']), int(row['source_y'])
                    if source_x <= max_x and source_y <= max_y:
                        activity_map[source_x, source_y] += 1
                
                activity_map = np.log1p(activity_map)
                
                img = ax.imshow(
                    activity_map.T,
                    origin='lower',
                    aspect='auto',
                    cmap='YlOrRd',
                    interpolation='nearest',
                    alpha=0.6
                )
                
                cbar = plt.colorbar(img, ax=ax)
                cbar.set_label('Log(1+Activity)')
                
                for _, row in arrow_sample.iterrows():
                    source_x, source_y = row['source_x'], row['source_y']
                    target_x, target_y = row['target_x'], row['target_y']
                    lag = row['lag']
                    
                    arrow_width = 0.3
                    arrow_head_width = 3 * arrow_width
                    arrow_head_length = 3 * arrow_width
                    
                    color_intensity = (lag - min_lag) / (max_lag - min_lag) if max_lag > min_lag else 0.5
                    arrow_color = plt.cm.Blues(0.5 + 0.5 * color_intensity)
                    
                    ax.arrow(
                        source_x, source_y,
                        target_x - source_x, target_y - source_y,
                        width=arrow_width,
                        head_width=arrow_head_width,
                        head_length=arrow_head_length,
                        fc=arrow_color,
                        ec=arrow_color,
                        length_includes_head=True,
                        alpha=0.7
                    )
                
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min_lag, vmax=max_lag))
                sm.set_array([])
                lag_cbar = plt.colorbar(sm, ax=ax, location='right', pad=0.1)
                lag_cbar.set_label('Lag (ms) - Arrow Color')
            
            ax.set_title(f'DIV {div} (All Chips Combined)', fontsize=12)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
        
        plt.suptitle('Neural Connectivity: Arrows from Source to Target', fontsize=16, y=1.05)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, 'connectivity_map_with_arrows.png'), dpi=300)
        plt.savefig(os.path.join(save_path, 'connectivity_map_with_arrows.pdf'), format='pdf', dpi=300)
        plt.close()
        
        # Create quiver plots for each DIV
        fig, axes = plt.subplots(1, n_divs, figsize=(6 * n_divs, 6))
        
        # Ensure axes is always an array
        if n_divs == 1:
            axes = np.array([axes])
        
        for i, div in enumerate(divs):
            ax = axes[i]
            
            # Filter data for this DIV
            filtered_df = all_pairs_df[all_pairs_df['div'] == div]
            
            if len(filtered_df) > 0:
                max_arrows = 200
                arrow_sample = filtered_df.sample(min(max_arrows, len(filtered_df)))
                
                X = arrow_sample['source_x'].values
                Y = arrow_sample['source_y'].values
                U = arrow_sample['target_x'].values - X
                V = arrow_sample['target_y'].values - Y
                C = arrow_sample['lag'].values
                
                quiver = ax.quiver(
                    X, Y, U, V,
                    C,
                    cmap='Blues',
                    scale=30,
                    width=0.003,
                    headwidth=3,
                    headlength=4,
                    alpha=0.7
                )
                
                cbar = plt.colorbar(quiver, ax=ax)
                cbar.set_label('Lag (ms)')
                
                activity_map = np.zeros((max_x + 1, max_y + 1))
                for _, row in filtered_df.iterrows():
                    source_x, source_y = int(row['source_x']), int(row['source_y'])
                    if source_x <= max_x and source_y <= max_y:
                        activity_map[source_x, source_y] += 1
                
                activity_map = np.log1p(activity_map)
                
                contour_levels = np.linspace(0, activity_map.max(), 10)
                contour = ax.contourf(
                    np.arange(activity_map.shape[0]),
                    np.arange(activity_map.shape[1]),
                    activity_map.T,
                    levels=contour_levels,
                    cmap='YlOrRd',
                    alpha=0.3
                )
            
            ax.set_title(f'DIV {div} Signal Flow Direction', fontsize=12)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_aspect('equal')
        
        plt.suptitle('Neural Signal Flow: Source to Target', fontsize=16, y=1.05)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, 'signal_flow_quiver.png'), dpi=300)
        plt.savefig(os.path.join(save_path, 'signal_flow_quiver.pdf'), format='pdf', dpi=300)
        plt.close()
    
    # Create the combined plots as well
    create_combined_div_plots()
    
    # Create a summary figure showing one example chip for each DIV
    if len(chips) > 0:
        fig, axes = plt.subplots(1, len(divs), figsize=(6 * len(divs), 6))
        
        # Ensure axes is always an array
        if len(divs) == 1:
            axes = np.array([axes])
        
        for i, div in enumerate(divs):
            ax = axes[i]
            
            # Get the first chip with data for this DIV
            div_chips = [chip for chip in chips if len(all_pairs_df[(all_pairs_df['div'] == div) & 
                                                                    (all_pairs_df['chip_id'] == chip)]) > 0]
            
            if div_chips:
                example_chip = div_chips[0]
                filtered_df = all_pairs_df[(all_pairs_df['div'] == div) & 
                                           (all_pairs_df['chip_id'] == example_chip)]
                
                # Sample arrows
                max_arrows = 100
                arrow_sample = filtered_df.sample(min(max_arrows, len(filtered_df)))
                
                # Extract positions and calculate direction vectors
                X = arrow_sample['source_x'].values
                Y = arrow_sample['source_y'].values
                U = arrow_sample['target_x'].values - X
                V = arrow_sample['target_y'].values - Y
                C = arrow_sample['lag'].values
                
                # Create quiver plot (arrows)
                quiver = ax.quiver(
                    X, Y, U, V,
                    C,
                    cmap='Blues',
                    scale=30,
                    width=0.003,
                    headwidth=3,
                    headlength=4,
                    alpha=0.7
                )
                
                # Add colorbar
                cbar = plt.colorbar(quiver, ax=ax)
                cbar.set_label('Lag (ms)')
                
                ax.set_title(f'DIV {div}, Chip {example_chip}', fontsize=12)
            else:
                ax.text(0.5, 0.5, f'No data for DIV {div}', 
                        horizontalalignment='center', verticalalignment='center')
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_aspect('equal')
        
        plt.suptitle('Example Chips: Neural Signal Flow', fontsize=16, y=1.05)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, 'example_chips_signal_flow.png'), dpi=300)
        plt.savefig(os.path.join(save_path, 'example_chips_signal_flow.pdf'), format='pdf', dpi=300)
        plt.close()
    
    print(f"Saved {len(divs) * len(chips)} individual chip-DIV plots to {per_chip_dir}")
    print(f"Saved combined plots to {save_path}")

def calculate_additional_metrics(all_pairs_df):
    """
    Calculate additional metrics by DIV across all chips
    
    Parameters:
    all_pairs_df (pd.DataFrame): DataFrame with all pairs data
    
    Returns:
    pd.DataFrame: Summary statistics by DIV
    """
    # Convert DIV to numeric for proper sorting
    all_pairs_df['div'] = pd.to_numeric(all_pairs_df['div'])
    
    # Calculate metrics by DIV
    div_metrics = all_pairs_df.groupby('div').agg(
        mean_lag=('lag', 'mean'),
        median_lag=('lag', 'median'),
        std_lag=('lag', 'std'),
        min_lag=('lag', 'min'),
        max_lag=('lag', 'max'),
        total_pairs=('lag', 'count'),
        num_chips=('chip_id', 'nunique')
    ).reset_index()
    
    # Calculate mean distance
    distance_by_div = all_pairs_df.groupby('div')['distance'].mean().reset_index()
    div_metrics = div_metrics.merge(distance_by_div, on='div')
    div_metrics.rename(columns={'distance': 'mean_distance'}, inplace=True)
    
    return div_metrics

def plot_metrics_by_div(save_path,div_metrics):
    """
    Plot various metrics by DIV
    
    Parameters:
    div_metrics (pd.DataFrame): DataFrame with metrics by DIV
    """
    plt.figure(figsize=(14, 10))
    
    # 1. Mean and median lag by DIV
    plt.subplot(2, 2, 1)
    plt.plot(div_metrics['div'], div_metrics['mean_lag'], 'o-', label='Mean Lag', color='blue')
    plt.plot(div_metrics['div'], div_metrics['median_lag'], 's--', label='Median Lag', color='darkblue')
    plt.fill_between(
        div_metrics['div'],
        div_metrics['mean_lag'] - div_metrics['std_lag'],
        div_metrics['mean_lag'] + div_metrics['std_lag'],
        alpha=0.2,
        color='blue'
    )
    plt.title('Lag by DIV (All Chips)', fontsize=12)
    plt.xlabel('DIV (Days In Vitro)', fontsize=11)
    plt.ylabel('Lag (ms)', fontsize=11)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    # 2. Total pairs by DIV
    plt.subplot(2, 2, 2)
    bars = plt.bar(div_metrics['div'], div_metrics['total_pairs'], color='steelblue')
    plt.title('Total Source-Target Pairs by DIV', fontsize=12)
    plt.xlabel('DIV (Days In Vitro)', fontsize=11)
    plt.ylabel('Number of Pairs', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # 3. Mean distance by DIV
    plt.subplot(2, 2, 3)
    plt.plot(div_metrics['div'], div_metrics['mean_distance'], 'o-', color='green')
    plt.title('Mean Electrode Distance by DIV', fontsize=12)
    plt.xlabel('DIV (Days In Vitro)', fontsize=11)
    plt.ylabel('Mean Distance', fontsize=11)
    plt.grid(linestyle='--', alpha=0.7)
    
    # 4. Lag min/max range by DIV
    plt.subplot(2, 2, 4)
    for i, row in div_metrics.iterrows():
        plt.vlines(
            x=row['div'],
            ymin=row['min_lag'],
            ymax=row['max_lag'],
            color='gray',
            alpha=0.5
        )
    plt.plot(div_metrics['div'], div_metrics['min_lag'], 'v', label='Min Lag', color='green')
    plt.plot(div_metrics['div'], div_metrics['max_lag'], '^', label='Max Lag', color='red')
    plt.plot(div_metrics['div'], div_metrics['mean_lag'], 'o', label='Mean Lag', color='blue')
    
    plt.title('Lag Range by DIV', fontsize=12)
    plt.xlabel('DIV (Days In Vitro)', fontsize=11)
    plt.ylabel('Lag (ms)', fontsize=11)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'metrics_by_div.png'), dpi=300)
    plt.savefig(os.path.join(save_path,'metrics_by_div.pdf'), format = 'pdf', dpi=300)

    plt.close()

def main():
    """
    Main function to run the analysis and generate plots
    """
    print("Starting neural data analysis...")
    
    # Analyze the data
    results = analyze_neural_data()
    
    # Print summary information
    print("\nSummary:")
    print(f"Number of files processed: {len(results['files_metadata'])}")
    print(f"Number of chips: {len(set(item['chip_id'] for item in results['files_metadata']))}")
    print(f"DIVs analyzed: {sorted(set(item['div'] for item in results['files_metadata']))}")
    
    if len(results['all_pairs_df']) > 0:
        print(f"Total source-target pairs: {len(results['all_pairs_df'])}")
        print(f"Lag range: {results['all_pairs_df']['lag'].min():.2f} - {results['all_pairs_df']['lag'].max():.2f} ms")
        print(f"Average lag: {results['all_pairs_df']['lag'].mean():.2f} ms")
    else:
        print("No data pairs were extracted.")
    
    # Generate and save plots
    if len(results['pairs_plot_df']) > 0:
        print("\nGenerating plots...")
        plot_pairs_per_div(results['pairs_plot_df'])
        print("- Saved mean_pairs_per_div.png")
    
    if len(results['all_pairs_df']) > 0:
        # Plot lag vs position with new visualizations
        plot_lag_vs_position(results['all_pairs_df'])
        print("- Saved lag_vs_position.png and lag_trends_by_div.png")
        
        # Plot spatial connectivity with arrows from source to target
        plot_spatial_lag_map(results['all_pairs_df'])
        print("- Saved connectivity_map_with_arrows.png and signal_flow_quiver.png")
        
        # Calculate and plot additional metrics
        div_metrics = calculate_additional_metrics(results['all_pairs_df'])
        plot_metrics_by_div(div_metrics)
        print("- Saved metrics_by_div.png")
        
        # Export summary metrics to CSV
        div_metrics.to_csv('div_metrics_summary.csv', index=False)
        print("- Saved div_metrics_summary.csv")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()