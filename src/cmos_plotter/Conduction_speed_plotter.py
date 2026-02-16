import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import spikeinterface as si
import spikeinterface.extractors as se
import seaborn as sns
import os
import re


def load_and_extract_templates(template_path):
    """
    Load templates from sorted data
    
    template_extractor = si.extract_templates(
        recording=recording,
        sorting=sorting,
        max_spikes_per_unit=100
    )
    templates = template_extractor.get_templates()
    """
    templates = np.load(os.path.join(template_path, 'templates_average.npy'))
    return templates


def extract_metadata_from_filename(filename):
    """Extract chip_id, area, and DIV from filename."""
    # Pattern to extract ID, area (N#), and DIV
    pattern = r"ID(\d+)_N(\d+)_DIV(\d+)_"
    match = re.search(pattern, filename)
    
    
    if match:
        chip_id = match.group(1)
        area = match.group(2)
        div = int(match.group(3))
        return chip_id, area, div
    else:
        pattern = r"ID(\d+)_(\d+)_DIV(\d+)_"
        match = re.search(pattern, filename)
        if match:
            chip_id = match.group(1)
            area = match.group(2)
            div = int(match.group(3))
            return chip_id, area, div
        else:
            # If no match, return None for all
            print(f"Could not extract metadata from filename: {filename}")
            return None, None, None

def estimate_conduction_speed(template, probe_locations, sampling_rate):
    """
    Estimate conduction speed from a template
    """
    # Find peak/trough times for each channel
    peak_times = np.zeros(template.shape[1])
    peak_amplitudes = np.zeros(template.shape[1])
    
    for chan in range(template.shape[1]):
        # Find peaks and their properties
        peaks, properties = find_peaks(np.abs(template[:, chan]), height=np.std(template[:, chan]))
        if len(peaks) > 0:
            # Get the highest peak
            max_peak_idx = np.argmax(properties['peak_heights'])
            peak_times[chan] = peaks[max_peak_idx]
            peak_amplitudes[chan] = properties['peak_heights'][max_peak_idx]
    
    # Only consider channels with significant peaks
    amplitude_threshold = np.median(peak_amplitudes) * 0.3 # 30% of median amplitude
    valid_channels = peak_amplitudes > amplitude_threshold
    
    if np.sum(valid_channels) < 2:
        return None, None
    
    # Convert peak times to milliseconds
    peak_times = peak_times[valid_channels] / sampling_rate * 1000  # ms
    valid_locations = probe_locations[valid_channels]
    
    # Calculate distances between valid electrodes (in micrometers)
    distances = np.zeros((len(valid_locations), len(valid_locations)))
    for i in range(len(valid_locations)):
        for j in range(len(valid_locations)):
            distances[i,j] = np.sqrt(np.sum((valid_locations[i] - valid_locations[j])**2))
    
    # Calculate speeds between each pair of channels
    speeds = []
    directions = []
    for i in range(len(valid_locations)):
        for j in range(i+1, len(valid_locations)):
            time_diff = peak_times[j] - peak_times[i]
            if abs(time_diff) > 0:  # Avoid division by zero
                speed = (distances[i,j] / abs(time_diff))  # μm/ms = mm/s
                speeds.append(speed)
                # Calculate direction vector
                direction = valid_locations[j] - valid_locations[i]
                directions.append(direction / np.linalg.norm(direction))
    
    if len(speeds) == 0:
        return None, None
        
    return np.median(speeds), np.mean(directions, axis=0)

def analyze_conduction_speeds(templates, probe_locations, sampling_rate, unit_ids):
    """
    Analyze conduction speeds for all units
    """
    results = []
    for i, unit_id in enumerate(unit_ids):
        speed, direction = estimate_conduction_speed(
            templates[i], 
            probe_locations, 
            sampling_rate
        )
        if speed is not None:
            results.append({
                'unit': unit_id,
                'speed': speed/1000,   # Convert to m/s
                'direction': direction
            })
            #print(f"Unit {unit}: {speed/1000:.2f} m/s")
    return results

def visualize_speeds_and_directions(save_path, filename, results, templates, probe_locations, figsize=(15, 6), fontsize=14):
    """
    Visualize conduction speeds and directions with enhanced styling
    
    Parameters:
    - results: list of dictionaries containing 'speed' and 'direction' for each unit
    - probe_locations: numpy array of shape (n_channels, 2) with x,y coordinates
    - figsize: tuple for figure size
    - fontsize: base font size for plots
    """
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize + 2,
        'axes.titlesize': fontsize + 4,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })

    for result in results:
        result['speed'] = result.pop('speed_ms-1')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot speed distribution
    speeds = [r['speed'] for r in results]
    sns.histplot(
        data=speeds,
        bins=20,
        ax=ax1,
        color='royalblue',
        alpha=0.7,
        edgecolor='black'
    )
    ax1.set_xlabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
    ax1.set_ylabel('Count', fontsize=fontsize + 2)
    ax1.set_title('Distribution of Conduction Speeds', fontsize=fontsize + 4, pad=20)
    
    # Add mean and median lines
    mean_speed = np.mean(speeds)
    median_speed = np.median(speeds)
    ax1.axvline(mean_speed, color='red', linestyle='--', label=f'Mean: {mean_speed:.1f} mm/s')
    ax1.axvline(median_speed, color='green', linestyle='--', label=f'Median: {median_speed:.1f} mm/s')
    ax1.legend(fontsize=fontsize)
    
    # Plot probe layout with directions
    electrode_counts = np.zeros(len(probe_locations))
    for r in results:
        direction = r['direction']
        
        # Find the electrode with maximum amplitude for this unit
        # (You'll need to pass the template amplitudes in the results)
        # Or alternatively, use the electrode closest to the actual signal origin
        
        # For each unit, find peak amplitudes across electrodes
        template = templates[r['unit']]  # You'll need to pass templates to this function
        peak_amplitudes = np.max(np.abs(template), axis=0)
        origin_electrode = np.argmax(peak_amplitudes)
        center = probe_locations[origin_electrode]
        
        # Plot direction arrows from the actual source
        ax2.arrow(center[0], center[1],
                direction[0]*50, direction[1]*50,
                head_width=5, head_length=10,
                fc='red', ec='red', alpha=0.3)
        
        # Count signal at and around the origin electrode
        for i, loc in enumerate(probe_locations):
            if np.linalg.norm(loc - center) < 100:  # within 100μm
                electrode_counts[i] += 1

    # Plot electrodes with color intensity based on activity
    scatter = ax2.scatter(probe_locations[:, 0], probe_locations[:, 1],
                        c=electrode_counts, cmap='viridis',
                        s=50, alpha=0.8, edgecolor='black')
    plt.colorbar(scatter, ax=ax2, label='Signal Count')
    
    ax2.set_aspect('equal')
    ax2.set_title('Propagation Directions\nand Electrode Activity', fontsize=fontsize + 4, pad=20)
    ax2.set_xlabel('X Position (μm)', fontsize=fontsize + 2)
    ax2.set_ylabel('Y Position (μm)', fontsize=fontsize + 2)
    
    # Add statistical information
    stats_text = (f'Total Units: {len(speeds)}\n'
                 f'Mean Speed: {mean_speed:.1f} m/s\n'
                 f'Median Speed: {median_speed:.1f} m/s\n'
                 f'Std Dev: {np.std(speeds):.1f} m/s')
    ax2.text(0.02, 0.98, stats_text,
             transform=ax2.transAxes,
             fontsize=fontsize - 2,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{filename}_speeds_and_directions.png"))
    plt.savefig(os.path.join(save_path, f"{filename}_speeds_and_directions.pdf"), format='pdf', dpi= 300)
    return fig, (ax1, ax2)


def visualize_unit_propagation(save_path, filename, results, templates, probe_locations, 
                              sampling_rate, figsize=(10, 8), fontsize=12, unit_ids=None):
    """
    Visualize propagation direction for each unit across electrodes
    
    Parameters:
    - save_path: directory to save the figures
    - filename: base filename for saved plots
    - results: list of dictionaries containing 'speed' and 'direction' for each unit
    - templates: templates for each unit (shape: time x channels)
    - probe_locations: numpy array of shape (n_channels, 2) with x,y coordinates
    - sampling_rate: sampling rate in Hz
    - figsize: tuple for figure size
    - fontsize: base font size for plots
    - unit_ids: optional list of specific unit IDs to plot (default: all units)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.signal import find_peaks
    
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize + 2,
        'axes.titlesize': fontsize + 4,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })
    
    # Create custom colormap for temporal progression
    colors = [(0, 0, 0.8), (0, 0.8, 0), (0.8, 0, 0)]  # Blue -> Green -> Red
    cmap_name = 'temporal_progression'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # Process unit IDs
    if unit_ids is None:
        unit_ids = [r['unit'] for r in results]
    
    # Process results to map unit IDs to their data
    unit_data = {}
    for result in results:
        unit_id = result['unit']
        if 'speed_ms-1' in result:
            result['speed'] = result.pop('speed_ms-1')
        unit_data[unit_id] = result
    
    figures = []
    
    # Create a figure for each unit
    for unit_id in unit_ids:
        if unit_id not in unit_data:
            print(f"Unit {unit_id} not found in results, skipping.")
            continue
            
        result = unit_data[unit_id]
        template = templates[unit_id]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                      gridspec_kw={'width_ratios': [1, 1.2]})
        
        # Find peak/trough times for each channel
        peak_times = np.zeros(template.shape[1])
        peak_amplitudes = np.zeros(template.shape[1])
       
        for chan in range(template.shape[1]):
            # Find peaks and their properties
            peaks, properties = find_peaks(np.abs(template[:, chan]), 
                                          height=np.std(template[:, chan]))
            if len(peaks) > 0:
                # Get the highest peak
                max_peak_idx = np.argmax(properties['peak_heights'])
                peak_times[chan] = peaks[max_peak_idx]
                peak_amplitudes[chan] = properties['peak_heights'][max_peak_idx]
        
        # Only consider channels with significant peaks
        amplitude_threshold = np.median(peak_amplitudes) * 0.3
        valid_channels = peak_amplitudes > amplitude_threshold
        valid_peak_times = peak_times[valid_channels]
        valid_locations = probe_locations[valid_channels]
        valid_amplitudes = peak_amplitudes[valid_channels]
        
        # Normalize peak times to [0, 1] for color mapping
        if len(valid_peak_times) > 0:
            min_time = np.min(valid_peak_times)
            max_time = np.max(valid_peak_times)
            time_range = max_time - min_time
            
            if time_range > 0:
                normalized_times = (valid_peak_times - min_time) / time_range
            else:
                normalized_times = np.zeros_like(valid_peak_times)
        else:
            normalized_times = np.array([])
        
        # Plot template waveforms
        ax1.set_title(f"Unit {unit_id} Template", fontsize=fontsize+2)
        time_axis = np.arange(template.shape[0]) / sampling_rate * 1000  # Convert to ms
        
        # Plot a subset of channels for clarity (top 10 by amplitude)
        if valid_channels.sum() > 10:
            top_channels = np.argsort(peak_amplitudes)[-10:]
            top_channels = top_channels[peak_amplitudes[top_channels] > amplitude_threshold]
        else:
            top_channels = np.where(valid_channels)[0]
        
        # Plot each channel with color based on peak time
        for i, chan in enumerate(top_channels):
            if peak_amplitudes[chan] > amplitude_threshold:
                norm_time = (peak_times[chan] - min_time) / time_range if time_range > 0 else 0
                ax1.plot(time_axis, template[:, chan], 
                        color=cm(norm_time), 
                        alpha=0.8,
                        linewidth=1.5)
        
        ax1.set_xlabel("Time (ms)", fontsize=fontsize)
        ax1.set_ylabel("Amplitude", fontsize=fontsize)
        ax1.grid(True, alpha=0.3)
        
        # Plot electrode map with propagation
        ax2.set_title(f"Unit {unit_id} Propagation", fontsize=fontsize+2)
        
        # Plot all electrode positions
        ax2.scatter(probe_locations[:, 0], probe_locations[:, 1], 
                  color='gray', alpha=0.3, s=30, zorder=1, marker = 's')
        
        # Plot valid electrodes with color based on activation time
        scatter = ax2.scatter(valid_locations[:, 0], valid_locations[:, 1],
                           c=normalized_times, cmap=cm,
                           s=valid_amplitudes/np.max(valid_amplitudes)*100 + 30,
                           alpha=0.8, edgecolor='black', zorder=2)
        
        # Add colorbar for time
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Relative Activation Time', fontsize=fontsize)
        
        # Find the electrode with maximum amplitude
        if len(valid_amplitudes) > 0:
            max_amp_idx = np.argmax(valid_amplitudes)
            center = valid_locations[max_amp_idx]
            
            # Draw propagation direction arrow
            direction = result['direction']
            speed = result['speed']
            
            # Scale arrow length based on speed
            arrow_scale = 50  # Adjust as needed
            ax2.arrow(center[0], center[1],
                    direction[0] * arrow_scale, direction[1] * arrow_scale,
                    head_width=10, head_length=20,
                    fc='red', ec='black', alpha=0.8,
                    linewidth=2, zorder=3)
            
            # Add sequential arrows showing the propagation path
            sorted_by_time = np.argsort(normalized_times)
            for i in range(len(sorted_by_time)-1):
                idx1 = sorted_by_time[i]
                idx2 = sorted_by_time[i+1]
                
                # Only draw arrows between sequential points
                ax2.annotate("",
                           xy=(valid_locations[idx2, 0], valid_locations[idx2, 1]),
                           xytext=(valid_locations[idx1, 0], valid_locations[idx1, 1]),
                           arrowprops=dict(arrowstyle="->", color=cm(0.5), 
                                          alpha=0.6, linewidth=1.5),
                           zorder=2)
        
        # Add statistics text
        stats_text = (f"Speed: {speed:.2f} m/s\n"
                     f"Direction: [{direction[0]:.2f}, {direction[1]:.2f}]\n"
                     f"Active channels: {valid_channels.sum()}")
        
        ax2.text(0.02, 0.98, stats_text,
               transform=ax2.transAxes,
               fontsize=fontsize,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel("X Position (μm)", fontsize=fontsize)
        ax2.set_ylabel("Y Position (μm)", fontsize=fontsize)
        ax2.set_aspect('equal')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(save_path, f"{filename}_unit{unit_id}_propagation.png"))
        plt.savefig(os.path.join(save_path, f"{filename}_unit{unit_id}_propagation.pdf"), 
                   format='pdf', dpi=300)
        
        figures.append((fig, (ax1, ax2)))
    
    # Create a summary figure with all unit directions
    fig_summary, ax_summary = plt.subplots(figsize=(12, 10))
    
    # Plot all electrode positions
    ax_summary.scatter(probe_locations[:, 0], probe_locations[:, 1], 
                    color='gray', alpha=0.3, s=30)
    
    # Define colors for different units
    unit_colors = plt.cm.tab20(np.linspace(0, 1, len(unit_ids)))
    
    # For each unit, draw its propagation direction
    for i, unit_id in enumerate(unit_ids):
        if unit_id not in unit_data:
            continue
            
        result = unit_data[unit_id]
        template = templates[unit_id]
        
        # Find peak amplitudes across electrodes
        peak_amplitudes = np.max(np.abs(template), axis=0)
        origin_electrode = np.argmax(peak_amplitudes)
        center = probe_locations[origin_electrode]
        
        direction = result['direction']
        speed = result['speed']
        
        # Scale arrow length based on speed
        arrow_scale = 50
        ax_summary.arrow(center[0], center[1],
                       direction[0] * arrow_scale, direction[1] * arrow_scale,
                       head_width=10, head_length=20,
                       fc=unit_colors[i], ec='black', alpha=0.8,
                       linewidth=2, label=f"Unit {unit_id}: {speed:.2f} m/s")
    
    ax_summary.set_title("Summary of Propagation Directions for All Units", 
                       fontsize=fontsize+4)
    ax_summary.set_xlabel("X Position (μm)", fontsize=fontsize+2)
    ax_summary.set_ylabel("Y Position (μm)", fontsize=fontsize+2)
    ax_summary.set_aspect('equal')
    ax_summary.legend(fontsize=fontsize-2, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{filename}_all_units_propagation.png"))
    plt.savefig(os.path.join(save_path, f"{filename}_all_units_propagation.pdf"), 
               format='pdf', dpi=300)
    
    figures.append((fig_summary, ax_summary))
    
    return figures

def visualize_unit_speed_distribution(save_path, filename, results, unit_properties=None, 
                                     figsize=(14, 10), fontsize=12):
    """
    Visualize conduction speed distribution across units with additional unit properties
    
    Parameters:
    - save_path: directory to save the figures
    - filename: base filename for saved plots
    - results: list of dictionaries containing 'speed', 'direction', and 'unit' for each unit
    - unit_properties: optional dictionary with additional unit properties like firing rate, depth, etc.
                     format: {unit_id: {'property1': value1, 'property2': value2, ...}}
    - figsize: tuple for figure size
    - fontsize: base font size for plots
    
    Returns:
    - figures: list of (fig, axes) tuples for each created figure
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize + 2,
        'axes.titlesize': fontsize + 4,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })
    
    # Process results into a more usable format
    units_data = {}
    for result in results:
        unit_id = result['unit']
        # Handle different key naming conventions in results
        if 'speed_ms-1' in result:
            speed = result.pop('speed_ms-1')
        else:
            speed = result['speed']
        
        if unit_id not in units_data:
            units_data[unit_id] = {
                'speeds': [],
                'directions': [],
                'mean_speed': 0,
                'std_speed': 0
            }
        
        units_data[unit_id]['speeds'].append(speed)
        units_data[unit_id]['directions'].append(result['direction'])
    
    # Calculate statistics for each unit
    for unit_id in units_data:
        speeds = np.array(units_data[unit_id]['speeds'])
        directions = np.array(units_data[unit_id]['directions'])
        
        units_data[unit_id]['mean_speed'] = np.mean(speeds)
        units_data[unit_id]['median_speed'] = np.median(speeds)
        units_data[unit_id]['std_speed'] = np.std(speeds)
        units_data[unit_id]['min_speed'] = np.min(speeds)
        units_data[unit_id]['max_speed'] = np.max(speeds)
        units_data[unit_id]['mean_direction'] = np.mean(directions, axis=0)
    
    # Extract unit IDs and mean speeds for plotting
    unit_ids = list(units_data.keys())
    mean_speeds = [units_data[u]['mean_speed'] for u in unit_ids]
    std_speeds = [units_data[u]['std_speed'] for u in unit_ids]
    
    # Sort units by mean speed for better visualization
    sorted_indices = np.argsort(mean_speeds)
    sorted_unit_ids = [unit_ids[i] for i in sorted_indices]
    sorted_mean_speeds = [mean_speeds[i] for i in sorted_indices]
    sorted_std_speeds = [std_speeds[i] for i in sorted_indices]
    
    figures = []
    
    # Create main figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[2, 1])
    
    # Plot 1: Bar chart of mean speeds per unit
    ax1 = fig.add_subplot(gs[0, 0])
    bar_plot = ax1.bar(
        range(len(sorted_unit_ids)), 
        sorted_mean_speeds,
        yerr=sorted_std_speeds,
        capsize=5,
        color='royalblue',
        alpha=0.7,
        edgecolor='black'
    )
    
    # Add horizontal line for global mean
    global_mean = np.mean(mean_speeds)
    ax1.axhline(global_mean, color='red', linestyle='--', 
                label=f'Global Mean: {global_mean:.2f} m/s')
    
    ax1.set_xlabel('Unit ID (sorted by speed)', fontsize=fontsize + 2)
    ax1.set_ylabel('Mean Conduction Speed (m/s)', fontsize=fontsize + 2)
    ax1.set_title('Mean Conduction Speed by Unit', fontsize=fontsize + 4, pad=20)
    
    # Replace x-ticks with actual unit IDs
    if len(sorted_unit_ids) <= 20:
        ax1.set_xticks(range(len(sorted_unit_ids)))
        ax1.set_xticklabels([str(u) for u in sorted_unit_ids], rotation=45, ha='right')
    else:
        # If too many units, show only some ticks
        step = max(1, len(sorted_unit_ids) // 10)
        ax1.set_xticks(range(0, len(sorted_unit_ids), step))
        ax1.set_xticklabels([str(sorted_unit_ids[i]) for i in range(0, len(sorted_unit_ids), step)], 
                           rotation=45, ha='right')
    
    ax1.legend(fontsize=fontsize)
    
    # Plot 2: Histogram of mean speeds
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(
        data=mean_speeds,
        bins=10,
        ax=ax2,
        color='green',
        alpha=0.7,
        edgecolor='black',
        kde=True
    )
    ax2.axvline(global_mean, color='red', linestyle='--', 
                label=f'Mean: {global_mean:.2f} m/s')
    ax2.set_xlabel('Mean Conduction Speed (m/s)', fontsize=fontsize + 2)
    ax2.set_ylabel('Count', fontsize=fontsize + 2)
    ax2.set_title('Distribution of Mean Speeds', fontsize=fontsize + 4, pad=20)
    ax2.legend(fontsize=fontsize)
    
    # Plot 3: Speed vs additional unit property if available
    ax3 = fig.add_subplot(gs[1, 0])
    
    if unit_properties and any('firing_rate' in unit_properties.get(u, {}) for u in unit_ids):
        # Extract firing rates for units that have them
        x_data = []
        y_data = []
        for unit_id in unit_ids:
            if unit_id in unit_properties and 'firing_rate' in unit_properties[unit_id]:
                x_data.append(unit_properties[unit_id]['firing_rate'])
                y_data.append(units_data[unit_id]['mean_speed'])
        
        # Create scatter plot
        ax3.scatter(x_data, y_data, c='purple', alpha=0.7, s=50, edgecolor='black')
        ax3.set_xlabel('Firing Rate (Hz)', fontsize=fontsize + 2)
        ax3.set_ylabel('Mean Conduction Speed (m/s)', fontsize=fontsize + 2)
        ax3.set_title('Speed vs Firing Rate', fontsize=fontsize + 4, pad=20)
        
        # Calculate and display correlation
        if len(x_data) > 1:
            corr = np.corrcoef(x_data, y_data)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                    transform=ax3.transAxes, fontsize=fontsize,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    elif unit_properties and any('depth' in unit_properties.get(u, {}) for u in unit_ids):
        # If firing rate not available but depth is, plot depth vs speed
        x_data = []
        y_data = []
        for unit_id in unit_ids:
            if unit_id in unit_properties and 'depth' in unit_properties[unit_id]:
                x_data.append(unit_properties[unit_id]['depth'])
                y_data.append(units_data[unit_id]['mean_speed'])
        
        ax3.scatter(x_data, y_data, c='orange', alpha=0.7, s=50, edgecolor='black')
        ax3.set_xlabel('Depth (μm)', fontsize=fontsize + 2)
        ax3.set_ylabel('Mean Conduction Speed (m/s)', fontsize=fontsize + 2)
        ax3.set_title('Speed vs Depth', fontsize=fontsize + 4, pad=20)
        
        if len(x_data) > 1:
            corr = np.corrcoef(x_data, y_data)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                    transform=ax3.transAxes, fontsize=fontsize,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    else:
        # If no additional properties available, show speed vs unit index
        ax3.scatter(range(len(unit_ids)), mean_speeds, c='teal', alpha=0.7, s=50, edgecolor='black')
        ax3.set_xlabel('Unit Index', fontsize=fontsize + 2)
        ax3.set_ylabel('Mean Conduction Speed (m/s)', fontsize=fontsize + 2)
        ax3.set_title('Speed by Unit Index', fontsize=fontsize + 4, pad=20)
    
    # Plot 4: Box plot of speed distribution
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Prepare data for boxplot
    box_data = []
    for unit_id in sorted_unit_ids:
        box_data.append(units_data[unit_id]['speeds'])
    
    # Create box plot
    ax4.boxplot(box_data, vert=True, patch_artist=True,
              boxprops=dict(facecolor='lightblue', color='black'),
              whiskerprops=dict(color='black'),
              medianprops=dict(color='red'))
    
    ax4.set_xlabel('Unit (sorted by mean speed)', fontsize=fontsize + 2)
    ax4.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
    ax4.set_title('Speed Distribution by Unit', fontsize=fontsize + 4, pad=20)
    
    # Only show some x-ticks if there are many units
    if len(sorted_unit_ids) <= 10:
        ax4.set_xticks(range(1, len(sorted_unit_ids) + 1))
        ax4.set_xticklabels([str(u) for u in sorted_unit_ids], rotation=45, ha='right')
    else:
        step = max(1, len(sorted_unit_ids) // 5)
        ax4.set_xticks(range(1, len(sorted_unit_ids) + 1, step))
        ax4.set_xticklabels([str(sorted_unit_ids[i-1]) for i in range(1, len(sorted_unit_ids) + 1, step)], 
                           rotation=45, ha='right')
    
    # Add statistics text to the figure
    stats_text = (f'Total Units: {len(unit_ids)}\n'
                 f'Global Mean Speed: {global_mean:.2f} m/s\n'
                 f'Global Median Speed: {np.median(mean_speeds):.2f} m/s\n'
                 f'Speed Range: {min(mean_speeds):.2f} - {max(mean_speeds):.2f} m/s\n'
                 f'Std Dev: {np.std(mean_speeds):.2f} m/s')
    
    fig.text(0.02, 0.02, stats_text,
           fontsize=fontsize,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    figures.append((fig, (ax1, ax2, ax3, ax4)))
    
    # Create a specialized figure if we have multiple speeds per unit
    multiple_speeds_units = [u for u in unit_ids if len(units_data[u]['speeds']) > 1]
    
    if multiple_speeds_units:
        fig2, ax_multi = plt.subplots(figsize=(12, 6))
        
        # Create violin plot for units with multiple speed measurements
        violin_data = [units_data[u]['speeds'] for u in multiple_speeds_units]
        
        ax_multi.violinplot(violin_data, showmeans=True, showmedians=True)
        
        ax_multi.set_xlabel('Unit ID', fontsize=fontsize + 2)
        ax_multi.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
        ax_multi.set_title('Speed Distribution for Units with Multiple Measurements', 
                         fontsize=fontsize + 4, pad=20)
        
        ax_multi.set_xticks(range(1, len(multiple_speeds_units) + 1))
        ax_multi.set_xticklabels([str(u) for u in multiple_speeds_units], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{filename}_multiple_speeds_distribution.png"))
        plt.savefig(os.path.join(save_path, f"{filename}_multiple_speeds_distribution.pdf"), 
                   format='pdf', dpi=300)
        
        figures.append((fig2, ax_multi))
    
    # Save the main figure
    plt.figure(fig.number)
    plt.savefig(os.path.join(save_path, f"{filename}_unit_speed_distribution.png"))
    plt.savefig(os.path.join(save_path, f"{filename}_unit_speed_distribution.pdf"), 
               format='pdf', dpi=300)
    
    # Create a supplementary analysis if unit_properties contains cell type information
    if unit_properties and any('cell_type' in unit_properties.get(u, {}) for u in unit_ids):
        fig3, ax_types = plt.subplots(figsize=(10, 6))
        
        # Group units by cell type
        cell_types = {}
        for unit_id in unit_ids:
            if unit_id in unit_properties and 'cell_type' in unit_properties[unit_id]:
                cell_type = unit_properties[unit_id]['cell_type']
                if cell_type not in cell_types:
                    cell_types[cell_type] = []
                cell_types[cell_type].append(units_data[unit_id]['mean_speed'])
        
        # Create grouped boxplot
        box_positions = list(range(1, len(cell_types) + 1))
        box_data = [cell_types[cell_type] for cell_type in cell_types]
        
        ax_types.boxplot(box_data, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', color='black'),
                       whiskerprops=dict(color='black'),
                       medianprops=dict(color='red'))
        
        # Add individual points
        for i, (cell_type, speeds) in enumerate(cell_types.items()):
            ax_types.scatter([i + 1] * len(speeds), speeds, 
                          alpha=0.6, s=40, edgecolor='black')
        
        ax_types.set_xlabel('Cell Type', fontsize=fontsize + 2)
        ax_types.set_ylabel('Mean Conduction Speed (m/s)', fontsize=fontsize + 2)
        ax_types.set_title('Conduction Speed by Cell Type', fontsize=fontsize + 4, pad=20)
        
        ax_types.set_xticks(box_positions)
        ax_types.set_xticklabels(list(cell_types.keys()))
        
        # Add statistical test if we have at least two groups
        if len(cell_types) >= 2:
            from scipy.stats import f_oneway
            
            # Perform ANOVA
            f_stat, p_value = f_oneway(*box_data)
            
            stats_text = f'ANOVA: F={f_stat:.2f}, p={p_value:.4f}'
            if p_value < 0.05:
                stats_text += '\nSignificant difference between groups'
            
            ax_types.text(0.05, 0.95, stats_text, 
                        transform=ax_types.transAxes, fontsize=fontsize,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{filename}_cell_type_speeds.png"))
        plt.savefig(os.path.join(save_path, f"{filename}_cell_type_speeds.pdf"), 
                   format='pdf', dpi=300)
        
        figures.append((fig3, ax_types))
    
    return figures

def visualize_unit_propagation_V2(save_path, filename, results, templates, probe_locations, 
                              sampling_rate, figsize=(10, 8), fontsize=16, unit_ids=None):
    """
    Visualize propagation direction for each unit across electrodes
    
    Parameters:
    - save_path: directory to save the figures
    - filename: base filename for saved plots
    - results: list of dictionaries containing 'speed' and 'direction' for each unit
    - templates: templates for each unit (shape: time x channels)
    - probe_locations: numpy array of shape (n_channels, 2) with x,y coordinates
    - sampling_rate: sampling rate in Hz
    - figsize: tuple for figure size
    - fontsize: base font size for plots
    - unit_ids: optional list of specific unit IDs to plot (default: all units)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.signal import find_peaks
    
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize + 2,
        'axes.titlesize': fontsize + 4,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })

    
    
    # Create custom colormap for temporal progression
    #colors = [(0, 0, 0.8), (0, 0.8, 0), (0.8, 0, 0)]  # Blue -> Green -> Red
    #colors = [(0.0, 0.0, 0.5),    # Dark navy blue
    #      (0.0, 0.3, 0.7),    # Medium blue
    #      (0.2, 0.5, 0.9),    # Light blue
    #      (0.4, 0.7, 1.0)]    # Very light blue (still visible)

    #cmap_name = 'enhanced_blue_contrast'
    #cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    colors = sns.color_palette("viridis", as_cmap=True).colors
    cmap_name = 'viridis'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    
    # Process unit IDs
    if unit_ids is None:
        unit_ids = [r['unit'] for r in results]
    
    # Process results to map unit IDs to their data
    unit_data = {}
    for result in results:
        unit_id = result['unit']
        if 'speed_ms-1' in result:
            result['speed'] = result.pop('speed_ms-1')
        unit_data[unit_id] = result
    
    figures = []
    
    # Create a figure for each unit
    for unit_id in unit_ids:
        if unit_id not in unit_data:
            print(f"Unit {unit_id} not found in results, skipping.")
            continue
            
        result = unit_data[unit_id]
        template = templates[unit_ids.index(unit_id)]
        
        # Create figure with two subplots (template and absolute time)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                      gridspec_kw={'width_ratios': [1, 1.2]})
        
        # Find peak/trough times for each channel
        peak_times = np.zeros(template.shape[1])
        peak_amplitudes = np.zeros(template.shape[1])
       
        for chan in range(template.shape[1]):
            # Find peaks and their properties
            peaks, properties = find_peaks(np.abs(template[:, chan]), 
                                          height=np.std(template[:, chan]))
            if len(peaks) > 0:
                # Get the highest peak
                max_peak_idx = np.argmax(properties['peak_heights'])
                peak_times[chan] = peaks[max_peak_idx]
                peak_amplitudes[chan] = properties['peak_heights'][max_peak_idx]
        
        # Only consider channels with significant peaks
        amplitude_threshold = np.median(peak_amplitudes) * 0.3 
        valid_channels = peak_amplitudes > amplitude_threshold
        valid_peak_times = peak_times[valid_channels]
        valid_locations = probe_locations[valid_channels]
        valid_amplitudes = peak_amplitudes[valid_channels]
        
        # Normalize peak times to [0, 1] for color mapping
        if len(valid_peak_times) > 0:
            min_time = np.min(valid_peak_times)
            max_time = np.max(valid_peak_times)
            time_range = max_time - min_time
            
            if time_range > 0:
                normalized_times = (valid_peak_times - min_time) / time_range
            else:
                normalized_times = np.zeros_like(valid_peak_times)
        else:
            normalized_times = np.array([])
        '''
        # Plot template waveforms
        ax1.set_title(f"Unit {unit_id} Template", fontsize=fontsize+2)
        time_axis = np.arange(template.shape[0]) / sampling_rate * 1000  # Convert to ms
        
        # Plot a subset of channels for clarity (top 10 by amplitude)
        if valid_channels.sum() > 10:
            top_channels = np.argsort(peak_amplitudes)[-10:]
            top_channels = top_channels[peak_amplitudes[top_channels] > amplitude_threshold]
        else:
            top_channels = np.where(valid_channels)[0]
        
        # Plot each channel with color based on peak time
        for i, chan in enumerate(top_channels):
            if peak_amplitudes[chan] > amplitude_threshold:
                norm_time = (peak_times[chan] - min_time) / time_range if time_range > 0 else 0
                ax1.plot(time_axis, template[:, chan], 
                        color=cm(norm_time), 
                        alpha=0.8,
                        linewidth=1.5)
        
        ax1.set_xlabel("Time (ms)", fontsize=fontsize)
        ax1.set_ylabel("Amplitude", fontsize=fontsize)
        ax1.grid(True, alpha=0.3)


        # Plot template waveforms
        ax1.set_title(f"Unit {unit_id} Template", fontsize=fontsize+2)
        time_axis = np.arange(template.shape[0]) / sampling_rate * 1000  # Convert to ms

        # Find the channel that has a peak closest to the center of the template
        center_time = template.shape[0] // 2
        center_time_diffs = np.abs(peak_times - center_time)
        center_channel_idx = np.argmin(center_time_diffs[valid_channels])
        center_channel = np.where(valid_channels)[0][center_channel_idx]

        # Get indices of all valid channels
        valid_channel_indices = np.where(valid_channels)[0]

        # Find position of center channel in the valid_channel_indices array
        center_channel_position = np.where(valid_channel_indices == center_channel)[0][0]

        # Calculate how many channels we can take before and after
        channels_before = min(5, center_channel_position)
        channels_after = min(5, len(valid_channel_indices) - center_channel_position - 1)

        # Get the channels to plot (5 before and 5 after the center one, if possible)
        start_idx = center_channel_position - channels_before
        end_idx = center_channel_position + channels_after + 1
        channels_to_plot = valid_channel_indices[start_idx:end_idx]

        # Plot each channel with color based on peak time
        for i, chan in enumerate(channels_to_plot):
            if peak_amplitudes[chan] > amplitude_threshold:
                norm_time = (peak_times[chan] - min_time) / time_range if time_range > 0 else 0
                ax1.plot(time_axis, template[:, chan], 
                        color=cm(norm_time), 
                        alpha=0.8,
                        linewidth=1.5)

        ax1.set_xlabel("Time [ms]", fontsize=fontsize)
        ax1.set_ylabel("Amplitude", fontsize=fontsize)
        ax1.grid(True, alpha=0.3)
        '''
        # Plot template waveforms
        ax1.set_title(f"Unit {unit_id} Template", fontsize=fontsize+2)
        time_axis = np.arange(template.shape[0]) / sampling_rate * 1000  # Convert to ms

        # Plot all valid channels instead of just a subset around the center
        valid_channel_indices = np.where(valid_channels)[0]

        # Plot each channel with color based on peak time
        for i, chan in enumerate(valid_channel_indices):
            if peak_amplitudes[chan] > amplitude_threshold:
                norm_time = (peak_times[chan] - min_time) / time_range if time_range > 0 else 0
                ax1.plot(time_axis, template[:, chan], 
                        color=cm(norm_time), 
                        alpha=0.8,
                        linewidth=1.5)

        ax1.set_xlabel("Time [ms]", fontsize=fontsize)
        ax1.set_ylabel("Amplitude", fontsize=fontsize)
        ax1.grid(True, alpha=0.3)

        # Plot electrode map with propagation    
        # Plot electrode map with propagation using absolute time
        ax2.set_title(f"Unit {unit_id} Propagation", fontsize=fontsize+2)
        
        # Plot all electrode positions
        ax2.scatter(probe_locations[:, 0], probe_locations[:, 1], 
                  color='gray', alpha=0.3, s=30, zorder=1, marker = 's')
        
        if len(valid_peak_times) > 0:
            # Calculate absolute time in ms
            absolute_times_ms = valid_peak_times / sampling_rate * 1000  # Convert to ms
            
            # Create a colormap from blue to red for absolute time
            time_norm = plt.Normalize(np.min(absolute_times_ms), np.max(absolute_times_ms))
            
            # Plot electrodes with color based on absolute activation time
            scatter = ax2.scatter(valid_locations[:, 0], valid_locations[:, 1],
                               c=absolute_times_ms, cmap=cm,
                               s=valid_amplitudes/np.max(valid_amplitudes)*100 + 30,
                               alpha=0.8, edgecolor='black', zorder=2, marker = 's')
            
            # Add colorbar for absolute time
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Activation Time [ms]', fontsize=fontsize)
            
            # Find the electrode with maximum amplitude
            if len(valid_amplitudes) > 0:
                max_amp_idx = np.argmax(valid_amplitudes)
                center = valid_locations[max_amp_idx]
                
                # Draw propagation direction arrow
                direction = result['direction']
                speed = result['speed']
                
                # Scale arrow length based on speed
                arrow_scale = 50  # Adjust as needed
                ax2.arrow(center[0], center[1],
                        direction[0] * arrow_scale, direction[1] * arrow_scale,
                        head_width=10, head_length=20,
                        fc='yellow', ec='black', alpha=0.9,
                        linewidth=2, zorder=3)
                
                # Add sequential arrows showing the propagation path   ##REMOVED 250509 KV
                #sorted_by_time = np.argsort(absolute_times_ms)
                #for i in range(len(sorted_by_time)-1):
                #    idx1 = sorted_by_time[i]
                #    idx2 = sorted_by_time[i+1]
                    
                #    # Only draw arrows between sequential points
                #    ax2.annotate("",
                #              xy=(valid_locations[idx2, 0], valid_locations[idx2, 1]),
                #              xytext=(valid_locations[idx1, 0], valid_locations[idx1, 1]),
                #              arrowprops=dict(arrowstyle="->", color='magenta', 
                #                            alpha=0.7, linewidth=1.5),
                #              zorder=2)
            
            # Add time range information
            time_range_ms = np.max(absolute_times_ms) - np.min(absolute_times_ms)
            ax2.text(0.02, 0.02, f"Time Range: {time_range_ms:.2f} ms",
                   transform=ax2.transAxes, fontsize=fontsize,
                   verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add statistics text
        stats_text = (f"Speed: {speed:.2f} m/s\n"
                     f"Direction: [{direction[0]:.2f}, {direction[1]:.2f}]\n"
                     f"Active channels: {valid_channels.sum()}")
        
        ax2.text(0.02, 0.98, stats_text,
               transform=ax2.transAxes,
               fontsize=fontsize,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel("X Position (μm)", fontsize=fontsize)
        ax2.set_ylabel("Y Position (μm)", fontsize=fontsize)
        ax2.set_aspect('equal')
        
        # Adjust layout
        plt.tight_layout()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{filename}_unit{unit_id}_propagation.png"))
        plt.savefig(os.path.join(save_path, f"{filename}_unit{unit_id}_propagation.pdf"), 
                   format='pdf', dpi=300)
        plt.close()
        
        figures.append((fig, (ax1, ax2)))
    
    # Create a summary figure with all unit directions
    fig_summary, ax_summary = plt.subplots(figsize=(12, 10))
    
    # Plot all electrode positions
    ax_summary.scatter(probe_locations[:, 0], probe_locations[:, 1], 
                    color='gray', marker='s', alpha=0.3, s=30)
    
    # Define more vibrant colors for different units
    unit_colors = plt.cm.tab10(np.linspace(0, 1, 10))  # More saturated colors
    # If we have more than 10 units, cycle through the colors
    if len(unit_ids) > 10:
        unit_colors = np.vstack([unit_colors] * (len(unit_ids) // 10 + 1))[:len(unit_ids)]
    
    # For each unit, draw its propagation direction
    for i, unit_id in enumerate(unit_ids):
        if unit_id not in unit_data:
            continue
            
        result = unit_data[unit_id]
        template = templates[unit_id]
        
        # Find peak amplitudes across electrodes
        peak_amplitudes = np.max(np.abs(template), axis=0)
        origin_electrode = np.argmax(peak_amplitudes)
        center = probe_locations[origin_electrode]
        
        direction = result['direction']
        speed = result['speed']
        
        # Scale arrow length based on speed
        arrow_scale = 60  # Increased for better visibility
        ax_summary.arrow(center[0], center[1],
                       direction[0] * arrow_scale, direction[1] * arrow_scale,
                       head_width=15, head_length=25,
                       fc=unit_colors[i], ec='white', alpha=1.0,  # Full opacity
                       linewidth=3, label=f"Unit {unit_id}: {speed:.2f} m/s")
    
    ax_summary.set_title("Summary of Propagation Directions for All Units", 
                       fontsize=fontsize+4)
    ax_summary.set_xlabel("X Position (μm)", fontsize=fontsize+2)
    ax_summary.set_ylabel("Y Position (μm)", fontsize=fontsize+2)
    ax_summary.set_aspect('equal')
    #ax_summary.legend(fontsize=fontsize-2, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{filename}_all_units_propagation.png"))
    plt.savefig(os.path.join(save_path, f"{filename}_all_units_propagation.pdf"), 
               format='pdf', dpi=300)
    plt.close()
    figures.append((fig_summary, ax_summary))
    
    return figures



def visualize_speed_vs_div_by_cell_type_df(save_path, data_df, 
                                         figsize=(14, 10), fontsize=12, 
                                         show_individual=True, group_by_chip=False):
    """
    Visualize conduction speeds across DIVs for different cell types using a DataFrame
    
    Parameters:
    - save_path: directory to save the figures
    - data_df: pandas DataFrame with columns:
        - unit: unit ID
        - direction: propagation direction vector
        - speed_ms-1: conduction speed in m/s
        - chip_id: ID of the chip/recording
        - div: days in vitro
        - cell_type: type of the cell/unit
        - filename: source recording file
    - figsize: tuple for figure size
    - fontsize: base font size for plots
    - show_individual: whether to show individual data points
    - group_by_chip: whether to include chip ID in grouping (helps with technical replicates)
    
    Returns:
    - figures: list of (fig, axes) tuples for each created figure
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy import stats
    from matplotlib.gridspec import GridSpec
    
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize + 2,
        'axes.titlesize': fontsize + 4,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })
    
    # Make a copy to avoid modifying the original dataframe
    df = data_df.copy()
    
    # Ensure the speed column is named consistently
    if 'speed_ms-1' in df.columns and 'speed' not in df.columns:
        df.rename(columns={'speed_ms-1': 'speed'}, inplace=True)
    
    # Get unique DIVs and cell types
    divs = sorted(df['div'].unique())
    cell_types = sorted(df['cell_type'].unique())
    
    # Create color palette for cell types
    if len(cell_types) <= 10:
        palette = sns.color_palette("tab10", len(cell_types))
    else:
        palette = sns.color_palette("husl", len(cell_types))
    
    cell_type_colors = dict(zip(cell_types, palette))
    
    figures = []
    
    # Figure 1: Line plot with error bars for each cell type across DIVs
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    # Calculate mean and standard error for each cell type at each DIV
    if group_by_chip:
        # First calculate mean for each chip, then average across chips
        chip_means = df.groupby(['div', 'cell_type', 'chip_id'])['speed'].mean().reset_index()
        grouped = chip_means.groupby(['div', 'cell_type'])['speed'].agg(['mean', 'std', 'count']).reset_index()
    else:
        # Direct grouping by DIV and cell type
        grouped = df.groupby(['div', 'cell_type'])['speed'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    
    # Plot line for each cell type
    for cell_type in cell_types:
        cell_data = grouped[grouped['cell_type'] == cell_type]
        
        if not cell_data.empty:
            ax1.errorbar(cell_data['div'], cell_data['mean'], yerr=cell_data['se'],
                      fmt='o-', linewidth=2, markersize=8, 
                      color=cell_type_colors[cell_type],
                      label=f"{cell_type} (n={df[df['cell_type'] == cell_type]['unit'].nunique()})")
            
            # Add individual data points if requested
            if show_individual:
                if group_by_chip:
                    # Show chip means
                    individual_data = chip_means[chip_means['cell_type'] == cell_type]
                else:
                    # Show all units
                    individual_data = df[df['cell_type'] == cell_type]
                
                ax1.scatter(individual_data['div'], individual_data['speed'], 
                         alpha=0.3, color=cell_type_colors[cell_type], s=30)
    
    ax1.set_xlabel('Days In Vitro (DIV)', fontsize=fontsize + 2)
    ax1.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
    ax1.set_title('Conduction Speed vs DIV by Cell Type', fontsize=fontsize + 4, pad=20)
    
    # Set better x-ticks
    ax1.set_xticks(divs)
    
    # Add legend
    ax1.legend(fontsize=fontsize, title="Cell Type", title_fontsize=fontsize)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Figure 2: Box plots for each cell type at each DIV
    fig2, ax2 = plt.subplots(figsize=figsize)
    
    # Create box plot
    if group_by_chip:
        sns.boxplot(x='div', y='speed', hue='cell_type', data=chip_means,
                  palette=cell_type_colors, ax=ax2)
        
        if show_individual:
            sns.stripplot(x='div', y='speed', hue='cell_type', data=chip_means,
                        palette=cell_type_colors, dodge=True, 
                        alpha=0.5, size=5, ax=ax2)
    else:
        sns.boxplot(x='div', y='speed', hue='cell_type', data=df,
                  palette=cell_type_colors, ax=ax2)
        
        if show_individual:
            sns.stripplot(x='div', y='speed', hue='cell_type', data=df,
                        palette=cell_type_colors, dodge=True, 
                        alpha=0.5, size=5, ax=ax2)
    
    ax2.set_xlabel('Days In Vitro (DIV)', fontsize=fontsize + 2)
    ax2.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
    ax2.set_title('Conduction Speed Distribution by DIV and Cell Type', fontsize=fontsize + 4, pad=20)
    
    # Adjust legend
    handles, labels = ax2.get_legend_handles_labels()
    n_cell_types = len(cell_types)
    ax2.legend(handles[:n_cell_types], labels[:n_cell_types], 
             fontsize=fontsize, title="Cell Type", title_fontsize=fontsize)
    
    # Create statistical analysis and visualization
    fig3 = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig3)
    
    # Plot 1: Mean speed vs DIV with regression lines
    ax3_1 = fig3.add_subplot(gs[0, 0])
    
    # Plot regression line for each cell type
    for cell_type in cell_types:
        if group_by_chip:
            cell_data = chip_means[chip_means['cell_type'] == cell_type]
        else:
            cell_data = df[df['cell_type'] == cell_type]
        
        if len(cell_data) > 1:
            sns.regplot(x='div', y='speed', data=cell_data,
                      scatter=True, ci=95, 
                      scatter_kws={'alpha': 0.4, 's': 30},
                      line_kws={'linewidth': 2},
                      color=cell_type_colors[cell_type],
                      label=cell_type, ax=ax3_1)
    
    ax3_1.set_xlabel('DIV', fontsize=fontsize)
    ax3_1.set_ylabel('Speed (m/s)', fontsize=fontsize)
    ax3_1.set_title('Regression Analysis', fontsize=fontsize + 2)
    ax3_1.legend(fontsize=fontsize - 2)
    
    # Plot 2: Bar chart of mean speed by cell type
    ax3_2 = fig3.add_subplot(gs[0, 1])
    
    # Calculate mean and standard error by cell type
    if group_by_chip:
        # First average by chip
        chip_means_by_ct = chip_means.groupby(['cell_type', 'chip_id'])['speed'].mean().reset_index()
        # Then calculate statistics across chips
        cell_type_stats = chip_means_by_ct.groupby('cell_type')['speed'].agg(['mean', 'std', 'count']).reset_index()
    else:
        cell_type_stats = df.groupby('cell_type')['speed'].agg(['mean', 'std', 'count']).reset_index()
    
    cell_type_stats['se'] = cell_type_stats['std'] / np.sqrt(cell_type_stats['count'])
    
    # Sort by mean speed
    cell_type_stats = cell_type_stats.sort_values('mean', ascending=False)
    
    # Plot bars
    bars = ax3_2.bar(cell_type_stats['cell_type'], cell_type_stats['mean'], 
                  yerr=cell_type_stats['se'], capsize=5,
                  color=[cell_type_colors[ct] for ct in cell_type_stats['cell_type']])
    
    # Add count labels
    for i, bar in enumerate(bars):
        ax3_2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cell_type_stats['se'].iloc[i] + 0.02,
                 f"n={cell_type_stats['count'].iloc[i]}", 
                 ha='center', va='bottom', fontsize=fontsize - 2)
    
    ax3_2.set_xlabel('Cell Type', fontsize=fontsize)
    ax3_2.set_ylabel('Mean Speed (m/s)', fontsize=fontsize)
    ax3_2.set_title('Average Speed by Cell Type', fontsize=fontsize + 2)
    plt.setp(ax3_2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: ANOVA test results
    ax3_3 = fig3.add_subplot(gs[1, 0])
    
    # Perform one-way ANOVA for cell types
    data_for_anova = chip_means if group_by_chip else df
    cell_types_with_enough_data = [ct for ct in cell_types 
                                if len(data_for_anova[data_for_anova['cell_type'] == ct]) >= 3]
    
    if len(cell_types_with_enough_data) >= 2:
        anova_data = [data_for_anova[data_for_anova['cell_type'] == ct]['speed'].values 
                   for ct in cell_types_with_enough_data]
        
        f_val, p_val = stats.f_oneway(*anova_data)
        
        # Create a visual representation of ANOVA results
        ax3_3.axis('off')
        
        anova_text = (f"One-way ANOVA Results:\n"
                    f"F-value: {f_val:.3f}\n"
                    f"p-value: {p_val:.4f}\n\n")
        
        if p_val < 0.05:
            anova_text += "There is a significant difference\nbetween cell types (p < 0.05)."
        else:
            anova_text += "No significant difference\nbetween cell types (p ≥ 0.05)."
        
        ax3_3.text(0.5, 0.5, anova_text, 
                 ha='center', va='center', 
                 fontsize=fontsize,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    else:
        ax3_3.text(0.5, 0.5, "Not enough data for\nANOVA analysis\n(need at least 2 cell types\nwith 3+ samples each)", 
                 ha='center', va='center', fontsize=fontsize)
    
    # Plot 4: DIV correlation analysis
    ax3_4 = fig3.add_subplot(gs[1, 1])
    
    # Calculate correlation between DIV and speed for each cell type
    corr_data = []
    
    for cell_type in cell_types:
        if group_by_chip:
            cell_data = chip_means[chip_means['cell_type'] == cell_type]
        else:
            cell_data = df[df['cell_type'] == cell_type]
        
        if len(cell_data) > 5:  # Require at least 5 data points for correlation
            corr, p = stats.pearsonr(cell_data['div'], cell_data['speed'])
            corr_data.append({
                'Cell Type': cell_type,
                'Correlation': corr,
                'p-value': p,
                'Significant': p < 0.05
            })
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        
        # Sort by correlation strength
        corr_df = corr_df.sort_values('Correlation', ascending=False)
        
        # Create bar colors based on significance
        bar_colors = [cell_type_colors[ct] if sig else 'lightgray' 
                   for ct, sig in zip(corr_df['Cell Type'], corr_df['Significant'])]
        
        # Plot bars
        bars = ax3_4.bar(corr_df['Cell Type'], corr_df['Correlation'], color=bar_colors)
        
        # Add significance markers
        for i, bar in enumerate(bars):
            if corr_df['Significant'].iloc[i]:
                ax3_4.text(bar.get_x() + bar.get_width()/2, 
                         bar.get_height() + 0.02 if bar.get_height() >= 0 else bar.get_height() - 0.08,
                         '*', ha='center', va='bottom', fontsize=fontsize + 4)
        
        ax3_4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3_4.set_xlabel('Cell Type', fontsize=fontsize)
        ax3_4.set_ylabel('Pearson Correlation\nwith DIV', fontsize=fontsize)
        ax3_4.set_title('Speed vs DIV Correlation', fontsize=fontsize + 2)
        plt.setp(ax3_4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax3_4.text(0.5, 0.5, "Not enough data for\ncorrelation analysis\n(need at least 5 data points\nper cell type)", 
                 ha='center', va='center', fontsize=fontsize)
    
    # Additional figure: Speed vs Chip ID
    if len(df['chip_id'].unique()) > 1:
        fig4, ax4 = plt.subplots(figsize=figsize)
        
        # Use different markers for different DIVs
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        div_markers = {div: markers[i % len(markers)] for i, div in enumerate(divs)}
        
        # Create a scatter plot for each cell type and DIV combination
        for cell_type in cell_types:
            for div in divs:
                subset = df[(df['cell_type'] == cell_type) & (df['div'] == div)]
                
                if not subset.empty:
                    # Calculate mean speed per chip
                    chip_speeds = subset.groupby('chip_id')['speed'].mean().reset_index()
                    
                    ax4.scatter(chip_speeds['chip_id'], chip_speeds['speed'],
                             marker=div_markers[div], s=100,
                             color=cell_type_colors[cell_type],
                             alpha=0.7,
                             label=f"{cell_type}, DIV {div}")
        
        ax4.set_xlabel('Chip ID', fontsize=fontsize + 2)
        ax4.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
        ax4.set_title('Conduction Speed by Chip ID', fontsize=fontsize + 4, pad=20)
        
        # Create a legend that doesn't duplicate entries
        handles, labels = ax4.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax4.legend(by_label.values(), by_label.keys(), 
                 fontsize=fontsize - 2, loc='best')
        
        # Set better x-ticks
        ax4.set_xticks(sorted(df['chip_id'].unique()))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"Speed_by_chip.png"))
        plt.savefig(os.path.join(save_path, f"Speed_by_chip.pdf"), format='pdf', dpi=300)
        
        figures.append((fig4, ax4))
    
    # Adjust layouts and save figures
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    
    plt.figure(fig1.number)
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_line.png"))
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_line.pdf"), format='pdf', dpi=300)
    
    plt.figure(fig2.number)
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_box.png"))
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_box.pdf"), format='pdf', dpi=300)
    
    plt.figure(fig3.number)
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_stats.png"))
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_stats.pdf"), format='pdf', dpi=300)
    
    figures = [(fig1, ax1), (fig2, ax2), (fig3, (ax3_1, ax3_2, ax3_3, ax3_4))]
    
    # Optional: Create plots for each DIV separately
    for div in divs:
        div_data = df[df['div'] == div]
        
        if len(div_data) > 0:
            fig_div, ax_div = plt.subplots(figsize=(10, 8))
            
            # Create violin plot for this DIV
            if group_by_chip:
                # Aggregate by chip first
                chip_div_data = div_data.groupby(['cell_type', 'chip_id'])['speed'].mean().reset_index()
                #sns.violinplot(x='cell_type', y='speed', data=chip_div_data,
                #             palette=cell_type_colors, ax=ax_div)
                sns.swarmplot(x='cell_type', y='speed', data=chip_div_data, jitter=True,
                            palette=cell_type_colors, ax=ax_div)
                
                if show_individual:
                    sns.stripplot(x='cell_type', y='speed', data=chip_div_data,
                                color='black', alpha=0.5, size=5, jitter=True, ax=ax_div)
            else:
                #sns.violinplot(x='cell_type', y='speed', data=div_data,
                #             palette=cell_type_colors, ax=ax_div)
                sns.swarmplot(x='cell_type', y='speed', data=div_data, jitter=True,
                            palette=cell_type_colors, ax=ax_div)
                
                if show_individual:
                    sns.stripplot(x='cell_type', y='speed', data=div_data,
                                color='black', alpha=0.5, size=5, jitter=True, ax=ax_div)
            
            ax_div.set_xlabel('Cell Type', fontsize=fontsize + 2)
            ax_div.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
            ax_div.set_title(f'Speed Distribution by Cell Type at DIV {div}', 
                           fontsize=fontsize + 4, pad=20)
            
            # Add count annotation
            for i, cell_type in enumerate(div_data['cell_type'].unique()):
                count = len(div_data[div_data['cell_type'] == cell_type])
                ax_div.text(i, div_data['speed'].max() * 1.05, f"n={count}", 
                         ha='center', fontsize=fontsize - 2)
            
            plt.setp(ax_div.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(os.path.join(save_path, f"Div{div}_by_cell_type.png"))
            plt.savefig(os.path.join(save_path, f"Div{div}_by_cell_type.pdf"), 
                       format='pdf', dpi=300)
            
            figures.append((fig_div, ax_div))
    
    # Additional: Direction visualization
    fig_dir, ax_dir = plt.subplots(figsize=(12, 10))
    
    # Extract x and y components of direction vectors
    df['dir_x'] = df['direction'].apply(lambda d: d[0] if isinstance(d, (list, np.ndarray)) else np.nan)
    df['dir_y'] = df['direction'].apply(lambda d: d[1] if isinstance(d, (list, np.ndarray)) else np.nan)
    
    # Plot direction vectors colored by cell type
    for cell_type in cell_types:
        cell_data = df[df['cell_type'] == cell_type]
        
        # Filter out any rows with NaN directions
        cell_data = cell_data.dropna(subset=['dir_x', 'dir_y'])
        
        if not cell_data.empty:
            ax_dir.quiver(np.zeros(len(cell_data)), np.zeros(len(cell_data)),
                       cell_data['dir_x'], cell_data['dir_y'],
                       color=cell_type_colors[cell_type], 
                       label=cell_type,
                       alpha=0.7, scale=2, width=0.005)
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax_dir.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Set equal aspect ratio
    ax_dir.set_aspect('equal')
    
    # Add grid and axes
    ax_dir.grid(True, alpha=0.3)
    ax_dir.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_dir.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    ax_dir.set_xlim(-1.2, 1.2)
    ax_dir.set_ylim(-1.2, 1.2)
    
    ax_dir.set_xlabel('X Direction', fontsize=fontsize + 2)
    ax_dir.set_ylabel('Y Direction', fontsize=fontsize + 2)
    ax_dir.set_title('Propagation Directions by Cell Type', fontsize=fontsize + 4, pad=20)
    
    ax_dir.legend(fontsize=fontsize)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Direction_by_cell_type.png"))
    plt.savefig(os.path.join(save_path, f"Direction_by_cell_type.pdf"), format='pdf', dpi=300)
    
    figures.append((fig_dir, ax_dir))
    
    return figures

def visualize_speed_vs_div_by_cell_type_df_grouped(save_path, data_df, 
                                         figsize=(14, 10), fontsize=12, 
                                         show_individual=True, group_by_chip=False):
    """
    Visualize conduction speeds across DIVs for different cell types using a DataFrame
    
    Parameters:
    - save_path: directory to save the figures
    - filename: base filename for saved plots
    - data_df: pandas DataFrame with columns:
        - unit: unit ID
        - direction: propagation direction vector
        - speed_ms-1: conduction speed in m/s
        - chip_id: ID of the chip/recording
        - div: days in vitro
        - cell_type: type of the cell/unit
        - filename: source recording file
    - figsize: tuple for figure size
    - fontsize: base font size for plots
    - show_individual: whether to show individual data points
    - group_by_chip: whether to include chip ID in grouping (helps with technical replicates)
    
    Returns:
    - figures: list of (fig, axes) tuples for each created figure
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy import stats
    from matplotlib.gridspec import GridSpec
    
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize + 2,
        'axes.titlesize': fontsize + 4,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })
    
    # Make a copy to avoid modifying the original dataframe
    df = data_df.copy()
    
    # Ensure the speed column is named consistently
    if 'speed_ms-1' in df.columns and 'speed' not in df.columns:
        df.rename(columns={'speed_ms-1': 'speed'}, inplace=True)
    
    # Get unique DIVs and cell types
    divs = sorted(df['div'].unique())
    cell_types = sorted(df['cell_type'].unique())
    
    # Create DIV ranges
    div_ranges = [
        {'name': 'DIV7-9', 'range': (7, 9)},
        {'name': 'DIV11-15', 'range': (11, 15)},
        {'name': 'DIV16-20', 'range': (16, 20)},
        {'name': 'DIV21-25', 'range': (21, 25)},
        {'name': 'DIV26-30', 'range': (26, 30)},
        {'name': 'DIV31-38', 'range': (31, 38)}
    ]
    
    # Add a 'div_group' column to categorize DIVs into ranges
    def assign_div_group(div):
        for range_info in div_ranges:
            min_div, max_div = range_info['range']
            if min_div <= div <= max_div:
                return range_info['name']
        return f'DIV{div}'  # For any DIVs outside the defined ranges
    
    df['week'] = df['div'].apply(assign_div_group)
    
    # Create color palette for cell types
    if len(cell_types) <= 10:
        palette = sns.color_palette("tab10", len(cell_types))
    else:
        palette = sns.color_palette("husl", len(cell_types))
    
    cell_type_colors = dict(zip(cell_types, palette))
    
    figures = []
    
    # Figure 1: Line plot with error bars for each cell type across DIVs
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    # Calculate mean and standard error for each cell type at each DIV group
    if group_by_chip:
        # First calculate mean for each chip, then average across chips
        chip_means = df.groupby(['week', 'cell_type', 'chip_id'])['speed'].mean().reset_index()
        grouped = chip_means.groupby(['week', 'cell_type'])['speed'].agg(['mean', 'std', 'count']).reset_index()
    else:
        # Direct grouping by DIV group and cell type
        grouped = df.groupby(['week', 'cell_type'])['speed'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    
    # Plot line for each cell type
    for cell_type in cell_types:
        cell_data = grouped[grouped['cell_type'] == cell_type]
        
        if not cell_data.empty:
            ax1.errorbar(cell_data['week'], cell_data['mean'], yerr=cell_data['se'],
                      fmt='o-', linewidth=2, markersize=8, 
                      color=cell_type_colors[cell_type],
                      label=f"{cell_type} (n={df[df['cell_type'] == cell_type]['unit'].nunique()})")
            
            # Add individual data points if requested
            if show_individual:
                if group_by_chip:
                    # Show chip means
                    individual_data = chip_means[chip_means['cell_type'] == cell_type]
                else:
                    # Show all units
                    individual_data = df[df['cell_type'] == cell_type]
                
                ax1.scatter(individual_data['week'], individual_data['speed'], 
                         alpha=0.3, color=cell_type_colors[cell_type], s=30)
    
    ax1.set_xlabel('Days In Vitro (DIV)', fontsize=fontsize + 2)
    ax1.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
    ax1.set_title('Conduction Speed vs DIV by Cell Type', fontsize=fontsize + 4, pad=20)
    
    # Set better x-ticks
    div_group_names = [r['name'] for r in div_ranges]
    ax1.set_xticks(range(len(div_group_names)))
    ax1.set_xticklabels(div_group_names, rotation=45, ha='right')
    
    # Add legend
    ax1.legend(fontsize=fontsize, title="Cell Type", title_fontsize=fontsize)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Figure 2: Box plots for each cell type at each DIV
    fig2, ax2 = plt.subplots(figsize=figsize)
    
    # Create box plot
    if group_by_chip:
        sns.boxplot(x='week', y='speed', hue='cell_type', data=chip_means,
                  palette=cell_type_colors, ax=ax2)
        
        if show_individual:
            sns.stripplot(x='week', y='speed', hue='cell_type', data=chip_means,
                        palette=cell_type_colors, dodge=True, 
                        alpha=0.5, size=5, ax=ax2)
    else:
        sns.boxplot(x='week', y='speed', hue='cell_type', data=df,
                  palette=cell_type_colors, ax=ax2)
        
        if show_individual:
            sns.stripplot(x='week', y='speed', hue='cell_type', data=df,
                        palette=cell_type_colors, dodge=True, 
                        alpha=0.5, size=5, ax=ax2)
    
    ax2.set_xlabel('Days In Vitro (DIV)', fontsize=fontsize + 2)
    ax2.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
    ax2.set_title('Conduction Speed Distribution by DIV and Cell Type', fontsize=fontsize + 4, pad=20)
    
    # Adjust legend
    handles, labels = ax2.get_legend_handles_labels()
    n_cell_types = len(cell_types)
    ax2.legend(handles[:n_cell_types], labels[:n_cell_types], 
             fontsize=fontsize, title="Cell Type", title_fontsize=fontsize)
    
    # Create statistical analysis and visualization
    fig3 = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig3)
    
    # Plot 1: Mean speed vs DIV with regression lines, using original DIVs for regression
    ax3_1 = fig3.add_subplot(gs[0, 0])
    
    # Plot regression line for each cell type using original DIV values (not groups)
    for cell_type in cell_types:
        if group_by_chip:
            # Create a temporary DataFrame with actual DIV values for regression
            temp_data = df[df['cell_type'] == cell_type].copy()
            temp_data = temp_data.groupby(['div', 'chip_id'])['speed'].mean().reset_index()
        else:
            temp_data = df[df['cell_type'] == cell_type].copy()
        
        if len(temp_data) > 1:
            sns.regplot(x='div', y='speed', data=temp_data,
                      scatter=True, ci=95, 
                      scatter_kws={'alpha': 0.4, 's': 30},
                      line_kws={'linewidth': 2},
                      color=cell_type_colors[cell_type],
                      label=cell_type, ax=ax3_1)
    
    ax3_1.set_xlabel('DIV', fontsize=fontsize)
    ax3_1.set_ylabel('Speed (m/s)', fontsize=fontsize)
    ax3_1.set_title('Regression Analysis', fontsize=fontsize + 2)
    ax3_1.legend(fontsize=fontsize - 2)
    
    # Plot 2: Bar chart of mean speed by cell type
    ax3_2 = fig3.add_subplot(gs[0, 1])
    
    # Calculate mean and standard error by cell type
    if group_by_chip:
        # First average by chip
        chip_means_by_ct = chip_means.groupby(['cell_type', 'chip_id'])['speed'].mean().reset_index()
        # Then calculate statistics across chips
        cell_type_stats = chip_means_by_ct.groupby('cell_type')['speed'].agg(['mean', 'std', 'count']).reset_index()
    else:
        cell_type_stats = df.groupby('cell_type')['speed'].agg(['mean', 'std', 'count']).reset_index()
    
    cell_type_stats['se'] = cell_type_stats['std'] / np.sqrt(cell_type_stats['count'])
    
    # Sort by mean speed
    cell_type_stats = cell_type_stats.sort_values('mean', ascending=False)
    
    # Plot bars
    bars = ax3_2.bar(cell_type_stats['cell_type'], cell_type_stats['mean'], 
                  yerr=cell_type_stats['se'], capsize=5,
                  color=[cell_type_colors[ct] for ct in cell_type_stats['cell_type']])
    
    # Add count labels
    for i, bar in enumerate(bars):
        ax3_2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cell_type_stats['se'].iloc[i] + 0.02,
                 f"n={cell_type_stats['count'].iloc[i]}", 
                 ha='center', va='bottom', fontsize=fontsize - 2)
    
    ax3_2.set_xlabel('Cell Type', fontsize=fontsize)
    ax3_2.set_ylabel('Mean Speed (m/s)', fontsize=fontsize)
    ax3_2.set_title('Average Speed by Cell Type', fontsize=fontsize + 2)
    plt.setp(ax3_2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Two-way ANOVA test results (cell type × DIV group)
    ax3_3 = fig3.add_subplot(gs[1, 0])
    
    # Prepare data for two-way ANOVA (if statsmodels is available)
    data_for_anova = chip_means if group_by_chip else df
    
    # First, check if we have enough data for a meaningful ANOVA
    cell_types_with_enough_data = [ct for ct in cell_types 
                                 if len(data_for_anova[data_for_anova['cell_type'] == ct]) >= 3]
    
    if len(cell_types_with_enough_data) >= 2:
        anova_data = [data_for_anova[data_for_anova['cell_type'] == ct]['speed'].values 
                   for ct in cell_types_with_enough_data]
        
        f_val, p_val = stats.f_oneway(*anova_data)
        
        # Create a visual representation of ANOVA results
        ax3_3.axis('off')
        
        anova_text = (f"One-way ANOVA Results:\n"
                    f"F-value: {f_val:.3f}\n"
                    f"p-value: {p_val:.4f}\n\n")
        
        if p_val < 0.05:
            anova_text += "There is a significant difference\nbetween cell types (p < 0.05)."
        else:
            anova_text += "No significant difference\nbetween cell types (p ≥ 0.05)."
        
        ax3_3.text(0.5, 0.5, anova_text, 
                 ha='center', va='center', 
                 fontsize=fontsize,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    else:
        ax3_3.text(0.5, 0.5, "Not enough data for\nANOVA analysis\n(need at least 2 cell types\nwith 3+ samples each)", 
                 ha='center', va='center', fontsize=fontsize)
    
    # Plot 4: DIV correlation analysis
    ax3_4 = fig3.add_subplot(gs[1, 1])
    
    # Calculate correlation between DIV and speed for each cell type (using original DIV values)
    corr_data = []
    
    for cell_type in cell_types:
        if group_by_chip:
            # Use original DIV values for more accurate correlation
            cell_data = df[df['cell_type'] == cell_type].copy()
            cell_data = cell_data.groupby(['div', 'chip_id'])['speed'].mean().reset_index()
        else:
            cell_data = df[df['cell_type'] == cell_type].copy()
        
        if len(cell_data) > 5:  # Require at least 5 data points for correlation
            corr, p = stats.pearsonr(cell_data['div'], cell_data['speed'])
            corr_data.append({
                'Cell Type': cell_type,
                'Correlation': corr,
                'p-value': p,
                'Significant': p < 0.05
            })
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        
        # Sort by correlation strength
        corr_df = corr_df.sort_values('Correlation', ascending=False)
        
        # Create bar colors based on significance
        bar_colors = [cell_type_colors[ct] if sig else 'lightgray' 
                   for ct, sig in zip(corr_df['Cell Type'], corr_df['Significant'])]
        
        # Plot bars
        bars = ax3_4.bar(corr_df['Cell Type'], corr_df['Correlation'], color=bar_colors)
        
        # Add significance markers
        for i, bar in enumerate(bars):
            if corr_df['Significant'].iloc[i]:
                ax3_4.text(bar.get_x() + bar.get_width()/2, 
                         bar.get_height() + 0.02 if bar.get_height() >= 0 else bar.get_height() - 0.08,
                         '*', ha='center', va='bottom', fontsize=fontsize + 4)
        
        ax3_4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3_4.set_xlabel('Cell Type', fontsize=fontsize)
        ax3_4.set_ylabel('Pearson Correlation\nwith DIV', fontsize=fontsize)
        ax3_4.set_title('Speed vs DIV Correlation', fontsize=fontsize + 2)
        plt.setp(ax3_4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax3_4.text(0.5, 0.5, "Not enough data for\ncorrelation analysis\n(need at least 5 data points\nper cell type)", 
                 ha='center', va='center', fontsize=fontsize)
    
    # Additional figure: Speed vs Chip ID
    if len(df['chip_id'].unique()) > 1:
        fig4, ax4 = plt.subplots(figsize=figsize)
        
        # Use different markers for different DIVs
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        div_markers = {div: markers[i % len(markers)] for i, div in enumerate(divs)}
        
        # Create a scatter plot for each cell type and DIV combination
        for cell_type in cell_types:
            for div in divs:
                subset = df[(df['cell_type'] == cell_type) & (df['div'] == div)]
                
                if not subset.empty:
                    # Calculate mean speed per chip
                    chip_speeds = subset.groupby('chip_id')['speed'].mean().reset_index()
                    
                    ax4.scatter(chip_speeds['chip_id'], chip_speeds['speed'],
                             marker=div_markers[div], s=100,
                             color=cell_type_colors[cell_type],
                             alpha=0.7,
                             label=f"{cell_type}, DIV {div}")
        
        ax4.set_xlabel('Chip ID', fontsize=fontsize + 2)
        ax4.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
        ax4.set_title('Conduction Speed by Chip ID', fontsize=fontsize + 4, pad=20)
        
        # Create a legend that doesn't duplicate entries
        handles, labels = ax4.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax4.legend(by_label.values(), by_label.keys(), 
                 fontsize=fontsize - 2, loc='best')
        
        # Set better x-ticks
        ax4.set_xticks(sorted(df['chip_id'].unique()))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"Speed_by_chip.png"))
        plt.savefig(os.path.join(save_path, f"Speed_by_chip.pdf"), format='pdf', dpi=300)
        
        figures.append((fig4, ax4))
    
    # Adjust layouts and save figures
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    
    plt.figure(fig1.number)
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_line.png"))
    plt.savefig(os.path.join(save_path, f"Sspeed_vs_div_line.pdf"), format='pdf', dpi=300)
    
    plt.figure(fig2.number)
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_box.png"))
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_box.pdf"), format='pdf', dpi=300)
    
    plt.figure(fig3.number)
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_stats.png"))
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_stats.pdf"), format='pdf', dpi=300)
    
    figures = [(fig1, ax1), (fig2, ax2), (fig3, (ax3_1, ax3_2, ax3_3, ax3_4))]
    
    # Optional: Create plots for each DIV group separately
    for div_range in div_ranges:
        div_group_name = div_range['name']
        div_data = df[df['week'] == div_group_name]
        
        if len(div_data) > 0:
            fig_div, ax_div = plt.subplots(figsize=(10, 8))
            
            # Create violin plot for this DIV
            if group_by_chip:
                # Aggregate by chip first
                chip_div_data = div_data.groupby(['cell_type', 'chip_id'])['speed'].mean().reset_index()
                sns.violinplot(x='cell_type', y='speed', data=chip_div_data,
                             palette=cell_type_colors, ax=ax_div)
                
                if show_individual:
                    sns.stripplot(x='cell_type', y='speed', data=chip_div_data,
                                color='black', alpha=0.5, size=5, jitter=True, ax=ax_div)
            else:
                sns.violinplot(x='cell_type', y='speed', data=div_data,
                             palette=cell_type_colors, ax=ax_div)
                
                if show_individual:
                    sns.stripplot(x='cell_type', y='speed', data=div_data,
                                color='black', alpha=0.5, size=5, jitter=True, ax=ax_div)
            
            ax_div.set_xlabel('Cell Type', fontsize=fontsize + 2)
            ax_div.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
            ax_div.set_title(f'Speed Distribution by Cell Type at DIV {div}', 
                           fontsize=fontsize + 4, pad=20)
            
            # Add count annotation
            for i, cell_type in enumerate(div_data['cell_type'].unique()):
                count = len(div_data[div_data['cell_type'] == cell_type])
                ax_div.text(i, div_data['speed'].max() * 1.05, f"n={count}", 
                         ha='center', fontsize=fontsize - 2)
            
            plt.setp(ax_div.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(os.path.join(save_path, f"Div{div}_by_cell_type.png"))
            plt.savefig(os.path.join(save_path, f"Div{div}_by_cell_type.pdf"), 
                       format='pdf', dpi=300)
            
            figures.append((fig_div, ax_div))
    
    # Additional: Direction visualization
    fig_dir, ax_dir = plt.subplots(figsize=(12, 10))
    
    # Extract x and y components of direction vectors
    df['dir_x'] = df['direction'].apply(lambda d: d[0] if isinstance(d, (list, np.ndarray)) else np.nan)
    df['dir_y'] = df['direction'].apply(lambda d: d[1] if isinstance(d, (list, np.ndarray)) else np.nan)
    
    # Plot direction vectors colored by cell type
    for cell_type in cell_types:
        cell_data = df[df['cell_type'] == cell_type]
        
        # Filter out any rows with NaN directions
        cell_data = cell_data.dropna(subset=['dir_x', 'dir_y'])
        
        if not cell_data.empty:
            ax_dir.quiver(np.zeros(len(cell_data)), np.zeros(len(cell_data)),
                       cell_data['dir_x'], cell_data['dir_y'],
                       color=cell_type_colors[cell_type], 
                       label=cell_type,
                       alpha=0.7, scale=2, width=0.005)
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax_dir.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Set equal aspect ratio
    ax_dir.set_aspect('equal')
    
    # Add grid and axes
    ax_dir.grid(True, alpha=0.3)
    ax_dir.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_dir.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    ax_dir.set_xlim(-1.2, 1.2)
    ax_dir.set_ylim(-1.2, 1.2)
    
    ax_dir.set_xlabel('X Direction', fontsize=fontsize + 2)
    ax_dir.set_ylabel('Y Direction', fontsize=fontsize + 2)
    ax_dir.set_title('Propagation Directions by Cell Type', fontsize=fontsize + 4, pad=20)
    
    ax_dir.legend(fontsize=fontsize)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Direction_by_cell_type.png"))
    plt.savefig(os.path.join(save_path, f"Direction_by_cell_type.pdf"), format='pdf', dpi=300)
    
    figures.append((fig_dir, ax_dir))
    
    return figures


def plot_simple_speed_div_analysis(save_path, data_df, 
                                   selected_div_group=None, 
                                   figsize=(12, 10), fontsize=12):
    """
    Simple function to plot conduction speed vs DIV group for each cell type
    
    Parameters:
    - save_path: directory to save the figures
    - filename: base filename for saved plots
    - data_df: pandas DataFrame with columns:
        - unit: unit ID
        - speed_ms-1: conduction speed in m/s (or 'speed')
        - div: days in vitro
        - cell_type: type of the cell/unit
    - selected_div_group: specific DIV group for violin plot (e.g., 'DIV11-15')
                        If None, the most populated DIV group will be used
    - figsize: tuple for figure size
    - fontsize: base font size for plots
    
    Returns:
    - fig1, ax1: Line plot figure and axis
    - fig2, ax2: Violin plot figure and axis for selected DIV group
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize + 2,
        'axes.titlesize': fontsize + 4,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })
    
    # Make a copy to avoid modifying the original dataframe
    df = data_df.copy()
    
    # Ensure the speed column is named consistently
    if 'speed_ms-1' in df.columns and 'speed' not in df.columns:
        df.rename(columns={'speed_ms-1': 'speed'}, inplace=True)
    
    # Create DIV ranges
    div_ranges = [
        {'name': 'Week 1', 'range': (0, 6)},
        {'name': 'Week 2', 'range': (7, 13)},
        {'name': 'Week 3', 'range': (14, 20)},
        {'name': 'Week 4', 'range': (21, 27)},
        {'name': 'Week 5', 'range': (28, 34)},

    ]
    
    # Define the chronological order of DIV groups
    div_group_order = [r['name'] for r in div_ranges]
    
    # Add a 'div_group' column to categorize DIVs into ranges
    def assign_div_group(div):
        for range_info in div_ranges:
            min_div, max_div = range_info['range']
            if min_div <= div <= max_div:
                return range_info['name']
        return f'DIV{div}'  # For any DIVs outside the defined ranges
    
    df['week'] = df['div'].apply(assign_div_group)
    
    # Get all unique values and sort them by the defined order
    available_div_groups = set(df['week'].unique())
    div_groups = [group for group in div_group_order if group in available_div_groups]
    
    # Get unique cell types
    cell_types = sorted(df['cell_type'].unique())
    
    # Create custom color palette for specific cell types
    custom_colors = {
        'CTRL': '#2ca02c',      # Green
        'TO1': '#1f77b4',       # Darker blue
        'TO1-B5': '#7fbfff',    # Lighter blue
        'TO2': '#e377c2',       # Darker pink
        'TO2-B5': '#fcb0d9'     # Lighter pink
    }
    
    # Create color palette for cell types using custom colors when available
    cell_type_colors = {}
    for cell_type in cell_types:
        if cell_type in custom_colors:
            cell_type_colors[cell_type] = custom_colors[cell_type]
        else:
            # For any cell types not explicitly defined, use default colors
            cell_type_colors[cell_type] = '#999999'  # Gray for any undefined types
    
    # Figure 1: Line plot with shadowed error region
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    # For each cell type, create a line plot with shaded error region
    for cell_type in cell_types:
        # Get data for this cell type
        cell_data = df[df['cell_type'] == cell_type]
        
        if len(cell_data) == 0:
            continue
        
        # Group by DIV group and compute statistics
        grouped = cell_data.groupby('week')['speed'].agg(['mean', 'std', 'count']).reset_index()
        
        if len(grouped) == 0:
            continue
            
        # Compute standard error
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
        
        # Sort by DIV group order using the predefined order
        # Create a mapping for sorting
        order_map = {div: i for i, div in enumerate(div_group_order)}
        # Sort grouped data according to the predefined div_group_order
        grouped['order'] = grouped['week'].map(lambda x: order_map.get(x, 999))  # Default high value for any DIV groups not in the list
        grouped = grouped.sort_values('order')
        grouped = grouped.drop('order', axis=1)
        
        # X values for plotting - create consistent positions based on div_group_order
        x_positions = {dg: i for i, dg in enumerate(div_group_order) if dg in div_groups}
        x_values = [x_positions[dg] for dg in grouped['week']]
        
        # Plot line with explicit color to ensure consistency
        line = ax1.plot(x_values, grouped['mean'], 'o-', 
                linewidth=2, markersize=8, 
                color=cell_type_colors[cell_type], 
                label=f"{cell_type} (n={len(cell_data)})")
        
        # Add shaded error region with explicitly matching color
        ax1.fill_between(x_values, 
                       grouped['mean'] - grouped['se'], 
                       grouped['mean'] + grouped['se'], 
                       alpha=0.2, color=cell_type_colors[cell_type])
    
    # Set x-ticks and labels - ensure only DIV groups in the data are shown
    present_div_groups = [dg for dg in div_group_order if dg in available_div_groups]
    ax1.set_xticks(range(len(present_div_groups)))
    ax1.set_xticklabels(present_div_groups, rotation=45, ha='right')
    
    # Set labels and title
    ax1.set_xlabel('Days In Vitro (DIV)', fontsize=fontsize + 2)
    ax1.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
    ax1.set_title('Conduction Speed vs DIV Group by Cell Type', fontsize=fontsize + 4, pad=20)
    
    # Add legend
    ax1.legend(fontsize=fontsize, loc='best')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Set y-axis to start at 0
    ax1.set_ylim(bottom=0)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_line.png"), dpi=300)
    plt.savefig(os.path.join(save_path, f"Speed_vs_div_line.pdf"), format='pdf', dpi=300)
    
    # Figure 2: Violin plot for selected DIV group
    if selected_div_group is None:
        # If no DIV group is selected, use the one with the most data points
        div_counts = df['week'].value_counts()
        if not div_counts.empty:
            selected_div_group = div_counts.index[0]
        else:
            print("No data available for any DIV group")
            return (fig1, ax1), (None, None)
    
    # Filter data for the selected DIV group
    div_group_data = df[df['week'] == selected_div_group]
    
    if len(div_group_data) > 0:
        fig2, ax2 = plt.subplots(figsize=figsize)
        
        # Get available cell types in this DIV group
        available_cell_types = sorted(div_group_data['cell_type'].unique())
        
        # Create violin plot using custom implementation to ensure consistent colors
        violin_data = []
        positions = []
        colors = []
        
        # Prepare data for violinplot
        for i, cell_type in enumerate(cell_types):
            if cell_type in available_cell_types:
                ct_data = div_group_data[div_group_data['cell_type'] == cell_type]['speed'].values
                if len(ct_data) > 0:
                    violin_data.append(ct_data)
                    positions.append(i)
                    colors.append(cell_type_colors[cell_type])
        
        # Create violin plot with explicit colors
        if violin_data:
            violins = ax2.violinplot(violin_data, positions=positions, showmeans=True, 
                                   showmedians=True, showextrema=True)
            
            
            # Customize violin colors
            for i, pc in enumerate(violins['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            
            # Add individual points
            for i, (pos, data) in enumerate(zip(positions, violin_data)):
                ax2.scatter([pos] * len(data), data, color='black', alpha=0.4, 
                         s=30, zorder=3, edgecolor='none')
        
            # Set x-ticks and labels for available cell types
            ax2.set_xticks(positions)
            ax2.set_xticklabels([cell_types[p] for p in positions], rotation=45, ha='right')
            
            # Set labels and title
            ax2.set_xlabel('Cell Type', fontsize=fontsize + 2)
            ax2.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
            ax2.set_title(f'Speed Distribution by Cell Type at {selected_div_group}', 
                       fontsize=fontsize + 4, pad=20)
            
            # Add count annotations
            for i, (pos, data) in enumerate(zip(positions, violin_data)):
                count = len(data)
                # Find the maximum value for this specific dataset instead of all data
                max_value = np.max(data) if len(data) > 0 else 0
                # Use a fixed offset or calculate a relative offset from the entire dataset
                y_offset = max([np.max(d) for d in violin_data if len(d) > 0], default=0) * 1.05
                ax2.text(pos, y_offset, f"n={count}", 
                       ha='center', fontsize=fontsize - 2)
            
            # Set y-axis to start at 0
            ax2.set_ylim(bottom=0)
        else:
            ax2.text(0.5, 0.5, f"No data available for {selected_div_group}", 
                   ha='center', va='center', fontsize=fontsize)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{selected_div_group.replace('-', 'to')}_violin.png"), dpi=300)
        plt.savefig(os.path.join(save_path, f"{selected_div_group.replace('-', 'to')}_violin.pdf"), format='pdf', dpi=300)
    else:
        print(f"No data available for DIV group {selected_div_group}")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.text(0.5, 0.5, f"No data available for {selected_div_group}", 
               ha='center', va='center', fontsize=fontsize)
        fig2.tight_layout()
    
    return (fig1, ax1), (fig2, ax2)


def plot_statistical_annotations(ax, data_dict):
    """
    Add statistical annotations to the plot
    
    Parameters:
    - ax: matplotlib axis to annotate
    - data_dict: dictionary with cell types as keys and their speed data as values
    """
    import scipy.stats as stats
    
    # Prepare cell types and their data
    cell_types = list(data_dict.keys())
    
    # Find the maximum y value for positioning annotations
    max_y = max(max(data) for data in data_dict.values())
    y_offset = max_y * 1.1
    
    # Significance symbol mapping
    sig_levels = {
        0.0001: '****',
        0.001: '***', 
        0.01: '**', 
        0.05: '*'
    }
    
    # Perform pairwise comparisons
    for i in range(len(cell_types)):
        for j in range(i+1, len(cell_types)):
            type1, type2 = cell_types[i], cell_types[j]
            data1, data2 = data_dict[type1], data_dict[type2]
            
            # Perform t-test
            _, p_value = stats.ttest_ind(data1, data2)
            
            # Determine significance symbol
            sig_symbol = 'n.s.'
            for threshold, symbol in sorted(sig_levels.items()):
                if p_value < threshold:
                    sig_symbol = symbol
                    break
            
            # If significant, add annotation
            if sig_symbol != 'n.s.':
                ax.plot([i, j], [y_offset, y_offset], color='black', linewidth=1)
                ax.text((i+j)/2, y_offset, sig_symbol, 
                        ha='center', va='bottom', fontsize=10)
                
                # Increment y_offset for next potential annotation
                y_offset += max_y * 0.05
    
    # Extend y-axis if needed
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], max(current_ylim[1], y_offset * 1.1))

def plot_simple_speed_div_analysis_w_test(save_path, data_df, 
                               selected_div_group=None, 
                               figsize=(12, 10), fontsize=12):
    """
    Simple function to plot conduction speed vs DIV group for each cell type
    
    Parameters:
    - save_path: directory to save the figures
    - filename: base filename for saved plots
    - data_df: pandas DataFrame with columns:
        - unit: unit ID
        - speed_ms-1: conduction speed in m/s (or 'speed')
        - div: days in vitro
        - cell_type: type of the cell/unit
    - selected_div_group: specific DIV group for violin plot (e.g., 'DIV11-15')
                        If None, the most populated DIV group will be used
    - figsize: tuple for figure size
    - fontsize: base font size for plots
    
    Returns:
    - fig1, ax1: Line plot figure and axis
    - fig2, ax2: Violin plot figure and axis for selected DIV group
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import scipy.stats as stats
    
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize + 2,
        'axes.titlesize': fontsize + 4,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })
    
    # Make a copy to avoid modifying the original dataframe
    df = data_df.copy()
    
    # Ensure the speed column is named consistently
    if 'speed_ms-1' in df.columns and 'speed' not in df.columns:
        df.rename(columns={'speed_ms-1': 'speed'}, inplace=True)
    
    # Create DIV ranges
    div_ranges = [
        {'name': 'Week 1', 'range': (0, 6)},
        {'name': 'Week 2', 'range': (7, 13)},
        {'name': 'Week 3', 'range': (14, 20)},
        {'name': 'Week 4', 'range': (21, 27)},
        {'name': 'Week 5', 'range': (28, 34)},
    ]
    
    # Define the chronological order of DIV groups
    div_group_order = [r['name'] for r in div_ranges]
    
    # Add a 'div_group' column to categorize DIVs into ranges
    def assign_div_group(div):
        for range_info in div_ranges:
            min_div, max_div = range_info['range']
            if min_div <= div <= max_div:
                return range_info['name']
        return f'DIV{div}'  # For any DIVs outside the defined ranges
    
    df['week'] = df['div'].apply(assign_div_group)
    
    # Get all unique values and sort them by the defined order
    available_div_groups = set(df['week'].unique())
    div_groups = [group for group in div_group_order if group in available_div_groups]
    
    # Get unique cell types
    cell_types = sorted(df['cell_type'].unique())
    
    # Custom color palette
    custom_colors = {
        'CTRL': '#2ca02c',      # Green
        'TO1': '#1f77b4',       # Darker blue
        'TO1-B5': '#7fbfff',    # Lighter blue
        'TO2': '#e377c2',       # Darker pink
        'TO2-B5': '#fcb0d9'     # Lighter pink
    }
    
    # Create color palette for cell types
    cell_type_colors = {}
    for cell_type in cell_types:
        if cell_type in custom_colors:
            cell_type_colors[cell_type] = custom_colors[cell_type]
        else:
            cell_type_colors[cell_type] = '#999999'  # Gray for any undefined types
    
    # [Line plot code would typically go here, but omitted for brevity]
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    # Figure 2: Violin plot for selected DIV group
    if selected_div_group is None:
        # If no DIV group is selected, use the one with the most data points
        div_counts = df['week'].value_counts()
        if not div_counts.empty:
            selected_div_group = div_counts.index[0]
        else:
            print("No data available for any DIV group")
            return (fig1, ax1), (None, None)
    
    # Filter data for the selected DIV group
    div_group_data = df[df['week'] == selected_div_group]
    
    if len(div_group_data) > 0:
        fig2, ax2 = plt.subplots(figsize=figsize)
        
        # Prepare data for violinplot
        violin_data = []
        positions = []
        colors = []
        data_dict = {}
        
        # Collect data for each cell type
        for i, cell_type in enumerate(cell_types):
            ct_data = div_group_data[div_group_data['cell_type'] == cell_type]['speed'].values
            if len(ct_data) > 0:
                violin_data.append(ct_data)
                positions.append(i)
                colors.append(cell_type_colors[cell_type])
                data_dict[cell_type] = ct_data
        
        # Create violin plot
        if violin_data:
            violins = ax2.violinplot(violin_data, positions=positions, showmeans=True, 
                                   showmedians=True, showextrema=True)
            
            # Customize violin colors
            for i, pc in enumerate(violins['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            
            # Add individual points
            for i, (pos, data) in enumerate(zip(positions, violin_data)):
                ax2.scatter([pos] * len(data), data, color='black', alpha=0.4, 
                         s=30, zorder=3, edgecolor='none')
            
            # Set x-ticks and labels
            ax2.set_xticks(positions)
            ax2.set_xticklabels([cell_types[p] for p in positions], rotation=45, ha='right')
            
            # Labels and title
            ax2.set_xlabel('Cell Type', fontsize=fontsize + 2)
            ax2.set_ylabel('Conduction Speed (m/s)', fontsize=fontsize + 2)
            ax2.set_title(f'Speed Distribution by Cell Type at {selected_div_group}', 
                       fontsize=fontsize + 4, pad=20)
            
            # Add count annotations
            max_speed = 0
            for i, (pos, data) in enumerate(zip(positions, violin_data)):
                count = len(data)
                max_speed = max(max_speed, np.max(data))
                ax2.text(pos, max_speed * 1.05, f"n={count}", 
                       ha='center', fontsize=fontsize - 2)
            
            # Add statistical annotations
            plot_statistical_annotations(ax2, data_dict)
            
            # Set y-axis to start at 0
            ax2.set_ylim(bottom=0)
        else:
            ax2.text(0.5, 0.5, f"No data available for {selected_div_group}", 
                   ha='center', va='center', fontsize=fontsize)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{selected_div_group.replace('-', 'to')}_violin.png"), dpi=300)
        plt.savefig(os.path.join(save_path, f"{selected_div_group.replace('-', 'to')}_violin.pdf"), format='pdf', dpi=300)
    else:
        print(f"No data available for DIV group {selected_div_group}")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.text(0.5, 0.5, f"No data available for {selected_div_group}", 
               ha='center', va='center', fontsize=fontsize)
        fig2.tight_layout()
    
    return (fig1, ax1), (fig2, ax2)


def visualize_propagation_by_recording(conduction_speeds_path, save_path, filename_filter=None):
    """
    Load saved conduction speed results and visualize them by recording
    
    Parameters:
    - conduction_speeds_path: directory containing saved results
    - save_path: directory to save the figures
    - filename_filter: optional filter to only process specific recordings
    """
    import os
    import pickle
    import numpy as np
    
    # Load the combined results
    with open(os.path.join(conduction_speeds_path, 'Conduction_speeds_all.pkl'), 'rb') as f:
        all_results = pickle.load(f)

    
    # Group results by filename
    results_by_recording = {}
    for result in all_results:
        filename = result['filename']
        if filename_filter and filename not in filename_filter:
            continue
            
        if filename not in results_by_recording:
            results_by_recording[filename] = []
        results_by_recording[filename].append(result)
    
    # Process each recording
    for filename, results in results_by_recording.items():
        print(f"Processing visualization for {filename}")
        
        # Load the recording-specific metadata
        try:
            with open(os.path.join(conduction_speeds_path, f'metadata_{filename}.pkl'), 'rb') as f:
                metadata = pickle.load(f)
                
            # Extract required data for visualization
            sampling_rate = metadata['sampling_rate']
            probe_locations = metadata['probe_locations']
            templates = metadata['templates']
            unit_ids = metadata['unit_ids']
            
            # Get unit IDs from the results
            #unit_ids = [r['unit'] for r in results]
            
            # Create visualization for this recording
            visualize_unit_propagation_V2(
                save_path=save_path,
                filename=f"propagation_{filename}",
                results=results,
                templates=templates,
                probe_locations=probe_locations,
                sampling_rate=sampling_rate,
                unit_ids=unit_ids
            )
            
            print(f"Visualization completed for {filename}")
            
        except Exception as e:
            print(f"Error visualizing {filename}: {e}")
    
    print("All visualizations completed")


def create_summary_visualization(conduction_speeds_path, save_path, group_by='chip_id'):
    """
    Create summary visualizations of conduction speeds grouped by a specific attribute
    
    Parameters:
    - conduction_speeds_path: directory containing saved results
    - save_path: directory to save the figures
    - group_by: attribute to group results by ('chip_id', 'div', 'area', or 'cell_type')
    """
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load the combined results
    with open(os.path.join(conduction_speeds_path, 'Conduction_speeds_all.pkl'), 'rb') as f:
        all_results = pickle.load(f)
    
    # Create a dataframe for easier analysis
    import pandas as pd
    df = pd.DataFrame(all_results)

    # Create violin plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Violin plot shows the distribution of speeds directly
    ax = sns.violinplot(x=group_by, y='speed_ms-1', data=df, inner='quartile', cut=0)

    # Add a swarm plot on top to show individual data points
    sns.swarmplot(x=group_by, y='speed_ms-1', data=df, color='black', alpha=0.5, size=3)

    # Add count labels and group means
    for i, (name, group) in enumerate(df.groupby(group_by)):
        count = len(group)
        mean = group['speed_ms-1'].mean()
        ax.text(i, df['speed_ms-1'].min(), f'n={count}', ha='center', fontsize=10)
        ax.plot(i, mean, 'ro', ms=8)  # Red dot for mean
        ax.text(i, mean, f'μ={mean:.2f}', ha='center', va='bottom', fontsize=9, color='red')

    plt.title(f'Conduction Speed Distribution by {group_by}', fontsize=14)
    plt.xlabel(group_by, fontsize=12)
    plt.ylabel('Conduction Speed (m/s)', fontsize=12)
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(save_path, f'summary_speed_by_{group_by}.png'))
    plt.savefig(os.path.join(save_path, f'summary_speed_by_{group_by}.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Create direction visualization
    plt.figure(figsize=(12, 12))

    # Group by the selected attribute
    groups = df.groupby(group_by)

    # First, find global min and max speeds for consistent color scaling
    all_speeds_global = df['speed_ms-1'].values
    vmin = np.min(all_speeds_global)
    vmax = np.max(all_speeds_global)

    # Create a polar scatter plot for directions
    for i, (name, group) in enumerate(groups):
        # We need to retrieve probe locations for each recording
        all_distances = []
        all_angles = []
        all_speeds = []
        
        # Group by filename to get recording-specific data
        for filename, file_group in group.groupby('filename'):
            try:
                # Load the recording-specific metadata to get probe locations
                with open(os.path.join(conduction_speeds_path, f'metadata_{filename}.pkl'), 'rb') as f:
                    metadata = pickle.load(f)
                    
                probe_locations = metadata['probe_locations']
                
                # For each unit in this recording
                for _, unit_data in file_group.iterrows():
                    # Get the direction vector
                    direction = unit_data['direction']
                    speed = unit_data['speed_ms-1']
                    
                    # Convert direction vector to angle
                    angle = np.arctan2(direction[1], direction[0])
                    
                    # Find the center of the probe array
                    center_x = np.mean(probe_locations[:, 0])
                    center_y = np.mean(probe_locations[:, 1])
                    
                    # Find the typical inter-electrode distance by calculating the average
                    # distance between adjacent electrodes
                    distances = []
                    for j in range(len(probe_locations)):
                        for k in range(j+1, len(probe_locations)):
                            dist = np.sqrt((probe_locations[j, 0] - probe_locations[k, 0])**2 + 
                                        (probe_locations[j, 1] - probe_locations[k, 1])**2)
                            distances.append(dist)
                    
                    # Get the minimum non-zero distance as the typical electrode spacing
                    min_distance = np.min([d for d in distances if d > 0])
                    
                    # Calculate the typical propagation distance (in μm)
                    vector_length = np.sqrt(direction[0]**2 + direction[1]**2)
                    propagation_distance = vector_length * min_distance
                    
                    all_distances.append(propagation_distance)
                    all_angles.append(angle)
                    all_speeds.append(speed)
                    
            except Exception as e:
                print(f"Error processing {filename} for direction plot: {e}")
                continue
        
        # Plot in polar coordinates
        ax = plt.subplot(2, 2, i+1, projection='polar')
        
        if all_distances:  # Check if we have data to plot
            # Use consistent vmin and vmax for color mapping across all subplots
            sc = ax.scatter(all_angles, all_distances, c=all_speeds, cmap='viridis', 
                        alpha=0.7, s=50, label=f'{name} (n={len(all_angles)})',
                        vmin=vmin, vmax=vmax)  # Set consistent color scale
            
            # Add colorbar to show speed mapping
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Speed (m/s)')
            
            # Calculate max distance for setting rticks
            max_distance = max(all_distances) if all_distances else 100
            # Round up to nearest 50 μm
            max_tick = np.ceil(max_distance / 50) * 50
            
            # Set radial ticks with reasonable spacing
            r_ticks = np.arange(0, max_tick + 1, 50)  # 0, 50, 100, ... μm
            ax.set_rticks(r_ticks)
            
            # Add custom labels for the radial ticks
            for tick in r_ticks:
                if tick > 0:  # Skip the origin
                    ax.text(0, tick, f'{int(tick)} μm', ha='center', va='bottom', 
                        transform=ax.transData, fontsize=8)
        
        ax.set_title(f'{group_by}={name}')
        ax.set_theta_zero_location('N')  # North = 0 degrees
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_rlabel_position(0)
        ax.grid(True)

    # Add a colorbar for the entire figure
    # This is optional - you can keep this or remove it if you prefer the individual colorbars
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = plt.axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Speed (m/s)', fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for the colorbar
    plt.savefig(os.path.join(save_path, f'direction_distance_by_{group_by}.png'))
    plt.savefig(os.path.join(save_path, f'direction_distance_by_{group_by}.pdf'), format='pdf', dpi=300)
    plt.close()
        
    return df  # Return the dataframe for further analysis


def create_summary_visualization_V2(conduction_speeds_path, save_path, group_by = 'chip_id'):
    """
    Create summary visualizations of conduction speeds grouped by a specific attribute
    
    Parameters:
    - conduction_speeds_path: directory containing saved results
    - save_path: directory to save the figures
    - group_by: attribute to group results by ('chip_id', 'div', 'area', or 'cell_type')
    """
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load the combined results
    with open(os.path.join(conduction_speeds_path, 'Conduction_speeds_all.pkl'), 'rb') as f:
        all_results = pickle.load(f)
    
    # Create a dataframe for easier analysis
    import pandas as pd
    df = pd.DataFrame(all_results)

    # Create violin plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Violin plot shows the distribution of speeds directly
    ax = sns.violinplot(x=group_by, y='speed_ms-1', data=df, inner='quartile', cut=0)

    # Add a swarm plot on top to show individual data points
    sns.swarmplot(x=group_by, y='speed_ms-1', data=df, color='black', alpha=0.5, size=3)

    # Add count labels and group means
    for i, (name, group) in enumerate(df.groupby(group_by)):
        count = len(group)
        mean = group['speed_ms-1'].mean()
        ax.text(i, df['speed_ms-1'].min(), f'n={count}', ha='center', fontsize=10)
        ax.plot(i, mean, 'ro', ms=8)  # Red dot for mean
        ax.text(i, mean, f'μ={mean:.2f}', ha='center', va='bottom', fontsize=9, color='red')

    plt.title(f'Conduction Speed Distribution by {group_by}', fontsize=14)
    plt.xlabel(group_by, fontsize=12)
    plt.ylabel('Conduction Speed (m/s)', fontsize=12)
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(save_path, f'summary_speed_by_{group_by}.png'))
    plt.savefig(os.path.join(save_path, f'summary_speed_by_{group_by}.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Create direction visualization
    plt.figure(figsize=(12, 12))

    # Group by the selected attribute
    groups = df.groupby(group_by)

    # Set fixed speed range from 0 to 3 m/s
    vmin = 0  # Fixed minimum speed
    vmax = 3  # Fixed maximum speed

    # Create a polar scatter plot for directions
    for i, (name, group) in enumerate(groups):
        # Extract direction vectors and speeds
        directions = group['direction'].values
        speeds = group['speed_ms-1'].values
        
        # Convert direction vectors to angles and magnitudes
        angles = np.arctan2([v[1] for v in directions], [v[0] for v in directions])
        magnitudes = np.array([np.sqrt(v[0]**2 + v[1]**2) for v in directions])
        
        # Plot in polar coordinates
        ax = plt.subplot(2, 2, i+1, projection='polar')
        
        if len(angles) > 0:  # Check if we have data to plot
            sc = ax.scatter(angles, magnitudes, c=speeds, cmap='viridis', 
                        alpha=0.7, s=50, label=f'{name} (n={len(angles)})',
                        vmin=vmin, vmax=vmax)  # Fixed color scale from 0 to 3 m/s
            
            # Add colorbar to show speed mapping
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Speed (m/s)')
        
        ax.set_title(f'{group_by}={name}')
        ax.set_theta_zero_location('N')  # North = 0 degrees
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_rlabel_position(0)
        
        # Set reasonable radial ticks (0 to 1 range for vector magnitudes)
        ax.set_rticks([0, 0.25, 0.5, 0.75, 1.0])
        
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'direction_by_{group_by}.png'))
    plt.savefig(os.path.join(save_path, f'direction_by_{group_by}.pdf'), format='pdf', dpi=300)
    plt.close()
        
    return df  # Return the dataframe for further analysis
