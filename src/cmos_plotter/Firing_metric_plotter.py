import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from collections import defaultdict

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

def calculate_firing_metrics(spike_times, time_unit='ms'):
    """Calculate various firing metrics from spike times.
    
    Args:
        spike_times: Array of spike times
        time_unit: Unit of time for spike_times ('ms' for milliseconds, 's' for seconds)
    """
    if len(spike_times) < 2:
        return {
            'firing_rate': 0.0,
            'isi_mean': 0.0,
            'isi_median': 0.0,
            'isi_std': 0.0,
            'isi_cv': 0.0,  # Coefficient of variation
            'burst_index': 0.0,
            'spike_count': len(spike_times)
        }
    
    # Sort spike times to ensure correct calculations
    spike_times = np.sort(spike_times)
    
    # Calculate recording duration
    recording_duration = spike_times[-1] - spike_times[0]
    if recording_duration == 0:
        recording_duration = 1  # Avoid division by zero
    
    # Calculate firing rate (Hz)
    if time_unit == 'ms':
        # If times are in ms, convert duration to seconds for Hz calculation
        firing_rate = len(spike_times) / (recording_duration / 1000)
    else:
        # If times are already in seconds
        firing_rate = len(spike_times) / recording_duration
    
    # Calculate inter-spike intervals (ISIs)
    isis = np.diff(spike_times)
    
    # Calculate ISI statistics (keep in original time units)
    isi_mean = np.mean(isis)
    isi_median = np.median(isis)
    isi_std = np.std(isis)
    
    # Calculate coefficient of variation (measure of spike train irregularity)
    isi_cv = isi_std / isi_mean if isi_mean > 0 else 0
    
    # Simple burst index (ratio of mean ISI to median ISI)
    # Values > 1 suggest bursting behavior
    burst_index = isi_mean / isi_median if isi_median > 0 else 0
    
    return {
        'firing_rate': firing_rate,
        'isi_mean': isi_mean,
        'isi_median': isi_median, 
        'isi_std': isi_std,
        'isi_cv': isi_cv,
        'burst_index': burst_index,
        'spike_count': len(spike_times)
    }

def process_spike_data(filepath, time_unit='ms'):
    """Process spike data from a single file and return metrics for both SPIKEMAT_EXTREMUM and SPIKEMAT.
    
    Args:
        filepath: Path to the pickle file
        time_unit: Unit of time for spike times ('ms' for milliseconds, 's' for seconds)
    """
    # Extract metadata from filename
    filename = os.path.basename(filepath)
    filename_add_h5 = filename.split('raw')[0] + 'raw.h5'
    chip_id, area, div = extract_metadata_from_filename(filename)
    
    if not all([chip_id, area, div]):
        print(f"Could not extract metadata from {filename}, skipping...")
        return [], []
    
    # Load pickle file
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return [], []
    
    # Initialize results for both types of data
    extremum_results = []
    spikemat_results = []
    
    

    # Process SPIKEMAT_EXTREMUM if it exists
    if 'SPIKEMAT_EXTREMUM' in data:
        spike_data = data['SPIKEMAT_EXTREMUM']
        
        # Check if spike_data is None or empty
        if spike_data is not None and len(spike_data) > 0:
            # Group spike data by UnitIdx and Electrode
            # Treat units with the same index but different area/electrode as separate units
            unit_electrode_groups = defaultdict(list)
            unit_ids = list(data['UNIT_TO_EL'].keys())
            for spike in spike_data:
                electrode = spike['Electrode']
                unit_idx = spike['UnitIdx']
                spike_time = spike['Spike_Time']
                unit_id = unit_ids[unit_idx]
                # Create a composite key that ensures units are treated separately
                key = (electrode, unit_idx, unit_id)
                unit_electrode_groups[key].append(spike_time)
            
            # Calculate metrics for each unit
            for (electrode, unit_idx, unit_id), spike_times in unit_electrode_groups.items():
                metrics = calculate_firing_metrics(spike_times, time_unit)
                # Create result entry
                result = {
                    'filename': filename_add_h5,
                    'chip_id': chip_id,
                    'area': area,
                    'div': div,
                    'electrode': electrode,
                    'unit_idx': unit_idx,
                    'unit_id': unit_id,
                    'data_type': 'SPIKEMAT_EXTREMUM',
                    **metrics  # Unpack all metrics
                }
                
                extremum_results.append(result)
        else:
            print(f"SPIKEMAT_EXTREMUM found in {filename} but it's empty or None.")
    else:
        print(f"SPIKEMAT_EXTREMUM not found in {filename}, skipping unit-level metrics.")
    
    # Process SPIKEMAT (electrode-level analysis)
    if 'SPIKEMAT' in data:
        spike_data = data['SPIKEMAT']
        
        # Check if spike_data is None or empty
        if spike_data is not None and len(spike_data) > 0:
            # Group spike data by Electrode
            electrode_groups = defaultdict(list)
            for spike in spike_data:
                electrode = spike['Electrode']
                spike_time = spike['Spike_Time']
                electrode_groups[electrode].append(spike_time)
            
            # Calculate metrics for each electrode
            for electrode, spike_times in electrode_groups.items():
                metrics = calculate_firing_metrics(spike_times, time_unit)
                
                # Create result entry
                result = {
                    'chip_id': chip_id,
                    'area': area,
                    'div': div,
                    'electrode': electrode,
                    'unit_idx': None,  # No unit for SPIKEMAT data
                    'data_type': 'SPIKEMAT',
                    **metrics  # Unpack all metrics
                }
                
                spikemat_results.append(result)
        else:
            print(f"SPIKEMAT found in {filename} but it's empty or None.")
    else:
        print(f"SPIKEMAT not found in {filename}, skipping electrode-level metrics.")
    
    return extremum_results, spikemat_results


def create_combined_div_plot(df, plot_prefix, plots_dir):
    """Create a combined subplot figure with all metrics over DIV.
    
    Args:
        df: DataFrame with the metrics
        plot_prefix: Prefix for plot filenames (e.g. "unit" or "electrode")
        plots_dir: Directory to save the plots
    """
    if df.empty:
        print("No data available for plotting.")
        return
    
    # Make sure DIV is treated as numeric
    df['div'] = pd.to_numeric(df['div'])
    
    # Determine if ISI values are in ms or s for labeling
    data_type = df['data_type'].iloc[0] if 'data_type' in df.columns else ""
    time_unit_label = "ms" if df['isi_mean'].median() > 10 else "s"  # Heuristic: if median ISI > 10, likely in ms
    
    # Define metrics to plot
    metrics = [
        ('firing_rate', 'Firing Rate (Hz)'),
        ('isi_mean', f'Mean ISI ({time_unit_label})'),
        ('isi_std', f'ISI Std Dev ({time_unit_label})'),
        ('isi_cv', 'ISI CV'),
        ('burst_index', 'Burst Index')
    ]
    
    # Create figure with subplots - use more balanced aspect ratio
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 5*len(metrics)), sharex=True)
    
    # Determine data type for plot titles
    data_type = df['data_type'].iloc[0] if 'data_type' in df.columns else ""
    title_prefix = f"{data_type} " if data_type else ""
    
    # Plot each metric
    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        
        # Group by DIV and calculate mean and standard error
        try:
            grouped = df.groupby('div')[metric].agg(['mean', 'count', 'std']).reset_index()
            # Calculate standard error of the mean (SEM)
            grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
            
            # Line plot with error bars using SEM
            ax.errorbar(grouped['div'], grouped['mean'], yerr=grouped['sem'], 
                        marker='o', linestyle='-', capsize=5)
            
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Only set x-label on bottom subplot
            if i == len(metrics) - 1:
                ax.set_xlabel('Days In Vitro (DIV)')
        except Exception as e:
            print(f"Error plotting {metric}: {e}")
            ax.text(0.5, 0.5, f"Error plotting {metric}: {str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
    
    # Set overall title
    plt.suptitle(f'{title_prefix}Neural Metrics vs DIV ({plot_prefix})', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
    
    # Save the plot
    filename = os.path.join(plots_dir, f'{plot_prefix}_combined_metrics_by_div.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

def create_specific_div_boxplot(df, div_value, plot_prefix, plots_dir):
    """Create box plots for a specific DIV value.
    
    Args:
        df: DataFrame with the metrics
        div_value: DIV value to filter for
        plot_prefix: Prefix for plot filenames (e.g. "unit" or "electrode")
        plots_dir: Directory to save the plots
    """
    # Filter for the specified DIV
    div_df = df[df['div'] == div_value]
    
    if len(div_df) == 0:
        print(f"No data available for DIV{div_value}.")
        return
    
    # Determine data type for plot titles
    data_type = df['data_type'].iloc[0] if 'data_type' in df.columns else ""
    title_prefix = f"{data_type} " if data_type else ""
    
    # Determine if ISI values are in ms or s for labeling
    time_unit_label = "ms" if df['isi_mean'].median() > 10 else "s"  # Heuristic: if median ISI > 10, likely in ms
    
    # Define metrics to plot
    metrics = [
        ('firing_rate', 'Firing Rate (Hz)'),
        ('isi_mean', f'Mean ISI ({time_unit_label})'),
        ('isi_std', f'ISI Std Dev ({time_unit_label})'),
        ('isi_cv', 'ISI CV'),
        ('burst_index', 'Burst Index')
    ]
    
    # Calculate figure size for better aspect ratio (approximately 4:3 for each subplot)
    n_metrics = len(metrics)
    fig_width = 3 * n_metrics  # 3 inches per subplot width
    fig_height = 4  # Fixed height
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, fig_height))
    
    # Ensure axes is always an array even if n_metrics=1
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        
        try:
            # Create box plot
            sns.boxplot(y=metric, x='area', data=div_df, ax=ax)
            
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel}')
            ax.grid(True, linestyle='--', alpha=0.5, axis='y')
            
            # Rotate x-axis labels if needed
            if len(div_df['area'].unique()) > 4:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add individual points if there aren't too many
            if len(div_df) <= 100:  # Limit to prevent overcrowding
                sns.stripplot(y=metric, x='area', data=div_df, ax=ax, 
                             color='black', alpha=0.5, jitter=True, size=3)
        except Exception as e:
            print(f"Error plotting {metric} boxplot: {e}")
            ax.text(0.5, 0.5, f"Error plotting {metric}: {str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
    
    # Set overall title
    plt.suptitle(f'{title_prefix}Neural Metrics at DIV{div_value} ({plot_prefix})', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save the plot
    filename = os.path.join(plots_dir, f'{plot_prefix}_div{div_value}_boxplots.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

def plot_by_chip_and_area(df, plot_prefix="", title_prefix=""):
    """Create plots showing metrics by chip_id and area."""
    # List of metrics to plot
    metrics = ['firing_rate', 'isi_mean', 'isi_cv', 'burst_index']
    
    # Create separate plots for each DIV
    div_values = sorted(df['div'].unique())
    
    for div in div_values:
        div_df = df[df['div'] == div]
        
        if div_df.empty:
            print(f"No data available for DIV{div}.")
            continue
            
        for metric in metrics:
            try:
                plt.figure(figsize=(12, 6))
                
                # Check if we have enough data to create a pivot table
                if len(div_df['chip_id'].unique()) <= 1 or len(div_df['area'].unique()) <= 1:
                    # Not enough unique chip_ids or areas for a pivot table
                    # Do a simpler plot instead
                    if len(div_df['area'].unique()) > 1:
                        # If we have multiple areas, group by area
                        group_col = 'area'
                        title_add = 'by Area'
                    else:
                        # Otherwise group by chip_id
                        group_col = 'chip_id'
                        title_add = 'by Chip'
                    
                    # Create bar chart
                    sns.barplot(x=group_col, y=metric, data=div_df)
                    plt.title(f'{title_prefix}{metric} at DIV{div} {title_add}')
                else:
                    # We have multiple chips and areas, create a pivot table
                    try:
                        # Group by chip_id and area
                        pivot_data = div_df.pivot_table(
                            index='chip_id', 
                            columns='area', 
                            values=metric, 
                            aggfunc='mean'
                        )
                        
                        # Plot as grouped bar chart
                        pivot_data.plot(kind='bar', ax=plt.gca())
                        plt.title(f'{title_prefix}{metric} at DIV{div} by Chip and Area')
                        plt.legend(title='Area')
                    except ValueError as e:
                        print(f"Error creating pivot table for {metric} at DIV{div}: {e}")
                        # Fallback to regular bar plot
                        sns.barplot(x='chip_id', y=metric, hue='area', data=div_df)
                        plt.title(f'{title_prefix}{metric} at DIV{div} by Chip and Area')
                        plt.legend(title='Area')
                
                # Get appropriate y-label
                if metric == 'firing_rate':
                    ylabel = 'Firing Rate (Hz)'
                elif metric == 'isi_mean':
                    ylabel = 'Mean Inter-Spike Interval (s)'
                elif metric == 'isi_std':
                    ylabel = 'ISI Standard Deviation (s)'
                elif metric == 'isi_cv':
                    ylabel = 'ISI Coefficient of Variation'
                elif metric == 'burst_index':
                    ylabel = 'Burst Index'
                else:
                    ylabel = metric
                
                plt.ylabel(ylabel)
                plt.grid(True, linestyle='--', alpha=0.5, axis='y')
                
                # Save the plot
                prefix = f"{plot_prefix}_" if plot_prefix else ""
                filename = f'plots/{prefix}div{div}_{metric}_by_chip_area.png'
                plt.tight_layout()
                plt.savefig(filename, dpi=300)
                plt.close()
                print(f"Saved {filename}")
            except Exception as e:
                print(f"Error creating plot for {metric} at DIV{div}: {e}")
                plt.close()  # Close the figure in case of error
