import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from pathlib import Path

def load_waveforms(folder_path):
    """
    Load waveform numpy files from a specified folder.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing the waveform numpy files.
        
    Returns
    -------
    dict
        Dictionary with unit IDs as keys and waveform arrays as values.
    """
    waveforms = {}
    for file in os.listdir(folder_path):
        if file.endswith('.npy') and 'waveforms_' in file:
            unit_id = int(file.split('_')[1].split('.')[0])
            waveform_path = os.path.join(folder_path, file)
            waveforms[unit_id] = np.load(waveform_path)
            print(f"Loaded unit {unit_id} waveforms with shape {waveforms[unit_id].shape}")
    return waveforms

def calculate_waveform_metrics(waveform, sampling_rate=20000):
    """
    Calculate metrics for a single waveform.
    
    Parameters
    ----------
    waveform : np.ndarray
        Waveform array with dimensions [time_points, voltage, channel]
    sampling_rate : int, optional
        Sampling rate in Hz, by default 20000
        
    Returns
    -------
    dict
        Dictionary containing calculated metrics.
    """
    # Find the channel with the largest amplitude
    n_channels = waveform.shape[2] if waveform.ndim > 2 else 1
    
    if n_channels > 1:
        channel_amplitudes = np.zeros(n_channels)
        for ch in range(n_channels):
            channel_amplitudes[ch] = np.ptp(waveform[:, 0, ch])
        max_channel = np.argmax(channel_amplitudes)
        wf = waveform[:, 0, max_channel]
    else:
        wf = waveform[:, 0, 0] if waveform.ndim > 2 else waveform
    
    # Find trough (minimum) and peak (maximum)
    trough_idx = np.argmin(wf)
    peak_idx = np.argmax(wf)
    
    # Ensure peak comes after trough for typical biphasic spike
    if peak_idx < trough_idx:
        # Look for peak after trough
        peak_after_trough_idx = trough_idx + np.argmax(wf[trough_idx:])
        if peak_after_trough_idx > trough_idx:  # If there's a peak after trough
            peak_idx = peak_after_trough_idx
    
    # Calculate amplitude in microvolts (peak-to-trough)
    trough_value = wf[trough_idx]
    peak_value = wf[peak_idx]
    amplitude = (peak_value - trough_value)  # Already in microvolts
    
    # Calculate peak-trough ratio
    peak_trough_ratio = abs(peak_value / trough_value) if trough_value != 0 else float('inf')
    
    # Calculate duration (time between trough and peak)
    duration_samples = abs(peak_idx - trough_idx)
    duration_ms = (duration_samples / sampling_rate) * 1000  # Convert to ms
    
    # Calculate repolarization slope (slope from start to trough)
    if trough_idx > 0:
        # Find the zero-crossing or start point before trough
        zero_crossings = np.where(np.diff(np.signbit(wf[:trough_idx])))[0]
        start_idx = zero_crossings[-1] if len(zero_crossings) > 0 else 0
        
        # Calculate slope
        time_diff = (trough_idx - start_idx) / sampling_rate
        amplitude_diff = wf[trough_idx] - wf[start_idx]
        repolarization_slope = amplitude_diff / time_diff if time_diff > 0 else 0
    else:
        repolarization_slope = 0
    
    # Calculate recovery slope (slope from peak to end)
    if peak_idx < len(wf) - 1:
        # Find the zero-crossing or end point after peak
        zero_crossings = np.where(np.diff(np.signbit(wf[peak_idx:])))[0]
        end_idx = peak_idx + zero_crossings[0] if len(zero_crossings) > 0 else len(wf) - 1
        
        # Calculate slope
        time_diff = (end_idx - peak_idx) / sampling_rate
        amplitude_diff = wf[end_idx] - wf[peak_idx]
        recovery_slope = amplitude_diff / time_diff if time_diff > 0 else 0
    else:
        recovery_slope = 0
    
    return {
        'amplitude': amplitude,
        'peak_trough_ratio': peak_trough_ratio,
        'peak_to_trough_duration_ms': duration_ms,
        'repolarization_slope': repolarization_slope,
        'recovery_slope': recovery_slope,
        'trough_idx': trough_idx,
        'peak_idx': peak_idx,
        'waveform': wf,
        'max_channel': max_channel if n_channels > 1 else 0
    }

def process_unit_waveforms(waveforms, sampling_rate=20000):
    """
    Process all waveforms for all units and calculate metrics.
    
    Parameters
    ----------
    waveforms : dict
        Dictionary with unit IDs as keys and waveform arrays as values.
    sampling_rate : int, optional
        Sampling rate in Hz, by default 20000
        
    Returns
    -------
    dict
        Dictionary with unit IDs as keys and metrics dictionaries as values.
    """
    all_metrics = {}
    
    for unit_id, unit_waveforms in waveforms.items():
        print(f"Processing unit {unit_id}...")
        
        # Initialize metrics for this unit
        unit_metrics = {
            'amplitude': [],
            'peak_trough_ratio': [],
            'peak_to_trough_duration_ms': [],
            'repolarization_slope': [],
            'recovery_slope': [],
            'waveforms': [],
            'trough_idx': [],
            'peak_idx': [],
            'max_channel': []
        }
        
        # For each waveform in the unit
        num_waveforms = min(100, len(unit_waveforms))  # Limit to 100 waveforms for efficiency
        for i in range(num_waveforms):
            try:
                # Calculate metrics for this waveform
                wf_metrics = calculate_waveform_metrics(unit_waveforms[i:i+1], sampling_rate)
                
                # Append metrics to unit metrics
                for key in unit_metrics.keys():
                    if key in wf_metrics:
                        unit_metrics[key].append(wf_metrics[key])
            except Exception as e:
                print(f"Error processing waveform {i} for unit {unit_id}: {e}")
                continue
        
        # Convert lists to numpy arrays
        for key in unit_metrics.keys():
            unit_metrics[key] = np.array(unit_metrics[key]) if key != 'waveforms' else unit_metrics[key]
        
        # Calculate summary statistics
        summary_metrics = {}
        for key, values in unit_metrics.items():
            if key not in ['waveforms', 'trough_idx', 'peak_idx', 'max_channel'] and len(values) > 0:
                summary_metrics[f'{key}_mean'] = np.mean(values)
                summary_metrics[f'{key}_median'] = np.median(values)
                summary_metrics[f'{key}_std'] = np.std(values)
                
        # Combine all metrics
        all_metrics[unit_id] = {
            'individual': unit_metrics,
            'summary': summary_metrics
        }
    
    return all_metrics

def save_metrics(metrics, output_path):
    """
    Save metrics to a pickle file.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing metrics to save.
    output_path : str
        Path to save the pickle file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Metrics saved to {output_path}")

def plot_metric_distributions(metrics, output_folder, prefix=""):
    """
    Plot distributions of metrics across units.
    
    Parameters
    ----------
    metrics : dict
        Dictionary with unit IDs as keys and metrics dictionaries as values.
    output_folder : str
        Folder to save the plots.
    prefix : str, optional
        Prefix for the output files, by default ""
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare data for plotting
    metric_data = {
        'amplitude': [],
        'peak_trough_ratio': [],
        'peak_to_trough_duration_ms': [],
        'repolarization_slope': [],
        'recovery_slope': []
    }
    
    unit_ids = []
    
    for unit_id, unit_metrics in metrics.items():
        for metric_name in metric_data.keys():
            mean_value = unit_metrics['summary'].get(f'{metric_name}_mean', np.nan)
            if not np.isnan(mean_value):
                metric_data[metric_name].append(mean_value)
                if metric_name == 'amplitude':  # Only need to append unit_id once
                    unit_ids.append(unit_id)
    
    # Plot histograms for each metric
    for metric_name, values in metric_data.items():
        if len(values) > 0:
            plt.figure(figsize=(10, 6))
            sns.histplot(values, kde=True)
            
            # Customize title and labels
            if metric_name == 'amplitude':
                plt.title('Distribution of Amplitude across units')
                plt.xlabel('Amplitude (μV)')
            elif metric_name == 'peak_to_trough_duration_ms':
                plt.title('Distribution of Peak-to-Trough Duration across units')
                plt.xlabel('Duration (ms)')
            elif metric_name == 'peak_trough_ratio':
                plt.title('Distribution of Peak-Trough Ratio across units')
                plt.xlabel('Peak-Trough Ratio')
            elif metric_name == 'repolarization_slope':
                plt.title('Distribution of Repolarization Slope across units')
                plt.xlabel('Repolarization Slope (μV/s)')
            elif metric_name == 'recovery_slope':
                plt.title('Distribution of Recovery Slope across units')
                plt.xlabel('Recovery Slope (μV/s)')
            
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{prefix}{metric_name}_histogram.png"))
            plt.close()
    
    # Plot pairwise relationships
    plt.figure(figsize=(15, 12))
    
    # Create a DataFrame for seaborn
    import pandas as pd
    df = pd.DataFrame({
        'Unit ID': unit_ids,
        'Amplitude (μV)': metric_data['amplitude'],
        'Peak-Trough Ratio': metric_data['peak_trough_ratio'],
        'Peak-to-Trough Duration (ms)': metric_data['peak_to_trough_duration_ms'],
        'Repolarization Slope (μV/s)': metric_data['repolarization_slope'],
        'Recovery Slope (μV/s)': metric_data['recovery_slope']
    })
    
    # Plot pairplot
    sns.pairplot(df, vars=['Amplitude (μV)', 'Peak-Trough Ratio', 'Peak-to-Trough Duration (ms)', 
                           'Repolarization Slope (μV/s)', 'Recovery Slope (μV/s)'],
                 diag_kind='kde')
    plt.suptitle('Pairwise Relationships Between Metrics', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{prefix}metric_relationships.png"))
    plt.close()
    
    # Plot example waveforms with metrics
    plot_example_waveforms_with_metrics(metrics, output_folder, prefix)

def plot_example_waveforms_with_metrics(metrics, output_folder, prefix=""):
    """
    Plot example waveforms with metrics overlay for visualization.
    
    Parameters
    ----------
    metrics : dict
        Dictionary with unit IDs as keys and metrics dictionaries as values.
    output_folder : str
        Folder to save the plots.
    prefix : str, optional
        Prefix for the output files, by default ""
    """
    # Let's plot a few example units
    unit_ids = list(metrics.keys())
    for unit_id in unit_ids[:min(5, len(unit_ids))]:  # Plot up to 5 units
        unit_metrics = metrics[unit_id]
        
        # Let's get individual metrics for a single waveform for visualization
        if 'individual' in unit_metrics and len(unit_metrics['individual']['amplitude']) > 0:
            # Find the waveform with median amplitude
            amplitudes = unit_metrics['individual']['amplitude']
            idx = np.argsort(amplitudes)[len(amplitudes)//2]
            
            # Get the actual waveform data
            if 'waveforms' in unit_metrics['individual'] and len(unit_metrics['individual']['waveforms']) > 0:
                waveform = unit_metrics['individual']['waveforms'][idx]
                trough_idx = unit_metrics['individual']['trough_idx'][idx]
                peak_idx = unit_metrics['individual']['peak_idx'][idx]
                
                plt.figure(figsize=(12, 8))
                
                # Plot the waveform
                time = np.arange(len(waveform))
                plt.plot(time, waveform, 'b-', linewidth=2)
                
                # Mark trough and peak
                plt.plot(trough_idx, waveform[trough_idx], 'go', markersize=8, label='Trough')
                plt.plot(peak_idx, waveform[peak_idx], 'bo', markersize=8, label='Peak')
                
                # Draw lines for slopes
                # Repolarization slope (before trough)
                repol_start = max(0, trough_idx - 10)
                repol_y_start = waveform[repol_start]
                repol_y_end = waveform[trough_idx]
                plt.plot([repol_start, trough_idx], [repol_y_start, repol_y_end], 'r-', linewidth=3, label='Repolarization Slope')
                
                # Recovery slope (after peak)
                recovery_end = min(len(waveform)-1, peak_idx + 10)
                recovery_y_start = waveform[peak_idx]
                recovery_y_end = waveform[recovery_end]
                plt.plot([peak_idx, recovery_end], [recovery_y_start, recovery_y_end], 'm-', linewidth=3, label='Recovery Slope')
                
                # Draw line for duration
                duration_y = (waveform[trough_idx] + waveform[peak_idx]) / 2
                plt.plot([trough_idx, peak_idx], [duration_y, duration_y], 'k-', linewidth=3, label='Duration')
                
                # Add text annotations with values
                amplitude = unit_metrics['individual']['amplitude'][idx]
                peak_trough_ratio = unit_metrics['individual']['peak_trough_ratio'][idx]
                duration_ms = unit_metrics['individual']['peak_to_trough_duration_ms'][idx]
                repol_slope = unit_metrics['individual']['repolarization_slope'][idx]
                recovery_slope = unit_metrics['individual']['recovery_slope'][idx]
                
                # Position text annotations
                plt.text(trough_idx - 10, waveform[trough_idx], f'Trough', ha='right', va='center', fontsize=10)
                plt.text(peak_idx + 5, waveform[peak_idx], f'Peak', ha='left', va='center', fontsize=10)
                
                y_range = np.ptp(waveform)
                text_y_positions = np.linspace(waveform.min() + 0.1*y_range, waveform.max() - 0.1*y_range, 5)
                
                plt.text(0.05, 0.95, f'Repol. Slope: {repol_slope:.2f}', transform=plt.gca().transAxes, fontsize=10)
                plt.text(0.05, 0.90, f'Recovery Slope: {recovery_slope:.2f}', transform=plt.gca().transAxes, fontsize=10)
                plt.text(0.05, 0.85, f'Peak-to-Trough Duration: {duration_ms:.2f} ms', transform=plt.gca().transAxes, fontsize=10)
                plt.text(0.05, 0.80, f'Amplitude: {amplitude:.2f} μV', transform=plt.gca().transAxes, fontsize=10)
                plt.text(0.05, 0.75, f'P/T Ratio: {peak_trough_ratio:.2f}', transform=plt.gca().transAxes, fontsize=10)
                
                plt.title(f'Unit {unit_id} Waveform with Metrics')
                plt.xlabel('Time (samples)')
                plt.ylabel('Amplitude (μV)')
                plt.grid(True)
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f"{prefix}unit_{unit_id}_waveform_metrics.png"))
                plt.close()

def main(waveforms_folder, output_folder, sampling_rate=20000):
    """
    Main function to process waveforms, calculate metrics, save results, and generate plots.
    
    Parameters
    ----------
    waveforms_folder : str
        Path to the folder containing waveform numpy files.
    output_folder : str
        Path to save outputs (metrics pickle and plots).
    sampling_rate : int, optional
        Sampling rate in Hz, by default 20000
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load waveforms
    print(f"Loading waveforms from {waveforms_folder}...")
    waveforms = load_waveforms(waveforms_folder)
    print(f"Loaded waveforms for {len(waveforms)} units.")
    
    # Process waveforms
    print("Calculating metrics for waveforms...")
    waveform_metrics = process_unit_waveforms(waveforms, sampling_rate)
    
    # Save metrics
    metrics_output_path = os.path.join(output_folder, "waveform_metrics.pkl")
    save_metrics(waveform_metrics, metrics_output_path)
    
    # Plot distributions
    plot_output_folder = os.path.join(output_folder, "plots")
    os.makedirs(plot_output_folder, exist_ok=True)
    
    print("Plotting metric distributions...")
    plot_metric_distributions(waveform_metrics, plot_output_folder)
    
    print("Done!")
    return waveform_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate waveform metrics and generate plots.')
    parser.add_argument('--waveforms', type=str, required=True, help='Path to the folder containing waveform numpy files.')
    parser.add_argument('--output', type=str, required=True, help='Path to save outputs (metrics pickle and plots).')
    parser.add_argument('--sampling_rate', type=int, default=20000, help='Sampling rate in Hz, default is 20000.')
    
    args = parser.parse_args()
    
    main(args.waveforms, args.output, args.sampling_rate)