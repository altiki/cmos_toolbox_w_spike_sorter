import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import argparse

# Import spikeinterface
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.qualitymetrics as sqm
import spikeinterface.postprocessing as spost

def load_waveforms_and_compute_metrics(waveforms_folder, output_folder, sampling_rate=20000):
    """
    Load waveform files and compute metrics using spikeinterface.
    
    Parameters
    ----------
    waveforms_folder : str
        Path to folder containing waveform numpy files.
    output_folder : str
        Path to save outputs.
    sampling_rate : int, optional
        Sampling rate in Hz, default is 20000.
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    plots_folder = os.path.join(output_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    # Build dictionary of waveforms
    waveforms_dict = {}
    unit_ids = []
    
    print(f"Loading waveforms from {waveforms_folder}...")
    for file in os.listdir(waveforms_folder):
        if file.endswith('.npy') and 'waveforms_' in file:
            unit_id = int(file.split('_')[1].split('.')[0])
            unit_ids.append(unit_id)
            waveform_path = os.path.join(waveforms_folder, file)
            waveforms_dict[unit_id] = np.load(waveform_path)
            print(f"Loaded unit {unit_id} with shape {waveforms_dict[unit_id].shape}")
    
    print(f"Loaded waveforms for {len(waveforms_dict)} units")
    
    # Create metrics dictionary
    all_metrics = {}
    
    # Process each unit
    for unit_id, waveforms in waveforms_dict.items():
        print(f"Processing unit {unit_id}...")
        unit_metrics = {}
        
        # Check waveform shape and adjust if needed
        if waveforms.ndim == 3:
            # Shape should be (n_spikes, n_samples, n_channels)
            # Determine if reshaping is needed
            if waveforms.shape[1] == 1:  # If voltage dimension is 1
                # Reshape to (n_spikes, n_samples, n_channels)
                waveforms = waveforms.squeeze(axis=1)
        
        # Compute metrics using spikeinterface functions
        try:
            # 1. Calculate amplitude (peak-to-trough in μV)
            amplitudes = np.max(waveforms, axis=1) - np.min(waveforms, axis=1)
            unit_metrics['amplitude'] = np.mean(amplitudes)
            
            # 2. Find peak and trough for each waveform
            troughs = np.argmin(waveforms, axis=1)
            peaks_after_trough = np.array([trough + np.argmax(wf[trough:]) for trough, wf in zip(troughs, waveforms)])
            
            # 3. Calculate peak-to-trough duration
            peak_trough_durations = np.abs(peaks_after_trough - troughs) / sampling_rate * 1000  # ms
            unit_metrics['peak_to_trough_duration'] = np.mean(peak_trough_durations)
            
            # 4. Calculate peak-trough ratio
            trough_values = np.array([waveforms[i, troughs[i]] for i in range(len(troughs))])
            peak_values = np.array([waveforms[i, peaks_after_trough[i]] for i in range(len(peaks_after_trough))])
            with np.errstate(divide='ignore', invalid='ignore'):
                peak_trough_ratios = np.abs(peak_values / trough_values)
                # Replace inf and nan with high values
                peak_trough_ratios[~np.isfinite(peak_trough_ratios)] = 10  # arbitrary high value
            unit_metrics['peak_trough_ratio'] = np.mean(peak_trough_ratios)
            
            # 5. Calculate repolarization and recovery slopes
            repolarization_slopes = []
            recovery_slopes = []
            
            for i in range(len(waveforms)):
                # Repolarization slope (before trough)
                if troughs[i] > 5:  # Need at least a few points
                    start_idx = max(0, troughs[i] - 10)
                    time_diff = (troughs[i] - start_idx) / sampling_rate
                    amplitude_diff = waveforms[i, troughs[i]] - waveforms[i, start_idx]
                    repolarization_slopes.append(amplitude_diff / time_diff if time_diff > 0 else 0)
                
                # Recovery slope (after peak)
                if peaks_after_trough[i] < len(waveforms[i]) - 5:
                    end_idx = min(len(waveforms[i]) - 1, peaks_after_trough[i] + 10)
                    time_diff = (end_idx - peaks_after_trough[i]) / sampling_rate
                    amplitude_diff = waveforms[i, end_idx] - waveforms[i, peaks_after_trough[i]]
                    recovery_slopes.append(amplitude_diff / time_diff if time_diff > 0 else 0)
            
            unit_metrics['repolarization_slope'] = np.mean(repolarization_slopes) if repolarization_slopes else 0
            unit_metrics['recovery_slope'] = np.mean(recovery_slopes) if recovery_slopes else 0
            
            # 6. Use spikeinterface quality metrics if available
            # Note: This requires a WaveformExtractor, which we don't have directly
            # If you set up a WaveformExtractor, you can use sqm.compute_waveforms_metrics
            
            # Store metrics for this unit
            all_metrics[unit_id] = unit_metrics
            
            # Visualize the metrics for this unit
            visualize_waveform_with_metrics(unit_id, waveforms[0], unit_metrics, 
                                            troughs[0], peaks_after_trough[0], 
                                            sampling_rate, plots_folder)
            
        except Exception as e:
            print(f"Error processing unit {unit_id}: {e}")
    
    # Save all metrics
    metrics_file = os.path.join(output_folder, "waveform_metrics.pkl")
    with open(metrics_file, 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f"Saved metrics to {metrics_file}")
    
    # Create dataframe for plotting distributions
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'unit_id'}, inplace=True)
    
    # Plot distributions
    plot_metric_distributions(metrics_df, plots_folder)
    
    return all_metrics

def visualize_waveform_with_metrics(unit_id, waveform, metrics, trough_idx, peak_idx, 
                                   sampling_rate, output_folder):
    """
    Visualize a waveform with its metrics.
    
    Parameters
    ----------
    unit_id : int
        Unit ID
    waveform : ndarray
        Waveform array
    metrics : dict
        Metrics for this unit
    trough_idx : int
        Index of trough
    peak_idx : int
        Index of peak
    sampling_rate : int
        Sampling rate in Hz
    output_folder : str
        Output folder for plots
    """
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
    plt.plot([repol_start, trough_idx], 
             [waveform[repol_start], waveform[trough_idx]], 
             'r-', linewidth=3, label='Repolarization Slope')
    
    # Recovery slope (after peak)
    recovery_end = min(len(waveform)-1, peak_idx + 10)
    plt.plot([peak_idx, recovery_end], 
             [waveform[peak_idx], waveform[recovery_end]], 
             'm-', linewidth=3, label='Recovery Slope')
    
    # Draw line for duration
    duration_y = (waveform[trough_idx] + waveform[peak_idx]) / 2
    plt.plot([trough_idx, peak_idx], [duration_y, duration_y], 
             'k-', linewidth=3, label='Peak-to-Trough Duration')
    
    # Add text for metrics
    plt.text(0.05, 0.95, f"Repol. Slope: {metrics['repolarization_slope']:.2f}",
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.90, f"Recovery Slope: {metrics['recovery_slope']:.2f}",
             transform=plt.gca().transAxes, fontsize=10) 
    plt.text(0.05, 0.85, f"Peak-to-Trough Duration: {metrics['peak_to_trough_duration']:.2f} ms",
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.80, f"Amplitude: {metrics['amplitude']:.2f} μV",
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.75, f"P/T Ratio: {metrics['peak_trough_ratio']:.2f}",
             transform=plt.gca().transAxes, fontsize=10)
    
    plt.title(f'Unit {unit_id} Waveform with Metrics')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (μV)')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_folder, f"unit_{unit_id}_waveform_metrics.png"))
    plt.close()

def plot_metric_distributions(metrics_df, output_folder):
    """
    Plot distributions of metrics.
    
    Parameters
    ----------
    metrics_df : pandas.DataFrame
        DataFrame with metrics
    output_folder : str
        Output folder for plots
    """
    # Rename columns for better labels
    rename_dict = {
        'amplitude': 'Amplitude (μV)',
        'peak_to_trough_duration': 'Peak-to-Trough Duration (ms)',
        'peak_trough_ratio': 'Peak-Trough Ratio',
        'repolarization_slope': 'Repolarization Slope (μV/s)',
        'recovery_slope': 'Recovery Slope (μV/s)'
    }
    
    plot_df = metrics_df.rename(columns=rename_dict)
    
    # 1. Histograms for each metric
    for old_col, new_col in rename_dict.items():
        if old_col in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(metrics_df[old_col], kde=True)
            plt.title(f'Distribution of {new_col} across units')
            plt.xlabel(new_col)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{old_col}_histogram.png"))
            plt.close()
    
    # 2. Pairplot for relationships between metrics
    if len(metrics_df) > 1:  # Need at least 2 units for pairplot
        plt.figure(figsize=(15, 12))
        
        # Create pairplot
        sns.pairplot(
            plot_df,
            vars=[col for col in rename_dict.values() if col.split(' ')[0].lower() in metrics_df.columns],
            diag_kind='kde'
        )
        
        plt.suptitle('Pairwise Relationships Between Metrics', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "metric_relationships.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate waveform metrics using SpikeInterface.')
    parser.add_argument('--waveforms', type=str, required=True, 
                        help='Path to folder containing waveform numpy files.')
    parser.add_argument('--output', type=str, required=True, 
                        help='Path to save outputs (metrics and plots).')
    parser.add_argument('--sampling_rate', type=int, default=20000, 
                        help='Sampling rate in Hz, default is 20000.')
    
    args = parser.parse_args()
    
    # Run analysis
    load_waveforms_and_compute_metrics(args.waveforms, args.output, args.sampling_rate)
    print("Analysis complete!")

if __name__ == "__main__":
    main()