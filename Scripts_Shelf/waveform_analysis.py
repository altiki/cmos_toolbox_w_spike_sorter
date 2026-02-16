import spikeinterface as si
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import json
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')
from src.utils.metadata_functions import load_metadata_as_dataframe
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

#MAIN_PATH  = '/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/3_Processed_Data/Pickle_files_1851/'

#RECORDINGS_PATH = '/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/2_Raw_Data/1851_recordings/'
#metadata = load_metadata_as_dataframe(file_path=RECORDINGS_PATH)

MAIN_PATH = '/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Processed_data/'
OUTPUT_PATH = '/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Processed_data/Results/'

with open(os.path.join(MAIN_PATH, 'Results/extremum_results.pkl'), 'rb') as f:
    data = pd.read_pickle(f)

for filename in data['filename'].unique():
        # Try to load the waveform extractor with path remapping
        waveform_folder = os.path.join(MAIN_PATH, f'Sorters/Sorter_{filename}/wf_folder_curated')
        output_folder = os.path.join(waveform_folder, 'waveform_metrics_output')
        if os.path.isfile(os.path.join(output_folder, 'waveform_metrics.pkl')):
            print(f"Waveform metrics already calculated for {filename}. Skipping...")
        else:
             
            try:
                # Get unit IDs
                with open(os.path.join(waveform_folder, 'sparsity.json'), 'r') as f:
                            sparsity = json.load(f)
                unit_ids = sparsity['unit_id_to_channel_ids'].keys()
                unit_ids = list(map(int, unit_ids))
                print(f"Successfully loaded WaveformExtractor with {len(unit_ids)} units")
                
                # Create output directory
                output_folder = os.path.join(waveform_folder, 'waveform_metrics_output')
                os.makedirs(output_folder, exist_ok=True)

                
                # Now we can use the quality metrics module
                import spikeinterface.qualitymetrics as sqm
                
                # Calculate all available metrics
                # In v0.100.6, we need to calculate metrics individually
                all_metrics = {}
                
                # Calculate amplitude, peak-to-trough metrics 
                for unit_id in unit_ids:
                    # Get waveforms for this unit
                    waveforms = np.load(os.path.join(waveform_folder, f'waveforms/waveforms_{unit_id}.npy'))
                    templates = np.mean(waveforms, axis=0)
                    
                    # For each unit, calculate metrics manually
                    unit_metrics = {}
                    
                    # 1. Find best channel
                    peak_channel = np.argmax(np.max(np.abs(templates), axis=0))
                    
                    # 2. Get template on best channel
                    template = templates[:, peak_channel]
                    
                    # 3. Find peak and trough
                    peak_idx = np.argmax(template)
                    trough_idx = np.argmin(template)
                    
                    # Make sure peak comes after trough for biphasic spike
                    if peak_idx < trough_idx:
                        # Try to find a peak after the trough
                        peak_after_trough = trough_idx + np.argmax(template[trough_idx:])
                        if peak_after_trough > trough_idx:
                            peak_idx = peak_after_trough
                    
                    # 4. Calculate metrics
                    # Amplitude (μV)
                    unit_metrics['amplitude'] = np.max(template) - np.min(template)
                    
                    # Peak-to-trough duration (ms)
                    sampling_rate = 20000
                    unit_metrics['peak_to_trough_duration'] = abs(peak_idx - trough_idx) / sampling_rate * 1000
                    
                    # Peak-trough ratio
                    peak_val = template[peak_idx]
                    trough_val = template[trough_idx]
                    unit_metrics['peak_trough_ratio'] = abs(peak_val / trough_val) if trough_val != 0 else 0
                    
                    # Repolarization slope (before trough)
                    if trough_idx > 5:
                        start_idx = max(0, trough_idx - 10)
                        time_diff = (trough_idx - start_idx) / sampling_rate
                        amplitude_diff = template[trough_idx] - template[start_idx]
                        unit_metrics['repolarization_slope'] = amplitude_diff / time_diff if time_diff > 0 else 0
                    else:
                        unit_metrics['repolarization_slope'] = 0
                    
                    # Recovery slope (after peak)
                    if peak_idx < len(template) - 5:
                        end_idx = min(len(template) - 1, peak_idx + 10)
                        time_diff = (end_idx - peak_idx) / sampling_rate
                        amplitude_diff = template[end_idx] - template[peak_idx]
                        unit_metrics['recovery_slope'] = amplitude_diff / time_diff if time_diff > 0 else 0
                    else:
                        unit_metrics['recovery_slope'] = 0


                    # Half width (ms) of trough
                    # Find the half-width of the trough
                    half_width_start = np.where(template[:trough_idx] <= (template[trough_idx] / 2))[0]
                    half_width_end = np.where(template[trough_idx:] >= (template[trough_idx] / 2))[0]
                    if len(half_width_start) > 0 and len(half_width_end) > 0:
                        half_width_start_idx = half_width_start[-1]
                        half_width_end_idx = half_width_end[0] + trough_idx
                        unit_metrics['half_width'] = (half_width_end_idx - half_width_start_idx) / sampling_rate * 1000
                    else:
                        unit_metrics['half_width'] = 0    
                    
                    # Store the template and indices for visualization
                    unit_metrics['template'] = template
                    unit_metrics['peak_idx'] = peak_idx
                    unit_metrics['trough_idx'] = trough_idx
                    
                    # Add to all metrics
                    all_metrics[unit_id] = unit_metrics
                    
                    '''
                    # Visualize each unit's waveform with metrics
                    plt.figure(figsize=(12, 8))
                    time = np.arange(len(template))
                    plt.plot(time, template, 'b-', linewidth=2)
                    
                    # Mark trough and peak
                    plt.plot(trough_idx, template[trough_idx], 'go', markersize=8, label='Trough')
                    plt.plot(peak_idx, template[peak_idx], 'bo', markersize=8, label='Peak')
                    
                    # Draw lines for slopes
                    repol_start = max(0, trough_idx - 10)
                    plt.plot([repol_start, trough_idx], 
                            [template[repol_start], template[trough_idx]], 
                            'r-', linewidth=3, label='Repolarization Slope')
                    
                    recovery_end = min(len(template)-1, peak_idx + 10)
                    plt.plot([peak_idx, recovery_end], 
                            [template[peak_idx], template[recovery_end]], 
                            'm-', linewidth=3, label='Recovery Slope')
                    
                    # Draw line for duration
                    duration_y = (template[trough_idx] + template[peak_idx]) / 2
                    plt.plot([trough_idx, peak_idx], [duration_y, duration_y], 
                            'k-', linewidth=3, label='Peak-to-Trough Duration')
                    
                    # Add text for metrics
                    plt.text(0.05, 0.95, f"Repol. Slope: {unit_metrics['repolarization_slope']:.2f}",
                            transform=plt.gca().transAxes, fontsize=10)
                    plt.text(0.05, 0.90, f"Recovery Slope: {unit_metrics['recovery_slope']:.2f}",
                            transform=plt.gca().transAxes, fontsize=10)
                    plt.text(0.05, 0.85, f"Peak-to-Trough Duration: {unit_metrics['peak_to_trough_duration']:.2f} ms",
                            transform=plt.gca().transAxes, fontsize=10)
                    plt.text(0.05, 0.80, f"Amplitude: {unit_metrics['amplitude']:.2f} μV",
                            transform=plt.gca().transAxes, fontsize=10)
                    plt.text(0.05, 0.75, f"P/T Ratio: {unit_metrics['peak_trough_ratio']:.2f}",
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
                    '''
            
                # Create metrics dataframe and save to pickle
                metrics_df = pd.DataFrame.from_dict(
                    {unit_id: {k: v for k, v in metrics.items() if k not in ['template', 'peak_idx', 'trough_idx']} 
                    for unit_id, metrics in all_metrics.items()},
                    orient='index'
                )
                
                # Save metrics to pickle
                with open(os.path.join(output_folder, 'waveform_metrics.pkl'), 'wb') as f:
                    pickle.dump(all_metrics, f)
                
                # Plot distributions
                '''
                for metric in ['amplitude', 'peak_to_trough_duration', 'peak_trough_ratio', 
                            'repolarization_slope', 'recovery_slope']:
                    if metric in metrics_df.columns:
                        plt.figure(figsize=(10, 6))
                        sns.histplot(metrics_df[metric], kde=True)
                        
                        if metric == 'amplitude':
                            plt.title('Distribution of Amplitude across units')
                            plt.xlabel('Amplitude (μV)')
                        elif metric == 'peak_to_trough_duration':
                            plt.title('Distribution of Peak-to-Trough Duration across units')
                            plt.xlabel('Duration (ms)')
                        elif metric == 'peak_trough_ratio':
                            plt.title('Distribution of Peak-Trough Ratio across units')
                            plt.xlabel('Peak-Trough Ratio')
                        elif metric == 'repolarization_slope':
                            plt.title('Distribution of Repolarization Slope across units')
                            plt.xlabel('Repolarization Slope (μV/s)')
                        elif metric == 'recovery_slope':
                            plt.title('Distribution of Recovery Slope across units')
                            plt.xlabel('Recovery Slope (μV/s)')
                        
                        plt.ylabel('Count')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_folder, f"{metric}_histogram.png"))
                        plt.close()
                
                # Create pairplot
                if len(metrics_df) > 1:
                    plt.figure(figsize=(15, 12))
                    metrics_df_renamed = metrics_df.rename(columns={
                        'amplitude': 'Amplitude (μV)',
                        'peak_to_trough_duration': 'Peak-to-Trough Duration (ms)',
                        'peak_trough_ratio': 'Peak-Trough Ratio',
                        'repolarization_slope': 'Repolarization Slope (μV/s)',
                        'recovery_slope': 'Recovery Slope (μV/s)'
                    })
                    sns.pairplot(metrics_df_renamed, diag_kind='kde')
                    plt.suptitle('Pairwise Relationships Between Metrics', y=1.02)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, "metric_relationships.png"))
                    plt.close()
                
                print(f"Analysis complete! Results saved to {output_folder}")
                '''
                print(f"Analysis complete! Results saved to {output_folder}")
            except Exception as e:
                print(f"Error loading WaveformExtractor: {e}")