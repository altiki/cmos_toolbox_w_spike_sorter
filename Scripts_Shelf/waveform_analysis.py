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


MAIN_PATH = '...'
OUTPUT_PATH = '...'

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

                

                import spikeinterface.qualitymetrics as sqm
                
                all_metrics = {}
                

                for unit_id in unit_ids:
                    waveforms = np.load(os.path.join(waveform_folder, f'waveforms/waveforms_{unit_id}.npy'))
                    templates = np.mean(waveforms, axis=0)
                    
                    unit_metrics = {}
                    
                    peak_channel = np.argmax(np.max(np.abs(templates), axis=0))
                    
                    template = templates[:, peak_channel]
                    
                    peak_idx = np.argmax(template)
                    trough_idx = np.argmin(template)
                    
                    if peak_idx < trough_idx:
                        peak_after_trough = trough_idx + np.argmax(template[trough_idx:])
                        if peak_after_trough > trough_idx:
                            peak_idx = peak_after_trough
                    

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
                    half_width_start = np.where(template[:trough_idx] <= (template[trough_idx] / 2))[0]
                    half_width_end = np.where(template[trough_idx:] >= (template[trough_idx] / 2))[0]
                    if len(half_width_start) > 0 and len(half_width_end) > 0:
                        half_width_start_idx = half_width_start[-1]
                        half_width_end_idx = half_width_end[0] + trough_idx
                        unit_metrics['half_width'] = (half_width_end_idx - half_width_start_idx) / sampling_rate * 1000
                    else:
                        unit_metrics['half_width'] = 0    
                    
                    unit_metrics['template'] = template
                    unit_metrics['peak_idx'] = peak_idx
                    unit_metrics['trough_idx'] = trough_idx
                    
                    all_metrics[unit_id] = unit_metrics
                    
                   
            
                # Create metrics dataframe and save to pickle
                metrics_df = pd.DataFrame.from_dict(
                    {unit_id: {k: v for k, v in metrics.items() if k not in ['template', 'peak_idx', 'trough_idx']} 
                    for unit_id, metrics in all_metrics.items()},
                    orient='index'
                )
                
                # Save metrics to pickle
                with open(os.path.join(output_folder, 'waveform_metrics.pkl'), 'wb') as f:
                    pickle.dump(all_metrics, f)
                
                print(f"Analysis complete! Results saved to {output_folder}")
            except Exception as e:
                print(f"Error loading WaveformExtractor: {e}")
