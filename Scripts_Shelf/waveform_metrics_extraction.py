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
a=0
for filename in data['filename'].unique():
        waveform_folder = os.path.join(MAIN_PATH, f'Sorters/Sorter_{filename}/wf_folder_curated')
        output_folder = os.path.join(waveform_folder, 'waveform_metrics_output')
        if os.path.isfile(os.path.join(output_folder, 'waveform_metrics.pkl')):
            print(f"Waveform metrics already calculated for {filename}. Skipping...")
        if a == 0:
             
            try:
                # Get unit IDs
                with open(os.path.join(waveform_folder, 'sparsity.json'), 'r') as f:
                            sparsity = json.load(f)
                unit_ids = sparsity['unit_id_to_channel_ids'].keys()
                unit_ids = list(map(int, unit_ids))
                print(f"Successfully loaded WaveformExtractor with {len(unit_ids)} units")
                
            
                output_folder = os.path.join(waveform_folder, 'waveform_metrics_output')
                os.makedirs(output_folder, exist_ok=True)

                
                import spikeinterface.qualitymetrics as sqm
                
                # Calculate all available metrics
                all_waveforms = {}
           
                for unit_id in unit_ids:

                    waveforms = np.load(os.path.join(waveform_folder, f'waveforms/waveforms_{unit_id}.npy'))
                    templates = np.mean(waveforms, axis=0)
                    

                    unit_metrics = {}
        
                    peak_channel = np.argmax(np.max(np.abs(templates), axis=0))
                    waveforms_best_channel = waveforms[:, :, peak_channel]

                    if 'unit_ids' not in all_waveforms:
                        all_waveforms['unit_ids'] = []
                        all_waveforms['waveforms_best_channel'] = []
                    all_waveforms['unit_ids'].append(unit_id)
                    all_waveforms['waveforms_best_channel'].append(waveforms_best_channel)

                # Save all_waveforms as waveforms.pkl after processing all units
                with open(os.path.join(output_folder, 'waveforms.pkl'), 'wb') as wf:
                    pickle.dump(all_waveforms, wf)

                print(f"Analysis complete! Results saved to {output_folder}")
            except Exception as e:
                print(f"Error loading WaveformExtractor: {e}")
