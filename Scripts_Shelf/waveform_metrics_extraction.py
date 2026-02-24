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
a=0
for filename in data['filename'].unique():
        # Try to load the waveform extractor with path remapping
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
                
                # Create output directory
                output_folder = os.path.join(waveform_folder, 'waveform_metrics_output')
                os.makedirs(output_folder, exist_ok=True)

                
                # Now we can use the quality metrics module
                import spikeinterface.qualitymetrics as sqm
                
                # Calculate all available metrics
                # In v0.100.6, we need to calculate metrics individually
                all_waveforms = {}
                
                # Calculate amplitude, peak-to-trough metrics 
                for unit_id in unit_ids:
                    # Get waveforms for this unit
                    waveforms = np.load(os.path.join(waveform_folder, f'waveforms/waveforms_{unit_id}.npy'))
                    templates = np.mean(waveforms, axis=0)
                    
                    #print(waveforms.shape)
                    # For each unit, calculate metrics manually
                    unit_metrics = {}
                    
                    # 1. Find best channel
                    peak_channel = np.argmax(np.max(np.abs(templates), axis=0))
                    
                    # 2. Get template on best channel
                    waveforms_best_channel = waveforms[:, :, peak_channel]
                    
                    #print(waveforms_best_channel.shape)
                    
                    # Store unit_id and waveforms_best_channel for each unit
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
