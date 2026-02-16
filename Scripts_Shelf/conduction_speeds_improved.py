# Configure logging first - add this at the very top of your script
import logging
#logging.basicConfig(level=logging.ERROR)  # Only show ERROR level and above
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import seaborn as sns
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')
#from src.utils.logger_functions import console
from src.cmos_plotter.Plotter_Helper_KV import *
from src.utils.metadata_functions import load_metadata_as_dataframe
from src.utils.logger_functions import console
from src.cmos_plotter import Conduction_speed_plotter as csp
import spikeinterface.extractors as se
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
# Silence just the matplotlib.font_manager logger
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)



MAIN_PATH = '/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/3_Processed_Data/Very_old_chips/'

#RECORDINGS_PATH = os.path.join(MAIN_PATH,f'Raw_Traces')
RECORDINGS_PATH = '/itet-stor/kvulic/neuronies/single_neurons/4_Varia/Very_old_chips_recording/'
SORTER_PATH = os.path.join(MAIN_PATH, f'Sorters/')
CONDUCTION_SPEEDS_PATH = os.path.join(MAIN_PATH, f'Results/')

if not os.path.exists(CONDUCTION_SPEEDS_PATH):
    os.makedirs(CONDUCTION_SPEEDS_PATH)


FIGURE_PATH = CONDUCTION_SPEEDS_PATH

if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

metadata = load_metadata_as_dataframe(file_path=RECORDINGS_PATH)


# Initialize results_combined
results_combined = []

# Check if Conduction_speeds_all.pkl already exists
pkl_path = os.path.join(CONDUCTION_SPEEDS_PATH, 'Conduction_speeds_all.pkl')
if os.path.exists(pkl_path):
    print('Loading existing conduction speeds data...')
    with open(pkl_path, 'rb') as f:
        results_combined = pickle.load(f)
    
    # Extract filenames that are already processed
    processed_filenames = set()
    for result in results_combined:
        if 'filename' in result:
            processed_filenames.add(result['filename'])
    
    print(f"Found {len(processed_filenames)} previously processed recordings")
    print(f"Total results in existing file: {len(results_combined)}")

# Process remaining recordings not in the existing data
new_results_count = 0
for idx, filename in enumerate(metadata.Filename):
    # Skip if this recording is already in the combined results
    if os.path.exists(pkl_path) and filename in processed_filenames:
        print(f'Skipping {filename} - already in combined results')
        continue
        
    chip_id = metadata.loc[idx, 'Chip_ID']
    div = metadata.loc[idx, 'DIV']
    cell_type = 'NGN2'
    area = metadata.loc[idx, 'Network_ID']

    if os.path.exists(os.path.join(CONDUCTION_SPEEDS_PATH, f'metadata_{filename}.pkl')):
        print(f'Metadata exists for {filename}, but loading to ensure it is in combined results')
        # We'll process this one to make sure it's in the combined results
        
    try:
        print('Calculating conduction speeds for:', filename)
        recording = se.MaxwellRecordingExtractor(os.path.join(RECORDINGS_PATH, f'{filename}'))
        template_path = os.path.join(SORTER_PATH, f'Sorter_{filename}/wf_folder_curated/')
        
        # Get sampling rate
        sampling_rate = recording.get_sampling_frequency()
        # Get probe locations
        probe_locations = recording.get_channel_locations()
        # Extract templates
        templates = csp.load_and_extract_templates(template_path)
        #Get unit ids
        with open(os.path.join(SORTER_PATH, f'Sorter_{filename}/wf_folder_curated/sparsity.json'), 'rb') as f:
            unit_id_data = json.load(f)
        unit_ids = unit_id_data['unit_ids']
        # Analyze conduction speeds
        results = csp.analyze_conduction_speeds(templates, probe_locations, sampling_rate, unit_ids)
        
        if results is not None and len(results) > 0:  # Check if results exist and not empty
            # Store recording-specific metadata for later use in visualization
            recording_metadata = {
                'chip_id': chip_id,
                'div': div,
                'cell_type': cell_type,
                'area': area,
                'sampling_rate': sampling_rate,
                'probe_locations': probe_locations,
                'unit_ids': unit_ids,
                'templates': templates,
                'filename': filename
            }
            
            # Add metadata to each unit's results
            for result in results:
                result['chip_id'] = chip_id
                result['div'] = div
                result['cell_type'] = cell_type
                result['area'] = area
                result['filename'] = filename
                # Makes sure the speed key is consistent
                if 'speed' in result and 'speed_ms-1' not in result:
                    result['speed_ms-1'] = result.pop('speed')
                
                # Add to the combined results list
                results_combined.append(result)
                new_results_count += 1
            
            # Save the metadata separately for this recording
            metadata_dict = {
                'sampling_rate': sampling_rate,
                'probe_locations': probe_locations,
                'templates': templates,
                'unit_ids': unit_ids
            }
            
            # Save individual recording metadata
            with open(os.path.join(CONDUCTION_SPEEDS_PATH, f'metadata_{filename}.pkl'), 'wb') as f:
                pickle.dump(metadata_dict, f)
                
        else:
            print(f'No conduction speeds calculated for {filename}')
            
    except Exception as e:
        print(f'Error calculating conduction speeds for {filename}: {e}')
        continue

# If we have results and either added new ones or are creating the file for the first time
if results_combined is not None and len(results_combined) > 0:
    if new_results_count > 0 or not os.path.exists(pkl_path):
        print(f"Saving combined results with {new_results_count} new entries (total: {len(results_combined)})")
        # Save the combined data
        with open(pkl_path, 'wb') as f:
            pickle.dump(results_combined, f)
    else:
        print("No new results to add to the combined file")

try:
    csp.visualize_propagation_by_recording(CONDUCTION_SPEEDS_PATH, FIGURE_PATH)
    df = csp.create_summary_visualization_V2(CONDUCTION_SPEEDS_PATH, FIGURE_PATH, group_by='div')
except Exception as e:
    print(f'Error visualizing conduction speeds: {e}')