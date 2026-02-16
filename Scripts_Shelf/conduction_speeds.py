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



MAIN_PATH = '/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Processed_data'

#RECORDINGS_PATH = os.path.join(MAIN_PATH,f'Raw_Traces')
RECORDINGS_PATH = '/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Raw_data'
SORTER_PATH = os.path.join(MAIN_PATH, f'Sorters/')
CONDUCTION_SPEEDS_PATH = os.path.join(MAIN_PATH, f'Results/Conduction_speeds/')

if not os.path.exists(CONDUCTION_SPEEDS_PATH):
    os.makedirs(CONDUCTION_SPEEDS_PATH)


FIGURE_PATH = CONDUCTION_SPEEDS_PATH

if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

#metadata = load_metadata_as_dataframe(file_path=RECORDINGS_PATH)

with open(os.path.join(MAIN_PATH, 'Results/extremum_results.pkl'), 'rb') as f:
    data = pd.read_pickle(f)

for filename in data['filename'].unique():
    chip_id, area, div = csp.extract_metadata_from_filename(filename)

#check if Conduction_speeds_all.pkl already exists
if os.path.exists(os.path.join(CONDUCTION_SPEEDS_PATH, 'Conduction_speeds_all.pkl')):
    print('Conduction speeds already calculated for all recordings')

else:
    results_combined = []  # Change to a list to collect results from all recordings

    for filename in data['filename'].unique():
        chip_id, area, div = csp.extract_metadata_from_filename(filename)
        cell_type = filename.split("_")[6][:-7]

        if os.path.exists(os.path.join(CONDUCTION_SPEEDS_PATH, f'metadata_{filename}.pkl')):
            print(f'Conduction speeds already calculated for {filename}')
            continue
        else:
            try:
                
                    
                print('Calculating conduction speeds for:', filename)
                recording = se.MaxwellRecordingExtractor(os.path.join(RECORDINGS_PATH, f'Chip{chip_id}/Chip{chip_id}_DIV{div}/{filename}'))
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

    if results_combined is not None and len(results_combined) > 0:
        # Path to the file
        file_path = os.path.join(CONDUCTION_SPEEDS_PATH, 'Conduction_speeds_all.pkl')    
        # Save the combined data
        with open(file_path, 'wb') as f:
            pickle.dump(results_combined, f)



try:
    csp.visualize_propagation_by_recording(CONDUCTION_SPEEDS_PATH, FIGURE_PATH, 'ID1765_N2_DIV35_DATE20241025_1638_spontaneous_NGN2.raw.h5')
    #df = csp.create_summary_visualization_V2(CONDUCTION_SPEEDS_PATH, FIGURE_PATH, group_by='div')
except Exception as e:
    print(f'Error visualizing conduction speeds: {e}')