import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import json
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')
#from src.utils.logger_functions import console
from src.cmos_plotter.Latency_calculator import *
from src.utils.logger_functions import console
from src.utils.metadata_functions import load_metadata_as_dataframe
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


MAIN_PATH = f'/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/3_Processed_Data/March2025_heart/'
#METADATA_PATH = f'/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/2_Raw_Data/Recordings/'

#metadata = load_metadata_as_dataframe(file_path = METADATA_PATH)

PROCESSED_DATA_PATH = os.path.join(MAIN_PATH, f'Sorters/')
PAIRINGS_PATH = os.path.join(MAIN_PATH, f'biTE_stimulation/250415_stimulation/Split_files/')
OUTPUT_PATH = os.path.join(MAIN_PATH,f'Latency_plots/Latency_plots_TE_new')
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

SORTER_PATH = PROCESSED_DATA_PATH
#list all pickle files in pairings path
pickle_files = glob.glob(os.path.join(PAIRINGS_PATH, '*.pkl'))
total = 0
already_done = 0
processed = 0

for pickle_file in pickle_files:
    total+= 1
    filename = os.path.basename(pickle_file)
    print(filename)
    filename_parts = filename.split('_')
    chip_id = int(filename_parts[0].replace('ID', ''))
    div = int(filename_parts[2].replace('DIV', ''))
    area = filename_parts[1]
    segment = filename_parts[9]

    data_te = np.load(pickle_file, allow_pickle=True)
    spikes_extremum = pd.DataFrame(data_te['SPIKEMAT_EXTREMUM'])
    spikes = np.array([[int(row[0]), float(row[1]), float(row[2])] for row in data_te['SPIKEMAT']])
        
    
    if 'validated_results' in data_te.keys():
        console.info(f'Pairs from {filename} were already validated')
        already_done += 1
    else:
        try:
            exp_duration = data_te['EXPERIMENT_DURATION']
            te_unit_pairs = pd.DataFrame(data_te['mTE'])
            pairings = te_unit_pairs
            
            unit_ids = list(data_te['UNIT_TO_EL'].keys())

            spikes, electrodes_pre_all, electrodes_post_all, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all = get_electrode_unit_info_te(data_te, pairings, area, unit_ids)
            #run_for_all_files_latency_plot(OUTPUT_PATH, filename, data_te,exp_duration,unit_ids,spikes,spikes_extremum, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all)

            run_for_all_files_latency_calculation_split(pickle_file, data_te, exp_duration, unit_ids, spikes, spikes_extremum, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all)
            #run_for_all_files_latency_plot(OUTPUT_PATH, filename, data_te,exp_duration,unit_ids,spikes,spikes_extremum, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all)

            console.info(f'Latencies for {filename} were calculated and pairs were filtered')
            processed+=1
        except Exception as e:
            console.error(f'Error in {filename}: {e}')
            continue
console.info(f'Processed {processed} out of {total} files.')
console.info(f'Already done aearlier {already_done} out of {total} files.')

