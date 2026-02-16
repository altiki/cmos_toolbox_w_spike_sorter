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
METADATA_PATH = f'/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/2_Raw_Data/Recordings/'

metadata = load_metadata_as_dataframe(file_path = METADATA_PATH)

PROCESSED_DATA_PATH = os.path.join(MAIN_PATH, f'Sorters')
PAIRINGS_PATH = os.path.join(MAIN_PATH, f'biTE_new')
OUTPUT_PATH = os.path.join(MAIN_PATH,f'Latency_plots/Latency_plots_TE_new')
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

SORTER_PATH = PROCESSED_DATA_PATH
a=0

for filename in metadata['Filename']:
    chip_id = int(filename.split('_')[0].replace('ID', ''))
    div = int(filename.split('_')[2].replace('DIV', ''))
    #if filename == 'ID2167_N6_DIV28_DATE20250411_1039_spontaneous_NGN2.raw.h5':
    try:
        data = np.load(os.path.join(PROCESSED_DATA_PATH, f'{filename[:-3]}_processed.pkl'), allow_pickle=True)
        spikes_extremum = pd.DataFrame(data['SPIKEMAT_EXTREMUM'])
        spikes = np.array([[int(row[0]), float(row[1]), float(row[2])] for row in data['SPIKEMAT']])
        
        FULL_SORTER_PATH = os.path.join(SORTER_PATH,'Sorter_'+filename)
        area = filename.split('_')[1]
        # Open the file in binary mode
        sys.modules['numpy.rec'] = np.rec
        #with open(os.path.join(PAIRINGS_PATH, f'{filename[:-3]}_processed_info_metrics.pkl'), 'rb') as f:
        #    data_te = np.array(pickle.load(f))
        data_te = np.load(os.path.join(PAIRINGS_PATH, f'{filename[:-3]}_processed_info_metrics.pkl'), allow_pickle=True)
        if 'validated_results' in data_te.keys():
            console.info(f'Pairs from {filename} were already validated')
        elif a ==0:
            exp_duration = data_te['EXPERIMENT_DURATION']
            te_unit_pairs = pd.DataFrame(data_te['mTE'])
            pairings = te_unit_pairs
            

            with open(os.path.join(FULL_SORTER_PATH, 'wf_folder_curated/sparsity.json'), 'r') as file:
                sorting_info = json.load(file)

            unit_ids = sorting_info['unit_ids']

            spikes, electrodes_pre_all, electrodes_post_all, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all = get_electrode_unit_info_te(data, pairings, area, unit_ids)
            #run_for_all_files_latency_plot(OUTPUT_PATH, filename, data_te,exp_duration,unit_ids,spikes,spikes_extremum, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all)

            run_for_all_files_latency_calculation(PAIRINGS_PATH, filename, data_te, exp_duration, unit_ids, spikes, spikes_extremum, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all)
            #run_for_all_files_latency_plot(OUTPUT_PATH, filename, data_te,exp_duration,unit_ids,spikes,spikes_extremum, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all)

            console.info(f'Latencies for {filename} were calculated and pairs were filtered')

    except Exception as e:
        console.error(f'Error in {filename}: {e}')
        pass


