import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import glob
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')
from src.cmos_plotter.Plotter_Helper_KV import *
from src.utils.metadata_functions import load_metadata_as_dataframe
import logging
from src.utils.logger_functions import console
import pickle
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


MAIN_PATH = '...'

PAIRINGS_PATH = os.path.join(MAIN_PATH, f'biTE/')
OUTPUT_PATH = os.path.join(MAIN_PATH,f'Results')
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

pickle_files = glob.glob(os.path.join(PAIRINGS_PATH, '*.pkl'))

total = 0
already_done = 0
processed = 0

for pickle_file in pickle_files:
    total += 1
    filename = os.path.basename(pickle_file)
    filename_parts = filename.split('_')
    chip_id = int(filename_parts[0].replace('ID', ''))
    div = int(filename_parts[2].replace('DIV', ''))
    area = filename_parts[1]

    data_te = np.load(pickle_file, allow_pickle=True)

    try:
        filtered_data = data_te['validated_results']

        if 'syn probability' in filtered_data[0]:
            console.info(f'Synaptic probability already calculated for {filename}')
            already_done += 1
        else:
            try:
                console.info(f'Calculating synaptic probability for {filename}...')        
                for pair in filtered_data:
                    if pair['validation'] == 'good':
                        lag = pair['lag']
                        latency = pair['latency_extremum']
                        data_output = latency[latency['category'] == 'output']
                        latency_filtered = data_output[(data_output['latency'] >= lag - 1) & (data_output['latency'] <= lag + 1)]
                        input_spike_counts = np.sum(latency['category'] == 'input')
                        output_spike_counts = np.sum(np.isin(latency_filtered['input spike'], latency['input spike'][latency['category'] == 'input']))
                        probability = output_spike_counts / input_spike_counts
                        pair['syn probability'] = probability
                    elif pair['validation'] == 'bad':
                        pair['syn probability'] = np.nan
                data_te['validated_results'] = filtered_data
                with open(pickle_file, 'wb') as f:
                    pickle.dump(data_te,f)     
                processed += 1       
            except Exception as e:
                console.error(f'Error in {filename}: {e}')
                continue
    except Exception as e:
        console.error(f'Error in {filename}: {e}')
        continue

console.info(f'Processed {processed} out of {total} files.')
console.info(f'Already done earlier {already_done} out of {total} files.')
