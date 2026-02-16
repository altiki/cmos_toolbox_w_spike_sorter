import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')
#from src.utils.logger_functions import console
from src.cmos_plotter.Plotter_Helper_KV import *
from src.utils.metadata_functions import load_metadata_as_dataframe

if __name__ == "__main__":
    MAIN_PATH = f'/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Small_Network_Plasticity/3_Processed_Data/BCM_Test/'
    metadata = load_metadata_as_dataframe(file_path=MAIN_PATH)

    for idx, filename in enumerate(metadata.Filename):
        #console.info(f"Processing file {filename}...")
        try:
        #if filename == 'ID1851_N2_DIV100_DATE20240530_1342_spontaneous_.raw.h5':

            print(f"Processing file {filename}...")
            CHIP_ID = metadata.Chip_ID.iloc[idx]
            DIV = metadata.DIV.iloc[idx]
            area = 'N' + str(metadata.Network_ID.iloc[idx])

            print(MAIN_PATH)
            PROCESSED_DATA_PATH = os.path.join(MAIN_PATH, f'Sorters_old_version/')
           
            PAIRINGS_PATH = os.path.join(MAIN_PATH, f'biTE/')
            OUTPUT_PATH = os.path.join(MAIN_PATH,f'Latency_plots_TE/')
            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)

            SORTER_PATH = PROCESSED_DATA_PATH
            FULL_SORTER_PATH = os.path.join(SORTER_PATH,'Sorter_'+filename)


            data = np.load(os.path.join(PROCESSED_DATA_PATH, f'{filename[:-3]}_processed.pkl'), allow_pickle=True)
            data_te = np.load(os.path.join(PAIRINGS_PATH, f'{filename[:-3]}_processed_info_metrics.pkl'), allow_pickle=True)
            te_unit_pairs = pd.DataFrame(data_te['biTE'])
            pairings = te_unit_pairs

            with open(os.path.join(FULL_SORTER_PATH, 'wf_folder_curated/sparsity.json'), 'r') as file:
                sorting_info = json.load(file)

            unit_ids = sorting_info['unit_ids']
            spikes_extremum = pd.DataFrame(data['SPIKEMAT_EXTREMUM'])
            spikes, electrodes_pre_all, electrodes_post_all, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all = get_electrode_unit_info_te(data, pairings, area, unit_ids)
            print(unit_pre_all, unit_post_all)
            for pre_extremum, post_extremum, electrodes_pre, electrodes_post, unit_pre, unit_post, lag in zip(pre_extremum_all, post_extremum_all, electrodes_pre_all, electrodes_post_all, unit_pre_all, unit_post_all, lag_all):
                
                if unit_pre != unit_post:

                    print(f"Plotting latencies for unit pre {unit_pre}, electrode pre {pre_extremum}, unit post {unit_post}, electrode post {post_extremum}")
        
                    '''
                    original_directionality = os.path.join(OUTPUT_PATH, 'original_directionality/')
                    if not os.path.exists(original_directionality):
                        os.makedirs(original_directionality)
                    inverted_directionality = os.path.join(OUTPUT_PATH, 'inverted_directionality/')
                    if not os.path.exists(inverted_directionality):
                        os.makedirs(inverted_directionality)
                    elec_to_color = plot_color_coded_electrodes(original_directionality, unit_pre, unit_post, electrodes_pre, electrodes_post, area)
                    #plot_latency_and_location(original_directionality, spikes, pre_extremum, post_extremum, elec_to_color, unit_pre, unit_post, area)
                    plot_latency_and_location_with_extremum(original_directionality, spikes, pre_extremum, post_extremum, elec_to_color, unit_pre, unit_post, area, unit_ids, spikes_extremum)
                    '''
                    #elec_to_color = plot_color_coded_electrodes(OUTPUT_PATH, unit_pre, unit_post, electrodes_pre, electrodes_post, filename)
                    #plot_latency_and_location(inverted_directionality, spikes, post_extremum, pre_extremum, elec_to_color, unit_post, unit_pre, area)
                    plot_latency_and_location_with_extremum_without_small(OUTPUT_PATH, filename, pre_extremum, post_extremum, electrodes_pre, electrodes_post, unit_pre, unit_post, unit_ids, spikes_extremum, lag)
                    #else:
                        #print(f"No units found for pair pre {unit_pre} and post {unit_post}")
            print(f"Done for {filename}")

        except:
            #console.error(f"Error processing file {filename}: {e}")
            continue