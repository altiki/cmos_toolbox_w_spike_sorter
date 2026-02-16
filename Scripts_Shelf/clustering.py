import sys

sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')

import os
from src.utils.logger_functions import console
from src.utils.metadata_functions import load_metadata_as_dataframe
from src.cmos_plotter.Latency_and_clustering import ClusteringPlotter


MAIN_PATH = '/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/3_Processed_Data/Ferrans_processed_data_heart/'
INPUT_PATH = '/Latency_plot_by_amplitude/'

if __name__ == "__main__":
    metadata = load_metadata_as_dataframe(file_path=MAIN_PATH)
    for idx, filename in enumerate(metadata.Filename):
        #try:
        if filename == 'ID1742_N0_DIV32_DATE20241217_1138_spontaneous.raw.h5':
            
            print(f"Processing file {filename}...")

            plotter = ClusteringPlotter(
                filename=filename,
                main_path=MAIN_PATH,
                input_path=INPUT_PATH,
                pair_path = None
            )

            plotter.run_clustering_for_all_pairs()

            print("Done!")

        #except Exception as e:
        #    console.warning(f"An error occured while processing file {filename}: {e}")
