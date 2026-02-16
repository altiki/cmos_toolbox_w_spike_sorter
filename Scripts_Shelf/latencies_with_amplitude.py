import sys

sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')

import os
from src.utils.logger_functions import console
from src.utils.metadata_functions import load_metadata_as_dataframe
from src.cmos_plotter.Latency_and_clustering import LatencyPlotter


MAIN_PATH = '/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/3_Processed_Data/Ferrans_processed_data_heart/'
PROCESSED_DATA_PATH = '/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Ferran/2_Processed_Data/'
PAIRINGS_PATH = '/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Ferran/4_Pairings/'
if __name__ == "__main__":
    metadata = load_metadata_as_dataframe(file_path=MAIN_PATH)
    for idx, filename in enumerate(metadata.Filename):
        try:
        #if a == 0:
            print(f"Processing file {filename}...")

            plotter = LatencyPlotter(
                filename=filename,
                main_path=MAIN_PATH,
                input_path=PROCESSED_DATA_PATH,
                pair_path=PAIRINGS_PATH
            )

            plotter.run_for_all_pairs()

            print("Done!")

        except Exception as e:
            console.warning(f"An error occured while processing file {filename}: {e}")
