#  Import libraries
import sys
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')

import os
from src.utils.logger_functions import console
from src.utils.metadata_functions import load_metadata_as_dataframe
from src.cmos_extractor.Extractor_Class_GA import SortingExtractor

from src.cmos_extractor.Extractor_Class_GA import ElectrodeActivityExtractor


# Define variables
RECORDINGS_PATH = "/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Raw_data/Chip1908/Chip1908_DIV56/"
#RECORDINGS_PATH = "/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Nono/Raw_Data/Processed_2024_05_10_40Hz_Stim_4x/spont/"
#OUTPUT_PATH = "/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/3_Processed_Data/Nonos_data/Processed_2024_05_10_40Hz_Stim_4x/spont/"
OUTPUT_PATH = "/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Processed_data/Sorters"
chip_ids =[1867, 1870, 1873, 1876, 1877, 1908, 2034, 2070, 2074]


if __name__ == "__main__":
    filename = sys.argv[1]
    filename_parts = filename.split("_")
    div = filename_parts[2].replace("DIV", "")
    chip_id = filename_parts[0].replace("ID", "")
    network_id = filename_parts[1].replace("N", "")

    spike_path = os.path.join(OUTPUT_PATH, f"{filename[:-3]}_processed.pkl")
    if os.path.isfile(spike_path):
        console.info(f"File {filename} was already processed.")
    else:
        try:
            console.info(f"Processing file {filename}...")
            
              
                        
            extractor = SortingExtractor(
                filename=filename,
                input_path=RECORDINGS_PATH,
                output_path=OUTPUT_PATH,
                div=div,
                id_chip=chip_id,
                id_network= network_id,
                waveform_extraction_method="best_channels",
                n_jobs=6,
                num_channels_per_unit=25,
            )
            # Get spikes
            extractor.load_data()
            extractor.create_dict()


            print("Done!")

        except Exception as e:
            console.warning(f"An error occured while processing file {filename}: {e}")
            pass
