#  Import libraries
import sys
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')

import os
from src.utils.logger_functions import console
from src.utils.metadata_functions import load_metadata_as_dataframe
from src.cmos_extractor.Extractor_Class_KVedit_kilosort import SortingExtractor

# Define variables
RECORDINGS_PATH = "/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Small_Network_Plasticity/3_Processed_Data/BCM_Test/241206_test_rec/"
#RECORDINGS_PATH = "/itet-stor/amosg/neuronies/single_neurons/1_Subprojects/Small_Network_Plasticity/2_Raw_Data/Plasticity_Experiments/Raw_Data/BCM_Test/Assess/Splitted/"
OUTPUT_PATH = "/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Small_Network_Plasticity/3_Processed_Data/BCM_Test/241206_sorters_test/"
#OUTPUT_PATH = "/itet-stor/amosg/neuronies/single_neurons/1_Subprojects/Small_Network_Plasticity/2_Raw_Data/Plasticity_Experiments/Raw_Data/BCM_Test/Assess/Splitted/"

if __name__ == "__main__":
    # Load (or create) metadata
    metadata = load_metadata_as_dataframe(file_path=RECORDINGS_PATH)
    for idx, filename in enumerate(metadata.Filename):
        spike_path = os.path.join(OUTPUT_PATH, f"{filename[:-3]}_kilosort_processed.pkl")
        if os.path.isfile(spike_path):
            console.info("File was already processed.")

        else:
        #elif "B5" not in filename:
            try: 
                console.info(f"Processing file {filename}...")
                extractor = SortingExtractor(
                    filename=filename,
                    input_path=RECORDINGS_PATH,
                    output_path=OUTPUT_PATH,
                    div=metadata["DIV"].iloc[idx],
                    id_chip=metadata["Chip_ID"].iloc[idx],
                    id_network=metadata["Network_ID"].iloc[idx],
                    waveform_extraction_method="best_channels",
                    n_jobs=6,
                    num_channels_per_unit=30
                )
    
                # Get spikes
                extractor.load_data()
                extractor.create_dict()

                print("Done!")

            except Exception as e:
                console.warning(f"An error occured while processing file {filename}: {e}")

    
