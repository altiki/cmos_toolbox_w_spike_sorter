#  Import libraries
import sys
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')

import os
from src.utils.logger_functions import console
from src.utils.metadata_functions import load_metadata_as_dataframe
from src.cmos_extractor.Extractor_Class_GA import SortingExtractor

from src.cmos_extractor.Extractor_Class_GA import ElectrodeActivityExtractor

# Define variables
RECORDINGS_PATH = "/itet-stor/kvulic/neuronies/Karan/Ephys_recordings/For_Kate/Spontaneous_data/"
#RECORDINGS_PATH = "/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Nono/Raw_Data/Processed_2024_05_10_40Hz_Stim_4x/spont/"
#OUTPUT_PATH = "/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/3_Processed_Data/Nonos_data/Processed_2024_05_10_40Hz_Stim_4x/spont/"
OUTPUT_PATH = "/itet-stor/kvulic/neuronies/Karan/Ephys_recordings/For_Kate/Processed_Data/"

if __name__ == "__main__":
    # Load (or create) metadata
    metadata = load_metadata_as_dataframe(file_path=RECORDINGS_PATH)
    for idx, filename in enumerate(metadata.Filename):
        spike_path = os.path.join(OUTPUT_PATH, f"{filename[:-3]}_processed.pkl")
        if os.path.isfile(spike_path):
            console.info(f"File {filename} was already processed.")

        else:
        #elif metadata["Chip_ID"].iloc[idx] == 2184 or metadata["DIV"].iloc[idx] == 21:
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
                    num_channels_per_unit=15,
                )
                '''
                extractor = ElectrodeActivityExtractor(
                    filename=filename,
                    input_path=RECORDINGS_PATH,
                    output_path=OUTPUT_PATH,
                    div=metadata["DIV"].iloc[idx],
                    id_chip=metadata["Chip_ID"].iloc[idx],
                    id_network=metadata["Network_ID"].iloc[idx],
                    spike_distance = 3,
                )
                '''
                # Get spikes
                extractor.load_data()
                extractor.create_dict()


                print("Done!")

            except Exception as e:
                console.warning(f"An error occured while processing file {filename}: {e}")
                continue
                '''
                try:
                    extractor = ElectrodeActivityExtractor(
                        filename=filename,
                        input_path=RECORDINGS_PATH,
                        output_path=OUTPUT_PATH,
                        div=metadata["DIV"].iloc[idx],
                        id_chip=metadata["Chip_ID"].iloc[idx],
                        id_network=metadata["Network_ID"].iloc[idx],
                        spike_distance = 3,
                    )
                    # Get spikes
                    extractor.load_data()
                    extractor.create_dict()
                    console.info(f'Pickle file for {filename} created without spike sorting.')
                    print("Done!")
                except:
                    console.warning(f"An error occured while processing file {filename}: {e}")
                    console.warning(f"File {filename} was not processed.")
                    continue
                '''
