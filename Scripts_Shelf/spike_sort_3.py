#  Import libraries
import sys
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')

import os
from src.utils.logger_functions import console
from src.utils.metadata_functions import load_metadata_as_dataframe
from src.cmos_extractor.Extractor_Class_GA import SortingExtractor

from src.cmos_extractor.Extractor_Class_GA import ElectrodeActivityExtractor

# Define variables
RECORDINGS_PATH = "/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Raw_data/"
#RECORDINGS_PATH = "/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Nono/Raw_Data/Processed_2024_05_10_40Hz_Stim_4x/spont/"
#OUTPUT_PATH = "/itet-stor/kvulic/neuronies/single_neurons/1_Subprojects/Neurons_As_DNNs/3_Processed_Data/Nonos_data/Processed_2024_05_10_40Hz_Stim_4x/spont/"
OUTPUT_PATH = "/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Processed_data/Sorters"
chip_ids =[1867, 1870, 1873, 1876, 1877, 1908, 2034, 2070, 2074]

# Get list of directories in the specified path
def get_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

directories = get_directories(RECORDINGS_PATH)
console.info(f"Directories in {RECORDINGS_PATH}: {directories}")


if __name__ == "__main__":
    for chip_id in chip_ids:
        if chip_id == 1908:
            directories = get_directories(os.path.join(RECORDINGS_PATH, f'Chip{chip_id}/'))
            for directory in directories:
                RECORDINGS_FULL_PATH = os.path.join(RECORDINGS_PATH, f'Chip{chip_id}/', f'{directory}/')
                # Load (or create) metadata
                metadata = load_metadata_as_dataframe(file_path=RECORDINGS_FULL_PATH)
                for idx, filename in enumerate(metadata.Filename):
                    spike_path = os.path.join(OUTPUT_PATH, f"{filename[:-3]}_processed.pkl")
                    if os.path.isfile(spike_path):
                        console.info("File was already processed.")

                    else:
                        try: 
                            console.info(f"Processing file {filename}...")
                            
                            extractor = SortingExtractor(
                                filename=filename,
                                input_path=RECORDINGS_FULL_PATH,
                                output_path=OUTPUT_PATH,
                                div=metadata["DIV"].iloc[idx],
                                id_chip=metadata["Chip_ID"].iloc[idx],
                                id_network=metadata["Network_ID"].iloc[idx],
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
                            continue

                    #If you want to use the ElectrodeActivityExtractor instead of the SortingExtractor, uncomment the following lines:    
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
