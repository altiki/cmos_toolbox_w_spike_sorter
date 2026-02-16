#  Import libraries
import sys

sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')



import os
from src.utils.logger_functions import console
from src.utils.metadata_functions import load_metadata_as_dataframe
from src.cmos_extractor.Extractor_Class_KVedit import SortingExtractor



# Define variables
#RECORDINGS_PATH = f"/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Raw_data/Chip{chip_id}/Chip{chip_id}_DIV42/"
#OUTPUT_PATH = f"/itet-stor/kvulic/neuronies/single_neurons/3_Student_Projects/Amelie/Processed_data/Chip{chip_id}/Chip{chip_id}_DIV42/"

#RECORDINGS_PATH = f"/itet-stor/kvulic/neuronies/gbm_project/3_Student_Projects/Luc/Raw_Data/Raw_Traces/Primary_MEA_3/"
#RECORDINGS_PATH = f"/itet-stor/kvulic/neuronies/information_capacity/1_Raw_Data/Electrophysiology/Recordings/DIV30/"
#OUTPUT_PATH = "/itet-stor/kvulic/neuronies/gbm_project/1_Subprojects/Information_Capacity/Recordings/Processed_Data/"
#OUTPUT_PATH = "/itet-stor/kvulic/neuronies/information_capacity/1_Raw_Data/Electrophysiology/Processed_Data/DIV30/"
RECORDINGS_PATH = "/itet-stor/kvulic/neuronies/Karan/Ephys_recordings/For_Kate/Test/"
OUTPUT_PATH = "/itet-stor/kvulic/neuronies/Karan/Ephys_recordings/For_Kate/Processed_Data/"
if __name__ == "__main__":
    # Load (or create) metadata
    metadata = load_metadata_as_dataframe(file_path=RECORDINGS_PATH)
    for idx, filename in enumerate(metadata.Filename):
        spike_path = os.path.join(OUTPUT_PATH, f"{filename[:-3]}_processed.pkl")
        if os.path.isfile(spike_path):
            console.info("File was already processed.")

        else:
        #elif "N3" in filename:
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
                    num_channels_per_unit=25
                )
    
                # Get spikes
                extractor.load_data()
                extractor.create_dict()

                print("Done!")

            except Exception as e:
                console.warning(f"An error occured while processing file {filename}: {e}")

    
