import pickle
import os
import sys
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')
from src.utils.metadata_functions import load_metadata_as_dataframe
from src.cmos_extractor.Extractor_Class_KVedit import ElectrodeActivityExtractor
import argparse

RECORDINGS_PATH = "..."
ELECTRODE_SELECTION_PATH = "..."
OUTPUT_PATH = ".../"


# Load (or create) metadata
metadata = load_metadata_as_dataframe(file_path=RECORDINGS_PATH)


parser = argparse.ArgumentParser(description="Process either stimulated or spontanoues data")
parser.add_argument('mode', type=str, choices=['stim', 'spont'], help="Mode to run the script in: 'stim' or 'spont'")
args = parser.parse_args()

for idx, filename in enumerate(metadata.Filename):
    spike_path = os.path.join(OUTPUT_PATH, f"{filename[:-3]}_processed.pkl")
    filename_parts = filename.split("_")
    if os.path.isfile(spike_path):
        print("File was already processed.")

    else:
        if args.mode == 'spont':
            electrode_dict_path = ''
            blank_block_bool = False
            delay = None
            spike_threshold = 5
        elif args.mode == 'stim':
            electrode_dict_name = f"{filename_parts[0]}_{filename_parts[2]}_electrodes.pkl"
            print("Electrode dictionary: ", electrode_dict_name)
            electrode_dict_path = os.path.join(ELECTRODE_SELECTION_PATH, electrode_dict_name)
            delay = 0
            spike_threshold = 10
            print("Delay: ", delay)
            blank_block_bool = True
        #try:    
        print(f"Processing recording file {filename}...")
        analyzer = ElectrodeActivityExtractor(
            filename = filename,
            input_path = RECORDINGS_PATH,
            output_path = OUTPUT_PATH,
            delay = delay,
            blank_block_bool = blank_block_bool,
            electrode_dict_path = electrode_dict_path,
            div = metadata["DIV"].iloc[idx],
            spike_threshold = spike_threshold,
            #cell_type=metadata["Cell_Type"].iloc[idx],
            id_chip = metadata["Chip_ID"].iloc[idx],
            id_network = metadata["Network_ID"].iloc[idx]
            
            )
        # Get spikes
        analyzer.load_data()
        print("...Loaded!")
        analyzer.create_dict()

        print("...Done!")
