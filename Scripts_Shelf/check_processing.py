import sys
sys.path.append('/home/kvulic/Vulic/cmos_toolbox_w_spike_sorter/')
from src.cmos_extractor.Extractor_Helper import find_unmatched_h5_files

# Specify the directory
folder_path = "/itet-stor/amosg/neuronies/gbm_project/3_Student_Projects/Luc/Processed_Data/Spike_Data/Primary_MEA_1/"

if __name__ == "__main__":
    # Get the list of unmatched .h5 files
    unmatched_h5_files = find_unmatched_h5_files(folder_path=folder_path, delete=True)
