import sys
import os
sys.path.append('/itet-stor/amosg/home/cmos_toolbox/')
from src.cmos_extractor.Extractor_Helper import split_hdf5_file

INPUT_PATH = "/itet-stor/amosg/neuronies/single_neurons/1_Subprojects/Small_Network_Plasticity/2_Raw_Data/Plasticity_Experiments/Raw_Data/BCM_Test/Assess/"

if __name__ == "__main__":
    # List all .h5 files in the input directory
    h5_files = [f for f in os.listdir(INPUT_PATH) if f.endswith('.h5')]

    # Process each .h5 file
    for h5_file in h5_files:
        split_hdf5_file(input_path=INPUT_PATH, filename=h5_file, num_splits=5)


