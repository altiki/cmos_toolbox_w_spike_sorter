import numpy as np
import os
import pickle
from src.utils.electrode_selection_functions_Joel import getElectrodeListsWithSelection


def select_stimulation_electrodes(chip_id: int, div: int, voltage_map: np.array, output_path: str) -> dict:
    recording_electrodes, stimulus_electrodes = getElectrodeListsWithSelection(voltage_map, output_path,
                                                                               loadFileBool=False, n_sample=1020,
                                                                               selection_threshold=15,
                                                                               filename= f"ID{chip_id}_DIV{div}_"
                                                                                         f"subselection")

    # Create dictionary
    electrode_selection = {"recording_electrodes": recording_electrodes, "stimulus_electrodes": stimulus_electrodes}

    # Save the dictionary to a pickle file
    file_path = os.path.join(output_path, f"ID{chip_id}_DIV{div}_electrodes.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(electrode_selection, f)

    return electrode_selection

