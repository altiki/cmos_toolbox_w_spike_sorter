import os
import pickle
import logging
from PIL import Image
from abc import ABC, abstractmethod
from matplotlib import ticker
import pandas as pd
import numpy as np
import os
from typing import List
import h5py

import spikeinterface.full as si
from matplotlib.colors import LinearSegmentedColormap
from src.cmos_plotter.Plotter_Helper import *
from src.utils.logger_functions import console

# Suppress DEBUG messages from matplotlib font manager
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
pd.set_option('mode.chained_assignment', None)


class ABCPlotter(ABC):
    @abstractmethod
    def __init__(self, input_path: str, output_path: str):
        self.label_size = 21
        self.legend_size = 18
        self.font_size = 24
        self.grid_style = ":"
        self.grid_alpha = 0.75
        self.output_path = output_path
        self.input_path = input_path
        self.dpi = 300

    def save_plot(self, fig, filename):
        fig.savefig(os.path.join(self.output_path, filename + ".pdf"), transparent=True, dpi=self.dpi)
        fig.savefig(os.path.join(self.output_path, filename + ".png"), transparent=True, dpi=self.dpi)

class SpontaneousActivityPlotter(ABCPlotter):
    def __init__(self, filename: str, input_path: str, pairing_path:str, output_path: str):
        super().__init__(input_path=input_path, output_path=output_path)

        self.filename = filename
        self.chip_id = self.filename.split("_")[0].replace("ID", "")
        self.area = self.filename.split("_")[1]
        self.div = self.filename.split("_")[2].replace("DIV", "")
        self.filepath = os.path.join(self.input_path, self.filename)
        self.pairing_path = os.path.join(self.pairing_path, "pairing_of_units.pkl")

        self.sample_frequency = 20000
        self.chip_height = 120
        self.chip_width = 220

        with open(self.filepath, 'rb') as f:
            try:
                self.spike_dict = np.load(f, allow_pickle=True)
                self.spike_mat = self.spike_dict["SPIKEMAT"]
                self.experiment_duration = self.spike_dict["EXPERIMENT_DURATION"]
                self.electrode_numbers = self.spike_dict["SPIKEMAT"][:, 0].astype(int)
                self.normalized_electrodes = self.normalize_electrodes()
                self.spike_times = self.spike_dict["SPIKEMAT"][:, 1].astype(int)
            except Exception as e:
                console.error(f"An error occurred while loading the spike dictionary: {e}")
        with open(self.pairing_path, 'rb') as f:
            self.pairing_dict = np.load(f, allow_pickle=True)
            except Exception as e:
                console.error(f"An error occurred while loading pairings: {e}")