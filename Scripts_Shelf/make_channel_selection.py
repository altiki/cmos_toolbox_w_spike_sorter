from __future__ import annotations

import os
import pickle
import numpy as np
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors


class RecOrganizer:
    """  This class handles the reading for one single recording file, performs basic processing steps such as
    reading in, extracting meta info, extracting spikes, saving processed data to output file (.h5). """

    def __init__(self, recpath: str, voltage_map: np.array):

        self._recpath = recpath
        self._hdf = h5.File(self._recpath, 'r')
        self.voltage_map = voltage_map

        self._conversion_factor = self._hdf.get("/settings/lsb")[0] * 1e6
        self._no_channels = 1028
        if self._no_channels != self._hdf["sig"].shape[0]:
            self._no_channels = self._hdf["sig"].shape[0]
            raise Warning(
                f"Unexpected number of channels. Found {self._hdf['sig'].shape[0]}, expected {self._no_channels}. "
                f"Adjusting.")

        self.fs = 20e3
        self.rms_coeff = 5
        self._baseline = 30  # in uV
        self._low_cut = 300  # cut off frequency for high pass filter
        self._high_cut = 3500
        self.pitch = 17.5
        self.arr_width = 120  # in um
        self.arr_len = 220
        self.map_size = self.arr_len * self.arr_width
        self.pre = 1e-3  # in sec
        self.post = 1.5e-3  # in sec

        self.signal = self._hdf.get('sig')  # None object if path does not exist
        self.is_signal = bool(self.signal) and self.signal.size > 1
        self.first_frameno = self.get_rel_frameno(self.signal)

        self.mapping = self.get_map_info(self._hdf)
        self.spikes = self.get_spike_info(self._hdf)

    def get_map_info(self, hdf):

        map_hdf = hdf.get('/mapping/')[()]
        map = {'channel': map_hdf['channel'],
               'electrode': map_hdf['electrode'],
               'x': map_hdf['x'],
               'y': map_hdf['y'],
               'row_ind': [int(i / self.pitch) for i in map_hdf['y']],
               'col_ind': [int(i / self.pitch) for i in map_hdf['x']]}

        return map

    def get_rel_frameno(self, raw_sig):

        if self.is_signal:  # if full traces saved
            raw_frameno = [raw_sig[-2, 0], raw_sig[-1, 0]]
            first_frameno = np.uint32(
                (raw_frameno[1] << 16) | raw_frameno[0]) + 1  # not sure why +1 but it centered spikes for sure
        else:
            first_frameno = self.spikes['frameno'][0] - 1
        return first_frameno

    def get_spike_info(self, hdf):

        spikes_data = hdf.get('/proc0/spikeTimes/')[()]
        spikes_dict = {'frameno': spikes_data['frameno'] - self.first_frameno,
                       'channel': spikes_data['channel'],
                       'amplitude': - spikes_data['amplitude'] * self._conversion_factor}

        # convert to dict for better handling when matching chan <--> electrode
        spikes = pd.DataFrame.from_dict(spikes_dict)
        mapping_df = pd.DataFrame.from_dict(self.mapping)
        # Match info from mapping (electrode, x, y)
        spikes = pd.merge(spikes, mapping_df, on='channel', sort=False)
        # Sort table by increasing framenos
        spikes.sort_values(by=['frameno'], ascending=True, inplace=True)
        # Remove negative spike times (corresponds to "ghost" zone in between switch matrix configurations)
        spikes.drop(spikes[spikes.frameno < 0].index, inplace=True)

        return spikes

    def click_electrode_area(self) -> tuple[list, list[int]]:

        img = self.plot_voltage_map()

        clicked_points = []
        limit = 20
        cursor = mplcursors.cursor(img, hover=False)

        @cursor.connect("add")
        # the cursor click function is passed as a callback function as argument to the cusrsor.connect function
        def cursor_clicked(sel):
            clicked_points.append(sel.index)
            print(clicked_points)
            if len(clicked_points) == limit:  # check if the limit is reached
                print(f"You have reached the limit of {limit} clicks.")
                plt.close()  # close the plot

        plt.show()

        # Remove duplicates (due to extra clicks to reach the limit for example)
        unique_clicked_points = uniqify(clicked_points)

        merged_lists = merge(self.mapping['row_ind'], self.mapping['col_ind'])
        try:
            indices = [merged_lists.index(i) for i in unique_clicked_points]
            electrodes = [int(self.mapping['electrode'][i]) for i in indices]

            return indices, electrodes

        except ValueError:
            print("Electrode was not in list. Restarting channel selection.")
            indices, electrodes = self.click_electrode_area()
            return indices, electrodes

    def plot_voltage_map(self, extra_electrode=None, show=False):

        img = plt.imshow(self.voltage_map)

        # Plot recorded electrodes on top of voltage map
        plt.scatter(self.mapping['col_ind'], self.mapping['row_ind'], c="black",
                    marker='.')

        if extra_electrode:
            indices = [list(self.mapping['electrode']).index(i) for i in extra_electrode]
            plt.scatter([self.mapping['col_ind'][i] for i in indices],
                        [self.mapping['row_ind'][i] for i in indices],
                        c="red", marker='.')

        ax_map = plt.gca()
        ax_map.set_title("Voltage map")

        if show:
            plt.show()

        return img


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


def uniqify(seq, idfun=None):
   # make a list of unique elements while preserving order
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result


def main():
    CHIP_ID = 1689
    NO_SELECTIONS = 22

    INPUT_PATH =  "Z:/gbm_project/1_Subprojects/2D_GBM_Coculture/2_Raw_Data/Microstructure_Activity/MS_Activity_5/Recordings"
    VOLTAGE_MAPS_PATH = "Z:/gbm_project/1_Subprojects/2D_GBM_Coculture/2_Raw_Data/Microstructure_Activity/MS_Activity_5/Voltage_Maps"
    OUTPUT_PATH = "Z:/gbm_project/1_Subprojects/2D_GBM_Coculture/3_Processed_Data/Microstructure_Activity/MS_Activity_5/Electrode_Selections"

    all_files = os.listdir(INPUT_PATH)
    files_to_process = [file_name for file_name in all_files if str(CHIP_ID) in file_name]
    files_to_process = [file_name for file_name in files_to_process if "DIV28" in file_name]
    print(files_to_process)

    all_maps = os.listdir(VOLTAGE_MAPS_PATH)
    map_to_process = [file_name for file_name in all_maps if str(CHIP_ID) in file_name][0]

    # Iterate through files
    for file in files_to_process[8:]:
        dict_subselections = {}
        file_path = os.path.join(INPUT_PATH, file)

        # Get voltage map
        voltage_map = np.load(os.path.join(VOLTAGE_MAPS_PATH, map_to_process))

        for i in range(NO_SELECTIONS):
            organizer = RecOrganizer(recpath=file_path, voltage_map=voltage_map)
            _, electrodes = organizer.click_electrode_area()

            # Add to dictionary
            dict_subselections[f"Channel_{i}"] = {"Electrodes": electrodes}
            print(f"Electrodes of Channel {i} were selected.")

        # Save the dictionary
        output_path = os.path.join(OUTPUT_PATH, f"Channel_Subselections_{file[:-4]}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(dict_subselections, f)

        print(f"Electrodes of Network were selected and saved to {output_path}.")

        print("Dictionary:\n")
        print(dict_subselections)

        print(f"File {file} was successfully processed and saved.")


if __name__ == "__main__":
    main()

    # run "python .\Notebook_Shelf\make_channel_selection.py" in terminal