import numpy as np
import os
import pickle
import logging
from abc import ABC, abstractmethod
import pandas as pd

from src.cmos_plotter.Plotter_Helper import *
from src.utils.logger_functions import console

# Suppress DEBUG messages from matplotlib font manager
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
pd.set_option('mode.chained_assignment', None)


class SpontaneousActivityPreprocessor(ABC):
    def __init__(self, input_path: str, output_path: str):
        self.output_path = output_path
        self.input_path = input_path

        self.data_all = pd.DataFrame()
        self.data_active = pd.DataFrame()

    def convert_files_to_dataframe(self, save_dataframe=True):
        """
        Converts all pkl files (one file per network and DIV) to a single pd.Dataframe.

        :param save_dataframe: If True, saves the DataFrame as a .pkl file.
        :return: Processed pd.DataFrame containing all data.
        """

        # Get a list of all .pkl files in the input path
        all_files = os.listdir(self.input_path)
        pkl_files = [file for file in all_files if file.endswith('.pkl')]

        data_list = []
        columns_list = []
        first_time = True

        # Loop through each .pkl file
        for pkl_file in pkl_files:
            try:
                with open(os.path.join(self.input_path, pkl_file), 'rb') as f:
                    # Load the dictionary from the .pkl file
                    data_dict = pickle.load(f)
                    spike_metrics = data_dict['ELECTRODE_METRICS']
                    network_metrics = data_dict['NETWORK_METRICS']

                    # Loop through the electrodes
                    for el in spike_metrics[:, 0]:
                        el_dict = {}

                        # Extract the information you need from the dictionary
                        for idx, key in enumerate(data_dict):
                            if idx < 6:
                                el_dict[key] = data_dict[key]
                                if first_time:
                                    columns_list.append(key)
                        first_time = False

                        # Add general info
                        el_dict["EL"] = int(el)
                        el_dict["FILENAME"] = pkl_file.split("/")[-1]

                        # Add network metrics
                        col_network = ["NBR", "NBD", "NIBIm", "NIBIstd", "NIBIcv", "NoChannels"]
                        for idx, col in enumerate(col_network):
                            el_dict[col] = float(network_metrics[0][idx])

                        # Add electrode metrics
                        metrics_el = spike_metrics[spike_metrics[:, 0] == el]
                        col_electrodes = ["FR", "ISIm", "ISIstd", "ISIcv", "BR", "BD", "IBIm", "IBIstd", "IBIcv", "BSR"]
                        for idx, col in enumerate(col_electrodes):
                            el_dict[col] = float(metrics_el[0][idx+1])

                        # Append the extracted data to data_list
                        data_list.append(list(el_dict.values()))

            except Exception as e:
                console.error(f"Error processing file {pkl_file}: {e}")

        # Convert the list of NumPy arrays to a dataframe
        columns_list.extend(["EL", "FILENAME"])
        columns_list.extend(col_network)
        columns_list.extend(col_electrodes)
        data_array = np.array(data_list)
        self.data_all = pd.DataFrame(data_array, columns=columns_list)

        # Convert datatypes of columns
        self.convert_column_datatypes()

        # Save dataframe as pkl file
        if save_dataframe:
            output_file_path = os.path.join(self.output_path, 'processed_data.pkl')
            self.data_all.to_pickle(output_file_path)

        console.info(f"Successfully saved the processed data under: {output_file_path}.")

        return self.data_all

    def convert_column_datatypes(self):
        """
        Convert selected columns to appropriate data types.
        """

        # Convert columns to float
        columns_to_convert = self.data_all.columns[[2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
        for column in columns_to_convert:
            self.data_all[column] = self.data_all[column].astype(float)

        # Convert columns to integers
        columns_to_convert_int = self.data_all.columns[[0, 1]]
        for column in columns_to_convert_int:
            self.data_all[column] = self.data_all[column].astype(int)

    def exclude_inactive_networks(self,
                                  save_dataframe=True,
                                  mfr_threshold=0.1,
                                  br_threshold=0.4):
        """
        Exclude inactive networks based on specified thresholds.

        save_dataframe: If True, saves the DataFrame as a .pkl file.
        mfr_threshold: Minimum mean firing rate threshold in Hz.
        br_threshold: Maximum burst rate threshold in bursts per min.

        Returns: Processed pd.DataFrame containing active networks.
        """

        self.data_active = self.data_all.copy()

        # Extract the mean activity of a network on a chip at baseline
        data_mean = self.data_active.groupby(
            ["DIV_NGN", "DIV_GBM", "CHIP_ID", "NW_ID", "GBM_Type"], as_index=False, sort=False
        ).mean(numeric_only=True)
        baseline_data = data_mean[data_mean["DIV_GBM"] == 0]

        # Iterate through the baseline networks
        for _, row in baseline_data.iterrows():
            # Check whether network activity is below the inclusion threshold
            c_id, n_id, mfr, br = row['CHIP_ID'], row['NW_ID'], row['FR'], row["BR"]
            if br < br_threshold or mfr_threshold < mfr_threshold:
                # If the network is not active enough, exclude the network
                self.data_active = self.data_active[
                    ~((self.data_active["NW_ID"] == n_id) & (self.data_active["CHIP_ID"] == c_id))
                ].reset_index(drop=True)

                console.info(
                    f"Network {n_id} on chip {c_id} was excluded as the activity was below the inclusion "
                    f"threshold at baseline (MFR: {mfr}, BR: {br})."
                )

        # Save dataframe as pkl file
        if save_dataframe:
            output_file_path = os.path.join(self.output_path, 'processed_and_included_data.pkl')
            self.data_active.to_pickle(output_file_path)

        console.info(f"Successfully saved the processed and included data under: {output_file_path}.")

        return self.data_active
