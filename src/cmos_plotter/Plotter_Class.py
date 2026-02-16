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
        fig.savefig(os.path.join(self.output_path, filename + ".svg"), transparent=True, dpi=self.dpi)
        fig.savefig(os.path.join(self.output_path, filename + ".png"), transparent=True, dpi=self.dpi)


class SpontaneousActivityPlotter(ABCPlotter):
    def __init__(self, filename: str, input_path: str, output_path: str):
        super().__init__(input_path=input_path, output_path=output_path)

        self.filename = filename
        self.filepath = os.path.join(self.input_path, self.filename)

        self.sample_frequency = 20000
        self.chip_height = 120
        self.chip_width = 220

        self.most_act_el_per_channel = {}
        self.cross_correlation_delays = None
        self.cross_correlation_matrix = None

        with open(self.filepath, 'rb') as f:
            try:
                self.spike_dict = pickle.load(f)
                self.spike_mat = self.spike_dict["SPIKEMAT"]
                self.experiment_duration = self.spike_dict["EXPERIMENT_DURATION"]
                self.electrode_numbers = self.spike_dict["SPIKEMAT"][:, 0].astype(int)
                self.normalized_electrodes = self.normalize_electrodes()
                self.spike_times = self.spike_dict["SPIKEMAT"][:, 1].astype(int)
            except Exception as e:
                console.error(f"An error occurred while loading the spike dictionary: {e}")

    def plot_STTRP(self,
                   triggering_electrodes: list,
                   electrode_subselection=None,
                   window: (int, int) = (0, 10),
                   voltage_map=None
                   ):
        """

        :param voltage_map:
        :param channel:
        :param triggering_electrodes:
        :param electrode_subselection
        :param window:
        :return:
        """
        colours = None

        for triggering_el in triggering_electrodes:
            responses = self.get_responses(triggering_electrode=triggering_el,
                                           electrode_subselection=electrode_subselection,
                                           window=window)

            if responses is not None:
                fig = plt.figure(figsize=(5, 1.5))

                # Get and plot STTRP image
                image, colour_map = self.response_delay_over_time(responses=responses, nr_of_bins=200)
                plt.imshow(image, aspect='auto')
                plt.xlabel("Latency [ms]")
                plt.ylabel("Spike No.")

                # Apply the custom ticker function to the x-axis
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_ticks))

                fig.tight_layout()

                # Save the plot
                self.save_plot(fig=fig, filename=f"STTRP_{self.filename}_Trigger{triggering_el}")

        if voltage_map is not None:
            self.plot_voltage_map(voltage_map=voltage_map,
                                  channel_electrodes=electrode_subselection,
                                  color_mapping=colour_map
                                  )

        console.info(f"STTRPs for {self.filename} was created and saved successfully.")

        return colour_map
    
    def convert_elno_to_xy(elno):
        chipWidth = 220
        x = int(elno/chipWidth)
        y = elno % chipWidth
        return x,y

    def response_delay_over_time(self,
                                 responses: np.ndarray,
                                 colour_coding=RainbowColourCoding(),
                                 window: int = 10,
                                 nr_of_bins: int = None
                                 ):
        """
        Create an image of a raster plot as a numpy array of shape (H,W,3).
        :param responses: An array of shape (4,n_spikes) with the first dimension corresponding to
        (electrodes, delay, amplitude, response). A list allows the processing of multiple experiments into one plot.
        :param colour_coding: A class that, when called, maps an array of electrodes to colours.
        :param window: The response window at which to look at.
        :param nr_of_bins: In how many bins the responses are sorted. Default is the number of samples in the window.
        :return: A numpy array resembling an image of shape (H,W,3).
        """
        # Specify the number of bins to plot
        if nr_of_bins is None:
            nr_of_bins = (window * self.sample_frequency) / 1000

        # Initialize variables and image
        nr_of_inputs = int(responses[3, :].max())
        image = np.zeros((nr_of_inputs + 1, nr_of_bins + 1, 3))
        avg_to_take = np.zeros(image.shape)
        colours = None

        # Extract the spikes that are within the specified window
        valid_spikes = np.copy(responses[:, np.where(responses[1] < window)[0]])

        if valid_spikes.shape[1] != 0:
            response_index = (valid_spikes[3]).astype(int)
            delay_index = np.round(valid_spikes[1] * nr_of_bins / window).astype(int)

            # Get all the electrodes and color code them
            electrodes = valid_spikes[0, :]
            colours = colour_coding(electrodes)

            # Create a color map for later use
            colour_map = {electrode: color for electrode, color in zip(electrodes, colours)}

            # Change the color of the pixel at the corresponding location
            image[response_index, delay_index] = colours[np.newaxis, :, :3]
            avg_to_take[response_index, delay_index] += 1

        # Default 1 to avoid dividing through 0
        image[np.where(avg_to_take == 0)] = 1
        avg_to_take[np.where(avg_to_take == 0)] = 1
        image /= avg_to_take
        image = np.flip(image, 0)

        return image, colour_map

    def get_responses(
            self,
            triggering_electrode: int,
            electrode_subselection=None,
            min_spike_no=50,
            window: (int, int) = (0, 10),
    ) -> (np.ndarray, np.ndarray):
        """
        Based on an array containing the start timings of a response, the spikes are labeled with the corresponding response.
        :param triggering_electrode: Electrode that should be used as the triggering electrode.
        :param electrode_subselection: List of electrodes that should be used for the response.
        :param window: Window, in which a spike has to occur such that it belongs to a specific response start.
        :return: A numpy array of shape (4,n_filtered_spikes), where the first dimension corresponds to
        (channel, delay, amplitude, response) and the count of spikes per channel.
        """

        # Only extract electrodes in subselection
        if electrode_subselection is not None:
            self.make_electrode_subselection(electrode_subselection)

        spike_mat = self.spike_mat

        # Extract spike information from spike dict
        triggering_spikes = spike_mat[spike_mat[:, 0] == float(triggering_electrode)][:, 1]
        electrodes = spike_mat[:, 0]
        spike_times = spike_mat[:, 1]
        amplitudes = spike_mat[:, 2]

        # Check that triggering spikes were found
        if len(triggering_spikes) < min_spike_no:
            console.warning(f"Spike number insufficient for triggering electrode {triggering_electrode}.")
            responses = None

        # Check that window was defined properly
        elif np.min(triggering_spikes) < window[0] or window[1] <= window[0]:
            console.warning("Inappropriate window.")
            responses = None

        else:
            # Initialize variables
            responses = np.zeros([4, spike_times.shape[0]])
            start_index = 0
            last_start = -np.inf

            # Iterate through triggering spikes and extract window
            for i, start in enumerate(triggering_spikes.astype(dtype=np.int64)):
                # Check that the triggering spike is sufficiently far from the previous spike
                if start - last_start > 2:
                    # Get responses that lie in the defined window
                    indices = \
                    np.where(np.logical_and(start - window[0] <= spike_times, spike_times < start + window[1]))[0]
                    if len(indices) != 0:
                        try:
                            responses[0, start_index:start_index + len(indices)] = electrodes[indices]
                            responses[1, start_index:start_index + len(indices)] = spike_times[indices] - start
                            responses[2, start_index:start_index + len(indices)] = amplitudes[indices]
                            responses[3, start_index:start_index + len(indices)] = i
                        except ValueError as e:
                            console.warning(f"The response indices do not match. Error: {e}")

                    start_index += len(indices)
                    last_start = start

                else:
                    console.info("Response kicked due to being to close to previous.")

            responses = responses[:, :start_index]

        return responses

    def plot_voltage_map(self, voltage_map: np.array, channel_electrodes=None, color_mapping=None, plot_color_map=True):
        fig, ax = plt.subplots()

        # Threshold the values for better visibility
        mask = voltage_map > 40
        voltage_map[mask] = 1
        voltage_map[~mask] = 0
        img = plt.imshow(voltage_map, cmap="gray")

        ax_map = plt.gca()
        ax_map.set_title("Voltage map")

        if channel_electrodes:
            try:
                x = [i % voltage_map.shape[1] for i in channel_electrodes]
                y = [i // voltage_map.shape[1] for i in channel_electrodes]
                colors = [color_mapping[i] for i in channel_electrodes]
                plt.scatter(x, y, c=colors, marker='.')
            except NameError as e:
                console.warning(f"No color map was provided for the channel electrodes. Electrodes were plotted in red"
                                f" Error: {e}")
                plt.scatter(x, y, c="red", marker='.')
            except TypeError as e:
                console.warning(f"No channel electrodes were provided. No electrodes can be plotted."
                                f" Error: {e}")
                x = [0]
                y = [0]
            except KeyError as e:
                console.warning(f"The color for the channel electrode could not be found."
                                f" Error: {e}")

            xlim = [min(x) - 15, max(x) + 15]
            ylim = [min(y) - 15, max(x) + 15]

            ax_map.set_xlim(xlim[0], xlim[1])
            ax_map.set_ylim(ylim[0], ylim[1])

        # save the image
        self.save_plot(fig, f"Voltage_Map_{self.filename}")

        if plot_color_map:
            fig, ax = plt.subplots()

            # Create a new dictionary with sorted keys
            sorted_keys = sorted(color_mapping.keys())
            sorted_color_map = {key: color_mapping[key] for key in sorted_keys}

            # Create a scatter plot for each electrode and color and annotate the scatters
            pos = 1
            for electrode, color in sorted_color_map.items():
                plt.scatter(pos, 0, color=color, s=3000, edgecolors='white')  # Adjust the y-coordinate as needed
                plt.text(pos, 0, str(electrode), ha='center', va='center', color='white')

                pos += 1

            # Customize plot
            plt.title('Electrode Colors')
            plt.xlabel('Electrode Number')
            plt.yticks([])  # Hide y-axis
            plt.grid(axis='x')  # Show grid on x-axis
            plt.tight_layout()
            plt.gcf().set_size_inches(len(sorted_color_map) * 0.9, 1)

            # save the image
            self.save_plot(fig, f"Color_Map_{self.filename}")

        return img

    def make_electrode_subselection(self, electrode_subselection: np.array):
        # Only extract electrodes in subselection
        if electrode_subselection is None:
            console.warning("Please provide the electrodes to be used by specifying the param electrode_subselections.")
        else:
            mask = np.isin(self.spike_mat[:, 0], electrode_subselection)
            self.spike_mat = self.spike_mat[mask]

    def plot_spike_raster_plot(self, electrode_subselection=None):
        if electrode_subselection is not None:
            self.make_electrode_subselection(electrode_subselection)
            self.electrode_numbers = self.spike_mat[:, 0].astype(int)
            self.spike_times = self.spike_mat[:, 1].astype(int)

        sns.set_style("ticks")
        self.font_size = 12

        fig, ax = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [1.75, 1.25]})

        firing_rate, bin_centers = self.get_firing_rates()
        image = self.get_spike_image()

        if image is None:
            console.warning(f"No rasterplot image could be created for {self.filename}.")

        else:
            ax[0].imshow(image, cmap="grey", aspect='auto')
            sns.lineplot(x=bin_centers, y=firing_rate, color='black', ax=ax[1])
            ax[1].fill_between(x=bin_centers, y1=firing_rate, y2=0, color='black')

            for i in range(0, 2):
                ax[i].yaxis.set_major_locator(ticker.MaxNLocator(4))
                ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
                ax[i].spines['right'].set_visible(False)
                ax[i].spines['top'].set_visible(False)

            ax[0].set_ylabel("Electrode No.", fontsize=self.font_size)
            ax[0].set_xticks([])
            ax[0].set_xlim(0.01)
            ax[0].set_xlabel(" ", fontsize=self.font_size)
            ax[0].set_ylim(0, max(self.normalized_electrodes))
            ax[0].spines['left'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)

            ax[1].set_ylabel("Firing Rate [kHz]", fontsize=self.font_size)
            ax[1].set_xlim(0.01, self.experiment_duration)
            ax[1].set_ylim(0, np.sort(firing_rate)[-2])
            ax[1].set_xlabel("Time [s]", fontsize=self.font_size)

            fig.tight_layout()

            # Save the plot
            self.save_plot(fig=fig, filename=f"Spike_Rasterplot_{self.filename}")

            console.info(f"Spike rasterplot for {self.filename} was created and saved successfully.")

    def get_spike_image(self, bin_size=50):
        # Round each spike time to the nearest multiple of bin_size
        rounded_spikes = np.round(self.spike_times / bin_size).astype(int)

        try:
            # Create a white image of the appropriate size
            nr_of_bins = int(self.experiment_duration * 1e3 / bin_size) + 1
            nr_of_electrodes = max(self.normalized_electrodes)
            image = np.ones((nr_of_electrodes + 1, nr_of_bins + 1))

            # Set the pixels corresponding to spike positions to 0 (black)
            image[self.normalized_electrodes, rounded_spikes] = 0

            return image

        except ValueError as e:
            console.warning(f"The image is empty. Error: {e}.")

            return None

    def get_firing_rates(self, time_window=50):
        # Set the bin edges based on time_windows
        bin_num = int(self.experiment_duration * 1e3 / time_window) - 1
        bin_edges = np.linspace(0, int(self.experiment_duration), num=bin_num)
        hist, edges = np.histogram(self.spike_times / 1e3, bins=bin_edges)

        # Calculate the bin centers
        bin_centers = (edges[:-1] + edges[1:]) / 2

        # Calculate firing rate by dividing the histogram values by the bin width
        bin_width = bin_edges[1] - bin_edges[0]
        firing_rate = (hist / bin_width) / 1e3  # kHz

        return firing_rate, bin_centers

    def normalize_electrodes(self):
        # Normalize the electrode_numbers
        normalized_numbers = [i + 1 for i in range(len(set(self.electrode_numbers)))]
        mapping_dict = dict(zip(sorted(set(self.electrode_numbers)), normalized_numbers))
        normalized_electrodes = [mapping_dict[num] for num in self.electrode_numbers]

        return normalized_electrodes

    def find_most_act_el_per_channel(self, channel_info: dict):
        # Iterate through the channels
        for channel, info in channel_info.items():
            electrodes = info["Electrodes"]
            max_firing_rate = -np.inf
            best_electrode = None

            # Iterate through electrodes in the current channel
            for electrode in electrodes:
                # Find the index of the electrode in the electrode_metrics matrix
                index = np.where(self.spike_dict["ELECTRODE_METRICS"][:, 0] == electrode)[0]
                if len(index) > 0:  # Ensure electrode exists in the matrix
                    firing_rate = self.spike_dict["ELECTRODE_METRICS"][
                        index, 1]  # Get the firing rate for the electrode
                    if firing_rate > max_firing_rate:
                        max_firing_rate = firing_rate
                        best_electrode = electrode

            # Store the electrode with the highest firing rate for the current channel
            self.most_act_el_per_channel[channel] = {'Electrode': best_electrode, 'Firing Rate': max_firing_rate}

        return self.most_act_el_per_channel

    def get_binary_traces_per_channel(self, channel_info: dict, bin_size=5, nr_of_channels=22):
        # Get most active electrodes in each channel, if necessary
        if not self.most_act_el_per_channel:
            self.find_most_act_el_per_channel(channel_info)

        # Create an empty matrix to store binary traces for each channel
        nr_of_bins = int(self.experiment_duration * 1e3 / bin_size) + 1
        binary_matrix = np.zeros((nr_of_channels, nr_of_bins + 1))
        channel_no = 0

        # Iterate through each channel in highest_spike_frequency_per_channel
        for channel, data in self.most_act_el_per_channel.items():
            # Get all spikes that occurred on the electrode in the channel
            electrode = data['Electrode']
            rows_for_electrode = self.spike_mat[self.spike_mat[:, 0] == electrode]
            spikes_on_electrode = rows_for_electrode[:, 1]

            # Round each spike time to the nearest multiple of bin_size
            rounded_spikes = np.round(spikes_on_electrode / bin_size).astype(int)

            # Set the pixels corresponding to spike positions to 0 (black)
            binary_matrix[channel_no, rounded_spikes] = 1

            channel_no += 1

        return binary_matrix

    def get_cross_correlation(self, channel_info: dict, save_max_corr=False):
        # Get point processes (binary spike traces)
        binary_matrix = self.get_binary_traces_per_channel(channel_info=channel_info)

        # Initialize the cross-correlation matrix
        self.cross_correlation_matrix = np.zeros((binary_matrix.shape[0], binary_matrix.shape[0]))
        self.cross_correlation_delays = np.zeros((binary_matrix.shape[0], binary_matrix.shape[0]))

        for i in range(binary_matrix.shape[0]):
            # Iterate through channels that are equal or greater than i to avoid redundant calculations
            for j in range(i, binary_matrix.shape[0]):
                # Calculate and normalize the cross correlation to be able to compare the similarity between
                # different spike trains. By dividing by the square root of the autocorrelation values of the two
                # individual spike trains at time lag 0 (which corresponds to the maximum overlap), we normalize
                # the cross-correlation value. This scales the values to the range [-1, 1],
                # where 0 indicates no correlation.
                n = len(binary_matrix[i, :])

                corr = (np.correlate(binary_matrix[j, :], binary_matrix[i, :], mode='same') /
                        np.sqrt(np.correlate(binary_matrix[i, :], binary_matrix[i, :], mode='same')[int(n / 2)] *
                                np.correlate(binary_matrix[j, :], binary_matrix[j, :], mode='same')[int(n / 2)]))

                # Prepare a delay array to get the lag
                delay_arr = np.linspace(-0.5 * n / self.sample_frequency, 0.5 * n / self.sample_frequency, n)

                # Save the value in the matrices
                self.cross_correlation_matrix[i, j] = np.max(corr)
                self.cross_correlation_delays[i, j] = delay_arr[np.argmax(corr)]

        if save_max_corr:
            network_metrics = self.spike_dict["NETWORK_METRICS"]
            values_to_add = np.array([[self.cross_correlation_matrix.mean()]])
            new_array = np.hstack((network_metrics, values_to_add))
            self.spike_dict["NETWORK_METRICS"] = new_array

            try:
                # Add "_with_corr.pkl" to the filename
                new_filepath = os.path.splitext(self.filepath)[0] + "_with_corr.pkl"
                with open(new_filepath, 'wb') as f:
                    pickle.dump(self.spike_dict, f)
                print(f"Spike_dict saved with correlations: {new_filepath}")
                return True
            except Exception as e:
                print(f"Error saving spike_dict with correlations: {e}")
                return False

        return self.cross_correlation_matrix, self.cross_correlation_delays

    def plot_correlation_matrix(self):
        # Check if cross correlation matrix was created
        if self.cross_correlation_matrix.size == 0:
            console.warning("Calculate cross correlation matrix first by running get_cross_correlation.")
            return

        # Create a full matrix with zeros
        full_matrix = np.zeros_like(self.cross_correlation_matrix)

        # Fill the upper triangle with the values from the half matrix
        i, j = np.triu_indices(self.cross_correlation_matrix.shape[0], k=1)
        full_matrix[i, j] = self.cross_correlation_matrix[i, j]

        # Mirror the upper triangle to the lower triangle
        full_matrix[j, i] = self.cross_correlation_matrix[i, j]

        # Fill diagonal elements with 1s
        np.fill_diagonal(full_matrix, 1)

        # Create a custom colormap with reversed colors
        colors = [(0.3, 0.3, 1), (0, 0, 0.5), (0, 0, 0), (0.5, 0, 0),
                  (1, 0.3, 0.3)]  # Blue, dark blue, black, dark red, red
        cm = LinearSegmentedColormap.from_list('custom', colors, N=256)

        # Plot the cross-correlation matrix
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(full_matrix, vmin=-1, vmax=1, center=0, cmap=cm, square=True)
        plt.tick_params(left=False, bottom=False)
        plt.title('Cross-Correlation Matrix between Spike Trains', fontsize=16)
        plt.xlabel('Channel Index', fontsize=13)
        plt.ylabel('Channel Index', fontsize=13)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Save the plot
        self.save_plot(fig=fig, filename=f"Correlation_Matrix_{self.filename}")


    


class MetricsPlotter(ABCPlotter):
    def __init__(self, input_path: str, df_name: str, output_path: str, multiple_exp=False):
        super().__init__(input_path=input_path, output_path=output_path)

        try:
            with open(os.path.join(self.input_path, df_name), 'rb') as f:
                self.data_all = pickle.load(f)

            self.multiple_exp = multiple_exp
            self.data_relative = self.get_relative_values()

        except Exception as e:
            console.error(f"Error opening file {df_name}: {e}")

    def get_relative_values(self):
        # Get the mean values of each metric per network
        data_mean = self.data_all.groupby(
            ["DIV_NGN", "DIV_GBM", "CHIP_ID", "NW_ID", "GBM_Type"], as_index=False, sort=False
        ).mean(numeric_only=True)
        self.data_relative = data_mean.copy()

        # Get the mean value at baseline
        baseline_df = data_mean[data_mean["DIV_GBM"].astype(float) == 0.0]
        # Fill NaN values, as difference returns NaN if one of the values is NaN
        baseline_df = baseline_df.fillna(0)

        # Iterate through all networks
        chip_ids = baseline_df["CHIP_ID"].unique()
        nw_ids = baseline_df["NW_ID"].unique()
        if self.multiple_exp:
            exp_ids = baseline_df["EXP_ID"].unique()
        for chip_id in chip_ids:
            for nw_id in nw_ids:
                if not self.multiple_exp:
                    # Extract the corresponding baseline
                    baseline = baseline_df[(baseline_df["NW_ID"] == nw_id) & (baseline_df["CHIP_ID"] == chip_id)]
                    if len(baseline) != 0:
                        # Subtract the baseline value
                        for col in self.data_relative.columns[5:]:
                            self.data_relative.loc[(self.data_relative["NW_ID"] == nw_id) &
                                                   (self.data_relative["CHIP_ID"] == chip_id), col] -= \
                            baseline[col].iloc[0]

                if self.multiple_exp:
                    for exp_id in exp_ids:
                        # Extract the corresponding baseline
                        baseline = baseline_df[(baseline_df["NW_ID"] == nw_id)
                                               & (baseline_df["CHIP_ID"] == chip_id)
                                               & (baseline_df["EXP_ID"] == exp_id)
                                               ]
                        if len(baseline) != 0:
                            # Subtract the baseline value
                            for col in self.data_relative.columns[5:]:
                                self.data_relative.loc[(self.data_relative["NW_ID"] == nw_id) &
                                                       (self.data_relative["CHIP_ID"] == chip_id) &
                                                       (self.data_relative["EXP_ID"] == exp_id), col] -= \
                                    baseline[col].iloc[0]

        return self.data_relative

    def plot_metrics_over_time(self, use_relatives=True, use_channels=False, font_size=12, plot_mean=True, polynomial=False):
        # Get the mean values of each metric per network (the networks are the "biological units")
        if use_relatives:
            prefix = "Change in "
            df = self.data_relative.copy()
        else:
            prefix = ""
            df = self.data_all

        if plot_mean:
            # Get the mean over each network
            df = df.groupby(["DIV_NGN", "DIV_GBM", "CHIP_ID", "NW_ID", "GBM_Type"], as_index=False, sort=False
                            ).mean(numeric_only=True)

        # Convert columns to floats
        columns_to_convert = df.columns[[1, 2]]
        for column in columns_to_convert:
            df[column] = df[column].astype(float)

        # Plot the metrics over time
        fig, ax = plt.subplots(3, 4, figsize=(16, 10))

        # Define palette (MOVE TO CONFIGS LATER)
        if use_channels:
            palette = {
                "control": "teal",
                "BG5_Cancer_Channel": "indigo",
                "BG5_Control_Channel": "midnightblue"
            }
        else:
            palette = {
                "control": "teal",
                "BG5": "indigo",
                "S24": "midnightblue"
            }

        sns.set_theme(style="ticks")

        metrics = ["FR", "ISIm", "ISIcv", "BSR", "BR", "BD", "IBIm", "IBIcv", "NBR", "NIBIm", "NIBIcv", "NoChannels"]
        for i, metric in enumerate(metrics):
            if i <= 3:
                j, k = 0, i
            elif 3 < i <= 7:
                j, k = 1, i - 4
            else:
                j, k = 2, i - 8

            if polynomial:
                custom_lineplot_with_polynomial(
                    axes=ax[j, k], data=df, x="DIV_GBM", y=metric, style="GBM_Type", palette=palette)
            elif plot_mean:
                custom_lineplot(
                    axes=ax[j, k], data=df, x="DIV_GBM", y=metric, style="GBM_Type", palette=palette)
            else:
                custom_lineplot(
                    axes=ax[j, k], data=df, x="DIV_GBM", y=metric, style="GBM_Type", palette=palette, plot_single=True)

        # Style plot
        for i in range(0, 3):
            for j in range(0, 4):
                ax[i, j].set_xlabel("Days in Vitro (GBM)", fontsize=font_size)
                ax[i, j].axvspan(xmin=min(df["DIV_GBM"]), xmax=0.0, color='gray', alpha=0.2, label="Baseline", zorder=0)
                ax[i, j].axhline(0.0, linestyle=":", color='grey', zorder=1)
                ax[i, j].set_xlim(-1.0)
                ax[i, j].yaxis.set_major_locator(ticker.MaxNLocator(5))
                ax[i, j].legend([], [], frameon=False)

        ax[0, 0].set_ylabel(prefix + 'Firing Rate [Hz]', fontsize=font_size)
        ax[0, 1].set_ylabel(prefix + 'Inter-spike Interval [s]', fontsize=font_size)
        ax[0, 2].set_ylabel(prefix + 'Coefficient of Variation of ISI', fontsize=font_size)
        ax[0, 3].set_ylabel(prefix + 'Burst Spike Rate [Hz]', fontsize=font_size)
        ax[1, 0].set_ylabel(prefix + 'Burst Rate [Burst/min]', fontsize=font_size)
        ax[1, 1].set_ylabel(prefix + 'Burst Duration [ms]', fontsize=font_size)
        ax[1, 2].set_ylabel(prefix + 'Inter-burst Interval [s]', fontsize=font_size)
        ax[1, 3].set_ylabel(prefix + 'Coefficient of Variation of IBI', fontsize=font_size)
        ax[2, 0].set_ylabel(prefix + 'Network Burst Rate [Burst/min]', fontsize=font_size)
        ax[2, 1].set_ylabel(prefix + 'Network Inter-burst Interval [s]', fontsize=font_size)
        ax[2, 2].set_ylabel(prefix + 'Coefficient of Variation of NIBI', fontsize=font_size)
        ax[2, 3].set_ylabel(prefix + 'No. of Network Burst Electrodes', fontsize=font_size)

        # Add legend
        legend = ax[0, 3].legend(bbox_to_anchor=(1.5, 1))
        legend.set_title("GBM Type")

        # Prevent overlap of figures
        fig.tight_layout()

        # Save the plot
        if polynomial:
            self.save_plot(fig=fig, filename="Metrics_over_time_polynomial")
        elif use_relatives:
            if plot_mean:
                self.save_plot(fig=fig, filename="Metrics_over_time_relative_mean")
            else:
                self.save_plot(fig=fig, filename="Metrics_over_time_relative_single")
        else:
            if plot_mean:
                self.save_plot(fig=fig, filename="Metrics_over_time_absolute_mean")
            else:
                self.save_plot(fig=fig, filename="Metrics_over_time_absolute_single")

        console.info(f"Plots saved under {self.output_path}")


class RawTracesPlotter(ABCPlotter):
    def __init__(self, filename: str, input_path: str, channel_subselection_path: str, channel_subselection: str, output_path: str):
        super().__init__(input_path=input_path, output_path=output_path)
        self.filename = filename
        self.filepath = os.path.join(self.input_path, self.filename)
        self.channel_subselection_path = channel_subselection_path
        self.channel_subselection = channel_subselection
        self.electrodes_by_channels = None
        self.traces = None

    def load_data(self,
                freq_min: int = 200,
                clipping: bool = True):
        
        self.recording = si.read_maxwell(file_path=os.path.join(self.input_path, self.filename))
        self.recording = si.bandpass_filter(recording=self.recording,
                                                              freq_min=freq_min,
                                                              dtype='float32')
        self.recording = si.common_reference(self.recording)
        if clipping:
            console.info(f"Clipping the recording {self.filename}...")
            self.recording = si.clip(recording=self.recording, a_max=1200)
        self.traces = self.recording.get_traces()
        return self.traces

    def plot_raw_traces(self):
        pass

    def get_electrode_channel_mapping(self, raw_data) -> np.array:
        """
        Retrieves electrode-to-channel mapping from the HDF5 file.

        :param raw_data: The HDF5 file data.
        :return: A np.array containing the electrode to channel mapping.
        """
        # Get the clean absolute and relative indices of the spiking information.
        electrode_info = np.asarray(raw_data["mapping"]["channel", "electrode"])
        mask = [i["electrode"] != -1 for i in electrode_info]
        clean_abs_ids = np.asarray([i[0]["electrode"][i[1]] for i in zip(electrode_info, mask)], dtype=np.int32)
        clean_rel_ids = np.asarray([i[0]["channel"][i[1]] for i in zip(electrode_info, mask)], dtype=np.int32)

        # Map the relative indices to the absolute indices of the spiking information
        self.electrode_channel_mapping = np.zeros([2, clean_rel_ids.shape[0]], dtype=np.int32)
        self.electrode_channel_mapping[0, :] = np.squeeze(clean_abs_ids)
        self.electrode_channel_mapping[1, :] = np.squeeze(clean_rel_ids)

        del electrode_info
        del mask
        del clean_abs_ids
        del clean_rel_ids

        return self.electrode_channel_mapping
    
    def convert_elno_to_xy(elno):
        chipWidth = 220
        x = int(elno/chipWidth)
        y = elno % chipWidth
        return x,y

    def plot_activity_videos_most_active_el_trigger(self, ):
        self.electrodes_by_channels = np.load(os.path.join(self.channel_subselection_path, self.channel_subselection), allow_pickle=True)
        traces = self.load_data()
        


    def video_cropper(self,video_path: str, video_name:str, frames_to_extract: list):
        """
        Extracts frames from a video and saves them as .png files.
        :param video_path: The path to the video.
        :param video_name: The name of the video.
        :param frames_to_extract: A list of frame numbers to extract.
        """
        gif = Image.open(os.path.join(video_path, video_name))
        for frame_number in frames_to_extract:
            gif.seek(frame_number)
            gif.save(f'frame_{frame_number}.png')
        print(f"Frames successfully extracted from {video_name}.")