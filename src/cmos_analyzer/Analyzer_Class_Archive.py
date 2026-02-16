import numpy as np
import h5py
import os
import pickle
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks as fp

from src.utils.logger_functions import console


class SpontaneousActivityAnalyzer:
    def __init__(
        self,
        filename: str,
        input_path: str,
        output_path: str,
        sample_frequency=20000,
        experiment_duration=None,
        id_chip=None,
        id_network=None,
        div_ngn=None,
        div_gbm=None,
        gbm_type=None,
        spike_threshold=5,
        spike_distance=100,
    ):

        self.filename = filename
        self.input_path = input_path
        self.output_path = output_path

        self.sample_frequency = sample_frequency
        self.experiment_duration = experiment_duration
        self.id_chip = id_chip
        self.id_network = id_network
        self.div_ngn = div_ngn
        self.div_gbm = div_gbm
        self.gbm_type = gbm_type

        self.chip_width = 220
        self.spike_threshold = spike_threshold
        self.spike_distance = spike_distance

        self.spikes = []
        self.traces = None
        self.channels = None
        self.spike_mat = None
        self.electrode_burst_mat = None
        self.spike_dict = None
        self.electrode_metrics_columns = None
        self.network_metrics_columns = None
        self.electrode_channel_mapping = None

    def load_data(
        self,
        cutoff_frequency=200,
        filter_order=2,
        loading_steps=10,
    ) -> list:
        """
        Loads all the traces of the h5 file and generates the spike list with custom filtering.

        :param cutoff_frequency: Low frequencies up to this are filtered out by a highpass filter.
        :param filter_order: Order of the butterworth filter used.
        :param loading_steps: The loading of the traces is split in n steps.
        For larger n less RAM is used, but calculations take longer.
        :return: A list containing the spike times and meta information about the peaks on the electrode.
        """
        # Load raw data
        try:
            raw_data = h5py.File(os.path.join(self.input_path, self.filename), "r")
        except Exception as e:
            console.error(f"An error occurred while loading data: {str(e)}")
            return []  # Return empty list to indicate an error

        # Get the electrode to channel mappings
        self.get_electrode_channel_mapping(raw_data=raw_data)

        # Filter the data
        first_time = True
        nyquist = self.sample_frequency/2.
        cut_off_discrete = cutoff_frequency / nyquist
        coeff_b, coeff_a = butter(filter_order, cut_off_discrete, btype="highpass", analog=False)

        # Set up the loading steps (enables to load traces in loading_steps steps)
        loading_indices_start = [0]
        loading_indices_end = []

        for i in range(1, loading_steps - 1):
            step = int(i / loading_steps * self.electrode_channel_mapping.shape[1])
            loading_indices_end.append(step)
            loading_indices_start.append(step)
        loading_indices_end.append(self.electrode_channel_mapping.shape[1])

        # Find peaks in the traces
        for j in range(len(loading_indices_start)):
            self.channels = self.electrode_channel_mapping[1, np.arange(loading_indices_start[j], loading_indices_end[j])]
            self.traces = raw_data.get("sig")[np.squeeze(self.channels), :]

            if self.traces.ndim == 1:
                self.traces = np.expand_dims(self.traces, axis=0)

            if first_time:
                # Get experiment duration
                self.experiment_duration = self.traces.shape[1] / self.sample_frequency
                first_time = False

            self.traces = lfilter(coeff_b, coeff_a, self.traces)
            self.spikes.extend(list(map(self.find_peaks, range(self.traces.shape[0]))))

        self.traces = None

        console.info(f"Spikes of file {self.filename} was successfully loaded.")

        return self.spikes

    def load_data_spikes_only(self) -> list:
        """
        Load spike-related data from the HDF5 file and organizes it.
        Note: This function should be used with recordings where only the spike data was saved.

        :return: A list containing the spike times and meta information about the peaks on the electrode.
        """

        # Load data
        try:
            raw_data = h5py.File(os.path.join(self.input_path, self.filename), "r")
        except Exception as e:
            console.error(f"An error occurred while loading data: {str(e)}")
            return []  # Return an empty list to indicate an error

        # Get the electrode to channel mappings
        self.get_electrode_channel_mapping(raw_data=raw_data)

        # Get the spike times, amplitudes and corresponding channels
        amplitudes = np.squeeze((raw_data.get("proc0")["spikeTimes"])["amplitude"])
        spike_times = np.squeeze((raw_data.get("proc0")["spikeTimes"])["frameno"])
        channels = np.squeeze((raw_data.get("proc0")["spikeTimes"])["channel"])

        # Get the relative spike times (relative to start of experiment)
        spike_times = spike_times - np.min(spike_times)

        # Get experiment duration (if not given)
        if self.experiment_duration is None:
            self.experiment_duration = (np.max(spike_times) - np.min(spike_times)) / self.sample_frequency

        # Iterate through all the channels in the dataset
        for i in range(self.electrode_channel_mapping.shape[1]):
            # Get all the indices where the channel corresponds to the current channel in the mapping
            channel_indices = np.argwhere(channels == np.squeeze(self.electrode_channel_mapping[1, i]))

            # Get the meta information about the peaks on the channel
            peaks = np.squeeze(spike_times[channel_indices], axis=1), {}
            peaks[1]["peak_heights"] = np.squeeze(amplitudes[channel_indices], axis=1)
            peaks[1]["nr_of_peaks"] = (peaks[0]).shape[0]
            peaks[1]["electrode"] = self.electrode_channel_mapping[0, i]

            # Save all spikes to self.spikes
            self.spikes.append(peaks)

        del peaks
        del channel_indices
        del spike_times
        del channels
        del amplitudes
        del raw_data

        console.info(f"Spikes of file {self.filename} were successfully loaded.")

        return self.spikes

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

    def find_peaks(self, i: int) -> tuple:
        """
        Extracts the peak heights, nr of peaks and electrode no. from the spike trace.
        :param i: The channel number to be processed.
        :return: A tuple containing peak information including peak heights, the number of peaks,
        and the corresponding electrode number.
        """

        # Calculate MAD
        MAD = np.median(np.abs(self.traces[i, :] - np.median(self.traces[i, :])))
        std = MAD * 1.4826

        # Find negative peaks in the data
        peaks = fp((self.traces[i, :] * -1),
                   height=self.spike_threshold * std,
                   distance=self.spike_distance,
                   )
        peaks[1]["peak_heights"] = np.asarray(self.traces[i, :][peaks[0]])
        peaks[1]["nr_of_peaks"] = (peaks[0]).shape[0]
        peaks[1]["electrode"] = self.electrode_channel_mapping[
            0,
            np.squeeze(
                np.argwhere(self.channels[i] == self.electrode_channel_mapping[1, :])
            ),
        ]

        return peaks

    def get_spike_matrix(self):
        """
        Creates a spike matrix from the loaded spike data.
        The spike matrix containing electrode, spike time and spike amplitude,
        """
        # Create a spiking matrix of the format: electrode, spike time and spike amplitude
        spike_list = []
        for s_per_el in self.spikes:
            for s_id, s in enumerate(s_per_el[0]):
                spike_list.append(
                    np.array(
                        [s_per_el[1]['electrode'],
                         s / self.sample_frequency * 1e3,
                         s_per_el[1]['peak_heights'][s_id]]
                    )
                )

        self.spike_mat = np.array(spike_list, dtype=float)

    def create_dict(self, export=True) -> dict:
        """
        Export analyzed data to a dictionary and save it as a pickle file

        :param export: (bool) If True, the dictionary is exported to the output path.
        :return network_dict: Returns a dictionary containing spike and culturing information.
        """

        # Check if spike matrix was already created
        if self.spike_mat is None:
            self.get_spike_matrix()

        # Load analyzer entries to network dict
        self.spike_dict = {
            'CHIP_ID':  self.id_chip,
            'NW_ID':    self.id_network,
            'DIV_NGN':  self.div_ngn,
            'DIV_GBM':  self.div_gbm,
            'GBM_Type': self.gbm_type,
            'EXPERIMENT_DURATION': self.experiment_duration,
            'SPIKEMAT': self.spike_mat,
            'ELECTRODE_METRICS': self.get_electrode_metrics(),
            'NETWORK_METRICS': self.get_network_metrics()
        }

        # Create export filename
        if export:
            export_filename = self.filename[:-3] + "_processed"
            export_filename = f"{self.output_path}/{export_filename}"

            with open(f"{export_filename}.pkl", "wb") as f:
                pickle.dump(self.spike_dict, f)

            console.info(f"Spike Dictionary was successfully created and exported as {export_filename}.")

        return self.spike_dict

    def get_electrode_metrics(self,
                              burst_time_window=50,
                              min_spike_count_in_burst=4,
                              ) -> np.array:
        """
        Calculate network metrics, including Burst Rate (BR), Burst Duration (BD), and Firing Rate (FR) for each electrode.

        :param burst_time_window: Time window (in ms) used to define a burst (default is 50 ms).
        :param min_spike_count_in_burst: Minimum spike count required to consider a burst (default is 4 spikes).
        :return: A np.array containing the electrode number, Mean Firing Rate (FR), Burst Rate (BR),
        and Burst Duration (BD) per electrode.
        """

        # Initialize variables to store information
        self.electrode_metrics_columns = ["Electrode", "FR", "ISIm", "ISIstd", "ISIcv",
                                          "BR", "BD", "IBIm", "IBIstd", "IBIcv", "BSR"]
        electrode_metrics = np.empty((0, len(self.electrode_metrics_columns)))
        self.electrode_burst_mat = np.empty((0, 3))

        # Iterate through each unique electrode
        unique_electrodes = np.unique(self.spike_mat[:, 0])

        for electrode in unique_electrodes:
            # Filter spikes for the current electrode and sort them by spike time
            electrode_spikes = self.spike_mat[self.spike_mat[:, 0] == electrode]
            electrode_spikes = electrode_spikes[np.argsort(electrode_spikes[:, 1])]

            burst_count = 0
            durations = []
            spike_intervals = []
            burst_intervals = []
            burst_spike_rates = []
            burst_start_time = None
            previous_burst_start_time = None
            burst_end_time = None

            for i in range(1, len(electrode_spikes)):
                time_diff = electrode_spikes[i, 1] - electrode_spikes[i - 1, 1]
                spike_intervals.append(time_diff / 1e3)  # in s

                if time_diff <= burst_time_window:
                    if burst_start_time is None:
                        # Start of a new burst
                        burst_start_time = electrode_spikes[i - 1, 1]

                        # Add burst interval
                        if previous_burst_start_time is not None:
                            burst_time_diff = (burst_start_time - previous_burst_start_time)
                            burst_intervals.append(burst_time_diff / 1e3)  # in s

                    # Update the end time of the burst
                    burst_end_time = electrode_spikes[i, 1]

                else:
                    if burst_start_time is not None:
                        # End of a burst
                        duration = (burst_end_time - burst_start_time) / 1e3  # in s
                        burst_count += 1

                        # Find the index of the first spike with time >= burst_start_time
                        burst_spike_count = i - np.where(electrode_spikes[:, 1] == burst_start_time)[0][0]

                        # Check if the burst contained enough spikes
                        if burst_spike_count >= min_spike_count_in_burst:
                            burst_count += 1
                            durations.append(duration)
                            burst_spike_rates.append(burst_spike_count/duration)

                            # Add info to electrode burst matrix
                            new_row_el = np.array([electrode, burst_start_time, duration])
                            self.electrode_burst_mat = np.vstack((self.electrode_burst_mat, new_row_el))

                        # Reset burst tracking
                        previous_burst_start_time = burst_start_time
                        burst_start_time = None
                        burst_end_time = None

            # Calculate metrics related to bursts
            burst_rate = burst_count / self.experiment_duration * 60  # in bursts per min
            burst_duration = np.mean(durations)
            ibi_mean = np.mean(burst_intervals)
            ibi_std = np.std(burst_intervals)
            ibi_cv = ibi_std / ibi_mean
            burst_spike_rate = np.mean(burst_spike_rates)

            # Calculate metrics related to spikes
            firing_rate = len(electrode_spikes) / self.experiment_duration
            isi_mean = np.mean(spike_intervals)
            isi_std = np.std(spike_intervals)
            isi_cv = isi_std / isi_mean

            # Store information
            new_row = np.array([electrode, firing_rate, isi_mean, isi_std, isi_cv,
                                burst_rate, burst_duration, ibi_mean, ibi_std, ibi_cv, burst_spike_rate])
            electrode_metrics = np.vstack((electrode_metrics, new_row))

        return electrode_metrics

    def get_network_metrics(self,
                            min_no_parallel_channels=10,
                            ) -> np.array:
        """
        Extracts network burst metrics from the electrode matrix. When a burst is occurring in more than 6 channels at
        the same time, and 6 are time locked, these bursts are classified as network burst.
        :param min_no_parallel_channels: Minimum number of channels required to consider a burst.
        :return: A np.array containing network burst metrics with columns
        ["NBR", "NBD", "NIBIm", "NIMIstd", "NIMIcv", "NoChannels"].
        """

        # Initialize variables to store information
        self.network_metrics_columns = ["NBR", "NBD", "NIBIm", "NIMIstd", "NIMIcv", "NoChannels"]
        network_metrics = np.empty((0, len(self.network_metrics_columns)))

        burst_count = 0
        burst_durations = []
        burst_intervals = []
        parallel_channels = []
        prev_start_time = None

        # Iterate through each unique burst start time
        unique_start_times = np.unique(self.electrode_burst_mat[:, 1])
        unique_start_times = np.sort(unique_start_times)

        for start_time in unique_start_times:
            # Find bursts with the same start time
            bursts_same_time = self.electrode_burst_mat[self.electrode_burst_mat[:, 1] == start_time]

            # Check if there are at least the min number of parallel channels
            if bursts_same_time.shape[0] >= min_no_parallel_channels:
                # Extract information for network burst
                burst_count += 1
                burst_durations.extend(bursts_same_time[:, 2])
                parallel_channels.append(bursts_same_time.shape[0])

                # Calculate burst intervals
                if prev_start_time is not None:
                    burst_intervals.append((start_time - prev_start_time) / 1e3)  # in s
                prev_start_time = start_time

        # Calculate metrics related to network bursts
        nbr = burst_count / self.experiment_duration * 60  # in bursts per min
        nbd = np.mean(burst_durations)
        nibi_m = np.mean(burst_intervals)
        nibi_std = np.std(burst_intervals)
        nibi_cv = nibi_std / nibi_m
        no_channels = np.mean(parallel_channels)

        # Store information
        new_row = np.array([nbr, nbd, nibi_m, nibi_std, nibi_cv, no_channels])
        network_metrics = np.vstack((network_metrics, new_row))

        return network_metrics
