import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn import metrics
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.logger_functions import console


class SpontaneousActivityAnalyzer:
    def __init__(
        self,
        filename: str,
        input_path: str,
        output_path: str,
    ):
        """
        Initializes the SpontaneousActivityAnalyzer with the given filename, input path, and output path.

        :param filename: (str) The name of the file containing the data.
        :param input_path: (str) The path where the input file is located.
        :param output_path: (str) The path where the output file will be saved.
        """

        self.filename = filename
        self.input_path = input_path
        self.output_path = output_path

        self.sample_frequency = 20000

        self.spike_mat = None
        self.spike_mat_extremum = None
        self.data_dict = None
        self.experiment_duration = None
        self.cross_correlation_matrix = None
        self.cross_correlation_delays = None
        self.cross_correlation_auc = None

        self.electrode_burst_mat = np.empty((0, 3))

    def load_data(self):
        """
        Loads the pickle file containing the processed data.

        :return: A dictionary containing the loaded data.
        """
        # Load processed data
        try:
            with open(os.path.join(self.input_path, self.filename), 'rb') as f:
                # Load the dictionary from the .pkl file
                self.data_dict = pickle.load(f)
                self.spike_mat = self.data_dict['SPIKEMAT']
                self.spike_mat_extremum = self.data_dict["SPIKEMAT_EXTREMUM"]
                self.experiment_duration = self.data_dict["EXPERIMENT_DURATION"]

        except Exception as e:
            console.error(f"Error processing file {self.filename}: {e}")

        return self.data_dict

    def create_dict(self, export=True) -> dict:
        """
        Export analyzed data to a dictionary and save it as a pickle file

        :param export: (bool) If True, the dictionary is exported to the output path.
        :return network_dict: Returns a dictionary containing spike and culturing information.
        """

        # Add keys to data_dict
        additional_keys = {
            'ELECTRODE_METRICS': self.get_electrode_metrics(),
            'ELECTRODE_METRICS_EXTREMUM': self.get_electrode_metrics(extremum_channels_only=True),
            'NETWORK_BURST_METRICS': self.get_network_burst_metrics(),
            'CORRELATION_METRICS': self.get_correlation_metrics(),
            'CROSS_CORRELATION_MATRIX': self.cross_correlation_matrix,
            'CROSS_CORRELATION_DELAYS': self.cross_correlation_delays,
            'CROSS_CORRELATION_AUC': self.cross_correlation_auc,
        }

        self.data_dict.update(additional_keys)

        # Create export filename
        if export:
            export_filename = self.filename[:-4] + "_metrics"
            export_filename = f"{self.output_path}/{export_filename}"

            with open(f"{export_filename}.pkl", "wb") as f:
                pickle.dump(self.data_dict, f)

            console.info(f"Data Dictionary was successfully created and exported as {export_filename}.")

        return self.data_dict

    def get_electrode_metrics(self,
                              burst_time_window=50,
                              min_spike_count_in_burst=4,
                              extremum_channels_only=False,
                              ) -> np.array:
        """
        Calculate network metrics, including Burst Rate (BR), Burst Duration (BD), and Firing Rate (FR) for each electrode.

        :param burst_time_window: Time window (in ms) used to define a burst (default is 50 ms).
        :param min_spike_count_in_burst: Minimum spike count required to consider a burst (default is 4 spikes).
        :param extremum_channels_only: If True, metrics are calculated only for extremum channels.
        :return: A np.array containing the electrode number, Mean Firing Rate (FR), Burst Rate (BR), Burst Duration (BD)
        and Percentage of Random Spikes (PSR) per electrode.
        """

        console.info("Extracting electrode metrics...")

        # Initialize variables to store information
        dtype = [('Electrode', '<i8'), ('FR', '<f8'), ('ISIm', '<f8'), ('ISIstd', '<f8'), ('ISIcv', '<f8'),
                 ('BR', '<f8'), ('BD', '<f8'), ('IBIm', '<f8'), ('IBIstd', '<f8'), ('IBIcv', '<f8'), ('BSR', '<f8'),
                 ('PRS', '<f8')]

        electrode_metrics = np.zeros((0, len(dtype)))
        electrode_burst_mat = np.empty((0, 3))

        # Iterate through each unique electrode
        if extremum_channels_only:
            spike_mat = self.spike_mat_extremum
        else:
            spike_mat = self.spike_mat

        unique_electrodes = np.unique(spike_mat["Electrode"])

        for electrode in unique_electrodes:
            # Filter spikes for the current electrode and sort them by spike time
            electrode_spikes = spike_mat[spike_mat["Electrode"] == electrode]
            electrode_spikes = electrode_spikes[np.argsort(electrode_spikes["Spike_Time"])]

            burst_count = 0
            durations = []
            spike_intervals = []
            burst_intervals = []
            burst_spike_rates = []
            burst_start_time = None
            previous_burst_start_time = None
            burst_end_time = None
            total_spikes_in_bursts = 0

            for i in range(1, len(electrode_spikes)):
                time_diff = electrode_spikes["Spike_Time"][i] - electrode_spikes["Spike_Time"][i - 1]
                spike_intervals.append(time_diff / 1e3)  # in s

                if time_diff <= burst_time_window:
                    if burst_start_time is None:
                        # Start of a new burst
                        burst_start_time = electrode_spikes["Spike_Time"][i - 1]

                        # Add burst interval
                        if previous_burst_start_time is not None:
                            burst_time_diff = (burst_start_time - previous_burst_start_time)
                            burst_intervals.append(burst_time_diff / 1e3)  # in s

                    # Update the end time of the burst
                    burst_end_time = electrode_spikes["Spike_Time"][i]

                else:
                    if burst_start_time is not None:
                        # End of a burst
                        duration = (burst_end_time - burst_start_time) / 1e3  # in s
                        burst_count += 1

                        # Find the index of the first spike with time >= burst_start_time
                        burst_spike_count = i - np.where(electrode_spikes["Spike_Time"][:] == burst_start_time)[0][0]

                        # Check if the burst contained enough spikes
                        if burst_spike_count >= min_spike_count_in_burst:
                            burst_count += 1
                            durations.append(duration)
                            burst_spike_rates.append(burst_spike_count/duration)
                            total_spikes_in_bursts += burst_spike_count

                            # Add info to electrode burst matrix
                            new_row_el = np.array([electrode, burst_start_time, duration])
                            electrode_burst_mat = np.vstack((electrode_burst_mat, new_row_el))

                        # Reset burst tracking
                        previous_burst_start_time = burst_start_time
                        burst_start_time = None
                        burst_end_time = None

            # Calculate metrics related to bursts
            burst_rate = burst_count / self.experiment_duration * 60  # in bursts per min
            burst_duration = np.mean(durations) if durations else 0
            ibi_mean = np.mean(burst_intervals) if burst_intervals else 0
            ibi_std = np.std(burst_intervals) if burst_intervals else 0
            ibi_cv = ibi_std / ibi_mean if ibi_mean != 0 else 0
            burst_spike_rate = np.mean(burst_spike_rates) if burst_spike_rates else 0

            # Calculate metrics related to spikes
            firing_rate = len(electrode_spikes) / self.experiment_duration
            isi_mean = np.mean(spike_intervals) if spike_intervals else 0
            isi_std = np.std(spike_intervals) if spike_intervals else 0
            isi_cv = isi_std / isi_mean if isi_mean != 0 else 0

            # Calculate percentage of random spikes
            total_spikes = len(electrode_spikes)
            random_spikes_count = total_spikes - total_spikes_in_bursts
            percentage_random_spikes = (random_spikes_count / total_spikes) * 100 if total_spikes > 0 else 0

            # Store information
            new_row = np.array([electrode.astype(int), firing_rate, isi_mean, isi_std, isi_cv, burst_rate, burst_duration,
                                ibi_mean, ibi_std, ibi_cv, burst_spike_rate, percentage_random_spikes])

            electrode_metrics = np.vstack((electrode_metrics, new_row))

        if extremum_channels_only:
            self.electrode_burst_mat = electrode_burst_mat

        # Copy data from the original array to the structured array
        structured_electrode_metrics = np.zeros(electrode_metrics.shape[0], dtype=dtype)

        for i, field in enumerate(structured_electrode_metrics.dtype.names):
            structured_electrode_metrics[field] = electrode_metrics[:, i]

        console.info("Electrode metrics extracted successfully.")

        return structured_electrode_metrics

    def get_network_burst_metrics(self,
                            min_no_parallel_channels=6,
                            time_window = 20,
                            ) -> np.array:
        """
        Extracts network burst metrics from the electrode matrix. When a burst is occurring in more than N channels at
        the same time, and N are time locked, these bursts are classified as network burst.
        :param min_no_parallel_channels: Minimum number of channels required to consider a burst (default is 6)
        :param time_window: Time window (in ms) to consider bursts as parallel (default is 20 ms).
        :return: A np.array containing network burst metrics with columns
        ["NBR", "NBD", "NIBIm", "NIBIstd", "NIBIcv", "NoUnits", "Burst_Synchrony"].
        """

        console.info("Extracting network burst metrics...")

        # Initialize variables to store information
        dtype = [('NBR', '<f8'), ('NBD', '<f8'), ('NIBIm', '<f8'), ('NIBIstd', '<f8'),
                 ('NIBIcv', '<f8'), ('No_Units', '<f8'), ('Burst_Synchrony', '<f8')]

        network_metrics = np.zeros((0, len(dtype)))

        if len(self.electrode_burst_mat) == 0:
            self.get_electrode_metrics(extremum_channels_only=True)

        tot_units = len(np.unique(self.spike_mat_extremum["UnitIdx"]))

        burst_count = 0
        burst_durations = []
        burst_intervals = []
        parallel_units = []
        prev_start_time = None

        # Iterate through each unique burst start time
        unique_start_times = np.unique(self.electrode_burst_mat[:, 1].astype(float))
        unique_start_times = np.sort(unique_start_times)

        for start_time in unique_start_times:
            # Find bursts with the same start time
            bursts_same_time = self.electrode_burst_mat[(self.electrode_burst_mat[:, 1] >= str(start_time - time_window)) &
                               (self.electrode_burst_mat[:, 1] <= str(start_time + time_window))]

            # Check if there are at least the min number of parallel channels
            if bursts_same_time.shape[0] >= min_no_parallel_channels:
                # Extract information for network burst
                burst_count += 1
                burst_durations.extend(bursts_same_time[:, 2].astype(float))
                parallel_units.append(int(bursts_same_time.shape[0]))

                # Calculate burst intervals
                if prev_start_time is not None:
                    burst_intervals.append(float((start_time - prev_start_time) / 1e3))  # in s
                prev_start_time = start_time

        # Calculate metrics related to network bursts
        nbr = burst_count / self.experiment_duration * 60  # in bursts per min
        nbd = np.mean(burst_durations) if burst_durations else 0
        nibi_m = np.mean(burst_intervals) if burst_intervals else 0
        nibi_std = np.std(burst_intervals) if burst_intervals else 0
        nibi_cv = nibi_std / nibi_m if nibi_m != 0 else 0
        no_units = np.mean(parallel_units) if parallel_units else 0
        burst_synchrony = no_units/tot_units if tot_units != 0 else 0

        # Store information
        new_row = np.array([nbr, nbd, nibi_m, nibi_std, nibi_cv, no_units, burst_synchrony])
        network_metrics = np.vstack((network_metrics, new_row))

        # Copy data from the original array to the structured array
        structured_network_metrics = np.zeros(network_metrics.shape[0], dtype=dtype)

        for i, field in enumerate(structured_network_metrics.dtype.names):
            structured_network_metrics[field] = network_metrics[:, i]

        console.info("Network burst metrics extracted successfully.")

        return structured_network_metrics

    def get_binary_traces_per_unit(self, bin_size=5):
        """
        Get binary spike traces for each unit.

        :param bin_size: (int) Size of the time bins in milliseconds. Defaults to 5.
        :return np.ndarray: Matrix containing binary spike traces for each unit.
        """

        # Get extremum electrodes per unit
        extremum_electrodes = np.unique(self.spike_mat_extremum["Electrode"])

        # Create an empty matrix to store binary traces for each channel
        nr_of_bins = int(self.experiment_duration * 1e3 / bin_size) + 1
        binary_matrix = np.zeros((len(extremum_electrodes), nr_of_bins + 1))
        unit_no = 0

        # Iterate through each channel in highest_spike_frequency_per_channel
        for electrode in extremum_electrodes:
            # Get all spikes that occurred on the electrode in the channel
            rows_for_electrode = self.spike_mat_extremum[self.spike_mat_extremum["Electrode"] == electrode]
            spikes_on_electrode = rows_for_electrode["Spike_Time"]

            # Round each spike time to the nearest multiple of bin_size
            rounded_spikes = np.floor(spikes_on_electrode / bin_size).astype(int)

            # Set the pixels corresponding to spike positions to 0 (black)
            binary_matrix[unit_no, rounded_spikes] = 1

            unit_no += 1

        return binary_matrix

    def get_cross_correlation(self):
        """
        Compute the cross-correlation between spike trains i and j, normalized to allow comparison of similarity between
        different spike trains.
        """
        # Get point processes (binary spike traces)
        binary_matrix = self.get_binary_traces_per_unit()

        # Precompute autocorrelation for all spike trains
        with ProcessPoolExecutor(max_workers=4) as executor:
            auto_corr = list(executor.map(self.compute_autocorrelation, binary_matrix))

        # Convert the result to a numpy array
        auto_corr = np.array(auto_corr)
        console.info(f"Computed autocorrelations.")

        # Initialize the cross-correlation matrix
        n_units = binary_matrix.shape[0]
        self.cross_correlation_matrix = np.zeros((n_units, n_units))
        self.cross_correlation_delays = np.zeros((n_units, n_units))
        self.cross_correlation_auc = np.zeros((n_units, n_units))

        # Total number of tasks to track for progress
        total_tasks = (n_units * (n_units + 1)) // 2

        with tqdm(total=total_tasks, desc="Computing cross correlations") as pbar:
            with ProcessPoolExecutor(max_workers=4) as executor:
                # Iterate through channels that are equal or greater than i to avoid redundant calculations
                for i in range(n_units):
                    futures = [executor.submit(self.compute_cross_correlation, i, j, binary_matrix, auto_corr)
                               for j in range(i, n_units)]

                    # Save the value in the matrices
                    for future in as_completed(futures):
                        i, j, max_corr, delay, auc = future.result()
                        self.cross_correlation_matrix[i, j] = max_corr
                        self.cross_correlation_delays[i, j] = delay
                        self.cross_correlation_auc[i, j] = auc

                        # If i != j, fill the symmetric element as well
                        if i != j:
                            self.cross_correlation_matrix[j, i] = max_corr
                            self.cross_correlation_delays[j, i] = -delay
                            self.cross_correlation_auc[j, i] = auc

                        # Update progress bar
                        pbar.update(1)

    @staticmethod
    def compute_autocorrelation(row):
        return np.correlate(row, row, mode='same')

    def compute_cross_correlation(self, i: int, j: int, binary_matrix, auto_corr):
        """
        Compute the cross-correlation between spike trains i and j. The cross-correlation value is normalized by
        dividing it by the square root of the autocorrelation values of the two individual spike trains at time lag 0.
        This scales the values to the range [-1, 1], where 0 indicates no correlation.

        :param i: (int) Index of the first spike train.
        :param j: (int) Index of the second spike train.
        :param binary_matrix: (np.ndarray) Binary matrix of spike trains where each row represents a spike train and
        each column column represents a time bin.
        :param auto_corr: (np.ndarray) Precomputed autocorrelation values for all spike trains.
        :return: A tuple containing i, j, max_corr (maximum normalised cross-correlation value), delay (delay
        corresponding to the maximum cross-correlation value) and auc (area under the cross-correlation curve).
        """

        # Calculate and normalize the cross correlation to be able to compare the similarity between
        # different spike trains.
        n = len(binary_matrix[i, :])

        corr = np.correlate(binary_matrix[j, :], binary_matrix[i, :], mode='same')

        norm_corr = corr / np.sqrt(auto_corr[i, binary_matrix.shape[1] // 2] *
                                   auto_corr[j, binary_matrix.shape[1] // 2])

        # Prepare a delay array to get the lag
        delay_arr = np.linspace(-0.5 * n / self.sample_frequency, 0.5 * n / self.sample_frequency, n)

        # Return the computed values
        max_corr = np.max(norm_corr)
        delay = delay_arr[np.argmax(norm_corr)]
        auc = metrics.auc(x=np.array(range(0, corr.shape[0])), y=norm_corr)

        return i, j, max_corr, delay, auc

    def get_correlation_metrics(self):
        """
        Extract correlation metrics from the cross-correlation matrix.

        :return: A np.array containing correlation metrics with columns ["Mean_Cross_Correlation", "Mean_AUC"].
        """

        console.info("Extracting the correlation metrics...")

        # Check if the cross correlation matrix is available
        if self.cross_correlation_matrix is None:
            self.get_cross_correlation()

        # Calculate mean cross correlation and area under curve
        mean_cross_correlation = self.cross_correlation_matrix.mean()
        mean_cross_correlation_auc = self.cross_correlation_auc.mean()

        # Copy data from the original array to the structured array
        dtype = [('Mean_Cross_Correlation', '<f8'), ('Mean_AUC', '<f8')]
        correlation_metrics = np.array([mean_cross_correlation, mean_cross_correlation_auc], dtype=dtype)

        console.info("Correlation metrics extracted successfully.")

        return correlation_metrics



