# Import libraries
import h5py
import os
import pickle
import json
import logging
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks as fp
import spikeinterface as si
import spikeinterface.extractors
import spikeinterface.preprocessing
import spikeinterface.sorters
import spikeinterface.curation
import spikeinterface.qualitymetrics
import spikeinterface.sortingcomponents.peak_detection
import matplotlib.pyplot as plt
import spikeinterface.postprocessing
import spikeinterface.widgets
from abc import ABC, abstractmethod

from src.utils.logger_functions import console
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


class ABCExtractor(ABC):
    """
    Abstract base class for extracting electrophysiological data and processing it for spike sorting.
    """

    @abstractmethod
    def __init__(self,
                 filename: str,
                 input_path: str,
                 output_path: str,
                 sample_frequency=20000,
                 experiment_duration=None,
                 id_chip=None,
                 id_network=None,
                 div=None,
                 spike_threshold=5,
                 spike_distance=1,
                 ):
        """
        Initializes the ABCExtractor with the given parameters.
        """

        self.filename = filename
        self.input_path = input_path
        self.output_path = output_path

        self.sample_frequency = sample_frequency
        self.experiment_duration = experiment_duration
        self.id_chip = id_chip
        self.id_network = id_network
        self.div = div

        self.chip_width = 220
        self.spike_threshold = spike_threshold
        self.spike_distance = spike_distance

        self.spikes = []
        self.traces = None
        self.channels = None
        self.spike_mat = None
        self.spike_mat_extremum = None
        self.spike_dict = None
        self.electrode_channel_mapping = None
        self.unit_id_to_electrode_ids = None
        self.blanking_indices = None

    def create_dict(self, export=True, extremum=True) -> dict:
        """
        Export analyzed data to a dictionary and save it as a pickle file

        :param export: (bool, optional) If True, the dictionary is exported to the output path. Default is True.
        :param extremum: (bool, optional) If True, creates spike matrix for extremum channels. Default is True.

        :return network_dict: Returns a dictionary containing spike and culturing information.
        """

        # Check if spike matrix was already created
        if self.spike_mat is None:
            self.get_spike_matrix()

        if extremum:
            self.get_spike_matrix_extremum()

        # Load analyzer entries to network dict
        self.spike_dict = {
            'CHIP_ID': self.id_chip,
            'NW_ID': self.id_network,
            'DIV': self.div,
            'EXPERIMENT_DURATION': self.experiment_duration,
            'SPIKEMAT_EXTREMUM': self.spike_mat_extremum,
            'SPIKEMAT': self.spike_mat,
            'UNIT_TO_EL': self.unit_id_to_electrode_ids,
            "BLANKING": self.blanking_indices
        }

        # Create export filename
        if export:
            export_filename = self.filename[:-3] + "_processed"
            export_filename = f"{self.output_path}/{export_filename}"

            with open(f"{export_filename}.pkl", "wb") as f:
                pickle.dump(self.spike_dict, f)

            console.info(f"Spike Dictionary was successfully created and exported as {export_filename}.")

        return self.spike_dict

    @abstractmethod
    def get_spike_matrix(self):
        """
        Abstract method to be implemented by subclasses for creating a spike matrix.
        Must be overridden by the subclass.
        """
        pass

    @abstractmethod
    def get_spike_matrix_extremum(self):
        """
        Abstract method to be implemented by subclasses for creating a spike matrix for extremum channels.
        Extremum channels are the channels with the largest peaks for each unit.
        Must be overridden by the subclass.
        """
        pass

    @abstractmethod
    def load_data(self):
        """
        Abstract method to be implemented by subclasses for loading data.
        Must be overridden by the subclass.
        """
        pass

    @abstractmethod
    def get_electrode_channel_mapping(self, raw_data) -> np.array:
        """
        Abstract method to be implemented by subclasses for creating a mapping between electrode and channel IDs.

        :param raw_data: Raw data object.
        :returns np.array: Array mapping electrode IDs to channel IDs.

        Must be overridden by the subclass.
        """
        pass


class ElectrodeActivityExtractor(ABCExtractor):
    def __init__(
            self,
            filename: str,
            input_path: str,
            output_path: str,
            sample_frequency=20000,
            experiment_duration=None,
            id_chip=None,
            id_network=None,
            div=None,
            spike_threshold=5,
            spike_distance=1,
    ):

        super().__init__(filename=filename,
                         input_path=input_path,
                         output_path=output_path,
                         sample_frequency=sample_frequency,
                         experiment_duration=experiment_duration,
                         id_chip=id_chip,
                         id_network=id_network,
                         div=div,
                         spike_threshold=spike_threshold,
                         spike_distance=spike_distance,
                         )

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
        #try:
        raw_data = h5py.File(os.path.join(self.input_path, self.filename), "r")
        #except Exception as e:
        #    console.error(f"An error occurred while loading data: {str(e)}")
        #    return []  # Return empty list to indicate an error

        # Get the electrode to channel mappings
        self.get_electrode_channel_mapping(raw_data=raw_data)

        # Filter the data
        first_time = True
        nyquist = self.sample_frequency / 2.
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
            self.channels = self.electrode_channel_mapping[
                1, np.arange(loading_indices_start[j], loading_indices_end[j])]
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
                   distance=self.spike_distance*self.sample_frequency/1000,
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

    def get_spike_matrix(self):
        """
        Creates a spike matrix from the loaded spike data.
        The spike matrix containing electrode, spike time and spike amplitude.
        """
        spike_list = []
        for s_per_el in self.spikes:
            for s_id, s in enumerate(s_per_el[0]):
                # Create a tuple with the three values
                spike_tuple = (
                    str(s_per_el[1]['electrode']),  # Convert electrode to string
                    float(s / self.sample_frequency * 1e3),  # Convert time to milliseconds
                    float(s_per_el[1]['peak_heights'][s_id])  # Amplitude
                )
                spike_list.append(spike_tuple)
        
        dtype = [('Electrode', 'U10'), ('Spike_Time', 'f8'), ('Amplitude', 'f8')]
        self.spike_mat = np.array(spike_list, dtype=dtype)
    
    def get_spike_matrix_extremum(self):
        self.spike_mat_extremum = None
        return self.spike_mat_extremum


class SortingExtractor(ABCExtractor):
    """
    SortingExtractor class for processing electrophysiological recording data and performing spike sorting.
    """

    def __init__(
            self,
            filename: str,
            input_path: str,
            output_path: str,
            sample_frequency=20000,
            experiment_duration=None,
            id_chip=None,
            id_network=None,
            div=None,
            spike_threshold=5,
            spike_distance=3,
            waveform_extraction_method="best_channels",
            n_jobs=6,
            num_channels_per_unit=15,
            perform_sorting=True,
    ):
        """
        Initializes the SortingExtractor with given parameters.

        :param filename: (str) Name of the file to be processed.
        :param input_path: (str) Path to the input data.
        :param output_path: (str) Path to save the output data.
        :param sample_frequency: (int, optional) Sample frequency of the recording. Default is 20000 Hz.
        :param experiment_duration: (float, optional) Duration of the experiment. Default is None.
        :param id_chip: (str, optional) Chip identifier. Default is None.
        :param id_network: (str, optional) Network identifier. Default is None.
        :param div: (int, optional) DIV (Days In Vitro) of the experiment. Default is None.
        :param spike_threshold: (int, optional) Threshold for spike detection. Default is 5.
        :param spike_distance: (int, optional) Minimum distance between spikes. Default is 1 ms.
        :param waveform_extraction_method: (str, optional) Method used to extract waveforms.
        Options are "snr", "ptp" or "best_channels". Default is "best_channels"
        :param n_jobs: (int, optional) Number of jobs to run im parallel.
        :param num_channels_per_unit: (init, optional) Number of channels to extract per unit for method "best_channels".
        :param perform_sorting: (bool) If True, spike sorting is performed on the electrophysiology data.
        """

        super().__init__(filename=filename,
                         input_path=input_path,
                         output_path=output_path,
                         sample_frequency=sample_frequency,
                         experiment_duration=experiment_duration,
                         id_chip=id_chip,
                         id_network=id_network,
                         div=div,
                         spike_threshold=spike_threshold,
                         spike_distance=spike_distance,
                         )

        self.sorting_output_path = os.path.join(self.output_path, "Sorter_" + self.filename)

        self.waveform_extraction_method = waveform_extraction_method
        self.n_jobs = n_jobs
        self.num_channels_per_unit = num_channels_per_unit
        self.perform_sorting = perform_sorting

        self.sorting = None
        self.recording = None
        self.channel_ids = None
        self.waveform_extractor = None
        self.good_unit_ids = []
        self.electrodes = []

    def load_data(
            self,
            freq_min=200,
            freq_max=None,
            blanking=False,
            blanking_threshold=80,
            blanking_distance=80,
            blanking_window=(3, 3),
            plot_sorting=True
    ):
        """
        Loads and preprocesses the data, performs spike sorting and curation, and optionally plots the sorting results.

        :param freq_min: (int, optional) Minimum frequency for bandpass filtering. Default is 200 Hz.
        :param freq_max: (int, optional) Maximum frequency for bandpass filtering. Default is None.
        :param blanking: (bool, optional) Whether to perform blanking. Default is False. Required for stimulation.
        :param blanking_threshold: (int, optional) Threshold for artifact indices detection for blanking. Default is 80.
        :param blanking_distance: (int, optional) Minimum distance between blanking events / artifacts. Default is 80.
        :param blanking_window: (tuple, optional) Window size for blanking in milliseconds. Default is (3, 3).
        :param plot_sorting: (bool, optional) Whether to plot sorting results. Default is True.
        """

        # Load recording
        try:
            self.recording = si.extractors.read_maxwell(file_path=os.path.join(self.input_path, self.filename))
            console.info(f"Successfully loaded the recording {self.filename}.")
        except Exception as e:
            console.error(f"An error occurred while loading data: {str(e)}")
            return []

        self.experiment_duration = self.recording.get_duration()
        self.sampling_rate = self.recording.get_sampling_frequency()

        # Filter recording
        if freq_max is not None:
            self.recording = si.preprocessing.bandpass_filter(recording=self.recording,
                                                              freq_min=freq_min,
                                                              freq_max=freq_max,
                                                              dtype='float32'
                                                              )
        else:
            self.recording = si.preprocessing.bandpass_filter(recording=self.recording,
                                                              freq_min=freq_min,
                                                              dtype='float32')

        # Apply common reference to reduce the noise
        self.recording = si.preprocessing.common_reference(self.recording)

        console.info(f"Successfully filtered the recording {self.filename}.")

        # Blank if necessary
        if blanking:
            self.blanking_indices = self.find_blanking_indices(blanking_threshold, blanking_distance)
            self.recording = si.preprocessing.remove_artifacts(self.recording,
                                                               self.blanking_indices,
                                                               ms_before=blanking_window[0],
                                                               ms_after=blanking_window[1],
                                                               mode='linear',
                                                               )

        if self.perform_sorting:
            # Check if sorting has already been done
            if os.path.exists(self.sorting_output_path):
                console.info(f"Sorting output path {self.sorting_output_path} already exists.")
                var = self.check_sorting(self.sorting_output_path)
                if var == 'error_true' or var == 'log not found':
                    # Perform spike sorting if the sorter contains an error
                    self.sorting = self.perform_spike_sorting()

                elif var == 'error_false':
                    # Read the sorter if it does not contain an error
                    self.sorting = si.sorters.read_sorter_folder(
                        output_folder=self.sorting_output_path,
                        register_recording=False
                    )

            else:
                console.info(f"Sorting output path {self.sorting_output_path} does not exist. Performing spike sorting.")
                self.sorting = self.perform_spike_sorting()

             # Perform curation
            if not os.path.exists(os.path.join(self.sorting_output_path, "wf_folder_curated")):

                # Curate the spike sorting
                console.info(f"Curating the identified units...")
                self.sorting, self.waveform_extractor, self.good_unit_ids = self.curate_spike_sorting()
                #self.sorting = self.merge_spike_sorting()

                # Update the waveform extractor
                self.waveform_extractor = self.extract_waveforms(method=self.waveform_extraction_method,
                                                                folder="wf_folder_curated",
                                                                )
                # Remove units that have less than 3 electrodes
                self.good_unit_ids = self.remove_small_units()

                console.info(f"Spikes of file {self.filename} were successfully sorted.")

                if plot_sorting:
                    self.plot_sorting_results()
                    console.info(f"The sorting results were plotted and saved.")

    def check_sorting(self, sorting_output_path):
        var = None
        try:
            # Open and read the JSON file
            with open(os.path.join(sorting_output_path, 'spikeinterface_log.json'), 'r') as file:
                data = json.load(file)
            # Check if the "error" key exists and its value
            if "error" in data:
                if data["error"]:
                    console.info("Error occurred during spike sorting. Performing spike sorting again.")
                    var = 'error_true'
                else:
                    console.info("No error occurred during spike sorting. Loading the sorting results.")
                    var = 'error_false'
            else:
                console.info("Key 'error' not found in the JSON file.")

        except FileNotFoundError:
            console.info(f"File spikeinterface_log not found.")
            var = 'log not found'
        except json.JSONDecodeError:
            console.info("Error decoding the JSON file.")
        except Exception as e:
            console.info(f"An unexpected error occurred: {e}")

        return var

    def extract_waveforms(self, method: str, folder: str, threshold=5):
        """
        Extracts the waveforms from the recordings based on the waveform templates.

        :param method: (str) Name of the method to use. Common options: "best_channels", "snr" or "ptp".
        :param folder: (str) Folder to which the waveforms should be saved.
        :param threshold: (int) Threshold to apply for template matching using the snr or ptp method.
        :return si.WaveformExtractor: Waveform extractor object containing the extracted waveforms.
        """
        if method not in ["snr", "ptp", "best_channels"]:
            method = "best_channels"
            console.warning("Method could not be recognized. Resorting to best_channels method.")

        self.waveform_extractor = si.extract_waveforms(recording=self.recording, sorting=self.sorting,
                                                       folder=os.path.join(self.sorting_output_path, folder),
                                                       ms_before=2, ms_after=2, max_spikes_per_unit=500,
                                                       n_jobs=1, chunk_size=20000, total_memory="1G",
                                                       method=f"{method}", threshold=threshold,
                                                       num_spikes_for_sparsity=100,
                                                       num_channels=self.num_channels_per_unit, overwrite=True)

        return self.waveform_extractor

    def perform_spike_sorting(self):
        """
        Performs spike sorting on the loaded recording using the specified sorting algorithm.

        :return si.sorters.SortingExtractor: Sorting object containing the spike sorting results.
        """
        console.info(f"Starting spike sorting of {self.filename}...")
        # Set parameters
        sorter_params = si.sorters.get_default_sorter_params(sorter_name_or_class="spykingcircus2")
        sorter_params["apply_preprocessing"] = False
        sorter_params["general"]["radius_um"] = 2000
        sorter_params["detection"]["detect_threshold"] = self.spike_threshold
        sorter_params["detection"]["exclude_sweep_ms"] = self.spike_distance
        sorter_params["job_kwargs"]["n_jobs"] = self.n_jobs

        # perform spike sorting
        sorting = si.sorters.run_sorter(sorter_name='spykingcircus2',
                                        recording=self.recording,
                                        output_folder=self.sorting_output_path,
                                        verbose=True,
                                        **sorter_params
                                        )

        console.info(f"Preliminary spike sorting of {self.filename} done.")

        del sorter_params

        return sorting

    def curate_spike_sorting(self, check_isi=False):
        """
        Curates the spike sorting results by removing empty units and checking quality metrics.

        :param check_isi: (bool, optional) Whether to check ISI violations. Default is False.
        :return tuple: Tuple containing the curated sorting, waveform extractor, and good unit IDs.
        """
        # Remove empty units
        self.sorting = self.sorting.remove_empty_units()

        # Check the quality of the clusters
        self.waveform_extractor = self.extract_waveforms(method="best_channels", folder="wf_folder")

        metrics = si.qualitymetrics.compute_quality_metrics(self.waveform_extractor,
                                                            metric_names=['snr', 'isi_violation', "firing_rate"])

        if check_isi:
            keep_mask = (metrics['snr'] > 5) & (metrics['isi_violations_ratio'] < 0.05) & (metrics['firing_rate'] > 0.10
                                                                                           )
        else:
            keep_mask = (metrics['snr'] > 5) & (metrics['firing_rate'] > 0.10)

        self.good_unit_ids = keep_mask[keep_mask].index.values

        self.sorting = self.sorting.select_units(self.good_unit_ids)
        self.waveform_extractor = self.waveform_extractor.select_units(self.good_unit_ids)

        console.info(f"Quality metrics were checked. {len(self.good_unit_ids)} clusters were kept.")

        # Save the numpy array containing the metrics
        np.save(os.path.join(self.sorting_output_path, "clustering_metrics.npy"), metrics)

        del metrics
        del keep_mask

        return self.sorting, self.waveform_extractor, self.good_unit_ids

    def merge_spike_sorting(self, threshold=0.8):
        """
        Merges similar clusters based on a similarity threshold.

        :param threshold: (float, optional) Similarity threshold for merging clusters. Default is 0.8.
        :return si.sorters.SortingExtractor: Sorting object after merging similar clusters.
        """
        # Extract similarity matrix
        similarity_matrix = si.postprocessing.compute_template_similarity(self.waveform_extractor,
                                                                          load_if_exists=False,
                                                                          method='cosine_similarity')

        # Iterate through the similarity matrix to find clusters that can be merged
        merges = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                if i < j:
                    value = similarity_matrix[i][j]
                    if value > threshold:
                        merges.append(([self.good_unit_ids[i], self.good_unit_ids[j]], value))

        try:
            # Sort pairs by value in descending order
            merges.sort(key=lambda x: x[1], reverse=True)

            # Track selected pairs
            selected_merges = []
            selected_indices = set()

            # Select pairs ensuring no index repeats
            for pair, value in merges:
                i, j = pair[0], pair[1]
                if i not in selected_indices and j not in selected_indices:
                    selected_merges.append(pair)
                    selected_indices.add(i)
                    selected_indices.add(j)

            # Merge clusters
            self.sorting = si.curation.MergeUnitsSorting(parent_sorting=self.sorting, units_to_merge=selected_merges)
            self.good_unit_ids = self.sorting.get_unit_ids()

            console.info(f"Clusters were checked for merging. {len(selected_merges)} cluster pairs were merged.")

            del similarity_matrix
            del merges
            del selected_indices
            del selected_merges

        except Exception as e:
            console.warning(f"Merging of clusters was not possible. Error: {e}")

        return self.sorting

    def remove_small_units(self, min_electrodes=3):
        """
        Removes units with fewer than the specified minimum number of electrodes.

        :param min_electrodes: (int, optional) Minimum number of electrodes a unit must have. Default is 3.
        :return list: List of good unit IDs after removing small units.
        """
        self.unit_id_to_electrode_ids = self.get_units_to_electrode_ids()
        keys_with_less_than_3_entries = [key for key, values in self.unit_id_to_electrode_ids.items() if
                                         len(values) < 3]
        self.good_unit_ids = [unit for unit in self.good_unit_ids if
                              unit.astype(str) not in keys_with_less_than_3_entries]

        self.unit_id_to_electrode_ids = {key: self.unit_id_to_electrode_ids[str(key)] for key in self.good_unit_ids}
        self.sorting = self.sorting.select_units(self.good_unit_ids)

        console.info(f"Clusters were checked for size (min. {min_electrodes} electrodes). "
                     f"{len(keys_with_less_than_3_entries)} clusters were removed from the dataset.")

        del keys_with_less_than_3_entries

        return self.good_unit_ids

    def plot_sorting_results(self):
        """
        Plots the results of the spike sorting and saves the plots to the output path.
        """
        console.info(f"Plotting sorting results...")
        si.postprocessing.compute_unit_locations(self.waveform_extractor)
        figure = si.widgets.plot_unit_templates(self.waveform_extractor, ncols=5, figsize=(8, 16), same_axis=True,
                                                plot_channels=True, unit_ids=self.good_unit_ids)
        plt.savefig(os.path.join(self.sorting_output_path, 'wf_folder_curated/templates.svg'))

        figure1 = si.widgets.plot_unit_templates(self.waveform_extractor, ncols=10, figsize=(15, 20), same_axis=False,
                                                 plot_channels=True, unit_ids=self.good_unit_ids)
        plt.savefig(os.path.join(self.sorting_output_path, 'wf_folder_curated/templates_single.svg'))

        # figure1 = si.widgets.plot_unit_waveforms(self.waveform_extractor, ncols=5, figsize=(8, 16), same_axis=True,
        #                                         plot_channels=True, unit_ids=self.good_unit_ids)
        # plt.savefig(os.path.join(self.sorting_output_path, 'wf_folder_curated/waveforms.svg'))

        figure2 = si.widgets.plot_unit_locations(self.waveform_extractor, figsize=(20, 10), plot_legend=True,
                                                 unit_ids=self.good_unit_ids)
        plt.savefig(os.path.join(self.sorting_output_path, 'wf_folder_curated/locations.svg'))

    def find_blanking_indices(self, blanking_threshold: int, blanking_distance: int):
        """
        Finds the indices for blanking based on the given threshold and distance.

        :param blanking_threshold: (int) Threshold for blanking.
        :param blanking_distance: (int) Minimum distance between blanking events.
        :return list: List of indices for blanking.
       """
        blanking_indices = [0]
        console.info(f"Starting blanking...")
        # Get traces
        traces_filtered = self.recording.get_traces()

        # Plot artefact removal
        for single_trace in traces_filtered.T[0::3]:
            artefacts = (fp(np.abs(single_trace), height=blanking_threshold, distance=blanking_distance))[0]
            blanking_indices.extend(artefacts)
            del artefacts

        blanking_indices = np.unique(blanking_indices)
        blanking_indices.sort()
        blanking_indices = np.squeeze(np.asarray(blanking_indices))

        console.info(f"Successfully found blanking indices for recording {self.filename}.")

        del traces_filtered

        return blanking_indices

    def get_electrode_channel_mapping(self, raw_data) -> np.array:
        """
        Creates a mapping between electrode and channel IDs.

        :param raw_data: Raw data object (not used in the method).
        :return np.array: Array mapping electrode IDs to channel IDs.
        """

        with open(os.path.join(self.sorting_output_path, "wf_folder_curated/recording_info/recording_attributes.json"),
                  'r') as file:
            # Load electrode and channel ids from the file
            rec_attributes = json.load(file)
            self.electrodes = rec_attributes["properties"]["electrode"]
            self.channel_ids = rec_attributes["channel_ids"]

            # Create a mapping dictionary
            self.electrode_channel_mapping = dict(zip(self.channel_ids, self.electrodes))

            del rec_attributes

            return self.electrode_channel_mapping

    def get_units_to_electrode_ids(self):
        """
        Creates a mapping of unit IDs to electrode IDs based on sparsity data.

        :return dict: Dictionary mapping unit IDs to electrode IDs.
        """
        with open(os.path.join(self.sorting_output_path, "wf_folder_curated/sparsity.json"), 'r') as file:
            # Load sparsity data from the file
            sparsity_data = json.load(file)
            unit_id_to_channel_ids = sparsity_data["unit_id_to_channel_ids"]
            self.get_electrode_channel_mapping(raw_data=None)
            self.unit_id_to_electrode_ids = {key: [self.electrode_channel_mapping[f'{x}'] for x in value] for key, value
                                             in unit_id_to_channel_ids.items()}

            del sparsity_data
            del unit_id_to_channel_ids

        return self.unit_id_to_electrode_ids

    def get_spike_matrix_extremum(self):
        """
        Creates a spike matrix from the loaded spike data based on extremum channels.
        Extremum channels are defined as the channels with the largest peaks in one unit.

        :return np.array: Numpy array containing spike matrix data.
        """
        # Get the electrode to channel mapping
        self.electrode_channel_mapping = self.get_electrode_channel_mapping(raw_data=None)

        # Extract the spiking information from the extremum channels
        extremum_channel = si.get_template_extremum_channel(self.waveform_extractor, outputs="index")
        spike_vector = self.sorting.to_spike_vector(concatenated=True, extremum_channel_inds=extremum_channel)
        spike_df = pd.DataFrame(spike_vector)

        # Convert sample indices to milliseconds
        time_in_ms = (spike_vector["sample_index"] / self.sample_frequency) * 1000
        spike_df["time"] = time_in_ms

        # Add the 'Electrode' column to the DataFrame
        spike_df["channel"] = spike_df["channel_index"].map(lambda x: self.channel_ids[x])
        spike_df['Electrode'] = spike_df['channel'].map(lambda x: self.electrode_channel_mapping[f'{x}'])

        # Convert to a numpy array
        df = spike_df[["Electrode", "time", "unit_index"]]
        dtype = [('Electrode', 'U10'), ('Spike_Time', 'f8'), ('UnitIdx', 'i4')]
        self.spike_mat_extremum = np.array(list(df.itertuples(index=False, name=None)), dtype=dtype)

        del extremum_channel
        del spike_vector
        del spike_df
        del time_in_ms
        del df
        del dtype

        return self.spike_mat_extremum

    def get_spike_matrix(self):
        """
        Creates a spike matrix from the detected peaks in the recording.

        :return np.array: Numpy array containing spike matrix data.
        """
        all_spikes = pd.DataFrame(si.sortingcomponents.peak_detection.detect_peaks(self.recording,
                                                                                   method="by_channel",
                                                                                   peak_sign="neg",
                                                                                   detect_threshold=self.spike_threshold,
                                                                                   exclude_sweep_ms=self.spike_distance,
                                                                                   n_jobs=self.n_jobs,
                                                                                   chunk_size=20000,
                                                                                   total_memory="1G",
                                                                                   ))

        if not self.perform_sorting:
            # Load electrode to channel mapping
            try:
                raw_data = h5py.File(os.path.join(self.input_path, self.filename), "r")
                self.electrodes = list(np.asarray(raw_data["mapping"]["electrode"]))
            except Exception as e:
                console.error(f"An error occurred while loading data: {str(e)}")
                return []  # Return empty list to indicate an error

        # convert channel_index to electrode number
        all_spikes['Electrode'] = all_spikes['channel_index'].map(lambda x: self.electrodes[x])

        # Convert sample indices to milliseconds
        time_in_ms = (all_spikes["sample_index"] / self.sample_frequency) * 1000
        all_spikes["time"] = time_in_ms

        # Convert to a numpy array
        df = all_spikes[["Electrode", "time", "amplitude"]]

        dtype = [('Electrode', 'U10'), ('Spike_Time', 'f8'), ('Amplitude', 'f8')]
        self.spike_mat = np.array(list(df.itertuples(index=False, name=None)), dtype=dtype)

        del all_spikes
        del time_in_ms
        del df
        del dtype

        return self.spike_mat
