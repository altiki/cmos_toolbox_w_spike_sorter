# Import libraries
import h5py
import os
from collections import Counter
import pickle
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks as fp
import spikeinterface.full as si
from spikeinterface.sortingcomponents import peak_detection
from abc import ABC, abstractmethod
import sys
import random

from src.utils.logger_functions import console
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


class ABCExtractor(ABC):
    """
    Abstract base class for extracting electrophysiological data and processing it for spike sorting.
    """

    @abstractmethod
    def __init__(
        self,
        filename: str,
        input_path: str,
        output_path: str,
        sample_frequency=20000, # Hz
        electrode_dict_path ='',
        experiment_duration=None,
        id_chip=None,
        id_network=None,
        div=None,
        cell_type=None,
        spike_threshold=5, # Standard deviations
        spike_distance=100, # Samples
        spike_window = [-75, 75], # Samples
        delay = None,
        blank_block_bool=False,
        no_stim_duration=3000, # Samples
        blanking_thresh=100, # Singal amplitude uV
        blanking_window=[-10, 100]
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
        self.cell_type = cell_type

        self.chip_width = 220
        self.spike_threshold = spike_threshold
        self.spike_distance = spike_distance
        self.spike_window = spike_window

        self.electrode_dict_path = electrode_dict_path
        self.electrode_dict = None
        self.stimulated_electrodes_list = None
        self.delay = delay
        self.blank_block_bool = blank_block_bool
        self.blanking_thresh = blanking_thresh
        self.blanking_window = blanking_window
        self.end_stim = None
        self.blank_block_bool = blank_block_bool
        self.no_stim_duration = no_stim_duration
        self.stim_electrode_idx = 0
        self.multiple_stim_electrodes = True

        self.spikes = []
        self.traces = None
        self.channels = None
        self.spike_mat = None
        self.spike_mat_extremum = None
        self.spike_dict = None
        self.electrode_channel_mapping = None
        self.unit_id_to_electrode_ids = None
        

        if len(self.electrode_dict_path) != 0:
            with open(electrode_dict_path, "rb") as f:
                self.electrode_dict = pickle.load(f)
                try:
                    self.stimulated_electrodes_list = self.electrode_dict["stimulus_electrodes"]
                    self.recording_electrodes = self.electrode_dict["recording_electrodes"]
                    console.info("Electrode dictionary was successfully loaded.")
                except KeyError as e:
                    console.error(f"An error occurred while loading stimulation/recording electrode dictionary: {str(e)}")

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
            'UNIT_TO_EL': self.unit_id_to_electrode_ids
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
        sample_frequency=20000, # Hz
        electrode_dict_path ='',
        experiment_duration=None,
        id_chip=None,
        id_network=None,
        div=None,
        cell_type=None,
        spike_threshold=5, # Standard deviations
        spike_distance=100, # Samples
        spike_window = [-75, 75], # Samples
        delay = None,
        blank_block_bool=False,
        no_stim_duration=3000, # Samples
        blanking_thresh=100, # Singal amplitude uV
        blanking_window=[-10, 100]

    ):

        self.filename = filename
        self.input_path = input_path
        self.output_path = output_path

        self.sample_frequency = sample_frequency
        self.experiment_duration = experiment_duration
        self.id_chip = id_chip
        self.id_network = id_network
        self.div = div
        self.cell_type = cell_type

        self.chip_width = 220
        self.spike_threshold = spike_threshold
        self.spike_distance = spike_distance
        self.spike_window = spike_window

        self.electrode_dict_path = electrode_dict_path
        self.electrode_dict = None
        self.stimulated_electrodes_list = None
        self.delay = delay
        self.blank_block_bool = blank_block_bool
        self.blanking_thresh = blanking_thresh
        self.blanking_window = blanking_window
        self.end_stim = None
        self.blank_block_bool = blank_block_bool
        self.no_stim_duration = no_stim_duration
        self.stim_electrode_idx = 0
        self.multiple_stim_electrodes = True

        self.spikes = []
        self.traces = None
        self.channels = None
        self.spike_mat = None
        self.spike_mat_extremum = None
        self.electrode_burst_mat = None
        self.spike_dict = None
        self.electrode_metrics_columns = None
        self.network_metrics_columns = None
        self.electrode_channel_mapping = None

        if len(self.electrode_dict_path) != 0:
            with open(electrode_dict_path, "rb") as f:
                self.electrode_dict = pickle.load(f)
                try:
                    if self.delay == 0:
                        self.stimulated_electrodes_list = [self.electrode_dict["stimulus_electrodes"][0]]
                    else:
                        self.stimulated_electrodes_list = self.electrode_dict["stimulus_electrodes"]

                    self.recording_electrodes = self.electrode_dict["recording_electrodes"]
                    console.info("Electrode dictionary was successfully loaded.")
                except KeyError as e:
                    console.error(f"An error occurred while loading stimulation/recording electrode dictionary: {str(e)}")

    def load_data(
        self,
        cutoff_frequency=200,
        filter_order=2,
        loading_steps=10, #usually 10
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
        blanking_indices = []
        begin_stim = None #for blanking block
        end_stim = None  #for blanking block
        

        if self.stimulated_electrodes_list is not None:
                console.info("Blanking in progress...")
            
                while self.multiple_stim_electrodes == True:
                    electrode = self.stimulated_electrodes_list[self.stim_electrode_idx]
                    #console.info(f"Blanking from a stimulation electrode {electrode}...")
                    channel = self.electrode_channel_mapping[
                        1, np.argwhere(self.electrode_channel_mapping[0, :] == electrode)
                    ]
                    if len(channel) != 0:
                        channel = channel[0]
                    else:
                        self.stim_electrode_idx +=1
                        if self.stim_electrode_idx == len(self.stimulated_electrodes_list):
                            sys.exit("No stimulation electrodes found in the recording!")
                            
                        continue
                    
                    traces = raw_data.get("sig")[np.squeeze(channel), :] ###
                    print(f"Traces shape for channel {channel}: {traces.shape}")

                    traces = lfilter(coeff_b, coeff_a, traces)

                    if first_time:
                        self.experiment_duration = len(traces)
                        self.experiment_duration_blanked = self.experiment_duration
                        first_time = False

                    artefacts = (
                        fp(np.abs(traces), height=self.blanking_thresh, distance=self.spike_distance)
                    )[0]
                    blanking_indices.extend(artefacts)
                    del artefacts
                    del traces
                        
                    blanking_indices.sort()
                    blanking_indices = np.squeeze(np.asarray(blanking_indices))
                    
                    if self.blank_block_bool:
                            differences = np.diff(blanking_indices)
                            end_temp = (
                                blanking_indices[np.argwhere(differences > self.no_stim_duration)]
                                + self.blanking_window[1]
                            )
                            end_temp[np.argwhere(end_temp >= self.experiment_duration)] = (
                                self.experiment_duration - 1
                            )
                            begin_temp = (
                                blanking_indices[np.argwhere(differences > self.no_stim_duration) + 1]
                                + self.blanking_window[0]
                            )
                            begin_temp[np.argwhere(begin_temp < 0)] = 0
                            end_stim = np.zeros(end_temp.shape[0] + 1, dtype=np.int32)
                            begin_stim = np.zeros(begin_temp.shape[0] + 1, dtype=np.int32)
                            begin_stim[0] = max(blanking_indices[0] + self.blanking_window[0], 0)
                            end_stim[-1] = min(
                                blanking_indices[-1] + self.blanking_window[1], self.experiment_duration
                            )
                            if end_stim.shape[0] > 1:
                                end_stim[0 : end_stim.shape[0] - 1] = np.squeeze(end_temp)
                            if begin_stim.shape[0] > 1:
                                begin_stim[1:] = np.squeeze(begin_temp)
                            del end_temp
                            del begin_temp
                            if self.delay == 0:
                                self.multiple_stim_electrodes = False
                            else:
                                self.multiple_electrodes = True
                                self.stim_electrode_idx += 1
                            if self.stim_electrode_idx == len(self.stimulated_electrodes_list):
                                self.multiple_stim_electrodes = False

                        
        if end_stim is not None:
            self.blanking_end = end_stim
        
        if len(blanking_indices) != 0:
            self.blanking_end = np.asarray(blanking_indices) + self.spike_window[1]


        # Set up the loading steps (enables to load traces in loading_steps steps)
        loading_indices_start = [0]
        loading_indices_end = []

        
        
        channels_for_traces = self.electrode_channel_mapping[1, :]
    

        #Load only non-stimulation channels
        if len(self.electrode_dict_path) != 0:
                print("Exculding stimulated electrodes from recording...")
                counts = Counter(channels_for_traces) 
                channels_for_traces = [elem for elem in channels_for_traces if counts[elem] == 1] #exclude repeating channels
            
        for i in range(1, loading_steps - 1):
            step = int(i / loading_steps * len(channels_for_traces))
            console.info(f"Loading step {i} of {loading_steps}...")
            loading_indices_end.append(step)
            loading_indices_start.append(step)
        loading_indices_end.append(len(channels_for_traces))

        # Find peaks in the traces
        for j in range(len(loading_indices_start)):
            self.channels = channels_for_traces[loading_indices_start[j]:loading_indices_end[j]]
            self.channels = np.unique(self.channels)
            self.traces = raw_data.get("sig")[np.squeeze(self.channels), :] ###
            #np.save(f"{self.output_path}/traces_before_blanking_{j}.npy", self.traces)



            if self.traces.ndim == 1:
                self.traces = np.expand_dims(self.traces, axis=0)

            if first_time:
                # Get experiment duration
                self.experiment_duration = self.traces.shape[1] / self.sample_frequency
                self.experiment_duration_blanked = self.experiment_duration                
                first_time = False

            self.traces = lfilter(coeff_b, coeff_a, self.traces)
            #np.save(f"{self.output_path}/traces_before_blanking_filtered_{j}.npy", self.traces)

            if begin_stim is not None:
                for i in range(begin_stim.shape[0]):
                    self.traces[:, np.arange(begin_stim[i], end_stim[i])] = 0
                if self.experiment_duration_blanked == self.experiment_duration:
                    self.experiment_duration_blanked -= np.sum(
                        np.squeeze(end_stim - begin_stim)
                    )
            else:
                duration_blanked = 0
                for i in blanking_indices:
                    self.traces[
                        :,
                        max(0, i + self.blanking_window[0]) : min(
                            self.traces.shape[1], i + self.blanking_window[1]
                        ),
                    ] = 0
                    duration_blanked += min(
                        self.traces.shape[1], i + self.blanking_window[1]
                    ) - max(0, i + self.blanking_window[0])
                if self.experiment_duration_blanked == self.experiment_duration:
                    self.experiment_duration_blanked -= duration_blanked


            #np.save(f"{self.output_path}/traces_after_blanking_{j}.npy", self.traces)
            self.spikes.extend(list(map(self.find_peaks, range(self.traces.shape[0]))))
            #np.save(f"{self.output_path}/spikes_{j}.npy", self.spikes)

            


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

        dtype = [('Electrode', 'U10'), ('Spike_Time', 'f8'), ('Amplitude', 'f8')]
        self.spike_mat = np.array(spike_list, dtype=dtype)
       

 
    def get_spike_matrix_extremum(self):
        unique_electrodes = np.unique(self.spike_mat['Electrode'])
        self.spike_mat_extremum = np.zeros(len(unique_electrodes), dtype=[('Electrode', 'U10'), ('Spike_Time', 'f8'), ('Amplitude', 'f8')])

        for i, unit in enumerate(unique_electrodes):
            unit_spikes = self.spike_mat[self.spike_mat['Electrode'] == unit]
            if unit_spikes.size == 0:
                print(f"No spikes found for electrode {unit}")
                continue
            
            max_amplitude_idx = np.argmax(unit_spikes['Amplitude'])
            
            # Assign values explicitly to each field
            self.spike_mat_extremum[i]['Electrode'] = unit
            self.spike_mat_extremum[i]['Spike_Time'] = unit_spikes[max_amplitude_idx]['Spike_Time']
            self.spike_mat_extremum[i]['Amplitude'] = unit_spikes[max_amplitude_idx]['Amplitude']

        #return self.spike_mat_extremum



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
            spike_distance=1,
            waveform_extraction_method="best_channels",
            n_jobs=6,
            num_channels_per_unit=25
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

        self.sorting_output_path = os.path.join(self.output_path, "Sorter_"+self.filename+"_kilosort")
        print(f"Sorting output path: {self.sorting_output_path}")

        self.waveform_extraction_method = waveform_extraction_method
        self.n_jobs = n_jobs
        self.num_channels_per_unit = num_channels_per_unit

        self.sorting = None
        self.recording = None
        self.analyzer = None
        self.metric_names = None
        self.good_unit_ids = []
        self.electrodes = []


    def check_sorting(self,sorting_output_path):
        var = None
        try:
            # Open and read the JSON file
            with open(os.path.join(sorting_output_path,'spikeinterface_log.json'), 'r') as file:
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

    def load_data(
            self,
            freq_min=200,
            freq_max=None,
            blanking=False,
            clipping=True,
            blanking_threshold=80,
            blanking_distance=80,
            blanking_window=(3, 3),
            plot_sorting=False
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
            self.recording = si.read_maxwell(file_path=os.path.join(self.input_path, self.filename))
            console.info(f"Successfully loaded the recording {self.filename}.")
        except Exception as e:
            console.error(f"An error occurred while loading data: {str(e)}")
            return []

        self.experiment_duration = self.recording.get_duration()

        # Filter recording
        if freq_max is not None:
            self.recording = si.bandpass_filter(recording=self.recording,
                                                              freq_min=freq_min,
                                                              freq_max=freq_max,
                                                              dtype='float32'
                                                              )
        else:
            self.recording = si.bandpass_filter(recording=self.recording,
                                                              freq_min=freq_min,
                                                              dtype='float32')

        # Apply common reference to reduce the noise
        self.recording = si.common_reference(self.recording)

        console.info(f"Successfully filtered the recording {self.filename}.")

        # Blank if necessary
        if blanking:
            blanking_indices = self.find_blanking_indices(blanking_threshold, blanking_distance)
            self.recording = si.remove_artifacts(self.recording,
                                                               blanking_indices,
                                                               ms_before=blanking_window[0],
                                                               ms_after=blanking_window[1],
                                                               mode='zeros'
                                                               )
        if clipping:
            console.info(f"Clipping the recording {self.filename}...")
            self.recording = si.clip(recording=self.recording, a_max=1200)

        # Perform spike sorting
        if os.path.exists(self.sorting_output_path):
            console.info(f"Sorting output path {self.sorting_output_path} already exists.")
            var = self.check_sorting(self.sorting_output_path)
            if var == 'error_true' or var == 'log not found':
                self.sorting = self.perform_spike_sorting()
            elif var == 'error_false':
                self.sorting = si.read_sorter_folder(self.sorting_output_path)
            
        else:
            console.info(f"Sorting output path {self.sorting_output_path} does not exist. Performing spike sorting.")
            self.sorting = self.perform_spike_sorting()    
        
        if os.path.exists(os.path.join(self.sorting_output_path, "analyzer_curated")):
            console.info("Analyzer curated folder already exists. Loading the analyzer...")
            self.analyzer = si.load_sorting_analyzer(os.path.join(self.sorting_output_path, "analyzer_curated"))
            self.good_unit_ids = self.analyzer.unit_ids
            self.get_spike_matrix()
            if plot_sorting:
                self.plot_sorting_results()
                console.info(f"The sorting results were plotted and saved.")
            else:
                console.info(f"Analysis of {self.filename} finished. No plots were generated.")
        else:
            console.info("Analyzer curated folder does not exist. Creating the analyzer curated...")
            # Curate the spike sorting
            console.info(f"Curating the identified units...")
            self.sorting, self.analyzer, self.good_unit_ids = self.curate_spike_sorting()

            self.sorting = self.merge_spike_sorting()

            # Update the waveform extractor
            self.analyzer, self.good_unit_ids = self.extract_waveforms(folder="analyzer_curated")
            if plot_sorting:
                self.plot_sorting_results()
                console.info(f"The sorting results were plotted and saved. Analysis of {self.filename} finished.")
            else:
                console.info(f"No plots were generated. Analysis of {self.filename} finished.")


    def extract_waveforms(self, folder: str):
        """
        Extracts the waveforms from the recordings based on the waveform templates.

        :param method: (str) Name of the method to use. Common options: "best_channels", "snr" or "ptp".
        :param folder: (str) Folder to which the waveforms should be saved.
        :param threshold: (int) Threshold to apply for template matching using the snr or ptp method.
        :return si.WaveformExtractor: Waveform extractor object containing the extracted waveforms.
        """ 

        self.analyzer.save_as(folder = os.path.join(self.sorting_output_path, folder), format="binary_folder")
        self.good_unit_ids = self.analyzer.unit_ids

        return self.analyzer, self.good_unit_ids

    def perform_spike_sorting(self):
        """
        Performs spike sorting on the loaded recording using the specified sorting algorithm.

        :return si.sorters.SortingExtractor: Sorting object containing the spike sorting results.
        """
        console.info(f"Starting spike sorting of {self.filename}...")
        # Set parameters
        sorter_params = si.get_default_sorter_params(sorter_name_or_class="kilosort3")
        #sorter_params["apply_preprocessing"] = False
        #sorter_params["general"]["radius_um"] = 2000
        #sorter_params["detection"]["detect_threshold"] = self.spike_threshold
        #sorter_params["job_kwargs"]["n_jobs"] = self.n_jobs
        #sorter_params["clustering"]["method"] = "hdbscan"
        sorter_params["detect_threshold"] = self.spike_threshold
        sorter_params["skip_kilosort_preprocessing"] = True
        sorter_params["keep_good_only"] = True

        # perform spike sorting
        sorting = si.run_sorter(sorter_name='kilosort3',
                                        recording=self.recording,
                                        output_folder=self.sorting_output_path,
                                        remove_existing_folder= True,
                                        verbose=True,
                                        singularity_image="spikeinterface/kilosort3-compiled-base:0.1.0",
                                        **sorter_params
                                        )

        console.info(f"Spike sorting of {self.filename} done.")

        return sorting

    def curate_spike_sorting(self, check_isi=False):
        """
        Curates the spike sorting results by removing empty units and checking quality metrics.

        :param check_isi: (bool, optional) Whether to check ISI violations. Default is False.
        :return tuple: Tuple containing the curated sorting, waveform extractor, and good unit IDs.
        """
        job_kwargs = dict(n_jobs=10, chunk_duration='1s', progress_bar=True)
        #sparsity_kwargs = job_kwargs
        # Remove empty units
        self.sorting = self.sorting.remove_empty_units()

        if os.path.exists(os.path.join(self.sorting_output_path, "analyzer")):
            console.info("Analyzer folder already exists. Loading the analyzer...")
            self.analyzer = si.load_sorting_analyzer(os.path.join(self.sorting_output_path, "analyzer"))
            metric_names = si.get_quality_metric_list()
            metrics = si.compute_quality_metrics(self.analyzer, metric_names=metric_names)
        else:
            console.info("Analyzer folder does not exist. Creating the analyzer...")
        
            self.analyzer = si.create_sorting_analyzer(self.sorting, self.recording, format = "memory", folder = os.path.join(self.sorting_output_path,"analyzer_folder"), sparse = False, **job_kwargs)


            metric_names = si.get_quality_metric_list()

            self.analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500,**job_kwargs)
            console.info("Random spikes - done")
            self.analyzer.compute("templates", operators=["average", "std"],**job_kwargs)
            console.info("Templates - done")
            self.analyzer.compute("noise_levels",**job_kwargs)
            console.info("Noise levels - done")
    
            metrics = si.compute_quality_metrics(self.analyzer, metric_names=metric_names)
            console.info(f"Quality metrics {metric_names} calculated. ")

            self.analyzer.save_as(folder = os.path.join(self.sorting_output_path, "analyzer"), format="binary_folder")

        #template_ext = self.analyzer.load_extension("templates")
        #templates = template_ext.get_templates(operator="average")
        #templates_object = si.Templates(templates_array=templates, sampling_frequency=self.recording.get_sampling_frequency(), nbefore = 20)
        noise_levels = self.analyzer.get_extension("noise_levels").get_data()
        if self.waveform_extraction_method == "best_channels":
            self.sparsity = si.compute_sparsity(self.analyzer, self.recording, method = 'best_channels', num_channels = self.num_channels_per_unit)
        elif self.waveform_extraction_method == "snr":
            self.sparsity = si.compute_sparsity(templates_or_sorting_analyzer=self.analyzer, method = 'snr', threshold = 5, noise_levels=noise_levels)


        self.analyzer = si.create_sorting_analyzer(self.sorting, self.recording, format = "memory", folder = os.path.join(self.sorting_output_path,"analyzer_folder"), overwrite = True, sparse = True, sparsity = self.sparsity, **job_kwargs)

        if check_isi:
            keep_mask = (metrics['snr'] > 5) & (metrics['isi_violations_ratio'] < 0.05) & (metrics['firing_rate'] > 0.10)
        else:
            print("This also happens")
            keep_mask = (metrics['snr'] > 5) & (metrics['firing_rate'] > 0.10)


        self.good_unit_ids = keep_mask[keep_mask].index.values

        # Remove units that have less than 3 electrodes
        self.good_unit_ids = self.remove_small_units()
        self.sorting = self.sorting.select_units(unit_ids = self.good_unit_ids)
        self.analyzer = self.analyzer.select_units(unit_ids = self.good_unit_ids)

        self.analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500,**job_kwargs)
        console.info("Random spikes - done")
        self.analyzer.compute("waveforms", **job_kwargs)
        console.info("Waveforms - done")
        self.analyzer.compute("templates", operators=["average", "std"],**job_kwargs)
        console.info("Templates - done")
        #self.analyzer.compute("template_similarity",**job_kwargs)
        #console.info("Template similarity - done")
        self.analyzer.compute("noise_levels",**job_kwargs)
        console.info("Noise levels - done")
        self.analyzer.compute("spike_locations",**job_kwargs)
        console.info("Spike locations - done")
        self.analyzer.compute("spike_amplitudes", **job_kwargs)
        console.info("Spike amplitudes - done")
        self.analyzer.compute("correlograms", window_ms = 20.0, bin_ms = 5.0, **job_kwargs)

        template_metrics = si.compute_template_metrics(self.analyzer, include_multi_channel_metrics = True)

        console.info(f"Quality metrics were checked. {len(self.good_unit_ids)} clusters were kept.")
        if len(self.good_unit_ids) == 0:
            console.error("No clusters were kept after quality check. Please check the quality metrics.")

        # Save the numpy array containing the metrics
        metrics.to_csv(os.path.join(self.sorting_output_path, "clustering_metrics.csv"))
        template_metrics.to_csv(os.path.join(self.sorting_output_path, "template_metrics.csv"))

        return self.sorting, self.analyzer, self.good_unit_ids

    def merge_spike_sorting(self, threshold=0.8):    
        """
        Merges similar clusters based on a similarity threshold.

        :param threshold: (float, optional) Similarity threshold for merging clusters. Default is 0.8.
        :return si.sorters.SortingExtractor: Sorting object after merging similar clusters.
        """
        # Extract similarity matrix
        similarity_matrix = si.compute_template_similarity(self.analyzer,
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
        self.sorting = si.MergeUnitsSorting(self.sorting, units_to_merge=selected_merges)
        self.good_unit_ids = self.sorting.get_unit_ids()

        console.info(f"Clusters were checked for merging. {len(selected_merges)} cluster pairs were merged.")

        return self.sorting

    def remove_small_units(self, min_electrodes=5):
        """
        Removes units with fewer than the specified minimum number of electrodes.

        :param min_electrodes: (int, optional) Minimum number of electrodes a unit must have. Default is 3.
        :return list: List of good unit IDs after removing small units.
        """
        self.unit_id_to_channel_ids = self.get_units_to_channel_ids()
        keys_with_less_than_min_el_entries = [key for key, values in self.unit_id_to_channel_ids.items() if
                                         len(values) < min_electrodes]
        print(keys_with_less_than_min_el_entries)
        #self.good_unit_ids = [unit for unit in self.good_unit_ids if unit.astype(str) not in keys_with_less_than_3_entries]
        self.good_unit_ids = [unit for unit in self.good_unit_ids if unit not in keys_with_less_than_min_el_entries]
        self.unit_id_to_channel_ids = {key: self.unit_id_to_channel_ids[key] for key in self.good_unit_ids}
        self.sorting = self.sorting.select_units(unit_ids = self.good_unit_ids)

        print(f"Clusters were checked for size (min. {min_electrodes} electrodes). "
                     f"{len(keys_with_less_than_min_el_entries)} clusters were removed from the dataset.")

        return self.good_unit_ids
    
    def plot_sorting_results(self):
        """
        Plots the results of the spike sorting and saves the plots to the output path.
        """
        console.info(f"Plotting sorting results...")
        job_kwargs = dict(n_jobs=6, chunk_duration='1s', progress_bar=True)
        si.compute_unit_locations(self.analyzer,**job_kwargs)
      
        figure = si.plot_unit_templates(self.analyzer, ncols=5, figsize=(8, 16), same_axis=True,
                                                plot_channels=True, unit_ids=self.good_unit_ids)
        plt.savefig(os.path.join(self.sorting_output_path, 'analyzer_curated/templates.svg'))
        for unit_id in self.good_unit_ids:
            print("Plotting templates...")
            template_dir = os.path.join(self.sorting_output_path, 'analyzer_curated', 'templates')
            os.makedirs(template_dir, exist_ok=True)
            figure1 = si.plot_unit_templates(self.analyzer, ncols=10, figsize=(15, 20), same_axis=False,
                                                    plot_channels=True, unit_ids=[unit_id])
            plt.savefig(os.path.join(template_dir, f'templates_unit_{unit_id}.svg'))
        
        if len(self.good_unit_ids) < 50:
            figure2 = si.plot_unit_locations(self.analyzer, figsize=(20, 10), plot_legend=True,
                                                 unit_ids=self.good_unit_ids)
        else:
            figure2 = si.plot_unit_locations(self.analyzer, figsize=(10, 10), plot_legend=True,
                                                 unit_ids= np.random.choice(self.good_unit_ids,50,replace=False))
        plt.savefig(os.path.join(self.sorting_output_path, 'analyzer_curated/locations.svg'))

        figure3 = si.plot_all_amplitudes_distributions(self.analyzer, unit_ids=self.good_unit_ids)
        plt.savefig(os.path.join(self.sorting_output_path, 'analyzer_curated/amplitudes.svg'))
        self.analyzer.compute("correlograms", **job_kwargs)
        for unit_id in self.good_unit_ids:
            print("Plotting waveforms...")
            waveform_dir = os.path.join(self.sorting_output_path, 'analyzer_curated', 'waveforms')
            os.makedirs(waveform_dir, exist_ok=True)
            figure4 = si.plot_unit_waveforms(self.analyzer, unit_ids=[unit_id])
            plt.savefig(os.path.join(waveform_dir,f'unit_{unit_id}_waveform.svg'))

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

        return blanking_indices

    def get_electrode_channel_mapping(self, raw_data) -> np.array:
        """
        Creates a mapping between electrode and channel IDs.

        :param raw_data: Raw data object (not used in the method).
        :return np.array: Array mapping electrode IDs to channel IDs.
        """

        with open(os.path.join(self.sorting_output_path, "analyzer_curated/recording_info/recording_attributes.json"),
                'r') as file:
            # Load electrode and channel ids from the file
            rec_attributes = json.load(file)
            self.electrodes = rec_attributes["properties"]["electrode"]
            channel_ids = rec_attributes["channel_ids"]

            #print("Len channel ids", len(channel_ids), "len self electrodes", len(self.electrodes))

            # Create a mapping dictionary
            self.electrode_channel_mapping = dict(zip(channel_ids, self.electrodes))

            del rec_attributes
            del channel_ids

            return self.electrode_channel_mapping

    def get_units_to_channel_ids(self):
        """
        Creates a mapping of unit IDs to electrode IDs based on sparsity data.

        :return dict: Dictionary mapping unit IDs to electrode IDs.
        """
        unit_id_to_channel_ids = self.analyzer.sparsity.unit_id_to_channel_indices
        print(type(unit_id_to_channel_ids.keys()))

        return unit_id_to_channel_ids
    
    def get_units_to_electrode_ids(self):
        """
        Creates a mapping of unit IDs to electrode IDs based on sparsity data.

        :return dict: Dictionary mapping unit IDs to electrode IDs.
        """
        unit_id_to_channel_ids = self.analyzer.sparsity.unit_id_to_channel_indices
        print(type(unit_id_to_channel_ids.keys()))
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
        # Extract the spiking information from the extremum channels
        extremum_channel = si.get_template_extremum_channel(self.analyzer)
        spike_vector = self.sorting.to_spike_vector(concatenated=True, extremum_channel_inds=extremum_channel)
        spike_df = pd.DataFrame(spike_vector)

        # Convert sample indices to milliseconds
        time_in_ms = (spike_vector["sample_index"] / self.sample_frequency) * 1000
        spike_df["time"] = time_in_ms

        # Add the 'Electrode' column to the DataFrame
        spike_df['Electrode'] = spike_df['channel_index'].map(lambda x: self.electrode_channel_mapping[f'{x}'])

        # Convert to a numpy array
        df = spike_df[["Electrode", "time", "unit_index"]]
        dtype = [('Electrode', 'U10'), ('Spike_Time', 'f8'), ('UnitIdx', 'i4')]
        self.spike_mat_extremum = np.array(list(df.itertuples(index=False, name=None)), dtype=dtype)

        return self.spike_mat_extremum
    
    def get_spike_matrix(self):
        """
        Creates a spike matrix from the detected peaks in the recording.

        :return np.array: Numpy array containing spike matrix data.
        """
        debug = True
        if debug:
            self.matrix = np.zeros((1000, 3))
        else:
            job_kwargs = dict(n_jobs=6, chunk_duration='1s', progress_bar=True)
            all_spikes = pd.DataFrame(peak_detection.detect_peaks(self.recording,
                                                                                    method="by_channel",
                                                                                    #peak_sign="neg",
                                                                                    #detect_threshold=self.spike_threshold,
                                                                                    #exclude_sweep_ms=self.spike_distance,
                                                                                    #n_jobs=self.n_jobs,
                                                                                    #chunk_size=20000,
                                                                                    **job_kwargs
                                                                                    ))
            self.electrode_channel_mapping = self.get_electrode_channel_mapping(raw_data=None)
            print(self.electrode_channel_mapping)
            print(all_spikes)
            # convert channel_index to electrode number
            all_spikes = all_spikes[all_spikes['channel_index'].map(lambda x: f'{x}' in self.electrode_channel_mapping)]
            all_spikes['Electrode'] = all_spikes['channel_index'].map(lambda x: self.electrode_channel_mapping[f'{x}']) ###ASK GIULIA
            print(all_spikes)
            # Convert sample indices to milliseconds
            time_in_ms = (all_spikes["sample_index"] / self.sample_frequency) * 1000
            all_spikes["time"] = time_in_ms

            # Convert to a numpy array
            df = all_spikes[["Electrode", "time", "amplitude"]]
            
            dtype = [('Electrode', 'U10'), ('Spike_Time', 'f8'), ('Amplitude', 'f8')]
            self.spike_mat = np.array(list(df.itertuples(index=False, name=None)), dtype=dtype)

        return self.spike_mat
        