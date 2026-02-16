import time
import os
import re
import datetime
import logging
import random
import pickle
import numpy as np
from abc import ABC, abstractmethod
from src.utils import stimulusLoopFunctions
from src.utils.logger_functions import console

# Importing MaxLab libraries
import maxlab
import maxlab.system
import maxlab.util
import maxlab.saving


class ABCRecorder(ABC):
    @abstractmethod
    def __init__(self,
                 electrode_selections_path: str,
                 output_path: str,
                 chip_id: int,
                 suffix: str,
                 div: float,
                 network_id=None,
                 electrodes=None,
                 record_only_spikes=False) -> None:
        """
        Initialize the Recorder.

        Args:
            electrode_selections_path (str): Path to electrode selections.
            output_path (str): Path to store recordings.
            chip_id (int): Chip ID.
            suffix (str): Suffix for output files (e.g. "control", "cancer", ...).
            div (int): Days in vitro of the culture (DIV0 corresponds to the day of cell thawing)
            network_id (Optional): Network ID.
            electrodes (Optional): Electrodes to record from.
            record_only_spikes (bool): Whether to record only spikes.
        """

        self.output_path = output_path
        self.electrode_selections_path = electrode_selections_path
        self.chip_id = chip_id
        self.div = div
        self.suffix = suffix
        self.network_id = network_id
        self.electrodes = electrodes
        self.record_only_spikes = record_only_spikes

        self.electrode_selections = []
        self.time_format = None
        self.date_format = None
        self.config_file = None

        self.create_log_and_config_files()
        self.saver = self.initialize_saver()

    def find_electrode_selections(self) -> list:
        """
        Searches through the electrode selections folder and stores all the electrode selections into a list.
        """
        # Load the electrode selections
        files = os.listdir(self.electrode_selections_path)

        for file in files:
            if str(self.chip_id) in file:
                self.electrode_selections.append(os.path.join(self.electrode_selections_path, file))

        if len(self.electrode_selections) == 0:
            logging.error(f"No electrode selections found for Chip ID {self.chip_id}.")
        else:
            logging.info(f"Found electrode selections for Chip ID {self.chip_id}. "
                         f"Total number: {len(self.electrode_selections)}")

        return self.electrode_selections

    @staticmethod
    def initialize_maxlab():
        """
        Connect to the maxlab system and initialize Scope.
        """
        try:
            # Initialize MaxLab system into a defined state
            maxlab.util.initialize()
            # Correct the zero-mean offset in each electrode reading
            maxlab.util.offset()

            # Power on MaxLab system
            status = maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
            logging.info("Power ON status: " + status)

        except TypeError as e:
            console.error(f"Could not initialize maxlab. Make sure Scope is up and running. \n Error: {e}.")

    def create_log_and_config_files(self):
        """
        Creates a log and a config file and stores them in the corresponding folders.
        """
        # Creating relevant filepaths and folders
        config_folder = os.path.join(os.getcwd(), 'config/')
        logs_folder = os.path.join(os.getcwd(), 'logs/')

        try:
            os.makedirs(config_folder)
            os.makedirs(logs_folder)
        except:
            pass

        # Setting the time and date for the configuration and log files
        time_now = datetime.datetime.now()
        self.time_format = time_now.strftime("%H%M")
        self.date_format = time_now.strftime("%Y%m%d")

        # Setting the log and config filenames
        config_filename = 'config_' + str(self.date_format) + '_Chip' + str(self.chip_id)
        self.config_file = config_folder + config_filename + '.cfg'

        log_filename = 'Log_' + str(self.date_format) + '_' + str(self.time_format) + '_Chip' + str(
            self.chip_id)
        log_file = logs_folder + log_filename + '.log'
        logging.basicConfig(filename=log_file, level=logging.INFO)

    def load_electrodes(self, electrodes, rec_type='array', stimulus_electrodes=None):
        """
        Loads the electrode selection into Scope.

        :param electrodes: List of electrodes to load.
        :param rec_type: Token ID to identify the instance on the server. 'array' is used for spontaneous recordings,
        'stimulation' is used for stimulation recordings.
        :param stimulus_electrodes: Electrodes used for stimulation, if any.
        """
        array = maxlab.chip.Array(rec_type)
        array.reset()
        array.clear_selected_electrodes()

        # Selecting the electrodes to be routed
        array.select_electrodes(electrodes)
        if stimulus_electrodes is not None:
            array.select_stimulation_electrodes(stimulus_electrodes)
        array.route()

        # Download the prepared array configuration to the chip
        array.download()

        # Save the configuration files
        array.save_config(self.config_file)

        logging.info("Successfully loaded the electrode selection.")

        return array

    def initialize_saver(self):
        """
        Initializes maxlab saver and returns it.
        """
        saver = maxlab.saving.Saving()
        saver.open_directory(self.output_path)
        saver.set_legacy_format(True)
        saver.group_delete_all()

        return saver

    @staticmethod
    def wait(duration):
        """
        Waits for a given duration.
        """
        if duration <= 0:
            return
        time.sleep(duration)


class SpontaneousActivityRecorder(ABCRecorder):
    def __init__(self,
                 electrode_selections_path: str,
                 output_path: str,
                 chip_id: int,
                 suffix: str,
                 div: float,
                 network_id=None,
                 electrodes=None,
                 record_only_spikes=False) -> None:
        """
        Initialize the SpontaneousActivityRecorder.
        """
        super().__init__(electrode_selections_path=electrode_selections_path,
                         output_path=output_path,
                         chip_id=chip_id,
                         suffix=suffix,
                         div=div,
                         network_id=network_id,
                         electrodes=electrodes,
                         record_only_spikes=record_only_spikes
                         )

    def spontaneous_recording(self, duration):
        """
        Records the spontaneous activity from the inserted chip.

        :param duration: Recording duration in s.
        """
        # Set filename
        filename = (f'ID{self.chip_id}_{self.network_id}_DIV{self.div}_DATE{self.date_format}_{self.time_format}_'
                    f'spontaneous_{self.suffix}')

        # Load electrode selections
        if self.electrodes is None:
            logging.warning(f"The electrode list to load is not given. Please provide an electrode list.")
        else:
            self.load_electrodes(self.electrodes)

        # Start recording
        self.saver.start_file(filename)
        logging.info(f"Starting recording of network {self.network_id} on chip {self.chip_id}...")

        # Input how many wells you want to record from (range(1) for MaxOne, range(6) for MaxTwo)
        wells = range(1)
        if not self.record_only_spikes:
            for well in wells:
                self.saver.group_define(well, "routed")
        self.saver.start_recording(wells)

        # Wait the predefined duration
        self.wait(duration=duration)

        # Stop recording data
        self.saver.stop_recording()
        self.saver.stop_file()
        self.saver.group_delete_all()

        logging.info(f"Recording of network {self.network_id} on chip {self.chip_id} completed.")

    def spontaneous_recording_from_multiple_networks(self, duration):
        """
        Iteratively records the spontaneous activity from multiple networks on a chip, based on the electrode selections
        found in self.electrode_selections_path.

        :param duration: Recording duration in s.
        """
        self.find_electrode_selections()

        # Run through the networks
        for selection_path in self.electrode_selections:
            # Extract network ID
            self.network_id = selection_path.split('.')[-2][-2:]

            # Load electrode selection into Scope
            self.electrodes = np.load(selection_path)

            # Record the spiking data
            self.spontaneous_recording(duration)

        logging.info("\n Done!")


class StimulationActivityRecorder(ABCRecorder):
    def __init__(self,
                 electrode_selections_path: str,
                 output_path: str,
                 chip_id: int,
                 suffix: str,
                 div: float,
                 stim_voltage: int,
                 stimulation_duration: int,
                 stimulation_repetitions: int,
                 pause_duration: int,
                 spontaneous_rec_duration: int,
                 pulse_frequency: int,
                 inter_electrode_delay: float,
                 electrode_dict: dict,
                 sampling_time=0.05,
                 pulse_duration=0.4,
                 network_id=None,
                 record_only_spikes=False) -> None:
        """
         Initialize the StimulationActivityRecorder.

         :param electrode_selections_path: (str) Path to the electrode selections.
         :param output_path: (str) Path to save the output files.
         :param chip_id: (int) ID of the chip.
         :param suffix: (str) Suffix for the output file names.
         :param div: (float) Division factor.
         :param stim_voltage: (int) Stimulation voltage.
         :param stimulation_duration: (int) Duration of stimulation episode in seconds.
         :param stimulation_repetitions: (int) Number of stimulation repetitions.
         :param pause_duration: (int) Duration of pause between stimulation episodes in seconds.
         :param spontaneous_rec_duration: (int) Duration of spontaneous recording in seconds before and after stimulating.
         :param pulse_frequency: (int) Frequency of stimulation pulses applied during stimulation episodes in Hz.
         :param inter_electrode_delay: (float) Delay in milliseconds (ms) between two paired electrodes.
         :param electrode_dict: (dict) Dictionary containing recording and stimulation electrodes.
         :param sampling_time: (float) Time step for one time point in milliseconds (ms) (corresponds to sampling rate).
         :param pulse_duration: (float) Duration in milliseconds (ms) for both phases of biphasic pulse.
         :param network_id: (Optional[str]) ID of the network.
         :param record_only_spikes: (bool) Flag indicating whether to record only spikes.
         """
        super().__init__(electrode_selections_path=electrode_selections_path,
                         output_path=output_path,
                         chip_id=chip_id,
                         suffix=suffix,
                         div=div,
                         network_id=network_id,
                         record_only_spikes=record_only_spikes
                         )

        try:
            self.recording_electrodes = electrode_dict["recording_electrodes"]
            self.stimulus_electrodes = electrode_dict["stimulus_electrodes"]
        except KeyError:
            console.warning("No valid electrode dictionary was provided.")
            self.recording_electrodes = []
            self.stimulus_electrodes = []

        # Set stimulation parameters
        self.stim_voltage = stim_voltage
        self.pulse_duration = pulse_duration
        self.sampling_time = sampling_time
        self.inter_electrode_delay = inter_electrode_delay

        self.stimulation_duration = stimulation_duration  # s
        self.pause_duration = pause_duration  # s
        self.spontaneous_rec_duration = spontaneous_rec_duration  # s
        self.stimulation_repetitions = stimulation_repetitions
        self.pulse_frequency = pulse_frequency  # Hz
        self.pulse_delay_time = 1000 / self.pulse_frequency  # Time between pulses from start of the pulse trains (ms)

        self.repetition = int(np.ceil(self.stimulation_duration + self.pause_duration))  # Total time of stimulation episode (s)
        self.total_time_experiment = self.repetition*self.stimulation_repetitions  # Total time of experiment (s)
        self.interpulse_interval = self.pulse_delay_time - self.pulse_duration  # Time between pulse trains (ms)
        self.sample_amount = int(self.interpulse_interval / self.sampling_time)  # No. of frames sampled in-between pulse trains
        self.inter_electrode_sample_amount = int(self.inter_electrode_delay / self.sampling_time)  # No. of frames between delayed stimulation units
        self.total_pulses = int(self.pulse_frequency * self.stimulation_duration)  # No. of pulse trains in one stimulation episode

        # If there is a delay between the stimulation of paired electrodes, redefine the variables
        if self.inter_electrode_delay != 0.0:
            self.first_stimulus_electrodes = self.stimulus_electrodes[::2]
            self.second_stimulus_electrodes = self.stimulus_electrodes[1::2]
            self.stimulus_electrodes = self.first_stimulus_electrodes + self.second_stimulus_electrodes
        else:
            self.first_stimulus_electrodes = []
            self.second_stimulus_electrodes = []

        # Initialize other parameters
        self.electrode_selections = []
        self.time_format = None
        self.date_format = None
        self.config_file = None

        # Initialize files and maxlab
        self.create_log_and_config_files()
        self.saver = self.initialize_saver()
        self.initialize_maxlab()

    def get_routed_electrodes(self, array) -> list:
        """
        Retrieves the list of electrodes that can be stimulated based on whether they have amplifier connectivity.

        :param array: The array object.
        Returns: List of electrodes that can be stimulated.
        """
        routed = []

        for electrode in self.stimulus_electrodes:
            if array.query_amplifier_at_electrode(electrode):
                routed.append(electrode)

        # Randomising the order of stimulation electrodes
        random.shuffle(routed)

        logging.info('Number of stimulation electrodes NOT routed: ' + str(len(self.stimulus_electrodes) - len(routed)))

        return routed

    def create_stimulation_sequence(self):
        """
        Creates a stimulation sequence based on the provided stimulation parameters.

        Returns: (tuple) A tuple containing the created sequence and voltage bit.
        """
        logging.info("Creating the stimulation sequence...")
        voltage_bit = stimulusLoopFunctions.convertVoltsToBits(self.stim_voltage)

        # Creating the sequence
        sequence = maxlab.Sequence()

        for rep in range(1, self.total_pulses):
            sequence = self.append_stimulation_pulse(sequence=sequence,
                                                     amplitude=voltage_bit,
                                                     pulse_duration=self.pulse_duration,
                                                     dac=0,
                                                     )

            if self.inter_electrode_delay != 0:
                # Wait for the delay between the two stimulation units
                sequence.append(maxlab.system.DelaySamples(self.inter_electrode_sample_amount))

                sequence = self.append_stimulation_pulse(sequence=sequence,
                                                         amplitude=voltage_bit,
                                                         pulse_duration=self.pulse_duration,
                                                         dac=1,
                                                         )

                # Wait to start the first pulse again
                sequence.append(maxlab.system.DelaySamples(self.sample_amount-self.inter_electrode_sample_amount))

            else:
                # Wait between two pulses
                sequence.append(maxlab.system.DelaySamples(self.sample_amount))

        logging.info("Sequence created for " + str(voltage_bit) + " mV.\n")

        return sequence, voltage_bit

    def append_stimulation_pulse(self, sequence, amplitude, pulse_duration, dac=0):
        """
        Appends a biphasic pulse to the stimulation protocol.

        :param sequence: Stimulation sequence to attend the pulse to.
        :param amplitude: Amplitude of the stimulation pulse.
        :param pulse_duration: Duration of both phases of the pulse.
        :param dac: DAC to append the stimulus to.
        """
        # Calculate phase duration in sample numbers
        sample = int(pulse_duration / self.sampling_time / 2)
        voltage_baseline = 512  # corresponds to 1.6 V

        # Append a biphasic pulse (first negative, then positive phase)
        sequence.append(maxlab.chip.DAC(dac, voltage_baseline - amplitude))
        sequence.append(maxlab.system.DelaySamples(sample))
        sequence.append(maxlab.chip.DAC(dac, voltage_baseline + amplitude))
        sequence.append(maxlab.system.DelaySamples(sample))
        sequence.append(maxlab.chip.DAC(dac, voltage_baseline))

        return sequence

    def connect_electrode_for_stimulation(self, array, routed_stim_electrodes):
        """
        Connects the specified electrode for stimulation.

        :param array: The array object.
        :param routed_stim_electrodes: (list) List of electrodes to be stimulated.
        Returns: (list) List of stimulation units.
        """
        # Load the configuration saved above
        array.load_config(self.config_file)

        if self.inter_electrode_delay == 0:
            stimulation_units = self.get_stimulation_units(routed_stim_electrodes, array)
            return stimulation_units

        else:
            stimulation_units_one = self.get_stimulation_units(self.first_stimulus_electrodes, array)
            stimulation_units_two = self.get_stimulation_units(self.second_stimulus_electrodes, array)

            if not set(stimulation_units_one).isdisjoint(set(stimulation_units_two)):
                console.warning(f"Routing Problem, stim one {stimulation_units_one} "
                                f"and stim two {stimulation_units_two}")

            return stimulation_units_one, stimulation_units_two

    @staticmethod
    def get_stimulation_units(electrode_list, array):
        """
        Connects the stimulation electrodes to the stimulation and returns the stimulation units.
        """
        stimulation_units = []
        # Connect the electrode to stimulation
        for stim_elec in electrode_list:
            array.connect_electrode_to_stimulation(stim_elec)
            # Check if a stimulation channel can be connected to the electrode
            stimulation_units.append(array.query_stimulation_at_electrode(stim_elec))

        # Repeated download of configuration to the chip executed each time to prevent unusual chip behaviour
        array.download()

        return stimulation_units

    @staticmethod
    def turn_maxlab_on_or_off(stimulation_units, turn_on, dac=0):
        """
        Turns the MaxLab stimulation unit on or off.

        :param stimulation_units: (list) The list of stimulation units.
        :param turn_on: (bool) A boolean indicating whether to turn on (True) or off (False) the stimulation.
        :param dac: (int) DAC the stimulation units should be mapped to.
        """
        # Power ON or OFF the stimulation
        for stimulation_unit in stimulation_units:
            stim = maxlab.chip.StimulationUnit(stimulation_unit).power_up(turn_on).connect(turn_on)

            if turn_on:
                maxlab.send(stim.set_voltage_mode().dac_source(dac))

            else:
                maxlab.send(stim)

        if not turn_on:
            maxlab.send(maxlab.system.DelaySamples(100))

    def apply_voltage_at_electrodes(self, array, sequence, routed_stim_electrodes):
        """
        Applies stimulation voltage at the specified electrode using the given sequence.

        :param array: The array object.
        :param sequence: The stimulation sequence.
        :param routed_stim_electrodes: (list) List of electrodes to be stimulated.
        """

        try:
            # Input how many wells you want to record from (range(1) for MaxOne, range(6) for MaxTwo)
            wells = range(1)
            if not self.record_only_spikes:
                for well in wells:
                    self.saver.group_define(well, "routed")

            # Get stimulation units and power OFF and ON the stimulation
            if self.inter_electrode_delay == 0:
                stimulation_units = self.connect_electrode_for_stimulation(array, routed_stim_electrodes)
                self.turn_maxlab_on_or_off(stimulation_units, turn_on=False)
                self.turn_maxlab_on_or_off(stimulation_units, turn_on=True)

            else:
                stimulation_units_one, stimulation_units_two = self.connect_electrode_for_stimulation(
                    array, routed_stim_electrodes)
                self.turn_maxlab_on_or_off(stimulation_units_one, turn_on=False,dac=0)
                self.turn_maxlab_on_or_off(stimulation_units_two, turn_on=False, dac=1)
                self.turn_maxlab_on_or_off(stimulation_units_one, turn_on=True, dac=0)
                self.turn_maxlab_on_or_off(stimulation_units_two, turn_on=True, dac=1)

            logging.info(f"Stimulation Unit connected to electrodes {routed_stim_electrodes}.")

            # Set filename
            filename = (f'ID{self.chip_id}_N{self.network_id}_DIV{self.div}_DATE{self.date_format}_{self.time_format}_'
                        f'stimulation_{self.suffix}_V{self.stim_voltage}_Freq{self.pulse_frequency}_'
                        f'El{self.stimulus_electrodes[0]}')

            # Start recording
            self.saver.start_file(filename)
            self.saver.start_recording(wells)

            # Run for however many seconds possible in sequence --> send stimulation pulse for x min
            for rep in range(self.stimulation_repetitions):
                console.info(f"Starting pause of {self.pause_duration} seconds for repetition {rep}.")
                time.sleep(int(self.pause_duration))
                console.info(f"Starting stimulation of {self.stimulation_duration} seconds.")
                sequence.send()
                time.sleep(self.pulse_delay_time * self.total_pulses / 1000)
                logging.info(f"Sequence of repetition {rep} sent successfully.")

            # Run a spontaneous recording after the stimulation if specified
            if self.spontaneous_rec_duration != 0:
                logging.info(f"Continue to record spontaneous activity for {self.spontaneous_rec_duration} s...")
                time.sleep(self.spontaneous_rec_duration)
                logging.info(f"... done with spontaneous recording.")

            # Stop recording data
            self.saver.stop_recording()
            self.saver.stop_file()
            self.saver.group_delete_all()

            # Turning OFF the stimulation
            if self.inter_electrode_delay == 0:
                self.turn_maxlab_on_or_off(stimulation_units, turn_on=False)
            else:
                self.turn_maxlab_on_or_off(stimulation_units_one, turn_on=False, dac=0)
                self.turn_maxlab_on_or_off(stimulation_units_two, turn_on=False, dac=0)

            logging.info("Powered OFF stimulation units. \n")

            # Disconnect the electrode from stimulation
            for stim_el in routed_stim_electrodes:
                array.disconnect_electrode_from_stimulation(stim_el)

        except:
            logging.warning("\tNo stimulation channel can connect to electrodes: " + str(routed_stim_electrodes) + "\n")

    def stimulation_recording(self):
        """
        Initiates stimulation recording process for the specified electrodes and stimulation parameters
        at a given network.
        """
        # Load electrode selections
        if self.recording_electrodes is None:
            logging.warning(f"The electrode list to load is not given. Please provide an electrode list.")
        else:
            array = self.load_electrodes(electrodes=self.recording_electrodes, rec_type="stimulation",
                                         stimulus_electrodes=self.stimulus_electrodes)

            routed = self.get_routed_electrodes(array=array)

            # Get stimulation sequence
            sequence, voltage_bit = self.create_stimulation_sequence()

            # Looping through all routed electrodes and apply a sequence of pulses
            logging.info(f'Stimulating at electrodes {routed}.')
            self.apply_voltage_at_electrodes(array=array, sequence=sequence, routed_stim_electrodes=routed)

            logging.info(f"Recording of network {self.network_id} on chip {self.chip_id} completed.")

    def stimulation_recording_from_multiple_networks(self, no_of_channels):
        """
        Iteratively stimulates and records activity from multiple networks on a chip, based on the electrode selections
        found in self.electrode_selections_path.

        :param no_of_channels: Number of channels to record from (ordered by activity).
        """
        self.find_electrode_selections()

        # Iterate through the no_of_channels most active electrodes
        for i in range(0, no_of_channels):
            # Run through the networks
            for selection_path in self.electrode_selections:
                # Extract network ID
                pattern = r'N(\d+)'
                match = re.search(pattern, selection_path)
                self.network_id = int(match.group(1))

                with open(selection_path, 'rb') as f:
                    electrode_dict = pickle.load(f)
                    self.recording_electrodes = electrode_dict["Subselection_Cancer"]["recording_electrodes"]
                    self.stimulus_electrodes = [electrode_dict["Subselection_Cancer"]["stimulus_electrodes"][i]]

                # Record the spiking data
                console.info(f"Stimulating and recording from Network {self.network_id} at electrode "
                             f"{self.stimulus_electrodes}.")
                self.stimulation_recording()

        logging.info("\n Done!")