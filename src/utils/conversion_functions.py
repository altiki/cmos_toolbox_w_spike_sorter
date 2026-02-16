import numpy as np


def convert_coordinates_to_electrode_number(x: float, y: float, chip_width=220):
    """
    Converts XY coordinates to electrode number.

    :param x: X-coordinate of electrode
    :param y: Y-Coordinate of electrode
    :param chip_width: Width of the CMOS chip (default 220)
    :return: Electrode number of the electrode at (x, y)
    """
    return y * chip_width + x % chip_width


def convert_electrode_number_to_coordinates(electrode_number: int, chip_width=220):
    """
    Converts electrode number to XY coordinates.

    :param electrode_number: Number of electrode to convert
    :param chip_width: Width of the CMOS chip (default 220)
    :return: Coordinates of the electrode
    """
    x = int(electrode_number / chip_width)
    y = electrode_number % chip_width

    return x, y


def convert_electrode_selection_to_voltage_map(electrodes: np.ndarray, base_value: int = 40, true_values=None):
    """
    Converts a list of electrode indices to a voltage map.

    :param true_values: If given, the voltage map is plotted using frequency values.
    :param electrodes: An array of electrode indices.
    :param base_value: The voltage value to assign to selected electrodes. Default is 40.
    :return: A 2D numpy array representing the voltage map with assigned values.
    """
    voltage_map = np.zeros((120, 220))
    coords_x = electrodes.astype(int) % voltage_map.shape[1]
    coords_y = electrodes.astype(int) // voltage_map.shape[1]

    if true_values is not None:
        base_value = np.max(true_values)/2
        voltage_map[coords_y, coords_x] = base_value + true_values

    else:
        voltage_map[coords_y, coords_x] = base_value

    return voltage_map



