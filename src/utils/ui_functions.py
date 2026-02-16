import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tkinter import Tk, Label, Button, Entry
from src.utils import conversion_functions

# Specify the backend
matplotlib.use("TkAgg")


class ElectrodeSelector:
    def __init__(self, path_to_electrode_selection: str):
        """
        A class for subsampling electrodes in an electrode array.

        :param path_to_electrode_selection: (str) The path to the electrode selection data.

        Attributes:
            selected_electrodes (list): A list to store selected electrodes.
            path_to_electrode_selection (str): The path to the electrode selection data file.
            electrodes (numpy.ndarray): An array of electrode data.
            coords (numpy.ndarray): An array of electrode coordinates.
            electrode_array (numpy.ndarray): A 2D array representing the electrode array.
            subselection_name (str): The name for the electrode subselection.
            root (Tkinter.Tk): The Tkinter root window.
            button (Tkinter.Button): The Tkinter button widget.
            entry (Tkinter.Entry): The Tkinter entry widget for user input.
            selection_completed (bool): A flag to track the completion of electrode selection.
        """

        self.selected_electrodes = []
        self.path_to_electrode_selection = path_to_electrode_selection

        # Load electrode selection
        self.electrodes = np.load(path_to_electrode_selection)
        self.coords = np.asarray([conversion_functions.convert_electrode_number_to_coordinates(i)
                                  for i in self.electrodes])

        # Initialize variables
        self.electrode_array = np.zeros((120, 220))
        self.subselection_name = None
        self.first_electrode = None
        self.root = None
        self.button = None
        self.entry = None
        self.selection_completed = False  # Flag to track selection completion

        # Get electrode array
        self.get_electrode_array()

    def get_electrode_array(self):
        """
        Populate the electrode_array with selected electrodes' coordinates.
        """
        for coord in self.coords:
            self.electrode_array[coord[0], coord[1]] = 1

    def user_define_channel_name(self):
        """
        Create a Tkinter GUI for the user to input a name for the electrode subselection.
        """
        # Create a Tkinter root window (main window)
        self.root = Tk()
        self.root.title("Subselection Naming:")

        # Create a label for the title
        title_label = Label(self.root, text="Enter a name for the subselection:")
        title_label.pack(pady=10)

        # Create an Entry widget for filename input
        self.entry = Entry(self.root)
        self.entry.pack(padx=20, pady=10)

        # Create a button to trigger the filename input
        self.button = Button(self.root, text="Enter", command=self.get_filename)
        self.button.pack(pady=10)

        # Start the Tkinter GUI event loop
        self.root.mainloop()

        # Check if a filename was entered and print it
        if self.subselection_name:
            print("Entered filename:", self.subselection_name)

    def user_select_electrode_area(self):
        """
        Create a GUI for the user to select an area by drawing a rectangle on the electrode plot.
        """
        # Create a figure and plot the data
        fig, ax = plt.subplots()
        ax.imshow(self.electrode_array)
        plt.title("Draw an area to select electrodes.")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # Create a RectangleSelector widget
        rs = RectangleSelector(ax, onselect=self.onselect, useblit=True, interactive=True)

        plt.show()

    def user_remove_electrodes(self):
        """
        Create a GUI for the user to remove electrodes by clicking on points.
        """
        # Create a figure and plot the data
        fig, ax = plt.subplots()
        ax.imshow(self.electrode_array)
        plt.title("Remove electrodes by clicking on points.")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        for x, y in self.selected_electrodes:
            plt.plot(y, x, 'go')  # Mark the selected point with a green dot
            plt.draw()

        # Connect the mouse click event to the select_area function
        fig.canvas.mpl_connect('button_press_event', self.remove_electrode)

        # Create a Tkinter GUI window
        self.root = Tk()
        self.root.title("Remove Electrodes.")

        # Add instructions and a confirm button to the GUI
        label = Label(self.root, text="Select points by clicking on the plot.")
        label.pack(pady=10)
        button_confirm = Button(self.root, text="Confirm Selection", command=self.confirm_selection)
        button_confirm.pack(pady=10)

        button_restart = Button(self.root, text="Restart Selection", command=self.restart_electrode_selection)
        button_restart.pack(pady=10)

        # Show the plot as a separate window
        plt.show()

    def user_select_first_electrode(self):
        """
        Create a GUI for the user to select the first electrode of a channel by clicking on a point.
        """

        # Create a figure and plot the data
        fig, ax = plt.subplots()
        ax.imshow(self.electrode_array)
        plt.title("Select first electrode of the selection.")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        for x, y in self.selected_electrodes:
            plt.plot(y, x, 'go')  # Mark the selected point with a green dot
            plt.draw()

        # Connect the mouse click event to the select_area function
        fig.canvas.mpl_connect('button_press_event', self.get_first_electrode)

        # Create a Tkinter GUI window
        self.root = Tk()
        self.root.title("Select First Electrodes.")

        # Add instructions and a confirm button to the GUI
        label = Label(self.root, text="Select first electrode by clicking on the plot.")
        label.pack(pady=10)
        button_confirm = Button(self.root, text="Confirm Selection", command=self.confirm_selection)
        button_confirm.pack(pady=10)

        button_restart = Button(self.root, text="Restart Selection", command=self.restart_electrode_selection)
        button_restart.pack(pady=10)

        # Show the plot as a separate window
        plt.show()

    def get_filename(self):
        """
        Get the filename entered by the user and close the GUI window and main loop.
        """
        # Get the text from the Entry widget
        self.subselection_name = self.entry.get()

        # Close GUI window and main loop
        self.root.destroy()
        self.root.quit()

    def onselect(self, eclick, erelease):
        """
        Handle the selection of electrodes within a rectangle.

        :param eclick: (matplotlib.backend_bases.PickEvent) The start point of the rectangle.
        :param erelease: (matplotlib.backend_bases.PickEvent) The end point of the rectangle.
        """
        y1, x1 = eclick.xdata, eclick.ydata
        y2, x2 = erelease.xdata, erelease.ydata

        # Select points within the rectangle
        self.selected_electrodes = [(x, y) for x, y in self.coords if x1 <= x <= x2 and y1 <= y <= y2]

        # Close the plot window
        plt.close()

        # Mark the electrode selection as completed
        self.selection_completed = True

    def remove_electrode(self, event):
        """
        Handle the removal of selected electrodes by clicking on points.

        :param event: (matplotlib.backend_bases.PickEvent) The mouse click event.
        """
        # Extract all x- and y-coordinates of the selected points
        x_points, y_points = self.get_selected_points_coordinates()

        # Remove electrode when the user clicks on it
        if event.inaxes is not None:
            y, x = event.xdata, event.ydata

            # Find the electrode closest to the mouse click
            closest_x = min(x_points, key=lambda x_point: abs(x_point - x))
            closest_y = min(y_points, key=lambda y_point: abs(y_point - y))

            # Remove the electrode from the list
            try:
                self.selected_electrodes.remove((closest_x, closest_y))
            except ValueError:
                print("Electrode could not be found in list.")
            plt.plot(closest_y, closest_x, 'ro')  # Mark the removed point with a red dot
            plt.draw()

    def get_first_electrode(self, event):
        """
        Handle the selection of the first electrodes by clicking on points.

        :param event: (matplotlib.backend_bases.PickEvent) The mouse click event.
        """
        # Extract all x- and y-coordinates of the selected points
        x_points, y_points = self.get_selected_points_coordinates()

        # Remove electrode when the user clicks on it
        if event.inaxes is not None:
            y, x = event.xdata, event.ydata

            # Find the electrode closest to the mouse click
            closest_x = min(x_points, key=lambda x_point: abs(x_point - x))
            closest_y = min(y_points, key=lambda y_point: abs(y_point - y))

            # Remove the electrode from the list
            try:
                self.first_electrode = conversion_functions.convert_coordinates_to_electrode_number(closest_x, closest_y)
            except ValueError:
                print("Electrode could not be found in list.")
            plt.plot(closest_y, closest_x, 'bo')  # Mark the removed point with a blue dot
            plt.draw()

    def get_selected_points_coordinates(self):
        """
        Extracts all x- and y-coordinates of the selected points.

        :return: Two lists x_points and y_points with the coordinates.
        """
        x_points = []
        y_points = []
        for x, y in self.selected_electrodes:
            x_points.append(x)
            y_points.append(y)

        return x_points, y_points

    def confirm_selection(self):
        """
        Close the plot window and the GUI window along with its main loop.
        """
        # Close the plot window
        plt.close()

        # Close GUI window and main loop
        self.root.destroy()
        self.root.quit()

    def restart_electrode_selection(self):
        """
        Close the plot window and the GUI window and restart the electrode selection.
        """
        # Close everything
        plt.close()
        self.root.destroy()
        self.root.quit()

        # Restart
        self.run_electrode_subselection()

    def run_electrode_subselection(self):
        """
        Run the electrode subselection process, including naming, selecting, and removing electrodes.

        :return: (list) A list of selected electrodes' coordinates.
        """
        self.user_define_channel_name()
        while self.selection_completed is not True:
            self.user_select_electrode_area()
        self.user_remove_electrodes()
        self.user_select_first_electrode()

        return [conversion_functions.convert_coordinates_to_electrode_number(x, y) for x, y in self.selected_electrodes]

