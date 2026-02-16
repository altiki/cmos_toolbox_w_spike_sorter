import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils.logger_functions import console
import seaborn as sns
import numpy as np
from matplotlib.colors import hsv_to_rgb
import random


def check_color_map(colormap):
    """
    Check if a specified colormap exists in Matplotlib's colormaps.

    :param colormap: (str) The name of the colormap to check.
    :return colormap: (str) The specified colormap if it exists; otherwise, returns the default colormap 'seismic'.
    """
    # Check if colormap exists
    if colormap not in plt.colormaps():
        console.warning("Colormap does not exist in Matplotlib.")
        return "seismic"
    else:
        return colormap


class RainbowColourCoding:
    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    def __call__(self, electrodes):
        unique_electrodes = sorted(set(electrodes))
        num_colors = len(unique_electrodes)

        # Generate evenly spaced hue values
        hue_values = np.linspace(0, 1, num_colors)

        # Shuffle the list of hue values if shuffle is True
        if self.shuffle:
            np.random.shuffle(hue_values)

        # Convert each hue value to RGB color and store in dictionary
        color_mapping = {}
        for hue, electrode in zip(hue_values, unique_electrodes):
            color_mapping[electrode] = hsv_to_rgb([hue, 1, random.uniform(0.5, 1)])

        # Assign colors to all electrodes
        rgb = [color_mapping[electrode] for electrode in electrodes]

        # Convert to a numpy array
        rgb = np.array(rgb)

        return rgb


class LinearColourCoding:
    def __init__(self, width: int = 220, cmap: str = "plasma"):
        self.colourMap = cm.get_cmap(cmap)
        self.colourMap.set_bad('black')
        self.width = width

    def __call__(self, electrodes):
        electrodes = np.asarray(electrodes)

        coordsX = electrodes % self.width
        coordsY = electrodes // self.width
        boundX = [np.min(coordsX), np.max(coordsX)+1]
        boundY = [np.min(coordsY), np.max(coordsY)+1]
        yAxisBool = (boundY[1]-boundY[0]) - (boundX[1]-boundX[0]) > 0

        if yAxisBool:
            bound = boundY
            coords = coordsY - boundY[0]
        else:
            bound = boundX
            coords = coordsX - boundX[0]
        span = bound[1] - bound[0]
        phi = coords / span

        rgb = self.colourMap(phi)

        return rgb


def custom_ticks(x, pos):
    return f'{x / 20:.0f}'


def custom_lineplot(
        axes,
        data, x: str, y: str,
        color="black", hue="None", palette="tab10", style="None",
        plot_single=False):
    """
    Create a custom line plot on one or more Matplotlib axes.

    :param axes: (matplotlib.axes.Axes) A single Matplotlib axis or a list of axes to plot on.
    :param data: (pd.DataFrame) The DataFrame containing the data to be plotted.
    :param x: (str) The column name to be used for the x-axis.
    :param y: (str) The column name to be used for the y-axis.
    :param color: (str) The color of the line plot when hue is 'None'. Default is 'black'.
    :param hue: (str): The column name to be used for coloring the lines based on categories. Default is 'None'.
    :param palette: (str or dict): Color palette to use for coloring lines when hue is used. Default is 'tab10'.
    :param style: (str): The column name to be used for styling the lines based on categories. Default is 'None'.
    """

    if hue != "None":
        sns.lineplot(ax=axes,
                     data=data, x=x, y=y,
                     hue=hue, errorbar='sd', marker="o", palette=palette,
                     linewidth=1.5, markersize=6
                     )

    elif style != "None":
        if plot_single:
            sns.lineplot(ax=axes,
                         data=data, x=x, y=y,
                         style=style, hue=style, palette=palette, alpha=0.8,
                         markers=False, errorbar=None,
                         units="NW_ID", estimator=None,
                         linewidth=1.5, markersize=6, markeredgecolor=None
                         )
        else:
            sns.lineplot(ax=axes,
                         data=data, x=x, y=y,
                         style=style, hue=style, palette=palette, alpha=0.8,
                         markers=True, errorbar='sd',
                         linewidth=1.5, markersize=6, markeredgecolor=None
                         )

    else:
        sns.lineplot(ax=axes,
                     data=data, x=x, y=y,
                     color=color, errorbar='sd', marker="o",
                     linewidth=1.5, markersize=6
                     )


def custom_lineplot_with_polynomial(
        axes,
        data, x: str, y: str,
        style: str, palette: dict,
        degree=4):
    """
    Create a custom scatter plot with a polynomal line on one or more Matplotlib axes.

    :param axes: (matplotlib.axes.Axes) A single Matplotlib axis or a list of axes to plot on.
    :param data: (pd.DataFrame) The DataFrame containing the data to be plotted.
    :param x: (str) The column name to be used for the x-axis.
    :param y: (str) The column name to be used for the y-axis.
    :param style: (str): The column name to be used for styling the lines based on categories. Default is 'None'.
    :param palette: (str or dict): Color palette to use for coloring lines when hue is used. Default is 'tab10'.
    :param degree: (int) The polynomal degree of the model (default=8).
    """
    styles = [(0, (1, 1)), (0, (5, 1)), "solid", "dashdotted"]
    for idx, type in enumerate(data[style].unique()):
        subdata = data[data[style] == type]
        color = palette[type]

        # Scatter plot
        sns.scatterplot(data=subdata, x=x, y=y, hue=style, palette=[color], alpha=0.4, s=5, ax=axes)

        # Polynomial fitting
        # Make sure no NaNs are present as polyfit cannot handle them
        mask = ~np.isnan(subdata[x]) & ~np.isnan(subdata[y])
        model = np.poly1d(np.polyfit(subdata[x][mask], subdata[y][mask], degree))

        # Generate polyline
        polyline = np.linspace(min(subdata[x]), max(subdata[x]), 50)

        # Line plot
        sns.lineplot(x=polyline, y=model(polyline), color=color, ls=styles[idx], linewidth=2.5, ax=axes)
