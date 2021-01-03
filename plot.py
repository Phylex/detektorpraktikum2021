""" Module responsible for the plotting of the hitmaps and histograms """
import matplotlib as mpl
from matplotlib import cm
import pixel as px


def plot_sensor_hit_histogram(axes: mpl.axes.Axes, sensor: px.Sensor)\
        -> mpl.axes.Axes:
    """ plot the hitmap of the sensor into a given axes

    plots the hitmap of the sensor into a provided axes instance and
    annotates the plot accordingly
    """
    cmap = cm.get_cmap('viridis', 1024)
    hist, x_edges, y_edges = sensor.hitmap_histogram(density=True)
    axes.set_title("Sensor Hitmap")
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    axes.pcolormesh(x_edges, y_edges, hist, cmap=cmap)
