""" Module responsible for the plotting of the hitmaps and histograms """
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pixel as px
import fitting as ft


def plot_sensor_hit_histogram(axes: mpl.axes.Axes, hist: np.ndarray,
                              x_edges: np.ndarray, y_edges: np.ndarray):
    """ plot the hitmap of the sensor into a given axes

    plots the hitmap of the sensor into a provided axes instance and
    annotates the plot accordingly
    """
    cmap = cm.get_cmap('viridis', 1024)
    axes.set_title("Sensor Hitmap")
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    axes.pcolormesh(x_edges, y_edges, hist, cmap=cmap)


def plot_2d_gauss_fit(axes: mpl.axes.Axes, popt_2d, x_centers: np.ndarray,
                      y_centers: np.ndarray):
    xx, yy = np.meshgrid(x_centers, y_centers)
    axes.contour(xx, yy, ft.gauss_2D(*popt_2d)(xx, yy, cmap='coolwarm'))
    axes.scatter(popt_2d[0], popt_2d[1], color='darkred')


def plot_hitmap_with_projections(hist_2d: np.ndarray, x_edges: np.ndarray,
                                 y_edges, x_hist, y_hist, x_popt, y_popt,
                                 popt_2d, figpath):
    """ generate a plot that shows the 2D view of the sensor with matching hists

    Generate a plot that shows the top down 2D view of the sensor with a
    histogram for the sum of the hits in every row and column. Show the fits
    on these histograms that determine the position of the peak. Draw
    horizontal and vertical lines at the peak so that the position can be
    marked.
    """
    # determine the aspect ratio of the sensor so that we can scale the
    # rest accordingly
    sensor_aspect_ratio = (x_edges[-1] - x_edges[0])/(y_edges[-1] - y_edges[0])

    # swap the variables so that we get an upright plot
    # this is better for printing on paper
    if sensor_aspect_ratio <= 1:
        x_edges, y_edges = (y_edges, x_edges)
        x_hist, y_hist = (y_hist, x_hist)
        x_popt, y_popt = (y_popt, x_popt)
        popt_2d = popt_2d[::-1]
        hitmap_hist = hitmap_hist.T

    x_centers = (x_edges[1:] + x_edges[:-1])/2
    y_centers = (y_edges[1:] + y_edges[:-1])/2

    # construct the shape of the axes
    fig_width = 4
    fig_height = fig_width / (2 * sensor_aspect_ratio)

    spacing = 0.005
    left, tdh_width = 0.1, 0.50
    bottom, tdh_height = 0.1, 0.7
    x_hist_h = 1 - (bottom + 2 * spacing + tdh_height)
    y_hist_w = 1 - (left + tdh_width + 2 * spacing)

    # construct the rectangles for the plots
    rect_2D_hist = [left, bottom, tdh_width, tdh_height]
    rect_x_hist = [left, bottom + tdh_height + spacing, tdh_width, x_hist_h]
    rect_y_hist = [left + tdh_width + spacing, bottom, y_hist_w, tdh_height]

    # create figures and the axes
    fig = plt.figure(1, figsize=(fig_width, fig_height))
    ax_2dh = plt.axes(rect_2D_hist)
    ax_hx = plt.axes(rect_x_hist)
    ax_hy = plt.axes(rect_y_hist)

    # hide the unneeded tick labels
    ax_2dh.tick_params(direction='in', top=True, right=True)
    ax_hx.tick_params(direction='in', labelbottom=False)
    ax_hy.tick_params(direction='in', labelleft=False)

    # plot the 2D histogram
    plot_sensor_hit_histogram(ax_2dh, hist_2d, x_edges, y_edges)

    # plot the rings from the 2D fit
    ax_2dh.contour(hitmap_coordinates[0].T,
                   hitmap_coordinates[1].T,
                   gauss_2D(*popt_2d)(hitmap_coordinates[0].T,
                                      hitmap_coordinates[1].T),
                   cmap='coolwarm')
    ax_2dh.scatter(popt_2d[0], popt_2d[1], color='darkred')

    # plot the x_axis histogram
    ax_hx.hist(x_bins[:-1], x_bins, weights=projections[0], )
    ax_hx.plot(x_centers, norm.pdf(x_centers, *x_popt))
    # plot vertical  line in 2D hist
    ax_2dh.vlines([x_popt[0]], 0, 1, color='orange')

    # plot the y_axis histogram
    ax_hy.set_ylim([y_bins[0], y_bins[-1]])
    ax_hy.hist(y_bins[:-1], y_bins, weights=projections[1], density=True,
               orientation='horizontal')
    ax_hy.plot(norm.pdf(y_centers, *y_popt), y_centers)
    # plot the horizontal line from the independent fit
    ax_2dh.hlines([y_popt[0]], 0, 1, color='orange')
    plt.savefig(figpath+'_test.jpg')
    plt.close(fig)
