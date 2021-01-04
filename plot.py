""" Module responsible for the plotting of the hitmaps and histograms """
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
# import pixel as px
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
    """ plot the contours of the Gaussian fit on the hitmap """
    xx, yy = np.meshgrid(x_centers, y_centers)
    axes.contour(xx, yy, ft.gauss_2d(*popt_2d)(xx, yy), cmap='coolwarm')
    axes.scatter(popt_2d[0], popt_2d[1], color='darkred')


def plot_histogram(axes: mpl.axes.Axes, hist, popt, bin_edges: np.ndarray,
                   axis: str):
    """ plots the histogram of one of the axis projections

    Plot the projection of histogram of hits. The plot is plotted vertically
    if the axis label is given as 'y'

    Parameters
    ----------
    axes : mpl.axes.Axes
        the axes to plot the histogram into. The plot will be vertical if the
        axis is specified to be 'y'
    hist : np.ndarray
        the histogramd hits
    popt : np.ndarray
        the parameters for the Gaussian that to be plotted over the hist
    bin_edges : np.ndarray
        the edges of the bins used for the histogram
    axis : str
        the axis on for which the results should be plotted (either 'x'
        or 'y'). Plots vertically if 'y' is specified.
    """
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    if axis == 'x':
        axes.set_xlim([bin_edges[0], bin_edges[-1]])
        axes.hist(bin_edges[:-1], bin_edges, weights=hist, density=True)
        axes.plot(bin_centers, norm.pdf(bin_centers, *popt))
    elif axis == 'y':
        axes.set_ylim([bin_edges[0], bin_edges[-1]])
        axes.hist(bin_edges[:-1], bin_edges, weights=hist, density=True,
                  orientation='horizontal')
        axes.plot(norm.pdf(bin_centers, *popt), bin_centers)
    else:
        raise ValueError("axis has to be either 'x' or 'y'")


def plot_crosshairs(axes: mpl.axes.Axes, x_popt, y_popt, color: str = 'gray'):
    """ plot the cross-hairs on the hitmap

    Plot the cross-hairs that are the results from the fit of the projections
    """
    axes.vlines([x_popt[0]], 0, 1, color=color)
    axes.hlines([y_popt[0]], 0, 1, color=color)


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
        hist_2d = hist_2d.T

    x_centers = (x_edges[1:] + x_edges[:-1])/2
    y_centers = (y_edges[1:] + y_edges[:-1])/2

    # construct the shape of the axes
    fig_width = 10
    fig_height = 2 * fig_width / sensor_aspect_ratio

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

    # plot the rings from the 2D fit and the cross-hairs
    plot_2d_gauss_fit(ax_2dh, popt_2d, x_centers, y_centers)
    plot_crosshairs(ax_2dh, x_popt, y_popt)

    # plot the x_axis histogram
    plot_histogram(ax_hx, x_hist, x_popt, x_edges, 'x')

    # plot the y_axis histogram
    plot_histogram(ax_hy, y_hist, y_popt, y_edges, 'y')

    # save the finished plot
    plt.savefig(figpath+'_test.jpg')
    plt.close(fig)


def plot_coord_transform(top_sns_points, bottom_sns_points, tp):
    """
    Plot the transformed coordinates alongside the untransformed
    coordinates
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.5))
    ax.scatter([top_sns_points[0][0], top_sns_points[1][0]],
               [top_sns_points[0][1], top_sns_points[1][1]], color='blue',
               label='positions of the signal peaks on the Top sensor')
    ax.scatter([bottom_sns_points[0][0], bottom_sns_points[1][0]],
               [bottom_sns_points[0][1], bottom_sns_points[1][1]],
               color='gold', label='original positions of peaks on bottom\
                                    sensor')
    ax.scatter([tp[0, 0], tp[1, 0]], [tp[0, 1], tp[1, 1]],
               color='orange',
               label='Transformed points of the bottom sensor')
    plt.grid()
    plt.legend()
    plt.savefig('coordinate_transformation_fit.svg')
    plt.close(fig)
