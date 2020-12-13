#!/usr/bin/python3
"""Script for analysing the data form the pixel telescope for the Detector
physics course 2021

This module performs the analysis of the data and produces the plots used
in the presentation The steps of the analysis are as follows:
1) figure out the alignment of the sensors relative to each other.
2) figure out the latency of the trigger system to retrieve the right data
from the sensor internal buffer.
3) use alignment information to correct the hits in the muon detection runs
4) calculate and histogram the angular distribution of the muons detected by
the sensor
"""
import uproot as ur
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import norm

LS_DIR = 'LatencyScan'
LScan_range_top = [format(elem, '03d') for elem in range(0, 157)]
LScan_range_btm = [format(elem, '03d') for elem in range(70, 94)]

ALIGNMENT_DIR = 'Alignment-Sr90'
Align_files = ['Alignment{}-Sr90_oKoll_3600.root'.format(string)
               for string in ['', '2']]

MUON_DIR = 'Muon-Runs'
mrun_files = ['muon-run{}.root'.format(i) for i in range(1, 3)]

SNSR_DIRS = ['M4587-Top', 'M4520-Bottom']


def get_alignment_data(alignment_fpath):
    """get the data for the alignment out of the root file

    The data for the alignment is inside a set of root files
    (one per sensor and measurement) extract the name and the data for
    every chip of a sensor and measurement form the root file and return
    the data

    Args:
        alignment_fpath (string): the name of the root file to extract
        the data form

    Returns:
        sensor_data (list of np.array): the data for every chip on the sensor
        for the measurement contained in the root file
        sensor_names (list of strings): the names of the chips on the sensor
        index matched to the data in `sensor_data`
    """
    sensor_names = ['C{}'.format(i) for i in range(0, 16)]
    sensor_data = []
    with ur.open(alignment_fpath) as file:
        for sensor in sensor_names:
            sensor_data.append(
                    file['Xray;1']['hMap_Sr90_{}_V0;1'.format(sensor)].allvalues
            )
    return sensor_data, sensor_names


def calc_reweight_from_arrays(reference, unweighted_array):
    """calculate the reweigh factor for the sensor remap"""
    return np.mean(reference)/np.mean(unweighted_array)


def remap_sensor(sensor_data):
    """remap the non standard pixel dimensions to the normal grid

    The sensors have differently sized pixels that have to be corrected
    Furthermore there is a dimensional error that has to be fixed (the
    sensor is to high by one row

    This function also determines the reweighing factors for the data
    by looking at the unweighed neighbouring row/column
    """
    # the columns and rows are referenced in matrix indexing convention
    # remove that extra column
    sensor_data = sensor_data[:, 1:]
    # remap the right column
    sensor_data[:, -1] = sensor_data[:, -2]/2
    sensor_data[:, -2] = sensor_data[:, -1]
    # reweigh the right column
    rwfactor = calc_reweight_from_arrays(sensor_data[:, -3],
                                         sensor_data[:, -2])
    sensor_data[:, -2] *= rwfactor
    sensor_data[:, -1] *= rwfactor
    # remap the top row
    sensor_data[0, :] = sensor_data[1, :]/2
    sensor_data[1, :] = sensor_data[0, :]
    # reweigh the top row
    rwfactor = calc_reweight_from_arrays(sensor_data[2, :],
                                         sensor_data[1, :])
    sensor_data[0, :] *= rwfactor
    sensor_data[1, :] *= rwfactor
    # remap the bottom row
    sensor_data[-1, :] = sensor_data[-2, :]/2
    sensor_data[-2, :] = sensor_data[-1, :]
    # reweigh the bottom row
    rwfactor = calc_reweight_from_arrays(sensor_data[-3, :],
                                         sensor_data[-2, :])
    sensor_data[-1, :] *= rwfactor
    sensor_data[-2, :] *= rwfactor
    return sensor_data


def flip_chip(sensor):
    """flip the chip over both axis

    The chips on the top row of the sensor need to be reoriented, as they
    are rotated by 180 deg. This is done via a flip of both axis.

    Args:
        sensor (np.array): the sensor with dimensions (54, 82)

    Returns:
        np.array: the sensor with the values rotated by 180 degrees
    """
    return sensor[::-1, ::-1]


def merge_data(sensor_names, sensor_data):
    """merge the data from the individual chips into a single array

    The data for the alignment hit map is stored inside a root file with
    one map for every chip this function takes the hit maps of every chip
    and converts them into a single hit map while correcting for the
    orientations of the different chips relative to each other.
    """
    # remap sensors
    for i, sensor in enumerate(sensor_data):
        sensor_data[i] = remap_sensor(sensor)
    width, height = sensor_data[0].shape

    # check the dimension of the sensors
    for data in sensor_data[1:]:
        if data.shape[0] != width or data.shape[1] != height:
            raise TypeError("Wrong shape of array")

    # merge the data taking into consideration the location and orientation
    # of the ROCs
    sensor_numbers = [int(name[1:]) for name in sensor_names]
    hitmap = np.zeros((8*width, 2*height), np.int32)
    for i, data in zip(sensor_numbers, sensor_data):
        chip_row = int(i/8)
        chip_column = i % 8
        if chip_row == 0:
            x_start = chip_column * width
            x_end = x_start + width
            y_start = chip_row * height
            y_end = y_start + height
            hitmap[x_start:x_end, y_start:y_end] = data
        else:
            x_start = (7 - chip_column)*width
            x_end = x_start + width
            y_start = chip_row * height
            y_end = y_start + height
            hitmap[x_start:x_end, y_start:y_end] = flip_chip(data)
    return hitmap


def generate_coordinates_from_map(hitmap_hits):
    """Function that generates the coordinate map for the hit histogram

    this function detects the orientation of the sensor (upright or lying on
    it's side and generates the corresponding positions)

    Args:
        hitmap_hits np.array: This is essentially the histogram with the hits

    Returns:
        coordinates np.array: two arrays, one for the x and one for the y
            coordinate of the hitmap bin center so that everything is index
            matched (it can be displayed with the help of matplotlibs 3D axis)
    """
    rows, cols = hitmap_hits.shape
    # determine the orientation of the sensor
    # the sensor is upright the x axis is the short side
    if rows > cols:
        h_x = np.array([i*150*10**-6 for i in range(rows+1)])
        h_y = np.array([i*100*10**-6 for i in range(cols+1)])
        hxc = (h_x[1:] + h_x[:-1])/2
        hyc = (h_y[1:] + h_y[:-1])/2
        return np.meshgrid(hyc, hxc)
    # the sensor is lying on its side (the x axis is the long axis
    else:
        h_y = np.array([i*150*10**-6 for i in range(rows+1)])
        h_x = np.array([i*100*10**-6 for i in range(cols+1)])
        hxc = (h_x[1:] + h_x[:-1])/2
        hyc = (h_y[1:] + h_y[:-1])/2
        return np.meshgrid(hxc, hyc)


def project_hits_onto_axis(hitmap, coordinates):
    """ projects the 2D hitmap onto two histograms

    Function that projects the 2D hitmap onto histograms
    for the x and y axis. The histograms are normalized by the
    function as to make fitting a normalized gauss easier

    Args:
        hitmap_hits: 2D array like
            the hitmap of the sensor that is to be projected
            onto the axis

    Returns:
        x_projection: np.array
            the projection of the hitmap onto the x axis
        y_projection: np.array
            the projection of the hitmap onto the y axis
    """
    # calculate the projections
    x_projection = []
    for row in hitmap:
        x_projection.append(sum(row))
    y_projection = []
    for row in hitmap.T:
        y_projection.append(sum(row))
    # normalize the projections
    x_bin_edges = coordinates[0][0]
    x_bin_widths = x_bin_edges[1:] - x_bin_edges[:-1]
    y_bin_edges = coordinates[1][:, 0]
    y_bin_widths = y_bin_edges[1:] - y_bin_edges[:-1]
    x_projection = np.array([val/sum(x_projection) for val in x_projection])
    y_projection = np.array([val/sum(y_projection) for val in y_projection])
    x_projection /= x_bin_widths
    y_projection /= y_bin_widths
    return x_projection, y_projection


def fit_gauss_indipendently(hitmap, hitmap_coordinates):
    """ fits a gauss curve to the projection of the map onto it's axis

    This function takes the hitmap of the sensor with the hits from the
    alignment exposure and tries to fit a gaussian pdf onto the projection
    of the hitmap onto each of it's axis.

    Args:
        hitmap: 2D array like
            The hitmap of the sensor
        hitmap_coordinates: 2 2D arrays
            output of np.meshgrid for the x and the y coordinates

    Returns:
        x_popt: tuple of floats
            mean and sigma for the x-axis projection fit
        x_pcov: 2D Array
            covariance matrix for the x-axis projection
        y_popt: tuple of floats
            mean and sigma for the y-axis projection fit
        y_pcov: 2D Array
            covariance matrix for the y-axis projection
    """
    x_proj, y_proj = project_hits_onto_axis(hitmap)
    x_coord = hitmap_coordinates[0, 0, :]
    y_coord = hitmap_coordinates[0, :, 0]
    x_popt, x_pcov = opt.curve_fit(norm.pdf, x_coord, x_proj, p0=(np.median(
                                   x_coord), 10*(x_coord[2]-x_coord[1])))
    y_popt, y_pcov = opt.curve_fit(norm.pdf, y_coord, y_proj, p0=(np.median(
                                   y_coord), 10*(y_coord[2]-y_coord[1])))
    return x_popt, x_pcov, y_popt, y_pcov


if __name__ == "__main__":
    # perform alignment of sensor
    # extract data from the alignment files we are only interested in the
    # hit maps, also remap and merge the data from the sensors into a hit
    # map for every sensor
    alignment_datasets = []
    hitmaps = []
    for snsdir in SNSR_DIRS:
        for fpath in Align_files:
            PATH = '/'.join([ALIGNMENT_DIR, snsdir, fpath])
            data, names = get_alignment_data(PATH)
            hitmap_hits = merge_data(names, data)
            hitmaps.append(["/".join([snsdir, fpath]), merge_data(names, data),
                           generate_coordinates_from_map(hitmap_hits)])

    # now that we have the data we need to fit the Gaussian distributions to
    # the data.
    for (hpath, hmap, coordinates) in hitmaps:
        x_popt, x_pcov, y_popt, y_pcov = fit_gauss_indipendently(hmap,
                                                                 coordinates)
        print(x_popt, y_popt)



