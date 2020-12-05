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


if __name__ == "__main__":
    # perform alignment of sensor
    # extract data from the alignment files we are only interested in the
    # hit maps, also remap and merge the data from the sensors into a hit
    # map for every sensor
    alignment_datasets = []
    for snsdir in SNSR_DIRS:
        for fpath in Align_files:
            PATH = '/'.join([ALIGNMENT_DIR, snsdir, fpath])
            data, names = get_alignment_data(PATH)
            hitmap = merge_data(names, data)
    # now that we have the data we need to fit the Gaussian distributions to
    # the data.
