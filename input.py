"""
Module containing the input functions of the analysis so the functions that
read in the data from the provided root files
"""

from os import listdir
from os.path import isfile, join
import uproot as ur


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


def get_latency_scan_data(latency_dir_path):
    """ open and extract the data from all files in the latency_directory """
    files_in_dir = [f for f in listdir(latency_dir_path)
                    if isfile(join(latency_dir_path, f))]
    delay_buffer_ind = [int(fname.split('.')[0]) for fname in files_in_dir]
    latency_scan = []
    # iterate over every file (a file holds the hitmap for all sensors with the
    # same buffer index. The buffer index is encoded in the file name
    for fname, delay_ind in zip(files_in_dir, delay_buffer_ind):
        with ur.open(latency_dir_path+'/'+fname) as file:
            names = ['hMap_Ag_C{}_V0'.format(i) for i in range(16)]
            ldata = [file['Xray'][name].allvalues for name in names]
            hitcount = sum([sum(sensor.flatten()) for sensor in ldata])
            latency_scan.append((delay_ind, hitcount))
    lindex, hits = (zip(*latency_scan))
    return lindex, hits
