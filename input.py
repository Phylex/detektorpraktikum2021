"""
Module containing the input functions of the analysis so the functions that
read in the data from the provided root files
"""

from os import listdir
from os.path import isfile, join
from os import path as pt
import uproot as ur
import transformations as tf

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
    sensor_names = range(0, 16)
    sensor_data = []
    with ur.open(alignment_fpath) as file:
        for sensor in sensor_names:
            sensor_data.append(
                file['Xray;1']['hMap_Sr90_C{}_V0;1'.format(sensor)].allvalues
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


def get_hits_from_muon_file(fpath):
    """
    extract the data of interest from the root file pointed to by fpath
    """
    rootf = ur.open(fpath)
    events = rootf['Xray']['events']
    npix = events['npix'].array()
    proc = events['proc'].array()
    pcol = events['pcol'].array()
    prow = events['prow'].array()
    pq = events['pq'].array()
    return (npix, proc, pcol, prow, pq)


def get_muon_hit_data(muon_dir):
    """
    find all relevant files for the muon hit data and read their contents
    """
    hit_files = []
    for d in SNSR_DIRS:
        for f in listdir(muon_dir+'/'+d):
            fpath = join(muon_dir, d, f)
            if isfile(fpath):
                hit_files.append((fpath, get_hits_from_muon_file(fpath)))
    return hit_files

def sort_muon_data(data):
    sorted_data = {}
    paths = [datum[0] for datum in data]
    runs = [pt.basename(path) for path in paths]
    sensors = [pt.basename(pt.dirname(path)) for path in paths]
    data = [datum[1] for datum in data]
    for unique_sensor in set(sensors):
        sorted_data[unique_sensor] = {}
        for uniqe_run in set(runs):
            sorted_data[unique_sensor][uniqe_run] = None
    for run, sns, datum in zip(runs, sensors, data):
        sorted_data[sns][run] = datum
    return sorted_data


if __name__ == "__main__":
    m_data = get_muon_hit_data('Muon-Runs')
    sorted_m_data = sort_muon_data(m_data)
    relevant_hits, all_hits, all_events = tf.transform_data_to_needed_format(
            sorted_m_data)
    relevant_hits = [elem for elem in relevant_hits]
    all_hits = len(list(all_hits))
    all_events = len(list(all_events))
