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
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import input as ip
import pixel as px
import plot as pt
import fitting as ft
import transformations as tf

LS_DIR = 'LatencyScan'

ALIGNMENT_DIR = 'Alignment-Sr90'
Align_files = ['Alignment{}-Sr90_oKoll_3600.root'.format(string)
               for string in ['', '2']]

MUON_DIR = 'Muon-Runs'
mrun_files = ['muon-run{}.root'.format(i) for i in range(1, 3)]

SNSR_DIRS = ['M4587-Top', 'M4520-Bottom']

DELTA_Z = 15.9

ROCSHAPE = (80, 52)

LATENCY_SCAN = False

if __name__ == "__main__":
    # perform latency scan for the sensor
    if LATENCY_SCAN:
        for d in SNSR_DIRS:
            latency, hits = ip.get_latency_scan_data(LS_DIR+'/'+d)
            plt.plot(latency, hits, linestyle='--')
            plt.xlabel('index of ring buffer')
            plt.ylabel('sum of hits on sensor')
            plt.grid()
            plt.savefig('latency_plot_for_{}.svg'.format(d))
            plt.close()
    # perform alignment of sensor
    # extract data from the alignment files we are only interested in the
    # hit maps, also remap and merge the data from the sensors into a hit
    # map for every sensor
    alignment_datasets = []
    runs = []
    for fpath in Align_files:
        sensors = ['', '']
        for snsdir in SNSR_DIRS:
            if snsdir == 'M4587-Top':
                snsr = 0
            else:
                snsr = 1
            PATH = '/'.join([ALIGNMENT_DIR, snsdir, fpath])
            data, chip_ids = ip.get_alignment_data(PATH)
            for i, roc_hits in enumerate(data):
                data[i] = px.strip_sensor_of_empty_pixels(roc_hits)
            sensors[snsr] = px.Sensor(data, chip_ids)
        runs.append(px.Telescope(sensors[0], sensors[1]))

    # now that we have the data we need to fit the Gaussian distributions to
    # the data. the thing is that there are two measurements one for the left
    # side and one for the right side and these measurements belong together
    # into a single telescope instance the peaks shoult be arranged per sensor
    # so some reshuffling is needed this is done with the indicies of the for
    # loops
    peaks = [[0, 0], [0, 0]]
    for i, telescope in enumerate(runs):
        for j, sensor in enumerate([telescope.top_sensor,
                                    telescope.bottom_sensor]):
            x_hist, x_bin_edges = sensor.histogram('x', density=True)
            y_hist, y_bin_edges = sensor.histogram('y', density=True)
            x_popt, x_pcov = ft.fit_gauss_normalized_histogram(x_hist,
                                                               x_bin_edges)
            y_popt, y_pcov = ft.fit_gauss_normalized_histogram(y_hist,
                                                               y_bin_edges)
            hit_hist, x_edges, y_edges = sensor.hitmap_histogram(density=True)
            y_centers = (y_edges[1:] + y_edges[:-1])/2
            x_centers = (x_edges[1:] + x_edges[:-1])/2
            xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
            param_start = (x_popt[0], y_popt[0], x_popt[1], y_popt[1])
            opt_min = opt.least_squares(lambda p: ft.gauss_2d(*p)(xx.flatten(),
                                                                  yy.flatten())
                                        - hit_hist.flatten(), param_start)
            popt_2d = opt_min.x
            pt.plot_hitmap_with_projections(hit_hist.T, x_edges, y_edges,
                                            x_hist, y_hist, x_popt, y_popt,
                                            popt_2d,
                                            'r{}_s{}_alnmnt.jpg'.format(i, j))
            peaks[j][i] = (np.mean([x_popt[0], popt_2d[0]]),
                           np.mean([y_popt[0], popt_2d[1]]))

    # find the translation and rotation parameters to position the sensors
    # above each other
    print(peaks)
    top_sns_points = peaks[0]
    bottom_sns_points = peaks[1]

    def errf(p):
        transformed_points = np.array([px.parametrize_transform(*p)(point)
                                      for point in bottom_sns_points])
        dist = 0
        for point, t_point in zip(top_sns_points, transformed_points):
            dist += np.abs(point[0] - t_point[0])
            dist += np.abs(point[1] - t_point[1])
        return dist

    opt_pars = opt.least_squares(errf, (0, 0, 0))

    # just to be sure the transformation should be visualized to optically
    # verify the result
    transform_params = opt_pars.x
    t_points = px.parametrize_transform(*transform_params)(bottom_sns_points)
    pt.plot_coord_transform(top_sns_points, bottom_sns_points, t_points)

    # now that we have the transformation parameters and the transformation
    # the transformation function we have to create a "telescope" without hits
    # that can be passed the transformation function
    data = [np.zeros(ROCSHAPE) for _ in range(16)]
    indicies = range(16)
    transform_func = px.config_telescope_transfrom(transform_params[0],
                                                   transform_params[1],
                                                   transform_params[2],
                                                   -DELTA_Z)

    # build the telescope from the sensors
    top_sensor = px.Sensor(data, indicies)
    bottom_sensor = px.Sensor(data, indicies)
    telescope = px.Telescope(top_sensor, bottom_sensor, transform_func)

    # read in and preprocess the muon data
    m_data = ip.get_muon_hit_data('Muon-Runs')
    sorted_m_data = ip.sort_muon_data(m_data)
    relevant_events, all_events_with_hits, all_events =\
        tf.transform_data_to_needed_format(sorted_m_data)
    all_events_with_hits = len(list(all_events_with_hits))
    all_events = len(list(all_events))

    # write every event to the telescope and get back the angle of the track
    relevant_events = list(relevant_events)
    angles = np.array(list(map(telescope.calc_angle, relevant_events)))
    angles = angles.flatten()
    # transform the angle from radian to degrees
    angles = np.array([(angle/np.pi)*180 for angle in angles])
    print(angles)
    print(len(angles))
    print(len(relevant_events))
    plt.hist(angles)
    plt.show()
