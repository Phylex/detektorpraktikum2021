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
from os import path
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import norm
import input as ip
import pixel as px
import plot as pt
import fitting as ft

LS_DIR = 'LatencyScan'

ALIGNMENT_DIR = 'Alignment-Sr90'
Align_files = ['Alignment{}-Sr90_oKoll_3600.root'.format(string)
               for string in ['', '2']]

MUON_DIR = 'Muon-Runs'
mrun_files = ['muon-run{}.root'.format(i) for i in range(1, 3)]

SNSR_DIRS = ['M4587-Top', 'M4520-Bottom']

DELTA_Z = 0.0159


if __name__ == "__main__":
    # perform latency scan for the sensor
    for d in SNSR_DIRS:
        latency, hits = ip.get_latency_scan_data(LS_DIR+'/'+d)
        plt.plot(latency, hits, linestyle='--')
        plt.xlabel('index of ring buffer')
        plt.ylabel('sum of hits on sensor')
        plt.grid()
        plt.savefig('latency_plot_for_{}.svg'.format(d))
        plt.show()
    # perform alignment of sensor
    # extract data from the alignment files we are only interested in the
    # hit maps, also remap and merge the data from the sensors into a hit
    # map for every sensor
    alignment_datasets = []
    hitmaps = []
    for snsdir in SNSR_DIRS:
        for fpath in Align_files:
            PATH = '/'.join([ALIGNMENT_DIR, snsdir, fpath])
            data, chip_ids = ip.get_alignment_data(PATH)
            for i, roc_hits in enumerate(data):
                data[i] = px.strip_sensor_of_empty_pixels(roc_hits)
            sensor = px.Sensor(data, chip_ids)
            hitmaps.append([PATH, sensor])

    # now that we have the data we need to fit the Gaussian distributions to
    # the data.
    peaks = []
    for hpath, sensor in hitmaps:
        x_hist, x_bin_edges = sensor.histogram('x', density=True)
        y_hist, y_bin_edges = sensor.histogram('y', density=True)
        x_popt, x_pcov = ft.fit_gauss_normalized_histogram(x_hist, x_bin_edges)
        y_popt, y_pcov = ft.fit_gauss_normalized_histogram(y_hist, y_bin_edges)
        hit_hist, x_edges, y_edges = sensor.hitmap_histogram(density=True)
        y_centers = (y_edges[1:] + y_edges[:-1])/2
        x_centers = (x_edges[1:] + x_edges[:-1])/2
        xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
        param_start = (x_popt[0], y_popt[0], x_popt[1], y_popt[1])
        opt_min = opt.least_squares(lambda p: ft.gauss_2d(*p)(xx.flatten(),
                                                              yy.flatten())
                                    - hit_hist.flatten(), param_start)
        popt_2d = opt_min.x
        pt.plot_hitmap_with_projections(hit_hist.T, x_edges, y_edges, x_hist,
                                        y_hist, x_popt, y_popt, popt_2d, hpath)
        peaks.append((np.mean([x_popt[0], popt_2d[0]]),
                      np.mean([y_popt[0], popt_2d[1]])))

    # find the translation and rotation parameters to position the sensors
    # above each other
    print(peaks)
    sns1_points = np.array(peaks[0:2])
    sns2_points = np.array(peaks[2:])
    errf = lambda p: np.sqrt(sum((sns1_points - px.parametrize_transform(*p)(
        sns2_points))**2))
    opt_pars = opt.least_squares(errf, (0, 0, 0))

    # just to be sure the transformation should be visualized to optically
    # verify the result
    transform_params = opt_pars.x
    t_points = px.parametrize_transform(*transform_params)(sns2_points)
    pt.plot_coord_transform(sns1_points, sns2_points, t_points)

    # now that we have the transformation parameters and the transformation
    # the transformation function we have to transform the grid of the bottom
    # sensor
    for i, (hpath, hmap, coordinates) in enumerate(hitmaps):
        if i in (2, 3):
            tcoord = transform_pixel_coordinates(coordinates, transform_params)
            hitmaps[i] = (hpath, hmap, tcoord)

    # finally we can use the vertical distance of the sensors and add a z
    # coordinate to the 2D coordinates of the sensors
    for i, (hpath, hmap, coordinates) in enumerate(hitmaps):
        # top sensor, is height 0
        if hpath == ALIGNMENT_DIR+'/'+SNSR_DIRS[0]:
            coordinates = np.array([coordinates[0], coordinates[1],
                                   np.zeros_like(coordinates[0])])
            hitmaps[i] = (hpath, hmap, coordinates)
        # bottom sensor is -dz
        else:
            coordinates = np.array([coordinates[0], coordinates[1],
                                   np.zeros_like(coordinates[0])-DELTA_Z])
            hitmaps[i] = (hpath, hmap, coordinates)

    # now we have the properly aligned coordinate system for the sensors
    # and can start to read in the data
#    muon_hit_data = ip.get_muon_hit_data(MUON_DIR)
#    def filter_muon_hits(muon_hit_data):
#        # TODO: map the read out chip to the proper coordinates
#        npix = muon_hit_data[0]
#        proc = muon_hit_data[1]
#        pcol = muon_hit_data[2]
#        prow = muon_hit_data[3]
#        pq = muon_hit_data[4]
#        npix = filter(lambda p: p > 0, npix)
#        proc = filter(lambda p: len(p) > 0, proc)
#        pcol = filter(lambda p: len(p) > 0, pcol)
#        prow = filter(lambda p: len(p) > 0, prow)
#        pq = filter(lambda p: len(p) > 0, pq)
#        for npix, proc, pcol, prow, pq in zip(npix, proc, prow, pcol, pq):
#            for 
#            if npix > 1:
#                distinct_hits.append(proc[0], pcol[0], prow[0], pq[0])
#            else:
#                distinct_hits = []
#                for proc, pcol, prow, pq in zip(proc[1:], pcol[1:],
#                                                prow[1:], pq[1:]):
#                    np.sqrt
#
#
#
#
