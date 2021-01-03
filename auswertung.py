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

LS_DIR = 'LatencyScan'

ALIGNMENT_DIR = 'Alignment-Sr90'
Align_files = ['Alignment{}-Sr90_oKoll_3600.root'.format(string)
               for string in ['', '2']]

MUON_DIR = 'Muon-Runs'
mrun_files = ['muon-run{}.root'.format(i) for i in range(1, 3)]

SNSR_DIRS = ['M4587-Top', 'M4520-Bottom']

DELTA_Z = 0.0159






def parametrize_transform(theta, t_x, t_y):
    """
    transforms the position of a set of points by translating them
    and then rotating around the origin (2D)
    """
    def rot_mat(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    def parametrized_transform(points):
        t_vec = np.array([t_x, t_y])
        return np.array([(rot_mat(theta)@point) + t_vec for point in points])

    return parametrized_transform


def plot_coord_transform(p1, p2, tp):
    """
    Plot the transformed coordinates alongside the untransformed
    coordinates
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.5))
    ax.scatter([p1[0, 0], p1[1, 0]], [p1[0, 1], p1[1, 1]], color='blue',
               label='positions of the signal peaks on the Top sensor')
    ax.scatter([p2[0, 0], p2[1, 0]], [p2[0, 1], p2[1, 1]], color='gold',
               label='original positions of peaks on bottom sensor')
    ax.scatter([tp[0, 0], tp[1, 0]], [tp[0, 1], tp[1, 1]],
               color='orange',
               label='Transformed points of the bottom sensor')
    plt.grid()
    plt.legend()
    plt.savefig('coordinate_transformation_fit.svg')
    plt.close(fig)


def transform_pixel_coordinates(coordinates, transform_params):
    """
    Take the local coordinate system of a sensor and transform it into
    the coordinate system of the telescope

    The local coordinate system of the top sensor is taken to be the
    coordinate system for the telescope
    """
    c_shape = coordinates[0].shape
    x_coordinate = coordinates[0].flatten()
    y_coordinate = coordinates[1].flatten()
    rearranged_coordinates = np.array(list(
        zip(x_coordinate, y_coordinate)))
    tp = parametrize_transform(*transform_params)(
            rearranged_coordinates)
    tp = tp.flatten()
    tx_coordinate = tp[::2]
    ty_coordinate = tp[1::2]
    tx_coordinate = tx_coordinate.reshape(c_shape)
    ty_coordinate = ty_coordinate.reshape(c_shape)
    tcoord = np.array([tx_coordinate, ty_coordinate])
    return tcoord


if __name__ == "__main__":
    # perform latency scan for the sensor
    #for d in SNSR_DIRS:
    #    latency, hits = get_latency_scan_data(LS_DIR+'/'+d)
    #    plt.plot(latency, hits, linestyle='--')
    #    plt.xlabel('index of ring buffer')
    #    plt.ylabel('sum of hits on sensor')
    #    plt.grid()
    #    plt.savefig('latency_plot_for_{}.svg'.format(d))
    #    plt.show()
    # perform alignment of sensor
    # extract data from the alignment files we are only interested in the
    # hit maps, also remap and merge the data from the sensors into a hit
    # map for every sensor
    alignment_datasets = []
    hitmaps = []
    for snsdir in SNSR_DIRS:
        for fpath in Align_files:
            PATH = '/'.join([ALIGNMENT_DIR, snsdir, fpath])
            data, names = ip.get_alignment_data(PATH)
            hitmap_hits = merge_data(names, data)
            hitmaps.append([PATH, hitmap_hits,
                           generate_coordinates_from_map(hitmap_hits)])

    # now that we have the data we need to fit the Gaussian distributions to
    # the data.
    peaks = []
    for (hpath, hmap, coordinates) in hitmaps:
        print("{} shape = {}".format(hpath, hmap.shape))
        xparams, yparams, projections = fit_gauss_indipendently(hmap,
                                                                coordinates)
        print(xparams, yparams)
        opt_min = fit_2D_gauss(hmap, coordinates,
                               (xparams[0], xparams[1],
                                yparams[0], yparams[1]))
        popt_2d = opt_min.x
        plot_hitmap_with_projections(hmap, coordinates, projections,
                                     xparams, yparams, popt_2d, hpath)
        peaks.append((np.mean([xparams[0], popt_2d[0]]),
                      np.mean([yparams[0], popt_2d[1]])))

    # find the translation and rotation parameters to position the sensors
    # above each other
    print(peaks)
    sns1_points = np.array(peaks[0:2])
    sns2_points = np.array(peaks[2:])
    errf = lambda p: np.sqrt(sum((sns1_points - parametrize_transform(*p)(
        sns2_points))**2))
    opt_pars = opt.least_squares(errf, (0, 0, 0))

    # just to be sure the transformation should be visualized to optically
    # verify the result
    transform_params = opt_pars.x
    t_points = parametrize_transform(*transform_params)(sns2_points)
    plot_coord_transform(sns1_points, sns2_points, t_points)

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
    muon_hit_data = ip.get_muon_hit_data(MUON_DIR)
    def filter_muon_hits(muon_hit_data):
        # TODO: map the read out chip to the proper coordinates
        npix = muon_hit_data[0]
        proc = muon_hit_data[1]
        pcol = muon_hit_data[2]
        prow = muon_hit_data[3]
        pq = muon_hit_data[4]
        npix = filter(lambda p: p > 0, npix)
        proc = filter(lambda p: len(p) > 0, proc)
        pcol = filter(lambda p: len(p) > 0, pcol)
        prow = filter(lambda p: len(p) > 0, prow)
        pq = filter(lambda p: len(p) > 0, pq)
        for npix, proc, pcol, prow, pq in zip(npix, proc, prow, pcol, pq):
            for 
            if npix > 1:
                distinct_hits.append(proc[0], pcol[0], prow[0], pq[0])
            else:
                distinct_hits = []
                for proc, pcol, prow, pq in zip(proc[1:], pcol[1:],
                                                prow[1:], pq[1:]):
                    np.sqrt




