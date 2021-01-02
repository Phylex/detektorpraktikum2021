"""
Module that contains the data structures that represent the sensors and pixels
with the input and output code to generate the data formats for plotting with
matplotlib and calculations with numpy
"""
import numpy as np


class Sensor():
    pass


class ReadoutChip():
    def __init__(self, data, x_pixel_borders, y_pixel_borders, orientation):
        self.orientation = orientation
        self.shape = data.shape
        xx, yy = np.meshgrid(x_pixel_borders, y_pixel_borders)


class pixel():
    def __init__(self, hitcount, x_dim, y_dim, top_left_corner_coord):
        self.hits = hitcount
        self.width = x_dim
        self.height = y_dim
        self.center = (top_left_corner_coord[0] + x_dim/2,
                       top_left_corner_coord[1] + y_dim/2)


def strip_sensor_of_empty_pixels(sensor_hitmap):
    """
    the sensor has a border of pixels that don't have any pixels, we strip this
    border before any other transformation occurs as it messes up the
    coordinates
    """
    return sensor_hitmap[1:-1, 1:-1]


def generate_sensor_coordinates(hitmap):
    """ Function that generates the coordinate map for a single sensor

    All measurements are given in millimeters
    Also detects the orientation of the sensor and rotates it to the following
    Orientation:

     y(i)
     79
      ^
      |
    0 +--> 51 x(j)
      0

    Parameters
    ----------
    hitmap : array 2D
        The hitmap with one entry per pixel with the same topological
        arrangement of the pixels as on the sensor

    Returns
    -------
    xc : 1D array
        the x coordinate of the bottom left corner of the pixel
    yc : 1D array
        the y coordinate of the bottom left corner of the pixel
    dx : 1D array
        the length in x of the pixel
    dy : 1D array
        the length in y of the pixel

    """
    # reorient the sensor if neccesary
    i, j = hitmap.shape
    if j > i:
        hitmap = hitmap.T
        i, j = hitmap.shape

    regular_column_part = np.array([0.3+k*0.15 for k in range(j-1)])
    xc = np.concatenate(([0], regular_column_part,
                        [regular_column_part[-1]+0.3]), axis=0)
    regular_row_part = np.array([k*0.1 for k in range(i)])
    yc = np.concatenate((regular_row_part, [regular_row_part[-1]+0.2]))
    dx = xc[1:] - xc[:-1]
    dy = yc[1:] - yc[:-1]
    return xc, yc, dx, dy
