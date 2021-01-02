"""
Module that contains the data structures that represent the sensors and pixels
with the input and output code to generate the data formats for plotting with
matplotlib and calculations with numpy
"""
import numpy as np
import pytest

Point = tuple[float, float]


class Sensor():
    pass


class ReadoutChip():
    def __init__(self, hitmap: np.ndarray, chip_nr: int):
        """ set up the readout chip from the hitmap and the chip ID

        Generate the neccessary meta information for the roc on the sensor.
        This function calculates the orientation and offset position of the
        roc relative to the global sensor coordinates so that the roc
        data structure can hold every pixel with it's local position on the
        sensor and then generate the global position from that
        """
        i, j = hitmap.shape
        if j > i:
            hitmap = hitmap.T
        x_coord, y_coord, dx, dy, roc_dim = generate_sensor_coordinates(hitmap)
        if chip_nr > 7:
            vertical_offset = 2 * roc_dim[1]
            horizontal_offset = (8 * roc_dim[0]) - ((chip_nr % 8) * roc_dim[0])
            self.offset = (vertical_offset, horizontal_offset)
            self.orientation_matrix = np.array([[1, 0], [0, -1]]) @ \
                np.array([[-1, 0], [0, 1]])
        else:
            vertical_offset = roc_dim[1] * chip_nr
            horizontal_offset = roc_dim[0] * (chip_nr % 8)
            self.offset = (vertical_offset, horizontal_offset)
        pixel_positions = np.meshgrid(x_coord, y_coord)
        pixel_dimensions = np.meshgrid(dx, dy)
        self.pixels = []
        for prow_x, prow_y, drow_x, drow_y, hitrow in zip(pixel_positions[0],
                                                          pixel_positions[1],
                                                          pixel_dimensions[0],
                                                          pixel_dimensions[1],
                                                          hitmap):
            pixel_row = []
            for px, py, dx, dy, hitcount in zip(prow_x, prow_y, drow_x, drow_y,
                                                hitrow):
                pixel_row.append(pixel(hitcount, dx, dy, (px, py)))
            self.pixels.append(pixel_row)

    def __setitem__(self, pixel_index: tuple[int, int], pixel_hitcount):
        """ set the hitcount and qval of the pixel indexed by pixel_index

        this method has two usecases:
        1. it is used to generate the hitmap for the alignment
            in which case the q_val is irrelevant and should not be set
        2. it is used to generate the individual muon hits when actually being
            used as muon telescope

        Parameters
        ----------
        pixel_index : tuple(int, int)
            the index of the pixel on the sensor as x and y indicies
            x from 0 to 51 and y from 0 to 79
        pixel_hitcount : int or tuple(int, float)
            depending on if it is a tuple or an int the q_val will be set
            if the q_val is set then the hitcount is ignored when retreiving
            the data from the sensor as it is considered to be 1 (we are in
            the second mode of operation as telescope)
        """
        try:
            curpix = self.pixels[pixel_index[0]][pixel_index[1]]
            self.pixels[pixel_index[0]][pixel_index[1]] = pixel(
                    pixel_hitcount[0],
                    curpix.width,
                    curpix.height,
                    (curpix.center[0] - curpix.width/2,
                        curpix.center - curpix.height/2),
                    pixel_hitcount[1])
        except TypeError:
            curpix = self.pixels[pixel_index[0]][pixel_index[1]]
            self.pixels[pixel_index[0]][pixel_index[1]] = pixel(
                    pixel_hitcount,
                    curpix.width,
                    curpix.height,
                    (curpix.center[0] - curpix.width/2,
                        curpix.center - curpix.height/2))

    def __getitem__(self, pixel_index: tuple[int, int]):
        """ gets either the hitcount or the q_val depending on
        if the q_val was set or not

        Parameters
        ----------
        pixel_index : tuple(int, int)
            the index of the pixel on the sensor as x and y indicies
            x from 0 to 51 and y from 0 to 79

        Returns
        -------
        hitcount: int
            the amount of hits of the cell (it is different from one only
            if q_val is 0)
        q_val: float
            a proxy for the charge that was gathered by the cell (this is
            only given if the hitcount is 1)
        """
        curpix = self.pixels[pixel_index[0]][pixel_index[1]]
        return (curpix.hits, curpix.q_val)

    def normalized_hitmap(self):
        """ return the hitcount/area for each pixel as a hitmap

        Generate a list of tuples of positions and associated hitcounts/pixel
        area

        Returns
        -------
        hits : list[tuple(float, float, int)]
            a list of positions and hits at the corresponding position
        """
        pixel_indicies = np.indices((len(self.pixels[0]), len(self.pixels)))
        x_index = pixel_indicies[0].flatten()
        y_index = pixel_indicies[1].flatten()
        hitmap = []
        for i, j in zip(y_index, x_index):
            local_position = self.pixels[i][j].get_position()
            global_position = (self.orientation_matrix @ local_position) \
                + self.offset
            hitmap.append((global_position,
                           self.pixels[i][j].get_normalized_hitcount()))
        return hitmap


class pixel():
    def __init__(self, hitcount: int, x_dim: float, y_dim: float,
                 top_left_corner_coord: Point, q_val=None):
        self.hits = hitcount
        self.width = x_dim
        self.height = y_dim
        self.center = (top_left_corner_coord[0] + x_dim/2,
                       top_left_corner_coord[1] + y_dim/2)
        self.q_val = q_val

    def get_normalized_hitcount(self):
        """ get the hits per pixel area

        This method corrects the hitcount for the different sizes of the pixels
        """
        return self.hits / (self.width * self.height)

    def get_position(self):
        """ returns the position of the pixel center coordinates """
        return self.center


def strip_sensor_of_empty_pixels(roc_hitmap: np.ndarray):
    """
    the sensor has a border of pixels that don't have any pixels, we strip this
    border before any other transformation occurs as it messes up the
    coordinates
    """
    return roc_hitmap[1:-1, 1:-1]


def generate_sensor_coordinates(roc_hitmap: np.ndarray):
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
    roc_dim : tuple(float, float)
        the dimensions of the read out chip along it's x and y axis
    """
    # reorient the sensor if neccesary
    i, j = roc_hitmap.shape
    if j > i:
        roc_hitmap = roc_hitmap.T
        i, j = roc_hitmap.shape

    regular_column_part = np.array([0.3+k*0.15 for k in range(j-1)])
    xc = np.concatenate(([0], regular_column_part,
                        [regular_column_part[-1]+0.3]), axis=0)
    regular_row_part = np.array([k*0.1 for k in range(i)])
    yc = np.concatenate((regular_row_part, [regular_row_part[-1]+0.2]))
    dx = xc[1:] - xc[:-1]
    dy = yc[1:] - yc[:-1]
    return xc[:-1], yc[:-1], dx, dy, (xc[-1], yc[-1])


def test_generate_sensor_coordinates():
    """ test the `generate_sensor_coordinates` function

    check and see if the dimensions of the coordinates line up
    with the dimensions of the hitmap given
    """
    hitmap = np.random.randint(0, 255, (100, 50))
    xc, yc, dx, dy, dim = generate_sensor_coordinates(hitmap)
    assert len(xc) == 50
    assert len(yc) == 100
    assert len(dx) == 50
    assert len(dy) == 100
    assert dim[1] == yc[-1]+0.2
    assert dim[0] == xc[-1]+0.3
