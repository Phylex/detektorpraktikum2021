"""
Module that contains the data structures that represent the sensors and pixels
with the input and output code to generate the data formats for plotting with
matplotlib and calculations with numpy
"""
import itertools as itt
import numpy as np

Point = tuple[float, float]


def strip_sensor_of_empty_pixels(roc_hitmap: np.ndarray):
    """
    the sensor has a border of pixels that don't have any pixels, we strip this
    border before any other transformation occurs as it messes up the
    coordinates
    """
    return roc_hitmap[1:-1, 1:-1]


def remove_duplicates_from_sorted_array(array):
    """remove duplicate entries in a sorted array"""
    cur = array[0]
    newarr = [cur]
    for elem in array[1:]:
        if elem != cur:
            newarr.append(elem)
            cur = elem
    return np.array(newarr)


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
    x_coord = np.concatenate(([0], regular_column_part,
                              [regular_column_part[-1]+0.3]), axis=0)
    regular_row_part = np.array([k*0.1 for k in range(i)])
    y_coord = np.concatenate((regular_row_part, [regular_row_part[-1]+0.2]))
    delta_x = x_coord[1:] - x_coord[:-1]
    delta_y = y_coord[1:] - y_coord[:-1]
    return x_coord[:-1], y_coord[:-1], delta_x, delta_y,\
        (x_coord[-1], y_coord[-1])


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


def roc_to_sensor_transfrom(position: np.ndarray, rot_mat: np.ndarray,
                            offset: np.ndarray) -> np.ndarray:
    """ function that transfrorms the position inside a read out chip into
    a position in the sensor coordinate system """
    return rot_mat @ position + offset


def sensor_to_telescope_transfrom(position: np.ndarray, rot_mat: np.ndarray,
                                  offset: np.ndarray) -> np.ndarray:
    """ function that transforms any coordinate from the sensor to a coordinate
    in the telescope coordinate system """
    position_3d = np.append(position, 0)
    return position_3d @ rot_mat + offset


class Telescope():
    """ Class representing the muon telescope """
    def __init__(self, top_sensor, bottom_sensor, tf_func):
        self.top_sensor = top_sensor
        self.bottom_sensor = bottom_sensor
        self.transform_func = tf_func
        self.transform_params = None

    def configure_bottom_transformation(self, tf_parameters):
        """ configure the coordinate transformation from
        sensor to telescope coordinates

        configure the transformation function for the coordinate
        alignment of the bottom and top sensors. The transformation
        is applied to the coordinate system of the bottom sensor
        """
        self.transform_params = tf_parameters

    def __getitem__(self, index):
        if len(index) != 4:
            raise ValueError("the index needs to be of length 4")
        if index[0] == 0:
            return self.top_sensor[index[1:]]
        if index[0] == 1:
            pos, hits, q_val = self.bottom_sensor[index[1:]]
            if self.transform_params is not None:
                pos = self.transform_func(pos, *self.transform_params)
            return (pos, hits, q_val)
        raise ValueError("the 0th index need to be either '0' or '1'")


class Sensor():
    """Class representing one entire sensor assembly

    This class represents an entire sensor consisting of the 16 read
    out chips and some meta information.
    Two sensors are used stacked on top of each other to build the
    muon telescope that should be analysed in the laboratory
    """
    def __init__(self, hitmaps: list[np.ndarray], chip_id: list[int],
                 tf_function=None,
                 tf_params: tuple[np.ndarray, np.ndarray] = None):
        """ produce the sensor datastructure from the hitmaps of the ROCs

        produces a sensor with all the neccesary metadata from the hitmaps
        read in from the rootfile hitmaps

        Parameters
        ----------
        hitmaps : list[np.ndarray]
            the hitmaps of the different readout chips. The index of the hitmap
            needs to be the index of the chip ID in chip_ids otherwise the
            mapping will be off
        chip_id : list[int]
            this is the ID of the ROC and is in the range 0 to 15
        offset : tuple[np.ndarray, np.ndarray] or None
            this is the offset and orientation (in the order in that they where
            mentioned) of the sensor relative to some global
            coordiante system (this is needed for the later alignment)
        """
        self.transform_function = tf_function
        self.transform_params = tf_params
        self.rocs = []
        for hmap, chip_nr in zip(hitmaps, chip_id):
            _, _, _, _, roc_dim = generate_sensor_coordinates(hmap)
            if chip_nr > 7:
                x_offset = (8 * roc_dim[0]) - ((chip_nr % 8) * roc_dim[0])
                y_offset = 2 * roc_dim[1]
                offset = (x_offset, y_offset)
                orientation_matrix = np.array([[1, 0], [0, -1]]) @ \
                    np.array([[-1, 0], [0, 1]])
            else:
                y_offset = 0
                x_offset = roc_dim[0] * chip_nr
                offset = (x_offset, y_offset)
                orientation_matrix = np.array([[1, 0], [0, 1]])
            tf_params = (orientation_matrix, offset)
            readout_chip = ReadoutChip(hmap, chip_nr, roc_to_sensor_transfrom,
                                       tf_params)
            self.rocs.append(readout_chip)

    def __getitem__(self, index):
        """ get the pixel at the index

        the pixel index has to be given as the roc local index together
        with the number of the read out chip on which the pixel is
        located

        Parameters
        ----------
        index : tuple[int, int, int]
            the index of the pixel. the first number is the chip ID of the
            read out chip and the following two indicies are the roc-local
            index of the pixel

        Returns
        -------
        pixel : px.Pixel
            the pixel indicated by the given indicies
        """
        if isinstance(index, (list, tuple)):
            raise TypeError(
                    "The index for this operation must be a list of length 3")
        if len(index) != 3:
            raise TypeError("The length of the index needs to be 3")
        pos, hits, q_val = self.rocs[index[0]][index[1:]]
        if self.transform_function is not None\
                and self.transform_params is not None:
            pos = self.transform_function(pos, *self.transform_params)
        return (pos, hits, q_val)

    def __setitem__(self, index, value):
        """set the value of a pixel with roc local index

        The [] syntax can now be used to write values to the pixels of the
        sensor as long as they are using ROC local indexes for the pixels
        this will be used to read in the events from the muon runs
        """
        self[index] = value

    def hististogram_data(self, axis: str):
        """ generate raw histogram data for the entire sensor """
        roc_hits = [roc.histogramm_data(axis) for roc in self.rocs]
        return np.concatenate(roc_hits)

    def histogram(self, axis: str, density=False) \
            -> tuple[np.ndarray, np.ndarray]:
        """ generate a histogram of the entire sensor along one of it's axis

        generates a pixel perfect histogram of the hits on the sensor in
        sensor local coordinates

        Parameters
        ----------
        axis : str
            the axis along which to generate the histogram. Either 'x' or 'y'
        density : bool
            switch that decides if the histogram is normalized to area 1 or
            given as "raw" histogram (with the amount of hits in each bin)
        """
        hist_data = self.hististogram_data(axis)
        borders = [roc.get_pixel_borders(axis) for roc in self.rocs]
        borders = remove_duplicates_from_sorted_array(
                np.sort(np.round(np.concatenate(borders), 3)))
        return np.histogram(hist_data, borders, density=density)

    def hitmap_histogram(self, density: bool = False) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """generates a 2D histogram of the hits on the sensor

        this function generates a 2D histogram of the hits on the sensor using
        the numpy.histogram2d method. It therefore has the same return values

        Parameters
        ----------
        density : bool
            this flag decides if the flux density will be given or the hits
            per pixel

        Returns
        -------
        hist : np.ndarray
            the 2D histogram of the hits on the sensor
        x_edges : np.ndarray
            the edges of the bins on the x axis
        y_edges : np.ndarray
            the edges of the bins on the y axis
        """
        x_borders = [roc.get_pixel_borders('x') for roc in self.rocs[:8]]
        x_borders = remove_duplicates_from_sorted_array(
                np.sort(np.round(np.concatenate(x_borders), 3)))
        y_borders = [roc.get_pixel_borders('y')
                     for roc in [self.rocs[0], self.rocs[8]]]
        y_borders = remove_duplicates_from_sorted_array(
                np.sort(np.round(np.concatenate(y_borders), 3)))
        hits = [roc.hitmap() for roc in self.rocs]
        hits = np.array(hits, dtype=object)
        hits = np.concatenate(hits)
        hits = np.array([[np.round(e[0][0], 3), np.round(e[0][1], 3),
                        np.round(e[1], 3)] for e in hits])
        x_coords, y_coords, weights = zip(*hits)
        return np.histogram2d(x_coords, y_coords, bins=[x_borders, y_borders],
                              weights=weights, density=density)


class ReadoutChip():
    def __init__(self, hitmap: np.ndarray, chip_nr: int, tf_func, tf_params):
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
        x_coord, y_coord, delta_x, delta_y, _ = \
            generate_sensor_coordinates(hitmap)
        pixel_positions = np.meshgrid(x_coord, y_coord)
        pixel_dimensions = np.meshgrid(delta_x, delta_y)
        self.pixels = []
        self.transfrom_function = tf_func
        self.trandform_params = tf_params
        self.chip_nr = chip_nr
        for prow_x, prow_y, drow_x, drow_y, hitrow in zip(pixel_positions[0],
                                                          pixel_positions[1],
                                                          pixel_dimensions[0],
                                                          pixel_dimensions[1],
                                                          hitmap):
            pixel_row = []
            for pos_x, pos_y, dx, dy, hitcount in zip(prow_x, prow_y, drow_x,
                                                      drow_y, hitrow):
                pixel_row.append(Pixel(hitcount, dx, dy, (pos_x, pos_y)))
            self.pixels.append(pixel_row)
        self.pixels = np.array(self.pixels, dtype=object)

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
        pixel_center: [x_center, y_center]
            the center coordinate of the pixel
        hitcount: int
            the amount of hits of the cell (it is different from one only
            if q_val is 0)
        q_val: float
            a proxy for the charge that was gathered by the cell (this is
            only given if the hitcount is 1)
        """
        pixel = self.pixels[pixel_index[0]][pixel_index[1]]
        center = self.transfrom_function(pixel.center, *self.trandform_params)
        return (center, pixel.hits, pixel.q_val)

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
        i, j = pixel_index
        try:
            curpix = self.pixels[i][j]
            self.pixels[i][j] = Pixel(
                    pixel_hitcount[0],
                    curpix.width,
                    curpix.height,
                    curpix.blc,
                    pixel_hitcount[1])
        # the pixel hitcount is a value not an array
        except TypeError:
            curpix = self.pixels[pixel_index[0]][pixel_index[1]]
            self.pixels[pixel_index[0]][pixel_index[1]] = Pixel(
                    pixel_hitcount,
                    curpix.width,
                    curpix.height,
                    curpix.blc)

    def get_pixel_borders(self, axis: str):
        """ get the sensor local borders of the readout chip (this uses
        the generate_sensor_coordinates function
        """
        pixels = self.pixels.flatten()
        corners = np.array([pixel.get_edge_corners() for pixel in pixels])
        corners = np.concatenate(corners)
        sns_corners = np.array([self.transfrom_function(corner,
                                *self.trandform_params)
                                for corner in corners])
        if axis == 'x':
            corner_index = 0
        elif axis == 'y':
            corner_index = 1
        else:
            raise ValueError("axis needs to be either 'x' or 'y'")
        borders = np.array([corner[corner_index] for corner in sns_corners])
        borders = remove_duplicates_from_sorted_array(
                    np.sort(np.round(borders, 3)))
        return borders

    def hitmap(self, normalized=False):
        """ return the (normalized) hitcount for each pixel as a hitmap

        Generate a list of tuples of positions and associated hits.
        if the normalized flag is set the hitcount is scaled to the pixel
        area giving the hits/area measurement correcting for different sized
        pixels

        Parameters
        ----------
        normalized : bool
            flag that decides if the hitcount per pixel is returned or
            hits/area (flux) is returned for each pixel

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
            sns_position = self[(i, j)][0]
            if normalized:
                hitmap.append((sns_position,
                               self.pixels[i][j].get_normalized_hitcount()))
            else:
                hitmap.append((sns_position,
                               self.pixels[i][j].hits))
        return hitmap

    def histogramm_data(self, axis: str):
        """ generate the raw hit data for a histogram

        Parameters
        ----------
        axis : str
            the axis that the data should be generated for either 'x' or 'y'

        Returns
        -------
        hits : list[float]
            depending on the axis parameter either the x or y coordinate for
            every hit. A hitcount of n > 1 will result in the coordinate being
            repeated n number of times so that the raw data can be fed directly
            to np.histogram or plt.hist
        """

        hitmap = self.hitmap()
        if axis == 'x':
            hits_per_pixel = np.array([[elem[0][0], elem[1]]
                                       for elem in hitmap])
        elif axis == 'y':
            hits_per_pixel = np.array([[elem[0][1], elem[1]]
                                       for elem in hitmap])
        else:
            raise ValueError("the axis has to be either 'x' or 'y'")
        hits = []
        for elem in hits_per_pixel:
            for x_coord in itt.repeat(elem[0], int(elem[1])):
                hits.append(x_coord)
        return hits


class Pixel():
    def __init__(self, hitcount: int, x_dim: float, y_dim: float,
                 bottom_left_corner_coord: Point, q_val=None):
        self.hits = hitcount
        self.width = x_dim
        self.height = y_dim
        self.center = (bottom_left_corner_coord[0] + x_dim/2,
                       bottom_left_corner_coord[1] + y_dim/2)
        self.blc = bottom_left_corner_coord
        self.q_val = q_val

    def get_normalized_hitcount(self):
        """ get the hits per pixel area

        This method corrects the hitcount for the different sizes of the pixels
        """
        return self.hits / (self.width * self.height)

    def get_edge_corners(self):
        """ get the bottom left (lower x and y pixel border) and the top right (upper
        x and y pixel boundary """
        return np.array([self.blc,
                         self.blc + np.array([self.width, self.height])])


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


def test_generate_x_histogram():
    """ check that the histogram functions of the sensor method work """
    roc_hitmaps = [np.random.randint(0, 255, (100, 50)) for _ in range(16)]
    hits = sum(np.array(roc_hitmaps).flatten())
    chip_ids = range(16)
    sensor = Sensor(roc_hitmaps, chip_ids)
    data, bins = sensor.histogram('x')
    assert len(bins) == 8*50+1
    assert sum(data) == hits
    data, bins = sensor.histogram('y')
    assert len(bins) == 2 * 100 + 1
    assert sum(data) == hits


def test_gen_2d_hist():
    """ test the dimensionality of the sensor hitmap"""
    roc_hitmaps = [np.random.randint(0, 255, (100, 50)) for _ in range(16)]
    # hits = sum(np.array(roc_hitmaps).flatten())
    chip_ids = range(16)
    sensor = Sensor(roc_hitmaps, chip_ids)
    data, x_edges, y_edges = sensor.hitmap_histogram()
    assert len(data) == len(x_edges)-1
    assert len(data[0]) == len(y_edges)-1


def test_hist_dimensions():
    """ test the compatibility of the dimensions of the axes hist with
    that of the 2d histogram of the sensor """
    roc_hitmaps = [np.random.randint(0, 255, (100, 50)) for _ in range(16)]
    # hits = sum(np.array(roc_hitmaps).flatten())
    chip_ids = range(16)
    sensor = Sensor(roc_hitmaps, chip_ids)
    x_hist, x_edges = sensor.histogram('x')
    y_hist, y_edges = sensor.histogram('y')
    data, x_edges_2d, y_edges_2d = sensor.hitmap_histogram()
    assert len(x_edges) == len(x_edges_2d)
    assert len(y_edges) == len(y_edges_2d)
    for edge, edge_2d in zip(x_edges, x_edges_2d):
        assert np.round(edge, 3) == np.round(edge_2d, 3)
    for edge, edge_2d in zip(y_edges, y_edges_2d):
        assert np.round(edge, 3) == np.round(edge_2d, 3)
    for row, dsum in zip(data, x_hist):
        assert sum(row) == dsum
    for row, dsum in zip(data.T, y_hist):
        assert sum(row) == dsum


if __name__ == "__main__":
    test_generate_x_histogram()
    test_gen_2d_hist()
