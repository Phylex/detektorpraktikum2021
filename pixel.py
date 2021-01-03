"""
Module that contains the data structures that represent the sensors and pixels
with the input and output code to generate the data formats for plotting with
matplotlib and calculations with numpy
"""
import numpy as np
import itertools as itt

Point = tuple[float, float]


class Sensor():
    """Class representing one entire sensor assembly

    This class represents an entire sensor consisting of the 16 read
    out chips and some meta information.
    Two sensors are used stacked on top of each other to build the
    muon telescope that should be analysed in the laboratory
    """
    def __init__(self, hitmaps: list[np.ndarray], chip_id: list[int],
                 offset: tuple[np.ndarray, np.ndarray] = None):
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
        self.rocs = []
        for hmap, index in zip(hitmaps, chip_id):
            self.rocs.append(ReadoutChip(hmap, index))
        if offset is not None:
            self.orientation_matrix = offset[1]
            self.offset = offset[0]
        else:
            self.orientation_matrix = np.array([[1, 0], [0, 1]])
            self.offset = np.array([0, 0])

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
        return self.rocs[index[0]][index[1:]]

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

    def hitmap_histogram(self, density: bool = False) -> np.ndarray:
        """generates a 2D hitmap of the sensor
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
        x, y, w = zip(*hits)
        return np.histogram2d(x, y, bins=[x_borders, y_borders],
                              weights=w, density=density)


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
            x_offset = (8 * roc_dim[0]) - ((chip_nr % 8) * roc_dim[0])
            y_offset = 2 * roc_dim[1]
            self.offset = (x_offset, y_offset)
            self.orientation_matrix = np.array([[1, 0], [0, -1]]) @ \
                np.array([[-1, 0], [0, 1]])
        else:
            y_offset = 0
            x_offset = roc_dim[0] * chip_nr
            self.offset = (x_offset, y_offset)
            self.orientation_matrix = np.array([[1, 0], [0, 1]])
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
                pixel_row.append(Pixel(hitcount, dx, dy, (px, py)))
            self.pixels.append(pixel_row)
        self.pixels = np.array(self.pixels, dtype=object)

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
            self.pixels[pixel_index[0]][pixel_index[1]] = Pixel(
                    pixel_hitcount[0],
                    curpix.width,
                    curpix.height,
                    (curpix.center[0] - curpix.width/2,
                        curpix.center - curpix.height/2),
                    pixel_hitcount[1])
        except TypeError:
            curpix = self.pixels[pixel_index[0]][pixel_index[1]]
            self.pixels[pixel_index[0]][pixel_index[1]] = Pixel(
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

    def get_pixel_borders(self, axis: str):
        """ get the sensor local borders of the readout chip (this uses
        the generate_sensor_coordinates function
        """
        pixels = self.pixels.flatten()
        if axis == 'x':
            borders = np.array([pixel.borders()[0]
                               for pixel in pixels]).flatten()
            borders *= self.orientation_matrix[0, 0]
            borders += self.offset[0]
        elif axis == 'y':
            borders = np.array([pixel.borders()[1]
                               for pixel in pixels]).flatten()
            borders *= self.orientation_matrix[1, 1]
            borders += self.offset[1]
        else:
            raise ValueError("axis needs to be either 'x' or 'y'")
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
            local_position = self.pixels[i][j].get_position()
            global_position = (self.orientation_matrix @ local_position) \
                + self.offset
            if normalized:
                hitmap.append((global_position,
                               self.pixels[i][j].get_normalized_hitcount()))
            else:
                hitmap.append((global_position,
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
            for x in itt.repeat(elem[0], int(elem[1])):
                hits.append(x)
        return hits


class Pixel():
    def __init__(self, hitcount: int, x_dim: float, y_dim: float,
                 bottom_left_corner_coord: Point, q_val=None):
        self.hits = hitcount
        self.width = x_dim
        self.height = y_dim
        self.center = (bottom_left_corner_coord[0] + x_dim/2,
                       bottom_left_corner_coord[1] + y_dim/2)
        self.q_val = q_val

    def get_normalized_hitcount(self):
        """ get the hits per pixel area

        This method corrects the hitcount for the different sizes of the pixels
        """
        return self.hits / (self.width * self.height)

    def borders(self):
        """ get the x and y borders of the pixel """
        return np.array([(self.center[0] - self.width/2,
                         self.center[0] + self.width/2),
                         (self.center[1] - self.height/2,
                         self.center[1] + self.height/2)])

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

def test_generate_x_histogram():
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
    roc_hitmaps = [np.random.randint(0, 255, (100, 50)) for _ in range(16)]
    # hits = sum(np.array(roc_hitmaps).flatten())
    chip_ids = range(16)
    sensor = Sensor(roc_hitmaps, chip_ids)
    data, x_edges, y_edges = sensor.hitmap_histogram()
    assert len(data) == len(x_edges)-1
    assert len(data[0]) == len(y_edges)-1

def test_hist_dimensions():
    roc_hitmaps = [np.random.randint(0, 255, (100, 50)) for _ in range(16)]
    # hits = sum(np.array(roc_hitmaps).flatten())
    chip_ids = range(16)
    sensor = Sensor(roc_hitmaps, chip_ids)
    x_hist, x_edges = sensor.histogram('x')
    y_hist, y_edges = sensor.histogram('y')
    data, x_edges_2D, y_edges_2D = sensor.hitmap_histogram()
    for e, e2d in zip(x_edges, x_edges_2D):
        assert np.round(e, 3) == np.round(e2d, 3)
    for edge, edge_2d in zip(y_edges, y_edges_2D):
        assert np.round(edge, 3) == np.round(edge_2d, 3)
    for row, dsum in zip(data, x_hist):
        assert sum(row) == dsum
    for row, dsum in zip(data.T, y_hist):
        assert sum(row) == dsum


if __name__ == "__main__":
    test_generate_x_histogram()
    test_gen_2d_hist()
