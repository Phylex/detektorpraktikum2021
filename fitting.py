""" module that holds all neccesary functions to do the fitting """
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm

def fit_gauss_normalized_histogram(hist: np.ndarray, bins: np.ndarray):
    """ fits a gauss curve to a histogram

    Function that fits a gaussian to a histogram, the histogram has to be
    normalized

    Parameters
    ----------
    hist : np.ndarray
        histogram of the data
    bins : np.ndarray
        edges of the histogram bins in ascending order

    Returns
    -------
    popt : tuple of floats
        mean and sigma for the fitted Gaussian
    pcov: 2D Array
        covariance matrix for the fit parameters
    """
    bin_centers = (bins[:-1] + bins[1:])/2
    sigma_start = 30 * np.mean(bins[1:] - bins[:-1])
    mu_start = np.median(bin_centers)
    popt, pcov = opt.curve_fit(norm.pdf, bin_centers, hist,
                               p0=(mu_start, sigma_start))
    return popt, pcov


def gauss_2D(mux, muy, sigx, sigy):
    """produce a parametrized function of a normalized 2D gaussian distribution

    The output of this function is again a function that now is configured
    according to the parameters given to this function.

    Parameters
    ----------
    mux: float
        the mean of the gaussian in x direction
    muy: float
        the mean of the gaussian in y direction
    sigx: float
        the standard deviation of the distribution in x direction
    sigy: float
        the standard deviation of the distribution in y direction

    Returns
    -------
    f : callable
        2D gaussian function that calcualtes the height of the parametrized
        gaussian at the coordinate x, y
    """
    return lambda x, y: 1/(2*np.pi*sigx*sigy)*np.exp(- (x - mux) ** 2 /
                                                     (2 * sigx ** 2) -
                                                     (y - muy) ** 2 /
                                                     (2 * sigy ** 2))


def fit_2D_gauss(hitmap, hitmap_coordinates, pstart):
    """ Fits a 2D Gaussian onto a 2D histogram """
    data_x = hitmap_coordinates[0].flatten()
    data_y = hitmap_coordinates[1].flatten()
    n_hitmap = normalize_hitmap_volume(hitmap, hitmap_coordinates)
    data_z = n_hitmap.flatten()
    errfunc = lambda p: gauss_2D(*p)(data_x, data_y) - data_z
    opt_min = opt.least_squares(errfunc, pstart)
    return opt_min
