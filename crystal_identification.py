import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.feature import peak_local_max


def get_fourier_transform_of_slice(image):
    """
    Returns the real valued Fourier transform of an image
    :param image: the image to apply the Fourier transform to
    :return: The Fourier transofrm of the image
    """
    return np.fft.fftshift(abs(np.fft.fft2(image)))


def crop_image(image, cut_size):
    """
    Returns a subsection of an array representing an image
    :param image: The image as a numpy array
    :param cut_size: The fraction to cut from each side of the image
    :return: The cropped image as a numpy array
    """
    cut_high = (1 - cut_size)
    x_min = int(image.shape[0] * cut_size)
    x_max = int(image.shape[0] * cut_high)
    y_min = int(image.shape[1] * cut_size)
    y_max = int(image.shape[1] * cut_high)
    cropped_image = image[x_min:x_max, y_min:y_max]
    return cropped_image


def find_maxima_in_image(image, minimum_intensity_threshold, minimum_peak_separation_distance):
    """
    Finds peaks within an image
    :param image: The original image
    :param minimum_intensity_threshold: The minimum value of a peak relative to the maximum intensity of the image
    :return: ndarray of coordinates corresponding to peaks in the image
    """
    maxima = peak_local_max(image, min_distance=minimum_peak_separation_distance, threshold_rel=minimum_intensity_threshold)
    maxima = remove_central_peak(image, maxima)

    return maxima


def remove_central_peak(image, maxima):
    """
    The Fourier transform always has a peak in the middle. We are not interested in this peak so remove it.
    :param image: ndarray of the real valued Fourier transform of an image
    :param maxima: ndarray of coordinates corresponding to peaks in the image
    :return: ndarray of coordinates corresponding to peaks in the image with the central peak removed
    """

    central_x_pixel = np.round(np.shape(image)[0] / 2.)
    central_y_pixel = np.round(np.shape(image)[1] / 2.)
    middle_mask = np.where((maxima == (central_x_pixel, central_y_pixel)).all(axis=1))
    maxima = np.delete(maxima, middle_mask, axis=0)

    return maxima


def ringintensity(image, maxima):
    """
    # Calculates the relative intensity of the discovered peaks to the sampled intensity at
    the same distance from the centre of the fourier transform.
    Gives us a way of checking for "liquidity"
    :param image:
    :param maxima:
    :return:
    """
    num_ring_samples = 100
    middlepixel = int(np.round(np.shape(image)[0] / 2.))
    # average peak height
    peakheight = np.average(image[(maxima[:, 0], maxima[:, 1])])
    # average distance between center and peaks, this will be our ring radius
    r = np.average(
        ((((maxima - middlepixel)[:, 0]) ** 2) + (((maxima - middlepixel)[:, 1]) ** 2)) ** 0.5)
    # generate our random angles along the ring for sampling
    theta = np.random.random(num_ring_samples) * np.pi * 2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    # create an array with the sampling positions
    positions_in_fourier = np.concatenate((np.split(x, x.shape[0]), np.split(y, y.shape[0])),
                                          axis=1) + middlepixel
    # take the intensity samples
    intensities = image[(positions_in_fourier[:, 0], positions_in_fourier[:, 1])]
    intensityratio = np.average(intensities) / peakheight
    return intensityratio


def scanfourier(original_image, threshold, size_of_scan_box, ring_threshold, rastering_interval, image_crop_factor):
    # scans a box along the original image, and uses the foruier transform in that box to assign a number to whether it's a crystal or a liquid

    num_x_rasters = int((original_image.shape[0] - size_of_scan_box) / rastering_interval)
    num_y_rasters = int((original_image.shape[1] - size_of_scan_box) / rastering_interval)
    results_array = np.empty((num_x_rasters, num_y_rasters))

    for lowestx in np.arange(0, original_image.shape[0] - size_of_scan_box, rastering_interval):
        # scanning over input image
        for lowesty in np.arange(0, original_image.shape[1] - size_of_scan_box, rastering_interval):

            image_subsection = original_image[lowestx:lowestx + size_of_scan_box, lowesty:lowesty + size_of_scan_box]
            ft_of_subsection = get_fourier_transform_of_slice(image_subsection)
            ft_of_subsection = np.log10(ft_of_subsection)
            cropped_ft_of_subsection = crop_image(ft_of_subsection, image_crop_factor)
            minimum_peak_separation_distance = int(ft_of_subsection.shape[0] / 20.)
            maxima = find_maxima_in_image(cropped_ft_of_subsection, threshold, minimum_peak_separation_distance)
            number_of_maxima = maxima.shape[0]
            subsection_index = [int(lowestx / rastering_interval), int(lowesty / rastering_interval)]
            # if we've found some peaks
            if number_of_maxima > 0:
                # calculate whether we have a ring
                ring_intensity = ringintensity(cropped_ft_of_subsection, maxima)
                if number_of_maxima == 6:
                    # clear hexagonal crystal
                    if ring_intensity < ring_threshold:
                        results_array[subsection_index[0], subsection_index[1]] = 6
                    else:
                        # potential boundary between hex+liquid
                        results_array[subsection_index[0], subsection_index[1]] = 1
                elif number_of_maxima == 4:
                    if ring_intensity < ring_threshold:  # clear cross crystal
                        results_array[subsection_index[0], subsection_index[1]] = 4
                    else:
                        # potential boundary between cross+liquid
                        results_array[subsection_index[0], subsection_index[1]] = 0.5
                else:
                    results_array[subsection_index[0], subsection_index[1]] = 0
            else:
                results_array[subsection_index[0], subsection_index[1]] = 0

    return results_array


def plot_result(array):
    ###plotting###
    cmap = matplotlib.colors.ListedColormap(
        ['darkblue', 'skyblue', 'cadetblue', 'limegreen', 'orangered'])
    # plot, 6 = hexagonal crystal, 4 = cross crystal 1.0 = hex+liquid ring, 0.5 = cross + liquid ring, 0.0 = liquid (neither 4 nor 6 peaks detected)
    bounds = [-0.4, 0.4, 0.6, 1.4, 5.9, 6.1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.imshow(array, interpolation='nearest', cmap=cmap, norm=norm)
    plt.colorbar(fig, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 0.5, 1, 4, 6])
    plt.savefig("crystal_measure.png")


def load_image(filename):
    image = pl.imread(filename)[:, :, 0]
    return image


def main():

    filename = 'sample_image.tif'
    rastering_interval = 4
    ring_threshold = 0.9
    size_of_scan_box = 128
    threshold = 0.5
    image_crop_factor = 0.3

    image = load_image(filename)
    array = scanfourier(image, threshold, size_of_scan_box, ring_threshold, rastering_interval, image_crop_factor)

    plot_result(array)


if __name__ == '__main__':
    main()
