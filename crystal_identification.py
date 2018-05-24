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
    fft = np.fft.fft2(image)
    power_spectrum = get_2d_power_spectrum(fft)
    return np.fft.fftshift(power_spectrum)


def get_2d_power_spectrum(image):
    return image.real ** 2 + image.imag ** 2


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
    The Fourier transform always has a peak in the middle.
    We are not interested in this peak so remove it from the list of maxima.
    :param image: ndarray of the real valued Fourier transform of an image
    :param maxima: ndarray of coordinates corresponding to peaks in the image
    :return: ndarray corresponding to coordinates of peaks in the image
    """
    central_coordinate = get_center_of_image(image)
    revised_maxima = [peak for peak in list(maxima) if np.all(peak != central_coordinate)]
    return np.array(revised_maxima)


def get_ring_intensity(image, maxima, pixel_distances, center_coordinate):
    """
    Calculates the relative intensity of the discovered peaks and the sampled intensity at
    the same distance from the centre of the fourier transform.
    Gives us a way of checking for "liquidity"
    :param image: An ndarray represnting an image
    :param maxima: a list of tuples representing coordinates of maxima
    :param pixel_distances: An array giving the distance of each pixel from the center of the image
    :return: The ratio between the intenisty of the peaks and the radial average of the intensity at the same
    distance from the center as the peaks
    """
    average_peak_intensity = np.average(image[(maxima[:, 0], maxima[:, 1])])

    # average distance between center and peaks, this will be our ring radius
    average_distance = np.average(np.linalg.norm((maxima - center_coordinate), axis=1))

    mask = np.logical_and(pixel_distances > average_distance - 0.5, pixel_distances < average_distance + 0.5)
    ring_intensity = np.average(image[mask])

    intensityratio = np.average(ring_intensity) / average_peak_intensity
    return intensityratio


def get_center_of_image(image):
    """
    Return the coordinates representing the center of an image
    :param image: An ndarray representing an image
    :return: A tuple representing the coordinates of the center of image
    """
    image_shape = np.array(np.shape(image))
    center_coordinate = np.rint(image_shape / 2.0)
    return center_coordinate


def setup_radial_average(image, cropped_center):
    binsize = 1
    y, x = np.indices(image.shape)
    r = np.hypot(x - cropped_center[0], y - cropped_center[1]) / binsize
    return r


def scanfourier(original_image, threshold, size_of_scan_box, ring_threshold, rastering_interval, image_crop_factor):
    # scans a box along the original image, and uses the foruier transform in that box to assign a number to whether it's a crystal or a liquid

    minimum_peak_separation_distance = pixel_distances = cropped_center = 0

    num_x_rasters = int((original_image.shape[0] - size_of_scan_box) / rastering_interval)
    num_y_rasters = int((original_image.shape[1] - size_of_scan_box) / rastering_interval)
    results_array = np.zeros((num_x_rasters, num_y_rasters))

    for x_bin in range(num_x_rasters):
        for y_bin in range(num_y_rasters):
            subsection_minima = np.array((x_bin * rastering_interval, y_bin * rastering_interval))
            subsection_maxima = subsection_minima + size_of_scan_box
            image_subsection = original_image[subsection_minima[0]:subsection_maxima[0],
                                              subsection_minima[1]:subsection_maxima[1]]
            ft_of_subsection = get_fourier_transform_of_slice(image_subsection)
            cropped_ft_of_subsection = crop_image(ft_of_subsection, image_crop_factor)
            with np.errstate(divide='ignore'):
                cropped_ft_of_subsection = np.log10(cropped_ft_of_subsection)
            if x_bin == 0 and y_bin == 0:
                cropped_center = get_center_of_image(cropped_ft_of_subsection)
                minimum_peak_separation_distance = int(ft_of_subsection.shape[0] / 20.)
                pixel_distances = setup_radial_average(cropped_ft_of_subsection, cropped_center)
            maxima = find_maxima_in_image(cropped_ft_of_subsection, threshold, minimum_peak_separation_distance)

            crystal_type = classify_image_region(cropped_ft_of_subsection, maxima, ring_threshold, pixel_distances, cropped_center)
            results_array[x_bin, y_bin] = crystal_type

    return results_array


def classify_image_region(image, maxima, ring_threshold, r, cropped_center):
    """
    Classify the subsection of the image depending on the number of maxima there are
    :param image: An ndarray representing an image
    :param maxima: ndarray of coordinates corresponding to peaks in the image
    :param ring_threshold: A peak intesnity threshold above which a crystal is identified
    """
    number_of_maxima = maxima.shape[0]
    if number_of_maxima == 0:
        # No maxima so just a liquid
        return 0
    else:

        if number_of_maxima == 6:
            ring_intensity = get_ring_intensity(image, maxima, r, cropped_center)
            if ring_intensity < ring_threshold:
                # clear hexagonal crystal
                return 6
            else:
                # potential boundary between hex+liquid
                return 1
        elif number_of_maxima == 4:
            ring_intensity = get_ring_intensity(image, maxima, r, cropped_center)
            if ring_intensity < ring_threshold:
                # clear cross crystal
                return 4
            else:
                # potential boundary between cross+liquid
                return 0.5
        else:
            # No maxima so just a liquid
            return 0


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
    rastering_interval = 1
    ring_threshold = 0.9
    size_of_scan_box = 128
    threshold = 0.5
    image_crop_factor = 0.3

    image = load_image(filename)
    array = scanfourier(image, threshold, size_of_scan_box, ring_threshold, rastering_interval, image_crop_factor)

    plot_result(array)


if __name__ == '__main__':
    main()
