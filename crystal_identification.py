import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.feature import peak_local_max


def get_fourier_transform_of_slice(image):
    """
    Returns the real valued Fourier transform of an image
    :param image: the image to apply the Fourier transform to
    :return: The Fourier transform of the image
    """
    fft = np.fft.fft2(image)
    power_spectrum = get_2d_power_spectrum(fft)
    return np.fft.fftshift(power_spectrum)


def get_2d_power_spectrum(image):
    """
    Return the power specrum of a fourier tranform
    :param image: The real and complex fourier transform
    :return: The power spectrum of the Fourier transform
    """
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
    :param minimum_peak_separation_distance: The minimum seperation of detected peaks
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
    num_maxima = len(maxima)
    # We only care if there might be 4 or 6 crystal maxima
    if 3 < num_maxima < 8:
        central_coordinate = get_center_of_image(image.shape)
        revised_maxima = [peak for peak in list(maxima) if np.all(peak != central_coordinate)]
        return np.array(revised_maxima)
    else:
        return maxima


def get_ring_intensity(image, maxima, pixel_distances, center_coordinate):
    """
    Calculates the relative intensity of the discovered peaks and the sampled intensity at
    the same distance from the centre of the fourier transform.
    Gives us a way of checking for "liquidity"
    :param image: An ndarray represnting an image
    :param maxima: a list of tuples representing coordinates of maxima
    :param pixel_distances: An array giving the distance of each pixel from the center of the image
    :param center_coordinate: The coordinate of the center of image
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


def get_center_of_image(image_shape):
    """
    Return the coordinates representing the center of an image
    :param image_shape: An tuple representing the size of an image
    :return: A tuple representing the coordinates of the center of image
    """
    center_coordinate = np.rint(np.array(image_shape) / 2.0)
    return center_coordinate


def scanfourier(original_image, minimum_intensity_threshold, size_of_scan_box, ring_threshold, rastering_interval, image_crop_factor):
    """
    The main loop. Raster over an image and determine whether the subsections are crystalline.
    :param original_image: The original image to process
    :param minimum_intensity_threshold: The minimum intensity of a peak in fourier transform
    :param size_of_scan_box: The size of the region which is Fourier transformed
    :param ring_threshold: The threshold defining the boundary between crystal and liquid regions
    :param rastering_interval: The interval in pixels over which to raster the Fourier transform
    :param image_crop_factor: A float giving the proportion of the image to cut off each side
    :return:
    """

    cropped_center, minimum_peak_separation_distance, pixel_distances = setup_fourier_scan(image_crop_factor, size_of_scan_box)

    num_x_rasters = int((original_image.shape[0] - size_of_scan_box) / rastering_interval)
    num_y_rasters = int((original_image.shape[1] - size_of_scan_box) / rastering_interval)
    results_array = np.zeros((num_x_rasters, num_y_rasters))

    for x_bin in range(num_x_rasters):
        for y_bin in range(num_y_rasters):
            ft_of_subimage = get_ft_of_subimage(image_crop_factor, original_image, rastering_interval, size_of_scan_box, x_bin, y_bin)
            with np.errstate(divide='ignore'):
                ft_of_subimage = np.log10(ft_of_subimage)

            maxima = find_maxima_in_image(ft_of_subimage, minimum_intensity_threshold, minimum_peak_separation_distance)
            crystal_type = classify_image_region(ft_of_subimage, maxima, ring_threshold, pixel_distances, cropped_center)
            results_array[x_bin, y_bin] = crystal_type

    return results_array


def setup_fourier_scan(image_crop_factor, size_of_scan_box):
    """
    Initialise some values to be used in finding and classifying maxima
    :param image_crop_factor: A float specifying how much to crop off each side of the fourier transformed image
    subsection prior to identifying peaks
    :param size_of_scan_box: The size of the subimage which is Fourier transformed
    :return: cropped_center - The coordinates of the center of the cropped box
    :return: minimum_peak_separation_distance - The minimum separation in pixels of identified peaks in the fourier transform
    :return: pixel_distances - An array giving the distance of each pixel from the center of the image
    """
    minimum_peak_separation_distance = int(size_of_scan_box / 20.)
    x_min = int(size_of_scan_box * image_crop_factor)
    x_max = int(size_of_scan_box * (1 - image_crop_factor))
    cropped_image_size = x_max - x_min
    cropped_center = get_center_of_image([cropped_image_size, cropped_image_size])

    binsize = 1
    y, x = np.indices([cropped_image_size, cropped_image_size])
    pixel_distances = np.hypot(x - cropped_center[0], y - cropped_center[1]) / binsize

    return cropped_center, minimum_peak_separation_distance, pixel_distances


def get_ft_of_subimage(image_crop_factor, original_image, rastering_interval, size_of_scan_box, x_bin, y_bin):
    """
    Get a subsection of an image, get the Fourier transform then crop the image
    :param image_crop_factor: A float giving the proportion of the image to cut off each side
    :param original_image: An ndarray representing the original image
    :param rastering_interval: The interval in pixels between rastered images
    :param size_of_scan_box: The size of the sub-image used for the Fourier transform
    :param x_bin: The bin number of the current sub-image
    :param y_bin: The bin number of the current sub-image
    :return: The cropped Fourier transform of the image subection
    """
    subimage_minima = np.array((x_bin * rastering_interval, y_bin * rastering_interval))
    subimgae_maxima = subimage_minima + size_of_scan_box
    subimage = original_image[subimage_minima[0]:subimgae_maxima[0], subimage_minima[1]:subimgae_maxima[1]]
    ft_of_subimage = get_fourier_transform_of_slice(subimage)
    cropped_ft_of_subimage = crop_image(ft_of_subimage, image_crop_factor)

    return cropped_ft_of_subimage


def classify_image_region(image, maxima, ring_threshold, pixel_distances, cropped_center):
    """
    Classify the subsection of the image depending on the number of maxima there are
    :param image: An ndarray representing an image
    :param maxima: ndarray of coordinates corresponding to peaks in the image
    :param ring_threshold: A peak intesnity threshold above which a crystal is identified
    :param pixel_distances: An array giving the distance of each pixel from the center of the image
    :param cropped_center: The coordinate of the center of image
    """
    number_of_maxima = maxima.shape[0]
    if number_of_maxima == 0:
        # No maxima so just a liquid
        return 0
    else:

        if number_of_maxima == 6:
            ring_intensity = get_ring_intensity(image, maxima, pixel_distances, cropped_center)
            if ring_intensity < ring_threshold:
                # clear hexagonal crystal
                return 6
            else:
                # potential boundary between hex+liquid
                return 1
        elif number_of_maxima == 4:
            ring_intensity = get_ring_intensity(image, maxima, pixel_distances, cropped_center)
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
    """
    Convert the result to a 2D image representation
    :param array: The result of the analysis
    """
    cmap = matplotlib.colors.ListedColormap(['darkblue', 'skyblue', 'cadetblue', 'limegreen', 'orangered'])
    # plot, 6 = hexagonal crystal, 4 = cross crystal 1.0 = hex+liquid ring, 0.5 = cross + liquid ring, 0.0 = liquid (neither 4 nor 6 peaks detected)
    bounds = [-0.4, 0.4, 0.6, 1.4, 5.9, 6.1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.imshow(array, interpolation='nearest', cmap=cmap, norm=norm)
    plt.colorbar(fig, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 0.5, 1, 4, 6])
    plt.savefig("crystal_measure.png")


def load_image(filename):
    """
    Open an image as an array
    :param filename: The path of the image to open
    :return: An ndarray of the image
    """
    image = plt.imread(filename)
    if len(image.shape) == 3:
        image = image[:, :, 0]
    return image


def main():

    filename = 'sample_image.tif'
    rastering_interval = 2
    ring_threshold = 0.9
    size_of_scan_box = 128
    minimum_peak_intensity_threshold = 0.5
    image_crop_factor = 0.35

    image = load_image(filename)
    array = scanfourier(image, minimum_peak_intensity_threshold, size_of_scan_box, ring_threshold, rastering_interval, image_crop_factor)

    plot_result(array)


if __name__ == '__main__':
    main()
