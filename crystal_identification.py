import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.feature import peak_local_max


def get_fourier_transform_of_slice(region):
    # returns the fourier transform of a region
    fourier = np.log10(np.fft.fftshift(abs(np.fft.fft2(region))))
    return fourier


def crop_image(image, cut_size):
    # crops an array
    cut_high = (1 - cut_size)
    x_min = int(image.shape[0] * cut_size)
    x_max = int(image.shape[0] * cut_high)
    y_min = int(image.shape[1] * cut_size)
    y_max = int(image.shape[1] * cut_high)
    cropped_image = image[x_min:x_max, y_min:y_max]
    return cropped_image


def findmaxima(fourier_central, threshold, min_distance):
    # finds the maxima within the cropped fourier transform
    maxima = peak_local_max(fourier_central, min_distance=min_distance, threshold_rel=threshold)
    middlepixel = np.round(np.shape(fourier_central)[0] / 2.)
    findmiddle = np.where((maxima == (middlepixel, middlepixel)).all(axis=1))
    # fourier always has a maximum at the central pixel, which we remove from the results
    if findmiddle[0].shape[0] > 0:
        maxima = np.delete(maxima,
                           np.where((maxima == (middlepixel, middlepixel)).all(axis=1))[0][0],
                           axis=0)
    return maxima


def ringintensity(fourier_cut, maxima, number_ring_samples):
    # calculates the relative intensity of the discovered peaks to the sampled intensity at the same distance from the centre of the fourier transform. Gives us a way of checking for "liquidity"
    middlepixel = int(np.round(np.shape(fourier_cut)[0] / 2.))
    # average peak height
    peakheight = np.average(fourier_cut[(maxima[:, 0], maxima[:, 1])])
    # average distance between center and peaks, this will be our ring radius
    r = np.average(
        ((((maxima - middlepixel)[:, 0]) ** 2) + (((maxima - middlepixel)[:, 1]) ** 2)) ** 0.5)
    # generate our random angles along the ring for sampling
    theta = np.random.random(number_ring_samples) * np.pi * 2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    # create an array with the sampling positions
    positions_in_fourier = np.concatenate((np.split(x, x.shape[0]), np.split(y, y.shape[0])),
                                          axis=1) + middlepixel
    # take the intensity samples
    intensities = fourier_cut[(positions_in_fourier[:, 0], positions_in_fourier[:, 1])]
    intensityratio = np.average(intensities) / peakheight
    return intensityratio


def scanfourier(original_image, threshold, size_of_scan_box, ring_threshold):
    # scans a box along the original image, and uses the foruier transform in that box to assign a number to whether it's a crystal or a liquid

    rastering_interval = 4  # Rastering size
    # when we take fourier transform of the scan box, crop it to highlight crystal peaks
    fourier_cut_low = 0.3

    num_x_rasters = int((original_image.shape[0] - size_of_scan_box) / rastering_interval)
    num_y_rasters = int((original_image.shape[1] - size_of_scan_box) / rastering_interval)
    results_array = np.empty((num_x_rasters, num_y_rasters))

    for lowestx in np.arange(0, original_image.shape[0] - size_of_scan_box, rastering_interval):
        # scanning over input image
        for lowesty in np.arange(0, original_image.shape[1] - size_of_scan_box, rastering_interval):
            # take the relevant slice
            image_subsection = original_image[lowestx:lowestx + size_of_scan_box, lowesty:lowesty + size_of_scan_box]
            ft_of_subsection = get_fourier_transform_of_slice(image_subsection)
            cropped_ft_of_subsection = crop_image(ft_of_subsection, fourier_cut_low)

            min_distance = int(ft_of_subsection.shape[0] / 20.)
            maxima = findmaxima(cropped_ft_of_subsection, threshold, min_distance)

            subsection_index = [int(lowestx / rastering_interval), int(lowesty / rastering_interval)]
            # if we've found some peaks
            if maxima.shape[0] > 0:
                # calculate whether we have a ring
                ring_intensity = ringintensity(cropped_ft_of_subsection, maxima, 100)
                if maxima.shape[0] == 6:
                    # clear hexagonal crystal
                    if ring_intensity < ring_threshold:
                        results_array[subsection_index[0], subsection_index[1]] = 6
                    else:
                        # potential boundary between hex+liquid
                        results_array[subsection_index[0], subsection_index[1]] = 1
                elif maxima.shape[0] == 4:
                    if ring_intensity < ring_threshold:  # clear cross crystal
                        results_array[subsection_index[0], subsection_index[1]] = 4
                    else:
                        # potential boundary between cross+liquid
                        results_array[
                            subsection_index[0], subsection_index[1]] = 0.5
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


def main():
    filename = 'sample_image.tif'
    im = pl.imread(filename)[:, :, 0]

    array = scanfourier(im, 0.5, 128, 0.9)

    plot_result(array)


if __name__ == '__main__':
    main()
