import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.feature import peak_local_max


def fourieronregion(region):
    # returns the fourier transform of a region
    fourier = np.log10(np.fft.fftshift(abs(np.fft.fft2(region))))
    return fourier


def cutfourier(fourier, fourier_cut_low):
    # crops the fourier transform to region that contains peaks, for peak detection
    fourier_cut_high = (1 - fourier_cut_low)
    fourier_central = fourier[
                      fourier.shape[0] * fourier_cut_low:fourier.shape[0] * fourier_cut_high,
                      fourier.shape[1] * fourier_cut_low:fourier.shape[1] * fourier_cut_high]
    return fourier_central


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


def scanfourier(im, threshold, size_scan_box,
                ring_threshold):  # scans a box along the original image, and uses the foruier transform in that box to assign a number to whether it's a crystal or a liquid

    size_of_small_box = 4
    size_of_scan_box = size_scan_box
    # when we take fourier transform of the scan box, crop it to highlight crystal peaks
    fourier_cut_low = 0.3

    dimension_results_arr = (im.shape[0] - size_of_scan_box) / size_of_small_box
    results_array = np.empty((dimension_results_arr, dimension_results_arr))

    for lowestx in np.arange(0, im.shape[0] - size_of_scan_box, size_of_small_box):
        # scanning over input image
        for lowesty in np.arange(0, im.shape[0] - size_of_scan_box, size_of_small_box):
            # take the relevant slice
            scan_box = im[lowestx:lowestx + size_of_scan_box, lowesty:lowesty + size_of_scan_box]
            fourier = fourieronregion(scan_box)
            fourier_central = cutfourier(fourier, fourier_cut_low)

            min_distance = fourier.shape[0] / 20.
            maxima = findmaxima(fourier_central, threshold, min_distance)

            # if we've found some peaks
            if maxima.shape[0] > 0:
                # calculate whether we have a ring
                ring_intensity = ringintensity(fourier_central, maxima, 100)
                if maxima.shape[0] == 6:
                    # clear hexagonal crystal
                    if ring_intensity < ring_threshold:
                        results_array[lowestx / size_of_small_box, lowesty / size_of_small_box] = 6
                    else:
                        # potential boundary between hex+liquid
                        results_array[lowestx / size_of_small_box, lowesty / size_of_small_box] = 1
                elif maxima.shape[0] == 4:
                    if ring_intensity < ring_threshold:  # clear cross crystal
                        results_array[lowestx / size_of_small_box, lowesty / size_of_small_box] = 4
                    else:
                        # potential boundary between cross+liquid
                        results_array[
                            lowestx / size_of_small_box, lowesty / size_of_small_box] = 0.5
                else:
                    results_array[lowestx / size_of_small_box, lowesty / size_of_small_box] = 0
            else:
                results_array[lowestx / size_of_small_box, lowesty / size_of_small_box] = 0

    return results_array


def main():
    filename = 'sample_image.tif'
    im = pl.imread(filename)[:, :, 0]

    array = scanfourier(im, 0.5, 128, 0.9)

    ###plotting###
    cmap = matplotlib.colors.ListedColormap(
        ['darkblue', 'skyblue', 'cadetblue', 'limegreen', 'orangered'])
    # plot, 6 = hexagonal crystal, 4 = cross crystal 1.0 = hex+liquid ring, 0.5 = cross + liquid ring, 0.0 = liquid (neither 4 nor 6 peaks detected)
    bounds = [-0.4, 0.4, 0.6, 1.4, 5.9, 6.1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.imshow(array, interpolation='nearest', cmap=cmap, norm=norm)
    plt.colorbar(fig, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 0.5, 1, 4, 6])
    plt.show()
