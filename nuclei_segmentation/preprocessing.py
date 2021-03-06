import matplotlib.pyplot as plt
import numpy as np
import math


def histogram_stretching(img, intensity_lvls=256, quantile=2):
    """
    This function peforms histogram stretching and ignores the outliers (below lower and above upper quantile).
    The minimum intensity of the stretched image is zero, the maximum intensity is 256 (.png) or 2**16 (.tif)

    :param quantile: Quantile (pixels above or under are cut out)
    :param img: Input image
    :param intensity_lvls: 256 (.png) or 2**16 (.tif)
    :return: Stretched Image
    """

    lower_quantile, upper_quantile = np.percentile(img, (quantile, 100 - quantile))

    workimg = img.copy()
    workimg[workimg < lower_quantile] = lower_quantile
    workimg[workimg > upper_quantile] = upper_quantile

    min_intensity_output = 0
    max_intensity_output = intensity_lvls
    min_intensity_input = int(np.min(workimg))
    max_intensity_input = int(np.max(workimg))

    stretched_image = (workimg - min_intensity_input) * (max_intensity_output - min_intensity_output) / \
                      (max_intensity_input - min_intensity_input)

    return stretched_image


def gaussian_kernel(length=5, sigma=1):
    """
    The function returns a square shaped gaussian filter mask of desired size and standard deviation.
    Only odd values for the filter size are possible.

    :param length: side length of the gaussian kernel (only odd numbers)
    :param sigma: standard deviation of the gaussian distribution
    :return: square shaped gaussian kernel
    """
    if length % 2 == 0:
        raise Exception('Only odd numbers!')

    border_width = length // 2
    kernel_1d = np.linspace(-border_width, border_width, length)
    xx, yy = np.meshgrid(kernel_1d, kernel_1d)
    kernel_2d = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    kernel_normalized = kernel_2d / np.sum(kernel_2d)

    return kernel_normalized


def convolution(image, kernel):
    """
    This function takes an image and an filter mask as Input and returns the filtered image.
    Only square shaped filter masks are possible

    :param image: Input Image
    :param kernel: filter mask
    :return: Filtered Image
    """

    kernel = np.fliplr(np.flipud(kernel))

    border_width = np.shape(kernel)[0] // 2

    padded_picture = np.pad(image, (border_width, border_width), 'reflect')

    filtered_image = np.zeros(padded_picture.shape)

    for p in np.ndindex(padded_picture.shape):

        if p[0] >= border_width and p[0] < padded_picture.shape[0] - border_width \
                and p[1] >= border_width and p[1] < padded_picture.shape[1] - border_width:
            neighborhood = padded_picture[p[0] - border_width:p[0] + border_width + 1,
                           p[1] - border_width:p[1] + border_width + 1]
            neighborhood_array = neighborhood * kernel
            filtered_image[p] = int(math.floor(np.sum(neighborhood_array)))

    final_image = filtered_image[border_width:filtered_image.shape[0] - border_width,
                  border_width:filtered_image.shape[1] - border_width]

    return final_image


def median_filter(img, filter_size=5):
    """
    This function applies median filter to a given image.
    To calculate the values for border pixels, the image array is extended by reflecting the values.

    :param img: Input image
    :param filter_size: The length of the square filter mask
    :return: Filtered image
    """

    workimg = img.copy()
    workimg = np.pad(workimg, pad_width=filter_size // 2, mode='reflect')
    workimg = np.array([np.median(workimg[x:x + filter_size, y:y + filter_size].flatten())
                        for x, y in np.ndindex(img.shape)])
    workimg = workimg.reshape(img.shape)

    return workimg


def binarize(img):
    """
    Transforms non-binary images to binary by setting values above 0 to 1.

    :param img: Image to be transformed
    :return: Transformed image
    """

    workimg = img.copy()
    workimg[img > 0] = 1
    return workimg
