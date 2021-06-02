import numpy as np


def stretch(img, intensity_lvls=256, quantile = 2):
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

    stretched_image = (workimg - min_intensity_input) * (max_intensity_output - min_intensity_output) /\
                      (max_intensity_input - min_intensity_input)

    return stretched_image



def gaussian_kernel(length = 5, sigma = 1):
    """
    The function returns a square shaped Gaussian filter mask of desired size and standard deviation.
    Only odd values for the filter size are possible.

    :param length: Side length of the gaussian kernel (only odd numbers)
    :param sigma: Standard deviation of the gaussian distribution
    :return: Square shaped Gaussian kernel
    """

    if length % 2 == 0:
        raise Exception('Only odd numbers!')

    N = length//2
    kernel_1D = np.linspace(-N, N, length)
    xx, yy = np.meshgrid(kernel_1D, kernel_1D)
    kernel_2D = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel_normalized = kernel_2D / np.sum(kernel_2D)

    return kernel_normalized



def convolution(image, kernel):
    """
    This function takes an image and an filter mask as Input and returns the filtered image.
    Only square shaped filter masks are possible

    :param image: Input Image
    :param kernel: Filter mask
    :return: Filtered Image
    """

    N = np.shape(kernel)[0]//2
    padded_picture = np.pad(image, (N, N), 'reflect')
    filtered_image = np.zeros(padded_picture.shape)
    for p in np.ndindex(padded_picture.shape):
        if p[0] >= N and p[0] < padded_picture.shape[0] - N and p[1] >= N and p[1] < padded_picture.shape[1] - N:
            neighborhood = padded_picture[p[0] - N:p[0] + N + 1, p[1] - N:p[1] + N + 1]
            neighborhood_array = neighborhood * kernel
            filtered_image[p] = np.sum(neighborhood_array)
    final_image = filtered_image[N:filtered_image.shape[0] - N, N:filtered_image.shape[1] - N]
    return final_image



r'''
png = imread(r'..\Data\NIH3T3\im\dna-0.png')
image_test_tif = imread(r'..\Data\N2DH-GOWT1\img\t01.tif')

stretchy = stretch(image_test_png, 256)

plt.imshow(image_test_png, 'gray')
plt.show()

plt.imshow(stretchy, 'gray')
plt.show()

import matplotlib.pyplot as plt

from skimage.io import imread

x = gaussian_kernel(21,10)
png = imread(r'..\Data\NIH3T3\img\dna-0.png')
conv = convolution(png,x)

print(conv.shape)
print(png.shape)
'''