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

r'''
png = imread(r'..\Data\NIH3T3\im\dna-0.png')
image_test_tif = imread(r'..\Data\N2DH-GOWT1\img\t01.tif')

stretchy = stretch(image_test_png, 256)

plt.imshow(image_test_png, 'gray')
plt.show()

plt.imshow(stretchy, 'gray')
plt.show()
'''