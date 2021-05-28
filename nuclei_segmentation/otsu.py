import numpy as np
from skimage.io import imread
from skimage.io import imshow
from matplotlib import pyplot as plt


def otsu(image, intesity_lvls = 256):
    """
    This function takes an image and calculates the probability of class occurrence
    and the mean value for all pixels to calculate the threshold according to the formula
    of Otsu Thresholding.
    Also it calculates the total variance and uses it to calculate the goodness of the threshold.

    Source:
    Otsu, N. "A threshold selection method from gray-level histograms."
    IEEE Transactions on Systems, Man, and Cybernetics 9:1 (1979), pp 62-66.

    :param image: image out of Data
    :return: threshold and goodness of the image
    """

    img = image.copy().flatten()
    number_of_pixels = img.size
    class_probability = np.zeros(256)
    class_mean = np.zeros(256)
    total_mean = np.mean(img)

    for threshold in range(intesity_lvls):

        class_mean[threshold] = np.sum(img[img <= threshold]) / number_of_pixels
        class_probability[threshold] = np.sum(np.where(img <= threshold, 1, 0)) / number_of_pixels

    # ignoring error because of division with 0
    with np.errstate(all='ignore'):
        inbetween_variance = (total_mean * class_probability - class_mean) ** 2 / (class_probability * (1 - class_probability))

    optimal_threshold = np.nanargmax(inbetween_variance)
    total_variance = np.var(img)
    goodness = inbetween_variance[optimal_threshold] / total_variance
    return optimal_threshold, goodness


def otsu_faster(image, intensity_lvls=256):
    """
    This function takes an image and calculates the probability of class occurrence and the mean value for all pixels to
     calculate the threshold according to the formula of Otsu Thresholding without using a for loop.
     Also it calculates the total variance and uses it to calculate the goodness of the threshold.
    :param image: image out of Data
    :param intensity_lvls:
    :return: threshold and goodness of the image
    """

    img = image.copy().flatten()
    histogram = np.histogram(img, bins=np.arange(intensity_lvls + 1), density=True)

    class_probability = np.cumsum(histogram[0])
    class_mean = np.cumsum(histogram[0] * np.arange(intensity_lvls))
    total_mean = np.mean(img)

    # ignoring error because of division with 0
    with np.errstate(all='ignore'):

        inbetween_variance = (total_mean * class_probability - class_mean) ** 2 / (class_probability * (1 - class_probability))

    optimal_threshold = np.nanargmax(inbetween_variance)
    total_variance = np.var(img)
    goodness = inbetween_variance[optimal_threshold] / total_variance

    return optimal_threshold, goodness


def otsu_twolevel(img, intensity_lvls=256):
    """
    This function takes an image and calculates two thresholds according to the formula of two-level Otsu Thresholding.

    :param img: image out of Data
    :return: threshold of the image
    """

    histogram = np.histogram(img, bins=np.arange(intensity_lvls+1), density=True)

    class_probability = histogram[0]
    class_mean = histogram[0] * np.arange(intensity_lvls)

    Probabilities = np.zeros((intensity_lvls, intensity_lvls))
    Means = np.zeros((intensity_lvls, intensity_lvls))

    # Calculate class probability and mean for all possible intervals
    for lower_border in range(intensity_lvls):
        for upper_border in range(lower_border, intensity_lvls):
            Probabilities[lower_border, upper_border] = np.sum(class_probability[0:upper_border + 1]) - np.sum(class_probability[0:lower_border])
            Means[lower_border, upper_border] = np.sum(class_mean[0:upper_border + 1]) - np.sum(class_mean[0:lower_border])

    inbetween_variance = np.zeros((intensity_lvls, intensity_lvls))
    for first_threshold in range(intensity_lvls):
        for second_threshold in range(first_threshold + 1, intensity_lvls-1):
            with np.errstate(all='ignore'):
                inbetween_variance[first_threshold, second_threshold] = Means[0, first_threshold] ** 2 / Probabilities[0, first_threshold] + Means[first_threshold + 1, second_threshold] ** 2 / Probabilities[first_threshold + 1, second_threshold] + Means[second_threshold + 1, intensity_lvls-1] ** 2 / Probabilities[second_threshold + 1, intensity_lvls-1]

    maximal_variance = np.nanargmax(inbetween_variance)
    optimal_thresholds = np.unravel_index(maximal_variance, inbetween_variance.shape)

    return (optimal_thresholds)


def clipping(img, threshold):
    """
    This function takes the intensity of every pixel and sets its value to 0 if the threshold is equal or smaller 0.
    If the intensity value is greater than the threshold, the value is set to 1.

    :param img: image out of Data
    :param threshold: uses the threshold derived from Otsu Thresholding
    :return: workimg that has been clipped
    """

    workimg = img.copy()
    workimg[workimg <= threshold] = 0
    workimg[workimg > threshold] = 1

    return workimg


def intensity_value(path_to_image_collection):
    """
    This function returns the total number of possible intensity values
     for an image depending on the image type. Only supports .png and .tif.

    :return: The total number of intensity values
    :param path_to_image_collection: Path to image collection as a string in a list.
    """

    if path_to_image_collection[0][-3:] == 'tif':
        intensity = 2 ** 16
    elif path_to_image_collection[0][-3:] == 'png':
        intensity = 256
    else:
        raise Exception('Not a tif or png!')

    return intensity


### Just for testing. Delete later ###

image_test = imread(r'..\Data\NIH3T3\im\dna-27.png')
threshold, goodness = otsu_faster(image_test)
clipped_img = clipping(image_test, threshold)


plt.imshow(clipped_img, 'gray')
plt.show()
plt.imshow(image_test, 'gray')
plt.show()

# Testing whether the output is the same as in the skimage function
from skimage.filters import threshold_otsu
t_skimage = threshold_otsu(image_test)

print(threshold, t_skimage)

# Testing whether twolevel_otsu is the same as the skimage funktion
from skimage.filters import threshold_multiotsu
t_twolevel_skimage = otsu_twolevel(image_test)

print(t_twolevel_skimage)
