import numpy as np
from contextlib import suppress

def otsu(image, intensity_lvls=256):
    """
    This function takes an image and calculates the probability of class occurrence and the mean value for all pixels to
     calculate the threshold according to the formula of Otsu Thresholding without using a for loop.
     Also it calculates the total variance and uses it to calculate the goodness of the threshold.

    :param image: Input image
    :param intensity_lvls:The total number of intensity levels
    :return: Threshold and goodness of the image
    """

    img = image.copy().flatten()
    histogram = np.histogram(img, bins=np.arange(intensity_lvls + 1), density=True)

    class_probability = np.cumsum(histogram[0])
    class_mean = np.cumsum(histogram[0] * np.arange(intensity_lvls))
    total_mean = np.mean(img)

    try:
        inbetween_variance = (total_mean * class_probability - class_mean) ** 2 / (
                class_probability * (1 - class_probability))
    except ZeroDivisionError:
        inbetween_variance = (total_mean * class_probability - class_mean) ** 2 / (
                class_probability * (1 - class_probability))

    optimal_threshold = np.nanargmax(inbetween_variance)
    total_variance = np.var(img)
    goodness = inbetween_variance[optimal_threshold] / total_variance

    return optimal_threshold, goodness


def otsu_twolevel(img, intensity_lvls=256):
    """
    This function takes an image and calculates two thresholds according to the formula of two-level Otsu Thresholding.

    :param intensity_lvls: The total number of intensity levels
    :param img: Input image
    :return: Threshold of the image
    """

    histogram = np.histogram(img, bins=np.arange(intensity_lvls + 1), density=True)

    class_probability = histogram[0]
    class_mean = histogram[0] * np.arange(intensity_lvls)

    Probabilities = np.zeros((intensity_lvls, intensity_lvls))
    Means = np.zeros((intensity_lvls, intensity_lvls))

    # Calculate class probability and mean for all possible intervals
    for lower_border in range(intensity_lvls):
        for upper_border in range(lower_border, intensity_lvls):
            Probabilities[lower_border, upper_border] = np.sum(class_probability[0:upper_border + 1]) - np.sum(
                class_probability[0:lower_border])
            Means[lower_border, upper_border] = np.sum(class_mean[0:upper_border + 1]) - np.sum(
                class_mean[0:lower_border])

    inbetween_variance = np.zeros((intensity_lvls, intensity_lvls))
    for first_threshold in range(intensity_lvls):
        for second_threshold in range(first_threshold + 1, intensity_lvls - 1):
            with np.errstate(all='ignore'):

                means_c1 = Means[0, first_threshold]
                means_c2 = Means[first_threshold + 1, second_threshold]
                means_c3 = Means[second_threshold + 1, intensity_lvls - 1]

                probs_c1 = Probabilities[0, first_threshold]
                probs_c2 = Probabilities[first_threshold + 1, second_threshold]
                probs_c3 = Probabilities[second_threshold + 1, intensity_lvls - 1]

                inbetween_variance[first_threshold, second_threshold] = means_c1 ** 2 / probs_c1 + \
                                                                        means_c2 ** 2 / probs_c2 + \
                                                                        means_c3 ** 2 / probs_c3

    maximal_variance = np.nanargmax(inbetween_variance)
    optimal_thresholds = np.unravel_index(maximal_variance, inbetween_variance.shape)

    return optimal_thresholds


def clipping(img, threshold):
    """
    This function takes the intensity of every pixel and sets its value to 0 if the threshold is equal or smaller 0.
    If the intensity value is greater than the threshold, the value is set to 1.

    :param img: Input image
    :param threshold: Threshold that defines the clipping
    :return: Clipped image
    """

    workimg = img.copy()
    workimg[workimg <= threshold] = 0
    workimg[workimg > threshold] = 1

    return workimg

def complete_segmentation (img, intensity_lvls=256):
    '''
    Performs complete image segmentation using Otsu threshold.

    :param img: Image to be segmented
    :return: Segmented binary image
    '''
    threshold, goodness = otsu(img, intensity_lvls)
    workimg = clipping(img, threshold)

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


# Just for testing. Delete later #
if __name__ == '__main__':
    from skimage.io import imread
    from skimage.io import imshow
    from matplotlib import pyplot as plt
    import pathlib
    image_test = imread(pathlib.Path(r'..\Data\NIH3T3\img\dna-27.png'))
    threshold, goodness = otsu(image_test)
    clipped_img = clipping(image_test, threshold)
    print(threshold,goodness)


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