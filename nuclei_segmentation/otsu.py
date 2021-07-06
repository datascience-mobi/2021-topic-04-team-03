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

    #img = image.copy().flatten()
    histogram = np.histogram(image, bins=np.arange(intensity_lvls + 1), density=True)

    class_probability = np.cumsum(histogram[0])
    class_mean = np.cumsum(histogram[0] * np.arange(intensity_lvls))
    total_mean = np.mean(image)

    with np.errstate(divide='ignore'):
        inbetween_variance = (total_mean * class_probability - class_mean) ** 2 / (
                class_probability * (1 - class_probability))


    # Inf values are invalid
    inbetween_variance[inbetween_variance == np.inf] = np.nan
    optimal_threshold = np.nanargmax(inbetween_variance)
    total_variance = np.var(image)
    goodness = inbetween_variance[optimal_threshold] / total_variance

    return optimal_threshold


def otsu_twolevel(img, intensity_lvls=256):
    """
    This function takes an image and calculates two thresholds according to the formula of two-level Otsu Thresholding.

    :param intensity_lvls: The total number of intensity levels
    :param img: Input image
    :return: Threshold of the image
    """
    intensity_lvls = intensity_lvls - 1
    histogram = np.histogram(img, bins=np.arange(intensity_lvls + 1), density=True)

    class_probability = histogram[0]
    class_mean = histogram[0] * np.arange(intensity_lvls)

    probabilities = np.zeros((intensity_lvls, intensity_lvls))
    means = np.zeros((intensity_lvls, intensity_lvls))

    # Calculate class probability and mean for all possible intervals
    for lower_border in range(intensity_lvls):
        for upper_border in range(lower_border, intensity_lvls):
            probabilities[lower_border, upper_border] = np.sum(class_probability[0:upper_border + 1]) - np.sum(
                class_probability[0:lower_border])
            means[lower_border, upper_border] = np.sum(class_mean[0:upper_border + 1]) - np.sum(
                class_mean[0:lower_border])

    inbetween_variance = np.zeros((intensity_lvls, intensity_lvls))

    for first_threshold in range(intensity_lvls):
        for second_threshold in range(first_threshold + 1, intensity_lvls - 1):
            means_c1 = means[0, first_threshold]
            means_c2 = means[first_threshold + 1, second_threshold]
            means_c3 = means[second_threshold + 1, intensity_lvls - 1]

            probs_c1 = probabilities[0, first_threshold]
            probs_c2 = probabilities[first_threshold + 1, second_threshold]
            probs_c3 = probabilities[second_threshold + 1, intensity_lvls - 1]

            with np.errstate(divide='ignore'):
                inbetween_variance[first_threshold, second_threshold] = means_c1 ** 2 / probs_c1 + \
                                                                        means_c2 ** 2 / probs_c2 + \
                                                                        means_c3 ** 2 / probs_c3
    # Inf values are invalid
    inbetween_variance[inbetween_variance == np.inf] = np.nan

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

    workimg = np.zeros(img.shape)
    workimg[img > threshold] = 1

    return workimg

def complete_segmentation (img, intensity_lvls=256):
    '''
    Performs complete image segmentation using Otsu threshold.

    :param img: Image to be segmented
    :return: Segmented binary image
    '''
    threshold = otsu(img, intensity_lvls)
    workimg = clipping(img, threshold)

    return workimg

def clipping_twolevel(img, thresholds):
    '''
    This function corrects bright reflections on image by setting pixels
    with intensity above the higher threshold black. Thresholds can be obtained
    by two-level Otsu algorithm.

    :param img: Image to be processed
    :param thresholds: List/array containing two thresholds
    :return: Corrected image
    '''

    workimg = np.zeros(img.shape)
    threshold_1 = min(thresholds)
    threshold_2 = max(thresholds)

    workimg[img > threshold_1] = 1
    workimg[img > threshold_2] = 0

    return workimg

def complete_segmentation_twolevel (img, intensity_lvls=256):
    '''
    Performs complete image segmentation using Two-Level Otsu thresholding.
    The purpose is to eliminate reflections.

    :param img: Image (with reflections)
    :param intensity_lvls: Total number of intensity levels
    :return: Segmented binary image
    '''

    thresholds = otsu_twolevel(img, intensity_lvls=intensity_lvls)
    segmented_img = clipping_twolevel(img, thresholds)

    return segmented_img


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

def alternative_reflection_otsu(img, intensity_lvls = 256):
    '''
    This is alternative for handling reflections in the images.
    It first calculates two thresholds with two-level Otsu.
    Than the values below the lower and above the higher threshold
    are set to the value of the lower threshold.
    The image is than stretched. A new threshold is calculated with one-level Otsu.
    The image is clipped with the new threshold.

    :param img: Input image
    :return: Segmented image
    '''
    two_lvl_thresholds = otsu_twolevel(img)
    threshold1, threshold2 = min(two_lvl_thresholds), max(two_lvl_thresholds)

    workimg = img.copy()
    workimg[workimg < threshold1] = threshold1
    workimg[workimg > threshold2] = threshold1

    min_intensity_output = 0
    max_intensity_output = intensity_lvls
    min_intensity_input = int(np.min(workimg))
    max_intensity_input = int(np.max(workimg))

    stretched_image = (workimg - min_intensity_input) * (max_intensity_output - min_intensity_output) / \
                      (max_intensity_input - min_intensity_input)

    one_lvl_threshold = otsu(stretched_image)
    img_clipped = clipping(stretched_image, one_lvl_threshold)

    return img_clipped

# Just for testing. Delete later #
if __name__ == '__main__':
    from skimage.io import imread
    from skimage.io import imshow
    from matplotlib import pyplot as plt
    import pathlib
    image_test = imread(pathlib.Path(r'..\Data\NIH3T3\img\dna-27.png'))

    seg_img = complete_segmentation(image_test)
    plt.imshow(seg_img)
    plt.show()
    # threshold = otsu(image_test)
    # clipped_img = clipping(image_test, threshold)
    # print(threshold)
    #
    #
    # plt.imshow(clipped_img, 'gray')
    # plt.show()
    # plt.imshow(image_test, 'gray')
    # plt.show()
    #
    #
    # # Testing whether the output is the same as in the skimage function
    # from skimage.filters import threshold_otsu
    #
    # t_skimage = threshold_otsu(image_test)
    #
    # print(threshold, t_skimage)
    #
    # # Testing whether twolevel_otsu is the same as the skimage funktion
    # from skimage.filters import threshold_multiotsu
    #
    # t_twolevel_skimage = otsu_twolevel(image_test)
    #
    # print(t_twolevel_skimage)