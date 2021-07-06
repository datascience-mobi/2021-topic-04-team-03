from nuclei_segmentation import preprocessing, otsu, evaluation, metrics
import pathlib as pl
from skimage.io import imread_collection
import json
import numpy as np

def without_preprocessing_function_application(col_img, col_gt, intensity_lvls = 256, mode = "one_level"):
    '''
    This function takes all images of a dataset and ground truth images and uses the Otsu algorithm on it. Afterwards
    the Dice score, the mean surface distance (msd) and the hausdorff distance.

    :param col_img: collection of images of a dataset
    :param col_gt: collection of ground truth of the dataset
    :param intensity_lvls: gives the number of intensity levels in an image used for Otsu's algorithm
    :param mode: defines whether one-level or two-level Otsu is used on the dataset
    :return: lists of dice, msd and hausdorff values for this dataset
    '''

    dice_list = []
    msd_list = []
    hausdorff_list = []

    for index in range(len(col_img)):
        if mode == "one_level":
            segmented_img = otsu.complete_segmentation(col_img[index], intensity_lvls)
        elif mode == "two_level":
            segmented_img = otsu.complete_segmentation_twolevel(col_img[index], intensity_lvls)
        else:
            raise Exception("Invalid mode!")

        dice_list.append(evaluation.dice(segmented_img, col_gt[index]))
        msd_list.append(evaluation.msd(segmented_img, col_gt[index]))
        hausdorff_list.append(evaluation.hausdorff(segmented_img, col_gt[index]))

    return dice_list, msd_list, hausdorff_list




def gauss_function_application(col_img, col_gt, intensity_lvls = 256, mode="one_level", filter_size=11, sigma=5):
    '''
    This function takes all images of a dataset and ground truth images, preprocesses them by using a gauss filter
    and uses the Otsu algorithm on it. Afterwards the Dice score, the mean surface distance (msd) and the
    hausdorff distance.

    :param col_img: collection of images of a dataset
    :param col_gt: collection of ground truth of the dataset
    :param intensity_lvls: gives the number of intensity levels in an image used for Otsu's algorithm
    :param mode: defines whether one-level or two-level Otsu is used on the dataset
    :param filter_size: defines the filter size of the gaussian filter
    :param sigma: standard deviation of the gaussian filter
    :return: lists of dice, msd and hausdorff values for this dataset
    '''
    kernel = preprocessing.gaussian_kernel(filter_size, sigma)

    dice_list = []
    msd_list = []
    hausdorff_list = []

    for index in range(len(col_img)):

        gauss_filter_img = preprocessing.convolution(col_img[index], kernel)

        if mode == "one_level":
            segmented_img = otsu.complete_segmentation(gauss_filter_img, intensity_lvls)
        elif mode == "two_level":
            segmented_img = otsu.complete_segmentation_twolevel(gauss_filter_img, intensity_lvls)
        else:
            raise Exception("Invalid mode!")


        dice_list.append(evaluation.dice(segmented_img, col_gt[index]))
        #msd_list.append(evaluation.msd(segmented_img, col_gt[index]))
        #hausdorff_list.append(evaluation.hausdorff(segmented_img, col_gt[index]))

    return dice_list, msd_list, hausdorff_list




def median_function_application(col_img, col_gt, intensity_lvls = 256, filter_size=3, mode = "one_level"):
    '''
    This function takes all images of a dataset and ground truth images, preprocesses them by using the median filter
    and uses the Otsu algorithm on it. Afterwards the Dice score, the mean surface distance (msd) and the
    hausdorff distance.

    :param col_img: collection of images of a dataset
    :param col_gt: collection of ground truth of the dataset
    :param intensity_lvls: gives the number of intensity levels in an image used for Otsu's algorithm
    :param filter_size: defines the filter size of the median filter
    :param mode: defines whether one-level or two-level Otsu is used on the dataset
    :return: lists of dice, msd and hausdorff values for this dataset
    '''
    dice_list = []
    msd_list = []
    hausdorff_list = []

    for index in range(len(col_img)):
        median_filter_img = preprocessing.median_filter(col_img[index], filter_size=filter_size)

        if mode == "one_level":
            segmented_img = otsu.complete_segmentation(median_filter_img, intensity_lvls)
        elif mode == "two_level":
            segmented_img = otsu.complete_segmentation_twolevel(median_filter_img, intensity_lvls)
        else:
            raise Exception("Invalid mode!")

        dice_list.append(evaluation.dice(segmented_img, col_gt[index]))
        msd_list.append(evaluation.msd(segmented_img, col_gt[index]))
        hausdorff_list.append(evaluation.hausdorff(segmented_img, col_gt[index]))

    return dice_list, msd_list, hausdorff_list




def histogram_stretching_function_application(col_img, col_gt, intensity_lvls = 256, mode="one_level"):
    '''
    This function takes all images of a dataset and ground truth images and preprocesses them with histogram stretching
    and uses the Otsu algorithm on it. Afterwards the Dice score, the mean surface distance (msd) and the
    hausdorff distance.

    :param col_img: collection of images of a dataset
    :param col_gt: collection of ground truth of the dataset
    :param intensity_lvls: gives the number of intensity levels in an image used for Otsu's algorithm
    :param mode: defines whether one-level or two-level Otsu is used on the dataset
    :return: lists of dice, msd and hausdorff values for this dataset
    '''
    dice_list = []
    msd_list = []
    hausdorff_list = []

    for index in range(len(col_img)):

        histogram_stretching_img = preprocessing.histogram_stretching(col_img[index], intensity_lvls=2**16)

        if mode == "one_level":
            segmented_img = otsu.complete_segmentation(histogram_stretching_img, intensity_lvls)
        elif mode == "two_level":
            segmented_img = otsu.complete_segmentation_twolevel(histogram_stretching_img, intensity_lvls)
        else:
            raise Exception("Invalid mode!")

        dice_list.append(evaluation.dice(segmented_img, col_gt[index]))
        msd_list.append(evaluation.msd(segmented_img, col_gt[index]))
        hausdorff_list.append(evaluation.hausdorff(segmented_img, col_gt[index]))

    return dice_list, msd_list, hausdorff_list




def median_histogram_stretching_function_application(col_img, col_gt, filter_size = 3, intensity_lvls = 256,
                                                     mode="one_level"):
    '''
    This function takes all images of a dataset and ground truth images, preprocesses them by using the median filter
    and histogram stretching and uses the Otsu algorithm on it. Afterwards the Dice score, the mean surface distance
    (msd) and the hausdorff distance.

    :param col_img: collection of images of a dataset
    :param col_gt: collection of ground truth of the dataset
    :param filter_size: defines the filter size of the median filter
    :param intensity_lvls: gives the number of intensity levels in an image used for Otsu's algorithm
    :param mode: defines whether one-level or two-level Otsu is used on the dataset
    :return: lists of dice, msd and hausdorff values for this dataset
    '''
    dice_list = []
    msd_list = []
    hausdorff_list = []

    for index in range(len(col_img)):

        median_filter_image = preprocessing.median_filter(col_img[index], filter_size)
        median_filter_histogram_stretch_img = preprocessing.histogram_stretching(median_filter_image, intensity_lvls)

        if mode == "one_level":
            segmented_img = otsu.complete_segmentation(median_filter_histogram_stretch_img, intensity_lvls)
        elif mode == "two_level":
            segmented_img = otsu.complete_segmentation_twolevel(median_filter_histogram_stretch_img, intensity_lvls)
        else:
            raise Exception("Invalid mode!")

        dice_list.append(evaluation.dice(segmented_img, col_gt[index]))
        msd_list.append(evaluation.msd(segmented_img, col_gt[index]))
        hausdorff_list.append(evaluation.hausdorff(segmented_img, col_gt[index]))

    return dice_list, msd_list, hausdorff_list



def gauss_histogram_stretching_function_application(col_img, col_gt, intensity_lvls = 256, mode="one_level",
                                                    filter_size=5, sigma=1):
    '''
    This function takes all images of a dataset and ground truth images, preprocesses them by using the gaussian filter
    and the histogram stretching function and uses the Otsu algorithm on it. Afterwards the Dice score, the mean surface
    distance (msd) and the hausdorff distance.

    :param col_img: collection of images of a dataset
    :param col_gt: collection of ground truth of the dataset
    :param intensity_lvls: gives the number of intensity levels in an image used for Otsu's algorithm
    :param mode: defines whether one-level or two-level Otsu is used on the dataset
    :param filter_size: defines the filter size of the gaussian filter
    :param sigma: standard deviation of the gaussian filter
    :return: lists of dice, msd and hausdorff values for this dataset
    '''
    kernel = preprocessing.gaussian_kernel(filter_size, sigma)

    dice_list = []
    msd_list = []
    hausdorff_list = []

    for index in range(len(col_img)):

        gauss_filter_image = preprocessing.convolution(col_img[index], kernel)
        gauss_filter_histogram_stretch_img = preprocessing.histogram_stretching(gauss_filter_image, intensity_lvls)

        if mode == "one_level":
            segmented_img = otsu.complete_segmentation(gauss_filter_histogram_stretch_img, intensity_lvls)
        elif mode == "two_level":
            segmented_img = otsu.complete_segmentation_twolevel(gauss_filter_histogram_stretch_img, intensity_lvls)
        else:
            raise Exception("Invalid mode!")

        dice_list.append(evaluation.dice(segmented_img, col_gt[index]))
        msd_list.append(evaluation.msd(segmented_img, col_gt[index]))
        hausdorff_list.append(evaluation.hausdorff(segmented_img, col_gt[index]))

    return dice_list, msd_list, hausdorff_list

def write_in_json(file_path, combinations, dataset, scores):
    '''

    :param file_path: this is the path to the file that is opened
    :param combinations: this
    :param dataset:
    :param scores:
    :return:
    '''
    with open(file_path, "r") as file:
        json_object = json.load(file)
    for index in range(len(combinations)):
        # create dictionary
        if combinations[index] in json_object:
            data = json_object[combinations[index]]
        else:
            data = {}
        data[dataset] = {"Dice Score": scores[index][0],
                          "MSD": scores[index][1],
                          "Hausdorff": scores[index][2]}

        json_object[combinations[index]] = data

    with open(file_path, "w") as file:
        json.dump(json_object, file, indent=3)


def cell_counting_analysis (collection):
    detected_number_list = []
    gt_number_list = []
    relative_differences = []
    absolute_differences =[]
    for image in collection:
        detected_number = metrics.cell_counting(image)
        gt_number = metrics.cell_counting_ground_truth(image)
        absolute_differences.append(detected_number-gt_number)
        relative_differences.append((detected_number-gt_number)/gt_number)
        detected_number_list.append(detected_number)
        gt_number_list.append(gt_number)
    return detected_number_list, gt_number_list, absolute_differences, relative_differences




if __name__ == "__main__":

    N2DH_GOWT1_img = str(pl.Path('../Data/N2DH-GOWT1/img/*.tif'))
    N2DH_GOWT1_gt = str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif'))

    N2DL_HeLa_img = str(pl.Path('../Data/N2DL-HeLa/img/*.tif'))
    N2DL_HeLa_gt = str(pl.Path('../Data/N2DL-HeLa/gt/*.tif'))

    NIH3T3_img = str(pl.Path('../Data/NIH3T3/img/*.png'))
    NIH3T3_gt = str(pl.Path('../Data/NIH3T3/gt/*.png'))

    col_img_GOWT1 = imread_collection(N2DH_GOWT1_img)
    col_gt_GOWT1 = imread_collection(N2DH_GOWT1_gt)

    col_img_HeLa = imread_collection(N2DL_HeLa_img)
    col_gt_HeLa = imread_collection(N2DL_HeLa_gt)

    col_img_NIH3T3 = imread_collection(NIH3T3_img)
    col_gt_NIH3T3 = imread_collection(NIH3T3_gt)

    # without preprocessing

    without_preprocessing_GOWT1 = without_preprocessing_function_application(col_img_GOWT1, col_gt_GOWT1,
                                                                                           intensity_lvls=2 ** 16)
    without_preprocessing_HeLa = without_preprocessing_function_application(col_img_HeLa, col_gt_HeLa,
                                                                                          intensity_lvls=2 ** 16)
    without_preprocessing_NIH3T3 = without_preprocessing_function_application(col_img_NIH3T3,
                                                                                            col_gt_NIH3T3)

    print(without_preprocessing_GOWT1)
    print(without_preprocessing_HeLa)
    print(without_preprocessing_NIH3T3)

    # gauss function - values for dice, msd and hsd

    gauss_GOWT1 = gauss_function_application(col_img_GOWT1, col_gt_GOWT1, intensity_lvls=2 ** 16)
    gauss_HeLa = gauss_function_application(col_img_HeLa, col_gt_HeLa, intensity_lvls=2 ** 16)
    gauss_NIH3T3 = gauss_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(gauss_GOWT1)
    print(gauss_HeLa)
    print(gauss_NIH3T3)

    # median function - values for dice, msd and hsd

    median_GOWT1 = median_function_application(col_img_GOWT1, col_gt_GOWT1, intensity_lvls=2 ** 16)
    median_HeLa = median_function_application(col_img_HeLa, col_gt_HeLa, intensity_lvls=2 ** 16)
    median_NIH3T3 = median_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(median_GOWT1)
    print(median_HeLa)
    print(median_NIH3T3)

    # histogram stretching function - values for dice, msd and hsd

    histogram_stretching_GOWT1 = histogram_stretching_function_application(col_img_GOWT1, col_gt_GOWT1,
                                                                                         intensity_lvls=2 ** 16)
    histogram_stretching_HeLa = histogram_stretching_function_application(col_img_HeLa, col_gt_HeLa,
                                                                                        intensity_lvls=2 ** 16)
    histogram_stretching_NIH3T3 = histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(histogram_stretching_GOWT1)
    print(histogram_stretching_HeLa)
    print(histogram_stretching_NIH3T3)

    #median and histogram stretching function - values for dice, msd and hsd

    median_histogram_stretching_GOWT1 = median_histogram_stretching_function_application(col_img_GOWT1, col_gt_GOWT1,
                                                                                         intensity_lvls=2 ** 16)
    median_histogram_stretching_HeLa = median_histogram_stretching_function_application(col_img_HeLa, col_gt_HeLa,
                                                                                        intensity_lvls=2 ** 16)
    median_histogram_stretching_NIH3T3 = median_histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(median_histogram_stretching_GOWT1)
    print(median_histogram_stretching_HeLa)
    print(median_histogram_stretching_NIH3T3)

    # median and histogram stretching function - values for dice, msd and hsd

    gauss_histogram_stretching_GOWT1 = gauss_histogram_stretching_function_application(col_img_GOWT1, col_gt_GOWT1,
                                                                                       intensity_lvls=2 ** 16)
    gauss_histogram_stretching_HeLa = gauss_histogram_stretching_function_application(col_img_HeLa, col_gt_HeLa,
                                                                                      intensity_lvls=2 ** 16)
    gauss_histogram_stretching_NIH3T3 = gauss_histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(gauss_histogram_stretching_GOWT1)
    print(gauss_histogram_stretching_HeLa)
    print(gauss_histogram_stretching_NIH3T3)
