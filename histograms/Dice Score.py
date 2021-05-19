

import opencv-python as cv2
import matplotlib.pyplot as plt
# from our data.path import list
import numpy as np

# from our data.path import list

def dice_score(otsu_thresholding, picture_name, dataset_flag):
    # A flag variable, in its simplest form, is a variable you define to have one value until some condition is true, in which case you change the variable's value.
    # It is a variable you can use to control the flow of a function or statement, allowing you to check for certain conditions while your function progresses.
    """Calculation of the Dice Score.

    The DSC calculates the True Positive (tp), False Negative (fn) and False Positives (fp) by using a specific formula.
    The DSC method compares the list of pixels in the ground truth image generated by the function control_images with the images derived from Otsu algorithm.
    The ground truth images for the pictures are saved after the first time the code is executed.
    For every following calculation of the Dice Score for the same image the ground truth images are loaded.

    :param dataset_flag: Specifies the dataset.
    :param otsu_data: image that went through Otsu Thresholding
    :param picture_name: Specifies the picture the control pictures are searched for.
    :return: dsc (Dice Score)
    """

    if dataset_flag == "otsu_data":
        control_img, count_control_pixel = preprocessing_BBBC020(picture_name)
    if dataset_flag == "otsu_data":
        control_img = preprocessing_BBBC007(picture_name, seeded_regions.shape)

    tp = 0
    fp = 0
    fn = 0

    for p in np.ndindex(seeded_regions.shape):
        if otsu_thresholding[p] == True and gt_img[p] == True:
            tp = tp + 1
        if otsu_thresholding[p] == True and gt_img[p] == False:
            fp = fp + 1
        if otsu_thresholding[p] == False and gt_img[p] == True:
            fn = fn + 1

    dsc = (2 * tp) / (2 * tp + fp + fn)

    if dataset_flag == "otsu_data":
        return dsc
