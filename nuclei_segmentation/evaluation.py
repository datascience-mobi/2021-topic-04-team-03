

import matplotlib.pyplot as plt
# from our data.path import list
import numpy as np

# from our data.path import list

def dice_score(clipped_image, ground_truth):
    """Calculation of the Dice Score.

        The DSC, also known as F1 score, calculates the True Positive (tp), False Negative (fn) and False Positives (fp) by using a specific formula.
        The DSC method compares the list of pixels in the ground truth image generated by the function control_images with the images derived from Otsu algorithm.
        :param otsu_thresholding: image derived after Otsu thresholding
        :param ground_truth: ground truth image
        :param dataset: dataset with all pictures
        :return: dice score coefficient
        """

    tp = 0
    fp = 0
    fn = 0

    for p in np.ndindex(clipped_image.shape):
        if clipped_image[p] == True and ground_truth[p] == True:
            tp = tp + 1
        if clipped_image[p] == True and ground_truth[p] == False:
            fp = fp + 1
        if clipped_image[p] == False and ground_truth[p] == True:
            fn = fn + 1

    dsc = (2 * tp) / (2 * tp + fp + fn)
    # if dataset == "otsu_data":
    return dsc

def dice_score_faster (clipped_image, ground_truth):
    """This is a version of the Dice score, that should be working faster.

    :param otsu_thresholding: image derived after Otsu thresholding
    :param ground_truth: ground truth image
    :return: dice score coefficient
    """
    # dice score without for-loop

    intersection = np.sum(clipped_image*ground_truth)
    union = np.sum(clipped_image) + np.sum(ground_truth)

    dsc = np.sum(2 * intersection) / np.sum(union)
    return dsc

def iou (clipped_image, ground_truth):

    """Calculation of the Intersection-Over-Union (IoU), also known as Jaccard Index.
    
        The IoU calculates the Area of Overlap divided by the Area of Union between our images derived from Otsu with the ground truth images.
        It is very similar to Dice, although the received value is always lower, so the IoU considers a higher discrepancy. 
        The other way round, the Dice score is less sensitive to outliers.
        :param otsu_thresholding: image derived after Otsu thresholding
        :param ground_truth: ground truth image
        :return: IoU coefficient
        """

    intersection = np.sum(clipped_image*ground_truth)
    union = np.sum(clipped_image) + np.sum(ground_truth) - intersection

    iou = np.mean(intersection) / union

    return iou
