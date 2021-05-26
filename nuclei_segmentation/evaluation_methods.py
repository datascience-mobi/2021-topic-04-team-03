

import matplotlib.pyplot as plt
# from our data.path import list
import numpy as np

# from our data.path import list

def dice_score(otsu_thresholding, dataset):
    """Calculation of the Dice Score.

        The DSC, also known as F1 score, calculates the True Positive (tp), False Negative (fn) and False Positives (fp) by using a specific formula.
        The DSC method compares the list of pixels in the ground truth image generated by the function control_images with the images derived from Otsu algorithm.
        """

    tp = 0
    fp = 0
    fn = 0

    for p in np.ndindex(otsu_thresholding):
        if otsu_thresholding[p] == True and gt_img[p] == True:
            tp = tp + 1
        if otsu_thresholding[p] == True and gt_img[p] == False:
            fp = fp + 1
        if otsu_thresholding[p] == False and gt_img[p] == True:
            fn = fn + 1

    dsc = (2 * tp) / (2 * tp + fp + fn)

    if dataset == "otsu_data":
        return dsc

def dice_score_faster (ostu_thresholding, picture_name, dataset)
    # dice score without for-loop

    intersection = np.sum(otsu_thresholding*gt_img)
    union = np.sum(otsu_thresholding) + np.sum(gt_img)

    dsc = np.sum(2 * intersection) / np.sum(union)

    return dsc

def IoU(otsu_thresholding, dataset)
    """Calculation of the Intersection-Over-Union (IoU), also known as Jaccard Index.
    
        The IoU calculates the Area of Overlap divided by the Area of Union between our images derived from Otsu with the ground truth images.
        It is very similar to Dice, although the received value is always lower, so the IoU considers a higher discrepancy. 
        The other way round, the Dice score is less sensitive to outliers.
        """

    intersection = np.sum(otsu_thresholding*gt_img)
    union = np.sum(otsu_thresholding) + np.sum(gt_img) - intersection

    IoU = np.mean(intersection) / (union)

    return IoU
