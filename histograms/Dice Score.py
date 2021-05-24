

import matplotlib.pyplot as plt
# from our data.path import list
import numpy as np

# from our data.path import list

def dice_score(otsu_thresholding, picture_name, dataset):


    for p in np.ndindex(seeded_regions.shape):
        if otsu_thresholding[p] == True and gt_img[p] == True:
            tp = tp + 1
        if otsu_thresholding[p] == True and gt_img[p] == False:
            fp = fp + 1
        if otsu_thresholding[p] == False and gt_img[p] == True:
            fn = fn + 1

    dsc = (2 * tp) / (2 * tp + fp + fn)

    if dataset == "otsu_data":
        return dsc
