import numpy as np
from nuclei_segmentation import visualisation
import json

with open("../Results/new_values.json", "r") as read_file:
    data = json.load(read_file)

# dataset 1

dice_scores_GOWT1 = []
msd_scores_GOWT1 = []

for method in ["No preprocessing", "Median filter", "Gaussian filter", "Histogram stretching",
               "Median filter and histogram stretching", "Gauss filter and histogram stretching"]:
    dice_scores_GOWT1.append(data[method]["N2DH-GOWT1"]["Dice Score"])
    msd_scores_GOWT1.append(data[method]["N2DH-GOWT1"]["MSD"])

dice_scores = np.array(dice_scores_GOWT1)
visualisation.comparison_preprocessing(dice_scores)

msd_scores = np.array(msd_scores_GOWT1)
visualisation.comparison_preprocessing(msd_scores, y_label='MSD Value')

# dataset 2

dice_scores_HeLa = []
msd_scores_HeLa = []

for method in ["No preprocessing", "Median filter", "Gaussian filter", "Histogram stretching",
               "Median filter and histogram stretching", "Gauss filter and histogram stretching"]:
    dice_scores_HeLa.append(data[method]["N2DL-HeLa"]["Dice Score"])
    msd_scores_HeLa.append(data[method]["N2DL-HeLa"]["MSD"])

dice_scores = np.array(dice_scores_HeLa)
visualisation.comparison_preprocessing(dice_scores)

msd_scores = np.array(msd_scores_HeLa)
visualisation.comparison_preprocessing(msd_scores, y_label='MSD Value')

# dataset 3

dice_scores_NIH3T3 = []
msd_scores_NIH3T3 = []

for method in ["No preprocessing", "Median filter", "Gaussian filter", "Histogram stretching",
               "Median filter and histogram stretching", "Gauss filter and histogram stretching"]:
    dice_scores_NIH3T3.append(data[method]["NIH3T3"]["Dice Score"])
    msd_scores_NIH3T3.append(data[method]["NIH3T3"]["MSD"])

dice_scores = np.array(dice_scores_NIH3T3)
visualisation.comparison_preprocessing(dice_scores)

msd_scores = np.array(msd_scores_NIH3T3)
visualisation.comparison_preprocessing(msd_scores, y_label='MSD Value')

