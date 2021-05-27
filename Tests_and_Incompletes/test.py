from skimage.io import imread_collection
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
from scipy.spatial import distance
from skimage.io import imread
from skimage.io import imshow
from matplotlib import pyplot as plt
import numpy as np



col_dir_img = [r'''..\Data\NIH3T3\im\*.png''']
col_dir_gt = [r'''..\Data\NIH3T3\gt\*.png''']

col_img = imread_collection(col_dir_img )
col_gt = imread_collection(col_dir_gt)

print(col_img)


clipped_list = []
gt_list = []


for i in col_img:
    threshold, goodness = otsu.otsu_faster(i)
    clipped_list.append(otsu.clipping(i,threshold))

for i in col_gt:
    gt_list.append(i)

for i in range(len(clipped_list)):
    print(evaluation.dice_score_faster(clipped_list[i], gt_list[i]))

