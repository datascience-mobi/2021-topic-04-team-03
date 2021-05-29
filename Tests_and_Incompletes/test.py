import nuclei_segmentation


from skimage.io import imread_collection
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
from scipy.spatial import distance
from skimage.io import imread
from skimage.io import imshow
from matplotlib import pyplot as plt
import numpy as np
import os


col_dir_img = [r'..\Data\N2DL-HeLa\img\*.tif']
col_dir_gt = [r'..\Data\N2DL-HeLa\gt\*.tif']

col_dir_img_a = [r'..\Data\NIH3T3\im\*.png']
col_dir_gt = [r'..\Data\NIH3T3\gt\*.png']

col_img = imread_collection(col_dir_img, col_dir_img_a)
col_gt = imread_collection(col_dir_gt)

print(col_img)


r'''
print(col_img)

clipped_list = []
gt_list = []

for i in col_img:
    threshold, goodness = otsu.otsu_faster(i)
    clipped_list.append(otsu.clipping(i, threshold))
    print(threshold)

for i in col_gt:
    gt_list.append(i)

for i in range(len(clipped_list)):
    print(evaluation.dice_score_faster(clipped_list[i], gt_list[i]))


# try to make this a function:

def pipeline_assembly():
    """ This function takes all images of the given dataset and uses our Otsu algorithm on them and then clips those images.
    Finally, it evaluates the Otsu Threshold by using the Dice score on those clipped images and their ground truth images.
    """
    if __name__ == "__main__":
        #try to use os.walk function to open all files at once, does not work yet -cannot distinguish img an gt
        for img in os.walk('./Data'):

            img_dataset = imread(img, plugin='tifffile')

            gt_dataset_dir = ['Data/dataset/gt/*.tif']
            gt_dataset = imread_collection(gt_dataset_dir, plugin='tifffile')

            clipped_list_dataset = []
            gt_list_dataset = []

            for img in img_dataset:
                threshold, goodness = otsu.otsu_faster(img)
                clipped_list_dataset.append(otsu.clipping(i, threshold))

            for img in gt_dataset:
                gt_list_dataset.append(img)

            for img in range(len(clipped_list_dataset)):
                print(evaluation.dice_score_faster(clipped_list_dataset[img], gt_list_dataset[img]))


gt = gt_list[6]
im = col_img[6]

from skimage.filters import threshold_otsu
t_skimage = threshold_otsu(im)
print(t_skimage)

print(otsu.otsu_faster(im))

'''