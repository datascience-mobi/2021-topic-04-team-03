from skimage.io import imread_collection
from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
import matplotlib.pyplot as plt

NIH3T3_gt = str(pl.Path('../Data/NIH3T3/img/*.png'))
NIH3T3_im = str(pl.Path('../Data/NIH3T3/gt/*.png'))

read_NIH3T3_gt = imread_collection(NIH3T3_gt)
read_NIH3T3_img = imread_collection(NIH3T3_im)

histogram_stretching_list = []

for index in range(len(read_NIH3T3_img)):
    histogram_stretching_NIH3T3 = preprocessing.histogram_stretching(read_NIH3T3_img[index])
    plt.imshow(histogram_stretching_NIH3T3, 'gray')
    plt.show()
    segmented_NIH3T3 = otsu.complete_segmentation(histogram_stretching_NIH3T3, intensity_lvls= 256)
    dsc = evaluation.dice(segmented_NIH3T3, read_NIH3T3_gt[index])
    iou = evaluation.iou(segmented_NIH3T3, read_NIH3T3_gt[index])
    msd = evaluation.msd(segmented_NIH3T3, read_NIH3T3_gt[index])

    histogram_stretching_list.append(dsc)
    histogram_stretching_list.append(iou)
    histogram_stretching_list.append(msd)

print(histogram_stretching_list)