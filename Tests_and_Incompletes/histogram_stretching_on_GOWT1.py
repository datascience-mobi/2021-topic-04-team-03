from skimage.io import imread_collection
from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
import matplotlib.pyplot as plt

N2DH_GOWT1_gt = str(pl.Path('../Data/N2DH-GOWT1/img/*.tif'))
N2DH_GOWT1_im = str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif'))

read_N2DH_GOWT1_gt = imread_collection(N2DH_GOWT1_gt)
read_N2DH_GOWT1_img = imread_collection(N2DH_GOWT1_im)

histogram_stretching_list = []

for index in range(len(read_N2DH_GOWT1_img)):
    histogram_stretching_N2DH_GOWT1 = preprocessing.histogram_stretching(read_N2DH_GOWT1_img[index])
    plt.imshow(histogram_stretching_N2DH_GOWT1, 'gray')
    plt.show()
    segmented_N2DH_GOWT1 = otsu.complete_segmentation(histogram_stretching_N2DH_GOWT1, intensity_lvls=2 ** 16)
    dsc = evaluation.dice(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])
    iou = evaluation.iou(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])
    msd = evaluation.msd(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])

    histogram_stretching_list.append(dsc)
    histogram_stretching_list.append(iou)
    histogram_stretching_list.append(msd)

print(histogram_stretching_list)