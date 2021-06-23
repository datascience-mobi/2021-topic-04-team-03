from skimage.io import imread_collection
from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
import matplotlib.pyplot as plt


N2DL_HeLa_gt = str(pl.Path('../Data/N2DL-HeLa/img/*.tif'))
N2DL_HeLa_im = str(pl.Path('../Data/N2DL-HeLa/gt/*.tif'))

read_N2DL_HeLa_gt = imread_collection(N2DL_HeLa_gt)
read_N2DL_HeLa_img = imread_collection(N2DL_HeLa_im)

histogram_stretching_list = []

for index in range(len(read_N2DL_HeLa_img)):
    histogram_stretching_N2DL_HeLa = preprocessing.histogram_stretching(read_N2DL_HeLa_img[index])
    plt.imshow(histogram_stretching_N2DL_HeLa, 'gray')
    plt.show()
    segmented_N2DL_HeLa = otsu.complete_segmentation(histogram_stretching_N2DL_HeLa, intensity_lvls=2 ** 16)
    dsc = evaluation.dice(segmented_N2DL_HeLa, read_N2DL_HeLa_gt[index])
    iou = evaluation.iou(segmented_N2DL_HeLa, read_N2DL_HeLa_gt[index])
    msd = evaluation.msd(segmented_N2DL_HeLa, read_N2DL_HeLa_gt[index])

    histogram_stretching_list.append(dsc)
    histogram_stretching_list.append(iou)
    histogram_stretching_list.append(msd)

print(histogram_stretching_list)