from skimage.io import imread_collection
from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
import matplotlib.pyplot as plt

# use of median filter on N2DH_GOWT1

N2DH_GOWT1_gt = str(pl.Path('../Data/N2DH-GOWT1/img/*.tif'))
N2DH_GOWT1_im = str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif'))

read_N2DH_GOWT1_gt = imread_collection(N2DH_GOWT1_gt)
read_N2DH_GOWT1_img = imread_collection(N2DH_GOWT1_im)

median_list = []

for index in range(len(read_N2DH_GOWT1_img)):
    median_filter_N2DH_GOWT1 = preprocessing.median_filter(read_N2DH_GOWT1_img[index])
    plt.imshow(median_filter_N2DH_GOWT1, 'gray')
    plt.show()
    segmented_N2DH_GOWT1 = otsu.complete_segmentation(median_filter_N2DH_GOWT1, intensity_lvls=2 ** 16)
    dsc = evaluation.dice(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])
    iou = evaluation.iou(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])
    msd = evaluation.msd(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])

    median_list.append(dsc)
    median_list.append(iou)
    median_list.append(msd)

print(median_list)

