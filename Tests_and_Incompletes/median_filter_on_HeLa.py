from skimage.io import imread_collection
from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
import matplotlib.pyplot as plt


# use of median filter on N2DL_HeLa

N2DL_HeLa_gt = str(pl.Path('../Data/N2DL-HeLa/gt/*.tif'))
N2DL_HeLa_im = str(pl.Path('../Data/N2DL-HeLa/img/*.tif'))

read_N2DL_HeLa_gt = imread_collection(N2DL_HeLa_gt)
read_N2DL_HeLa_img = imread_collection(N2DL_HeLa_im)


median_list = []

for index in range(len(read_N2DL_HeLa_img)):
    median_filter_N2DL_HeLa = preprocessing.median_filter(read_N2DL_HeLa_img[index])
    plt.imshow(median_filter_N2DL_HeLa, 'gray')
    plt.show()
    segmented_N2DL_HeLa = otsu.complete_segmentation(median_filter_N2DL_HeLa, intensity_lvls=2 ** 16)
    dsc = evaluation.dice(segmented_N2DL_HeLa, read_N2DL_HeLa_gt[index])
    iou = evaluation.iou(segmented_N2DL_HeLa, read_N2DL_HeLa_gt[index])
    msd = evaluation.msd(segmented_N2DL_HeLa, read_N2DL_HeLa_gt[index])

    median_list.append(dsc)
    median_list.append(iou)
    median_list.append(msd)

print(median_list)
