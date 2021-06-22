from skimage.io import imread_collection
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
import matplotlib.pyplot as plt
'''
# dataset 2

N2DL_HeLa_gt = str(pl.Path('../Data/N2DL-HeLa/img/*.tif'))
N2DL_HeLa_im = str(pl.Path('../Data/N2DL-HeLa/gt/*.tif'))

read_N2DL_HeLa_gt = imread_collection(N2DL_HeLa_gt)
read_N2DL_HeLa_img = imread_collection(N2DL_HeLa_im)

dice_list = []
iou_list = []
msd_list = []

for index in range(len(read_N2DL_HeLa_img)):
    plt.imshow(read_N2DL_HeLa_img[index], 'gray')
    plt.show()
    segmented_N2DL_HeLa = otsu.complete_segmentation(read_N2DL_HeLa_img[index], intensity_lvls=2 ** 16)
    dsc = evaluation.dice(segmented_N2DL_HeLa, read_N2DL_HeLa_gt[index])
    iou = evaluation.iou(segmented_N2DL_HeLa, read_N2DL_HeLa_gt[index])
    msd = evaluation.msd(segmented_N2DL_HeLa, read_N2DL_HeLa_gt[index])

    dice_list.append(dsc)
    iou_list.append(iou)
    msd_list.append(msd)

print(dice_list)
print(iou_list)
print(msd_list)

#dataset 1

N2DH_GOWT1_gt = str(pl.Path('../Data/N2DH-GOWT1/img/*.tif'))
N2DH_GOWT1_im = str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif'))

read_N2DH_GOWT1_gt = imread_collection(N2DH_GOWT1_gt)
read_N2DH_GOWT1_img = imread_collection(N2DH_GOWT1_im)

dice_list = []
iou_list = []
msd_list = []

for index in range(len(read_N2DH_GOWT1_img)):
    plt.imshow(read_N2DH_GOWT1_img[index], 'gray')
    plt.show()
    segmented_N2DH_GOWT1 = otsu.complete_segmentation(read_N2DH_GOWT1_img[index], intensity_lvls=2 ** 16)
    dsc = evaluation.dice(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])
    iou = evaluation.iou(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])
    msd = evaluation.msd(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])

    dice_list.append(dsc)
    iou_list.append(iou)
    msd_list.append(msd)

print(dice_list)
print(iou_list)
print(msd_list)
'''
# dataset 3

NIH3T3_gt = str(pl.Path('../Data/NIH3T3/img/*.png'))
NIH3T3_im = str(pl.Path('../Data/NIH3T3/gt/*.png'))

read_NIH3T3_gt = imread_collection(NIH3T3_gt)
read_NIH3T3_img = imread_collection(NIH3T3_im)

dice_list = []
iou_list = []
msd_list = []

for index in range(len(read_NIH3T3_img)):
    plt.imshow(read_NIH3T3_img[index], 'gray')
    plt.show()
    segmented_NIH3T3 = otsu.complete_segmentation(read_NIH3T3_img[index], intensity_lvls=2 ** 16)
    dsc = evaluation.dice(segmented_NIH3T3, read_NIH3T3_gt[index])
    iou = evaluation.iou(segmented_NIH3T3, read_NIH3T3_gt[index])
    msd = evaluation.msd(segmented_NIH3T3, read_NIH3T3_gt[index])

    dice_list.append(dsc)
    iou_list.append(iou)
    msd_list.append(msd)

print(dice_list)
print(iou_list)
print(msd_list)