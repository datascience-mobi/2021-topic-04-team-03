from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
import matplotlib.pyplot as plt

def without_preprocessing_function_application(col_img, col_gt, intensity_lvls):
    dice_list = []
    msd_list = []
    hausdorff_list = []
    for index in range(len(col_img)):
        plt.imshow(col_img[index], 'gray')
        plt.show()
        segmented_NIH3T3 = otsu.complete_segmentation(col_img[index], intensity_lvls)
        dsc = evaluation.dice(segmented_NIH3T3, col_gt[index])
        msd = evaluation.msd(segmented_NIH3T3, col_gt[index])
        hausdorff = evaluation.hausdorff(segmented_NIH3T3, col_gt[index])

        dice_list.append(dsc)
        msd_list.append(msd)
        hausdorff_list.append(hausdorff)
    return dice_list, msd_list, hausdorff_list




def gauss_function_application(col_img, col_gt, intensity_lvls):

    kernel = preprocessing.gaussian_kernel(11, 5)
    gauss_list = []
    a = 0
    for image in col_img:
        gauss_list.append(preprocessing.convolution(image, kernel))
        print(a)
        a += 1
    print(len(gauss_list))
    clipped_images = []
    for gauss_img in gauss_list:
        clipped_images.append(otsu.complete_segmentation(gauss_img, intensity_lvls))
    print(len(clipped_images))
    gt_list = []
    for gt_image in col_gt:
        gt_list.append(gt_image)
    dice_list = []
    msd_list = []
    hausdorff_list = []
    for i in range(len(clipped_images)):
        dice_list.append(evaluation.dice(clipped_images[i], gt_list[i]))
        msd_list.append(evaluation.msd(clipped_images[i], gt_list[i]))
        hausdorff_list.append(evaluation.hausdorff(clipped_images[i], gt_list[i]))

    return dice_list, msd_list, hausdorff_list




# use of median filter on N2DH_GOWT1

def median_function_application(col_img, col_gt, intensity_lvls):
    dice_list = []
    msd_list = []
    hausdorff_list = []
    for index in range(len(col_img)):
        median_filter_N2DH_GOWT1 = preprocessing.histogram_stretching(col_img[index])
        plt.imshow(median_filter_N2DH_GOWT1, 'gray')
        plt.show()
        segmented_N2DH_GOWT1 = otsu.complete_segmentation(median_filter_N2DH_GOWT1, intensity_lvls)
        dsc = evaluation.dice(segmented_N2DH_GOWT1, col_gt[index])
        msd = evaluation.msd(segmented_N2DH_GOWT1, col_gt[index])
        hausdorff = evaluation.iou(segmented_N2DH_GOWT1, col_gt[index])

        dice_list.append(dsc)
        msd_list.append(msd)
        hausdorff_list.append(hausdorff)
    return dice_list, msd_list, hausdorff_list


