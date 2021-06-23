from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
from skimage.io import imread_collection

def without_preprocessing_function_application(col_img, col_gt, intensity_lvls = 256):
    dice_list = []
    msd_list = []
    hausdorff_list = []
    for index in range(len(col_img)):
        segmented_img = otsu.complete_segmentation(col_img[index], intensity_lvls)
        dsc = evaluation.dice(segmented_img, col_gt[index])
        msd = evaluation.msd(segmented_img, col_gt[index])
        hausdorff = evaluation.hausdorff(segmented_img, col_gt[index])

        dice_list.append(dsc)
        msd_list.append(msd)
        hausdorff_list.append(hausdorff)
    return dice_list, msd_list, hausdorff_list




def gauss_function_application(col_img, col_gt, intensity_lvls = 256):

    kernel = preprocessing.gaussian_kernel(11, 5)
    gauss_list = []
    for image in col_img:
        gauss_list.append(preprocessing.convolution(image, kernel))
    clipped_images = []
    for gauss_img in gauss_list:
        clipped_images.append(otsu.complete_segmentation(gauss_img, intensity_lvls))
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




def median_function_application(col_img, col_gt, intensity_lvls = 256):
    dice_list = []
    msd_list = []
    hausdorff_list = []
    for index in range(len(col_img)):
        median_filter_img = preprocessing.histogram_stretching(col_img[index])
        segmented_img = otsu.complete_segmentation(median_filter_img, intensity_lvls)
        dsc = evaluation.dice(segmented_img, col_gt[index])
        msd = evaluation.msd(segmented_img, col_gt[index])
        hausdorff = evaluation.iou(segmented_img, col_gt[index])

        dice_list.append(dsc)
        msd_list.append(msd)
        hausdorff_list.append(hausdorff)
    return dice_list, msd_list, hausdorff_list




def histogram_stretching_function_application(col_img, col_gt, intensity_lvls = 256):
    histogram_stretching_list = []
    for index in range(len(col_img)):
        histogram_stretching_img = preprocessing.histogram_stretching(col_img[index])
        segmented_img = otsu.complete_segmentation(histogram_stretching_img, intensity_lvls)
        dsc = evaluation.dice(segmented_img, col_gt[index])
        iou = evaluation.iou(segmented_img, col_gt[index])
        msd = evaluation.msd(segmented_img, col_gt[index])

        histogram_stretching_list.append(dsc)
        histogram_stretching_list.append(iou)
        histogram_stretching_list.append(msd)
    print(histogram_stretching_list)




def median_histogram_stretching_function_application(col_img, col_gt, filter_size = 3, intensity_lvls = 256):
    dice_list = []
    msd_list = []
    hausdorff_list = []
    for image_index in range(len(col_img)):
        image = col_img[image_index].copy()

        gt = col_gt[image_index]
        gt[gt > 0] = 1
        image = preprocessing.median_filter(image, filter_size)
        image_filtered = preprocessing.histogram_stretching(image, intensity_lvls)
        image_seg = otsu.complete_segmentation(image_filtered)

        dice_list.append(evaluation.dice(image_seg, gt))
        msd_list.append(evaluation.msd(image_seg, gt))
        hausdorff_list.append(evaluation.hausdorff(image_seg, gt))
    return dice_list, msd_list, hausdorff_list



def gauss_histogram_stretching_function_application(col_img, col_gt, intensity_lvls = 256):

    kernel = preprocessing.gaussian_kernel(5,1)

    gauss_list = []
    for image in col_img:
        gauss_list.append(preprocessing.convolution(image, kernel))

    stretch_list = []

    for gauss_img in gauss_list:
        stretch_list.append(preprocessing.histogram_stretching(gauss_img, intensity_lvls))

    clipped_images = []

    for stretch_img in stretch_list:
        clipped_images.append(otsu.complete_segmentation(stretch_img, intensity_lvls))

    gt_list = []

    for gt_image in col_gt:
        gt_list.append(gt_image)

    dice_list = []
    msd_list = []
    hausdorff_list = []

    for i in range(len(clipped_images)):
        dice_list.append(evaluation.dice(clipped_images[i], gt_list[i]))
        msd_list.append(evaluation.msd(clipped_images[i],gt_list[i]))
        hausdorff_list.append(evaluation.hausdorff(clipped_images[i],gt_list[i]))

    return dice_list, msd_list, hausdorff_list


if __name__ == "__main__":

    N2DH_GOWT1_img = str(pl.Path('../Data/N2DH-GOWT1/img/*.tif'))
    N2DH_GOWT1_gt = str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif'))

    N2DL_HeLa_img = str(pl.Path('../Data/N2DL-HeLa/img/*.tif'))
    N2DL_HeLa_gt = str(pl.Path('../Data/N2DL-HeLa/gt/*.tif'))

    NIH3T3_img = str(pl.Path('../Data/NIH3T3/img/*.png'))
    NIH3T3_gt = str(pl.Path('../Data/NIH3T3/gt/*.png'))

    col_img_GOWT1 = imread_collection(N2DH_GOWT1_img)
    col_gt_GOWT1 = imread_collection(N2DH_GOWT1_gt)

    col_img_HeLa = imread_collection(N2DL_HeLa_img)
    col_gt_HeLa = imread_collection(N2DL_HeLa_gt)

    col_img_NIH3T3 = imread_collection(NIH3T3_img)
    col_gt_NIH3T3 = imread_collection(NIH3T3_gt)

    # without preprocessing

    without_preprocessing_GOWT1 = without_preprocessing_function_application(col_img_GOWT1, col_gt_GOWT1,
                                                                                           intensity_lvls=2 ** 16)
    without_preprocessing_HeLa = without_preprocessing_function_application(col_img_HeLa, col_gt_HeLa,
                                                                                          intensity_lvls=2 ** 16)
    without_preprocessing_NIH3T3 = without_preprocessing_function_application(col_img_NIH3T3,
                                                                                            col_gt_NIH3T3)

    print(without_preprocessing_GOWT1)
    print(without_preprocessing_HeLa)
    print(without_preprocessing_NIH3T3)

    # gauss function - values for dice, msd and hsd

    gauss_GOWT1 = gauss_function_application(col_img_GOWT1, col_gt_GOWT1, intensity_lvls=2 ** 16)
    gauss_HeLa = gauss_function_application(col_img_HeLa, col_gt_HeLa, intensity_lvls=2 ** 16)
    gauss_NIH3T3 = gauss_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(gauss_GOWT1)
    print(gauss_HeLa)
    print(gauss_NIH3T3)

    # median function - values for dice, msd and hsd

    median_GOWT1 = median_function_application(col_img_GOWT1, col_gt_GOWT1, intensity_lvls=2 ** 16)
    median_HeLa = median_function_application(col_img_HeLa, col_gt_HeLa, intensity_lvls=2 ** 16)
    median_NIH3T3 = median_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(median_GOWT1)
    print(median_HeLa)
    print(median_NIH3T3)

    # histogram stretching function - values for dice, msd and hsd

    histogram_stretching_GOWT1 = histogram_stretching_function_application(col_img_GOWT1, col_gt_GOWT1,
                                                                                         intensity_lvls=2 ** 16)
    histogram_stretching_HeLa = histogram_stretching_function_application(col_img_HeLa, col_gt_HeLa,
                                                                                        intensity_lvls=2 ** 16)
    histogram_stretching_NIH3T3 = histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(histogram_stretching_GOWT1)
    print(histogram_stretching_HeLa)
    print(histogram_stretching_NIH3T3)

    # median and histogram stretching function - values for dice, msd and hsd

    median_histogram_stretching_GOWT1 = median_histogram_stretching_function_application(col_img_GOWT1, col_gt_GOWT1,
                                                                                                intensity_lvls=2 ** 16)
    median_histogram_stretching_HeLa = median_histogram_stretching_function_application(col_img_HeLa, col_gt_HeLa,
                                                                                                intensity_lvls=2 ** 16)
    median_histogram_stretching_NIH3T3 = median_histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(median_histogram_stretching_GOWT1)
    print(median_histogram_stretching_HeLa)
    print(median_histogram_stretching_NIH3T3)

    # median and histogram stretching function - values for dice, msd and hsd

    gauss_histogram_stretching_GOWT1 = gauss_histogram_stretching_function_application(col_img_GOWT1, col_gt_GOWT1,
                                                                                                     intensity_lvls=2 ** 16)
    gauss_histogram_stretching_HeLa = gauss_histogram_stretching_function_application(col_img_HeLa, col_gt_HeLa,
                                                                                                    intensity_lvls=2 ** 16)
    gauss_histogram_stretching_NIH3T3 = gauss_histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3)

    print(gauss_histogram_stretching_GOWT1)
    print(gauss_histogram_stretching_HeLa)
    print(gauss_histogram_stretching_NIH3T3)
