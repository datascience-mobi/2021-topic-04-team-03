from skimage.io import imread_collection
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
from nuclei_segmentation import preprocessing
import numpy as np
import matplotlib.pyplot as plt


def comparison_plot (image, processed_image, image_seg, gt, dice_score):
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Dice Score: ' + str(dice_score))
    ax[0][0].imshow(image, 'gray')
    ax[0][0].set_title('Original image')
    ax[0][1].imshow(processed_image, 'gray')
    ax[0][1].set_title('Processed image')
    ax[1][0].imshow(image_seg, 'gray')
    ax[1][0].set_title('Segmented image')
    ax[1][1].imshow(gt, 'gray')
    ax[1][1].set_title('Ground truth')
    ax[0][0].set_axis_off()
    ax[0][1].set_axis_off()
    ax[1][0].set_axis_off()
    ax[1][1].set_axis_off()
    # for ax_i in ax.ravel():
    #     ax_i.set_axis_off()
    fig.show()

col_dir_img_nih = str(pl.Path('Data/N2DL-HeLa/img/*.tif'))
col_dir_gt_nih = str(pl.Path('Data/N2DL-HeLa/gt/*.tif'))

N2DL_img_collection = imread_collection(col_dir_img_nih)
N2DL_gt_collection = imread_collection(col_dir_gt_nih)

dice_list = []
msd_list = []
hausdorff_list = []
for index in range(len(N2DL_img_collection)):
    image = N2DL_img_collection[index]
    gt = N2DL_gt_collection[index]
    gt[gt > 0 ] = 1

    stretched_image = preprocessing.histogram_stretching(image, intensity_lvls=2**16)

    clipped_image = otsu.complete_segmentation(stretched_image, intensity_lvls=2**16)
    dice_score = evaluation.dice(clipped_image,gt)
    dice_list.append(dice_score)
    msd_list.append(evaluation.msd(clipped_image, gt))
    hausdorff_list.append(evaluation.hausdorff(clipped_image, gt))
    comparison_plot(image, stretched_image, clipped_image, gt, dice_score)

print('Mean Dice Score of the N2DL-HeLa dataset: ' +str(np.mean(dice_list)))





