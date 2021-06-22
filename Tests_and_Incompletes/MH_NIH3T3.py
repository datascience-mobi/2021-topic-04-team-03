from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation

from skimage import io
import matplotlib.pyplot as plt
import pathlib as pl
import numpy as np

def mh_no_seg(image, int_lvls, size):
    image = preprocessing.median_filter(image, filter_size=size)
    image = preprocessing.histogram_stretching(image, intensity_lvls=int_lvls)
    return image

def comparison_plot (image, processed_image, image_seg, gt, dice_score):
    fig, ax = plt.subplots(2, 2)
    fig.suptitle(dice_score)
    ax[0][0].imshow(image)
    ax[0][0].set_title('Original image')
    ax[0][1].imshow(processed_image)
    ax[0][1].set_title('Processed image')
    ax[1][0].imshow(image_seg)
    ax[1][0].set_title('Segmented image')
    ax[1][1].imshow(gt)
    ax[1][1].set_title('Ground truth')
    ax[0][0].set_axis_off()
    ax[0][1].set_axis_off()
    ax[1][0].set_axis_off()
    ax[1][1].set_axis_off()
    # for ax_i in ax.ravel():
    #     ax_i.set_axis_off()
    fig.show()

NIH3T3_img_collection = io.imread_collection(str(pl.Path('../Data/NIH3T3/img/*.png')))
NIH3T3_gt_collection = io.imread_collection(str(pl.Path('../Data/NIH3T3/gt/*.png')))

dice_list = []
hausdorff_list = []
msd_list = []
for image_index in range(len(NIH3T3_img_collection)):
    image = NIH3T3_img_collection[image_index].copy()

    gt = NIH3T3_gt_collection[image_index]
    gt[gt>0] = 1

    image_filtered = mh_no_seg(image, 256, 3)
    thresholds = otsu.otsu_twolevel(image_filtered)
    image_seg = otsu.clipping_twolevel(image_filtered, thresholds)
    dice_score = evaluation.dice(image_seg, gt)
    dice_list.append(dice_score)
    print(dice_score)
    comparison_plot(image, image_filtered, image_seg, gt, dice_score)

    hausdorff_list.append(evaluation.hausdorff(image_seg, gt))
    msd_list.append(evaluation.msd(image_seg, gt))

print('Dice: ' + str(np.mean(dice_list)))
print('Hausdorff: ' + str(np.mean(hausdorff_list)))
print('MSD: ' + str(np.mean(msd_list)))

