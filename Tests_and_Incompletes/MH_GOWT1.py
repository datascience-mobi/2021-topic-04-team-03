from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation

from skimage import io
import matplotlib.pyplot as plt
import pathlib as pl
import numpy as np

def mh(image, int_lvls, size):
    image = preprocessing.median_filter(image, filter_size=size)
    image = preprocessing.histogram_stretching(image, intensity_lvls=int_lvls)
    image = otsu.complete_segmentation(image, intensity_lvls=int_lvls)
    return image

def comparison_plot (image, image_seg, gt, dice_score):
    fig, ax = plt.subplots(1, 3)
    fig.suptitle(dice_score)
    ax[0].imshow(image)
    ax[0].set_title('Original image')
    ax[1].imshow(image_seg)
    ax[1].set_title('Segmented image')
    ax[2].imshow(gt)
    ax[2].set_title('Ground truth')
    for ax_i in ax.ravel():
        ax_i.set_axis_off()
    fig.show()

GOWT1_img_collection = io.imread_collection(str(pl.Path('../Data/N2DH-GOWT1/img/*.tif')))
GOWT1_gt_collection = io.imread_collection(str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif')))

dice_list = []
hausdorff_list = []
msd_list = []
for image_index in range(len(GOWT1_img_collection)):
    image = GOWT1_img_collection[image_index].copy()

    gt = GOWT1_gt_collection[image_index]
    gt[gt>0] = 1

    image_seg = mh(image, 2 ** 16, 3)
    dice_score = evaluation.dice(image_seg, gt)
    dice_list.append(dice_score)
    print(dice_score)
    comparison_plot(image, image_seg, gt, dice_score)

    hausdorff_list.append(evaluation.hausdorff(image_seg, gt))
    msd_list.append(evaluation.msd(image_seg, gt))

print('Dice: ' + str(np.mean(dice_list)))
print('Hausdorff: ' + str(np.mean(hausdorff_list)))
print('MSD: ' + str(np.mean(msd_list)))


#### Test for the best filter size ----> 3 ####

# dice_list = []
# for image_index in range(len(GOWT1_img_collection)):
#     image = GOWT1_img_collection[image_index].copy()
#
#     gt = GOWT1_gt_collection[image_index]
#     gt[gt>0] = 1
#     dice_list.append([])
#     for i in range (3, 70, 7):
#         image_seg = mh(image, 2 ** 16, i)
#         dice_score = evaluation.dice(image_seg, gt)
#         dice_list[image_index].append(dice_score)
#     #comparison_plot(image, image_seg, gt, dice_score)
#
# x = range(3, 70, 7)
#
# plt.plot(x, dice_list[0], '-o', color = 'black')
# plt.plot(x, dice_list[1], '-o', color = 'pink')
# plt.plot(x, dice_list[2], '-o', color = 'red')
# plt.plot(x, dice_list[3], '-o', color = 'blue')
# plt.plot(x, dice_list[4], '-o', color = 'green')
# plt.plot(x, dice_list[5], '-o', color = 'orange')
# plt.show()
