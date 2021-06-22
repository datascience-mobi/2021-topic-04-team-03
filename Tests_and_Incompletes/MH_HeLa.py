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

HeLa_img_collection = io.imread_collection(str(pl.Path('../Data/N2DL-HeLa/img/*.tif')))
HeLa_gt_collection = io.imread_collection(str(pl.Path('../Data/N2DL-HeLa/gt/*.tif')))

dice_list = []
hausdorff_list = []
msd_list = []
for image_index in range(len(HeLa_img_collection)):
    image = HeLa_img_collection[image_index].copy()

    gt = HeLa_gt_collection[image_index]
    gt[gt>0] = 1

    image_seg = mh(image, 2 ** 16, 9)
    dice_score = evaluation.dice(image_seg, gt)
    dice_list.append(dice_score)
    print(dice_score)
    comparison_plot(image, image_seg, gt, dice_score)

    hausdorff_list.append(evaluation.hausdorff(image_seg, gt))
    msd_list.append(evaluation.msd(image_seg, gt))

print('Dice: ' + str(np.mean(dice_list)))
print('Hausdorff: ' + str(np.mean(hausdorff_list)))
print('MSD: ' + str(np.mean(msd_list)))