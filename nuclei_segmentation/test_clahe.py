
import matplotlib.pyplot as plt
import pathlib as pl
from nuclei_segmentation import otsu
from nuclei_segmentation import preprocessing
from nuclei_segmentation import evaluation
from nuclei_segmentation import metrics
from nuclei_segmentation import visualisation
from skimage.io import imshow
from skimage.io import imread
import numpy as np
from skimage import exposure


def stretch(img, threshold1, threshold2, intensity_lvls=256):
    """
    This function peforms histogram stretching and ignores the outliers (below lower and above upper quantile).
    The minimum intensity of the stretched image is zero, the maximum intensity is 256 (.png) or 2**16 (.tif)

    :param quantile: Quantile (pixels above or under are cut out)
    :param img: Input image
    :param intensity_lvls: 256 (.png) or 2**16 (.tif)
    :return: Stretched Image
    """

    #lower_quantile, upper_quantile = np.percentile(img, (threshold1, threshold2))

    workimg = img.copy()
    workimg[workimg < threshold1] = threshold1
    workimg[workimg > threshold2] = threshold1

    min_intensity_output = 0
    max_intensity_output = intensity_lvls
    min_intensity_input = int(np.min(workimg))
    max_intensity_input = int(np.max(workimg))

    stretched_image = (workimg - min_intensity_input) * (max_intensity_output - min_intensity_output) / \
                      (max_intensity_input - min_intensity_input)

    return stretched_image

img_NIH3T3 = imread(str(pl.Path(r'..\Data\NIH3T3\img\dna-42.png')))
gt_NIH3T3 = imread(str(pl.Path(r'..\Data\NIH3T3\gt\42.png')))
#
thresholds = otsu.otsu_twolevel(img_NIH3T3)
# img_stretched = stretch(img_NIH3T3, min(thresholds), max(thresholds))
# plt.imshow(img_stretched)
# plt.show()
# t = otsu.otsu(img_stretched)
# img_clipped = otsu.clipping(img_stretched, t)
# plt.imshow(img_clipped)
# plt.show()
# print(evaluation.dice(img_clipped, gt_NIH3T3))
#
# plt.imshow(gt_NIH3T3)
# plt.show()
#
# new_img = otsu.alternative_reflection_otsu(img_NIH3T3)

old = preprocessing.two_level_reflection(img_NIH3T3, thresholds)
print('old ' + str(evaluation.dice(old, gt_NIH3T3)))

# CLAHE filter
clahe = exposure.equalize_adapthist(img_NIH3T3)
# The function returns an image with maximal value 1, therefore the correction
clahe = clahe*255

new_t = otsu.otsu_twolevel(clahe)
new = preprocessing.two_level_reflection(clahe, new_t)
print('new ' + str(evaluation.dice(new, gt_NIH3T3)))

fig, ax = plt.subplots(1, 3)
gt_plot = ax[0]
old_plot = ax[1]
new_plot = ax[2]
gt_plot.imshow(gt_NIH3T3)
gt_plot.set_title('Ground truth')
old_plot.imshow(old)
old_plot.set_title('Old')
new_plot.imshow(new)
new_plot.set_title('CLAHE')
fig.show()