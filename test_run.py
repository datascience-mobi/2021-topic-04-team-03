import numpy as np
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt
from nuclei_segmentation import otsu
from nuclei_segmentation import preprocessing
from nuclei_segmentation import evaluation

img_NIH3T3 = imread(r'Data\NIH3T3\img\dna-42.png')
gt_NIH3T3 = imread(r'Data\NIH3T3\gt\42.png')

plt.imshow(img_NIH3T3)
plt.title ("Original")
plt.show()

plt.imshow(gt_NIH3T3)
plt.title ("Ground truth")
plt.show()

# One level Otsu
threshold_NIH3T3, goodness = otsu.otsu_faster(img_NIH3T3)
clipped_NIH3T3 = otsu.clipping(img_NIH3T3, threshold_NIH3T3)

dc_clipped_NIH3T3 = evaluation.dice(clipped_NIH3T3, gt_NIH3T3)
print("One level Otsu: " + str(dc_clipped_NIH3T3))

plt.imshow(clipped_NIH3T3)
plt.title ("One level clipped")
plt.show()

# Two level Otsu for reflection correction

two_level_thresholds = otsu.otsu_twolevel(img_NIH3T3)
two_level_clipped_NIH3T3 = preprocessing.two_level_reflection(img_NIH3T3, two_level_thresholds)

dc_two_level_NIH3T3 = evaluation.dice(two_level_clipped_NIH3T3, gt_NIH3T3)
print("Two level Otsu (reflection correction): " + str(dc_two_level_NIH3T3))

plt.imshow(two_level_clipped_NIH3T3)
plt.title ("Two level clipped")
plt.show()

# ----------
# Other dataset

img_GOWT1 = imread(r'Data\N2DH-GOWT1\img\t01.tif')
gt_GOWT1 = imread(r'Data/N2DH-GOWT1/gt/man_seg01.tif')

plt.imshow(img_GOWT1, 'gray')
plt.title ("GOWT1 original")
plt.show()

plt.imshow(gt_GOWT1, 'gray')
plt.title ("GOWT1 groundtruth")
plt.show()

GOWT1_segmented = otsu.complete_segmentation(img_GOWT1, 2*16)

plt.imshow(GOWT1_segmented, 'gray')
plt.title ("GOWT1 original, segmented")
plt.show()

dc_GOWT1_segmented = evaluation.dice(GOWT1_segmented, gt_GOWT1)
print('Original GOWT1: ' + str(dc_GOWT1_segmented))

# Preprocessing with histogram stretching

stretched_GOWT1 = preprocessing.stretch(img_GOWT1, intensity_lvls=2**16)
stretched_GOWT1_segmented = otsu.complete_segmentation(stretched_GOWT1, 2**16)

plt.imshow(stretched_GOWT1_segmented, 'gray')
plt.title ("GOWT1 histogram stretched, segmented")
plt.show()

dc_stretched_GOWT1 = evaluation.dice(stretched_GOWT1_segmented, gt_GOWT1)
print("GOWT1 histogram stretched: " + str(dc_stretched_GOWT1))

# Preprocessing with filters
# gauss filter

### DOESN'T WORK
gauss_kernel = preprocessing.gaussian_kernel(length=5, sigma=1)
gauss_GOWT1 = preprocessing.convolution(img_GOWT1, gauss_kernel)
gauss_GOWT1_segmented = otsu.complete_segmentation(gauss_GOWT1, 2**16)

plt.imshow(gauss_GOWT1_segmented, 'gray')
plt.title ("GOWT1 Gaussian filter, segmented")
plt.show()

dc_gauss_GOWT1 = evaluation.dice(gauss_GOWT1_segmented, gt_GOWT1)
print("GOWT1 Gaussian filter: " + str(dc_gauss_GOWT1))

# median filter
median_GOWT1 = preprocessing.median_filter(img_GOWT1)
median_GOWT1_segmented = otsu.complete_segmentation(median_GOWT1, 2**16)

plt.imshow(median_GOWT1_segmented, 'gray')
plt.title ("GOWT1 median filter, segmented")
plt.show()

dc_median_GOWT1 = evaluation.dice(median_GOWT1_segmented, gt_GOWT1)
print("GOWT1 median filter: " + str(dc_median_GOWT1))

# Median filter after histogram stretching
median_stretched_GOWT1 = preprocessing.median_filter(stretched_GOWT1)
median_stretched_GOWT1_segmented = otsu.complete_segmentation(median_stretched_GOWT1, 2**16)

plt.imshow(median_stretched_GOWT1_segmented, 'gray')
plt.title ("GOWT1 median filter after histogram stretching, segmented")
plt.show()

dc_median_stretched_GOWT1 = evaluation.dice(median_stretched_GOWT1_segmented, gt_GOWT1)
print('GOWT1 median filter after histogram stretching: ' + str(dc_median_stretched_GOWT1))

# Histogram stretching after median filter
stretched_median_GOWT1 = preprocessing.stretch(median_GOWT1, 2**16)
stretched_median_GOWT1_segmented = otsu.complete_segmentation(stretched_median_GOWT1, 2**16)

plt.imshow(stretched_median_GOWT1_segmented, 'gray')
plt.title ("GOWT1 histogram stretching after median filter, segmented")
plt.show()

dc_stretched_median_GOWT1 = evaluation.dice(stretched_median_GOWT1_segmented, gt_GOWT1)
print('GOWT1 histogram stretching after median filter: ' + str(dc_stretched_median_GOWT1))