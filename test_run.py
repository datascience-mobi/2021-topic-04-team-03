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


