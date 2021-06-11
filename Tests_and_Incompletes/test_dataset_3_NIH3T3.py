import numpy as np
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt
import pathlib as pl
from nuclei_segmentation import otsu
from nuclei_segmentation import preprocessing
from nuclei_segmentation import evaluation
from nuclei_segmentation import metrics
from nuclei_segmentation import visualisation

img_NIH3T3 = imread(str(pl.Path(r'Data\NIH3T3\img\dna-42.png')))
gt_NIH3T3 = imread(str(pl.Path(r'Data\NIH3T3\gt\42.png')))

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

# Cell counting

cell_number = metrics.cell_counting(gt_NIH3T3)
print('There are ' + str(cell_number) + ' cells on the NIH3T3 image.')

border_pixels_NIH3T3 = metrics.find_border(gt_NIH3T3)
visualisation.border_image(gt_NIH3T3, border_pixels_NIH3T3, 5)