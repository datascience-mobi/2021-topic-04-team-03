import numpy as np
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt
from matplotlib import colors
from nuclei_segmentation import otsu
from nuclei_segmentation import preprocessing
from nuclei_segmentation import evaluation
from nuclei_segmentation import metrics
from nuclei_segmentation import visualisation

img_NIH3T3 = imread(r'Data\NIH3T3\img\dna-42.png')
gt_NIH3T3 = imread(r'Data\NIH3T3\gt\42.png')

# plt.imshow(img_NIH3T3)
# plt.title ("Original")
# plt.show()
#
plt.imshow(gt_NIH3T3)
plt.title ("Ground truth")
plt.show()

two_level_thresholds = otsu.otsu_twolevel(img_NIH3T3)
two_level_clipped_NIH3T3 = preprocessing.two_level_reflection(img_NIH3T3, two_level_thresholds)

dc_two_level_NIH3T3 = evaluation.dice(two_level_clipped_NIH3T3, gt_NIH3T3)
print("Two level Otsu (reflection correction): " + str(dc_two_level_NIH3T3))

plt.imshow(two_level_clipped_NIH3T3)
plt.title ("Two level clipped")
plt.show()

intensity_lvls = 256

ground_thruth = gt_NIH3T3.copy()
test_image = two_level_clipped_NIH3T3.copy()
false_pixels = np.ma.masked_where(ground_thruth == test_image, test_image)
false_negatives = np.ma.masked_where(ground_thruth == 0, false_pixels)
#false_negatives = np.ma.mask_or(false_pixels, gt_0)
false_positives= np.ma.masked_where(ground_thruth ==  intensity_lvls-1, false_pixels)
#false_positives = np.ma.mask_or(false_pixels, gt_1)

plt.figure()
plt.imshow(test_image, 'gray', alpha = 0.8)
cmap_false_negatives = colors.ListedColormap(['red', 'none'])
plt.imshow(false_negatives, cmap=cmap_false_negatives)

cmap_false_positives = colors.ListedColormap(['blue', 'none'])
plt.imshow(false_positives, cmap=cmap_false_positives)

plt.plot(0, 0, ".", c = 'red', label = 'False negatives')
plt.plot(0, 0, ".", c = 'blue', label = 'False positives')
plt.legend()

#plt.title(title)

plt.show()
