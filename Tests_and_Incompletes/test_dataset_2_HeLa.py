
from skimage.io import imread
import matplotlib.pyplot as plt
import pathlib as pl
from nuclei_segmentation import otsu
from nuclei_segmentation import preprocessing
from nuclei_segmentation import evaluation

img_HeLa = imread(str(pl.Path(r'..\Data\N2DL-HeLa\img\t13.tif')))
gt_HeLa = imread(str(pl.Path(r'..\Data\N2DL-HeLa\gt\man_seg13.tif')))

plt.imshow(img_HeLa, 'gray')
plt.title("HeLa original")
plt.show()

plt.imshow(gt_HeLa, 'gray')
plt.title("HeLa ground truth")
plt.show()

HeLa_segmented = otsu.complete_segmentation(img_HeLa)

plt.imshow(HeLa_segmented, 'gray')
plt.title("HeLa original, segmented")
plt.show()

dc_HeLa_segmented = evaluation.dice(HeLa_segmented, gt_HeLa)
print('Original HeLa: ' + str(dc_HeLa_segmented))

# gauss filter

gauss_kernel = preprocessing.gaussian_kernel(length=5, sigma=1)
gauss_HeLa = preprocessing.convolution(img_HeLa, gauss_kernel)
gauss_HeLa_segmented = otsu.complete_segmentation(gauss_HeLa)

plt.imshow(gauss_HeLa_segmented, 'gray')
plt.title ("HeLa Gaussian filter, segmented")
plt.show()

dc_gauss_HeLa = evaluation.dice(gauss_HeLa_segmented, gt_HeLa)
print("HeLa Gaussian filter: " + str(dc_gauss_HeLa))

# median filter
median_HeLa = preprocessing.median_filter(img_HeLa)
median_HeLa_segmented = otsu.complete_segmentation(median_HeLa)

plt.imshow(median_HeLa_segmented, 'gray')
plt.title("HeLa median filter, segmented")
plt.show()

dc_median_GOWT1 = evaluation.dice(median_HeLa_segmented, gt_HeLa)
print("GOWT1 median filter: " + str(dc_median_GOWT1))
