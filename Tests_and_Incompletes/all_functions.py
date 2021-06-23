import nuclei_segmentation
from skimage.io import imread_collection
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
from nuclei_segmentation import preprocessing

col_dir_img = str(pl.Path('../Data/N2DH-GOWT1/img/*.tif'))
col_dir_gt = str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif'))

col_img = imread_collection(col_dir_img)
col_gt = imread_collection(col_dir_gt)

kernel = preprocessing.gaussian_kernel(11, 5)

gauss_list = []
a = 0
for image in col_img:
    gauss_list.append(preprocessing.convolution(image, kernel))
    print(a)
    a += 1

print(len(gauss_list))

clipped_images = []

for gauss_img in gauss_list:
    clipped_images.append(otsu.complete_segmentation(gauss_img, intensity_lvls=2**16))

print(len(clipped_images))

gt_list = []

for gt_image in col_gt:
    gt_list.append(gt_image)



dice = []
msd = []
hausdorff_list = []

for i in range(len(clipped_images)):
    dice.append(evaluation.dice(clipped_images[i], gt_list[i]))
    msd.append(evaluation.msd(clipped_images[i], gt_list[i]))
    hausdorff_list.append(evaluation.hausdorff(clipped_images[i], gt_list[i]))

print(dice)
print(msd)
print(hausdorff_list)