import nuclei_segmentation
from skimage.io import imread_collection
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
from nuclei_segmentation import preprocessing
from nuclei_segmentation import all_functions

N2DH_GOWT1_img = str(pl.Path('../Data/N2DH-GOWT1/img/*.tif'))
N2DH_GOWT1_gt = str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif'))

N2DL_HeLa_img = str(pl.Path('../Data/N2DL-HeLa/img/*.tif'))
N2DL_HeLa_gt = str(pl.Path('../Data/N2DL-HeLa/gt/*.tif'))

NIH3T3_img = str(pl.Path('../Data/NIH3T3/img/*.png'))
NIH3T3_gt = str(pl.Path('../Data/NIH3T3/gt/*.png'))

col_img_GOWT1 = imread_collection(N2DH_GOWT1_img)
col_gt_GOWT1 = imread_collection(N2DH_GOWT1_gt)

col_img_HeLa = imread_collection(N2DL_HeLa_img)
col_gt_HeLa = imread_collection(N2DL_HeLa_gt)

col_img_NIH3T3 = imread_collection(NIH3T3_img)
col_gt_NIH3T3 = imread_collection(NIH3T3_gt)


# without preprocessing

without_preprocessing_GOWT1 = all_functions.without_preprocessing_function(col_img_GOWT1, col_gt_GOWT1, intensity_lvls=2**16)
without_preprocessing_HeLa = all_functions.without_preprocessing_function(col_img_HeLa, col_gt_HeLa, intensity_lvls=2**16)
without_preprocessing_NIH3T3 = all_functions.without_preprocessing_function(col_img_NIH3T3, col_gt_NIH3T3, intensity_lvls=256)

print(without_preprocessing_GOWT1)
print(without_preprocessing_HeLa)
print(without_preprocessing_NIH3T3)

# gauss function - values for dice, msd and hsd

gauss_GOWT1 = all_functions.gauss_function_application(col_img_GOWT1, col_gt_GOWT1, intensity_lvls=2**16)
gauss_HeLa = all_functions.gauss_function_application(col_img_HeLa, col_gt_HeLa, intensity_lvls=2**16)
gauss_NIH3T3 = all_functions.gauss_function_application(col_img_NIH3T3, col_gt_NIH3T3, intensity_lvls=256)

print(gauss_GOWT1)
print(gauss_HeLa)
print(gauss_NIH3T3)

# median function - values for dice, msd and hsd

median_GOWT1 = all_functions.median_function_application(col_img_GOWT1, col_gt_GOWT1, intensity_lvls=2**16)
median_HeLa = all_functions.median_function_application(col_img_HeLa, col_gt_HeLa, intensity_lvls=2**16)
median_NIH3T3 = all_functions.median_function_application(col_img_NIH3T3, col_gt_NIH3T3, intensity_lvls=256)

print(median_GOWT1)
print(median_HeLa)
print(median_NIH3T3)