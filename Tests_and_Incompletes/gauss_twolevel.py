
from skimage.io import imread_collection
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
from nuclei_segmentation import preprocessing

col_dir_img = str(pl.Path('../Data/NIH3T3/img/*.png'))
col_dir_gt = str(pl.Path('../Data/NIH3T3/gt/*.png'))

col_img = imread_collection(col_dir_img)
col_gt = imread_collection(col_dir_gt)

kernel = preprocessing.gaussian_kernel(11, 5)


dice = []
msd = []
hausdorff_list = []
for index in range(len(col_img)):
    image = col_img[index]
    gt = col_gt[index]
    gauss_img = preprocessing.convolution(image, kernel)

    thresholds = otsu.otsu_twolevel(gauss_img)
    clipped_image =otsu.clipping_twolevel(gauss_img, thresholds)
    dice.append(evaluation.dice(clipped_image, gt))
    msd.append(evaluation.msd(clipped_image, gt))
    hausdorff_list.append(evaluation.hausdorff(clipped_image, gt))

print(dice)
print(msd)
print(hausdorff_list)


