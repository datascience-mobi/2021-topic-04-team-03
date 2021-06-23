import nuclei_segmentation
from skimage.io import imread_collection
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
from nuclei_segmentation import preprocessing

col_dir_img_gowt = str(pl.Path('Data/N2DH-GOWT1/img/*.tif'))
col_dir_gt_gowt = str(pl.Path('Data/N2DH-GOWT1/gt/*.tif'))

col_dir_img_hela = str(pl.Path('Data/N2DL-HeLa/img/*.tif'))
col_dir_gt_hela = str(pl.Path('Data/N2DL-HeLa/gt/*.tif'))

col_dir_img_nih = str(pl.Path('Data/NIH3T3/img/*.png'))
col_dir_gt_nih = str(pl.Path('Data/NIH3T3/gt/*.png'))



def gauss_stretch(path_img, path_gt, intensity = 256):
    col_img = imread_collection(path_img)
    col_gt = imread_collection(path_gt)

    kernel = preprocessing.gaussian_kernel(5,1)

    gauss_list = []
    for image in col_img:
        gauss_list.append(preprocessing.convolution(image, kernel))

    stretch_list = []

    for gauss_img in gauss_list:
        stretch_list.append(preprocessing.histogram_stretching(gauss_img, intensity_lvls = intensity))


    clipped_images = []

    for stretch_img in stretch_list:
        clipped_images.append(otsu.complete_segmentation(stretch_img, intensity_lvls = intensity))

    print(len(clipped_images))

    gt_list = []

    for gt_image in col_gt:
        gt_list.append(gt_image)



    dice = []
    msd = []
    hausdorff_list = []

    for i in range(len(clipped_images)):
        dice.append(evaluation.dice(clipped_images[i], gt_list[i]))
        msd.append(evaluation.msd(clipped_images[i],gt_list[i]))
        hausdorff_list.append(evaluation.hausdorff(clipped_images[i],gt_list[i]))

    return dice, msd,hausdorff_list

dc_hela,msd_hela,hd_hela = gauss_stretch(col_dir_img_hela,col_dir_gt_hela, intensity=2**16)

dc_gowt,msd_gowt,hd_gowt = gauss_stretch(col_dir_img_gowt,col_dir_gt_gowt, intensity=2**16)

dc_nih,msd_nih,hd_nih = gauss_stretch(col_dir_img_nih,col_dir_gt_nih, intensity=256)


print(dc_hela)
print(msd_hela)
print(hd_hela)

print(dc_gowt)
print(msd_gowt)
print(hd_gowt)

print(dc_nih)
print(msd_nih)
print(hd_nih)






