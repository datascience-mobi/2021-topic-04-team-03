from skimage.io import imread_collection
from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
import matplotlib.pyplot as plt

# use of median filter on NIH3T3

NIH3T3_gt = str(pl.Path('../Data/NIH3T3/img/*.png'))
NIH3T3_im = str(pl.Path('../Data/NIH3T3/gt/*.png'))

read_NIH3T3_gt = imread_collection(NIH3T3_gt)
read_NIH3T3_img = imread_collection(NIH3T3_im)

median_list = []
dice_list = []
msd_list = []
hausdorff_list = []

for index in range(len(read_NIH3T3_img)):
    image = read_NIH3T3_img[index].copy()

    gt = read_NIH3T3_img[index]
    gt[gt > 0] = 1
    image_fil = preprocessing.median_filter(image, 9)
    thresholds_NIH3T3 = otsu.otsu_twolevel(image_fil)
    image_seg = otsu.clipping_twolevel(image_fil, thresholds_NIH3T3)
    dice_score = evaluation.dice(image_seg, gt)
    dice_list.append(dice_score)
    print(dice_score)

    hausdorff_list.append(evaluation.hausdorff(image_seg, gt))
    msd_list.append(evaluation.msd(image_seg, gt))

    '''
    median_filter_NIH3T3 = preprocessing.median_filter(read_NIH3T3_img[index])
    plt.imshow(median_filter_NIH3T3, 'gray')
    plt.show()
    thresholds_NIH3T3 = otsu.otsu_twolevel(median_filter_NIH3T3)
    segmented_NIH3T3 = otsu.clipping_twolevel(median_filter_NIH3T3, thresholds_NIH3T3)
    dsc = evaluation.dice(segmented_NIH3T3, read_NIH3T3_gt[index])
    iou = evaluation.iou(segmented_NIH3T3, read_NIH3T3_gt[index])
    msd = evaluation.msd(segmented_NIH3T3, read_NIH3T3_gt[index])

    median_list.append(dsc)
    median_list.append(iou)
    median_list.append(msd)

print(median_list)
'''
print(dice_list)