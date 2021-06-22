from skimage.io import imread_collection
from nuclei_segmentation import preprocessing
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
import pathlib as pl
import matplotlib.pyplot as plt

# use of median filter on N2DH_GOWT1

N2DH_GOWT1_gt = str(pl.Path('../Data/N2DH-GOWT1/img/*.tif'))
N2DH_GOWT1_im = str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif'))

read_N2DH_GOWT1_gt = imread_collection(N2DH_GOWT1_gt)
read_N2DH_GOWT1_img = imread_collection(N2DH_GOWT1_im)

median_list = []
dice_list = []
iou_list = []
msd_list = []

for index in range(len(read_N2DH_GOWT1_img)):
    median_filter_N2DH_GOWT1 = preprocessing.histogram_stretching(read_N2DH_GOWT1_img[index])
    plt.imshow(median_filter_N2DH_GOWT1, 'gray')
    plt.show()
    segmented_N2DH_GOWT1 = otsu.complete_segmentation(median_filter_N2DH_GOWT1, intensity_lvls=2 ** 16)
    dsc = evaluation.dice(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])
    iou = evaluation.iou(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])
    msd = evaluation.msd(segmented_N2DH_GOWT1, read_N2DH_GOWT1_gt[index])

    dice_list.append(dsc)
    iou_list.append(iou)
    msd_list.append(msd)

print(dice_list)
print(iou_list)
print(msd_list)
'''
######
# show plots

#median_fig, median_ax = plt.subplots(nrows=2, ncols=3)
#median_ax.imshow(median_list)


#### Test for the best filter size ----> 3 ####

dice_list = []
for image_index in range(len(read_N2DH_GOWT1_img)):
    print('image')
    image = read_N2DH_GOWT1_img[image_index].copy()

    gt = read_N2DH_GOWT1_gt[image_index]
    gt[gt>0] = 1
    dice_list.append([])
    for i in range(3, 70, 7):
        print('size')
        image_seg = preprocessing.median_filter(image, i)
        image_seg = otsu.complete_segmentation(image_seg, 2 ** 16)
        dice_score = evaluation.dice(image_seg, gt)
        dice_list[image_index].append(dice_score)

x = range(3, 70, 7)

plt.plot(x, dice_list[0], '-o', color = 'black')
plt.plot(x, dice_list[1], '-o', color = 'pink')
plt.plot(x, dice_list[2], '-o', color = 'red')
plt.plot(x, dice_list[3], '-o', color = 'blue')
plt.plot(x, dice_list[4], '-o', color = 'green')
plt.plot(x, dice_list[5], '-o', color = 'orange')
plt.show()
'''