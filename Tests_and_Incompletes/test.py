from skimage.io import imread_collection
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation
from scipy.spatial import distance
from skimage.io import imread
from skimage.io import imshow
from matplotlib import pyplot as plt
import numpy as np



col_dir_img = [r'''..\Data\NIH3T3\im\*.png''']
col_dir_gt = [r'''..\Data\NIH3T3\gt\*.png''']

col_img = imread_collection(col_dir_img )
col_gt = imread_collection(col_dir_gt)

print(col_img)


clipped_list = []
gt_list = []


from skimage.filters import threshold_otsu
#t_skimage = threshold_otsu(col_img[6])
#print('fertige funktion: ' + str(t_skimage))

for i in col_img:
    threshold, goodness = otsu.otsu_faster(i)
    clipped_list.append(otsu.clipping(i,threshold))
    #t_skimage = threshold_otsu(i)
    #print(threshold)
    #print('funktion: ' + str(t_skimage))




for i in col_gt:
    gt_list.append(i)

for i in range(len(clipped_list)):
    print(evaluation.dice_score_faster(clipped_list[i], gt_list[i]))

r'''
plt.imshow(clipped_list[6], 'gray')
plt.show()

plt.imshow(gt_list[6], 'gray')
plt.show()


from skimage.filters import threshold_otsu
t_skimage = threshold_otsu(col_img[6])
print('fertige funktion: ' + str(t_skimage))

'''
r'''
im1 = clipped_list[0]
im2 = col_gt[0]
#print(distance.dice(im1.flatten(), im2.flatten()))



plt.imshow(im1,'gray')
plt.show()

plt.imshow(gt, 'gray')
plt.show()

#print(im1)
#print(gt)

gt[gt!= 0] = 1

print(gt)
print(im1)

intersection = np.sum(gt*im1)

print(np.sum(gt))
print(np.sum(im1))

print(intersection)

union = np.sum(im1) + np.sum(gt)
print(union)

dsc = 2*intersection/union

print(dsc)

print(evaluation.dice_score_faster(im1,gt))

'''
im1 = clipped_list[0]
gt = gt_list[0]

gt[gt != 0] = 1

im1[im1 != 0] = 1

intersection = np.sum(im1*gt)

plt.imshow(col_img[0], 'gray')
plt.show()

plt.imshow(col_gt[0], 'gray')
plt.show()