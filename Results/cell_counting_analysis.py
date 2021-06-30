import json
import pathlib as pl
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread_collection
from nuclei_segmentation import all_functions, metrics, visualisation

col_img_GOWT1 = imread_collection(str(pl.Path('../Data/N2DH-GOWT1/img/*.tif')))
col_gt_img_GOWT1 = imread_collection(str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif')))

col_img_HeLa = imread_collection(str(pl.Path('../Data/N2DL-HeLa/img/*.tif')))
col_gt_img_HeLa = imread_collection(str(pl.Path('../Data/N2DL-HeLa/gt/*.tif')))

calculated_number_GOWT1, gt_number_GOWT1, absolute_differences_GOWT1_gt, relative_differences_GOWT1_gt = all_functions.cell_counting_analysis(col_gt_img_GOWT1)
calculated_number_HeLa, gt_number_HeLa, absolute_differences_HeLa_gt, relative_differences_HeLa_gt = all_functions.cell_counting_analysis(col_gt_img_HeLa)

print(np.unique(absolute_differences_GOWT1_gt))
print(np.unique(absolute_differences_HeLa_gt))

json_object = {
    "N2DH-GOWT1":{
        "Calculated number": calculated_number_GOWT1,
        "Ground truth number": gt_number_GOWT1,
        "Absolute difference": absolute_differences_GOWT1_gt,
        "Relative difference": relative_differences_GOWT1_gt
},
    "N2DL-HeLa":{
        "Calculated number": calculated_number_HeLa,
        "Ground truth number": gt_number_HeLa,
        "Absolute difference": absolute_differences_HeLa_gt,
        "Relative difference": relative_differences_HeLa_gt
}
}
with open("cell_counting_results.json", 'w') as json_file:
    json.dump(json_object, json_file, indent=3)

img = col_gt_img_HeLa[2]
# border = metrics.find_border(img)
# visualisation.border_image(img, border, width = 0.05)

im2 = img[430:500, 380:480]
plt.imshow(im2)
plt.show()


