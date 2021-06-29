import pathlib as pl
import numpy as np
from skimage.io import imread_collection
from nuclei_segmentation import all_functions as af

col_img_NIH3T3 = imread_collection(str(pl.Path('../Data/NIH3T3/img/*.png')))
col_gt_NIH3T3 = imread_collection(str(pl.Path('../Data/NIH3T3/gt/*.png')))

#
# no_preprocessing_scores = af.without_preprocessing_function_application(col_img_NIH3T3, col_gt_NIH3T3,
#                                                                         intensity_lvls=256,
#                                                                         mode="two_level")
# print("no preprocessing")
# hs_scores = af.histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3,
#                                                                            intensity_lvls=256,
#                                                                            mode="two_level")
# print("hs")
#
# af.write_in_json(str(pl.Path("two_lvl.json")), ["No preprocessing", "Histogram stretching"],
#                  "NIH3T3", [no_preprocessing_scores, hs_scores])

#
# for filter_size in range(31, 70, 7):
    # print(filter_size)
    # median_scores = af.median_function_application(col_img_NIH3T3, col_gt_NIH3T3,
    #                                                       intensity_lvls=256,
    #                                                       mode="two_level",
    #                                                       filter_size=filter_size)
    # print("median")
    # median_hs_scores = af.median_histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3,
    #                                                                              intensity_lvls=256,
    #                                                                              mode="two_level",
    #                                                                              filter_size=filter_size)
    # print("median + hs")
    # af.write_in_json(str(pl.Path("two_lvl.json")),
    #                  ["Median filter"],
    #                  filter_size,
    #                  [median_scores])

for filter_size in range(11, 50, 10):
    for sigma in range(1, 50, 10):

        sigma = sigma / 5
        print(filter_size, sigma)
        gauss_scores = af.gauss_function_application(col_img_NIH3T3, col_gt_NIH3T3,
                                                            intensity_lvls=256,
                                                            mode="two_level",
                                                            filter_size=filter_size,
                                                            sigma=sigma)
        print("gauss")

        # gauss_hs_scores = af.gauss_histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3,
        #                                                                            intensity_lvls=256,
        #                                                                            mode="two_level",
        #                                                                            filter_size=filter_size,
        #                                                                            sigma=sigma)
        # print("gauss and hs")

        af.write_in_json(str(pl.Path("two_lvl.json")),
                         ["Gaussian filter"],
                         (str(filter_size) + ", " + str(sigma)),
                         [gauss_scores])
