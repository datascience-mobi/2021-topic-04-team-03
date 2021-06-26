import pathlib as pl
from skimage.io import imread_collection
from nuclei_segmentation import all_functions as af

col_img_GOWT1 = imread_collection(str(pl.Path('../Data/N2DH-GOWT1/img/*.tif')))
col_gt_GOWT1 = imread_collection(str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif')))

col_img_HeLa = imread_collection(str(pl.Path('../Data/N2DL-HeLa/img/*.tif')))
col_gt_HeLa = imread_collection(str(pl.Path('../Data/N2DL-HeLa/gt/*.tif')))

col_img_NIH3T3 = imread_collection(str(pl.Path('../Data/NIH3T3/img/*.png')))
col_gt_NIH3T3 = imread_collection(str(pl.Path('../Data/NIH3T3/gt/*.png')))

img_col_list = [col_img_GOWT1, col_img_HeLa, col_img_NIH3T3]
gt_col_list = [col_img_GOWT1, col_gt_HeLa, col_gt_NIH3T3]
col_names = ["N2DH-GOWT1", "N2DL-HeLa", "NIH3T3"]
intensity_lvls_list = [2 ** 16, 2 ** 16, 256]
median_filter_sizes = [3, 9, 3]
gauss_filter_sizes = [25, 25, 25]
sigma_list = [5, 5, 5 ]

combinations = ["No preprocessing", "Median filter", "Gaussian filter", "Histogram stretching",
                "Median filter and histogram stretching", "Gauss filter and histogram stretching"]

for i in range(len(img_col_list)):
    print("Started with "+col_names[i])
    no_preprocessing_scores = af.without_preprocessing_function_application(img_col_list[i], gt_col_list[i],
                                                                            intensity_lvls=intensity_lvls_list[i])
    print("no preprocessing")
    median_filter_scores = af.median_function_application(img_col_list[i], gt_col_list[i],
                                                          intensity_lvls=intensity_lvls_list[i],
                                                          filter_size=median_filter_sizes[i])
    print("median")
    gauss_filter_scores = af.gauss_function_application(img_col_list[i], gt_col_list[i],
                                                        intensity_lvls=intensity_lvls_list[i],
                                                        filter_size=gauss_filter_sizes[i],
                                                        sigma=sigma_list[i])
    print("gauss")
    histogram_stretching_socres = af.histogram_stretching_function_application(img_col_list[i], gt_col_list[i],
                                                                               intensity_lvls=intensity_lvls_list[i])
    print("hs")
    median_and_hist_scores = af.median_histogram_stretching_function_application(img_col_list[i], gt_col_list[i],
                                                                                 intensity_lvls=intensity_lvls_list[i],
                                                                                 filter_size=median_filter_sizes[i])
    print("median + hs")
    gauss_and_hist_scores = af.gauss_histogram_stretching_function_application(img_col_list[i], gt_col_list[i],
                                                                               intensity_lvls=intensity_lvls_list[i],
                                                                               filter_size=gauss_filter_sizes[i],
                                                                               sigma=sigma_list[i])
    print("gauss + hs")
    all_scores = [no_preprocessing_scores, median_filter_scores, gauss_filter_scores, histogram_stretching_socres,
                  median_and_hist_scores, gauss_and_hist_scores]
    af.write_in_json(str(pl.Path("new_values.json")), combinations, col_names[i], all_scores)
