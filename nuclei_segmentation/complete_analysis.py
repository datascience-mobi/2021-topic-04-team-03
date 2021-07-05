import json
import numpy as np
import pathlib as pl
from skimage.io import imread_collection
from nuclei_segmentation import all_functions as af


def optimal_combination_filter_size(path_to_file, method):
    with open(path_to_file, "r") as file:
        json_object = json.load(file)
    scores = []
    filter_sizes = []
    for filter_size in json_object[method]:
        filter_sizes.append(filter_size)
        scores.append(np.mean(json_object[method][filter_size]["Dice Score"]))

    return filter_sizes[np.argmax(scores)], np.max(scores)

def optimal_combination_no_filter_size(path_to_file, evaluation_method, dataset):
    with open(path_to_file, "r") as file:
        json_object = json.load(file)
    scores = []
    methods = []
    for method in json_object:
        methods.append(method)
        if dataset in json_object[method]:
            scores.append(np.mean(json_object[method][dataset][evaluation_method]))

    return methods[np.argmax(scores)], max(scores)

def get_all_two_level_results(path_to_file, dataset):
    scores = []

    with open(path_to_file, "r") as file:
        json_object = json.load(file)
    for method in ["No preprocessing", "Histogram stretching"]:
        scores.append(json_object[method][dataset]["Dice Score"])
    for method in ["Median filter", "Median filter and histogram stretching", "Gaussian filter"]:
        optimal_size = optimal_combination_filter_size(path_to_file, method)[0]
        scores.append(json_object[method][optimal_size]["Dice Score"])

    return scores

def get_one_lvl_dice_scores(data_one_level, dataset = 'NIH3T3'):
    dice_scores = []
    for method in ["No preprocessing", "Median filter", "Gaussian filter", "Histogram stretching",
                   "Median filter and histogram stretching", "Gauss filter and histogram stretching"]:
        dice_scores.append(data_one_level[method][dataset]["Dice Score"])
    return dice_scores

def get_one_lvl_msd(data_one_level, dataset = 'NIH3T3'):
    dice_scores = []
    for method in ["No preprocessing", "Median filter", "Gaussian filter", "Histogram stretching",
                   "Median filter and histogram stretching", "Gauss filter and histogram stretching"]:
        dice_scores.append(data_one_level[method][dataset]["MSD"])
    return dice_scores

def get_one_lvl_hd(data_one_level, dataset = 'NIH3T3'):
    dice_scores = []
    for method in ["No preprocessing", "Median filter", "Gaussian filter", "Histogram stretching",
                   "Median filter and histogram stretching", "Gauss filter and histogram stretching"]:
        dice_scores.append(data_one_level[method][dataset]["Hausdorff"])
    return dice_scores

def one_level_complete_calculation():
    col_img_GOWT1 = imread_collection(str(pl.Path('../Data/N2DH-GOWT1/img/*.tif')))
    col_gt_GOWT1 = imread_collection(str(pl.Path('../Data/N2DH-GOWT1/gt/*.tif')))

    col_img_HeLa = imread_collection(str(pl.Path('../Data/N2DL-HeLa/img/*.tif')))
    col_gt_HeLa = imread_collection(str(pl.Path('../Data/N2DL-HeLa/gt/*.tif')))

    col_img_NIH3T3 = imread_collection(str(pl.Path('../Data/NIH3T3/img/*.png')))
    col_gt_NIH3T3 = imread_collection(str(pl.Path('../Data/NIH3T3/gt/*.png')))

    img_col_list = [col_img_GOWT1, col_img_HeLa, col_img_NIH3T3]
    gt_col_list = [col_gt_GOWT1, col_gt_HeLa, col_gt_NIH3T3]
    col_names = ["N2DH-GOWT1", "N2DL-HeLa", "NIH3T3"]
    intensity_lvls_list = [2 ** 16, 2 ** 16, 256]
    median_filter_sizes = [3, 9, 3]
    gauss_filter_sizes = [25, 25, 25]
    sigma_list = [5, 5, 5]

    combinations = ["No preprocessing", "Median filter", "Gaussian filter", "Histogram stretching",
                    "Median filter and histogram stretching", "Gauss filter and histogram stretching"]

    for i in range(len(img_col_list)):
        # print("Started with "+col_names[i])
        no_preprocessing_scores = af.without_preprocessing_function_application(img_col_list[i], gt_col_list[i],
                                                                                intensity_lvls=intensity_lvls_list[i])
        # print("no preprocessing")
        median_filter_scores = af.median_function_application(img_col_list[i], gt_col_list[i],
                                                              intensity_lvls=intensity_lvls_list[i],
                                                              filter_size=median_filter_sizes[i])
        # print("median")
        gauss_filter_scores = af.gauss_function_application(img_col_list[i], gt_col_list[i],
                                                            intensity_lvls=intensity_lvls_list[i],
                                                            filter_size=gauss_filter_sizes[i],
                                                            sigma=sigma_list[i])
        # print("gauss")
        histogram_stretching_socres = af.histogram_stretching_function_application(img_col_list[i], gt_col_list[i],
                                                                                   intensity_lvls=intensity_lvls_list[
                                                                                       i])
        # print("hs")
        median_and_hist_scores = af.median_histogram_stretching_function_application(img_col_list[i], gt_col_list[i],
                                                                                     intensity_lvls=intensity_lvls_list[
                                                                                         i],
                                                                                     filter_size=median_filter_sizes[i])
        # print("median + hs")
        gauss_and_hist_scores = af.gauss_histogram_stretching_function_application(img_col_list[i], gt_col_list[i],
                                                                                   intensity_lvls=intensity_lvls_list[
                                                                                       i],
                                                                                   filter_size=gauss_filter_sizes[i],
                                                                                   sigma=sigma_list[i])
        # print("gauss + hs")
        all_scores = [no_preprocessing_scores, median_filter_scores, gauss_filter_scores, histogram_stretching_socres,
                      median_and_hist_scores, gauss_and_hist_scores]
        af.write_in_json(str(pl.Path("values.json")), combinations, col_names[i], all_scores)



def result_evaluation(json_file, dataset_names = ["NIH3T3", "N2DH-GOWT1", "N2DL-HeLa"]):
    '''

    :param json_file: File to save the data
    :param dataset_names: Names of the datasets
    :return: Prints mean dsc, msd and hd for best preprocessing method of every dataset
    '''
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        methods = [name for name in data]
        for dataset in dataset_names:
            dice_scores = []
            msd_scores = []
            hd_scores = []
            for combination in data:
                dice_scores.append(np.mean(data[combination][dataset]["Dice Score"]))
                msd_scores.append(np.mean(data[combination][dataset]["MSD"]))
                hd_scores.append(np.mean(data[combination][dataset]["Hausdorff"]))
            print(dataset + ' (dice) ' + ': ' + str(round(np.max(dice_scores),3)) + '   --->   ' + str(
                methods[np.argmax(dice_scores)]))
            print(dataset + ' (msd) ' + ': ' + str(round(np.min(msd_scores),3)) + '   --->   ' + str(
                methods[np.argmin(msd_scores)]))
            print(
                dataset + ' (hd) ' + ': ' + str(round(np.min(hd_scores),3)) + '   --->   ' + str(methods[np.argmin(hd_scores)]))


def recalculation_desired(recalculate_data = False, path_to_data = "Results/values.json"):
    '''
    If recalculation of the data in the respective json files is desired, the parameter recalculate_data can be set to True.

    :param recalculate_data: True if data is wanted to be recalculated
    :param path_to_data: Path to the json file
    :return: -
    '''
    recalculate_one_level = False

    result_json_path_one_level = pl.Path(path_to_data)
    if recalculate_one_level or not result_json_path_one_level.exists():
        data = one_level_complete_calculation()
        with open(result_json_path_one_level) as file:
            file.write(data)
    else:
        with open(str(result_json_path_one_level), "r") as file:
            data = json.load(file)
    return data


if __name__ == '__main__':

    result_evaluation('../Results/values.json', dataset_names = ["N2DH-GOWT1", "N2DL-HeLa", "NIH3T3"])
    recalculation_desired(recalculate_data=False, path_to_data='../Results/values.json')