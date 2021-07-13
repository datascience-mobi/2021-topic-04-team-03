import json
import numpy as np
import pathlib as pl
from skimage.io import imread_collection
from nuclei_segmentation import all_functions as af


def optimal_combination_filter_size(path_to_file, method):
    """
    This function determines the optimal filter size from a method of interest from data in a json file.

    :param path_to_file: Path to the file with results
    :param method: Method of interest
    :return: optimal filter size, attributed score
    """
    with open(path_to_file, "r") as file:
        json_object = json.load(file)
    scores = []
    filter_sizes = []
    for filter_size in json_object[method]:
        filter_sizes.append(filter_size)
        scores.append(np.mean(json_object[method][filter_size]["Dice Score"]))

    return filter_sizes[np.argmax(scores)], np.max(scores)


def optimal_combination_no_filter_size(path_to_file, evaluation_method, dataset):
    """
    Determines the optimal preprocessing strategy for data with only one or no filter size from a json file.

    :param path_to_file: Path to the file with results
    :param evaluation_method: Method use for evaluation of results
    :param dataset: Dataset of interest
    :return: Optimal preprocessing strategy, attributed score
    """
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
    """
    Extracts all data for two-level Otsu's segmentation from a json file.

    :param path_to_file: Path to file with results
    :param dataset: Dataset of interest
    :return: All scores for the given dataset
    """
    scores = []

    with open(path_to_file, "r") as file:
        json_object = json.load(file)
    for method in ["No preprocessing", "Histogram stretching"]:
        scores.append(json_object[method][dataset]["Dice Score"])
    for method in ["Median filter", "Median filter and histogram stretching", "Gaussian filter"]:
        optimal_size = optimal_combination_filter_size(path_to_file, method)[0]
        scores.append(json_object[method][optimal_size]["Dice Score"])

    return scores


def one_level_complete_calculation():
    """
    Carries out complete analysis (all datasets, all preprocessing strategies) with one-level Otsu's segmentation.
    Saves result in values.json

    :return: None
    """
    col_img_GOWT1 = imread_collection(str(pl.Path('Data/N2DH-GOWT1/img/*.tif')))
    col_gt_GOWT1 = imread_collection(str(pl.Path('Data/N2DH-GOWT1/gt/*.tif')))

    col_img_HeLa = imread_collection(str(pl.Path('Data/N2DL-HeLa/img/*.tif')))
    col_gt_HeLa = imread_collection(str(pl.Path('Data/N2DL-HeLa/gt/*.tif')))

    col_img_NIH3T3 = imread_collection(str(pl.Path('Data/NIH3T3/img/*.png')))
    col_gt_NIH3T3 = imread_collection(str(pl.Path('Data/NIH3T3/gt/*.png')))

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
        no_preprocessing_scores = af.without_preprocessing_function_application(img_col_list[i], gt_col_list[i],
                                                                                intensity_lvls=intensity_lvls_list[i])

        median_filter_scores = af.median_function_application(img_col_list[i], gt_col_list[i],
                                                              intensity_lvls=intensity_lvls_list[i],
                                                              filter_size=median_filter_sizes[i])

        gauss_filter_scores = af.gauss_function_application(img_col_list[i], gt_col_list[i],
                                                            intensity_lvls=intensity_lvls_list[i],
                                                            filter_size=gauss_filter_sizes[i],
                                                            sigma=sigma_list[i])

        histogram_stretching_scores = af.histogram_stretching_function_application(img_col_list[i], gt_col_list[i],
                                                                                   intensity_lvls=intensity_lvls_list[
                                                                                       i])

        median_and_hist_scores = af.median_histogram_stretching_function_application(img_col_list[i], gt_col_list[i],
                                                                                     intensity_lvls=intensity_lvls_list[
                                                                                         i],
                                                                                     filter_size=median_filter_sizes[i])

        gauss_and_hist_scores = af.gauss_histogram_stretching_function_application(img_col_list[i], gt_col_list[i],
                                                                                   intensity_lvls=intensity_lvls_list[
                                                                                       i],
                                                                                   filter_size=gauss_filter_sizes[i],
                                                                                   sigma=sigma_list[i])

        all_scores = [no_preprocessing_scores, median_filter_scores, gauss_filter_scores, histogram_stretching_scores,
                      median_and_hist_scores, gauss_and_hist_scores]
        af.write_in_json(str(pl.Path("Results/values.json")), combinations, col_names[i], all_scores)


def result_evaluation(file_pathway, dataset_names=["NIH3T3", "N2DH-GOWT1", "N2DL-HeLa"], return_data=False):
    """
    Prints mean dsc, msd and hd for best preprocessing method of every dataset.

    :param return_data: If True returns all scores in a list
    :param file_pathway: File with saved the data
    :param dataset_names: Names of the datasets
    :return: List with all scores (if return_data == True)
    """
    with open(file_pathway, "r") as read_file:
        data = json.load(read_file)
    methods = [name for name in data]
    if return_data:
        all_scores = [[], [], []]
    for dataset in dataset_names:
        dice_scores = []
        msd_scores = []
        hd_scores = []
        for combination in data:
            dice_scores.append(np.mean(data[combination][dataset]["Dice Score"]))
            msd_scores.append(np.mean(data[combination][dataset]["MSD"]))
            hd_scores.append(np.mean(data[combination][dataset]["Hausdorff"]))
            if return_data:
                all_scores[0].append(data[combination][dataset]["Dice Score"])
                all_scores[1].append(data[combination][dataset]["MSD"])
                all_scores[2].append(data[combination][dataset]["Hausdorff"])
        print(dataset + ' (dice) ' + ': ' + str(round(np.max(dice_scores), 3)) + '   --->   ' + str(
            methods[np.argmax(dice_scores)]))
        print(dataset + ' (msd) ' + ': ' + str(round(np.min(msd_scores), 3)) + '   --->   ' + str(
            methods[np.argmin(msd_scores)]))
        print(
            dataset + ' (hd) ' + ': ' + str(round(np.min(hd_scores), 3)) + '   --->   ' + str(
                methods[np.argmin(hd_scores)]))
        if return_data:
            return all_scores


def two_level_complete_calculation():
    """
    Carries out complete analysis (all datasets, all preprocessing strategies) with one-level Otsu's segmentation.
    Saves result in two_lvl.json

    :return: None
    """
    col_img_NIH3T3 = imread_collection(str(pl.Path('Data/NIH3T3/img/*.png')))
    col_gt_NIH3T3 = imread_collection(str(pl.Path('Data/NIH3T3/gt/*.png')))

    no_preprocessing_scores = af.without_preprocessing_function_application(col_img_NIH3T3, col_gt_NIH3T3,
                                                                            intensity_lvls=256,
                                                                            mode="two_level")

    hs_scores = af.histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3,
                                                             intensity_lvls=256,
                                                             mode="two_level")

    af.write_in_json(str(pl.Path("Results/two_lvl.json")), ["No preprocessing", "Histogram stretching"],
                     "NIH3T3", [no_preprocessing_scores, hs_scores])

    for filter_size in range(31, 70, 7):
        print(filter_size)
        median_scores = af.median_function_application(col_img_NIH3T3, col_gt_NIH3T3,
                                                       intensity_lvls=256,
                                                       mode="two_level",
                                                       filter_size=filter_size)

        median_hs_scores = af.median_histogram_stretching_function_application(col_img_NIH3T3, col_gt_NIH3T3,
                                                                               intensity_lvls=256,
                                                                               mode="two_level",
                                                                               filter_size=filter_size)

        af.write_in_json(str(pl.Path("Results/two_lvl.json")), ["Median filter"], filter_size, [median_scores])
        af.write_in_json(str(pl.Path("Results/two_lvl.json")), ["Median filter and histogram stretching"],
                         filter_size, [median_hs_scores])

    for filter_size in range(11, 50, 10):
        for sigma in range(1, 50, 10):
            sigma = sigma / 5
            print(filter_size, sigma)
            gauss_scores = af.gauss_function_application(col_img_NIH3T3, col_gt_NIH3T3,
                                                         intensity_lvls=256,
                                                         mode="two_level",
                                                         filter_size=filter_size,
                                                         sigma=sigma)

            af.write_in_json(str(pl.Path("two_lvl.json")),
                             ["Gaussian filter"],
                             (str(filter_size) + ", " + str(sigma)),
                             [gauss_scores])


def recalculation_desired_one_lvl(recalculate_one_level=False, path_to_data="Results/values.json"):
    """
    If recalculation of the data in the respective json files is desired, the parameter recalculate_data can be set to
    True.

    :param recalculate_one_level: True if data is wanted to be recalculated
    :param path_to_data: Path to the json file
    :return: None
    """

    result_json_path_one_level = pl.Path(path_to_data)
    if recalculate_one_level or not result_json_path_one_level.exists():
        one_level_complete_calculation()
    with open(str(result_json_path_one_level), "r") as file:
        data = json.load(file)
    return data


def recalculation_desired_two_lvl(recalculate_two_level=False, path_to_data="Results/two_lvl.json"):
    """
    If recalculation of the data in the respective json files is desired, the parameter recalculate_data can be set to
    True.

    :param recalculate_two_level: True if data is wanted to be recalculated
    :param path_to_data: Path to the json file
    :return: None
    """

    result_json_path_two_level = pl.Path(path_to_data)
    if recalculate_two_level or not result_json_path_two_level.exists():
        two_level_complete_calculation()
    with open(str(result_json_path_two_level), "r") as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    result_evaluation('../Results/values.json', dataset_names=["N2DH-GOWT1", "N2DL-HeLa", "NIH3T3"])
    x = recalculation_desired_one_lvl(recalculate_one_level=False, path_to_data='../Results/values.json')
    y = recalculation_desired_two_lvl(recalculate_two_level=False, path_to_data='../Results/two_lvl.json')
    print(y)
