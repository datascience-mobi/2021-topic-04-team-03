import json
import numpy as np


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


if __name__ == "__main__":
    from nuclei_segmentation import visualisation, complete_analysis

    scores = get_all_two_level_results("two_lvl.json", "NIH3T3")
    print(np.argmax([np.mean(i) for i in scores]))
    scores_one_lvl = complete_analysis.result_evaluation("values.json", dataset_names=["NIH3T3"], return_data=True)

    ax1 = visualisation.comparison_swarmplot(scores,
                                             x_label=["No preprocessing", "Histogram stretching", "Median filter",
                                                      "Median filter and\nhistogram stretching", "Gaussian filter"],
                                             plot=False)

    ax2 = visualisation.comparison_boxplot(scores,
                                           x_label=["No preprocessing", "Histogram stretching", "Median filter",
                                                    "Median filter and\nhistogram stretching", "Gaussian filter"],
                                           plot=False)
