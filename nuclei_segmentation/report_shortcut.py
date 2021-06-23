import json
import numpy as np

def result_evaluation(json_file, dataset_names = ["NIH3T3", "N2DH-GOWT1", "N2DL-HeLa"]):
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


if __name__ == '__main__':

    result_evaluation('../Tests_and_Incompletes/values.json')
