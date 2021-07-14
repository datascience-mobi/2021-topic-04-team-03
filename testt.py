import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns

from nuclei_segmentation import metrics, complete_analysis, visualisation, evaluation, otsu, preprocessing
from skimage import io
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
import pathlib as pl
import json
import pandas as pd
import warnings
import matplotlib as mpl

def comparison_boxplot(scores,
                       x_label=['No preprocessing', 'Median filter', 'Gaussian filter',
                                'Histogram stretching', 'Median filter and\nhistogram stretching',
                                'Gauss filter and\nhistogram stretching'],
                       y_label='Dice Score',
                       plot=True):
    """
    Plots a boxplot, that shows evaluation scores for single images
    sorted by preprocessing methods.

    :param scores: List in which every element is a list with scores for the image
    :param x_label: List of preprocessing methods (same order as in scores)
    :param y_label: Description pof the y axis
    :return:
    """

    dataframe = pd.DataFrame(data=np.transpose(scores), columns=x_label)
    fig, ax = plt.subplots()

    ax = sns.boxplot(data=dataframe,
                     showmeans=True,
                     meanline=True,
                     palette='PuRd',
                     meanprops={'visible': True, 'color': 'k', 'ls': '-', 'lw': 1, 'label': 'Mean'},
                     medianprops={'visible': True, 'color': 'lightslategray', 'linestyle': '--', 'label': 'Median'},
                     )
    ax.set(ylabel=y_label,
           title='Comparison of different preprocessing methods',
           )
    h, l = ax.get_legend_handles_labels()
    position = ax.get_position()
    ax.legend(h[0:2], l[0:2], bbox_to_anchor=(position.width + 0.6, 0.6))
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=30,
                       horizontalalignment='right')
    fig.tight_layout()
    if plot:
        fig.savefig('plot.png')
    else:
        return fig

data_one_level = complete_analysis.recalculation_desired_one_lvl(recalculate_one_level=False,
                                                                 path_to_data="Results/values.json")

comparison_boxplot(complete_analysis.get_one_lvl_dice_scores(data_one_level, dataset = "N2DH-GOWT1"))