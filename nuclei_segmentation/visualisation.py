import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns
from nuclei_segmentation import complete_analysis

def border_image(image, border_pixels, width=5):
    """
    This Function takes a binary image and a list with the position of the border pixels of the image
    and plots an image with cell borders of custom width.
    To recieve the border pixels use 'find_border' in metrics

    :param image: Ground truth image
    :param border_pixels: List of border pixels of input image
    :param width: border width
    :return:Plots the image with cell borders
    """

    fig, ax = plt.subplots()

    ax.imshow(image, 'gray')
    ax.set_axis_off()
    ax.set_title('Cell Border Visualization')
    x, y = zip(*border_pixels)
    ax.scatter(y, x, c='deeppink', s=width)

    fig.show()


def overlay(test_image, ground_truth,
            title='Overlay of ground truth and test image',
            plot=True):
    """
    This function plots overlay of ground truth and a test image.
    False negatives are shown in red, while false positives in blue.

    :param plot: if False it will return the figure instead of plotting
    :param test_image: Image to be compared with ground truth
    :param ground_truth: Ground truth image
    :param title: Title of the plot
    :return: None
    """

    ground_truth_corrected = np.zeros(ground_truth.shape)
    ground_truth_corrected[ground_truth > 0] = 1

    false_pixels = np.ma.masked_where(ground_truth_corrected == test_image, test_image)
    false_negatives = np.ma.masked_where(ground_truth_corrected == 0, false_pixels)
    false_positives = np.ma.masked_where(ground_truth_corrected == 1, false_pixels)

    cmap_false_negatives = colors.ListedColormap(['palevioletred', 'none'])
    cmap_false_positives = colors.ListedColormap(['cornflowerblue', 'none'])

    fig, ax = plt.subplots()

    ax.imshow(test_image, 'gray', alpha=0.8)
    ax.imshow(false_negatives, cmap=cmap_false_negatives)
    ax.imshow(false_positives, cmap=cmap_false_positives)

    ax.plot(0, 0, ".", c='palevioletred', label='False negatives')
    ax.plot(0, 0, ".", c='cornflowerblue', label='False positives')
    h, l = ax.get_legend_handles_labels()
    position = ax.get_position()
    ax.legend(h, l, bbox_to_anchor=(position.width + 0.8, 0.6))

    ax.set(title=title)
    ax.set_axis_off()

    fig.tight_layout()
    if plot:
        fig.show()
    else:
        return fig


def comparison_swarmplot(scores,
                         x_label=['No preprocessing', 'Median filter', 'Gaussian filter',
                                  'Histogram stretching', 'Median filter and\nhistogram stretching',
                                  'Gauss filter and\nhistogram stretching'],
                         y_label='Dice Score',
                         plot=True,
                         scale_axis = 0):
    """
    Plots a swarmplot, that shows evaluation scores for single images
    sorted by preprocessing methods. It also draws a line through the mean value.

    :param plot: if False it will return the figure instead of plotting
    :param scores: List in which every element is a list with scores for the image
    :param x_label: List of preprocessing methods (same order as in scores)
    :param y_label: Description pof the y axis
    :return: none
    """

    dataframe = pd.DataFrame(data=np.transpose(scores), columns=x_label)

    fig, ax = plt.subplots()
    ax = sns.swarmplot(data=dataframe,
                       size=7,
                       palette='Set2')
    ax = sns.boxplot(showmeans=True,
                     meanline=True,
                     meanprops={'color': 'k', 'ls': '-', 'lw': 1, 'label': 'Mean'},
                     medianprops={'visible': True, 'color': 'lightslategray', 'linestyle': '--', 'label': 'Median'},
                     whiskerprops={'visible': False},
                     zorder=10,
                     data=dataframe,
                     showfliers=False,
                     showbox=False,
                     showcaps=False,
                     ax=ax)
    h, l = ax.get_legend_handles_labels()
    position = ax.get_position()
    ax.legend(h[0:2], l[0:2], bbox_to_anchor=(position.width + 0.6, 0.6))

    ax.set(ylabel=y_label,
           title='Comparison of different preprocessing methods')
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=30,
                       horizontalalignment='right')
    fig.tight_layout()
    if scale_axis==1:
        ax.set_ylim([0,1.1])
    if plot:
        fig.show()
    else:
        return fig


def comparison_boxplot(scores,
                       x_label=['No preprocessing', 'Median filter', 'Gaussian filter',
                                'Histogram stretching', 'Median filter and\nhistogram stretching',
                                'Gauss filter and\nhistogram stretching'],
                       y_label='Dice Score',
                       plot=True, scale_axis = 0):
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
                     palette='Set2',
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
    if scale_axis==1:
        ax.set_ylim([0,1])
    fig.tight_layout()
    if plot:
        fig.show()
    else:
        return fig


def comparison_plot(image1, image2, image3, image4,
                    title1, title2, title3, title4,
                    figure_title):
    """
    Neat plot of four images + title

    :param image1: Image 1
    :param image2: Image 2
    :param image3: Image 3
    :param image4: Image 4
    :param title1: Title 1
    :param title2: Title 2
    :param title3: Title 3
    :param title4: Title 4
    :param figure_title: Heading
    :return: Neat plot of the four images
    """

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(figure_title)
    ax[0][0].imshow(image1, 'gray')
    ax[0][0].set_title(title1)
    ax[0][1].imshow(image2, 'gray')
    ax[0][1].set_title(title2)
    ax[1][0].imshow(image3, 'gray')
    ax[1][0].set_title(title3)
    ax[1][1].imshow(image4, 'gray')
    ax[1][1].set_title(title4)
    ax[0][0].set_axis_off()
    ax[0][1].set_axis_off()
    ax[1][0].set_axis_off()
    ax[1][1].set_axis_off()
    fig.show()


def two_img_plot(image1, image2, title1, title2, plot=True):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image1, 'gray')
    ax[0].set_title(title1, pad=20, loc="left")
    ax[1].imshow(image2, 'gray')
    ax[1].set_title(title2, pad=20, loc="left")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()
    if plot:
        fig.show()
    else:
        return fig


def three_img_plot(image1, image2, image3, title1, title2, title3, plot=True):
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image1, 'gray')
    ax[0].set_title(title1, pad=20, loc="left")
    ax[1].imshow(image2, 'gray')
    ax[1].set_title(title2, pad=20, loc="left")
    ax[2].imshow(image3, 'gray')
    ax[2].set_title(title3, pad=20, loc="left")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    plt.tight_layout()
    if plot:
        fig.show()
    else:
        return fig



if __name__ == '__main__':
    from nuclei_segmentation import otsu
    from skimage.io import imread
    import pathlib as pl

    img = imread(str(pl.Path(r'..\Data\NIH3T3\img\dna-47.png')))
    our_img = otsu.complete_segmentation(img)
    gt = imread(str(pl.Path(r'..\Data\NIH3T3\gt\47.png')))

    data_one_level = complete_analysis.recalculation_desired_one_lvl(recalculate_one_level=False,
                                                                     path_to_data=str(pl.Path(r'..\Results\values.json')))

    comparison_boxplot(complete_analysis.get_one_lvl_dice_scores(data_one_level, dataset="N2DH-GOWT1"),
                                       y_label="Hausdorff", scale_axis=1)