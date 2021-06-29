import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns

def border_image(image, border_pixels, width=5):
    '''
    This Function takes a binary image and a list with the position of the border pixels of the image
    and plots an image with cell borders of custom width.
    To recieve the border pixels use 'find_border' in metrics

    :param image: Ground truth image
    :param border_pixels: List of border pixels of input image
    :param width: border width
    :return:Plots the image with cell borders
    '''

    fig, ax = plt.subplots()

    ax.imshow(image, 'gray')
    ax.set_axis_off()
    ax.set_title('Cell Border Visualization')
    x, y = zip(*border_pixels)
    ax.scatter(y, x, c='deeppink', s=width)

    fig.show()

def overlay (test_image, ground_truth, intensity_lvls = 256, title ='Overlay of groundtruth and test image'):
    '''
    This function plots overlay of ground truth and a test image.
    False negatives are shown in red, while false positives in blue.

    :param test_image: Image to be compared with ground truth
    :param ground_truth: Ground truth image
    :param intensity_lvls: Total number of intensity levels of the images
    :param title: Title of the plot
    :return: None
    '''
    test_image_corrected = test_image.copy()*intensity_lvls
    false_pixels = np.ma.masked_where(ground_truth == test_image_corrected, test_image_corrected)
    false_negatives = np.ma.masked_where(ground_truth == 0, false_pixels)
    false_positives = np.ma.masked_where(ground_truth == intensity_lvls - 1, false_pixels)

    cmap_false_negatives = colors.ListedColormap(['red', 'none'])
    cmap_false_positives = colors.ListedColormap(['blue', 'none'])

    plt.figure()

    plt.imshow(test_image_corrected, 'gray', alpha=0.8)
    plt.imshow(false_negatives, cmap=cmap_false_negatives)
    plt.imshow(false_positives, cmap=cmap_false_positives)

    plt.plot(0, 0, ".", c='red', label='False negatives')
    plt.plot(0, 0, ".", c='blue', label='False positives')
    plt.legend()

    plt.title(title)

    plt.show()


def comparison_preprocessing (scores, x_label = ['None', 'G', 'M', 'H', 'GH', 'MH'], y_label = 'Dice Score'):
    '''
    Plots a swarmplot, that shows evaluation scores for single images
    sorted by preprocessing methods. It also draws a line through the mean value.

    :param scores: list in which every element is a list with scores for an image
    :param x_label: List of preprocessing methods (same order as in scores)
    :return: none
    '''

    dataframe = pd.DataFrame(data=np.transpose(scores), columns=x_label)
    ax = sns.swarmplot(data=dataframe,
                       size=7,
                       palette='magma_r')
    ax = sns.boxplot(showmeans=True,
                     meanline=True,
                     meanprops={'color': 'k', 'ls': '-', 'lw': 1},
                     medianprops={'visible': False},
                     whiskerprops={'visible': False},
                     zorder=10,
                     data=dataframe,
                     showfliers=False,
                     showbox=False,
                     showcaps=False,
                     ax=ax)
    ax.set(xlabel='Preprocessing',
           ylabel=y_label,
           title='Comparison of different preprocessing methods')
    plt.show()


def comparison_plot (image1, image2, image3, image4,
                     title1, title2, title3, title4,
                     figure_title):
    '''
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
    '''

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




if __name__ == '__main__':
    from nuclei_segmentation import otsu
    from skimage.io import imread
    import pathlib as pl

    img = imread(str(pl.Path(r'..\Data\NIH3T3\img\dna-47.png')))
    our_img = otsu.complete_segmentation(img)
    gt = imread(str(pl.Path(r'..\Data\NIH3T3\gt\47.png')))

    overlay(our_img, gt)

    plt.imshow(our_img)
    plt.show()

    plt.imshow(gt)
    plt.show()

