import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

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

if __name__ == '__main__':
    from nuclei_segmentation import otsu
    from skimage.io import imread
    import pathlib as pl

    img = imread(str(pl.Path(r'..\Data\NIH3T3\img\dna-0.png')))
    our_img = otsu.complete_segmentation(img)
    gt = imread(str(pl.Path(r'..\Data\NIH3T3\gt\0.png')))

    overlay(our_img, gt)

    plt.imshow(our_img)
    plt.show()

    plt.imshow(gt)
    plt.show()