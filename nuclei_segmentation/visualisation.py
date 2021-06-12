import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def border_image(image, border_pixels, width):
    '''
    This Function takes a binary image and a list with the position of the border pixels of the image
    and plots an image with cell borders of custom width.
    To recieve the border pixels use 'find_border' in metrics

    :param image: Ground truth image
    :param border_pixels: List of border pixels of input image
    :param width: border width
    :return:Plots the imag with cell borders
    '''

    half_width = width//2

    border_image = image.copy()

    for pixel in border_pixels:
        for a in range(-half_width, half_width+1):
            for b in range(-half_width, half_width+1):
                if pixel[0] + a < border_image.shape[0] and pixel[1] + b < border_image.shape[1]:
                    border_image[pixel[0] + a][pixel[1] + b] = 120

    plt.imshow(border_image, 'PiYG')
    plt.show()

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

    img = imread(r'..\Data\NIH3T3\img\dna-0.png')
    our_img = otsu.complete_segmentation(img)
    gt = imread(r'..\Data\NIH3T3\gt\0.png')

    overlay(our_img, gt)

    plt.imshow(our_img)
    plt.show()

    plt.imshow(gt)
    plt.show()