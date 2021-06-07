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

def overlay (test_image, ground_thruth):
    # does not work
    false_pixels = np.ma.masked_where(ground_thruth == test_image, test_image)
    gt_0 = np.ma.masked_where(ground_thruth == 0, false_pixels)
    false_negatives = np.ma.mask_or(false_pixels, gt_0)
    gt_1 = np.ma.masked_where(ground_thruth == 1, false_pixels)
    false_positives = np.ma.mask_or(false_pixels, gt_1)

    plt.figure()
    plt.imshow(test_image, 'gray', interpolation='none')
    plt.imshow(false_negatives, 'jet', interpolation='none')
    plt.imshow(false_positives, 'gnuplot', interpolation='none')
    plt.show()

# from nuclei_segmentation import otsu
# from skimage.io import imread
#
# img = imread(r'..\Data\NIH3T3\img\dna-0.png')
# our_img = otsu.otsu(img)
# gt = imread(r'..\Data\NIH3T3\gt\0.png')
#
# overlay(our_img, gt)
