import math
import numpy as np


def find_border(image):
    """
    This Function takes a binary image and appends every pixel with intensity 255, that is next to or diagonal to
    a pixel of intensity 0, to a list.

    :param image: Binary image
    :return: List that contains the positions of all border pixels
    """

    edge_pixels = []
    workimg = np.zeros(image.shape)
    workimg[image > 0] = 1

    for index in np.ndindex(image.shape):
        if workimg[index[0]][index[1]] == 1:
            if 0 in workimg[(index[0] - 1):(index[0] + 2), (index[1] - 1):(index[1] + 2)]:
                edge_pixels.append(index)

    return edge_pixels


def group_border(border_pixels):
    """
    This Function takes a list containing the positions of all border pixels and groups
    all border pixels belonging to the same shape. Therefore, a list containing lists is returned.

    :param border_pixels: List containing the positions of all border pixels
    :return: List containing border pixels belonging to one segment in separate lists
    """

    all_groups = []

    for start_pixel in border_pixels:
        old_group = [start_pixel]
        new_group = []
        first_run = True
        while old_group != new_group:
            if not first_run:
                old_group = new_group
            first_run = False
            for pixel in old_group:
                for other_pixel in border_pixels:
                    if math.dist(pixel, other_pixel) < 2:
                        new_group.append(other_pixel)
                        border_pixels.remove(other_pixel)
                if pixel in border_pixels:
                    border_pixels.remove(pixel)
        all_groups.append(new_group)

    return all_groups


def cell_counting(image):
    """
    This function takes a binary image and returns the number of shapes.
    For more detail see functions 'find_border' and 'group_border'

    :param image: Binary Image
    :return: Number of shapes in the binary image
    """

    border = find_border(image)
    cells = group_border(border)

    return len(cells)


def cell_counting_ground_truth(ground_truth_image):
    """
    # (shades of gray in the ground truth) = # (number of cells) - 1
    datasets: N2DH-GOWT1 and N2DL-HeLa

    :param ground_truth_image: Ground truth image
    :return:Number of cells
    """
    return len(np.unique(ground_truth_image)) - 1
