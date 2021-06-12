import math
import numpy as np


def find_border(img):
    """
    This Function takes a binary image and appends every pixel with intensity 255, that is next to or diagonal to
    a pixel of intensity 0, to a list.

    :param img: Binary image
    :return: List that contains the positions of all border pixels
    """

    edge_pixels = []

    for index in np.ndindex(img.shape):
        if img[index[0]][index[1]] == 255:
            if 0 in img[(index[0] - 1):(index[0] + 2), (index[1] - 1):(index[1] + 2)]:
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



def cell_counting (img):
    """
    This function takes a binary image and returns the number of shapes.
    For more detail see functions 'find_border' and 'group_border'

    :param img: Binary Image
    :return: Number of shapes in the binary image
    """

    border = find_border(img)
    cells = group_border(border)

    return len(cells)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from skimage.io import imread
    from skimage.io import imshow
    import matplotlib as mpl # big plots
    import pathlib as pl
    mpl.rcParams['figure.dpi'] = 10000
    img = imread(str(pl.Path(r'..\Data\NIH3T3\gt\0.png')))
    fig, ax = plt.subplots(figsize=(10.24, 13.44))
    plt.imshow(img, 'gray')
    plt.show()
