import math

import numpy as np
def cell_counting (img):
    # select border pixels
    border_pixels = []
    for index in np.ndindex(img.shape):
        if img[index[0]][index[1]] == 255:
            if 0 in img[(index[0]-1):(index[0]+2), (index[1]-1):(index[1]+2)]:
                border_pixels.append(index)

    # group border pixels
    # pixel_groups = []
    # for start_pixel in border_pixels:
    #     single_group = [start_pixel]
    #     for other_pixel in border_pixels:
    #         if other_pixel != start_pixel and math.dist(other_pixel, start_pixel) < 2:
    #             single_group.append(other_pixel)
    #             border_pixels.remove(other_pixel)
    #     border_pixels.remove(start_pixel)
    #     pixel_groups.append(single_group)


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
                border_pixels.remove(pixel)


    return pixel_groups
