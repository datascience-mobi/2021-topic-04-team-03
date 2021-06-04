import numpy as np
def cell_counting (img):
    # select border pixels
    border_pixels = []
    for index in np.ndindex(img.shape):
        if img[index[0]][index[1]] == 255:
            if 0 in img[(index[0]-1):(index[0]+2), (index[1]-1):(index[1]+2)]:
                border_pixels.append(index)

    # group border pixels
    pixel_groups = []
    for start_pixel in border_pixels:
        single_group = [start_pixel]
        for other_pixel in border_pixels:
            if other_pixel != start_pixel and math.dist(other_pixel, start_pixel) < 2:
                single_group.append(other_pixel)
                border_pixels.remove(other_pixel)
        border_pixels.remove(start_pixel)
        pixel_groups.append(single_group)

    return pixel_groups
