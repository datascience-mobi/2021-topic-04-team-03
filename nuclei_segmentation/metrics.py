import numpy as np
def cell_counting (img):
    # select border pixels
    border_pixels = []
    for index in np.ndindex(img.shape):
        if img[index[0]][index[1]] == 255:
            if 0 in img[(index[0]-1):(index[0]+2), (index[1]-1):(index[1]+2)]:
                border_pixels.append(index)

    return border_pixels
    # group border pixels
    for pixel