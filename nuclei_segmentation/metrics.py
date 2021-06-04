import numpy as np
def cell_counting (img):
    # select border pixels
    border_pixels = []
    for index in np.ndindex(img.shape):
        if img[index] == 1:
            if 0 in img[(index[0]-1):(index[0]+1)][(index[1]-1):(index[1]+1)]:
                border_pixels.append(index)
    # group border pixels
    for pixel