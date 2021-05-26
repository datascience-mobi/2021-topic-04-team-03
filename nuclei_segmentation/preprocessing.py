from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np


def stretch_tif(img):
    p2, p98 = np.percentile(img, (2, 98))
    workimg = img.copy()
    workimg[workimg < p2] = p2
    workimg[workimg > p98] = p98
    a = 0
    b = 2 ** 16
    c = int(np.min(workimg))
    d = int(np.max(workimg))
    img_stretch = (workimg - c) * (b - a) / (d - c)
    return img_stretch



r'''
image_test = imread(r'..\Data\N2DL-HeLa\img\t13.tif')
stretchy = stretch_tif(image_test)

plt.imshow(image_test, 'gray')
plt.show()

plt.imshow(stretchy, 'gray')
plt.show()
'''