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

def stretch_png(img):
    p2, p98 = np.percentile(img, (2, 98))
    workimg = img.copy()
    workimg[workimg < p2] = p2
    workimg[workimg > p98] = p98
    a = 0
    b = 255
    c = int(np.min(workimg))
    d = int(np.max(workimg))
    img_stretch = (workimg - c) * (b - a) / (d - c)
    return img_stretch


image_test_png = imread(r'..\Data\NIH3T3\im\dna-0.png')
image_test_tif = imread(r'..\Data\N2DH-GOWT1\img\t01.tif')


stretchy = stretch_tif(image_test_png)

plt.imshow(image_test_png, 'gray')
plt.show()

plt.imshow(stretchy, 'gray')
plt.show()
