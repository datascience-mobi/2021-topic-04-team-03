import numpy as np
from skimage.io import imread
from skimage.io import imshow
from matplotlib import pyplot as plt

def otsu (image):
    img = image.copy().flatten()
    #Number of pixels
    N = img.size
    #probability of class occurence
    w = np.zeros(256)
    #mean value
    m = np.zeros(256)
    #Total variance
    #sigma_tot = 0
    # total mean value
    m_tot = np.mean(img)
    #iterate over all thresholds
    for t in range (256):
        #calculate mu of pixels below t
        m[t] = np.sum(img[img<=t]) / N
        #calculate probabilty of class occurency for pixels below t
        w[t] = np.sum(np.where(img <= t, 1, 0)) / N
        #total variance
        #sigma_tot += ((t-m_tot)**2)*(np.sum(np.where(img == t, 1, 0)) / N)

    zero_index = sum(np.where(w == 0, 1, 0))
    m = m[w != 0]
    w = w[w != 0]
    m = m[w != 1]
    w = w[w != 1]
    #in-between class variance
    sigma_b = (m_tot*w - m)**2/(w*(1-w))
    print(sigma_b)
    #optimal threshold
    #threshold = np.where(sigma_b == max(sigma_b))[0][0]
    threshold = np.argmax(sigma_b) + zero_index
    #total variance (same for every threshold)
    sigma_tot = np.var(img)
    goodness = sigma_b[threshold]/sigma_tot

    return threshold, goodness, sigma_b


def clipping (img,threshold):
    #Copy of Image
    workimg = img.copy()
    #All pixels with intensity below theshold to 0
    workimg[workimg <= threshold] = 0
    #All pixels abow threshold to 1
    workimg[workimg > threshold] = 1
    return workimg

image_test = imread('dna-49.png')
threshold, goodness, s = otsu(image_test)
clipped_img = clipping(image_test, threshold)
plt.imshow(clipped_img, 'gray')
plt.show()
plt.imshow(image_test, 'gray')
plt.show()
