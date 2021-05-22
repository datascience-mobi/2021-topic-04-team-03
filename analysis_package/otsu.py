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


def otsuna (image):
    img = image.copy().flatten()
    #Number of pixels
    N = img.size
    #probability of class occurence
    w_lower = np.zeros(256)
    w_upper = np.zeros(256)
    #mean value
    m_lower = np.zeros(256)
    m_upper = np.zeros(256)
    # total mean value
    m_tot = np.mean(img)
    #iterate over all thresholds
    for t in range(256):
        #calculate probabilty of class occurency for pixels below or equal/above t
        w_lower[t] = np.sum(np.where(img <= t, 1, 0)) / N
        w_upper[t] = np.sum(np.where(img > t, 1, 0)) / N
        #Calculate mu of both classes, considering that you cannot divide with zero
        if w_lower[t]>0 and w_upper[t]>0:
            m_lower[t] = np.sum(img[img <= t]) / (w_lower[t]*N)
            m_upper[t] = np.sum(img[img > t]) / (w_upper[t]*N)
        else:
            m_lower[t] = np.nan
            m_lower[t] = np.nan
        #total variance
        #sigma_tot += ((t-m_tot)**2)*(np.sum(np.where(img == t, 1, 0)) / N)
    sigma_b = w_lower*(w_upper)*((m_upper-m_lower)**2)
    threshold = np.nanargmax(sigma_b)
    #Calculate the goodness of our computet threshold
    sigma_tot = np.var(img)
    goodness = sigma_b[threshold] / sigma_tot
    return(threshold,sigma_b,goodness)


def clipping (img,threshold):
    #Copy of Image
    workimg = img.copy()
    #All pixels with intensity below theshold to 0
    workimg[workimg <= threshold] = 0
    #All pixels abow threshold to 1
    workimg[workimg > threshold] = 1
    return workimg

image_test = imread(r'''..\Data\NIH3T3\im\dna-27.png''')
threshold, goodness, s = otsu(image_test)
clipped_img = clipping(image_test, threshold)


plt.imshow(clipped_img, 'gray')
plt.show()
plt.imshow(image_test, 'gray')
plt.show()

print(threshold)