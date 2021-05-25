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
    # total mean value
    m_tot = np.mean(img)
    #iterate over all thresholds
    for t in range (256):
        #calculate mu of pixels below t
        m[t] = np.sum(img[img<=t]) / N
        #calculate probabilty of class occurency for pixels below t
        w[t] = np.sum(np.where(img <= t, 1, 0)) / N

    # ignoring error because of division with 0
    with np.errstate(all='ignore'):
        # in-between class variance
        sigma_b = (m_tot*w - m)**2/(w*(1-w))

    #optimal threshold
    threshold = np.nanargmax(sigma_b)
    #total variance (same for every threshold)
    sigma_tot = np.var(img)
    goodness = sigma_b[threshold]/sigma_tot
    return threshold, goodness

def otsu_faster(image, intensity_lvls = 256):
    img = image.copy().flatten()
    # Number of pixels
    N = img.size
    # image histogram
    hist = np.histogram(img, bins = np.arange(intensity_lvls + 1), density = True)

    # probability of class occurence
    w = np.cumsum(hist[0])
    # mean value
    m = np.cumsum(hist[0]*np.arange(intensity_lvls))
    # total mean value
    m_tot = np.mean(img)

    # ignoring error because of division with 0
    with np.errstate(all='ignore'):
        # in-between class variance
        sigma_b = (m_tot * w - m) ** 2 / (w * (1 - w))

    # optimal threshold
    threshold = np.nanargmax(sigma_b)
    # total variance (same for every threshold)
    sigma_tot = np.var(img)
    goodness = sigma_b[threshold] / sigma_tot
    return threshold, goodness

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

    sigma_b = w_lower*(w_upper)*((m_upper-m_lower)**2)
    threshold = np.nanargmax(sigma_b)
    #Calculate the goodness of our computet threshold
    sigma_tot = np.var(img)
    goodness = sigma_b[threshold] / sigma_tot
    return(threshold,goodness)

def otsu_twolevel (img):
    # compute histogram of img
    hist = np.histogram(img, bins=np.arange(257), density=True)

    #zeroth_order moment = wk
    zeroth_order = hist[0]

    #first_order moment = mu(k) of the kth class
    first_order = hist[0] * np.arange(256)

    # Zeroth order moment P(u,v) and First order moment S(u,v) are stored in tables for all possible combinations of u and v
    P = np.zeros((256, 256))
    S = np.zeros((256, 256))

    for u in range(256):
        for v in range(u, 256):
            P[u, v] = np.sum(zeroth_order[0:v + 1]) - np.sum(zeroth_order[0:u])
            if P[u, v] == 0:
                P[u, v] = np.nan
            S[u, v] = np.sum(first_order[0:v + 1]) - np.sum(first_order[0:u])

    #Calculate the in between class variance using the values in the Tables P and S
    sigma = np.zeros((256, 256))
    for s in range(255):
        for t in range(s + 1, 255):
            sigma[s, t] = S[0, s] ** 2 / P[0, s] + S[s + 1, t] ** 2 / P[s + 1, t] + S[t + 1, 255] ** 2 / P[t + 1, 255]

    # Compute the maximum variance
    max = np.nanargmax(sigma)
    # Get postition of maximum ( = optimal Threshold values)
    thresholds = np.unravel_index(max, sigma.shape)
    return(thresholds)



def clipping (img,threshold):
    #Copy of Image
    workimg = img.copy()
    #All pixels with intensity below theshold to 0
    workimg[workimg <= threshold] = 0
    #All pixels abow threshold to 1
    workimg[workimg > threshold] = 1
    return workimg

### Just for testing. Delete later ###

r'''image_test = imread(r'..\Data\NIH3T3\im\dna-27.png')
threshold, goodness = otsu_faster(image_test)
clipped_img = clipping(image_test, threshold)


plt.imshow(clipped_img, 'gray')
plt.show()
plt.imshow(image_test, 'gray')
plt.show()

# Testing whether the output is the same as in the skimage function
from skimage.filters import threshold_otsu
t_skimage = threshold_otsu(image_test)

print(threshold, t_skimage)

# Testing whether twolevel_otsu is the same as the skimage funktion
from skimage.filters import threshold_multiotsu
t_twolevel_skimage = threshold_multiotsu(image_test)

print(t_twolevel_skimage)'''
