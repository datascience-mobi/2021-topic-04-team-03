import numpy as np
from numpy import unravel_index
from skimage.io import imread

img = imread(r'''Data\NIH3T3\im\dna-42.png''')
#Number of pixels
N = img.size
#probability of class occurence
w_lower = np.zeros((256,256))
w_middle = np.zeros((256,256))
w_upper = np.zeros((256,256))
#mean value
m_lower = np.zeros((256,256))
m_middle = np.zeros((256,256))
m_upper = np.zeros((256,256))

# total mean value
m_tot = np.mean(img)

#iterate over all thresholds
for t in range(0,256):
    for s in range(t+1,256):
        #calculate probabilty of class occurency for all classes
        w_lower[t,s] = np.sum(np.where(img <= t, 1, 0)) / N
        w_upper[t,s] = np.sum(np.where(img > s, 1, 0)) / N
        w_middle[t,s] = 1 - w_lower[t,s] - w_upper[t,s]
        #Calculate mu of all classes, considering that you cannot divide with zero
        if w_lower[t,s]>0 and w_upper[t,s]>0 and w_middle[t,s]>0:
            m_lower[t,s] = np.sum(img[img <= t]) / (w_lower[t,s]*N)
            m_middle[t,s] = np.sum(img[(img > t) & (img <= s)]) / (w_middle[t,s]*N)
            m_upper[t,s] = np.sum(img[img > t]) / (w_upper[t,s]*N)
        else:
            m_lower[t,s] = np.nan
            m_middle[t,s] = np.nan
            m_lower[t,s] = np.nan

#Calculate the between class variance
sigma_b = w_lower*((m_lower-m_tot)**2) + w_middle*((m_middle-m_tot)**2) + w_upper*((m_upper-m_tot)**2)
#determine position of max variance
maxvariance = np.nanargmax(sigma_b)
#get position of max variance as a tuple of both thresholds
thresholds = np.unravel_index(maxvariance, sigma_b.shape)

# The thresholds are not equal to the thresholds computed by the skimage function, so theres something wrong with the code
print(thresholds)