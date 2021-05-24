from skimage.io import imread
from skimage.io import imshow

import numpy
from matplotlib import pyplot as plt

img = imread('man_seg01.tif')

# make onedimensional array
imflat = img.flatten()

#Make an array without zeros
print(type(imflat))
print(imflat.ndim)

print(len(imflat))

imflat_nozero = imflat[imflat != 0]

print(len(imflat_nozero))
#plot imflat

#Wir haben diskrete Werte aufer x-Achse und keine kont. Werte, das schiebt alles schön hin
d = numpy.diff(numpy.unique(imflat_nozero)).min()
left_of_first_bin = imflat_nozero.min() - float(d)/2
right_of_last_bin = imflat_nozero.max() + float(d)/2


#make histogram
plt.hist(imflat_nozero, numpy.arange(left_of_first_bin, right_of_last_bin + d, d), color = 'gray', ec = 'black' )
#Titel
plt.title('Man_seg01 - Erstes Histogramm',fontsize=15)
#X-Achse Beschriftung
plt.xlabel('Intensity')
#y Achse Beschriftung
plt.ylabel('Frequency')
#customize number of x-ticks
plt.xticks(range(1,25), fontsize = 7)


plt.show()

# plt.hist(imflat, bins=range(0,25))
# plt.show()

print(d)


#print(type(img))

#imshow(img)
#pyplot.show()