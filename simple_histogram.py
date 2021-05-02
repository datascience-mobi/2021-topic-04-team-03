from PIL import Image
import numpy
from matplotlib import pyplot as plt

#open image using Pillow
image44 = Image.open('dna-44.png')

#make numpy array
imarray44 = numpy.array(image44)

#flatten the array
imflat44 = imarray44.flatten()


#We are using discrete values - so let's align..
d = numpy.diff(numpy.unique(imflat44)).min()
left_of_first_bin = imflat44.min() - float(d)/2
right_of_last_bin = imflat44.max() + float(d)/2


#Plot the histogram
plt.hist(imflat44, numpy.arange(left_of_first_bin,right_of_last_bin+d,1), color = 'steelblue', ec = 'steelblue' )

#customize x-ticks
plt.xticks(numpy.arange(0,256,15), fontsize = 7)

#title and label
plt.title('Gray-value histogram')
plt.xlabel('Intensity', fontsize = 9)
plt.ylabel('Frequency', fontsize = 9)

#plt.show()
plt.show()
