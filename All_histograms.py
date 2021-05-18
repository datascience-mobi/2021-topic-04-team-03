from PIL import Image
import numpy
from matplotlib import pyplot as plt

#open image using Pillow
#you probably have to alter the path to the image

##Dataset 1
N2DH_GOWT1_t01 = Image.open('t01.jpg')
N2DH_GOWT1_t21 = Image.open('t21.jpg')
N2DH_GOWT1_t31 = Image.open('t31.jpg')
N2DH_GOWT1_t39 = Image.open('t39.jpg')
N2DH_GOWT1_t52 = Image.open('t52.jpg')
N2DH_GOWT1_t72 = Image.open('t72.jpg')

##Dataset 1 gt
N2DH_GOWT1_gt01 = Image.open('man_seg01.jpg')
N2DH_GOWT1_gt21 = Image.open('man_seg21.jpg')
N2DH_GOWT1_gt31 = Image.open('man_seg31.jpg')
N2DH_GOWT1_gt39 = Image.open('man_seg39.jpg')
N2DH_GOWT1_gt52 = Image.open('man_seg52.jpg')
N2DH_GOWT1_gt72 = Image.open('man_seg72.jpg')

##Dataset 2
N2DL_HeLa_t13 = Image.open('t13.jpg')
N2DL_HeLa_t52 = Image.open('t52.jpg')
N2DL_HeLa_t75 = Image.open('t75.jpg')
N2DL_HeLa_t79 = Image.open('t79.jpg')

##Dataset 2 gt
N2DL_HeLa_gt13 = Image.open('man_seg13.jpg')
N2DL_HeLa_gt52 = Image.open('man_seg52.jpg')
N2DL_HeLa_gt75 = Image.open('man_seg75.jpg')
N2DL_HeLa_gt79 = Image.open('man_seg79.jpg')

##Dataset 3
NIH3T3_dna_0 = Image.open('dna-0.jpg')
NIH3T3_dna_1 = Image.open('dna-1.jpg')
NIH3T3_dna_26 = Image.open('dna-26.jpg')
NIH3T3_dna_27 = Image.open('dna-27.jpg')
NIH3T3_dna_28 = Image.open('dna-28.jpg')
NIH3T3_dna_29 = Image.open('dna-29.jpg')
NIH3T3_dna_30 = Image.open('dna-30.jpg')
NIH3T3_dna_31 = Image.open('dna-31.jpg')
NIH3T3_dna_32 = Image.open('dna-32.jpg')
NIH3T3_dna_33 = Image.open('dna-33.jpg')
NIH3T3_dna_37 = Image.open('dna-37.jpg')
NIH3T3_dna_40 = Image.open('dna-40.jpg')
NIH3T3_dna_42 = Image.open('dna-42.jpg')
NIH3T3_dna_44 = Image.open('dna-44.jpg')
NIH3T3_dna_45 = Image.open('dna-45.jpg')
NIH3T3_dna_46 = Image.open('dna-46.jpg')
NIH3T3_dna_47 = Image.open('dna-47.jpg')
NIH3T3_dna_49 = Image.open('dna-49.jpg')

##Dataset 3 gt
NIH3T3_dna_gt0 = Image.open('0.jpg')
NIH3T3_dna_gt1 = Image.open('1.jpg')
NIH3T3_dna_gt26 = Image.open('26.jpg')
NIH3T3_dna_gt27 = Image.open('27.jpg')
NIH3T3_dna_gt28 = Image.open('28.jpg')
NIH3T3_dna_gt29 = Image.open('29.jpg')
NIH3T3_dna_gt30 = Image.open('30.jpg')
NIH3T3_dna_gt31 = Image.open('31.jpg')
NIH3T3_dna_gt32 = Image.open('32.jpg')
NIH3T3_dna_gt33 = Image.open('33.jpg')
NIH3T3_dna_gt37 = Image.open('37.jpg')
NIH3T3_dna_gt40 = Image.open('40.jpg')
NIH3T3_dna_gt42 = Image.open('42.jpg')
NIH3T3_dna_gt44 = Image.open('44.jpg')
NIH3T3_dna_gt45 = Image.open('45.jpg')
NIH3T3_dna_gt46 = Image.open('46.jpg')
NIH3T3_dna_gt47 = Image.open('47.jpg')
NIH3T3_dna_gt49 = Image.open('49.jpg')

##iterate over alle images with the following code...tbd

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