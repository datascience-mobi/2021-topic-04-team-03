#from PIL import Image
#path = r'Data\NIH3T3\gt\0.png'
#image = Image.open(path)

#image.show()

#To open tif (but also png) images

import skimage.io
img = skimage.io.imread(r"\Data\NIH3T3\im\dna-26.png")
imshow(img)


