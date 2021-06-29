# Implementation and evaluation of Otsu’s thresholding
Our project, "Otsu's Thresholding" is a useful method for detecting the ideal threshold of an image. It is therefore used frequently in image segmentation for biological and medical purposes. Here we are going to use the implementation of this algorithm for segmentation of cell nuclei from three different datasets. For pre-processing the images, we implemented histogram stretching for images with low resolution, a gaussian filter, and a median filter as well as two-level Otsu thresholding for excluding reflections in some images. The implemented Otsu algorithm used on the pre-processed images was then evaluated with the Dice score, the median surface distance function and the Hausdorff metric.

The most important packages that defintely need to be installed are skimage, pathlib and numpy. 

There are several folders to arrange the project clearly: 

1. .ipynb_checkpoints
2. Data
3. histograms
4. Meetings
5. nuclei segmentation 
6. proposals
7. Results
8. Tests_and_incompletes


