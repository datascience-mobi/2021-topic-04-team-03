# Implementation and evaluation of Otsu’s thresholding

Supervisor:

* PD Dr. Karl Rohr (k.rohr@uni-heidelberg.de)
*Christian Ritter (christian.ritter@bioquant.uni-heidelberg.de)
*Roman Spilger (roman.spilger@bioquant.uni-heidelberg.de)
*Leonid Kostrykin (leonid.kostrykin@bioquant.uni-heidelberg.de)
*Svenja Reith (svenja.reith@bioquant.uni-heidelberg.de)
*Qi Gao (qi.gao@bioquant.uni-heidelberg.de)

# Introduction
Our project, "Otsu's Thresholding" is a useful method for detecting the ideal threshold of an image. It is therefore used frequently in image segmentation for biological and medical purposes. Here we are going to use the implementation of this algorithm for segmentation of cell nuclei from three different datasets. For pre-processing the images, we implemented histogram stretching for images with low resolution, a gaussian filter, and a median filter as well as two-level Otsu thresholding for excluding reflections in some images. The implemented Otsu algorithm used on the pre-processed images was then evaluated with the Dice score, the median surface distance function and the Hausdorff metric.

# Description of datasets

**N2DH-GOWT1 cells**

The dataset N2DH-GOWT1 of the cell tracking challenge (Bártová et al., 2011) contains images of GFP-GOWT1 mouse
embryonic stem cells that have been derived with time-lapse confocal microscopy with a Leica TCS SP5 microscope.
The varying brightness of the cells makes it hard to distinguish all the cells from the background.

**N2HL-HeLa cells**

The dataset N2DL-HeLa of the cell tracking challenge (Neumann et al., 2010) contains images of human epithelial cells
of cervical cancer. Those images have been derived with an Olympus IX81 microscope used for live imaging of
fluorescently labelled chromosomes. The challenge in these images is the variety of brightness of the cells.

**NIH3T3 cells**

The dataset NIH3T3 (Coelho et al., 2009) contains images of several mouse embryonic fibroblast cells. These images
have also been derived with fluorescence microscopy images and the difficulty in segmenting these images mainly
lies in the bright light spots, probably from the used microscope, that makes it difficult for the algorithm to choose
a threshold between the brightness of the cells and the background and not between the brightness of light spots
and the cells.

# Packages 
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

# Literature 



