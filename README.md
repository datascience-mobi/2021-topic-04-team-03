# Implementation and evaluation of Otsu’s thresholding

Supervisors:

- PD Dr. Karl Rohr (k.rohr@uni-heidelberg.de)
- Christian Ritter (christian.ritter@bioquant.uni-heidelberg.de)
- Carola Krug (carola.krug@bioquant.uni-heidelberg.de)
- Leonid Kostrykin (leonid.kostrykin@bioquant.uni-heidelberg.de)

# Overview
Thresholding is a useful method that is frequently used in the context of
image segmentation.
In this project, we used Otsu's thresholding algorithm in order to find the optimal threshold value,
that optimizes the image segmentation.
The algorithm was applied to a number of images from different datasets (N2DH-GOWT1, N2DL-HeLa, NIH3T3).
To improve the results, several preprocessing methods (mainly filters) were used.
The final segmentations were compared to reference images and evaluated with several methods (Dice Score, MSD, Hausdorff Distance).
The different datasets are characterized by different features, like reflections or low contrast.
For this reason, it is likely that different preprocessing methods will lead to the optimal result.
Our overall analysis confirmed this hypothesis.

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
1. all_functions.py
2. complete_analysis.py
3. evaluation.py
4. metrics.py
5. otsu.py
6. preprocessing.py
7. visualisation.py



