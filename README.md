# Implementation and evaluation of Otsu’s thresholding
Elizaveta Chernova, Veronika Schuler, Laura Wächter, Hannah Winter

Supervisors:

- PD Dr. Karl Rohr (k.rohr@uni-heidelberg.de)
- Christian Ritter (christian.ritter@bioquant.uni-heidelberg.de)
- Carola Krug (carola.krug@bioquant.uni-heidelberg.de)
- Leonid Kostrykin (leonid.kostrykin@bioquant.uni-heidelberg.de)

# Overview
Thresholding is a useful method that is frequently used in the context of image segmentation. In this project, we used Otsu's thresholding algorithm in order to find the optimal threshold value, that optimizes image segmentation. The algorithm was applied to a number of images from different datasets (N2DH-GOWT1, N2DL-HeLa, NIH3T3). In order to improve the results, several preprocessing methods (mainly filters) were used. The final segmentations were compared to reference images and evaluated with several methods (Dice Score, MSD, Hausdorff Distance). The different datasets are characterized by different features, like reflections or low contrast. For this reason, we hypothesized that for the seperate datasets different preprocessing methods would lead to the optimal segmentation result. This was confirmed by our overall analysis. In addition, a cell nuclei counting algorithm was developed.

# Description of the datasets

**N2DH-GOWT1 cells**

The dataset N2DH-GOWT1 of the cell tracking challenge contains images of GFP-GOWT1 mouse embryonic stem cells that were captured using time-lapse confocal microscopy (Leica TCS SP5 microscope). The varying brightness of the cells makes it difficult to distinguish the cells from the background. Further, low contrast and the noise in the images present challenges to the segmentation algorithm.

**N2HL-HeLa cells**

The dataset N2DL-HeLa of the cell tracking challenge contains images of human epithelial cells of cervical cancer. Those images were captured with an Olympus IX81 microscope used for live imaging of fluorescently labelled chromosomes. The challenge of these images is the variable brightness of the cells.

**NIH3T3 cells**

The dataset NIH3T3 contains images of several mouse embryonic fibroblast cells. These images were also captured using fluorescence microscopy. The difficulty when segmenting these images lies mainly in certain bright spots (reflections).

# Modules 
1. all_functions.py
2. complete_analysis.py
3. evaluation.py
4. metrics.py
5. otsu.py
6. preprocessing.py
7. visualisation.py



