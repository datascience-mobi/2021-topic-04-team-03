from skimage.io import imread_collection
from nuclei_segmentation import preprocessing
import numpy as np
from nuclei_segmentation import visualisation
from nuclei_segmentation import metrics
from skimage.io import imread
import matplotlib.pyplot as plt
import pathlib as pl
from nuclei_segmentation import otsu
from nuclei_segmentation import evaluation


def comparison_plot (image, processed_image, image_seg, gt, dice_score):
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Dice Score: ' + str(dice_score))
    ax[0][0].imshow(image, 'gray')
    ax[0][0].set_title('Original image')
    ax[0][1].imshow(processed_image, 'gray')
    ax[0][1].set_title('Processed image')
    ax[1][0].imshow(image_seg, 'gray')
    ax[1][0].set_title('Segmented image')
    ax[1][1].imshow(gt, 'gray')
    ax[1][1].set_title('Ground truth')
    ax[0][0].set_axis_off()
    ax[0][1].set_axis_off()
    ax[1][0].set_axis_off()
    ax[1][1].set_axis_off()
    # for ax_i in ax.ravel():
    #     ax_i.set_axis_off()
    fig.show()

col_dir_img_nih = str(pl.Path('Data/N2DL-HeLa/img/*.tif'))
col_dir_gt_nih = str(pl.Path('Data/N2DL-HeLa/gt/*.tif'))

N2DL_img_collection = imread_collection(col_dir_img_nih)
N2DL_gt_collection = imread_collection(col_dir_gt_nih)

dice_list = []
msd_list = []
hausdorff_list = []
for index in range(len(N2DL_img_collection)):
    image = N2DL_img_collection[index]
    gt = N2DL_gt_collection[index]
    gt[gt > 0 ] = 1

    stretched_image = preprocessing.histogram_stretching(image, intensity_lvls=2**16)
    clipped_image = otsu.complete_segmentation(stretched_image, intensity_lvls=2**16)

    # for later analysis (cell counting)
    if index == 0:
        cell_counting_sample = clipped_image

    dice_score = evaluation.dice(clipped_image,gt)
    dice_list.append(dice_score)
    msd_list.append(evaluation.msd(clipped_image, gt))
    hausdorff_list.append(evaluation.hausdorff(clipped_image, gt))

    visualisation.comparison_plot(image, stretched_image, clipped_image, gt,
                    title1='Original image', title2='Preprocessed image', title3='Segmented image', title4='Ground truth',
                    figure_title=('Dice Score: ' + str(dice_score)))

print('Mean Dice Score of the N2DL-HeLa dataset: ' +str(np.mean(dice_list)))

# precalculated data
dc_gowt = [0.20095467802323116, 0.18615249210830676, 0.14723089179436713, 0.17247176784418858, 0.23942897882574002, 0.2800658197354881]
g_dc_gowt = [0.6049033767200361, 0.545156565294437, 0.607365210111527, 0.6337398735867533, 0.628395754259946, 0.6101785396524112]
m_dc_gowt = [0.6037089513424323, 0.5424736733429693, 0.5957841500444236, 0.6207683116695029, 0.6151315544639202, 0.6342945864207978]
hs_dc_gowt = [0.7305905136874669, 0.777013844515442, 0.7818812453546948, 0.7545458227027902, 0.7540217593453599, 0.7841191962191998]
gs_dc_gowt = [0.7891630367381934, 0.774270861172657, 0.7936481543066178, 0.778829001019368, 0.7923810256015992, 0.7848321671557089]
dice_mh_GOWT1 = [0.8479139784946237, 0.7897543357524959, 0.7991618631153515, 0.7936914767322101, 0.8372021654002431, 0.83254629515168]

dice_scores = np.array([dc_gowt, g_dc_gowt, m_dc_gowt, hs_dc_gowt, gs_dc_gowt, dice_mh_GOWT1])
visualisation.comparison_preprocessing(dice_scores)


msd_gowt = [93.14500783281446, 138.74403266791705, 150.45660270160226, 137.00494187548546, 72.56997880215009, 81.213996269387]
gs_msd_gowt = [5.266364362378366, 16.891292176072785, 14.848960822525951, 15.79853712125804, 7.020092819137251, 5.167408977948047]
hs_msd_gowt = [9.32115590110038, 0.9935213685903743, 1.1562701354988933, 1.259951709461983, 4.749376664514487, 2.8531301479357114]
m_msd_gowt =[17.240628033923354, 36.67875121818343, 33.382121763815015, 33.43979093514925, 18.626395846524083, 20.79117451137494]
g_msd_gowt = [24.26923556237997, 49.406291084433015, 35.86219047128074, 34.783610573022486, 30.55086265130697, 33.18067226935093]
msd_mh_GOWT1 = [2.8858543997286246, 13.547702014094002, 14.179070113137572, 13.396876081690408, 2.0279260699536343, 2.5968995136705164]

msd_scores = np.array([msd_gowt, g_msd_gowt, m_msd_gowt, hs_msd_gowt, gs_msd_gowt, msd_mh_GOWT1])
visualisation.comparison_preprocessing(msd_scores, y_label= 'MSD Value')

# Example for cell counting
cell_number = metrics.cell_counting(cell_counting_sample)
print('Number of cells: ' + str(cell_number))

border = metrics.find_border(cell_counting_sample)
visualisation.border_image(cell_counting_sample, border, width= 0.3)


# plot of original and ground truth


img_NIH3T3 = imread(str(pl.Path('./Data/NIH3T3/img/dna-42.png')))
gt_NIH3T3 = imread(str(pl.Path('./Data/NIH3T3/gt/42.png')))


# One level Otsu

threshold_NIH3T3 = otsu.otsu(img_NIH3T3)
clipped_NIH3T3 = otsu.clipping(img_NIH3T3, threshold_NIH3T3)

dc_clipped_NIH3T3 = evaluation.dice(clipped_NIH3T3, gt_NIH3T3)
print("One level Otsu: " + str(dc_clipped_NIH3T3))

# Two level Otsu for reflection correction

two_level_threshold_NIH3T3 = otsu.otsu_twolevel(img_NIH3T3)
two_level_clipped_NIH3T3 = otsu.clipping_twolevel(img_NIH3T3, two_level_threshold_NIH3T3)

dc_two_level_NIH3T3 = evaluation.dice(two_level_clipped_NIH3T3, gt_NIH3T3)
print("Two level Otsu (reflection correction): " + str(dc_two_level_NIH3T3))

visualisation.comparison_plot(img_NIH3T3, gt_NIH3T3, clipped_NIH3T3, two_level_clipped_NIH3T3,
                    title1='Original Image', title2='Ground Truth', title3='One Level Clipped', title4= 'Two Level Clipped',
                    figure_title='Comparison of One and Two Level Otsu')
