import numpy as np
from scipy.ndimage import morphology
from scipy import spatial

def dice(clipped_image, ground_truth):
    """
    This function returns the Dice Score Coefficient (value between 0 and 1), that evaluates the segmentation.

    :param clipped_image: Segmented (binary) image
    :param ground_truth: Ground truth image
    :return: Dice Score Coefficient
    """

    work_clipped = np.zeros(clipped_image.shape)
    work_gt = np.zeros(ground_truth.shape)

    # Assign 1 to all pixels, that have a non-zero intensity
    work_gt[ground_truth!= 0] = 1
    work_clipped[clipped_image != 0] = 1

    intersection = np.sum(work_clipped * work_gt)
    sum_all = np.sum(work_clipped) + np.sum(work_gt)
    dice_score = (2 * intersection) / sum_all

    return dice_score


def iou(clipped_image, ground_truth):
    """
    Calculation of the Intersection-Over-Union (value between 0 and 1), also known as Jaccard Index.

    :param clipped_image: Segmented (binary) image
    :param ground_truth: Ground truth image
    :return: Intersection-Over-Union
    """

    work_clipped = np.zeros(clipped_image.shape)
    work_gt = np.zeros(ground_truth.shape)

    # Assign 1 to all pixels, that have a non-zero intensity
    work_gt[ground_truth!= 0] = 1
    work_clipped[clipped_image != 0] = 1

    intersection = np.sum(clipped_image * ground_truth)
    union = np.sum(clipped_image) + np.sum(ground_truth) - intersection

    iou_score = intersection / union

    return iou_score


def msd(segmentation,ground_truth):
    '''
    This function computes Mean surface distance between the segmentation and the ground truth.
    To accelerate the process, we used scipy.spatial.KDTree:
    'This class provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point.'

    :param segmentation: Segmented, binary picture
    :param ground_truth: Ground Truth
    :return: Mean surface distance
    '''
    seg_pixels = []
    for index in np.ndindex(segmentation.shape):
        if segmentation[index[0]][index[1]] != 0:
            seg_pixels.append(index)

    gt_pixels = []
    for index1 in np.ndindex(ground_truth.shape):
        if ground_truth[index1[0]][index1[1]] != 0:
            gt_pixels.append(index1)

    seg_array = np.array(seg_pixels)
    gt_array = np.array(gt_pixels)

    # calculate minimum distances for each point in seg to the sets of points in gt
    tree_seg_gt = spatial.cKDTree(gt_array)
    mindist_seg_gt, minid_seg_gt = tree_seg_gt.query(seg_array)

    # calculate sum and length of arrays with minimal distances
    sum_seg_gt = np.sum(mindist_seg_gt)
    size_seg_gt = len(mindist_seg_gt)


    # calculate minimum distances for each point in gt to the sets of points in seg
    tree_gt_seg = spatial.cKDTree(seg_array)
    mindist_gt_seg, minid_gt_seg = tree_gt_seg.query(gt_array)

    # calculate sum and length of arrays with minimal distances
    sum_gt_seg = np.sum(mindist_gt_seg)
    size_gt_seg = len(mindist_gt_seg)

    mean_surface_distance = (1/(size_gt_seg+size_seg_gt))*(sum_gt_seg + sum_seg_gt)

    return mean_surface_distance

def hausdorff(segmentation,ground_truth):
    '''
    This function computes Haussdorf distance between the segmentation and the ground truth.
    To accelerate the process, we used scipy.spatial.KDTree:
    'This class provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point.'

    :param segmentation: Segmented, binary picture
    :param ground_truth: Ground Truth
    :return: Hausdorff Distance
    '''
    seg_pixels = []
    for index in np.ndindex(segmentation.shape):
        if segmentation[index[0]][index[1]] != 0:
            seg_pixels.append(index)

    gt_pixels = []
    for index1 in np.ndindex(ground_truth.shape):
        if ground_truth[index1[0]][index1[1]] != 0:
            gt_pixels.append(index1)

    seg_array = np.array(seg_pixels)
    gt_array = np.array(gt_pixels)

    # calculate minimum distances for each point in seg to the sets of points in gt
    tree_seg_gt = spatial.cKDTree(gt_array)
    mindist_seg_gt, minid_seg_gt = tree_seg_gt.query(seg_array)

    # maximum value in set of minimal distances
    max_seg_gt = np.max(mindist_seg_gt)


    # calculate minimum distances for each point in gt to the sets of points in seg
    tree_gt_seg = spatial.cKDTree(seg_array)
    mindist_gt_seg, minid_gt_seg = tree_gt_seg.query(gt_array)

    # maximum value in set of minimal distances
    max_gt_seg = np.max(mindist_gt_seg)

    hausdorff_distance = max(max_gt_seg,max_seg_gt)

    return hausdorff_distance


if __name__ == "__main__":
    from nuclei_segmentation import otsu
    from skimage.io import imread
    import pathlib as pl

    img = imread(str(pl.Path('../Data/NIH3T3/img/dna-29.png')))
    our_img = otsu.complete_segmentation(img)
    gt = imread(str(pl.Path('../Data/NIH3T3/gt/29.png')))

    mean_surface_dist = msd(our_img,gt)
    hd = hausdorff(our_img,gt)
    print(hd)
    print(mean_surface_dist)

