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


def surface_distance(clipped_image, ground_truth, pixel_size=1, connectivity=1):
    """
    This function calculates the surface distance between the pixels of the clipped and the ground truth image.
    It therefore creates a kernel to detect the edges of the segmentations, named conn, and then removes the outermost
    pixels from the edges to receive a single-pixel-wide surface.
    Then the distances between the surfaces are calculated and the surface distance is returned.


    :param clipped_image: Segmented (binary) image
    :param ground_truth: Ground truth image
    :param pixel_size: the pixel resolution or pixel size, entered as a 2D vector, default value is one
    :param connectivity: creates a 2D matrix defining the neighborhood around which the function looks
    for neighbouring pixels, default is a six-neighborhood kernel
    :return: surface distance between the clipped and the ground truth image
    """
    clipped = np.atleast_1d(clipped_image.astype(np.bool_))
    gt = np.atleast_1d(ground_truth.astype(np.bool_))

    conn = morphology.generate_binary_structure(clipped.ndim, connectivity)

    surface = np.subtract(clipped, morphology.binary_erosion(clipped, conn), dtype=np.float32)
    surface_prime = np.subtract(gt, morphology.binary_erosion(gt, conn), dtype=np.float32)

# not sure if ~ and 1-surface are the same in this case
    distance_1 = morphology.distance_transform_edt(1-surface, pixel_size)
    distance_2 = morphology.distance_transform_edt(1-surface_prime, pixel_size)

    surf_dist = np.concatenate([np.ravel(distance_1[surface_prime != 0]), np.ravel(distance_2[surface != 0])])

    return surf_dist

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

def haussdorf(segmentation,ground_truth):
    '''
    This function computes Haussdorf distance between the segmentation and the ground truth.
    To accelerate the process, we used scipy.spatial.KDTree:
    'This class provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point.'

    :param segmentation: Segmented, binary picture
    :param ground_truth: Ground Truth
    :return: Haussdorf Distance
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

    haussdorf_distance = max(max_gt_seg,max_seg_gt)

    return haussdorf_distance


if __name__ == "__main__":
    from nuclei_segmentation import otsu
    from skimage.io import imread
    import pathlib as pl

    img = imread(str(pl.Path('Data/NIH3T3/img/dna-29.png')))
    our_img = otsu.complete_segmentation(img)
    gt = imread(str(pl.Path('/Data/NIH3T3/gt/29.png')))

    mean_surface_dist = msd(our_img,gt)
    hd = haussdorf(our_img,gt)
    print(hd)
    print(mean_surface_dist)



    #msd_hd = surface_distance_functions(our_img, gt, [1344, 1024], 1)
    #print(msd_hd)

    #surface_distance_applied = surface_distance(our_img, gt, pixel_size=[1024, 1344], connectivity=1)

    #weird values
    #msd = surface_distance_applied.mean()
    #print(msd)
    #hd = surface_distance_applied.max()
    #print(hd)
