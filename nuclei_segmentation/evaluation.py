import numpy as np
from scipy.ndimage import morphology


def dice(clipped_image, ground_truth):
    """
    This function returns the Dice Score Coefficient (value between 0 and 1), that evaluates the segmentation.

    :param clipped_image: Segmented (binary) image
    :param ground_truth: Ground truth image
    :return: Dice Score Coefficient
    """

    work_clipped = clipped_image.copy()
    work_gt = ground_truth.copy()

    # Assign 1 to all pixels, that have a non-zero intensity
    work_gt[work_gt != 0] = 1

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
    clipped = np.atleast_1d(clipped_image)
    gt = np.atleast_1d(ground_truth)

    conn = morphology.generate_binary_structure(clipped.ndim, connectivity)

    surface = clipped - morphology.binary_erosion(clipped, conn)
    surface_prime = gt - morphology.binary_erosion(gt, conn)

    distance_1 = morphology.distance_transform_edt(~clipped, pixel_size)
    distance_2 = morphology.distance_transform_edt(~surface_prime, pixel_size)

    surface_dist = np.concatenate([np.ravel(distance_1[surface_prime != 0]), np.ravel(distance_2[surface != 0])])

    return surface_dist


def mean_surface_distance(clipped_image, ground_truth, surface_dist):
    """
    This function applies the surface distance function to our images and returns its mean.

    :param clipped_image: Segmented (binary) image
    :param ground_truth: Ground truth image
    :param surface_dist: Use our implemented function surface_dist to calculate the surface distances of our images.
    :return: msd: mean surface distance
    """
    surf_dist = surface_dist(clipped_image(clipped_image == 1), 
                             ground_truth(ground_truth == 1), pixel_size=1, connectivity=1)

    msd = surf_dist.mean()

    return msd
