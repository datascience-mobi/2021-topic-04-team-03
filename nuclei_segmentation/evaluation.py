import numpy as np

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
    sum = np.sum(work_clipped) + np.sum(work_gt)
    dice_score = (2 * intersection) / sum

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
