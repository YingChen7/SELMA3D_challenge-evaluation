import numpy as np
from typing import List,Dict
from skimage.measure import euler_number, label
import SimpleITK as sitk

def extract_labels(gt_array, pred_array):
    """
    Extracts union of labels in gt and pred masks
    
    Args:
        gt_array: ndarray, ground truth
        pred_array: ndarray, prediction
    Returns:
        labels: list of union of unique elements in gt and pred
    """
    labels_gt = np.unique(gt_array)
    labels_pred = np.unique(pred_array)
    labels = list(set().union(labels_gt, labels_pred))
    labels = [int(x) for x in labels]
    return labels

def betti_number(img):
    """
    calculates the Betti number B0 and B1 for a 3D array from the Euler characteristic number

    code prototyped by
    - Martin Menten (Imperial College)
    - Suprosanna Shit (Technical University Munich)
    - Johannes C. Paetzold (Imperial College)

    Args:
        img: ndarray
    Returns:
        [b0, b1]: betti number B0 and B1 for the input image
    """
    # make sure the image is 3D (for connectivity settings)
    assert img.ndim == 3

    # 6 or 26 neighborhoods are defined for 3D images,
    # (connectivity 1 and 3, respectively)
    # If foreground is 26-connected, then background is 6-connected, and conversely
    N6 = 1
    N26 = 3

    # important first step is to
    # pad the image with background (0) around the border!
    padded = np.pad(img, pad_width=1)

    # make sure the image is binary with
    assert set(np.unique(padded)).issubset({0, 1})

    # calculate the Betti numbers B0, B2
    # then use Euler characteristic to get B1

    # get the label connected regions for foreground
    _, b0 = label(
        padded,
        # return the number of assigned labels
        return_num=True,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    euler_char_num = euler_number(
        padded,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    # get the label connected regions for background
    _, b2 = label(
        1 - padded,
        # return the number of assigned labels
        return_num=True,
        # 6 neighborhoods for background
        connectivity=N6,
    )

    # NOTE: need to substract 1 from b2
    b2 -= 1

    b1 = b0 + b2 - euler_char_num  # Euler number = Betti:0 - Bett:1 + Betti:2

    return [b0, b1]


def betti_number_error(gt, pred):
    """
    Compute the betti number error for binary segmentation 
    (N. Stucki, J. C. Paetzold, S. Shit, B. Menze, U. Bauer. Topologically faithful image segmentation via induced matching of persistence barcodes. In International Conference on Machine Learning, 2023, pp. 32698-32727.)

    Args:
        gt: ndarray, gounrd truth
        pred: ndarray, prediction

    Returns:
        b0_error: betti number error of dimension-0
        b1_error: betti number error of dimension-1
    """
    labels = extract_labels(gt_array=gt, pred_array=pred)
    labels.remove(0)

    # when there are no labels in the ROI, return blank betti_num_err_dict
    if len(labels) == 0:
        b0_error = 0
        b1_error = 0
    else:
        gt_betti_numbers = betti_number(gt)
        pred_betti_numbers = betti_number(pred)

        b0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        b1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])
    return b0_error, b1_error


