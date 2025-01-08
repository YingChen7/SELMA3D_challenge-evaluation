import numpy as np

def volume_dice(gt, pred):
    """
    Compute dice score for binary segmentation

    Args:
        - gt: ndarray, ground truth
        - pred: ndarray, prediction
    Returns:
        - dsc: float, dice score value
    """
    
    inter = np.sum(pred * gt)
    l = np.sum(gt)
    r = np.sum(pred)
    if float(l + r)==0:
      dsc = 1.0
    else:
      dsc = (2 * float(inter)) / float(l + r)
    # print(dsc)
    return dsc





