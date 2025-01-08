import numpy as np
from skimage.morphology import skeletonize_3d

def cl_score(volume, skel):
    """
    Computes the skeleton volume overlap

    Args:
        volume: ndarray, binary segmentation
        skel: ndarray, skeleton of another binary segmentation

    Returns:
        output: float,  skeleton volume intersection ratio
    """
    if np.sum(skel) == 0:
        output = 0
    else:
        output = np.sum(volume*skel)/np.sum(skel)
    return output

def centerline_dice(gt, pred):
    """
    Computes the centerline dice score for binary segmentation 
    (S. Shit et al. clDice - a novel topology-preserving loss function for tubular structure segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 16555-16564.) 

    Args:
        gt: ndarray, ground truth
        pred: ndarray, prediction

    Returns:
        cldice: cldice value
    """
    smooth = 1e-6
    tprec = cl_score(pred,skeletonize_3d(gt))
    tsens = cl_score(gt,skeletonize_3d(pred))

    cldice = (2*tprec*tsens+smooth)/(tprec+tsens+smooth)
    print(cldice)
    return cldice
 