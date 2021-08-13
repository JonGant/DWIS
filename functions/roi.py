"""
Author: Jonathan Gant
Modification Dates: August 10th 2021
General Description: This script contains all the functions used for calculating the ROI mask.

Global Variables: None
List of variables: Refer to individual subroutine documentation.
Common Blocks: None

List of subroutines (functions):
roi_signal - calculates an isotropic diffusion weighted MRI signal given a bvector and bvalue. Current version does not use this function.
roi_mask - checks if a given voxel is contained in an ROI and returns the ROI number if the voxel is indeed within an ROI. This function is used in diffusion.py to calculate an ROI mask.
"""

import numpy as np


def roi_signal(bvec, bval):
    """
    Calculates an isotropic diffusion weighted MRI signal given a bvector and bvalue.
    Parameters:
        bvec - unit vector specifying the direction of the bvector (direction of the gradient) 
        bval - value specifying the magnitude of the bvalue (strength of the gradient) in units s/mm^2
    Returns:
        isotropic signal calculation for the given bvec and bval.
    Variables:
        bvec - unit vector specifying the direction of the bvector (direction of the gradient)
        bval - value specifying the magnitude of the bvalue (strength of the gradient) in units s/mm^2
        isotropic_dt - 3 by 3 isotropic diffusion tensor array
        isotropic_term - product of diffusion tensor matrix with a given bvector
        isotropic_signal - mono-exponential model of the diffusion signal
    """
    isotropic_dt = np.array([[.003, 0, 0], [0, .003, 0], [0, 0, .003]])
    isotropic_term = np.matmul(bvec, np.matmul(isotropic_dt, np.transpose(bvec)))
    isotropic_signal = np.exp(-bval * isotropic_term)
    return isotropic_signal


def roi_mask(rois, i, j, k):
    """
    Checks if a given voxel is contained in an ROI and returns the ROI number if the voxel is indeed within an ROI.
    Parameters:
        rois - labeled array (each entry along axis=0 is a distinct ROI) containing a list of voxels contained within the user specified ROIs
        i, j, k - voxel coordinates
    Returns:
        ROI label for voxel i, j, k. 0 if the voxel is not in an ROI.
    Variables:
        rois - labeled array containing a list of voxels contained within the user specified ROIs
        i, j, k - x, y, z coordinates of a voxel
    """
    for a in range(len(rois)):
        for b in range(len(rois[a])):
            if rois[a][b][0] == i and rois[a][b][1] == j and rois[a][b][2] == k:
                return a + 1
    return 0