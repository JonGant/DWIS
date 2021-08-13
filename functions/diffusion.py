"""
Author: Jonathan Gant
Modification Dates: August 10th 2021
General Description: This script contains the functions which calculate the bvector distribution based on electrostatic repulsion developed by Caruyer, and simulate the diffusion weighted signal.
References:
Caruyer, Emmanuel, Christophe Lenglet, Guillermo Sapiro, and Rachid Deriche.
    "Design of multishell sampling schemes with uniform coverage in diffusion MRI."
    Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-1540.

Global Variables: None
List of variables: Refer to individual subroutine documentation.
Common Blocks: None

List of subroutines (functions):
create_bvecs_and_bvals - creates an array of bvalues and bvectors based on input bvalues and directions per shell. This function utilizies multishell.optimize to generate an evenly
                         distributed set of bvectors based on an electrostatic repulsion model (Caruyer et al. 2013).
rician_noise - add Rician distributed noise to the input image. 
signal_function - calculates a diffusion signal dependent of the tissue type (CSF, GM, WM).
simulate_dwi_calc - simulates the diffusion tensor image by calling create_bvecs_and_bvals, rician_noise, and signal_function. Outputs diffusion weighted images saved as Nifti files.
create_dt_data -  takes in a list of rois and a list of parameters determining the width and curvature of each fiber
                  and creates the diffusion tensor information associated with the fibers.
"""

import numpy as np
import os
import nibabel as nib
from . import multishell
from .geometry import linear_fit, multi_linear_fit, circular_fit
from .roi import roi_mask

def create_bvecs_and_bvals(bvalues, dirpshell):
    """
    Creates an array of bvalues and bvectors based on input bvalues and directions per shell. This function utilizies multishell.optimize to generate an evenly distributed set of bvectors based on an electrostatic repulsion model (Caruyer et al. 2013).
    Parameters:
        bvalues - array of bvalues in units s/mm^2.
        dirpshell - array containing integers specifying the number of directions per bvalue shell.
    Returns:
        [bvecs, bvals] - concatenated array containing the direction vectors for the bvectors as well as an array of bvals with repeated entries corresponding to a given bvalue shell. Hence 
                         bvecs and bvals are the same length.
    """
    totalpoints = sum(dirpshell)
    bvals = []
    for i in range(len(bvalues)):
        for j in range(dirpshell[i]):
            bvals.append(bvalues[i])
    weights = np.ones((totalpoints, totalpoints)) / 2
    bvecs = multishell.optimize(len(dirpshell), dirpshell, weights)

    return [bvecs, bvals]

def rician_noise(image, sigma, rng=None):
    """
    Add Rician distributed noise to the input image.
    Parameters
    ----------
    image : array-like, shape ``(dim_x, dim_y, dim_z)`` or ``(dim_x, dim_y,
        dim_z, K)``
    sigma : double
    rng : random number generator (a numpy.random.RandomState instance).
    """
    n1 = rng.normal(loc=0, scale=sigma, size=image.shape)
    n2 = rng.normal(loc=0, scale=sigma, size=image.shape)
    return np.sqrt((image + n1)**2 + n2**2)

def signal_function(bvec, bval, dt_data, roi):
    """
    Calculates a diffusion signal dependent of the tissue type (CSF, GM, WM).
    Parameters:
        bvec - bvector, direction of the gradient.
        bval - bvalue, strength of the gradient. Units of s/mm^2.
        dt_data - array containing the diffusion tensors for every voxel in the image. 0 if there is no fiber in that voxel.
        roi - labeled array containing the voxels which correspond to a ROI.
    Returns:
        signal - monoexponential model of the diffusion signal.
    """
    gm_dt = np.array([[.0007, 0, 0], [0, .0007, 0], [0, 0, .0007]])
    csf_dt = np.array([[.003, 0, 0], [0, .003, 0], [0, 0, .003]])
    iso_on = False
    n_fibers = len(dt_data)
    if n_fibers >= 1:
        exponent_terms = []
        for fiber_dt in dt_data:
            exponent_terms.append(np.matmul(bvec, np.matmul(fiber_dt, np.transpose(bvec))))
        signal_terms = []
        for term in exponent_terms:
            signal_terms.append(np.exp(-bval * term))
        if iso_on:
            isotropic_term = np.matmul(bvec, np.matmul(csf_dt, np.transpose(bvec)))
            isotropic_signal = np.exp(-bval * isotropic_term)
            signal = (isotropic_signal + sum(signal_terms)) / (n_fibers + 1)
        else:
            signal = sum(signal_terms) / n_fibers
    elif roi > 0:
        gm_term = np.matmul(bvec, np.matmul(gm_dt, np.transpose(bvec)))
        signal = np.exp(-bval * gm_term)
    else:
        csf_term = np.matmul(bvec, np.matmul(csf_dt, np.transpose(bvec)))
        signal = np.exp(-bval * csf_term)
    return signal

def simulate_dwi_calc(xsize, ysize, zsize, bvalue, dirpershell, dt_data, filename, rois, snr, res):
    """
    Simulates the diffusion tensor image by calling create_bvecs_and_bvals, rician_noise, and signal_function. Outputs diffusion weighted images saved as Nifti files.
    Parameters:
        xsize, ysize, zsize - integers defining the size of the simulation space.
        bvalue - strength of the applied gradient. Units of s/mm^2.
        dirpershell - array containing integers specifying the number of directions per bvalue shell.
        dt_data - array containing the diffusion tensors for every voxel in the image. 0 if there is no fiber in that voxel.
        filename - string used to save the image. 
        rois - labeled array containing the voxels which correspond to a ROI.
        snr - signal to noise ratio.
        res - image resolution in mm^3.
    Returns:
        False
    """
    # create b vectors and values
    bvecval = create_bvecs_and_bvals(bvalue, dirpershell)
    bvecs = bvecval[0]
    bvals = bvecval[1]
    # Create a data array, set values of parameters
    dirs = np.size(bvecs, 0)
    data = np.zeros((xsize, ysize, zsize, dirs), float)
    roi = np.zeros((xsize, ysize, zsize), int)
    mask = np.zeros((xsize, ysize, zsize), int)
    # reformat dt_data to be per voxel instead of per fiber
    dt_data_collated = np.empty((xsize, ysize, zsize), dtype='object')
    for i in range(xsize):
        for j in range(ysize):
            for k in range(zsize):
                voxel_dts = []
                for fiber in dt_data:
                    if not np.equal(fiber[i, j, k], np.zeros((3, 3))).all():
                        voxel_dts.append(fiber[i, j, k])
                        mask[i, j, k] = 1
                dt_data_collated[i, j, k] = voxel_dts
    print("calculating signal for each voxel...")
    S0 = 1.0
    for k in range(zsize):
        for j in range(ysize):
            for i in range(xsize):
                roi[i, j, k] = roi_mask(rois, i, j, k)
                for m in range(dirs):
                    data[i, j, k, m] = S0 * signal_function(bvecs[m], bvals[m], dt_data_collated[i, j, k], roi[i, j, k])

    # add noise
    if snr != 0:
        sigma = S0 / snr
        rng = np.random.RandomState(42)
        dwis = rician_noise(data, sigma, rng=rng)
    elif snr == 0:
        dwis = data
    # set the affine
    affine = np.diag([res, res, res, 1])
    # make an image file from the numpy array
    array_img = nib.Nifti1Image(dwis, affine)
    # make a mask image
    mask_img = nib.Nifti1Image(mask, affine)
    # make a roi image
    roi_img = nib.Nifti1Image(roi, affine)
    # create result directory if it does not already exist
    if not os.path.isdir("nifti_images"):
        os.mkdir("nifti_images")
    # save the simulated data as a nifti file
    nib.save(array_img, "nifti_images/" + str(filename) + ".nii.gz")
    # save the mask
    nib.save(mask_img, "nifti_images/" + str(filename) + "_mask.nii.gz")
    # save the roi
    nib.save(roi_img, "nifti_images/" + str(filename) + "_roi.nii.gz")
    # print bvals
    file = open("nifti_images/" + str(filename) + ".bval", "w+")
    for i in range(len(bvals)):
        file.write(str(bvals[i])+" ")
    file.close()
    # print bvecs
    file = open("nifti_images/" + str(filename) + ".bvec", "w+")
    for i in range(3):
        for j in range(len(bvecs)):
            file.write(str(bvecs[j][i]) + " ")
        file.write("\n")
    file.close()

    return False

def create_dt_data(xsize, ysize, zsize, fiber_voxels, eigenvalues, radii, fit_types):
    """
    Takes in a list of rois and a list of parameters determining the width and curvature of each fiber and creates the diffusion tensor information associated with the fibers.
    Parameters:
        xsize, ysize, zsize - integers defining the size of the simulation space.
        fiber_voxels - user inputted information for the start/mid/end points of the fibers.
        eigenvalues - user inputted diffusion tensor eigenvalues for each fiber.
        radii - array of fiber radii defining how think each fiber is.
        fit_types - array containing strings specifying whihc fit type to use.
    Returns:
        fiber_data - array containing the diffusion tensors for every voxel in the image. 0 if there is no fiber in that voxel.
    """
    fiber_data = []
    for i in range(len(fiber_voxels)):
        print("starting fiber " + str(i+1) + " calculation...")
        diff = np.array([[eigenvalues[i][0], 0, 0], [0, eigenvalues[i][1], 0], [0, 0, eigenvalues[i][2]]])
        if fit_types[i] == 'linear':
            if len(fiber_voxels[i]) == 2:
                fiber_data.append(linear_fit(xsize, ysize, zsize, fiber_voxels[i], diff, radii[i]))
            elif len(fiber_voxels[i]) == 3:
                fiber_data.append(multi_linear_fit(xsize, ysize, zsize, fiber_voxels[i], diff, radii[i]))
        elif fit_types[i] == 'circular':
            if len(fiber_voxels[i]) == 2:
                # since a circle cannot be uniquely defined with only points default to linear
                print("WARNING: Fiber " + str(i+1) + " cannot be fit with circular fiber since only two reference "
                                                     "points were specified. As such as linear fit was done instead.")
                fiber_data.append(linear_fit(xsize, ysize, zsize, fiber_voxels[i], diff, radii[i]))
            elif len(fiber_voxels[i]) == 3:
                fiber_data.append(circular_fit(xsize, ysize, zsize, fiber_voxels[i], diff, radii[i]))
    return fiber_data