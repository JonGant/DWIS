# This program creates a simulated data set of dMRI images for use in  the creation of weighted edge adjacency matrix
# February 3rd 2020
# Author: Jonathan Gant
# References: Colon-Perez LM, Spindler C, Goicochea S, Triplett W, Parekh M, Montie E, et al. (2015) Dimensionless,
#             Scale Invariant, Edge Weight Metric for the Study of Complex Structural Networks.
#             PLoS ONE 10(7): e0131493. https://doi.org/10.1371/journal.pone.0131493

# import statements
import numpy as np
import math
import nibabel as nib
import os


# initialize b vectors
def create_bvecs_64_6_2(nshells):
    bvecs = np.array([[0.0, 0.0, 0.0], [0.714505, -0.014881, -0.699472], [-0.714505, -0.014881, -0.699472],
             [0.006989, 0.724845, -0.688877], [0.006989, 0.724845, 0.688877], [0.696386, 0.717367, -0.020778],
             [-0.696386, 0.717367, -0.020778], [0.0, 0.0, 0.0], [0.999975, -0.005050, -0.005045],
             [0.000000, 0.999988, -0.004980], [-0.024949, 0.654640, -0.755529], [0.589703, -0.769552, -0.245032],
             [-0.235508, -0.530035, -0.814616], [-0.893552, -0.264193, -0.362996], [0.797988, 0.133987, -0.587590],
             [0.232856, 0.932211, -0.277056], [0.936737, 0.145261, -0.318470], [0.503830, -0.847151, 0.168788],
             [0.344880, -0.850958, 0.396143], [0.456580, -0.636053, 0.622070], [-0.487215, -0.395128, -0.778778],
             [-0.616849, 0.677327, -0.400906], [-0.578228, -0.109670, 0.808471], [-0.825176, -0.525574, -0.207020],
             [0.895346, -0.044861, 0.443106], [0.289606, -0.546793, 0.785586], [0.114907, -0.964118, 0.239319],
             [-0.799923, 0.408399, 0.439696], [0.511672, 0.842750, -0.167226], [-0.789853, 0.158242, 0.592530],
             [0.949160, -0.238854, -0.205046], [0.231882, 0.787519, -0.571003], [-0.019682, -0.193328, 0.980937],
             [0.216050, -0.957283, -0.192175], [0.772394, -0.608092, -0.183388], [-0.160407, 0.361349, -0.918530],
             [-0.146296, 0.735992, 0.660994], [0.887130, 0.421934, -0.187009], [-0.562752, 0.237427, 0.791795],
             [-0.381197, 0.148276, -0.912526], [-0.306491, -0.204151, 0.929723], [-0.332448, -0.135350, -0.933359],
             [-0.961969, -0.270561, -0.037590], [-0.959690, 0.209869, -0.186949], [0.451014, -0.890373, -0.061818],
             [-0.770666, 0.631995, -0.081582], [0.709698, 0.413843, -0.570143], [-0.694854, 0.028740, -0.718577],
             [0.681158, 0.533870, 0.501005], [-0.141170, -0.730096, -0.668603], [-0.740626, 0.393661, -0.544521],
             [-0.102000, 0.826127, -0.554175], [0.584088, -0.600963, -0.545605], [-0.086976, -0.340311, -0.936282],
             [-0.550628, -0.795773, -0.252098], [0.836941, -0.463602, 0.290864], [0.362769, -0.566671, -0.739785],
             [-0.183912, 0.397731, 0.898881], [-0.718323, -0.695704, -0.002830], [0.432372, 0.687090, 0.583919],
             [0.501455, 0.695404, -0.514739], [-0.169795, -0.514666, 0.840410], [0.463674, 0.428929, -0.775259],
             [0.383808, -0.813147, -0.437588], [-0.714437, -0.252685, -0.652480], [0.258407, 0.887816, 0.380801],
             [0.000000, 0.081483, 0.996675], [0.036369, -0.905214, -0.423396], [0.570920, -0.309049, 0.760617],
             [-0.281910, 0.150149, 0.947619], [0.720033, 0.612825, -0.325574], [0.265891, 0.960683, 0.079935]])
    if nshells > 2:
        for i in range(nshells-2):
            dirs = np.array([[0.0, 0.0, 0.0], [0.999975, -0.005050, -0.005045], [0.000000, 0.999988, -0.004980],
                    [-0.024949, 0.654640, -0.755529], [0.589703, -0.769552, -0.245032], [-0.235508, -0.530035, -0.814616],
                    [-0.893552, -0.264193, -0.362996], [0.797988, 0.133987, -0.587590], [0.232856, 0.932211, -0.277056],
                    [0.936737, 0.145261, -0.318470], [0.503830, -0.847151, 0.168788], [0.344880, -0.850958, 0.396143],
                    [0.456580, -0.636053, 0.622070], [-0.487215, -0.395128, -0.778778], [-0.616849, 0.677327, -0.400906],
                    [-0.578228, -0.109670, 0.808471], [-0.825176, -0.525574, -0.207020], [0.895346, -0.044861, 0.443106],
                    [0.289606, -0.546793, 0.785586], [0.114907, -0.964118, 0.239319], [-0.799923, 0.408399, 0.439696],
                    [0.511672, 0.842750, -0.167226], [-0.789853, 0.158242, 0.592530], [0.949160, -0.238854, -0.205046],
                    [0.231882, 0.787519, -0.571003], [-0.019682, -0.193328, 0.980937], [0.216050, -0.957283, -0.192175],
                    [0.772394, -0.608092, -0.183388], [-0.160407, 0.361349, -0.918530], [-0.146296, 0.735992, 0.660994],
                    [0.887130, 0.421934, -0.187009], [-0.562752, 0.237427, 0.791795], [-0.381197, 0.148276, -0.912526],
                    [-0.306491, -0.204151, 0.929723], [-0.332448, -0.135350, -0.933359], [-0.961969, -0.270561, -0.037590],
                    [-0.959690, 0.209869, -0.186949], [0.451014, -0.890373, -0.061818], [-0.770666, 0.631995, -0.081582],
                    [0.709698, 0.413843, -0.570143], [-0.694854, 0.028740, -0.718577], [0.681158, 0.533870, 0.501005],
                    [-0.141170, -0.730096, -0.668603], [-0.740626, 0.393661, -0.544521], [-0.102000, 0.826127, -0.554175],
                    [0.584088, -0.600963, -0.545605], [-0.086976, -0.340311, -0.936282], [-0.550628, -0.795773, -0.252098],
                    [0.836941, -0.463602, 0.290864], [0.362769, -0.566671, -0.739785], [-0.183912, 0.397731, 0.898881],
                    [-0.718323, -0.695704, -0.002830], [0.432372, 0.687090, 0.583919], [0.501455, 0.695404, -0.514739],
                    [-0.169795, -0.514666, 0.840410], [0.463674, 0.428929, -0.775259], [0.383808, -0.813147, -0.437588],
                    [-0.714437, -0.252685, -0.652480], [0.258407, 0.887816, 0.380801], [0.000000, 0.081483, 0.996675],
                    [0.036369, -0.905214, -0.423396], [0.570920, -0.309049, 0.760617], [-0.281910, 0.150149, 0.947619],
                    [0.720033, 0.612825, -0.325574], [0.265891, 0.960683, 0.079935]])
            bvecs = np.transpose(np.concatenate((np.transpose(bvecs), np.transpose(dirs)), axis=1))

    return bvecs

# initialize b values
def create_bvals_64_6_2(low_bval, high_bval, nshells):
    num = nshells + 6 + (nshells - 1) * 64
    bvals = np.empty(num, float)
    bvals[0] = 0.0
    for m in range(1,7):
        bvals[m] = low_bval
    for m in range(7,num):
        n = m - 7
        hbval = high_bval * (math.floor(n / 65) + 1)
        bvals[m] = hbval
        counter2 = n % 65
        if counter2 == 0:
            bvals[m] = 0

    return bvals

# create the diffusion tensor elements
def initialize_D(num1, num2, fract_ang1, fract_ang2, phi_ang, FA_value, AD, AD01, sf):
    if sf == 0:
        alpha1 = np.pi/3 + (np.pi/6) * fract_ang1
    elif sf == 1:
        alpha1 = np.pi/2 * fract_ang1
    elif sf == 2:
        alpha1 = np.pi/4

    d1 = .0017
    d2 = .0003
    d3 = .0003

    Dz = np.array([[d2, 0, 0], [0, d2, 0], [0, 0, d1]])
    D2z = np.array([[d3, 0, 0], [0, d3, 0], [0, 0, d1]])
    rotalpha1_y = np.array([[np.cos(alpha1), 0, np.sin(alpha1)], [0, 1, 0], [-np.sin(alpha1), 0, np.cos(alpha1)]])
    Dalpha1_zx = np.matmul(np.matmul(rotalpha1_y, Dz), np.transpose(rotalpha1_y))

    return [D2z, Dalpha1_zx]

# main part of the program which simulates the diffusion tensor data
def simulate_dwi_calc():
    nshells = 2  # number of shells, must be a number greater than or equal to 2
    prefix = '/export/faraday/manishamin/Downloads/paper_data/DOT_test'
    filedir = '/export/faraday/manishamin/Downloads/DOT_sav/'
    filename = filedir + '2f_db.sav'
    sd = 0.0  # standard deviation for noise

    # create b vectors and values
    bvecs = create_bvecs_64_6_2(nshells)
    bvals = create_bvals_64_6_2(100, 1000, nshells)

    # Create a data array, set values of parameters
    r = 5
    ph = 5
    sl = 5
    dirs = np.size(bvecs, 0)
    print(dirs)
    data = np.zeros((r, ph, sl, dirs), float)
    sf = 1  # Has to do with angles. Check initialize_D function
    S0 = 1.0
    f01 = 0.0  # volume fraction of the isotropic diffusion
    f02 = 0.0  # volume fraction of free water
    f3 = (1. / 3.) * (1.0 - f01)  # volume fraction of each fiber population
    f2 = 0.5 * (1.0 - f01)
    f1 = 1.0 - f01

    FA_value = 0.8  # FA value of each fiber population
    AD = 1.0e-03  # AD value of each fiber population
    AD01 = .003  # AD value of isotropic diffusion

    # generate diffusion weighted data
    xsize = r
    xmin = ph
    zsize = sl

    for k in range(zsize):
        for j in range(ysize):
            for i in range(xsize):
                fract_ang1 = j / (ysize - 1.)
                fract_ang2 = i / (xsize - 1.)
                phi_ang = 1
                VF1 = 1 # .5 + .3 * (i / (xsize - 1))

                VF2 = 1 - VF1
                z_ang = (np.pi / 4)
                num1 = 0
                num2 = 0
                D_matrix = np.array([[.0017, 0, 0], [0, .0003, 0], [0, 0, .0003]])#initialize_D(num1, num2, fract_ang1, fract_ang2, phi_ang, FA_value, AD, AD01, sf)

                for m in range(dirs):
                    term1 = np.matmul(bvecs[m,:], np.matmul(D_matrix, np.transpose(bvecs[m,:])))
                    # term2 = np.matmul(bvecs[m,:], np.matmul(D_matrix[1], np.transpose(bvecs[m,:])))
                    data[i, j, k, m] = S0 * (VF1 * np.exp(-bvals[m] * term1)) # + VF2 * np.exp(-bvals[m] * term2))

    DWdata = data
    # set the affine
    affine = np.diag([2, 2, 2, 1])
    # make an image file from the numpy array
    array_img = nib.Nifti1Image(DWdata, affine)
    # create result directory if it does not already exist
    if not os.path.isdir("nifti_images"):
        os.mkdir("nifti_images")
    # save the simulated data as a nifti file
    nib.save(array_img, "nifti_images/dwi.nii.gz")
    # print bvals
    file = open("simulated_data.bval", "w+")
    for i in range(len(bvals)):
        file.write(str(bvals[i])+" ")
    file.close()
    # print bvecs
    file = open("simulated_data.bvec", "w+")
    for i in range(3):
        for j in range(len(bvecs)):
            file.write(str(bvecs[j][i]) + " ")
        file.write("\n")
    file.close()

    return DWdata


# runs the functions defined above and prints out the saved file to check that it saved properly
simulate_dwi_calc()
image = nib.load("nifti_images/dwi.nii.gz")
print(image)