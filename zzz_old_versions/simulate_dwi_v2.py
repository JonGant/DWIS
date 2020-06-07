# Description: This program creates a simulated diffusion weighted image for use in  the creation of weighted edge
# adjacency matrix
# Date: April 6th 2020
# Author: Jonathan Gant
# References: Colon-Perez LM, Spindler C, Goicochea S, Triplett W, Parekh M, Montie E, et al. (2015) Dimensionless,
#             Scale Invariant, Edge Weight Metric for the Study of Complex Structural Networks.
#             PLoS ONE 10(7): e0131493. https://doi.org/10.1371/journal.pone.0131493

# import statements
import numpy as np
import math
import nibabel as nib
import os
from tkinter import *
from tkinter import messagebox

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

# main part of the program which simulates the diffusion tensor data
def simulate_dwi_calc(low_bval, high_bval, nshells, xsize, ysize, zsize, xmin, xmax, ymin, ymax, zmin, zmax, dt_data):
    # create b vectors and values
    bvecs = create_bvecs_64_6_2(nshells=nshells)
    bvals = create_bvals_64_6_2(low_bval=low_bval, high_bval=high_bval, nshells=nshells)

    # Create a data array, set values of parameters
    dirs = np.size(bvecs, 0)
    data = np.zeros((xsize, ysize, zsize, dirs), float)
    S0 = 1.0
    VF1 = 1 # Volume Fraction
    for k in range(zsize):
        for j in range(ysize):
            for i in range(xsize):
                for m in range(dirs):
                    term = np.matmul(bvecs[m,:], np.matmul(dt_data[i][j][k], np.transpose(bvecs[m,:])))
                    data[i, j, k, m] = S0 * (VF1 * np.exp(-bvals[m] * term))
    # set the affine
    affine = np.diag([2, 2, 2, 1])
    # make an image file from the numpy array
    array_img = nib.Nifti1Image(data, affine)
    # create result directory if it does not already exist
    if not os.path.isdir("nifti_images"):
        os.mkdir("nifti_images")
    # save the simulated data as a nifti file
    nib.save(array_img, "nifti_images/dwi_" + str(low_bval) + "_" + str(high_bval) + "_" + str(nshells) + "_" +
             str(xsize) + "_" + str(ysize) + "_" + str(zsize) + str(xmin) + str(xmax) + str(ymin) + str(ymax) +
             str(zmin) + str(zmax) + ".nii.gz")
    # print bvals
    file = open("nifti_images/" + str(low_bval) + "_" + str(high_bval) + "_" + str(nshells) + "_" +
             str(xsize) + "_" + str(ysize) + "_" + str(zsize) + str(xmin) + str(xmax) + str(ymin) + str(ymax) +
             str(zmin) + str(zmax) + ".bval", "w+")
    for i in range(len(bvals)):
        file.write(str(bvals[i])+" ")
    file.close()
    # print bvecs
    file = open("nifti_images/" + str(low_bval) + "_" + str(high_bval) + "_" + str(nshells) + "_" +
             str(xsize) + "_" + str(ysize) + "_" + str(zsize) + str(xmin) + str(xmax) + str(ymin) + str(ymax) +
             str(zmin) + str(zmax) + ".bvec", "w+")
    for i in range(3):
        for j in range(len(bvecs)):
            file.write(str(bvecs[j][i]) + " ")
        file.write("\n")
    file.close()

    return

# this function creates the diffusion tensor information file containing the principal eigenvectors for each voxel
def create_dt_data(xsize, ysize, zsize, xmin, xmax, ymin, ymax, zmin, zmax):
    # create data structure for diffusion data
    dt_data = np.zeros([xsize, ysize, zsize, 3, 3], float)
    # Set max values for diffusion data
    max_d = .002
    min_d = .0002
    # check if min is less than max, i.e. the linear gradient is increasing as you move in the positive voxel direction
    if xmin < xmax:
        # cutoff min max values to fit the bounds above
        if xmin < min_d:
            xmin = min_d
        if xmax > max_d:
            xmax = max_d
    elif xmin > xmax:
        # gradient is going in the other direction so we switch the cutoffs
        if xmax < min_d:
            xmax = min_d
        if xmin > max_d:
            xmin = max_d
    elif xmin == xmax:
        if xmin < min_d:
            xmin = min_d
        if xmin > max_d:
            xmin = max_d
        xmax = xmin
    if ymin < ymax:
        # cutoff min max values to fit the bounds above
        if ymin < min_d:
            ymin = min_d
        if ymax > max_d:
            ymax = max_d
    elif ymin > ymax:
        # gradient is going in the other direction so we switch the cutoffs
        if ymax < min_d:
            ymax=min_d
        if ymin > max_d:
            ymin = max_d
    elif ymin == ymax:
        if ymin < min_d:
            ymin = min_d
        if ymin > max_d:
            ymin = max_d
        ymax = ymin
    if zmin < zmax:
        # cutoff min max values to fit the bounds above
        if zmin < min_d:
            zmin = min_d
        if zmax > max_d:
            zmax = max_d
    elif zmin > zmax:
        # gradient is going in the other direction so we switch the cutoffs
        if zmax < min_d:
            zmax = min_d
        if zmin > max_d:
            zmin = max_d
    elif zmin == zmax:
        if zmin < min_d:
            zmin = min_d
        if zmin > max_d:
            zmin = max_d
        zmax = zmin
    # start filling in the diffusion data based on min max values
    # start with x values and create a list of the x gradient values
    if xmin == xmax:
        # there is no gradient!
        xvals = np.full(xsize, xmin)
    elif xmin < xmax:
        # this is a positive gradient in the +x voxel direction
        xvals = np.linspace(xmin, xmax, xsize)
    elif xmin > xmax:
        # this is a negative gradient in the +x voxel direction
        xvals = np.linspace(xmax, xmin, xsize)
    # do the same for the other directions
    if ymin == ymax:
        # there is no gradient!
        yvals = np.full(ysize, ymin)
    elif ymin < ymax:
        # this is a positive gradient in the +x voxel direction
        yvals = np.linspace(ymin, ymax, ysize)
    elif ymin > ymax:
        # this is a negative gradient in the +x voxel direction
        yvals = np.linspace(ymax, ymin, ysize)
    if zmin == zmax:
        # there is no gradient!
        zvals = np.full(zsize, zmin)
    elif zmin < zmax:
        # this is a positive gradient in the +x voxel direction
        zvals = np.linspace(zmin, zmax, zsize)
    elif zmin > zmax:
        # this is a negative gradient in the +x voxel direction
        zvals = np.linspace(zmax, zmin, zsize)
    # go through the full array
    for i in range(xsize):
        for j in range(ysize):
            for k in range(zsize):
                dt_data[i][j][k][0][0] = xvals[i]
                dt_data[i][j][k][1][1] = yvals[j]
                dt_data[i][j][k][2][2] = zvals[k]
    return dt_data

# if file is run as the main file then execute the following code allowing for user input in a simple GUI
if __name__ == "__main__":
    # take user input
    # create window
    window = Tk()
    window.title("Diffusion Weighted Data Simulator")
    window.geometry('500x400')
    #Initialize label widgets and input text boxes
    xsize_lbl = Label(window, text="X size: ", font=("Arial", 12))
    xsize_lbl.grid(column=0, row=0)
    xsize = Entry(window, width=10)
    xsize.grid(column=1,row=0)
    xsize.focus()
    ysize_lbl = Label(window, text="Y size: ", font=("Arial", 12))
    ysize_lbl.grid(column=0, row=1)
    ysize = Entry(window, width=10)
    ysize.grid(column=1, row=1)
    zsize_lbl = Label(window, text="Z size: ", font=("Arial", 12))
    zsize_lbl.grid(column=0, row=2)
    zsize = Entry(window, width=10)
    zsize.grid(column=1, row=2)
    xmin_lbl = Label(window, text="First X diffusion value: ", font=("Arial", 12))
    xmin_lbl.grid(column=0, row=3)
    xmin = Entry(window, width=10)
    xmin.grid(column=1, row=3)
    xmax_lbl = Label(window, text="Last X diffusion value: ", font=("Arial", 12))
    xmax_lbl.grid(column=0, row=4)
    xmax = Entry(window, width=10)
    xmax.grid(column=1, row=4)
    ymin_lbl = Label(window, text="First Y diffusion value: ", font=("Arial", 12))
    ymin_lbl.grid(column=0, row=5)
    ymin = Entry(window, width=10)
    ymin.grid(column=1, row=5)
    ymax_lbl = Label(window, text="Last Y diffusion value: ", font=("Arial", 12))
    ymax_lbl.grid(column=0, row=6)
    ymax = Entry(window, width=10)
    ymax.grid(column=1, row=6)
    zmin_lbl = Label(window, text="First Z diffusion value: ", font=("Arial", 12))
    zmin_lbl.grid(column=0, row=7)
    zmin = Entry(window, width=10)
    zmin.grid(column=1, row=7)
    zmax_lbl = Label(window, text="Last Z diffusion value: ", font=("Arial", 12))
    zmax_lbl.grid(column=0, row=8)
    zmax = Entry(window, width=10)
    zmax.grid(column=1, row=8)
    low_bval_lbl = Label(window, text="Low b-value: ", font=("Arial", 12))
    low_bval_lbl.grid(column=0, row=9)
    low_bval = Entry(window, width=10)
    low_bval.grid(column=1, row=9)
    high_bval_lbl = Label(window, text="High b-value: ", font=("Arial", 12))
    high_bval_lbl.grid(column=0, row=10)
    high_bval = Entry(window, width=10)
    high_bval.grid(column=1, row=10)
    nshells_lbl = Label(window, text="Number of shells: ", font=("Arial", 12))
    nshells_lbl.grid(column=0, row=11)
    nshells = Entry(window, width=10)
    nshells.grid(column=1, row=11)
    # Define and Initialize button to run simulation
    start_lbl = Label(window, text="Waiting for user input: ", font=("Arial", 12))
    start_lbl.grid(column=0, row=12)
    def clicked():
        # Check to make sure user input is correct
        if (xsize.get().isdigit() and ysize.get().isdigit() and zsize.get().isdigit() and low_bval.get().isdigit()
                and high_bval.get().isdigit() and nshells.get().isdigit()):
            if int(nshells.get()) >= 2:
                if int(low_bval.get()) < int(high_bval.get()):
                    # create result directory if it does not already exist
                    if not os.path.isdir("nifti_images"):
                        os.mkdir("nifti_images")
                    # check to see if file of these parameters already exists
                    if not (os.path.exists("nifti_images/dwi_" + str(low_bval) + "_" + str(high_bval) + "_" + str(nshells) + "_" +
                                            str(xsize) + "_" + str(ysize) + "_" + str(zsize) + str(xmin) + str(xmax) + str(ymin) + str(ymax) +
                                            str(zmin) + str(zmax) + ".nii.gz")):
                        if 0 < int(xsize.get()) <= 100 and 0 < int(ysize.get()) <= 100 and 0 < int(zsize.get()) <= 100:
                            try:
                                # create the diffusion tensor info file
                                dt_data = create_dt_data(int(xsize.get()), int(ysize.get()), int(zsize.get()), float(xmin.get()), float(xmax.get()),
                                               float(ymin.get()), float(ymax.get()), float(zmin.get()), float(zmax.get()))
                                # Start the simulation
                                messagebox.showinfo('Simulation Notification', 'Simulation has started')
                                simulate_dwi_calc(int(low_bval.get()), int(high_bval.get()), int(nshells.get()), int(xsize.get()),
                                                  int(ysize.get()), int(zsize.get()), float(xmin.get()), float(xmax.get()),
                                               float(ymin.get()), float(ymax.get()), float(zmin.get()), float(zmax.get()), dt_data)
                            except:
                                messagebox.showinfo('Simulation Notification', 'Please enter float values for XYZ minima and maxima')
                        else:
                            messagebox.showinfo('Simulation Notification', 'Please enter values greater than 0 and less than 100 for XYZ sizes')
                    else:
                        messagebox.showinfo('Simulation Notification', 'File already exists, please enter different parameters')
                else:
                    messagebox.showinfo('Simulation Notification', 'Please ensure that low b-value is less than high b-value')
            else:
                messagebox.showinfo('Simulation Notification', 'Please enter integers greater than 2 for the number of shells')
        else:
            messagebox.showinfo('Simulation Notification', 'Please enter acceptable input type: integers')
    start = Button(window, text="Start Simulation", command=clicked)
    start.grid(column=1, row=12)
    # run the window
    window.mainloop()