# Description: This program creates a simulated diffusion weighted image with complex fiber geometry
# Date: March 7 2021
# Author: Jonathan Gant
# References:   Caruyer, Emmanuel, Christophe Lenglet, Guillermo Sapiro, and Rachid Deriche.
#                   "Design of multishell sampling schemes with uniform coverage in diffusion MRI."
#                   Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-1540.

# import statements
import numpy as np
from sympy import solve, symbols, Eq
import nibabel as nib
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from qspace import multishell


# initialize b vectors
def create_bvecs_and_bvals(bvalues, dirpshell):
    totalpoints = sum(dirpshell)
    bvals = []
    for i in range(len(bvalues)):
        for j in range(dirpshell[i]):
            bvals.append(bvalues[i])
    weights = np.ones((totalpoints, totalpoints)) / 2
    bvecs = multishell.optimize(len(dirpshell), dirpshell, weights)

    return [bvecs, bvals]


def roi_signal(bvec, bval):
    isotropic_dt = np.array([[.003, 0, 0], [0, .003, 0], [0, 0, .003]])
    isotropic_term = np.matmul(bvec, np.matmul(isotropic_dt, np.transpose(bvec)))
    isotropic_signal = np.exp(-bval * isotropic_term)
    return isotropic_signal


def roi_mask(rois, i, j, k):
    # check if there is a roi or not, returns roi number if there is a roi
    for a in range(len(rois)):
        for b in range(len(rois[a])):
            if rois[a][b][0] == i and rois[a][b][1] == j and rois[a][b][2] == k:
                return a + 1
    return 0


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


# main part of the program which simulates the diffusion tensor data
def simulate_dwi_calc(xsize, ysize, zsize, bvalue, dirpershell, dt_data, filename, rois, snr, res):
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


# function rotates diffusion data in a voxel to be in the direction of the two angles phi and theta
def rotate(voxel_data, phi, theta):
    rot_y = np.array([[np.cos(theta - np.pi/2), 0, -np.sin(theta - np.pi/2)], [0, 1, 0],
                          [np.sin(theta - np.pi/2), 0, np.cos(theta - np.pi/2)]])
    rot_z = np.array([[np.cos(-phi), -np.sin(-phi), 0], [np.sin(-phi), np.cos(-phi), 0], [0, 0, 1]])
    rotated_voxel_data = np.dot(np.dot(rot_z, np.dot(np.dot(rot_y, voxel_data), rot_y.T)), rot_z.T)
    return rotated_voxel_data


def find_angles(direction):
    phi = np.arctan2(direction[1], direction[0])
    theta = np.arccos(direction[2])
    return phi, theta


# find the plane containing three points
def find_plane(points):
    vec1 = (points[1] - points[0])/np.linalg.norm(points[1] - points[0])
    vec2 = (points[2] - points[0])/np.linalg.norm(points[2] - points[0])
    plane_vec = np.cross(vec1, vec2)
    plane_vec = plane_vec / np.linalg.norm(plane_vec)
    d = np.dot(plane_vec, points[0])
    return plane_vec, d


# find the sphere containing three points
def find_circle(points):
    plane_vec, d = find_plane(points)
    a, b, c, r = symbols("a b c r", real=True)
    eq1 = Eq((points[0][0] - a) ** 2 + (points[0][1] - b) ** 2 + (points[0][2] - c) ** 2, r ** 2)
    eq2 = Eq((points[1][0] - a) ** 2 + (points[1][1] - b) ** 2 + (points[1][2] - c) ** 2, r ** 2)
    eq3 = Eq((points[2][0] - a) ** 2 + (points[2][1] - b) ** 2 + (points[2][2] - c) ** 2, r ** 2)
    eq4 = Eq(plane_vec[0] * a + plane_vec[1] * b + plane_vec[2] * c, d)
    solution = solve([eq1, eq2, eq3, eq4])
    center = np.array([float(solution[1][a]), float(solution[1][b]), float(solution[1][c])])
    radius = float(solution[1][r])
    return center, radius, plane_vec, d


# interpolate circle function with given radius to define a curved fiber
def circular_fit(xsize, ysize, zsize, points, eigenvalues, fiber_radius):
    fiber_dts = np.zeros((xsize, ysize, zsize, 3, 3))
    n_steps = int((xsize + ysize + zsize) / 3)
    center, circle_radius, plane_vec, d = find_circle(points)
    v1 = (points[0] - center)/np.linalg.norm(points[0] - center)
    v2 = np.cross(plane_vec, v1)
    v2 = v2 / np.linalg.norm(v2)

    # define inner parametric function for the circle in three dimensions
    def circle_func(angle):
        return center + circle_radius * np.cos(angle) * v1 + circle_radius * np.sin(angle) * v2

    # find angle sweeped out by the circle function
    start = points[0] - center
    middle = points[1] - center
    finish = points[2] - center
    sweep_angle = np.arccos(np.dot(start, middle) / (np.linalg.norm(start) * np.linalg.norm(middle))) + \
                  np.arccos(np.dot(finish, middle) / (np.linalg.norm(finish) * np.linalg.norm(middle)))
    step_size = sweep_angle / n_steps
    t = step_size
    while t < sweep_angle:
        circle_point = circle_func(t)
        tangent_vec = np.cross(plane_vec, circle_point - center)
        tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)
        circular_d = np.dot(tangent_vec, circle_point)
        phi, theta = find_angles(tangent_vec)
        rotated_dt = rotate(eigenvalues, phi, theta)
        for i in range(xsize):
            for j in range(ysize):
                for k in range(zsize):
                    voxel = np.array([i, j, k])
                    s = circular_d - np.dot(tangent_vec, voxel)
                    closest_point_on_plane = voxel + s * tangent_vec
                    if np.linalg.norm(voxel - closest_point_on_plane) <= 1:
                        # find the point on the line which minimizes the distance to the test point
                        line_min_dist = np.linalg.norm(circle_point - closest_point_on_plane)
                        if line_min_dist <= fiber_radius:
                            if np.equal(fiber_dts[i, j, k], np.zeros((3, 3))).all():
                                fiber_dts[i, j, k] = rotated_dt
        t += step_size
    return fiber_dts


# interpolate linear function with a given radius to define a straight fiber
def linear_fit(xsize, ysize, zsize, points, eigenvalues, radius):
    # initialize fiber diffusion tensor array
    fiber_dts = np.zeros((xsize, ysize, zsize, 3, 3))

    # calculate the direction of the fiber
    displacement = points[1] - points[0]
    range_t = np.linalg.norm(displacement)
    direction = displacement / range_t

    # find theta and phi for the direction vector
    phi, theta = find_angles(direction)
    rotated_dt = rotate(eigenvalues, phi, theta)

    def linear_eq(t):
        return points[0] + t * direction

    for i in range(xsize):
        for j in range(ysize):
            for k in range(zsize):
                voxel = np.array([i, j, k])
                d = np.dot(direction, voxel)
                t = d - np.dot(direction, points[0])
                if 0 < t <= range_t:
                    closest_point_on_line = linear_eq(t)
                    if np.linalg.norm(closest_point_on_line - voxel) <= radius:
                        fiber_dts[i, j, k] = rotated_dt

    return fiber_dts


def multi_linear_fit(xsize, ysize, zsize, points, eigenvalues, radius):
    fiber_dts = np.zeros((xsize, ysize, zsize, 3, 3))
    fiber1 = linear_fit(xsize, ysize, zsize, points[0:2], eigenvalues, radius)
    fiber2 = linear_fit(xsize, ysize, zsize, points[1:3], eigenvalues, radius)
    for i in range(xsize):
        for j in range(ysize):
            for k in range(zsize):
                # check voxel has info from both fibers or not
                if not np.equal(fiber1[i, j, k], np.zeros((3,3))).all() and not np.equal(fiber2[i, j, k], np.zeros((3,3))).all():
                    # create new dt data with direction being the addition of the two directions
                    direction = (points[2] - points[0])/np.linalg.norm(points[2] - points[0])
                    # find theta and phi for the direction vector
                    phi, theta = find_angles(direction)
                    rotated_dt = rotate(eigenvalues, phi, theta)
                    fiber_dts[i, j, k] = rotated_dt
                elif not np.equal(fiber1[i][j][k], np.zeros((3,3))).all():
                    fiber_dts[i, j, k] = fiber1[i, j, k]
                elif not np.equal(fiber2[i][j][k], np.zeros((3,3))).all():
                    fiber_dts[i, j, k] = fiber2[i, j, k]
    return fiber_dts


# this function takes in a list of rois and a list of parameters determining the width and curvature of each fiber
# and creates the diffusion tensor information associated with the fibers
def create_dt_data(xsize, ysize, zsize, fiber_voxels, eigenvalues, radii, fit_types):
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


# if file is run as the main file then execute the following code allowing for user input in a simple GUI
if __name__ == "__main__":

    LARGE_FONT = ("Arial", 12)

    class DWIS(tk.Tk):
        def __init__(self, *args, **kwargs):

            tk.Tk.__init__(self, *args, **kwargs)
            tk.Tk.wm_title(self, "DWIS")

            tk.Tk.iconbitmap(self, default="DWIS_icon.ico")

            container = tk.Frame(self)
            container.pack(side="top", fill="both", expand=True)
            container.grid_rowconfigure(0, weight=1)
            container.grid_columnconfigure(0, weight=1)

            self.frames = {}

            for F in (MainMenu, AddROI, AddFiber):

                frame = F(container, self)

                self.frames[F] = frame

                frame.grid(row=0, column=0, sticky="nsew")

            self.show_frame(MainMenu)

        def show_frame(self, cont):
            frame = self.frames[cont]
            frame.tkraise()

        def get_frame(self, PageName):
            frame = self.frames[PageName]
            return frame

    class MainMenu(tk.Frame):
        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)

            # Initialize label widgets, input text boxes and buttons to navigate between frames

            label = ttk.Label(self, width=25, text="Main Menu", font=LARGE_FONT, anchor='e')
            label.grid(row=0, column=0, columnspan=2)

            button1 = tk.Button(self, text="Add ROIs", command=lambda: controller.show_frame(AddROI))
            button1.grid(row=1, column=0, sticky='E')

            button2 = tk.Button(self, text="Add Fibers", command=lambda: controller.show_frame(AddFiber))
            button2.grid(row=1, column=1)

            xsize_lbl = ttk.Label(self, width=24, text="X size:", font=LARGE_FONT, anchor='e')
            xsize_lbl.grid(row=2, column=0)
            xsize = ttk.Entry(self, width=10)
            xsize.grid(row=2, column=1)
            xsize.focus()

            ysize_lbl = ttk.Label(self, width=24, text="Y size:", font=LARGE_FONT, anchor='e')
            ysize_lbl.grid(row=3, column=0)
            ysize = ttk.Entry(self, width=10)
            ysize.grid(row=3, column=1)

            zsize_lbl = ttk.Label(self, width=24, text="Z size:", font=LARGE_FONT, anchor='e')
            zsize_lbl.grid(row=4, column=0)
            zsize = ttk.Entry(self, width=10)
            zsize.grid(row=4, column=1)

            # enter as comma separated list
            bval_lbl = ttk.Label(self, width=25, text="List of b-values:", font=LARGE_FONT, anchor='e')
            bval_lbl.grid(row=5, column=0)
            bval = ttk.Entry(self, width=10)
            bval.grid(row=5, column=1)

            # enter as comma separated list
            nshells_lbl = ttk.Label(self, width=25, text="Number of directions per shell:", font=LARGE_FONT, anchor='e')
            nshells_lbl.grid(row=6, column=0)
            nshells = ttk.Entry(self, width=10)
            nshells.grid(row=6, column=1)

            te_lbl = ttk.Label(self, width=24, text="TE:", font=LARGE_FONT, anchor='e')
            te_lbl.grid(row=7, column=0)
            te = ttk.Entry(self, width=10)
            te.grid(row=7, column=1)

            tr_lbl = ttk.Label(self, width=24, text="TR:", font=LARGE_FONT, anchor='e')
            tr_lbl.grid(row=8, column=0)
            tr = ttk.Entry(self, width=10)
            tr.grid(row=8, column=1)

            snr_lbl = ttk.Label(self, width=24, text="SNR:", font=LARGE_FONT, anchor='e')
            snr_lbl.grid(row=9, column=0)
            snr = ttk.Entry(self, width=10)
            snr.grid(row=9, column=1)

            res_lbl = ttk.Label(self, width=24, text="Resolution:", font=LARGE_FONT, anchor='e')
            res_lbl.grid(row=10, column=0)
            res = ttk.Entry(self, width=10)
            res.grid(row=10, column=1)

            filename_lbl = ttk.Label(self, width=25, text="File name:", font=LARGE_FONT, anchor='e')
            filename_lbl.grid(row=11, column=0)
            filename = ttk.Entry(self, width=10)
            filename.grid(row=11, column=1)

            # get roi info
            def get_roi_info():
                AddROIFrame = controller.get_frame(AddROI)
                return AddROIFrame.get_roi_list()

            def set_roi_info(roi_list):
                AddROIFrame = controller.get_frame(AddROI)
                return AddROIFrame.set_roi_list(roi_list)

            # get fiber info
            def get_fiber_info():
                AddfiberFrame = controller.get_frame(AddFiber)
                return AddfiberFrame.get_fiber_list()

            def set_fiber_info(fiber_list):
                AddfiberFrame = controller.get_frame(AddFiber)
                return AddfiberFrame.set_fiber_list(fiber_list)

            # start simulation command
            def start_sim():
                save_func()
                roi_list = get_roi_info()
                fiber_list = get_fiber_info()
                # format roi list
                roi_list_formatted = []
                for i in range(len(roi_list)):
                    if roi_list[i] is not None and roi_list[i] != '':
                        if roi_list[i][0] == "cuboid":
                            voxel = roi_list[i][1].split(sep=', ')
                            voxels_in_cuboid = []
                            for j in range(len(voxel)):
                                if '(' in voxel[j]:
                                    voxel[j] = voxel[j][1:]
                                if ')' in voxel[j]:
                                    voxel[j] = voxel[j][:len(voxel[j]) - 1]
                            center = np.array([int(voxel[0]), int(voxel[1]), int(voxel[2])])
                            voxel = roi_list[i][2].split(sep=', ')
                            for j in range(len(voxel)):
                                if '(' in voxel[j]:
                                    voxel[j] = voxel[j][1:]
                                if ')' in voxel[j]:
                                    voxel[j] = voxel[j][:len(voxel[j]) - 1]
                            sizes = np.array([int(voxel[0]), int(voxel[1]), int(voxel[2])])
                            voxels_in_cuboid.append(center)
                            even = sizes % 2 == 0
                            if even[0]:
                                for i in range(1, int((sizes[0] / 2))):
                                    if i != (sizes[0] / 2) - 1:
                                        voxels_in_cuboid.append(center + np.array([i, 0, 0]))
                                    voxels_in_cuboid.append(center - np.array([i, 0, 0]))
                            elif not even[0] and sizes[0] > 1:
                                for i in range(1, 1 + int((sizes[0] - 1) / 2)):
                                    voxels_in_cuboid.append(center + np.array([i, 0, 0]))
                                    voxels_in_cuboid.append(center - np.array([i, 0, 0]))
                            if even[1]:
                                base_voxels = voxels_in_cuboid.copy()
                                for j in range(1, int((sizes[0] / 2))):
                                    for voxel in base_voxels:
                                        if j != (sizes[1] / 2) - 1:
                                            voxels_in_cuboid.append(voxel + np.array([0, j, 0]))
                                        voxels_in_cuboid.append(voxel - np.array([0, j, 0]))
                            elif not even[1] and sizes[1] > 1:
                                base_voxels = voxels_in_cuboid.copy()
                                for j in range(1, 1 + int((sizes[1] - 1) / 2)):
                                    for voxel in base_voxels:
                                        voxels_in_cuboid.append(voxel + np.array([0, j, 0]))
                                        voxels_in_cuboid.append(voxel - np.array([0, j, 0]))
                            if even[2]:
                                base_voxels = voxels_in_cuboid.copy()
                                for k in range(1, int((sizes[0] / 2))):
                                    for voxel in base_voxels:
                                        if k != (sizes[2] / 2) - 1:
                                            voxels_in_cuboid.append(voxel + np.array([0, 0, k]))
                                        voxels_in_cuboid.append(voxel - np.array([0, 0, k]))
                            elif not even[2] and sizes[2] > 1:
                                base_voxels = voxels_in_cuboid.copy()
                                for k in range(1, 1 + int((sizes[2] - 1) / 2)):
                                    for voxel in base_voxels:
                                        voxels_in_cuboid.append(voxel + np.array([0, 0, k]))
                                        voxels_in_cuboid.append(voxel - np.array([0, 0, k]))
                            roi_list_formatted.append(voxels_in_cuboid)
                        elif roi_list[i][0] == "sphere":
                            voxel = roi_list[i][1].split(sep=', ')
                            voxels_in_sphere = []
                            for j in range(len(voxel)):
                                if '(' in voxel[j]:
                                    voxel[j] = voxel[j][1:]
                                if ')' in voxel[j]:
                                    voxel[j] = voxel[j][:len(voxel[j]) - 1]
                            center = np.array([int(voxel[0]), int(voxel[1]), int(voxel[2])])
                            radius = float(roi_list[i][2])
                            for i in range(int(xsize.get())):
                                for j in range(int(ysize.get())):
                                    for k in range(int(zsize.get())):
                                        test_point = np.array([i, j, k])
                                        if np.linalg.norm(test_point - center) <= radius:
                                            voxels_in_sphere.append(test_point)
                            roi_list_formatted.append(voxels_in_sphere)
                        elif roi_list[i][0] == "manual":
                            split_roi = roi_list[i][1].split(sep='), (')
                            coord_list = []
                            for voxel in split_roi:
                                if '(' in voxel:
                                    voxel = voxel[1:]
                                if ')' in voxel:
                                    voxel = voxel[:len(voxel) - 1]
                                coords = voxel.split(sep=', ')
                                coord_list.append(np.array([int(coords[0]), int(coords[1]), int(coords[2])]))
                            roi_list_formatted.append(coord_list)
                # format fiber lists
                fiber_list_formatted = []
                eigenvalues = np.empty((len(fiber_list), 3))
                radii = []
                fiber_voxels = []
                fit_types = []
                for i in range(len(fiber_list)):
                    if fiber_list[i][0] is not None and fiber_list[i][0] != '':
                        split_fiber = fiber_list[i][0].split(sep=', ')
                        fiber_list_formatted.append([int(split_fiber[0]), int(split_fiber[1])])
                    if fiber_list[i][1] is not None and fiber_list[i][1] != '':
                        split_eigen = fiber_list[i][1].split(sep=', ')
                        eigenvalues[i] = np.array([float(split_eigen[0]), float(split_eigen[1]), float(split_eigen[2])])
                    if fiber_list[i][2] is not None and fiber_list[i][2] != '':
                        radius = float(fiber_list[i][2])
                        radii.append(radius)
                    if fiber_list[i][3] is not None and fiber_list[i][3] != '':
                        fiber_voxel = []
                        split_roi = fiber_list[i][3].split(sep='), (')
                        for j in range(len(split_roi)):
                            if '(' in split_roi[j]:
                                voxel = split_roi[j][1:]
                            elif ')' in split_roi[j]:
                                voxel = split_roi[j][:len(split_roi[j])-1]
                            else:
                                voxel = split_roi[j]
                            coords = voxel.split(sep=', ')
                            fiber_voxel.append(np.array([int(coords[0]), int(coords[1]), int(coords[2])]))
                        fiber_voxels.append(fiber_voxel)
                    if fiber_list[i][4] is not None and fiber_list[i][4] != '':
                        fit_types.append(str(fiber_list[i][4]))

                # format bvals and directions
                bvals = []
                split_bval = bval.get().split(", ")
                for term in split_bval:
                    bvals.append(float(term))

                dir_shell = []
                split_nshells = nshells.get().split(", ")
                for term in split_nshells:
                    dir_shell.append(int(term))

                # check that fiber voxels are in the roi and if a third point is specified that it is unique
                correct_num_voxels = True
                two_voxels_same = False
                roi_check = []
                for i in range(len(fiber_voxels)):
                    if len(fiber_voxels[i]) == 2:
                        for j in range(2):
                            voxels = roi_list_formatted[fiber_list_formatted[i][j] - 1]
                            voxel_check = False
                            for k in range(len(voxels)):
                                if np.equal(voxels[k], fiber_voxels[i][j]).all():
                                    voxel_check = True
                            roi_check.append(voxel_check)
                    elif len(fiber_voxels[i]) == 3:
                        for j in range(len(fiber_voxels[i])):
                            if j != 1:
                                if j == 2:
                                    voxels = roi_list_formatted[fiber_list_formatted[i][j - 1] - 1]
                                    voxel_check = False
                                    for k in range(len(voxels)):
                                        if np.equal(voxels[k], fiber_voxels[i][j]).all():
                                            voxel_check = True
                                    roi_check.append(voxel_check)
                                else:
                                    voxels = roi_list_formatted[fiber_list_formatted[i][j] - 1]
                                    voxel_check = False
                                    for k in range(len(voxels)):
                                        if np.equal(voxels[k], fiber_voxels[i][j]).all():
                                            voxel_check = True
                                    roi_check.append(voxel_check)
                        if np.array_equal(fiber_voxels[i][0], fiber_voxels[i][1]):
                            two_voxels_same = True
                        elif np.array_equal(fiber_voxels[i][0], fiber_voxels[i][2]):
                            two_voxels_same = True
                        elif np.array_equal(fiber_voxels[i][1], fiber_voxels[i][2]):
                            two_voxels_same = True
                    elif len(fiber_voxels[i]) > 3 or len(fiber_voxels[i]) < 2:
                        correct_num_voxels = False

                fiber_format = all(roi_check)

                # start simulation functions
                if not two_voxels_same:
                    if correct_num_voxels:
                        if fiber_format:
                            if xsize.get().isdigit() and ysize.get().isdigit() and zsize.get().isdigit():
                                if not (',' in bvals or ',' in dir_shell):
                                    if len(bvals) == len(dir_shell):
                                        # create result directory if it does not already exist
                                        if not os.path.isdir("nifti_images"):
                                            print("creating directory for images...")
                                            os.mkdir("nifti_images")
                                        if 0 < int(xsize.get()) <= 100 and 0 < int(ysize.get()) <= 100 and 0 < int(zsize.get()) <= 100:
                                            messagebox.showinfo('Simulation Notification', 'Simulation has started')
                                            # create the diffusion tensor info file
                                            print("starting fiber geometry calculation...")
                                            dt_data = create_dt_data(int(xsize.get()), int(ysize.get()),
                                                                     int(zsize.get()), fiber_voxels,
                                                                     eigenvalues, radii, fit_types)
                                            # Start the simulation
                                            param = True
                                            while(param):
                                                print("simulating image based on fiber geometry...")
                                                param = simulate_dwi_calc(int(xsize.get()), int(ysize.get()), int(zsize.get()),
                                                                          bvals, dir_shell, dt_data, filename.get(), roi_list_formatted, float(snr.get()), float(res.get()))
                                            messagebox.showinfo('Simulation Notification', 'Simulation successfully finished')
                                        else:
                                            messagebox.showerror('Simulation Error',
                                                                'Please enter values greater than 0 and less than 100 for XYZ sizes')
                                    else:
                                        messagebox.showerror('Simulation Error',
                                                            'Please ensure that the list of b-values and directions per shell are the same length')
                                else:
                                    messagebox.showerror('Simulation Error',
                                                        'Please enter a comma separated list for b-values and directions per shell')
                            else:
                                messagebox.showerror('Simulation Error', 'Please enter acceptable input type: integers')
                        else:
                            messagebox.showerror('Simulation Error', 'Please enter fiber voxels that are in the respective ROIs')
                    else:
                        messagebox.showerror('Simulation Error',
                                            'Please enter two fiber voxels that are in the respective ROIs or three voxels where the first and third voxels are in the respective ROIs and the second voxel is unique')
                else:
                    messagebox.showerror('Simulation Error',
                                        'Please enter fiber voxels that are distinct from each other')

                return
            # button for starting simulation
            button3 = tk.Button(self, text="Start Simulation", command=start_sim)
            button3.grid(row=12, column=1)

            def save_func():
                if not os.path.isdir("simulation_parameters"):
                    os.mkdir("simulation_parameters")
                file = open("simulation_parameters/" + filename.get() + ".txt", "w+")
                # roi_list, fiber_list, bval, nshells
                # write xyz
                file.write(xsize.get() + "\n")
                file.write(ysize.get() + "\n")
                file.write(zsize.get() + "\n")
                # write bvals
                file.write(bval.get() + "\n")
                # write nshells
                file.write(nshells.get() + "\n")
                # write te and tr
                file.write(te.get() + "\n")
                file.write(tr.get() + "\n")
                # write SNR
                file.write(snr.get() + "\n")
                # write resolution
                file.write(res.get() + "\n")
                # write roi list
                numrois = 0
                for roi in get_roi_info():
                    if roi is not None and roi != "":
                        numrois += 1
                file.write(str(numrois) + "\n")
                for roi in get_roi_info():
                    if roi is not None and roi != "":
                        for info in roi:
                            file.write(info + "\n")
                # write fiber list
                numfiber = 0
                for edge in get_fiber_info():
                    if edge[0] is not None and edge[0] != "":
                        numfiber += 1
                file.write(str(numfiber) + "\n")
                for fiber in get_fiber_info():
                    for element in fiber:
                        if element is not None and element != "":
                            file.write(element + "\n")
                    # file.write(fiber + "\n")
                file.flush()
                file.close()
                return True

            save = tk.Button(self, text="Save", command=save_func)
            save.grid(row=0, column=2)

            def load_func():
                if not os.path.isdir("simulation_parameters"):
                    messagebox.showerror('Load Error', 'There are no saved parameters, please manually enter simulation parameters.')
                    return True
                else:
                    file_name = filedialog.askopenfilename()
                    file = open(file_name, "r")
                    # set simulation parameters
                    xsize.delete(0, 'end')
                    xsize.insert(0, file.readline()[:-1])
                    ysize.delete(0, 'end')
                    ysize.insert(0, file.readline()[:-1])
                    zsize.delete(0, 'end')
                    zsize.insert(0, file.readline()[:-1])
                    bval.delete(0, 'end')
                    bval.insert(0, file.readline()[:-1])
                    nshells.delete(0, 'end')
                    nshells.insert(0, file.readline()[:-1])
                    te.delete(0, 'end')
                    te.insert(0, file.readline()[:-1])
                    tr.delete(0, 'end')
                    tr.insert(0, file.readline()[:-1])
                    snr.delete(0, 'end')
                    snr.insert(0, file.readline()[:-1])
                    res.delete(0, 'end')
                    res.insert(0, file.readline()[:-1])
                    filename.delete(0, 'end')
                    filename.insert(0, file_name.split(sep="/")[-1][:-4])
                    # set roi info
                    numrois = int(file.readline()[:-1])
                    roi_info = []
                    for i in range(numrois):
                        roi = [file.readline()[:-1]]
                        if roi[0] != "manual":
                            roi.append(file.readline()[:-1])
                            roi.append(file.readline()[:-1])
                        else:
                            roi.append(file.readline()[:-1])
                        roi_info.append(roi)
                    set_roi_info(roi_info)
                    # set fiber info
                    numfibers = int(file.readline()[:-1])
                    fiber_info = []
                    for i in range(numfibers):
                        fiber = []
                        for j in range(5):
                            fiber.append(file.readline()[:-1])
                        fiber_info.append(fiber)
                    set_fiber_info(fiber_info)
                    file.close()
                    return True

            load = tk.Button(self, text="Load", command=load_func)
            load.grid(row=1, column=2)

    class AddROI(tk.Frame):
        num_rois = 60
        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)

            button1 = tk.Button(self, text="Main Menu", command=lambda: controller.show_frame(MainMenu))
            button1.grid(row=0, column=0)

            button2 = tk.Button(self, text="Add Fibers", command=lambda: controller.show_frame(AddFiber))
            button2.grid(row=0, column=1)

            self.orig_color = button1.cget("background")

            self.roi_lbls = [ttk.Label(self, text="ROI " + str(i+1) + ":", font=LARGE_FONT) for i in range(self.num_rois)]
            self.cuboid_centers = [ttk.Entry(self, width=25) for _ in range(self.num_rois)]
            self.cuboid_sizes = [ttk.Entry(self, width=25) for _ in range(self.num_rois)]
            self.sphere_centers = [ttk.Entry(self, width=25) for _ in range(self.num_rois)]
            self.sphere_radii = [ttk.Entry(self, width=25) for _ in range(self.num_rois)]
            self.manual = [ttk.Entry(self, width=50) for _ in range(self.num_rois)]

            def set_cuboid(i):
                def func():
                    self.sphere_centers[i].grid_forget()
                    self.sphere_radii[i].grid_forget()
                    self.manual[i].grid_forget()
                    self.sphere_centers[i].delete(0, 'end')
                    self.sphere_radii[i].delete(0, 'end')
                    self.manual[i].delete(0, 'end')
                    if i < 30:
                        self.cuboid_centers[i].grid(row=i+2, column=5)
                        self.cuboid_sizes[i].grid(row=i+2, column=6)
                    elif i >= 30:
                        self.cuboid_centers[i].grid(row=i-28, column=11)
                        self.cuboid_sizes[i].grid(row=i-28, column=12)
                    self.cuboid_buttons[i].configure(bg=self.orig_color)
                    self.sphere_buttons[i].configure(bg='gray')
                    self.manual_buttons[i].configure(bg='gray')
                return func

            self.cuboid_buttons = [tk.Button(self, text="Cuboid", command=set_cuboid(i)) for i in range(self.num_rois)]

            def set_sphere(i):
                def func():
                    self.cuboid_centers[i].grid_forget()
                    self.cuboid_sizes[i].grid_forget()
                    self.manual[i].grid_forget()
                    self.cuboid_centers[i].delete(0, 'end')
                    self.cuboid_sizes[i].delete(0, 'end')
                    self.manual[i].delete(0, 'end')
                    if i < 30:
                        self.sphere_centers[i].grid(row=i+2, column=5)
                        self.sphere_radii[i].grid(row=i+2, column=6)
                    elif i >= 30:
                        self.sphere_centers[i].grid(row=i-28, column=11)
                        self.sphere_radii[i].grid(row=i-28, column=12)
                    self.cuboid_buttons[i].configure(bg='gray')
                    self.sphere_buttons[i].configure(bg=self.orig_color)
                    self.manual_buttons[i].configure(bg='gray')
                return func

            self.sphere_buttons = [tk.Button(self, text="Sphere", command=set_sphere(i)) for i in range(self.num_rois)]

            def set_manual(i):
                def func():
                    self.cuboid_centers[i].grid_forget()
                    self.cuboid_sizes[i].grid_forget()
                    self.sphere_centers[i].grid_forget()
                    self.sphere_radii[i].grid_forget()
                    self.cuboid_centers[i].delete(0, 'end')
                    self.cuboid_sizes[i].delete(0, 'end')
                    self.sphere_centers[i].delete(0, 'end')
                    self.sphere_radii[i].delete(0, 'end')
                    if i < 30:
                        self.manual[i].grid(row=i+2, column=5, columnspan=2)
                    elif i >= 30:
                        self.manual[i].grid(row=i-28, column=11, columnspan=2)
                    self.cuboid_buttons[i].configure(bg='gray')
                    self.sphere_buttons[i].configure(bg='gray')
                    self.manual_buttons[i].configure(bg=self.orig_color)
                return func

            self.manual_buttons = [tk.Button(self, text="Manual", command=set_manual(i)) for i in range(self.num_rois)]

            # grid the first roi and label
            self.roi_lbls[0].grid(row=2, column=1)
            self.cuboid_buttons[0].grid(row=2, column=2)
            self.cuboid_buttons[0].configure(bg=self.orig_color)
            self.sphere_buttons[0].grid(row=2, column=3)
            self.sphere_buttons[0].configure(bg='gray')
            self.manual_buttons[0].grid(row=2, column=4)
            self.manual_buttons[0].configure(bg='gray')
            self.cuboid_centers[0].grid(row=2, column=5)
            self.cuboid_sizes[0].grid(row=2, column=6)

            def add_roi():
                for i in range(self.num_rois):
                    if self.cuboid_buttons[i].grid_info() == {} and i < 30:
                        self.roi_lbls[i].grid(row=i+2, column=1)
                        self.cuboid_buttons[i].configure(bg=self.orig_color)
                        self.cuboid_buttons[i].grid(row=i+2, column=2)
                        self.sphere_buttons[i].configure(bg='gray')
                        self.sphere_buttons[i].grid(row=i+2, column=3)
                        self.manual_buttons[i].configure(bg='gray')
                        self.manual_buttons[i].grid(row=i+2, column=4)
                        self.cuboid_centers[i].grid(row=i+2, column=5)
                        self.cuboid_sizes[i].grid(row=i+2, column=6)
                        break
                    elif self.cuboid_buttons[i].grid_info() == {} and i >= 30:
                        self.roi_lbls[i].grid(row=i-28, column=7)
                        self.cuboid_buttons[i].configure(bg=self.orig_color)
                        self.cuboid_buttons[i].grid(row=i-28, column=8)
                        self.sphere_buttons[i].configure(bg='gray')
                        self.sphere_buttons[i].grid(row=i-28, column=9)
                        self.manual_buttons[i].configure(bg='gray')
                        self.manual_buttons[i].grid(row=i-28, column=10)
                        self.cuboid_centers[i].grid(row=i-28, column=11)
                        self.cuboid_sizes[i].grid(row=i-28, column=12)
                        break
                    elif i == self.num_rois - 1:
                        messagebox.showerror('Add ROI Error', 'The maximum number of ROIs have been added')

            button3 = tk.Button(self, text="Add ROI", command=add_roi)
            button3.grid(row=1, column=0)

            def remove_roi():
                for i in reversed(range(self.num_rois)):
                    if not (self.cuboid_buttons[i].grid_info() == {}) and i > 0:
                        self.roi_lbls[i].grid_forget()
                        self.cuboid_buttons[i].grid_forget()
                        self.cuboid_centers[i].grid_forget()
                        self.cuboid_centers[i].delete(0, 'end')
                        self.cuboid_sizes[i].grid_forget()
                        self.cuboid_sizes[i].delete(0, 'end')
                        self.sphere_buttons[i].grid_forget()
                        self.sphere_centers[i].grid_forget()
                        self.sphere_centers[i].delete(0, 'end')
                        self.sphere_radii[i].grid_forget()
                        self.sphere_radii[i].delete(0, 'end')
                        self.manual_buttons[i].grid_forget()
                        self.manual[i].grid_forget()
                        self.manual[i].delete(0, 'end')
                        break
                    elif i == 0:
                        self.cuboid_centers[i].delete(0, 'end')
                        self.cuboid_sizes[i].delete(0, 'end')
                        self.sphere_centers[i].delete(0, 'end')
                        self.sphere_radii[i].delete(0, 'end')
                        self.manual[i].delete(0, 'end')

            button4 = tk.Button(self, text="Remove ROI", command=remove_roi)
            button4.grid(row=1, column=1)

            def clear_all():
                for i in reversed(range(self.num_rois)):
                    if i > 0:
                        self.roi_lbls[i].grid_forget()
                        self.cuboid_buttons[i].grid_forget()
                        self.cuboid_centers[i].grid_forget()
                        self.cuboid_centers[i].delete(0, 'end')
                        self.cuboid_sizes[i].grid_forget()
                        self.cuboid_sizes[i].delete(0, 'end')
                        self.sphere_buttons[i].grid_forget()
                        self.sphere_centers[i].grid_forget()
                        self.sphere_centers[i].delete(0, 'end')
                        self.sphere_radii[i].grid_forget()
                        self.sphere_radii[i].delete(0, 'end')
                        self.manual_buttons[i].grid_forget()
                        self.manual[i].grid_forget()
                        self.manual[i].delete(0, 'end')
                    elif i == 0:
                        self.cuboid_centers[i].delete(0, 'end')
                        self.cuboid_sizes[i].delete(0, 'end')
                        self.sphere_centers[i].delete(0, 'end')
                        self.sphere_radii[i].delete(0, 'end')
                        self.manual[i].delete(0, 'end')

            button5 = tk.Button(self, text="Clear ROIs", command=clear_all)
            button5.grid(row=2, column=0)

            blank = ttk.Label(self, text=" ", font=LARGE_FONT)
            blank.grid(row=32, column=0)

            info1 = ttk.Label(self, width=100,
                              text="Cuboid: In the first box enter a center point (a, b, c) and in the second box enter integers (i, j, k) for the x, y, and z size of the cuboid.",
                              font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info1.grid(row=33, column=0, columnspan=12)

            info2 = ttk.Label(self, width=100,
                              text="Sphere: In the first box enter a center point (a, b, c) and in the second box enter a value for the radius.",
                              font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info2.grid(row=34, column=0, columnspan=12)

            info3 = ttk.Label(self, width=100, text="Manual: Enter a list of tuples (a, b, c), ..., (x, y, z) for each ROI.", font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info3.grid(row=35, column=0, columnspan=12)

        def get_roi_list(self):
            roi_list = np.empty(self.num_rois, dtype='object')
            for i in range(self.num_rois):
                if self.cuboid_centers[i].get() is not None and self.cuboid_centers[i].get() != "":
                    roi_list[i] = ["cuboid", self.cuboid_centers[i].get(), self.cuboid_sizes[i].get()]
                elif self.sphere_centers[i].get() is not None and self.sphere_centers[i].get() != "":
                    roi_list[i] = ["sphere", self.sphere_centers[i].get(), self.sphere_radii[i].get()]
                elif self.manual[i].get() is not None and self.manual[i].get() != "":
                    roi_list[i] = ["manual", self.manual[i].get()]
                else:
                    roi_list[i] = None
            return roi_list

        def set_roi_list(self, roi_info):
            for i in reversed(range(self.num_rois)):
                if i > 0:
                    self.roi_lbls[i].grid_forget()
                    self.cuboid_buttons[i].grid_forget()
                    self.cuboid_centers[i].grid_forget()
                    self.cuboid_centers[i].delete(0, 'end')
                    self.cuboid_sizes[i].grid_forget()
                    self.cuboid_sizes[i].delete(0, 'end')
                    self.sphere_buttons[i].grid_forget()
                    self.sphere_centers[i].grid_forget()
                    self.sphere_centers[i].delete(0, 'end')
                    self.sphere_radii[i].grid_forget()
                    self.sphere_radii[i].delete(0, 'end')
                    self.manual_buttons[i].grid_forget()
                    self.manual[i].grid_forget()
                    self.manual[i].delete(0, 'end')
                elif i == 0:
                    self.cuboid_centers[i].delete(0, 'end')
                    self.cuboid_sizes[i].delete(0, 'end')
                    self.sphere_centers[i].delete(0, 'end')
                    self.sphere_radii[i].delete(0, 'end')
                    self.manual[i].delete(0, 'end')
            for i in range(len(roi_info)):
                if i < 30:
                    self.roi_lbls[i].grid(row=i + 2, column=1)
                    self.cuboid_buttons[i].grid(row=i + 2, column=2)
                    self.sphere_buttons[i].grid(row=i + 2, column=3)
                    self.manual_buttons[i].grid(row=i + 2, column=4)
                    if roi_info[i][0] == "cuboid":
                        self.cuboid_centers[i].grid(row=i + 2, column=5)
                        self.cuboid_centers[i].insert(0, roi_info[i][1])
                        self.cuboid_sizes[i].grid(row=i + 2, column=6)
                        self.cuboid_sizes[i].insert(0, roi_info[i][2])
                        self.cuboid_buttons[i].configure(bg=self.orig_color)
                        self.sphere_buttons[i].configure(bg='gray')
                        self.manual_buttons[i].configure(bg='gray')
                    elif roi_info[i][0] == "sphere":
                        self.sphere_centers[i].grid(row=i + 2, column=5)
                        self.sphere_centers[i].insert(0, roi_info[i][1])
                        self.sphere_radii[i].grid(row=i + 2, column=6)
                        self.sphere_radii[i].insert(0, roi_info[i][2])
                        self.sphere_buttons[i].configure(bg=self.orig_color)
                        self.manual_buttons[i].configure(bg='gray')
                        self.cuboid_buttons[i].configure(bg='gray')
                    elif roi_info[i][0] == "manual":
                        self.manual[i].grid(row=i + 2, column=5)
                        self.manual[i].insert(0, roi_info[i][1])
                        self.manual_buttons[i].configure(bg=self.orig_color)
                        self.cuboid_buttons[i].configure(bg='gray')
                        self.sphere_buttons[i].configure(bg='gray')
                elif i >= 30:
                    self.roi_lbls[i].grid(row=i - 28, column=7)
                    self.cuboid_buttons[i].grid(row=i - 28, column=8)
                    self.sphere_buttons[i].grid(row=i - 28, column=9)
                    self.manual_buttons[i].grid(row=i - 28, column=10)
                    if roi_info[i][0] == "cuboid":
                        self.cuboid_centers[i].grid(row=i - 28, column=11)
                        self.cuboid_centers[i].insert(0, roi_info[i][1])
                        self.cuboid_sizes[i].grid(row=i - 28, column=12)
                        self.cuboid_sizes[i].insert(0, roi_info[i][2])
                        self.cuboid_buttons[i].configure(bg=self.orig_color)
                        self.sphere_buttons[i].configure(bg='gray')
                        self.manual_buttons[i].configure(bg='gray')
                    elif roi_info[i][0] == "sphere":
                        self.sphere_centers[i].grid(row=i - 28, column=11)
                        self.sphere_centers[i].insert(0, roi_info[i][1])
                        self.sphere_radii[i].grid(row=i - 28, column=12)
                        self.sphere_radii[i].insert(0, roi_info[i][2])
                        self.sphere_buttons[i].configure(bg=self.orig_color)
                        self.manual_buttons[i].configure(bg='gray')
                        self.cuboid_buttons[i].configure(bg='gray')
                    elif roi_info[i][0] == "manual":
                        self.manual[i].grid(row=i - 28, column=11)
                        self.manual[i].insert(0, roi_info[i][1])
                        self.manual_buttons[i].configure(bg=self.orig_color)
                        self.cuboid_buttons[i].configure(bg='gray')
                        self.sphere_buttons[i].configure(bg='gray')
            return

    class AddFiber(tk.Frame):
        num_fibers = 30

        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)

            button1 = tk.Button(self, text="Main Menu", command=lambda: controller.show_frame(MainMenu))
            button1.grid(row=0, column=0)

            button2 = tk.Button(self, text="Add ROIs", command=lambda: controller.show_frame(AddROI))
            button2.grid(row=0, column=1)

            self.fiber_lbls = [ttk.Label(self, text="Fiber " + str(i + 1) + ":", font=LARGE_FONT) for i in range(self.num_fibers)]
            self.fibers = [ttk.Entry(self, width=10) for _ in range(self.num_fibers)]
            self.fiber_dirs = [ttk.Entry(self, width=20) for _ in range(self.num_fibers)]
            self.fiber_radii = [ttk.Entry(self, width=10) for _ in range(self.num_fibers)]
            self.fiber_voxels = [ttk.Entry(self, width=20) for _ in range(self.num_fibers)]
            self.fit_type = [ttk.Entry(self, width=10) for _ in range(self.num_fibers)]

            self.fiber_lbls[0].grid(row=2, column=1)
            self.fibers[0].grid(row=2, column=2)
            self.fiber_dirs[0].grid(row=2, column=3)
            self.fiber_radii[0].grid(row=2, column=4)
            self.fiber_voxels[0].grid(row=2, column=5)
            self.fit_type[0].grid(row=2, column=6)

            def add_fiber():
                for i in range(len(self.fibers)):
                    if self.fibers[i].grid_info() == {}:
                        self.fiber_lbls[i].grid(row=i+2, column=1)
                        self.fibers[i].grid(row=i+2, column=2)
                        self.fiber_dirs[i].grid(row=i+2, column=3)
                        self.fiber_radii[i].grid(row=i+2, column=4)
                        self.fiber_voxels[i].grid(row=i+2, column=5)
                        self.fit_type[i].grid(row=i+2, column=6)
                        break
                    elif i == len(self.fibers) - 1:
                        tk.messagebox.showerror('Add Fiber Error', 'The maximum number of fibers have been added')

            button3 = tk.Button(self, text="Add Fiber", command=add_fiber)
            button3.grid(row=1, column=0)

            def remove_fiber():
                for i in reversed(range(len(self.fibers))):
                    if not (self.fibers[i].grid_info() == {}) and i > 0:
                        self.fiber_lbls[i].grid_forget()
                        self.fibers[i].grid_forget()
                        self.fibers[i].delete(0, 'end')
                        self.fiber_dirs[i].grid_forget()
                        self.fiber_dirs[i].delete(0, 'end')
                        self.fiber_radii[i].grid_forget()
                        self.fiber_radii[i].delete(0, 'end')
                        self.fiber_voxels[i].grid_forget()
                        self.fiber_voxels[i].delete(0, 'end')
                        self.fit_type[i].grid_forget()
                        self.fit_type[i].delete(0, 'end')
                        break
                    elif i == 0:
                        self.fibers[i].delete(0, 'end')
                        self.fiber_dirs[i].delete(0, 'end')
                        self.fiber_radii[i].delete(0, 'end')
                        self.fiber_voxels[i].delete(0, 'end')
                        self.fit_type[i].delete(0, 'end')

            button4 = tk.Button(self, text="Remove Fiber", command=remove_fiber)
            button4.grid(row=1, column=1)

            def clear_all():
                for i in reversed(range(len(self.fibers))):
                    if i > 0:
                        self.fiber_lbls[i].grid_forget()
                        self.fibers[i].grid_forget()
                        self.fibers[i].delete(0, 'end')
                        self.fiber_dirs[i].grid_forget()
                        self.fiber_dirs[i].delete(0, 'end')
                        self.fiber_radii[i].grid_forget()
                        self.fiber_radii[i].delete(0, 'end')
                        self.fiber_voxels[i].grid_forget()
                        self.fiber_voxels[i].delete(0, 'end')
                        self.fit_type[i].grid_forget()
                        self.fit_type[i].delete(0, 'end')
                    elif i == 0:
                        self.fibers[i].delete(0, 'end')
                        self.fiber_dirs[i].delete(0, 'end')
                        self.fiber_radii[i].delete(0, 'end')
                        self.fiber_voxels[i].delete(0, 'end')
                        self.fit_type[i].delete(0, 'end')

            button5 = tk.Button(self, text="Clear Fibers", command=clear_all)
            button5.grid(row=2, column=0)

            blank = ttk.Label(self, text=" ", font=LARGE_FONT)
            blank.grid(row=62, column=0)
            info = ttk.Label(self, width=110, text="In the first entry box enter two ROI numbers "
                                                   "(e.g. 1, 2 for ROI1 and ROI2) with a comma and space after the "
                                                   "first ROI.", font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info.grid(row=63, column=1, columnspan=10)
            info1 = ttk.Label(self, width=110, text="Enter a tuple x, y, z with spaces in the second entry box "
                                                    "corresponding to the eigenvalues of the diffusion tensor "
                                                    "(e.g. .0015, .00025, .00025).", font=LARGE_FONT, anchor='w',
                                                    justify=tk.LEFT)
            info1.grid(row=64, column=1, columnspan=10)
            info2 = ttk.Label(self, width=110, text=" In the third box enter enter a float for the radius of the "
                                                    "fiber.", font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info2.grid(row=65, column=1, columnspan=10)
            info3 = ttk.Label(self, width=110, text="In the fourth box enter two tuples (e.g. (a, b, c), (d, e, f)) "
                                                    "from the two respective ROIs and an optional tuple to define a "
                                                    "midpoint to define the fiber between the two ROIs.",
                              font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info3.grid(row=66, column=1, columnspan=10)
            info4 = ttk.Label(self, width=110, text="In the last box enter either linear or circular to define the "
                                                    "geometry of the fiber.", font=LARGE_FONT, anchor='w',
                              justify=tk.LEFT)
            info4.grid(row=67, column=1, columnspan=10)

        def get_fiber_list(self):
            fiber_list = np.empty((self.num_fibers, 5), dtype='object')
            for i in range(len(self.fibers)):
                fiber_list[i][0] = self.fibers[i].get()
                fiber_list[i][1] = self.fiber_dirs[i].get()
                fiber_list[i][2] = self.fiber_radii[i].get()
                fiber_list[i][3] = self.fiber_voxels[i].get()
                fiber_list[i][4] = self.fit_type[i].get()
            return fiber_list

        def set_fiber_list(self, fiber_info):
            for i in reversed(range(len(self.fibers))):
                if not (self.fibers[i].grid_info() == {}) and i > 0:
                    self.fiber_lbls[i].grid_forget()
                    self.fibers[i].grid_forget()
                    self.fibers[i].delete(0, 'end')
                    self.fiber_dirs[i].grid_forget()
                    self.fiber_dirs[i].delete(0, 'end')
                    self.fiber_radii[i].grid_forget()
                    self.fiber_radii[i].delete(0, 'end')
                    self.fiber_voxels[i].grid_forget()
                    self.fiber_voxels[i].delete(0, 'end')
                    self.fit_type[i].grid_forget()
                    self.fit_type[i].delete(0, 'end')
                elif i == 0:
                    self.fibers[i].delete(0, 'end')
                    self.fiber_dirs[i].delete(0, 'end')
                    self.fiber_radii[i].delete(0, 'end')
                    self.fiber_voxels[i].delete(0, 'end')
                    self.fit_type[i].delete(0, 'end')
            for i in range(len(fiber_info)):
                self.fibers[i].insert(0, fiber_info[i][0])
                self.fiber_dirs[i].insert(0, fiber_info[i][1])
                self.fiber_radii[i].insert(0, fiber_info[i][2])
                self.fiber_voxels[i].insert(0, fiber_info[i][3])
                self.fit_type[i].insert(0, fiber_info[i][4])
                self.fiber_lbls[i].grid(row=i+2, column=1)
                self.fibers[i].grid(row=i+2, column=2)
                self.fiber_dirs[i].grid(row=i+2, column=3)
                self.fiber_radii[i].grid(row=i+2, column=4)
                self.fiber_voxels[i].grid(row=i+2, column=5)
                self.fit_type[i].grid(row=i+2, column=6)

    app = DWIS()
    app.mainloop()