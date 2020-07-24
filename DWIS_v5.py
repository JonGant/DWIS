# Description: This program creates a simulated diffusion weighted image for use in  the creation of weighted edge
# adjacency matrix
# Date: April 6th 2020
# Author: Jonathan Gant
# References: Colon-Perez LM, Spindler C, Goicochea S, Triplett W, Parekh M, Montie E, et al. (2015) Dimensionless,
#             Scale Invariant, Edge Weight Metric for the Study of Complex Structural Networks.
#             PLoS ONE 10(7): e0131493. https://doi.org/10.1371/journal.pone.0131493

# import statements
import numpy as np
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
    weights = np.ones((totalpoints, totalpoints))
    bvecs = multishell.optimize(len(dirpshell), dirpshell, weights)

    return [bvecs, bvals]


def mask_function(nodes, i, j, k):
    # check if there is a fiber or not, returns true if there is a fiber
    for a in range(len(nodes)):
        for b in range(len(nodes[a])):
            if nodes[a][b][0] == i and nodes[a][b][1] == j and nodes[a][b][2] == k:
                return a + 1
    return 0

def signal_function(S0, bvecs, bvals, dt_data, nedges, i, j, k, m):
    # check if there is a fiber or not, returns true if there is a fiber
    fiber1 = not np.equal(dt_data[i][j][k][0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]).all()
    fiber2 = not np.equal(dt_data[i][j][k][1], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]).all()
    fiber3 = not np.equal(dt_data[i][j][k][2], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]).all()
    isotropic_dt = np.array([[.0017, 0, 0], [0, .0017, 0], [0, 0, .0017]])
    isotropic_term = np.matmul(bvecs[m, :], np.matmul(isotropic_dt, np.transpose(bvecs[m, :])))
    VF_iso = 0
    if nedges is 1:
        # no fibers
        if not (fiber1 or fiber2 or fiber3):
            return S0 * VF_iso * (np.exp(-bvals[m] * isotropic_term))
        # one fiber
        elif fiber1:
            term = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][0], np.transpose(bvecs[m, :])))
            return S0 * (0.5 * VF_iso * np.exp(-bvals[m] * isotropic_term) + 0.5 * np.exp(-bvals[m] * term))
        elif fiber2:
            term = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][1], np.transpose(bvecs[m, :])))
            return S0 * (0.5 * VF_iso * np.exp(-bvals[m] * isotropic_term) + 0.5 * np.exp(-bvals[m] * term))
        elif fiber3:
            term = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][2], np.transpose(bvecs[m, :])))
            return S0 * (0.5 * VF_iso * np.exp(-bvals[m] * isotropic_term) + 0.5 * np.exp(-bvals[m] * term))
    elif nedges is 2:
        # two fibers
        if fiber1 and fiber2:
            term1 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][0], np.transpose(bvecs[m, :])))
            term2 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][1], np.transpose(bvecs[m, :])))
            return S0 * ((1 / 3) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 3) * np.exp(-bvals[m] * term1)
                                     + (1 / 3) * np.exp(-bvals[m] * term2))
        elif fiber1 and fiber3:
            term1 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][0], np.transpose(bvecs[m, :])))
            term3 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][2], np.transpose(bvecs[m, :])))
            return S0 * ((1 / 3) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 3) * np.exp(-bvals[m] * term1)
                                     + (1 / 3) * np.exp(-bvals[m] * term3))
        elif fiber2 and fiber3:
            term2 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][1], np.transpose(bvecs[m, :])))
            term3 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][2], np.transpose(bvecs[m, :])))
            return S0 * ((1 / 3) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 3) * np.exp(-bvals[m] * term2)
                                     + (1 / 3) * np.exp(-bvals[m] * term3))
        # one fiber
        elif fiber1:
            term1 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][0], np.transpose(bvecs[m, :])))
            return S0 * ((2 / 3) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 3) * np.exp(-bvals[m] * term1))
        elif fiber2:
            term2 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][1], np.transpose(bvecs[m, :])))
            return S0 * ((2 / 3) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 3) * np.exp(-bvals[m] * term2))
        elif fiber3:
            term3 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][2], np.transpose(bvecs[m, :])))
            return S0 * ((2 / 3) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 3) * np.exp(-bvals[m] * term3))
        # no fibers
        else:
            return S0 * VF_iso * (np.exp(-bvals[m] * isotropic_term))
    elif nedges is 3:
        # three fibers
        if fiber1 and fiber2 and fiber3:
            term1 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][0], np.transpose(bvecs[m, :])))
            term2 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][1], np.transpose(bvecs[m, :])))
            term3 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][2], np.transpose(bvecs[m, :])))
            return S0 * ((1 / 4) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 4) * np.exp(-bvals[m] * term1)
                                     + (1 / 4) * np.exp(-bvals[m] * term2) + (1 / 4) * np.exp(-bvals[m] * term3))
        # two fibers
        elif fiber1 and fiber2:
            term1 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][0], np.transpose(bvecs[m, :])))
            term2 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][1], np.transpose(bvecs[m, :])))
            return S0 * ((1 / 2) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 4) * np.exp(
                -bvals[m] * term1) + (1 / 4) * np.exp(-bvals[m] * term2))
        elif fiber1 and fiber3:
            term1 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][0], np.transpose(bvecs[m, :])))
            term3 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][2], np.transpose(bvecs[m, :])))
            return S0 * ((1 / 2) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 4) * np.exp(
                -bvals[m] * term1) + (1 / 4) * np.exp(-bvals[m] * term3))
        elif fiber2 and fiber3:
            term2 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][1], np.transpose(bvecs[m, :])))
            term3 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][2], np.transpose(bvecs[m, :])))
            return S0 * ((1 / 2) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 4) * np.exp(
                -bvals[m] * term2) + (1 / 4) * np.exp(-bvals[m] * term3))
        # one fiber
        elif fiber1:
            term1 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][0], np.transpose(bvecs[m, :])))
            return S0 * ((3 / 4) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 4) * np.exp(
                -bvals[m] * term1))
        elif fiber2:
            term2 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][1], np.transpose(bvecs[m, :])))
            return S0 * ((3 / 4) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 4) * np.exp(
                -bvals[m] * term2))
        elif fiber3:
            term3 = np.matmul(bvecs[m, :], np.matmul(dt_data[i][j][k][2], np.transpose(bvecs[m, :])))
            return S0 * ((3 / 4) * VF_iso * np.exp(-bvals[m] * isotropic_term) + (1 / 4) * np.exp(
                -bvals[m] * term3))
        # no fibers
        else:
            return S0 * VF_iso * (np.exp(-bvals[m] * isotropic_term))


# main part of the program which simulates the diffusion tensor data
def simulate_dwi_calc(xsize, ysize, zsize, bvalue, dirpershell, dt_data, nedges, filename, nodes):
    # create b vectors and values
    bvecval = create_bvecs_and_bvals(bvalue, dirpershell)
    bvecs = bvecval[0]
    bvals = bvecval[1]
    # Create a data array, set values of parameters
    dirs = np.size(bvecs, 0)
    data = np.zeros((xsize, ysize, zsize, dirs), float)
    mask = np.zeros((xsize, ysize, zsize, dirs), int)
    S0 = 1.0
    for k in range(zsize):
        for j in range(ysize):
            for i in range(xsize):
                for m in range(dirs):
                    data[i, j, k, m] = signal_function(S0, bvecs, bvals, dt_data, nedges, i, j, k, m)
                    mask[i, j, k, m] = mask_function(nodes, i, j, k)
    # set the affine
    affine = np.diag([2, 2, 2, 1])
    # make an image file from the numpy array
    array_img = nib.Nifti1Image(data, affine)
    # make a mask image
    mask_img = nib.Nifti1Image(mask, affine)
    # create result directory if it does not already exist
    if not os.path.isdir("nifti_images"):
        os.mkdir("nifti_images")
    # save the simulated data as a nifti file
    nib.save(array_img, "nifti_images/" + str(filename) + ".nii.gz")
    # save the mask
    nib.save(mask_img, "nifti_images/" + str(filename) + "_mask.nii.gz")
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


# function rotates diffusion data in a voxel to be in the direction of the two angles phi and theta (from spherical coordinates)
def rotate(voxel_data, phi, theta):
    rot_y = np.array([[np.cos((np.pi/2)-theta), 0, -np.sin((np.pi/2)-theta)], [0, 1, 0], [np.sin((np.pi/2)-theta), 0 , np.cos((np.pi/2)-theta)]])
    rot_z = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    rotated_voxel_data = np.matmul(rot_z, np.matmul(np.matmul(np.matmul(rot_y, voxel_data), np.transpose(rot_y)), np.transpose(rot_z)))
    return rotated_voxel_data


# this function takes in a list of nodes and a list of parameters determining the width and curvature of each edge
# and creates the diffusion tensor information associated with the edges
def create_dt_data(xsize, ysize, zsize, nodes, edges, edge_voxels, eigenvalues, radii):
    failed = False
    # # create blank data array
    dt_data = np.zeros((xsize, ysize, zsize, 3, 3, 3), float)

    # define a list of vectors which define the direction of each edge
    directions = []
    displacements = []
    for i in range(len(edges)):
        node1 = edges[i][0]
        node2 = edges[i][1]
        if node1 != 0 or node2 != 0:
            displacement = np.array([edge_voxels[i][1][0] - edge_voxels[i][0][0], edge_voxels[i][1][1] - edge_voxels[i][0][1], edge_voxels[i][1][2] - edge_voxels[i][0][2]])
            displacements.append(displacement)
            directions.append(displacement/np.linalg.norm(displacement))
    num_edges = len(displacements)
    # defines the characteristic function
    # iterate numerically over the normalized slope vectors on four vertices of voxel to find voxels that are "hit" by the slope vectors
    hit_voxels = []
    for i in range(len(displacements)):
        voxels = []
        # first vertex
        startpoint1 = np.array([edge_voxels[i][0][0], edge_voxels[i][0][1], edge_voxels[i][0][2]])
        currentstartpoint = startpoint1
        while np.linalg.norm(currentstartpoint) < np.linalg.norm(startpoint1) + np.linalg.norm(displacements[i]):
            currentstartpoint = currentstartpoint + .25*directions[i]
            # check if hit new voxel
            found_point = False
            if voxels != []:
                for j in range(len(voxels)):
                    if np.equal(np.floor(currentstartpoint), voxels[j]).all():
                        found_point = True
            if not found_point:
                if 0 <= currentstartpoint[0] < xsize and 0 <= currentstartpoint[1] < ysize and 0 <= currentstartpoint[2] < zsize:
                    voxels.append(np.floor(currentstartpoint))
        # second vertex
        startpoint2 = np.array([edge_voxels[i][0][0] + radii[i], edge_voxels[i][0][1], edge_voxels[i][0][2]])
        currentstartpoint = startpoint2
        while np.linalg.norm(currentstartpoint) < np.linalg.norm(startpoint2) + np.linalg.norm(displacements[i]):
            currentstartpoint = currentstartpoint + .25*directions[i]
            # check if hit new voxel
            found_point = False
            if voxels != []:
                for j in range(len(voxels)):
                    if np.equal(np.floor(currentstartpoint), voxels[j]).all():
                        found_point = True
            if not found_point:
                if 0 <= currentstartpoint[0] < xsize and 0 <= currentstartpoint[1] < ysize and 0 <= currentstartpoint[2] < zsize:
                    voxels.append(np.floor(currentstartpoint))
        # third vertex
        startpoint3 = np.array([edge_voxels[i][0][0], edge_voxels[i][0][1] + radii[i], edge_voxels[i][0][2]])
        currentstartpoint = startpoint3
        while np.linalg.norm(currentstartpoint) < np.linalg.norm(startpoint3) + np.linalg.norm(displacements[i]):
            currentstartpoint = currentstartpoint + .25*directions[i]
            # check if hit new voxel
            found_point = False
            if voxels != []:
                for j in range(len(voxels)):
                    if np.equal(np.floor(currentstartpoint), voxels[j]).all():
                        found_point = True
            if not found_point:
                if 0 <= currentstartpoint[0] < xsize and 0 <= currentstartpoint[1] < ysize and 0 <= currentstartpoint[2] < zsize:
                    voxels.append(np.floor(currentstartpoint))
        # fourth vertex
        startpoint4 = np.array([edge_voxels[i][0][0], edge_voxels[i][0][1], edge_voxels[i][0][2] + radii[i]])
        currentstartpoint = startpoint4
        while np.linalg.norm(currentstartpoint) < np.linalg.norm(startpoint4) + np.linalg.norm(displacements[i]):
            currentstartpoint = currentstartpoint + .25*directions[i]
            # check if hit new voxel
            found_point = False
            if voxels != []:
                for j in range(len(voxels)):
                    if np.equal(np.floor(currentstartpoint), voxels[j]).all():
                        found_point = True
            if not found_point:
                if 0 <= currentstartpoint[0] < xsize and 0 <= currentstartpoint[1] < ysize and 0 <= currentstartpoint[2] < zsize:
                    voxels.append(np.floor(currentstartpoint))
        # add all the hit voxels to the overall list of hit voxels which is per edge
        hit_voxels.append(voxels)
    # find the angles of each direction vector
    angles = np.empty((len(directions), 2))
    # check if direction is positive or negative
    for i in range(len(directions)):
        if directions[i][0] == 0:
            angles[i][0] = np.pi/2
        else:
            angles[i][0] = np.arctan(directions[i][1]/directions[i][0])
        if directions[i][2] == 0:
            angles[i][1] = np.pi/2
        else:
            angles[i][1] = np.arctan((np.sqrt(np.square(directions[i][0])+np.square(directions[i][1])))/np.abs(directions[i][2]))
    # take the voxels that were hit and construct the data associated with those voxels
    for i in range(len(hit_voxels)):
        # sample diffusion profile (to be rotated)
        diff = np.array([[eigenvalues[i][0], 0, 0], [0, eigenvalues[i][1], 0], [0, 0, eigenvalues[i][2]]])
        mean_diff = (eigenvalues[i][0]+eigenvalues[i][1]+eigenvalues[i][2])/3
        print("Mean diffusivity of fiber " + str(i + 1) + ": ")
        print(mean_diff)
        fa = (np.sqrt(3) * np.sqrt(np.square(eigenvalues[i][0] - mean_diff) + np.square(eigenvalues[i][1] - mean_diff) + np.square(eigenvalues[i][2] - mean_diff)))/(np.sqrt(2) * np.sqrt(np.square(eigenvalues[i][0]) + np.square(eigenvalues[i][1]) + np.square(eigenvalues[i][2])))
        print("Fractional anistropy of fiber " + str(i + 1) + ": ")
        print(fa)

        # rotate data
        rotated = rotate(diff, angles[i][0], angles[i][1])
        voxel_list = hit_voxels[i]
        for voxel in voxel_list:
            x = int(voxel[0])
            y = int(voxel[1])
            z = int(voxel[2])
            # write diffusion data to array
            if np.equal(dt_data[x][y][z][0], [[0,0,0],[0,0,0],[0,0,0]]).all():
                dt_data[x][y][z][0] = rotated
            elif np.equal(dt_data[x][y][z][1], [[0,0,0],[0,0,0],[0,0,0]]).all():
                dt_data[x][y][z][1] = rotated
            elif np.equal(dt_data[x][y][z][2], [[0,0,0],[0,0,0],[0,0,0]]).all():
                dt_data[x][y][z][2] = rotated
            else:
                failed = True
                break
                # throw error
        # fill in the nodes with the same tensor info for the given edge
        for node in edges[i]:
            for voxel in nodes[node-1]:
                dt_data[voxel[0]][voxel[1]][voxel[2]][i] = rotated
    return [dt_data, failed, num_edges]


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

            for F in (MainMenu, AddNode, AddEdge):

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

            button1 = ttk.Button(self, text="Add Nodes", command=lambda: controller.show_frame(AddNode))
            button1.grid(row=1, column=0, sticky='E')

            button2 = ttk.Button(self, text="Add Edges", command=lambda: controller.show_frame(AddEdge))
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
            bval_lbl = ttk.Label(self, width=25, text="List of b-values: ", font=("Arial", 12), anchor='e')
            bval_lbl.grid(row=5, column=0)
            bval = ttk.Entry(self, width=10)
            bval.grid(row=5, column=1)

            # enter as comma separated list
            nshells_lbl = ttk.Label(self, width=25, text="Number of directions per shell: ", font=("Arial", 12), anchor='e')
            nshells_lbl.grid(row=6, column=0)
            nshells = ttk.Entry(self, width=10)
            nshells.grid(row=6, column=1)

            filename_lbl = ttk.Label(self, width=25, text="File name: ", font=("Arial", 12), anchor='e')
            filename_lbl.grid(row=7, column=0)
            filename = ttk.Entry(self, width=10)
            filename.grid(row=7, column=1)

            # get node info
            def get_node_info():
                AddNodeFrame = controller.get_frame(AddNode)
                return AddNodeFrame.get_node_list()

            def set_node_info(node_list):
                AddNodeFrame = controller.get_frame(AddNode)
                return AddNodeFrame.set_node_list(node_list)

            # get edge info
            def get_edge_info():
                AddEdgeFrame = controller.get_frame(AddEdge)
                return AddEdgeFrame.get_edge_list()

            def set_edge_info(edge_list):
                AddEdgeFrame = controller.get_frame(AddEdge)
                return AddEdgeFrame.set_edge_list(edge_list)
            # start simulation command
            def start_sim():
                save_func()
                node_list = get_node_info()
                edge_list = get_edge_info()
                # format node list
                node_list_formatted = []
                for i in range(len(node_list)):
                    if node_list[i] is not None and node_list[i] is not "":
                        split_node = node_list[i].split(sep='), (')
                        coord_list =[]
                        for voxel in split_node:
                            if '(' in voxel:
                                voxel = voxel[1:]
                            if ')' in voxel:
                                voxel = voxel[:len(voxel)-1]
                            coords = voxel.split(sep=', ')
                            coord_list.append([int(coords[0]), int(coords[1]), int(coords[2])])
                        node_list_formatted.append(coord_list)
                # format edge lists
                edge_list_formatted = []
                eigenvalues = np.empty((len(edge_list), 3))
                edge_voxels = []
                radii = []
                for i in range(len(edge_list)):
                    if edge_list[i][0] is not None and edge_list[i][0] is not "":
                        split_edge = edge_list[i][0].split(sep=', ')
                        edge_list_formatted.append([int(split_edge[0]), int(split_edge[1])])
                    if edge_list[i][1] is not None and edge_list[i][1] is not "":
                        split_eigen = edge_list[i][1].split(sep=', ')
                        eigenvalues[i] = np.array([float(split_eigen[0]), float(split_eigen[1]), float(split_eigen[2])])
                    if edge_list[i][2] is not None and edge_list[i][2] is not "":
                        edge_voxel = []
                        split_node = edge_list[i][2].split(sep='), (')
                        for j in range(len(split_node)):
                            if '(' in split_node[j]:
                                voxel = split_node[j][1:]
                            elif ')' in split_node[j]:
                                voxel = split_node[j][:len(split_node[j])-1]
                            else:
                                voxel = split_node[j]
                            coords = voxel.split(sep=', ')
                            edge_voxel.append([int(coords[0]), int(coords[1]), int(coords[2])])
                        edge_voxels.append(edge_voxel)
                    if edge_list[i][3] is not None and edge_list[i][3] is not "":
                        radius = float(edge_list[i][3])
                        radii.append(radius)
                # format bvals and directions
                bvals = []
                split_bval = bval.get().split(", ")
                for term in split_bval:
                    bvals.append(float(term))

                dir_shell = []
                split_nshells = nshells.get().split(", ")
                for term in split_nshells:
                    dir_shell.append(int(term))
                # check that edge voxels are in the node
                edge_format = False
                node_check = []
                for i in range(len(edge_voxels)):
                    for j in range(len(edge_voxels[i])):
                        voxels = node_list_formatted[edge_list_formatted[i][j]-1]
                        voxel_check = False
                        for voxel in voxels:
                            if np.equal(voxel, edge_voxels[i][j]).all():
                                voxel_check = True
                        node_check.append(voxel_check)
                edge_format = all(node_check)

                # start simulation functions
                if edge_format:
                    if xsize.get().isdigit() and ysize.get().isdigit() and zsize.get().isdigit():
                        if not (',' in bvals or ',' in dir_shell):
                            if len(bvals) == len(dir_shell):
                                # create result directory if it does not already exist
                                if not os.path.isdir("nifti_images"):
                                    os.mkdir("nifti_images")
                                if 0 < int(xsize.get()) <= 100 and 0 < int(ysize.get()) <= 100 and 0 < int(zsize.get()) <= 100:
                                    messagebox.showinfo('Simulation Notification', 'Simulation has started')
                                    # create the diffusion tensor info file
                                    dt_data = create_dt_data(int(xsize.get()), int(ysize.get()), int(zsize.get()),
                                                             node_list_formatted, edge_list_formatted, edge_voxels,
                                                             eigenvalues, radii)
                                    # Start the simulation
                                    param = True
                                    while(param):
                                        param = simulate_dwi_calc(int(xsize.get()), int(ysize.get()), int(zsize.get()),
                                                                  bvals, dir_shell, dt_data[0], dt_data[2], filename.get(), node_list_formatted)
                                    messagebox.showinfo('Simulation Notification', 'Simulation successfully finished')
                                else:
                                    messagebox.showinfo('Simulation Notification',
                                                        'Please enter values greater than 0 and less than 100 for XYZ sizes')
                            else:
                                messagebox.showinfo('Simulation Notification',
                                                    'Please ensure that the list of b-values and directions per shell are the same length')
                        else:
                            messagebox.showinfo('Simulation Notification',
                                                'Please enter a comma separated list for b-values and directions per shell')
                    else:
                        messagebox.showinfo('Simulation Notification', 'Please enter acceptable input type: integers')
                else:
                    messagebox.showinfo('Simulation Notification', 'Please enter edge voxels that are in the respective node')
                return
            # button for starting simulation
            button3 = ttk.Button(self, text="Start Simulation", command=start_sim)
            button3.grid(row=8, column=1)

            def save_func():
                if not os.path.isdir("simulation_parameters"):
                    os.mkdir("simulation_parameters")
                file = open("simulation_parameters/" + filename.get() + ".txt", "w+")
                # node_list, edge_list, bval, nshells
                # write xyz
                file.write(xsize.get() + "\n")
                file.write(ysize.get() + "\n")
                file.write(zsize.get() + "\n")
                # write bvals
                file.write(bval.get() + "\n")
                file.write(nshells.get() + "\n")
                # write nshells
                # write node list
                numnodes = 0
                for node in get_node_info():
                    if node is not None and node is not "":
                        numnodes += 1
                file.write(str(numnodes) + "\n")
                for node in get_node_info():
                    if node is not None and node is not "":
                        file.write(node + "\n")
                # write edge list
                numedge = 0
                for edge in get_edge_info():
                    if edge[0] is not None and edge[0] is not "":
                        numedge += 1
                file.write(str(numedge) + "\n")
                for edge in get_edge_info():
                    for element in edge:
                        if element is not None and element is not "":
                            file.write(element + "\n")
                    # file.write(edge + "\n")
                file.flush()
                file.close()
                return True

            save = ttk.Button(self, text="Save", command=save_func)
            save.grid(row=0, column=2)

            def load_func():
                if not os.path.isdir("simulation_parameters"):
                    messagebox.showinfo('Load Error', 'There are no saved parameters, please manually enter simulation parameters.')
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
                    filename.delete(0, 'end')
                    filename.insert(0, file_name.split(sep="/")[-1][:-4])
                    # set node info
                    numnodes = int(file.readline()[:-1])
                    node_info = []
                    for i in range(numnodes):
                        node_info.append(file.readline()[:-1])
                    set_node_info(node_info)
                    # set edge info
                    numedges = int(file.readline()[:-1])
                    edge_info = []
                    for i in range(numedges):
                        edge = []
                        for j in range(4):
                            edge.append(file.readline()[:-1])
                        edge_info.append(edge)
                    set_edge_info(edge_info)
                    file.close()
                    return True

            load = ttk.Button(self, text="Load", command=load_func)
            load.grid(row=1, column=2)

    class AddNode(tk.Frame):
        node_list = np.empty(6, dtype='object')

        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)

            button1 = ttk.Button(self, text="Main Menu", command=lambda: controller.show_frame(MainMenu))
            button1.grid(row=0, column=0)

            button2 = ttk.Button(self, text="Add Edges", command=lambda: controller.show_frame(AddEdge))
            button2.grid(row=0, column=1)

            def add_node():
                # check which nodes are visible and make visible the next node
                if self.node2.grid_info() == {}:
                    self.node2_lbl.grid(row=3, column=0)
                    self.node2.grid(row=3, column=1, columnspan=20)
                elif self.node3.grid_info() == {}:
                    self.node3_lbl.grid(row=4, column=0)
                    self.node3.grid(row=4, column=1, columnspan=20)
                elif self.node4.grid_info() == {}:
                    self.node4_lbl.grid(row=5, column=0)
                    self.node4.grid(row=5, column=1, columnspan=20)
                elif self.node5.grid_info() == {}:
                    self.node5_lbl.grid(row=6, column=0)
                    self.node5.grid(row=6, column=1, columnspan=20)
                elif self.node6.grid_info() == {}:
                    self.node6_lbl.grid(row=7, column=0)
                    self.node6.grid(row=7, column=1, columnspan=20)
                else:
                    messagebox.showinfo('Add Node Error', 'The maximum number of nodes have been added')

            button3 = ttk.Button(self, text="Add Node", command=add_node)
            button3.grid(row=1, column=0)

            def remove_node():
                # check which nodes are visible and make visible the next node
                if not (self.node6.grid_info() == {}):
                    self.node6_lbl.grid_forget()
                    self.node6.grid_forget()
                    self.node6.delete(0, 'end')
                elif not (self.node5.grid_info() == {}):
                    self.node5_lbl.grid_forget()
                    self.node5.grid_forget()
                    self.node5.delete(0, 'end')
                elif not (self.node4.grid_info() == {}):
                    self.node4_lbl.grid_forget()
                    self.node4.grid_forget()
                    self.node4.delete(0, 'end')
                elif not (self.node3.grid_info() == {}):
                    self.node3_lbl.grid_forget()
                    self.node3.grid_forget()
                    self.node3.delete(0, 'end')
                elif not (self.node2.grid_info() == {}):
                    self.node2_lbl.grid_forget()
                    self.node2.grid_forget()
                    self.node2.delete(0, 'end')

            button4 = ttk.Button(self, text="Remove Node", command=remove_node)
            button4.grid(row=1, column=1)

            def callback1():
                self.node_list[0] = self.node1.get()
                return True

            self.node1_lbl = ttk.Label(self, text="Node 1:", font=LARGE_FONT)
            self.node1_lbl.grid(row=2, column=0)
            self.node1 = ttk.Entry(self, width=90, validate="all", validatecommand=callback1)
            self.node1.grid(row=2, column=1, columnspan=20)

            def callback2():
                self.node_list[1] = self.node2.get()
                return True

            self.node2_lbl = ttk.Label(self, text="Node 2:", font=LARGE_FONT)
            self.node2_lbl.grid(row=3, column=0)
            self.node2 = ttk.Entry(self, width=90, validate="all", validatecommand=callback2)
            self.node2.grid(row=3, column=1, columnspan=20)
            self.node2_lbl.grid_forget()
            self.node2.grid_forget()

            def callback3():
                self.node_list[2] = self.node3.get()
                return True

            self.node3_lbl = ttk.Label(self, text="Node 3:", font=LARGE_FONT)
            self.node3_lbl.grid(row=4, column=0)
            self.node3 = ttk.Entry(self, width=90, validate="all", validatecommand=callback3)
            self.node3.grid(row=4, column=1, columnspan=20)
            self.node3_lbl.grid_forget()
            self.node3.grid_forget()

            def callback4():
                self.node_list[3] = self.node4.get()
                return True

            self.node4_lbl = ttk.Label(self, text="Node 4:", font=LARGE_FONT)
            self.node4_lbl.grid(row=5, column=0)
            self.node4 = ttk.Entry(self, width=90, validate="all", validatecommand=callback4)
            self.node4.grid(row=5, column=1, columnspan=20)
            self.node4_lbl.grid_forget()
            self.node4.grid_forget()

            def callback5():
                self.node_list[4] = self.node5.get()
                return True

            self.node5_lbl = ttk.Label(self, text="Node 5:", font=LARGE_FONT)
            self.node5_lbl.grid(row=6, column=0)
            self.node5 = ttk.Entry(self, width=90, validate="all", validatecommand=callback5)
            self.node5.grid(row=6, column=1, columnspan=20)
            self.node5_lbl.grid_forget()
            self.node5.grid_forget()

            def callback6():
                self.node_list[5] = self.node6.get()
                return True

            self.node6_lbl = ttk.Label(self, text="Node 6:", font=LARGE_FONT)
            self.node6_lbl.grid(row=7, column=0)
            self.node6 = ttk.Entry(self, width=90, validate="all", validatecommand=callback6)
            self.node6.grid(row=7, column=1, columnspan=20)
            self.node6_lbl.grid_forget()
            self.node6.grid_forget()

            blank = ttk.Label(self, text=" ", font=LARGE_FONT)
            blank.grid(row=8, column=0)
            info = ttk.Label(self, width=50, text=" Enter a list of tuples (a, b, c), ..., (x, y, z) for each node.", font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info.grid(row=9, column=0, columnspan=10)

            self.node_list[0] = self.node1.get()
            self.node_list[1] = self.node2.get()
            self.node_list[2] = self.node3.get()
            self.node_list[3] = self.node4.get()
            self.node_list[4] = self.node5.get()
            self.node_list[5] = self.node6.get()

        def get_node_list(self):
            self.node_list[0] = self.node1.get()
            self.node_list[1] = self.node2.get()
            self.node_list[2] = self.node3.get()
            self.node_list[3] = self.node4.get()
            self.node_list[4] = self.node5.get()
            self.node_list[5] = self.node6.get()
            return self.node_list

        def set_node_list(self, node_info):
            count = 0
            self.node6_lbl.grid_forget()
            self.node6.grid_forget()
            self.node6.delete(0, 'end')
            # self.node6_input.set(str())
            self.node5_lbl.grid_forget()
            self.node5.grid_forget()
            self.node5.delete(0, 'end')
            # self.node5_input.set(str())
            self.node4_lbl.grid_forget()
            self.node4.grid_forget()
            self.node4.delete(0, 'end')
            # self.node4_input.set(str())
            self.node3_lbl.grid_forget()
            self.node3.grid_forget()
            self.node3.delete(0, 'end')
            # self.node3_input.set(str())
            self.node2_lbl.grid_forget()
            self.node2.grid_forget()
            self.node2.delete(0, 'end')
            # self.node2_input.set(str())
            self.node1.delete(0, 'end')
            # self.node1_input.set(str())
            for node in node_info:
                count += 1
                if count == 1:
                    self.node1.insert(0, node_info[0])
                elif count == 2:
                    self.node2.insert(0, node_info[1])
                    self.node2_lbl.grid(row=3, column=0)
                    self.node2.grid(row=3, column=1, columnspan=20)
                elif count == 3:
                    self.node3.insert(0, node_info[2])
                    self.node3_lbl.grid(row=4, column=0)
                    self.node3.grid(row=4, column=1, columnspan=20)
                elif count == 4:
                    self.node4.insert(0, node_info[3])
                    self.node4_lbl.grid(row=5, column=0)
                    self.node4.grid(row=5, column=1, columnspan=20)
                elif count == 5:
                    self.node5.insert(0, node_info[4])
                    self.node5_lbl.grid(row=6, column=0)
                    self.node5.grid(row=6, column=1, columnspan=20)
                elif count == 6:
                    self.node6.insert(0, node_info[5])
                    self.node6_lbl.grid(row=7, column=0)
                    self.node6.grid(row=7, column=1, columnspan=20)
            return

    class AddEdge(tk.Frame):
        # edge weights have units in mm^2/s (common value is .0017, .0003, .0003)
        edge_list = np.empty((3, 4), dtype='object')

        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)

            button1 = ttk.Button(self, text="Main Menu", command=lambda: controller.show_frame(MainMenu))
            button1.grid(row=0, column=0)

            button2 = ttk.Button(self, text="Add Nodes", command=lambda: controller.show_frame(AddNode))
            button2.grid(row=0, column=1)

            def add_edge():
                # check which edges are visible and make visible the next edge
                if self.edge2.grid_info() == {}:
                    self.edge2_lbl.grid(row=3, column=0)
                    self.edge2.grid(row=3, column=1)
                    self.edge2_dir.grid(row=3, column=2)
                    self.edge2_voxels.grid(row=3, column=3)
                    self.edge2_rad.grid(row=3, column=4)
                elif self.edge3.grid_info() == {}:
                    self.edge3_lbl.grid(row=4, column=0)
                    self.edge3.grid(row=4, column=1)
                    self.edge3_dir.grid(row=4, column=2)
                    self.edge3_voxels.grid(row=4, column=3)
                    self.edge3_rad.grid(row=4, column=4)
                else:
                    tk.messagebox.showinfo('Add Edge Error', 'The maximum number of edges have been added')

            button3 = ttk.Button(self, text="Add Edge", command=add_edge)
            button3.grid(row=1, column=0)

            def remove_edge():
                # check which edges are visible and make visible the next edge
                if not (self.edge3.grid_info() == {}):
                    self.edge3_lbl.grid_forget()
                    self.edge3.grid_forget()
                    self.edge3.delete(0, 'end')
                    self.edge3_dir.grid_forget()
                    self.edge3_dir.delete(0, 'end')
                    self.edge3_voxels.grid_forget()
                    self.edge3_voxels.delete(0, 'end')
                    self.edge3_rad.grid_forget()
                    self.edge3_rad.delete(0, 'end')
                elif not (self.edge2.grid_info() == {}):
                    self.edge2_lbl.grid_forget()
                    self.edge2.grid_forget()
                    self.edge2.delete(0, 'end')
                    self.edge2_dir.grid_forget()
                    self.edge2_dir.delete(0, 'end')
                    self.edge2_voxels.grid_forget()
                    self.edge2_voxels.delete(0, 'end')
                    self.edge2_rad.grid_forget()
                    self.edge2_rad.delete(0, 'end')

            button4 = ttk.Button(self, text="Remove Edge", command=remove_edge)
            button4.grid(row=1, column=1)

            def callback1():
                self.edge_list[0][0] = self.edge1.get()
                return True

            def dircallback1():
                self.edge_list[0][1] = self.edge1_dir.get()
                return True

            def vcallback1():
                self.edge_list[0][2] = self.edge1_voxels.get()
                return True

            def rcallback1():
                self.edge_list[0][3] = self.edge1_rad.get()
                return True

            self.edge1_lbl = ttk.Label(self, text="Edge 1:", font=LARGE_FONT)
            self.edge1_lbl.grid(row=2, column=0)
            self.edge1 = ttk.Entry(self, width=10, validate="all", validatecommand=callback1)
            self.edge1.grid(row=2, column=1)
            self.edge1_dir = ttk.Entry(self, width=20, validate="all", validatecommand=dircallback1)
            self.edge1_dir.grid(row=2, column=2)
            self.edge1_voxels = ttk.Entry(self, width=20, validate="all", validatecommand=vcallback1)
            self.edge1_voxels.grid(row=2, column=3)
            self.edge1_rad = ttk.Entry(self, width=10, validate="all", validatecommand=rcallback1)
            self.edge1_rad.grid(row=2, column=4)

            def callback2():
                self.edge_list[1][0] = self.edge2.get()
                return True

            def dircallback2():
                self.edge_list[1][1] = self.edge2_dir.get()
                return True

            def vcallback2():
                self.edge_list[1][2] = self.edge2_voxels.get()
                return True

            def rcallback2():
                self.edge_list[1][3] = self.edge2_rad.get()
                return True

            self.edge2_lbl = ttk.Label(self, text="Edge 2:", font=LARGE_FONT)
            self.edge2_lbl.grid(row=3, column=0)
            self.edge2 = ttk.Entry(self, width=10, validate="all", validatecommand=callback2)
            self.edge2.grid(row=3, column=1)
            self.edge2_dir = ttk.Entry(self, width=20, validate="all", validatecommand=dircallback2)
            self.edge2_dir.grid(row=3, column=2)
            self.edge2_voxels = ttk.Entry(self, width=20, validate="all", validatecommand=vcallback2)
            self.edge2_voxels.grid(row=3, column=3)
            self.edge2_rad = ttk.Entry(self, width=10, validate="all", validatecommand=rcallback2)
            self.edge2_rad.grid(row=3, column=4)

            self.edge2_lbl.grid_forget()
            self.edge2.grid_forget()
            self.edge2_dir.grid_forget()
            self.edge2_voxels.grid_forget()
            self.edge2_rad.grid_forget()

            def callback3():
                self.edge_list[2][0] = self.edge3.get()
                return True

            def dircallback3():
                self.edge_list[2][1] = self.edge3_dir.get()
                return True

            def vcallback3():
                self.edge_list[2][2] = self.edge3_voxels.get()
                return True

            def rcallback3():
                self.edge_list[2][3] = self.edge3_rad.get()
                return True

            self.edge3_lbl = ttk.Label(self, text="Edge 3:", font=LARGE_FONT)
            self.edge3_lbl.grid(row=4, column=0)
            self.edge3 = ttk.Entry(self, width=10, validate="all", validatecommand=callback3)
            self.edge3.grid(row=4, column=1)
            self.edge3_dir = ttk.Entry(self, width=20, validate="all", validatecommand=dircallback3)
            self.edge3_dir.grid(row=4, column=2)
            self.edge3_voxels = ttk.Entry(self, width=20, validate="all", validatecommand=vcallback3)
            self.edge3_voxels.grid(row=4, column=3)
            self.edge3_rad = ttk.Entry(self, width=10, validate="all", validatecommand=rcallback3)
            self.edge3_rad.grid(row=4, column=4)

            self.edge3_lbl.grid_forget()
            self.edge3.grid_forget()
            self.edge3_dir.grid_forget()
            self.edge3_voxels.grid_forget()
            self.edge3_rad.grid_forget()

            blank = ttk.Label(self, text=" ", font=LARGE_FONT)
            blank.grid(row=5, column=0)
            info = ttk.Label(self, width=100, text=" In the first entry box enter two node numbers (e.g. node1, node2) with a comma and space after the first node.", font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info.grid(row=6, column=0, columnspan=10)
            info1 = ttk.Label(self, width=100, text=" Enter a tuple x, y, z with spaces in the second entry box corresponding to the eigenvalues of the diffusion tensor.",
                             font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info1.grid(row=7, column=0, columnspan=10)
            info2 = ttk.Label(self, width=100, text=" In the thrid box enter two tuples (e.g. (a, b, c), (d, e, f)) from the two respective nodes to define the edge between them.",
                              font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info2.grid(row=8, column=0, columnspan=10)
            info3 = ttk.Label(self, width=100, text=" In the last box enter a float for the radius of the edge.", font=LARGE_FONT, anchor='w', justify=tk.LEFT)
            info3.grid(row=9, column=0, columnspan=10)

            self.edge_list[0][0] = self.edge1.get()
            self.edge_list[0][1] = self.edge1_dir.get()
            self.edge_list[0][2] = self.edge1_voxels.get()
            self.edge_list[0][3] = self.edge1_rad.get()
            self.edge_list[1][0] = self.edge2.get()
            self.edge_list[1][1] = self.edge2_dir.get()
            self.edge_list[1][2] = self.edge2_voxels.get()
            self.edge_list[1][3] = self.edge2_rad.get()
            self.edge_list[2][0] = self.edge3.get()
            self.edge_list[2][1] = self.edge3_dir.get()
            self.edge_list[2][2] = self.edge3_voxels.get()
            self.edge_list[2][3] = self.edge3_rad.get()

        def get_edge_list(self):
            self.edge_list[0][0] = self.edge1.get()
            self.edge_list[0][1] = self.edge1_dir.get()
            self.edge_list[0][2] = self.edge1_voxels.get()
            self.edge_list[0][3] = self.edge1_rad.get()
            self.edge_list[1][0] = self.edge2.get()
            self.edge_list[1][1] = self.edge2_dir.get()
            self.edge_list[1][2] = self.edge2_voxels.get()
            self.edge_list[1][3] = self.edge2_rad.get()
            self.edge_list[2][0] = self.edge3.get()
            self.edge_list[2][1] = self.edge3_dir.get()
            self.edge_list[2][2] = self.edge3_voxels.get()
            self.edge_list[2][3] = self.edge3_rad.get()
            return self.edge_list

        def set_edge_list(self, edge_info):
            self.edge3_lbl.grid_forget()
            self.edge3.grid_forget()
            self.edge3.delete(0, 'end')
            self.edge3_dir.grid_forget()
            self.edge3_dir.delete(0, 'end')
            self.edge3_voxels.grid_forget()
            self.edge3_voxels.delete(0, 'end')
            self.edge3_rad.grid_forget()
            # self.edge3_input.set(str())
            # self.edge3_dir_input.set(str())
            # self.edge3_voxel_input.set(str())
            # self.edge3_radius.set(str())
            self.edge3_rad.delete(0, 'end')
            self.edge2_lbl.grid_forget()
            self.edge2.grid_forget()
            self.edge2.delete(0, 'end')
            self.edge2_dir.grid_forget()
            self.edge2_dir.delete(0, 'end')
            self.edge2_voxels.grid_forget()
            self.edge2_voxels.delete(0, 'end')
            self.edge2_rad.grid_forget()
            self.edge2_rad.delete(0, 'end')
            # self.edge2_input.set(str())
            # self.edge2_dir_input.set(str())
            # self.edge2_voxel_input.set(str())
            # self.edge2_radius.set(str())
            self.edge1.delete(0, 'end')
            self.edge1_dir.delete(0, 'end')
            self.edge1_voxels.delete(0, 'end')
            self.edge1_rad.delete(0, 'end')
            # self.edge1_input.set(str())
            # self.edge1_dir_input.set(str())
            # self.edge1_voxel_input.set(str())
            # self.edge1_radius.set(str())
            for i in range(len(edge_info)):
                if i == 0:
                    self.edge1.insert(0, edge_info[i][0])
                    self.edge1_dir.insert(0, edge_info[i][1])
                    self.edge1_voxels.insert(0, edge_info[i][2])
                    self.edge1_rad.insert(0, edge_info[i][3])
                elif i == 1:
                    self.edge2.insert(0, edge_info[i][0])
                    self.edge2_dir.insert(0, edge_info[i][1])
                    self.edge2_voxels.insert(0, edge_info[i][2])
                    self.edge2_rad.insert(0, edge_info[i][3])
                    self.edge2_lbl.grid(row=3, column=0)
                    self.edge2.grid(row=3, column=1)
                    self.edge2_dir.grid(row=3, column=2)
                    self.edge2_voxels.grid(row=3, column=3)
                    self.edge2_rad.grid(row=3, column=4)
                elif i == 2:
                    self.edge3.insert(0, edge_info[i][0])
                    self.edge3_dir.insert(0, edge_info[i][1])
                    self.edge3_voxels.insert(0, edge_info[i][2])
                    self.edge3_rad.insert(0, edge_info[i][3])
                    self.edge3_lbl.grid(row=4, column=0)
                    self.edge3.grid(row=4, column=1)
                    self.edge3_dir.grid(row=4, column=2)
                    self.edge3_voxels.grid(row=4, column=3)
                    self.edge3_rad.grid(row=4, column=4)
            return

    app = DWIS()
    app.mainloop()
