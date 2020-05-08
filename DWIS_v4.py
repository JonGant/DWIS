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
from qspace import multishell


# initialize b vectors
def create_bvecs_and_bvals(low_bval, high_bval, nshells, res=2.5):
    bval_list = np.empty((nshells))
    for i in range(nshells):
        bval_list[i] = low_bval + i*(high_bval)/nshells
    npoints = []
    totalpoints = 0
    bvals = []
    for bval in bval_list:
        npoints.append(int(np.floor(res*np.sqrt(bval))))
        totalpoints = totalpoints + int(np.floor(res*np.sqrt(bval)))
        for i in range(int(np.floor(res*np.sqrt(bval)))):
            if i == 0:
                bvals.append(0)
            else:
                bvals.append(bval)
    weights = 0.5*np.ones((totalpoints, totalpoints))
    bvecs = multishell.optimize(nshells, npoints, weights)

    return [bvecs, bvals]


# main part of the program which simulates the diffusion tensor data
def simulate_dwi_calc(low_bval, high_bval, nshells, xsize, ysize, zsize, dt_data):
    # create b vectors and values
    bvecval = create_bvecs_and_bvals(low_bval=low_bval, high_bval=high_bval, nshells=nshells)
    bvecs = bvecval[0]
    bvals = bvecval[1]
    # Create a data array, set values of parameters
    print(bvecval)
    dirs = np.size(bvecs, 0)
    data = np.zeros((xsize, ysize, zsize, dirs), float)
    S0 = 1.0
    VF1 = 1 # Volume Fraction
    for k in range(zsize):
        for j in range(ysize):
            for i in range(xsize):
                for m in range(dirs):
                    term = np.matmul(bvecs[m,:], np.matmul(dt_data[i][j][k][0], np.transpose(bvecs[m,:])))
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
             str(xsize) + "_" + str(ysize) + "_" + str(zsize) + ".nii.gz")
    # print bvals
    file = open("nifti_images/" + str(low_bval) + "_" + str(high_bval) + "_" + str(nshells) + "_" +
             str(xsize) + "_" + str(ysize) + "_" + str(zsize) + ".bval", "w+")
    for i in range(len(bvals)):
        file.write(str(bvals[i])+" ")
    file.close()
    # print bvecs
    file = open("nifti_images/" + str(low_bval) + "_" + str(high_bval) + "_" + str(nshells) + "_" +
             str(xsize) + "_" + str(ysize) + "_" + str(zsize) + ".bvec", "w+")
    for i in range(3):
        for j in range(len(bvecs)):
            file.write(str(bvecs[j][i]) + " ")
        file.write("\n")
    file.close()

    return False


# function rotates diffusion data in a voxel to be in the direction of the two angles phi and theta (from spherical coordinates)
def rotate(voxel_data, phi, theta):
    rot = np.array([[np.cos(phi)*np.cos(theta), -np.sin(phi), np.cos(phi)*np.sin(theta)], [np.sin(phi)*np.cos(theta), np.cos(phi),
                                                                                         np.sin(phi)*np.sin(theta)], [-np.sin(theta), 0, np.cos(theta)]])
    rotated_voxel_data = np.matmul(np.matmul(rot, voxel_data), np.transpose(rot))
    return rotated_voxel_data


# this function takes in a list of nodes and a list of parameters determining the width and curvature of each edge
# and creates the diffusion tensor information associated with the edges
# TODO make it so node size is a parameter
def create_dt_data(xsize, ysize, zsize, nodes, edges, radius=1):
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
            displacement = np.array([nodes[node2-1][0]-nodes[node1-1][0], nodes[node2-1][1]-nodes[node1-1][1], nodes[node2-1][2]-nodes[node1-1][2]])
            displacements.append(displacement)
            directions.append(displacement/np.linalg.norm(displacement))
    # defines the characteristic function
    # iterate numerically over the normalized slope vectors on four vertices of voxel to find voxels that are "hit" by the slope vectors
    hit_voxels = []
    for i in range(len(displacements)):
        voxels = []
        # first vertex
        startpoint1 = np.array([nodes[edges[i][0]-1][0], nodes[edges[i][0]-1][1], nodes[edges[i][0]-1][2]])
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
                if currentstartpoint[0] < xsize and currentstartpoint[1] < ysize and currentstartpoint[2] < zsize:
                    voxels.append(np.floor(currentstartpoint))
        # second vertex
        startpoint2 = np.array([nodes[edges[i][0]-1][0] + radius, nodes[edges[i][0]-1][1], nodes[edges[i][0]-1][2]])
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
                if currentstartpoint[0] < xsize and currentstartpoint[1] < ysize and currentstartpoint[2] < zsize:
                    voxels.append(np.floor(currentstartpoint))
        # third vertex
        startpoint3 = np.array([nodes[edges[i][0]-1][0], nodes[edges[i][0]-1][1] + radius, nodes[edges[i][0]-1][2]])
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
                if currentstartpoint[0] < xsize and currentstartpoint[1] < ysize and currentstartpoint[2] < zsize:
                    voxels.append(np.floor(currentstartpoint))
        # fourth vertex
        startpoint4 = np.array([nodes[edges[i][0]-1][0], nodes[edges[i][0]-1][1], nodes[edges[i][0]-1][2]+ radius])
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
                if currentstartpoint[0]<xsize and currentstartpoint[1]<ysize and currentstartpoint[2]<zsize:
                    voxels.append(np.floor(currentstartpoint))
        # add all the hit voxels to the overall list of hit voxels which is per edge
        hit_voxels.append(voxels)
    # TODO somewhere in here make error message for if more than three edges are on top of each other
    # find the angles of each direction vector
    angles = np.empty((len(directions), 2))
    for i in range(len(directions)):
        angles[i][0] = np.arctan(directions[i][1]/directions[i][0])
        angles[i][1] = np.arctan(np.sqrt(np.square(directions[i][0])+np.square(directions[i][1]))/directions[i][2])
    # take the voxels that were hit and construct the data associated with those voxels
    for i in range(len(hit_voxels)):
        # sample diffusion profile (to be rotated)
        diff = np.array([[.0017, 0, 0], [0, .0003, 0], [0, 0, .0003]])
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
    return [dt_data, failed]


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

            label = ttk.Label(self, text="Main Menu", font=LARGE_FONT)
            label.grid(row=0, column=0)

            button1 = ttk.Button(self, text="Add Nodes", command=lambda: controller.show_frame(AddNode))
            button1.grid(row=1, column=0)

            button2 = ttk.Button(self, text="Add Edges", command=lambda: controller.show_frame(AddEdge))
            button2.grid(row=1, column=1)

            xsize_lbl = ttk.Label(self, text="X size:", font=LARGE_FONT)
            xsize_lbl.grid(row=2, column=0)
            xsize = ttk.Entry(self, width=10)
            xsize.grid(row=2, column=1)
            xsize.focus()

            ysize_lbl = ttk.Label(self, text="Y size:", font=LARGE_FONT)
            ysize_lbl.grid(row=3, column=0)
            ysize = ttk.Entry(self, width=10)
            ysize.grid(row=3, column=1)

            zsize_lbl = ttk.Label(self, text="Z size:", font=LARGE_FONT)
            zsize_lbl.grid(row=4, column=0)
            zsize = ttk.Entry(self, width=10)
            zsize.grid(row=4, column=1)

            low_bval_lbl = ttk.Label(self, text="Low b-value: ", font=("Arial", 12))
            low_bval_lbl.grid(row=5, column=0)
            low_bval = ttk.Entry(self, width=10)
            low_bval.grid(row=5, column=1)

            high_bval_lbl = ttk.Label(self, text="High b-value: ", font=("Arial", 12))
            high_bval_lbl.grid(row=6, column=0)
            high_bval = ttk.Entry(self, width=10)
            high_bval.grid(row=6, column=1)

            nshells_lbl = ttk.Label(self, text="Number of shells: ", font=("Arial", 12))
            nshells_lbl.grid(row=7, column=0)
            nshells = ttk.Entry(self, width=10)
            nshells.grid(row=7, column=1)

            # get node info
            def get_node_info():
                AddNodeFrame = controller.get_frame(AddNode)
                return AddNodeFrame.get_node_list()

            # get edge info
            def get_edge_info():
                AddEdgeFrame = controller.get_frame(AddEdge)
                return AddEdgeFrame.get_edge_list()

            # start simulation command
            def start_sim():
                node_list = get_node_info()
                edge_list = get_edge_info()
                # format lists correctly
                node_list_formatted = np.zeros((len(node_list), 3), dtype='int8')
                for i in range(len(node_list)):
                    if node_list[i] is not None:
                        split_node = node_list[i].split(sep=',')
                        node_list_formatted[i]=np.array([int(split_node[0]), int(split_node[1]), int(split_node[2])])
                edge_list_formatted = np.zeros((len(edge_list), 2), dtype='int32')
                for i in range(len(edge_list)):
                    if edge_list[i] is not None:
                        split_edge = edge_list[i].split(sep=',')
                        edge_list_formatted[i]=np.array([int(split_edge[0]), int(split_edge[1])])
                if (xsize.get().isdigit() and ysize.get().isdigit() and zsize.get().isdigit() and low_bval.get().isdigit()
                        and high_bval.get().isdigit() and nshells.get().isdigit()):
                    if int(nshells.get()) >= 2:
                        if int(low_bval.get()) < int(high_bval.get()):
                            # create result directory if it does not already exist
                            if not os.path.isdir("nifti_images"):
                                os.mkdir("nifti_images")
                            # check to see if file of these parameters already exists
                            if not (os.path.exists(
                                    "nifti_images/dwi_" + str(low_bval) + "_" + str(high_bval) + "_" + str(
                                            nshells) + "_" +
                                    str(xsize) + "_" + str(ysize) + "_" + str(zsize) + ".nii.gz")):
                                if 0 < int(xsize.get()) <= 100 and 0 < int(ysize.get()) <= 100 and 0 < int(
                                    zsize.get()) <= 100:
                                    messagebox.showinfo('Simulation Notification', 'Simulation has started')
                                    # create the diffusion tensor info file
                                    dt_data = create_dt_data(int(xsize.get()), int(ysize.get()), int(zsize.get()),
                                                             node_list_formatted, edge_list_formatted)
                                    if not dt_data[1]:
                                        # Start the simulation
                                        param = True
                                        while(param):
                                            param = simulate_dwi_calc(int(low_bval.get()), int(high_bval.get()), int(nshells.get()),
                                                              int(xsize.get()), int(ysize.get()), int(zsize.get()), dt_data[0])
                                        messagebox.showinfo('Simulation Notification', 'Simulation successfully finished')
                                    else:
                                        messagebox.showinfo('Simulation Notification',
                                                            'More than three edges crossed in a single voxel! Please enter edges such that only a maximum of three edges cross in a single voxel.')
                                else:
                                    messagebox.showinfo('Simulation Notification',
                                                        'Please enter values greater than 0 and less than 100 for XYZ sizes')
                            else:
                                messagebox.showinfo('Simulation Notification',
                                                    'File already exists, please enter different parameters')
                        else:
                            messagebox.showinfo('Simulation Notification',
                                                'Please ensure that low b-value is less than high b-value')
                    else:
                        messagebox.showinfo('Simulation Notification',
                                            'Please enter integers greater than 2 for the number of shells')
                else:
                    messagebox.showinfo('Simulation Notification', 'Please enter acceptable input type: integers')
                return
            # button for starting simulation
            button3 = ttk.Button(self, text="Start Simulation", command=start_sim)
            button3.grid(row=8, column=1)


    class AddNode(tk.Frame):
        node_list = np.empty(10, dtype='object')
        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)

            button1 = ttk.Button(self, text="Main Menu", command=lambda: controller.show_frame(MainMenu))
            button1.grid(row=0, column=0)

            button2 = ttk.Button(self, text="Add Edges", command=lambda: controller.show_frame(AddEdge))
            button2.grid(row=0, column=1)

            def add_node():
                # check which nodes are visible and make visible the next node
                if node2.grid_info() == {}:
                    node2_lbl.grid(row=3, column=0)
                    node2.grid(row=3, column=1)
                elif node3.grid_info() == {}:
                    node3_lbl.grid(row=4, column=0)
                    node3.grid(row=4, column=1)
                elif node4.grid_info() == {}:
                    node4_lbl.grid(row=5, column=0)
                    node4.grid(row=5, column=1)
                elif node5.grid_info() == {}:
                    node5_lbl.grid(row=6, column=0)
                    node5.grid(row=6, column=1)
                elif node6.grid_info() == {}:
                    node6_lbl.grid(row=2, column=3)
                    node6.grid(row=2, column=4)
                elif node7.grid_info() == {}:
                    node7_lbl.grid(row=3, column=3)
                    node7.grid(row=3, column=4)
                elif node8.grid_info() == {}:
                    node8_lbl.grid(row=4, column=3)
                    node8.grid(row=4, column=4)
                elif node9.grid_info() == {}:
                    node9_lbl.grid(row=5, column=3)
                    node9.grid(row=5, column=4)
                elif node10.grid_info() == {}:
                    node10_lbl.grid(row=6, column=3)
                    node10.grid(row=6, column=4)
                else:
                    messagebox.showinfo('Add Node Error', 'The maximum number of nodes have been added')

            button3 = ttk.Button(self, text="Add Node", command=add_node)
            button3.grid(row=1, column=0)

            def remove_node():
                # check which nodes are visible and make visible the next node
                if not (node10.grid_info() == {}):
                    node10_lbl.grid_forget()
                    node10.grid_forget()
                elif not (node9.grid_info() == {}):
                    node9_lbl.grid_forget()
                    node9.grid_forget()
                elif not (node8.grid_info() == {}):
                    node8_lbl.grid_forget()
                    node8.grid_forget()
                elif not (node7.grid_info() == {}):
                    node7_lbl.grid_forget()
                    node7.grid_forget()
                elif not (node6.grid_info() == {}):
                    node6_lbl.grid_forget()
                    node6.grid_forget()
                elif not (node5.grid_info() == {}):
                    node5_lbl.grid_forget()
                    node5.grid_forget()
                elif not (node4.grid_info() == {}):
                    node4_lbl.grid_forget()
                    node4.grid_forget()
                elif not (node3.grid_info() == {}):
                    node3_lbl.grid_forget()
                    node3.grid_forget()
                elif not (node2.grid_info() == {}):
                    node2_lbl.grid_forget()
                    node2.grid_forget()

            button4 = ttk.Button(self, text="Remove Node", command=remove_node)
            button4.grid(row=1, column=1)

            node1_input = tk.StringVar()

            def callback1():
                self.node_list[0] = node1_input.get()
                return True
            node1_lbl = ttk.Label(self, text="Node 1:", font=LARGE_FONT)
            node1_lbl.grid(row=2, column=0)
            node1 = ttk.Entry(self, width=10, textvariable=node1_input, validate="focusout", validatecommand=callback1)
            node1.grid(row=2, column=1)

            node2_input = tk.StringVar()

            def callback2():
                self.node_list[1] = node2_input.get()
                return True
            node2_lbl = ttk.Label(self, text="Node 2:", font=LARGE_FONT)
            node2_lbl.grid(row=3, column=0)
            node2 = ttk.Entry(self, width=10, textvariable=node2_input, validate="focusout", validatecommand=callback2)
            node2.grid(row=3, column=1)
            node2_lbl.grid_forget()
            node2.grid_forget()

            node3_input = tk.StringVar()

            def callback3():
                self.node_list[2] = node3_input.get()
                return True
            node3_lbl = ttk.Label(self, text="Node 3:", font=LARGE_FONT)
            node3_lbl.grid(row=4, column=0)
            node3 = ttk.Entry(self, width=10, textvariable=node3_input, validate="focusout", validatecommand=callback3)
            node3.grid(row=4, column=1)
            node3_lbl.grid_forget()
            node3.grid_forget()

            node4_input = tk.StringVar()

            def callback4():
                self.node_list[3] = node4_input.get()
                return True
            node4_lbl = ttk.Label(self, text="Node 4:", font=LARGE_FONT)
            node4_lbl.grid(row=5, column=0)
            node4 = ttk.Entry(self, width=10, textvariable=node4_input, validate="focusout", validatecommand=callback4)
            node4.grid(row=5, column=1)
            node4_lbl.grid_forget()
            node4.grid_forget()

            node5_input = tk.StringVar()

            def callback5():
                self.node_list[4] = node5_input.get()
                return True
            node5_lbl = ttk.Label(self, text="Node 5:", font=LARGE_FONT)
            node5_lbl.grid(row=6, column=0)
            node5 = ttk.Entry(self, width=10, textvariable=node5_input, validate="focusout", validatecommand=callback5)
            node5.grid(row=6, column=1)
            node5_lbl.grid_forget()
            node5.grid_forget()

            node6_input = tk.StringVar()

            def callback6():
                self.node_list[5] = node6_input.get()
                return True
            node6_lbl = ttk.Label(self, text="Node 6:", font=LARGE_FONT)
            node6_lbl.grid(row=2, column=3)
            node6 = ttk.Entry(self, width=10, textvariable=node6_input, validate="focusout", validatecommand=callback6)
            node6.grid(row=2, column=4)
            node6_lbl.grid_forget()
            node6.grid_forget()

            node7_input = tk.StringVar()

            def callback7():
                self.node_list[6] = node7_input.get()
                return True
            node7_lbl = ttk.Label(self, text="Node 7:", font=LARGE_FONT)
            node7_lbl.grid(row=3, column=3)
            node7 = ttk.Entry(self, width=10, textvariable=node7_input, validate="focusout", validatecommand=callback7)
            node7.grid(row=3, column=4)
            node7_lbl.grid_forget()
            node7.grid_forget()

            node8_input = tk.StringVar()

            def callback8():
                self.node_list[7] = node8_input.get()
                return True
            node8_lbl = ttk.Label(self, text="Node 8:", font=LARGE_FONT)
            node8_lbl.grid(row=4, column=3)
            node8 = ttk.Entry(self, width=10, textvariable=node8_input, validate="focusout", validatecommand=callback8)
            node8.grid(row=4, column=4)
            node8_lbl.grid_forget()
            node8.grid_forget()

            node9_input = tk.StringVar()

            def callback9():
                self.node_list[8] = node9_input.get()
                return True
            node9_lbl = ttk.Label(self, text="Node 9:", font=LARGE_FONT)
            node9_lbl.grid(row=5, column=3)
            node9 = ttk.Entry(self, width=10, textvariable=node9_input, validate="focusout", validatecommand=callback9)
            node9.grid(row=5, column=4)
            node9_lbl.grid_forget()
            node9.grid_forget()

            node10_input = tk.StringVar()

            def callback10():
                self.node_list[9] = node10_input.get()
                return True
            node10_lbl = ttk.Label(self, text="Node 10:", font=LARGE_FONT)
            node10_lbl.grid(row=6, column=3)
            node10 = ttk.Entry(self, width=10, textvariable=node10_input, validate="focusout", validatecommand=callback10)
            node10.grid(row=6, column=4)
            node10_lbl.grid_forget()
            node10.grid_forget()

            info = ttk.Label(self, text="Enter a tuple x,y,z without spaces", font=LARGE_FONT)
            info.grid(row=7, columnspan=4)

        def get_node_list(self):
            return self.node_list

    class AddEdge(tk.Frame):
        edge_list = np.empty(10, dtype='object')
        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)

            button1 = ttk.Button(self, text="Main Menu", command=lambda: controller.show_frame(MainMenu))
            button1.grid(row=0, column=0)

            button2 = ttk.Button(self, text="Add Nodes", command=lambda: controller.show_frame(AddNode))
            button2.grid(row=0, column=1)

            def add_edge():
                # check which edges are visible and make visible the next edge
                if edge2.grid_info() == {}:
                    edge2_lbl.grid(row=3, column=0)
                    edge2.grid(row=3, column=1)
                elif edge3.grid_info() == {}:
                    edge3_lbl.grid(row=4, column=0)
                    edge3.grid(row=4, column=1)
                elif edge4.grid_info() == {}:
                    edge4_lbl.grid(row=5, column=0)
                    edge4.grid(row=5, column=1)
                elif edge5.grid_info() == {}:
                    edge5_lbl.grid(row=6, column=0)
                    edge5.grid(row=6, column=1)
                elif edge6.grid_info() == {}:
                    edge6_lbl.grid(row=2, column=3)
                    edge6.grid(row=2, column=4)
                elif edge7.grid_info() == {}:
                    edge7_lbl.grid(row=3, column=3)
                    edge7.grid(row=3, column=4)
                elif edge8.grid_info() == {}:
                    edge8_lbl.grid(row=4, column=3)
                    edge8.grid(row=4, column=4)
                elif edge9.grid_info() == {}:
                    edge9_lbl.grid(row=5, column=3)
                    edge9.grid(row=5, column=4)
                elif edge10.grid_info() == {}:
                    edge10_lbl.grid(row=6, column=3)
                    edge10.grid(row=6, column=4)
                else:
                    tk.messagebox.showinfo('Add Edge Error', 'The maximum number of edges have been added')

            button3 = ttk.Button(self, text="Add Edge", command=add_edge)
            button3.grid(row=1, column=0)

            def remove_edge():
                # check which edges are visible and make visible the next edge
                if not (edge10.grid_info() == {}):
                    edge10_lbl.grid_forget()
                    edge10.grid_forget()
                elif not (edge9.grid_info() == {}):
                    edge9_lbl.grid_forget()
                    edge9.grid_forget()
                elif not (edge8.grid_info() == {}):
                    edge8_lbl.grid_forget()
                    edge8.grid_forget()
                elif not (edge7.grid_info() == {}):
                    edge7_lbl.grid_forget()
                    edge7.grid_forget()
                elif not (edge6.grid_info() == {}):
                    edge6_lbl.grid_forget()
                    edge6.grid_forget()
                elif not (edge5.grid_info() == {}):
                    edge5_lbl.grid_forget()
                    edge5.grid_forget()
                elif not (edge4.grid_info() == {}):
                    edge4_lbl.grid_forget()
                    edge4.grid_forget()
                elif not (edge3.grid_info() == {}):
                    edge3_lbl.grid_forget()
                    edge3.grid_forget()
                elif not (edge2.grid_info() == {}):
                    edge2_lbl.grid_forget()
                    edge2.grid_forget()

            button4 = ttk.Button(self, text="Remove Edge", command=remove_edge)
            button4.grid(row=1, column=1)

            edge1_input = tk.StringVar()

            def callback1():
                self.edge_list[0] = edge1_input.get()
                return True

            edge1_lbl = ttk.Label(self, text="Edge 1:", font=LARGE_FONT)
            edge1_lbl.grid(row=2, column=0)
            edge1 = ttk.Entry(self, width=10, textvariable=edge1_input, validate="focusout", validatecommand=callback1)
            edge1.grid(row=2, column=1)

            edge2_input = tk.StringVar()

            def callback2():
                self.edge_list[1] = edge2_input.get()
                return True

            edge2_lbl = ttk.Label(self, text="Edge 2:", font=LARGE_FONT)
            edge2_lbl.grid(row=3, column=0)
            edge2 = ttk.Entry(self, width=10, textvariable=edge2_input, validate="focusout", validatecommand=callback2)
            edge2.grid(row=3, column=1)
            edge2_lbl.grid_forget()
            edge2.grid_forget()

            edge3_input = tk.StringVar()

            def callback3():
                self.edge_list[2] = edge3_input.get()
                return True

            edge3_lbl = ttk.Label(self, text="Edge 3:", font=LARGE_FONT)
            edge3_lbl.grid(row=4, column=0)
            edge3 = ttk.Entry(self, width=10, textvariable=edge3_input, validate="focusout", validatecommand=callback3)
            edge3.grid(row=4, column=1)
            edge3_lbl.grid_forget()
            edge3.grid_forget()

            edge4_input = tk.StringVar()

            def callback4():
                self.edge_list[3] = edge4_input.get()
                return True

            edge4_lbl = ttk.Label(self, text="Edge 4:", font=LARGE_FONT)
            edge4_lbl.grid(row=5, column=0)
            edge4 = ttk.Entry(self, width=10, textvariable=edge4_input, validate="focusout", validatecommand=callback4)
            edge4.grid(row=5, column=1)
            edge4_lbl.grid_forget()
            edge4.grid_forget()

            edge5_input = tk.StringVar()

            def callback5():
                self.edge_list[4] = edge5_input.get()
                return True

            edge5_lbl = ttk.Label(self, text="Edge 5:", font=LARGE_FONT)
            edge5_lbl.grid(row=6, column=0)
            edge5 = ttk.Entry(self, width=10, textvariable=edge5_input, validate="focusout", validatecommand=callback5)
            edge5.grid(row=6, column=1)
            edge5_lbl.grid_forget()
            edge5.grid_forget()

            edge6_input = tk.StringVar()

            def callback6():
                self.edge_list[5] = edge6_input.get()
                return True

            edge6_lbl = ttk.Label(self, text="Edge 6:", font=LARGE_FONT)
            edge6_lbl.grid(row=2, column=3)
            edge6 = ttk.Entry(self, width=10, textvariable=edge6_input, validate="focusout", validatecommand=callback6)
            edge6.grid(row=2, column=4)
            edge6_lbl.grid_forget()
            edge6.grid_forget()

            edge7_input = tk.StringVar()

            def callback7():
                self.edge_list[6] = edge7_input.get()
                return True

            edge7_lbl = ttk.Label(self, text="Edge 7:", font=LARGE_FONT)
            edge7_lbl.grid(row=3, column=3)
            edge7 = ttk.Entry(self, width=10, textvariable=edge7_input, validate="focusout", validatecommand=callback7)
            edge7.grid(row=3, column=4)
            edge7_lbl.grid_forget()
            edge7.grid_forget()

            edge8_input = tk.StringVar()

            def callback8():
                self.edge_list[7] = edge8_input.get()
                return True

            edge8_lbl = ttk.Label(self, text="Edge 8:", font=LARGE_FONT)
            edge8_lbl.grid(row=4, column=3)
            edge8 = ttk.Entry(self, width=10, textvariable=edge8_input, validate="focusout", validatecommand=callback8)
            edge8.grid(row=4, column=4)
            edge8_lbl.grid_forget()
            edge8.grid_forget()

            edge9_input = tk.StringVar()

            def callback9():
                self.edge_list[8] = edge9_input.get()
                return True

            edge9_lbl = ttk.Label(self, text="Edge 9:", font=LARGE_FONT)
            edge9_lbl.grid(row=5, column=3)
            edge9 = ttk.Entry(self, width=10, textvariable=edge9_input, validate="focusout", validatecommand=callback9)
            edge9.grid(row=5, column=4)
            edge9_lbl.grid_forget()
            edge9.grid_forget()

            edge10_input = tk.StringVar()

            def callback10():
                self.edge_list[9] = edge10_input.get()
                return True

            edge10_lbl = ttk.Label(self, text="Edge 10:", font=LARGE_FONT)
            edge10_lbl.grid(row=6, column=3)
            edge10 = ttk.Entry(self, width=10, textvariable=edge10_input, validate="focusout",
                               validatecommand=callback10)
            edge10.grid(row=6, column=4)
            edge10_lbl.grid_forget()
            edge10.grid_forget()

            info = ttk.Label(self, text="Enter a tuple node1,node2 without spaces starting from 1 to 10", font=LARGE_FONT)
            info.grid(row=7, columnspan=5)

        def get_edge_list(self):
            return self.edge_list

    app = DWIS()
    app.mainloop()