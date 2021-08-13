"""
Author: Jonathan Gant
Modification Dates: August 10th 2021
General Description: This script contains the classes defining the tkinter GUI for the image simulator.
References:

Global Variables: None
List of variables: Refer to individual subroutine documentation.
Common Blocks: None

List of subroutines (functions):
MainMenu - class defining the main menu frame.
    __init__ - constructor for the MainMenu class. Implements label & entry initialization/placement.
        get_roi_info - gets AddROI frame and runs get_roi_list which reads out the inputted info for use in the start_sim function. Used in save_func.
        set_roi_info - gets AddROI frame and runs set_roi_list which initializes input from a text file saved previously. Used in load_func.
        get_fiber_info - gets AddFiber frame and runs get_fiber_list which retreivres the inputted information on the fibers for use in the start_sim function. Used in save_func.
        set_fiber_info - gets AddFibr frame and runs set_fiber_list which initialzes input from a text file saved previously. Used in load_func.
        start_sim - reformats user inputed strings into appropriate numpy arrays for use in the back end of the simulation. Additionally, there are many checks on the user input to ensure
                    the input is correct. This functions uses create_dt_data and simulate_dwi_calc to run the image simulation. 
        save_func - saves the user inputted data to a text file with the same file name as the filename input box on the main menu. Automatically called by start_sim so that every
                    simulations parameters are saved.
        load_func - loads simulation parameters (general, ROIs, fibers) from a text file.
AddROI - class defining the add ROI frame.
    __init__ - constructor for the AddROI class. Implements label & entry initialization/placement.
        set_cuboid - sets the ROI type fo cuboid and changes the entry boxes to match the type of ROI.
            func - implements setting the type to cuboid by adding and removing entry boxes as necessary. Also darkens the button for the ROI type chosen.
        set_sphere - sets the ROI type fo spherical and changes the entry boxes to match the type of ROI. 
            func - implements setting the type to spherical by adding and removing entry boxes as necessary. Also darkens the button for the ROI type chosen.
        set_manual - sets the ROI type fo manual and changes the entry boxes to match the type of ROI.
            func - implements setting the type to manual by adding and removing entry boxes as necessary. Also darkens the button for the ROI type chosen.
        add_roi - reveals another ROI entryline for user input
        remove_roi - hides the highest numbered ROI and deletes all the ROI parameters the user inputted for that ROI.
        clear_all - hides all entrylines (except ROI 1) and deletes all user entered ROI information.
    get_roi_list - acessor which reads back all the user enetered ROI info. Used by save_func.
    set_roi_list - sets the values of the ROIs to the parameters loaded from a text file. Used by load_func.
AddFiber - class defining the add fiber frame.
    __init__ - constructor for the AddFiber class. Implements label & entry initialization/placement.
        add_fiber - reveals another fiber entryline for user input 
        remove_fiber - hides the highest numbered fiber entryline and deletes all the fiber parameters the user inputted for that fiber. 
        clear_all - hides all entrylines (except fiber 1) and deletes all user entered fiber information.
    get_fiber_list - acessor which reads back all the user enetered fiber info. Used by save_func. 
    set_fiber_list - sets the values of the fiber entrylines to the parameters loaded from a text file. Used by load_func.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import os
from functions.diffusion import create_dt_data, simulate_dwi_calc


# Main Menu class. Contains general parameter input boxes. Buttons to switch to the AddROI and AddFiber menu as well as
# start the simulation. Implements a save/load function as well for quickly loading previous simulations.
class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        """
        Constructor for the MainMenu class. Implements label & entry initialization/placement.
        Parameters:
            parent - tkinter container frame inherited by the MainMenu frame.
            controller - the original self from the parent class DWIS.
        Returns:
            None
        """
        tk.Frame.__init__(self, parent)
        
        LARGE_FONT = ("Arial", 12)

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
            """
            Gets AddROI frame and runs get_roi_list which reads out the inputted info for use in the start_sim function. Used in save_func.
            Parameters:
                None
            Returns:
                Array of entries from user input with each entry containing an array of parameters necessary for computing the voxels defined by the user enetered ROI parameters.
            """
            AddROIFrame = controller.get_frame(AddROI)
            return AddROIFrame.get_roi_list()

        def set_roi_info(roi_list):
            """
            Gets AddROI frame and runs set_roi_list which initializes input from a text file saved previously. Used in load_func.
            Parameters:
                roi_list - array of entry information with each entry containing an array of parameters necessary for computing the voxels defined for each ROI the user specified.
            Returns:
                None
            """
            AddROIFrame = controller.get_frame(AddROI)
            return AddROIFrame.set_roi_list(roi_list)

        # get fiber info
        def get_fiber_info():
            """
            Gets AddFiber frame and runs get_fiber_list which retreivres the inputted information on the fibers for use in the start_sim function. Used in save_func.
            Parameters:
                None
            Returns:
               Array of entries from user input with each entry containing an array of parameters necessary for computing the voxels defined by the user enetered fiber parameters.
            """
            AddfiberFrame = controller.get_frame(AddFiber)
            return AddfiberFrame.get_fiber_list()

        def set_fiber_info(fiber_list):
            """
            Gets AddFibr frame and runs set_fiber_list which initialzes input from a text file saved previously. Used in load_func. 
            Parameters:
                fiber_list - array of entry information with each entry containing an array of parameters necessary for computing the voxels defined for each fiber the user specified. 
            Returns:
                None
            """
            AddfiberFrame = controller.get_frame(AddFiber)
            return AddfiberFrame.set_fiber_list(fiber_list)

        # start simulation command
        def start_sim():
            """
            Reformats user inputed strings into appropriate numpy arrays for use in the back end of the simulation. Additionally, there are many checks on the user input to ensure the input is correct. This functions uses create_dt_data and simulate_dwi_calc to run the image simulation. 
            Parameters:
                None
            Returns:
                None
            """
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
            """
            Saves the user inputted data to a text file with the same file name as the filename input box on the main menu. Automatically called by start_sim so that every simnulations parameters are saved.
            Parameters:
                None
            Returns:
                True
            """
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
            """
            Loads simulation parameters (general, ROIs, fibers) from a text file.
            Parameters:
                None
            Returns:
                None
            """
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


# Add ROI class. Contains entry boxes for ROI parameters, buttons to switch back the Main Menu or Add Fiber frame, and
# accessor/mutator functions which allow for the data to be read into other subroutines or changed using save_func &
# load_func.
class AddROI(tk.Frame):
    # max number of ROIs
    num_rois = 60
    def __init__(self, parent, controller):
        """
        Constructor for the AddROI class. Implements label & entry initialization/placement.
        Parameters:
            parent - tkinter container frame inherited by the AddROI frame.
            controller - the original self from the parent class DWIS.
        Returns:
            None
        """
        tk.Frame.__init__(self, parent)

        LARGE_FONT = ("Arial", 12)

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
            """
            Sets the ROI type fo cuboid and changes the entry boxes to match the type of ROI.
            Parameters:
                i - index of the ROI number where the switch in ROI type is occuring.
            Returns:
                func - implements setting the type to cuboid by adding and removing entry boxes as necessary. Also darkens the button for the ROI type chosen. 
            """
            def func():
                """
                Implements setting the type to cuboid by adding and removing entry boxes as necessary. Also darkens the button for the ROI type chosen.
                Parameters:
                    None
                Returns:
                    None
                """
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
            """
            Sets the ROI type fo spherical and changes the entry boxes to match the type of ROI. 
            Parameters:
                i - index of the ROI number where the switch in ROI type is occuring.
            Returns:
                func - implements setting the type to spherical by adding and removing entry boxes as necessary. Also darkens the button for the ROI type chosen. 
            """
            def func():
                """
                Implements setting the type to spherical by adding and removing entry boxes as necessary. Also darkens the button for the ROI type chosen.
                Parameters:
                    None
                Returns:
                    None
                """
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
            """
            Sets the ROI type fo manual and changes the entry boxes to match the type of ROI.
            Parameters:
                i - index of the ROI number where the switch in ROI type is occuring. 
            Returns:
                func - implements setting the type to manual by adding and removing entry boxes as necessary. Also darkens the button for the ROI type chosen.
            """
            def func():
                """
                Implements setting the type to manual by adding and removing entry boxes as necessary. Also darkens the button for the ROI type chosen.
                Parameters:
                    None
                Returns:
                    None
                """
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
            """
            Reveals another ROI entryline for user input.
            Parameters:
                None
            Returns:
                None
            """
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
            """
            Hides the highest numbered ROI and deletes all the ROI parameters the user inputted for that ROI.
            Parameters:
                None
            Returns:
                None
            """
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
            """
            Hides all entrylines (except ROI 1) and deletes all user entered ROI information.
            Parameters:
                None
            Returns:
                None
            """
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
        """
        Accessor which reads back all the user enetered ROI info. Used by save_func.
        Parameters:
            self - class level instantiation necessary for accessing class variables.
        Returns:
            roi_list - Array of entries from user input with each entry containing an array of parameters necessary for computing the voxels defined by the user enetered ROI parameters.
        """
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
        """
        Sets the values of the ROIs to the parameters loaded from a text file. Used by load_func.
        Parameters:
            self - class level instantiation necessary for accessing class variables. 
            roi_info - Array of entries from input text file with each entry containing an array of parameters necessary for computing the voxels defined by the user enetered ROI parameters.
        Returns:
            None
        """
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


# Add Fiber class. Contains entry boxes for fiber parameters, buttons to switch back the Main Menu or Add ROI frame, and
# accessor/mutator functions which allow for the data to be read into other subroutines or changed using save_func &
# load_func.
class AddFiber(tk.Frame):
    # max number of fibers
    num_fibers = 30
    def __init__(self, parent, controller):
        """
        Constructor for the AddFiber class. Implements label & entry initialization/placement.
        Parameters:
            parent - tkinter container frame inherited by the MainMenu frame.
            controller - the original self from the parent class DWIS.
        Returns:
            None
        """ 
        tk.Frame.__init__(self, parent)

        LARGE_FONT = ("Arial", 12)

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
            """
            Reveals another fiber entryline for user input.
            Parameters:
                None
            Returns:
                None
            """
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
            """
            Hides the highest numbered fiber entryline and deletes all the fiber parameters the user inputted for that fiber.
            Parameters:
                None
            Returns:
                None
            """
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
            """
            Hides all entrylines (except fiber 1) and deletes all user entered fiber information.
            Parameters:
                None
            Returns:
                None
            """
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
        info2 = ttk.Label(self, width=110, text="In the third box enter enter a float for the radius of the "
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
        """
        Acessor which reads back all the user enetered fiber info. Used by save_func.
        Parameters:
            self - class level instantiation necessary for accessing class variables. 
        Returns:
            fiber_list - Array of entries from user input with each entry containing an array of parameters necessary for computing the voxels defined by the user enetered fiber parameters. 
        """
        fiber_list = np.empty((self.num_fibers, 5), dtype='object')
        for i in range(len(self.fibers)):
            fiber_list[i][0] = self.fibers[i].get()
            fiber_list[i][1] = self.fiber_dirs[i].get()
            fiber_list[i][2] = self.fiber_radii[i].get()
            fiber_list[i][3] = self.fiber_voxels[i].get()
            fiber_list[i][4] = self.fit_type[i].get()
        return fiber_list

    def set_fiber_list(self, fiber_info):
        """
        Sets the values of the fiber entrylines to the parameters loaded from a text file. Used by load_func.
        Parameters:
            self - class level instantiation necessary for accessing class variables. 
            fiber_info - Array of entries from input text file with each entry containing an array of parameters necessary for computing the voxels defined by the user enetered fiber parameters.
        Returns:
            None
        """
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