"""
Author: Jonathan Gant
Modification Dates: August 10th 2021 - Added documentation and split script into helper files which are imported for 
                                       better readability
                    March 7th 2021
General Description: This program creates a simulated diffusion weighted image with complex fiber geometry. To achieve this goal a GUI is created and takes input from the user to define ROIs and fibers. For each fiber direction vectors are calculated and the voxels contained by the fiber are also selected. The diffusion tensors for each voxel are then calculated and the diffusion weighted signal is then calculated and saved as a NIFTI image. The following algorithms are used in this program: 
References:


Global Variables: None
List of variables: Refer to individual subroutine documentation.
Common Blocks: None

List of subroutines (functions):
DWIS - class defining the parents class of the GUI.
    __init__ - constructor for the DWIS class.
    show_frame - shows the frame requested by the controller.
    get_frame - shows specific frame with a given name.
"""

# import statements
import functions.gui
import tkinter as tk


# DWIS class. Parent class of the program. Defines basic setup of the three frames as well as a GUI title and spatial configuration of labels and entry boxes.
class DWIS(tk.Tk):
    def __init__(self, *args, **kwargs):
        """
        Constructor for the DWIS class.
        Parameters:
            self - class level instantiation necessary for accessing class variables.
            *args - command line arguments
            **kwards - key word arguments
        Returns:
            None
        """
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "DWIS")

        # tk.Tk.iconbitmap(self, default="DWIS_icon.ico")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (functions.gui.MainMenu, functions.gui.AddROI, functions.gui.AddFiber):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(functions.gui.MainMenu)

    def show_frame(self, cont):
        """
        Shows the frame requested by the controller.
        Parameters:
            self - class level instantiation necessary for accessing class variables.
            cont - controller which contains info on which frame to show.
        Returns:
            None
        """
        frame = self.frames[cont]
        frame.tkraise()

    def get_frame(self, PageName):
        """
        Parameters:
            self - class level instantiation necessary for accessing class variables.
            PageName - string which refers to a specific frame being called.
        Returns:
            frame - frame specified by PageName.
        """
        frame = self.frames[PageName]
        return frame

# if file is run as the main file then execute the following code allowing for user input in a simple GUI
if __name__ == "__main__":
    app = DWIS()
    app.mainloop()
