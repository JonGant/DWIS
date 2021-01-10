# DWIS
Diffusion Weighted Image Simulator

This program simulates diffusion weighted images. The user is constrained to the creation of 6 nodes or ROIs and 3 edges which are representative of white matter fiber bundles in the brain. To run the program, simply execute the python file DWIS_v5.py. Folders containing the simulated images and simulation parameters will automatically be created once a simulation is run. The folder qspace contains the code used to determine the set of b-vectors based on simulated "electrostatic" repulsion.

Previous versions of the program are contained in the folder zzz_old_versions.

Package Dependencies: scipy, nibabel, numpy, tkinter, os
