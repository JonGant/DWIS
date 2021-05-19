# DWIS
Diffusion Weighted Image Simulator

This program simulates diffusion weighted images based on user defined ROIs and fiber geometries representative of white matter fiber bundles in the brain. To run the program, simply execute the python file DWIS_v9.py. Folders containing the simulated images and simulation parameters will automatically be created once a simulation is run. The folder qspace contains the code used to generate the set of b-vectors based on simulated "electrostatic" repulsion.

Package Dependencies: scipy, nibabel, numpy, sympy

Installation Instructions:
Download package dependencies

```
pip install -r requirements.txt
```

Using Python 3 run

```
python DWIS_v9.py
```

Code written by Jonathan Gant at the University of Florida
