# This script calculates the edge weight introduced in Colon-Perez et al. 2015
# References: Colon-Perez LM, Spindler C, Goicochea S, Triplett W, Parekh M, Montie E, et al. (2015) Dimensionless,
#             Scale Invariant, fiber Weight Metric for the Study of Complex Structural Networks.
#             PLoS ONE 10(7): e0131493. https://doi.org/10.1371/journal.pone.0131493

import numpy as np
import nibabel as nib
import glob

for fname in list(glob.glob('nifti_images/jacks_wb_sn.txt')):
    res = 2
    den = 5
    if fname[39:-10] != "12" or fname[39:-10] != "8" or fname[39:-10] != "4":
        # load simulation parameters
        file = open("simulation_parameters/" + fname[13:-10] + ".txt", "r")

        # clear extraneous file info
        for i in range(9):
            file.readline()

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
        file.close()

        sa = np.zeros(numrois)
        for i in range(len(roi_info)):
            if roi_info[i][0] == "cuboid":
                voxel = roi_info[i][2].split(sep=', ')
                for j in range(len(voxel)):
                    if '(' in voxel[j]:
                        voxel[j] = voxel[j][1:]
                    if ')' in voxel[j]:
                        voxel[j] = voxel[j][:len(voxel[j]) - 1]
                sizes = np.array([int(voxel[0]), int(voxel[1]), int(voxel[2])])
                sa[i] = res**2 * (2 * sizes[0] * sizes[1] + 2 * sizes[0] * sizes[2] + 2 * sizes[1] * sizes[2])
            elif roi_info[i][0] == "sphere":
                radius = float(roi_info[i][2])
                sa[i] = 4 * np.pi * (radius ** 2) * res**2
            elif roi_info[i][0] == "manual":
                print("not implemented")
        print(sa)
        adj_mat = np.zeros((numrois, numrois))
        # load track file and hyperparameters
        for exemplar in list(glob.glob(fname[:-10] + "_exemplars/*")):
            node1 = int(exemplar[-7]) - 1
            node2 = int(exemplar[-5]) - 1
            if adj_mat[node1, node2] == 0 or adj_mat[node2, node1] == 0:
                tracks = nib.streamlines.load(exemplar, lazy_load=True)
                gen = iter(tracks.tractogram.streamlines)
                sum = 0
                for streamline in gen:
                    length = 0
                    for i in range(len(streamline) - 1):
                        length += np.linalg.norm(streamline[i + 1] - streamline[i])
                    sum += (1 / length)
                print(sum)
                # edge weight calculation
                adj_mat[node1, node2] = ((res**3)/(den**3)) * (2 / (sa[node1] + sa[node2])) * sum
                adj_mat[node2, node1] = adj_mat[node1, node2]
        np.savetxt(fname[:-10] + "_cp_ew.txt", adj_mat, delimiter=" ")

# 2d slant
# 0.02718881786678025 d=5
# 0.029039656844674078 d=10
# 0.02901829472096349 d=20