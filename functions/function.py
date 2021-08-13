"""
Author: Jonathan Gant
Modification Dates: August 10th 2021
General Description: This script contains helper functions which are used to calculate the diffusion tensors based on the fiber geometries.
References:
Caruyer, Emmanuel, Christophe Lenglet, Guillermo Sapiro, and Rachid Deriche.
    "Design of multishell sampling schemes with uniform coverage in diffusion MRI."
    Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-1540.

Global Variables:
List of variables:
Common Blocks:

List of subroutines (functions):
"""

import numpy as np
from sympy import solve, symbols, Eq

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