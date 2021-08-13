"""
Author: Jonathan Gant
Modification Dates: August 10th 2021
General Description: This script contains all the functions used for calculating the voxels contained within a specified fiber geometry

Global Variables: None
List of variables: Refer to individual subroutine documentation.
Common Blocks: None

List of subroutines (functions):
circular_fit - interpolate circle function with given radius to define a curved fiber.
    circle_func - calculates a parametric function of a circle in 3D.
linear_fit - interpolate linear function with a given radius to define a straight fiber.
    linear_eq - calulates a parametric function of a line in 3D.
multi_linear_fit - repeats linear_fit with multiple directions to create two connected straight segments at some angle with each other.
"""

import numpy as np
from .function import rotate, find_angles, find_plane, find_circle

def circular_fit(xsize, ysize, zsize, points, eigenvalues, fiber_radius):
    """
    Interpolate circle function with given radius to define a curved fiber.
    Parameters:
        xsize, ysize, zsize - integers defining the size of the simulation space.
        points - voxels defining the start/mid/end points of the fiber.
        eigenvalues - eigenvalues of diffusion tensor for the fiber.
        fiber_radius - radius/thickness of the fiber.
    Returns:
        fiber_dts - array of diffusion tensors where each entry corresponds to a diffusion tensor for a voxel in the fiber.
    """
    fiber_dts = np.zeros((xsize, ysize, zsize, 3, 3))
    n_steps = int((xsize + ysize + zsize) / 3)
    center, circle_radius, plane_vec, d = find_circle(points)
    v1 = (points[0] - center)/np.linalg.norm(points[0] - center)
    v2 = np.cross(plane_vec, v1)
    v2 = v2 / np.linalg.norm(v2)

    def circle_func(angle):
        """
        Calculates a parametric function of a circle in 3D.
        Parameters:
            angle - angle in radians from starting point to some point on the circle relative to the center of the circle.
        Returns:
            3D coordinates for point on the circle at the given angle.
        """
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

def linear_fit(xsize, ysize, zsize, points, eigenvalues, radius):
    """
    Interpolate linear function with a given radius to define a straight fiber.
    Parameters:
        xsize, ysize, zsize - integers defining the size of the simulation space.
        points - voxels defining the start/end points of the fiber.
        eigenvalues - eigenvalues of diffusion tensor for the fiber.
        radius - radius/thickness of the fiber.
    Returns:
        fiber_dts - array of diffusion tensors where each entry corresponds to a diffusion tensor for a voxel in the fiber.
    """
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
        """
        Calulates a parametric function of a line in 3D.
        Parameters:
            t - parametric variable used to move along the line between the start and end points of the fiber.
        Returns:
            point on the 3D line defined by the fiber start/end points.
        """
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
    """
    Repeats linear_fit with multiple directions to create two connected straight segments at some angle with each other.
    Parameters:
        xsize, ysize, zsize - integers defining the size of the simulation space.
        points - voxels defining the start/mid/end points of the fiber.
        eigenvalues - eigenvalues of diffusion tensor for the fiber.
        radius - radius/thickness of the fiber.
    Returns:
        fiber_dts - array of diffusion tensors where each entry corresponds to a diffusion tensor for a voxel in the fiber. 
    """
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