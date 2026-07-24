"""
NeuXtalVizModel
--------------
Base model for crystallographic and reciprocal lattice operations using Mantid workspaces.

This class provides methods for handling UB matrices, checking oriented lattices, and computing
various crystallographic and reciprocal axes and transformations for visualization and analysis.

Attributes
----------
UB : np.ndarray or None
    The 3x3 UB matrix for the current workspace, or None if not set.

Methods
-------
has_UB(ws)
    Check if the oriented lattice exists on a workspace.
set_UB(UB)
    Update the UB-matrix.
get_oriented_lattice_parameters()
    Obtain the oriented lattice parameters (a, b, c, alpha, beta, gamma, u, v).
orientation_matrix()
    Return the current UB matrix (orientation matrix).
get_transform(reciprocal)
    Transformation matrix describing the reciprocal or crystal axes.
ab_star_axes(), bc_star_axes(), ca_star_axes()
    Cartesian camera/upward view vectors for various axes.
ab_axes(), bc_axes(), ca_axes()
    Cartesian camera/upward view vectors for reciprocal axes.
get_vector(axes_type, ind)
    Vector corresponding to a particular crystallographic direction.
"""

from importlib.resources import files
from importlib.resources import files
import os

from mantid.simpleapi import HasUB
from mantid.geometry import OrientedLattice

import logging
from mantid.utils.logging import log_to_python

log_to_python(level="notice")
log = logging.getLogger("Mantid")

import numpy as np
import scipy.spatial.transform


class NeuXtalVizModel:
    def __init__(self):
        """
        Initialize the NeuXtalVizModel with no UB matrix.
        """

        self.UB = None

    def files_exist(self, files):
        """
        Check if the specified files exist and are readable.

        Parameters
        ----------
        files : list of str
            List of file paths to check.

        Returns
        -------
        exist : bool
            True if all files exist and are readable, False otherwise.

        """

        return [
            f for f in files if os.path.isfile(f) and os.access(f, os.R_OK)
        ]

    def has_UB(self, ws):
        """
        Check if the oriented lattice exists on a workspace.

        Parameters
        ----------
        ws : str
            Name of workspace.

        Returns
        -------
        ol : bool
            True if oriented lattice exists, False otherwise.

        """

        ol = HasUB(Workspace=ws)

        return ol

    def set_UB(self, UB):
        """
        Update the UB-matrix.

        Parameters
        ----------
        UB : 3x3 element 2d array
            UB-matrix to set for the model.

        """

        self.UB = UB.copy()

    def get_oriented_lattice_parameters(self):
        """
        Obtain the oriented lattice parameters.

        Returns
        -------
        a, b, c : float
            Lattice constants.
        alpha, beta, gamma : float
            Lattice angles.
        u, v : np.ndarray
            Normalized u and v vectors.

        """

        if self.UB is not None:
            ol = OrientedLattice()
            ol.setUB(self.UB)

            params = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

            u, v = ol.getuVector(), ol.getvVector()

            u = np.array(u) / np.abs(u).max()
            v = np.array(v) / np.abs(v).max()

            return *params, u, v

    def orientation_matrix(self):
        """
        Return the current UB matrix (orientation matrix).

        Returns
        -------
        np.ndarray
            The UB matrix.

        """

        return self.UB.copy()

    def get_transform(self, reciprocal=True):
        """
        Transformation matrix describing the reciprocal or crystal axes.

        Parameters
        ----------
        reciprocal : bool, optional
            Option for the reciprocal (True) or crystal lattice axes (False).
            Default is True.

        Returns
        -------
        T : 3x3 element 2d array
            Normalized transformation matrix.

        """

        if self.UB is not None:
            if reciprocal:
                T = self.orientation_matrix()
            else:
                T = np.column_stack(
                    [
                        np.cross(self.UB[:, 1], self.UB[:, 2]),
                        np.cross(self.UB[:, 2], self.UB[:, 0]),
                        np.cross(self.UB[:, 0], self.UB[:, 1]),
                    ]
                )

            return T / np.linalg.norm(T, axis=0)

    def _reciprocal_axes(self):
        """
        :math:`a^\ast`/:math:`b^\ast`/:math:`c^\ast` in cartesian coordinates.

        Shared building block for the six ``*_axes``/``*_star_axes``
        camera-direction methods below, so their up-vector priority
        only needs to be encoded once: prefer :math:`c`/:math:`c^\ast`
        as "up", falling back to :math:`b`/:math:`b^\ast` only when
        :math:`c`/:math:`c^\ast` itself is the view direction --
        :math:`a`/:math:`a^\ast` is never used as "up".

        Returns
        -------
        a_star, b_star, c_star : 3 element 1d array
            Cartesian directions of the reciprocal axes.
        """

        M = self.orientation_matrix()

        return np.dot(M, [1, 0, 0]), np.dot(M, [0, 1, 0]), np.dot(M, [0, 0, 1])

    def _real_axes(self):
        """
        :math:`a`/:math:`b`/:math:`c` in cartesian coordinates.

        See :meth:`_reciprocal_axes` -- shared building block for the
        real-space camera-direction methods below.

        Returns
        -------
        a, b, c : 3 element 1d array
            Cartesian directions of the real-space axes.
        """

        a_star, b_star, c_star = self._reciprocal_axes()

        return (
            np.cross(b_star, c_star),
            np.cross(c_star, a_star),
            np.cross(a_star, b_star),
        )

    def ab_star_axes(self):
        """
        :math:`c^\ast`-direction in cartesian coordinates.

        Returns
        -------
        camera : 3 element 1d array
            Cartesian camera view vector.
        upward : 3 element 1d array
            Cartesian upward view vector.

        """

        if self.UB is not None:
            a_star, b_star, c_star = self._reciprocal_axes()
            return c_star, b_star

    def bc_star_axes(self):
        """
        :math:`a^\ast`-direction in cartesian coordinates.

        Returns
        -------
        camera : 3 element 1d array
            Cartesian camera view vector.
        upward : 3 element 1d array
            Cartesian upward view vector.

        """

        if self.UB is not None:
            a_star, b_star, c_star = self._reciprocal_axes()
            return a_star, c_star

    def ca_star_axes(self):
        """
        :math:`b^\ast`-direction in cartesian coordinates.

        Returns
        -------
        camera : 3 element 1d array
            Cartesian camera view vector.
        upward : 3 element 1d array
            Cartesian upward view vector.

        """

        if self.UB is not None:
            a_star, b_star, c_star = self._reciprocal_axes()
            return b_star, c_star

    def ab_axes(self):
        """
        :math:`c`-direction in cartesian coordinates.

        Returns
        -------
        camera : 3 element 1d array
            Cartesian camera view vector.
        upward : 3 element 1d array
            Cartesian upward view vector.

        """

        if self.UB is not None:
            a, b, c = self._real_axes()
            return c, b

    def bc_axes(self):
        """
        :math:`a`-direction in cartesian coordinates.

        Returns
        -------
        camera : 3 element 1d array
            Cartesian camera view vector.
        upward : 3 element 1d array
            Cartesian upward view vector.

        """

        if self.UB is not None:
            a, b, c = self._real_axes()
            return a, c

    def ca_axes(self):
        """
        :math:`b`-direction in cartesian coordinates.

        Returns
        -------
        camera : 3 element 1d array
            Cartesian camera view vector.
        upward : 3 element 1d array
            Cartesian upward view vector.

        """

        if self.UB is not None:
            a, b, c = self._real_axes()
            return b, c

    def get_vector(self, axes_type, ind):
        """
        Vector corresponding to a particular crystallographic direction.

        Parameters
        ----------
        axes_type : str, [hkl] or [uvw]
            Miller index or fractional coordinate.
        ind : 3-element 1d array-like
            Indices.

        Returns
        -------
        vec : 3 element 1d array
            Cartesian vector.

        """

        if self.UB is not None:
            if axes_type == "[hkl]":
                matrix = self.orientation_matrix()
            else:
                matrix = np.cross(
                    np.dot(
                        self.orientation_matrix(), np.roll(np.eye(3), 2, 1)
                    ).T,
                    np.dot(
                        self.orientation_matrix(), np.roll(np.eye(3), 1, 1)
                    ).T,
                ).T

            vec = np.dot(matrix, ind)

            return vec

    # ------------------------------------------------------------------
    # Camera rotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _camera_vectors_from_state(position, focal_point, up):
        """Compute an orthonormal camera basis from raw position, focal point, and up.

        Parameters
        ----------
        position : array-like, shape (3,)
            Camera position in world coordinates.
        focal_point : array-like, shape (3,)
            Camera focal point in world coordinates.
        up : array-like, shape (3,)
            Camera up vector in world coordinates.

        Returns
        -------
        fp : ndarray, shape (3,)
            Focal point.
        d : ndarray, shape (3,)
            Normalized viewing direction (from position to focal point).
        up_ortho : ndarray, shape (3,)
            Normalized up vector orthogonal to d.
        dist : float
            Distance from position to focal point.
        """

        pos = np.array(position, dtype=float)
        fp = np.array(focal_point, dtype=float)
        up_vec = np.array(up, dtype=float)

        d = fp - pos
        if np.linalg.norm(d) == 0.0:
            d = np.array([0.0, 0.0, 1.0])
        d = d / np.linalg.norm(d)

        if np.linalg.norm(up_vec) == 0.0:
            up_vec = np.array([0.0, 1.0, 0.0])
        # Make up orthogonal to direction
        up_vec = up_vec - np.dot(up_vec, d) * d
        if np.linalg.norm(up_vec) == 0.0:
            up_vec = np.array([0.0, 1.0, 0.0])
            up_vec = up_vec - np.dot(up_vec, d) * d
        up_vec = up_vec / np.linalg.norm(up_vec)

        dist = np.linalg.norm(fp - pos)
        if dist == 0.0:
            dist = 1.0

        return fp, d, up_vec, dist

    @staticmethod
    def _rotate_vector(v, axis, angle_deg):
        """Rotate vector ``v`` around ``axis`` by ``angle_deg`` degrees.

        Uses Rodrigues' rotation formula. Both ``v`` and ``axis`` are treated
        as 3D vectors.
        """

        v = np.array(v, dtype=float)
        axis = np.array(axis, dtype=float)

        if np.linalg.norm(axis) == 0.0:
            return v

        axis = axis / np.linalg.norm(axis)

        theta = np.deg2rad(angle_deg)
        k = axis

        return (
            v * np.cos(theta)
            + np.cross(k, v) * np.sin(theta)
            + k * np.dot(k, v) * (1.0 - np.cos(theta))
        )

    @staticmethod
    def _state_from_basis(focal_point, direction, up, dist):
        """Reconstruct camera state from focal point, direction, up, and distance.

        Parameters
        ----------
        focal_point : array-like, shape (3,)
            Camera focal point.
        direction : array-like, shape (3,)
            Approximate viewing direction.
        up : array-like, shape (3,)
            Approximate up vector.
        dist : float
            Distance from camera position to focal point.

        Returns
        -------
        position : ndarray, shape (3,)
            Camera position.
        fp : ndarray, shape (3,)
            Focal point (same as input, but as ndarray).
        up_ortho : ndarray, shape (3,)
            Normalized, orthogonal up vector.
        """

        fp = np.array(focal_point, dtype=float)
        d = np.array(direction, dtype=float)
        up_vec = np.array(up, dtype=float)

        if np.linalg.norm(d) == 0.0:
            d = np.array([0.0, 0.0, 1.0])
        d = d / np.linalg.norm(d)

        if np.linalg.norm(up_vec) == 0.0:
            up_vec = np.array([0.0, 1.0, 0.0])
        up_vec = up_vec - np.dot(up_vec, d) * d
        if np.linalg.norm(up_vec) == 0.0:
            up_vec = np.array([0.0, 1.0, 0.0])
            up_vec = up_vec - np.dot(up_vec, d) * d
        up_vec = up_vec / np.linalg.norm(up_vec)

        position = fp - d * float(dist)

        return position, fp, up_vec

    def rotate_camera_roll(self, position, focal_point, up, angle_deg):
        """Apply a roll about the current viewing axis.

        Parameters
        ----------
        position, focal_point, up : array-like
            Current camera position, focal point, and up vector.
        angle_deg : float
            Roll angle in degrees (positive is counter‑clockwise).

        Returns
        -------
        new_position, new_focal_point, new_up : ndarray
            Updated camera state after applying the roll.
        """

        fp, d, up_vec, dist = self._camera_vectors_from_state(
            position, focal_point, up
        )
        new_up = self._rotate_vector(up_vec, d, angle_deg)
        new_position, new_fp, new_up = self._state_from_basis(
            fp, d, new_up, dist
        )
        return new_position, new_fp, new_up

    def rotate_camera_elevation(self, position, focal_point, up, angle_deg):
        """Tilt the camera up/down about its right axis.

        Parameters
        ----------
        position, focal_point, up : array-like
            Current camera position, focal point, and up vector.
        angle_deg : float
            Elevation angle in degrees (positive tilts up).

        Returns
        -------
        new_position, new_focal_point, new_up : ndarray
            Updated camera state after applying the elevation.
        """

        fp, d, up_vec, dist = self._camera_vectors_from_state(
            position, focal_point, up
        )

        right = np.cross(d, up_vec)
        if np.linalg.norm(right) == 0.0:
            right = np.array([1.0, 0.0, 0.0])

        new_d = self._rotate_vector(d, right, angle_deg)
        new_up = self._rotate_vector(up_vec, right, angle_deg)
        new_position, new_fp, new_up = self._state_from_basis(
            fp, new_d, new_up, dist
        )
        return new_position, new_fp, new_up

    def rotate_camera_azimuth(self, position, focal_point, up, angle_deg):
        """Rotate the camera left/right about a global up axis.

        Parameters
        ----------
        position, focal_point, up : array-like
            Current camera position, focal point, and up vector.
        angle_deg : float
            Azimuth angle in degrees (positive rotates to the right).

        Returns
        -------
        new_position, new_focal_point, new_up : ndarray
            Updated camera state after applying the azimuth.
        """

        fp, d, up_vec, dist = self._camera_vectors_from_state(
            position, focal_point, up
        )

        world_up = np.array([0.0, 0.0, 1.0])

        new_d = self._rotate_vector(d, world_up, angle_deg)
        new_up = self._rotate_vector(up_vec, world_up, angle_deg)
        new_position, new_fp, new_up = self._state_from_basis(
            fp, new_d, new_up, dist
        )
        return new_position, new_fp, new_up

    def calculate_camera_parameters(self, direction, roll):
        """Calculate roll, elevation, and azimuth from camera direction and roll.

        Parameters
        ----------
        direction : array-like, shape (3,)
            Camera viewing direction.
        roll : float
            Camera roll angle in degrees.

        Returns
        -------
        roll, elevation, azimuth : float
            Camera roll, elevation, and azimuth angles in degrees.
        """

        direction = np.array(direction, dtype=float)
        if np.linalg.norm(direction) == 0.0:
            direction = np.array([0.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)

        elevation = np.rad2deg(np.arcsin(direction[2]))
        azimuth = np.rad2deg(np.arctan2(direction[0], direction[1]))

        return roll, elevation, azimuth
