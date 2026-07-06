from mantid.simpleapi import (
    CreateSingleValuedWorkspace,
    SetSample,
    SetGoniometer,
    LoadIsawUB,
    mtd,
)

import numpy as np
import scipy.spatial

from NeuXtalViz.models.base_model import NeuXtalVizModel


class SampleModel(NeuXtalVizModel):
    """
    Model for defining and characterizing a sample for absorption corrections.

    Wraps a single-valued Mantid workspace (``"sample"``) used to hold the
    oriented lattice, goniometer, shape, and material information needed to
    compute absorption/scattering parameters and to render the sample mesh.

    """

    def __init__(self):
        """
        Initialize the sample model and create the underlying workspace.
        """

        super(SampleModel, self).__init__()

        CreateSingleValuedWorkspace(OutputWorkspace="sample")

    def load_UB(self, filename):
        """
        Load a UB matrix from an ISAW UB file and update the model.

        Parameters
        ----------
        filename : str
            Path to the ISAW UB matrix file.

        """

        LoadIsawUB(InputWorkspace="sample", Filename=filename)

        UB = mtd["sample"].sample().getOrientedLattice().getUB().copy()

        self.set_UB(UB)

    def get_volume(self):
        """
        Return the unit cell volume of the oriented lattice, if set.

        Returns
        -------
        volume : float
            Unit cell volume in Angstrom^3, or None if no UB matrix is set.

        """

        if self.has_UB("sample"):
            return mtd["sample"].sample().getOrientedLattice().volume()

    def get_euler_angles(self, u_vector, v_vector):
        """
        Compute Euler angles that orient a sample face along given directions.

        The `u_vector` and `v_vector` are Miller index / fractional
        coordinate directions that are transformed to Cartesian coordinates
        via the UB matrix to build a rotation matrix, which is then
        decomposed into intrinsic ZYX Euler angles.

        Parameters
        ----------
        u_vector : 3-element 1d array-like
            Crystallographic direction to align with the sample's primary
            (height/thickness) axis.
        v_vector : 3-element 1d array-like
            Crystallographic direction used together with `u_vector` to
            define the orientation plane.

        Returns
        -------
        alpha, beta, gamma : float
            Rotation angles in degrees about the X, Y, and Z axes
            respectively, or None if no UB matrix is set or the vectors
            are collinear.

        """

        w_vector = np.cross(u_vector, v_vector)

        if self.UB is not None and np.linalg.norm(w_vector) > 0:
            u = np.dot(self.UB, u_vector)
            v = np.dot(self.UB, v_vector)

            u /= np.linalg.norm(u)

            w = np.cross(u, v)
            w /= np.linalg.norm(w)

            v = np.cross(w, u)

            T = np.column_stack([v, w, u])
            R = scipy.spatial.transform.Rotation.from_matrix(T)

            gamma, beta, alpha = R.as_euler("ZYX", degrees=True)

            return alpha, beta, gamma

    def get_shape_dict(self, shape, params, alpha=0, beta=0, gamma=0):
        """
        Build a Mantid CSG shape dictionary describing the sample geometry.

        Dimensions in `params` are given in the view's display units
        (mm for sphere/cylinder radius, cm for cuboid width/height/depth)
        and are converted to meters for the generated XML.

        Parameters
        ----------
        shape : str
            Sample shape name: ``"Sphere"``, ``"Cylinder"``, or any other
            value (treated as ``"Plate"``/cuboid).
        params : list of float
            Shape dimensions. For ``"Sphere"``, ``params[0]`` is the
            diameter (mm). For ``"Cylinder"``, ``params[0]`` is the
            diameter (mm) and ``params[1]`` is the height (cm). Otherwise,
            ``params[0:3]`` are the width, height, and depth (cm).
        alpha : float, optional
            Rotation about the x-axis in degrees. Default is 0.
        beta : float, optional
            Rotation about the y-axis in degrees. Default is 0.
        gamma : float, optional
            Rotation about the z-axis in degrees. Default is 0.

        Returns
        -------
        shape_dict : dict
            Dictionary with keys ``"Shape"`` (``"CSG"``) and ``"Value"``
            (the CSG XML string), suitable for the ``Geometry`` argument
            of Mantid's ``SetSample`` algorithm.

        """

        if shape == "Sphere":
            radius = params[0] / 200
            shape = ' \
            <sphere id="sphere"> \
              <radius val="{}" /> \
              <centre x="0.0" y="0.0" z="0.0" /> \
              <rotate x="{}" y="{}" z="{}" /> \
            </sphere> \
            '.format(
                radius, alpha, beta, gamma
            )
        elif shape == "Cylinder":
            radius, height = params[0] / 200, params[1] / 100
            shape = ' \
            <cylinder id="cylinder"> \
              <centre-of-bottom-base x="0.0" y="{}" z="0.0" /> \
              <axis x="0.0" y="1.0" z="0" /> \
              <radius val="{}" /> \
              <height val="{}" /> \
              <rotate x="{}" y="{}" z="{}" /> \
            </cylinder> \
            '.format(
                -height / 2, radius, height, alpha, beta, gamma
            )
        else:
            width, height, depth = (
                params[0] / 100,
                params[1] / 100,
                params[2] / 100,
            )
            shape = ' \
            <cuboid id="cuboid"> \
              <width val="{}" /> \
              <height val="{}" /> \
              <depth val="{}" /> \
              <centre x="0.0" y="0.0" z="0.0" /> \
              <rotate x="{}" y="{}" z="{}" /> \
            </cuboid> \
            '.format(
                width, height, depth, alpha, beta, gamma
            )

        return {"Shape": "CSG", "Value": shape}

    def get_material_dict(self, chemical_formula, z_parameter, volume):
        """
        Build a Mantid material dictionary for the sample.

        Parameters
        ----------
        chemical_formula : str
            Chemical formula of the sample material.
        z_parameter : float
            Number of formula units per unit cell.
        volume : float
            Unit cell volume in Angstrom^3.

        Returns
        -------
        mat_dict : dict
            Dictionary with keys ``"ChemicalFormula"``, ``"ZParameter"``,
            and ``"UnitCellVolume"``, suitable for the ``Material``
            argument of Mantid's ``SetSample`` algorithm.

        """

        mat_dict = {
            "ChemicalFormula": chemical_formula,
            "ZParameter": z_parameter,
            "UnitCellVolume": volume,
        }

        return mat_dict

    def get_goniometer_strings(self, goniometers):
        """
        Format goniometer axis definitions for Mantid's ``SetGoniometer``.

        Only axes with a nonzero rotation vector are kept, and exactly
        three such axes are required for a valid result.

        Parameters
        ----------
        goniometers : list of list
            Goniometer definitions, each a 6-element sequence
            ``(name, x, y, z, sense, angle)`` where ``(x, y, z)`` is the
            rotation axis, ``sense`` is the rotation sense, and ``angle``
            is the rotation angle in degrees.

        Returns
        -------
        axes : list of str
            Three axis strings of the form ``"angle,x,y,z,sense"``, or
            None if fewer than three axes have a nonzero rotation vector.

        """

        axes = []
        for goniometer in goniometers:
            name, x, y, z, sense, angle = goniometer
            if np.linalg.norm([x, y, z]) > 0:
                axes.append("{},{},{},{},{}".format(angle, x, y, z, sense))

        if len(axes) == 3:
            return axes

    def set_sample(self, shape_dict, mat_dict, axes):
        """
        Set the goniometer, shape, and material on the sample workspace.

        Parameters
        ----------
        shape_dict : dict
            Shape dictionary as returned by :meth:`get_shape_dict`, used
            as the ``Geometry`` argument of Mantid's ``SetSample``.
        mat_dict : dict
            Material dictionary as returned by :meth:`get_material_dict`,
            used as the ``Material`` argument of Mantid's ``SetSample``.
        axes : list of str
            Three goniometer axis strings as returned by
            :meth:`get_goniometer_strings`, used as the ``Axis0``,
            ``Axis1``, and ``Axis2`` arguments of Mantid's
            ``SetGoniometer``.

        """

        SetGoniometer(
            Workspace="sample", Axis0=axes[0], Axis1=axes[1], Axis2=axes[2]
        )

        SetSample(
            InputWorkspace="sample", Geometry=shape_dict, Material=mat_dict
        )

    def get_absorption_dict(self):
        """
        Compute absorption and scattering parameters for the sample material.

        Returns
        -------
        abs_dict : dict
            Dictionary with keys:

            - ``sigma_a`` : float, absorption cross section (barn).
            - ``sigma_s`` : float, total scattering cross section (barn).
            - ``mu_a`` : float, linear absorption coefficient (1/cm).
            - ``mu_s`` : float, linear scattering coefficient (1/cm).
            - ``N`` : float, total number of atoms.
            - ``M`` : float, relative molecular mass (g/mol).
            - ``n`` : float, effective number density (1/Angstrom^3).
            - ``rho`` : float, mass density (g/cm^3).
            - ``V`` : float, sample volume (cm^3).
            - ``m`` : float, sample mass (g).

        """

        mat = mtd["sample"].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective
        N = mat.totalAtoms

        V = abs(mtd["sample"].sample().getShape().volume() * 100**3)

        rho = (n / N) / 0.6022 * M
        m = rho * V

        mu_s = n * sigma_s
        mu_a = n * sigma_a

        abs_dict = {
            "sigma_a": sigma_a,  # barn
            "sigma_s": sigma_s,  # barn
            "mu_a": mu_a,  # 1/cm
            "mu_s": mu_s,  # 1/cm
            "N": N,  # atoms
            "M": M,  # g/mol
            "n": n,  # 1/A^3
            "rho": rho,  # g/cm^3
            "V": V,  # cm^3
            "m": m,
        }  # g

        return abs_dict

    def sample_mesh(self):
        """
        Return the triangulated mesh of the sample shape for visualization.

        Returns
        -------
        mesh : ndarray
            Array of triangle vertices describing the sample shape mesh,
            scaled from meters to centimeters.

        """

        shape = mtd["sample"].sample().getShape()

        return shape.getMesh() * 100
