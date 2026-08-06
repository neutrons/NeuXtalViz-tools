from mantid.kernel import V3D

from mantid.geometry import (
    CrystalStructure,
    Goniometer,
    ReflectionGenerator,
    ReflectionConditionFilter,
    PointGroup,
    PointGroupFactory,
    SpaceGroupFactory,
)

from mantid.simpleapi import (
    CloneWorkspace,
    CreateSampleWorkspace,
    CreatePeaksWorkspace,
    DeleteWorkspace,
    LoadCIF,
    LoadIsawUB,
    LoadNexus,
    LoadSampleShape,
    PredictPeaks,
    SaveINS,
    SetGoniometer,
    SetSample,
    SetUB,
    HFIRCalculateGoniometer,
    AddAbsorptionWeightedPathLengths,
    mtd,
)

import os
import tempfile

import numpy as np
import scipy.linalg

import pyvista as pv

from NeuXtalViz.config.instruments import beamlines
from NeuXtalViz.models.periodic_table import PeriodicTableModel
from NeuXtalViz.models.base_model import NeuXtalVizModel


class CrystalStructureModel(NeuXtalVizModel):
    """
    Model for crystal structure and space group handling using Mantid.

    Wraps a Mantid ``crystal`` sample workspace to manage the crystal
    structure (lattice, space group, and scatterers), compute the UB
    matrix, and generate derived crystallographic information such as
    structure factors, chemical formula, and atom positions.
    """

    PEAKS_WORKSPACE = "absorption_peaks"

    SIM_RESPONSE_WORKSPACE = "simulator_response"
    SIM_BACKGROUND_WORKSPACE = "simulator_background"
    SIM_PEAKS_WORKSPACE = "simulator_peaks"

    # Coulombs of proton charge per minute of counting at full nominal
    # beam power (per the simulator's calibration convention: 0.08 C
    # is equivalent to 1 minute at full power).
    CHARGE_PER_MINUTE = 0.08

    # Structure-factor-squared threshold below which a PredictPeaks
    # reflection is treated as systematically absent. Must be an
    # absolute (not exactly-zero) cutoff: ReflectionGenerator.getFsSquared
    # evaluates trig functions of pi multiples for forbidden
    # reflections, so it returns floating-point noise (e.g. 1e-30)
    # rather than an exact 0 -- filtering on "<= 0" alone lets that
    # noise through as a spuriously "observed" reflection.
    F2_EPSILON = 1e-3

    def __init__(self):
        """
        Initialize the model and create the underlying sample workspace.
        """
        super(CrystalStructureModel, self).__init__()

        CreateSampleWorkspace(OutputWorkspace="crystal")

    def save_ins(self, filename):
        """
        Set the sample material and save the crystal structure to an INS file.

        Parameters
        ----------
        filename : str
            Path of the INS file to write.
        """
        self.set_material()

        SaveINS(
            InputWorkspace="crystal",
            Filename=filename,
            UseNaturalIsotopicAbundances=True,
        )

    def has_crystal_structure(self):
        """
        Check whether the sample workspace has a crystal structure set.

        Returns
        -------
        has_structure : bool
            True if the ``crystal`` workspace sample has a crystal
            structure, False otherwise.
        """
        return mtd["crystal"].sample().hasCrystalStructure()

    def set_material(self):
        """
        Set the sample material from the chemical formula and number density.

        Computes the chemical formula and Z parameter from the current
        crystal structure and derives the sample number density from the
        unit cell volume, then applies it to the ``crystal`` workspace.
        """
        chemical_formula, z = self.get_chemical_formula_z_parameter()
        n = z / self.get_unit_cell_volume()

        chemical_formula = " ".join(chemical_formula.split("-"))
        SetSample(
            InputWorkspace="crystal",
            Material={
                "ChemicalFormula": chemical_formula,
                "SampleNumberDensity": float(str(n)),
            },
        )

    @staticmethod
    def generate_space_groups_from_crystal_system(system):
        """
        Generate the list of space groups belonging to a crystal system.

        A static method (uses no instance state) so it can be reused
        by other tools (e.g. the Volume Slicer's delta-PDF Transform
        tab) without instantiating a full ``CrystalStructureModel``,
        which would create/clobber its own ``"crystal"`` sample
        workspace.

        Parameters
        ----------
        system : str
            Name of the crystal system (e.g. ``"Cubic"``, ``"Hexagonal"``),
            matching a member of ``PointGroup.CrystalSystem``.

        Returns
        -------
        space_group : list of str
            Space groups formatted as ``"{number}: {symbol}"``, one per
            unique space group number in the crystal system.
        """
        pg_system = getattr(PointGroup.CrystalSystem, system)
        pgs = list(PointGroupFactory.getPointGroupSymbols(pg_system))
        pgs = [PointGroupFactory.createPointGroup(pg) for pg in pgs]
        sgs = [SpaceGroupFactory.getSpaceGroupsForPointGroup(pg) for pg in pgs]
        sgs = [sg for sg_list in sgs for sg in sg_list]
        sgs = [SpaceGroupFactory.createSpaceGroup(sg) for sg in sgs]

        nos = np.unique([sg.getNumber() for sg in sgs]).tolist()

        space_group = []
        for no in nos:
            symbol = SpaceGroupFactory.subscribedSpaceGroupSymbols(no)[0]
            space_group.append("{}: {}".format(no, symbol))

        return space_group

    @staticmethod
    def generate_settings_from_space_group(sg):
        """
        Generate the list of alternative settings (symbols) for a space group.

        A static method (uses no instance state) -- see
        :meth:`generate_space_groups_from_crystal_system`.

        Parameters
        ----------
        sg : str
            Space group formatted as ``"{number}: {symbol}"``, as returned
            by :meth:`generate_space_groups_from_crystal_system` or
            :meth:`get_space_group`.

        Returns
        -------
        settings : list of str
            Subscribed space group symbols (settings) sharing the same
            space group number as `sg`.
        """
        no, symbol = sg.split(": ")

        return list(SpaceGroupFactory.subscribedSpaceGroupSymbols(int(no)))

    def load_CIF(self, filename):
        """
        Load a crystal structure from a CIF file and recompute the UB matrix.

        Parameters
        ----------
        filename : str
            Path to the CIF file to load.
        """
        LoadCIF(Workspace="crystal", InputFile=filename)

        self.calculate_UB()

    def set_crystal_structure(self, params, space_group, scatterers):
        """
        Set the crystal structure on the sample workspace and recompute UB.

        Parameters
        ----------
        params : list of float
            Lattice parameters ``[a, b, c, alpha, beta, gamma]``.
        space_group : str
            Space group symbol or setting (Hermann-Mauguin symbol) used to
            build the :class:`~mantid.geometry.CrystalStructure`.
        scatterers : list of list
            List of scatterers, each formatted as
            ``[atom, x, y, z, occupancy, Uiso]``.
        """
        line = " ".join(["{}"] * 6)

        constants = line.format(*params)

        atom_info = ";".join([line.format(*s) for s in scatterers])

        cs = CrystalStructure(constants, space_group, atom_info)

        mtd["crystal"].sample().setCrystalStructure(cs)

        self.calculate_UB()

    def update_lattice_parameters(self, a, b, c, alpha, beta, gamma):
        """
        Update the lattice parameters, keeping the current setting and atoms.

        Parameters
        ----------
        a, b, c : float
            Lattice constants.
        alpha, beta, gamma : float
            Lattice angles in degrees.
        """
        scatterers = self.get_scatterers()

        sg = self.get_setting()

        params = [a, b, c, alpha, beta, gamma]

        self.set_crystal_structure(params, sg, scatterers)

    def calculate_UB(self):
        """
        Calculate and set the UB matrix from the current crystal structure.

        Computes the UB matrix from the unit cell metric tensor using
        Cholesky decomposition, sets it on the ``crystal`` workspace, and
        stores it on the model via :meth:`set_UB`.
        """
        cs = mtd["crystal"].sample().getCrystalStructure()

        uc = cs.getUnitCell()

        G = uc.getG()
        G_star = np.linalg.inv(G)

        A = scipy.linalg.cholesky(G, lower=False)
        B = scipy.linalg.cholesky(G_star, lower=False)

        U = np.linalg.inv(A).T @ np.linalg.inv(B)

        UB = np.dot(U, B)

        SetUB(Workspace="crystal", UB=UB)

        self.set_UB(UB)

    def generate_F2(self, d_min=0.7):
        """
        Generate unique reflections and their structure factors down to d_min.

        Parameters
        ----------
        d_min : float, optional
            Minimum d-spacing to generate reflections for (default 0.7).

        Returns
        -------
        hkls : numpy.ndarray
            Array of unique HKL indices, sorted by decreasing d-spacing.
        ds : numpy.ndarray
            Array of d-spacings corresponding to `hkls`.
        F2s : numpy.ndarray
            Array of squared structure factors corresponding to `hkls`.
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        generator = ReflectionGenerator(cryst_struct)

        sf_filt = ReflectionConditionFilter.StructureFactor

        unit_cell = cryst_struct.getUnitCell()

        d_max = np.max([unit_cell.a(), unit_cell.b(), unit_cell.c()])

        hkls = generator.getUniqueHKLsUsingFilter(d_min, d_max, sf_filt)

        ds = generator.getDValues(hkls)

        F2s = generator.getFsSquared(hkls)

        sort = np.argsort(ds)[::-1]

        return np.array(hkls)[sort], np.array(ds)[sort], np.array(F2s)[sort]

    def calculate_F2(self, h, k, l):
        """
        Calculate the structure factor and symmetry equivalents of a reflection.

        Parameters
        ----------
        h, k, l : float
            Miller indices of the reflection.

        Returns
        -------
        equivalents : list of V3D
            Symmetry-equivalent HKL reflections under the point group.
        d : float
            d-spacing of the reflection.
        F2 : float
            Squared structure factor of the reflection.
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        generator = ReflectionGenerator(cryst_struct)

        hkl = V3D(h, k, l)

        d = generator.getDValues([hkl])[0]

        F2 = generator.getFsSquared([hkl])[0]

        pg = cryst_struct.getSpaceGroup().getPointGroup()

        equivalents = pg.getEquivalents(hkl)

        return equivalents, d, F2

    def get_euler_angles(self, u_vector, v_vector, UB):
        """
        Compute Euler angles that orient a sample face along given directions.

        The `u_vector` and `v_vector` are Miller index / fractional
        coordinate directions that are transformed to Cartesian coordinates
        via ``UB`` to build a rotation matrix, which is then decomposed
        into ZYX Euler angles via Mantid's own ``Goniometer`` (matching
        ``garnet.reduction.sample.SampleMaterial.set_shape`` exactly,
        rather than scipy's rotation decomposition, so the angles are
        guaranteed to be in the convention Mantid's own
        ``LoadSampleShape`` (``XDegrees``/``YDegrees``/``ZDegrees``)
        expects).

        Note this takes an explicit ``UB`` rather than reading
        ``self.UB``: the shape must be oriented relative to the *same*
        orientation used for the rest of the absorption prediction
        (typically the sample-orientation UB from
        :meth:`get_UB_from_vectors`), not the model's own default
        (Busing-Levy) orientation, or the shape and the beam/goniometer
        geometry would be computed in inconsistent frames.

        Parameters
        ----------
        u_vector : 3-element 1d array-like
            Crystallographic direction to align with the sample's primary
            (height/thickness) axis.
        v_vector : 3-element 1d array-like
            Crystallographic direction used together with `u_vector` to
            define the orientation plane.
        UB : (3, 3) ndarray
            Orientation matrix to transform `u_vector`/`v_vector` into
            Cartesian coordinates.

        Returns
        -------
        alpha, beta, gamma : float
            Rotation angles in degrees about the X, Y, and Z axes
            respectively, or None if the vectors are collinear.
        """
        w_vector = np.cross(u_vector, v_vector)

        if np.linalg.norm(w_vector) > 0:
            u = np.dot(UB, u_vector)
            v = np.dot(UB, v_vector)

            u /= np.linalg.norm(u)

            w = np.cross(u, v)
            w /= np.linalg.norm(w)

            v = np.cross(w, u)

            T = np.column_stack([v, w, u])

            gon = Goniometer()
            gon.setR(T)
            gamma, beta, alpha = gon.getEulerAngles("ZYX")

            return alpha, beta, gamma

    def write_ellipsoid_stl(self, params):
        """
        Build a triaxial ellipsoid mesh and save it as a temporary STL file.

        Mantid's shape system (CSG XML / ``SetSample``) has no native
        ellipsoid primitive, so the ellipsoid is instead built as a
        unit icosphere non-uniformly scaled to the requested semi-axis
        lengths, then loaded as the sample shape via
        :meth:`set_sample_shape`'s ``LoadSampleShape`` call.

        Parameters
        ----------
        params : list of float
            Thickness, width, and height, in mm.

        Returns
        -------
        path : str
            Path to the written temporary ``.stl`` file. The caller is
            responsible for deleting it once ``LoadSampleShape`` has
            consumed it.
        volume : float
            Ellipsoid volume in cm^3 (from the PyVista mesh directly --
            Mantid's ``MeshObject`` shape has no ``volume()`` method, so
            this is needed by :meth:`get_absorption_dict`).
        """
        thickness, width, height = params

        sph = pv.Icosphere(radius=0.5)
        ell = sph.scale([width, height, thickness], inplace=False)

        fd, path = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        ell.save(path)

        return path, ell.volume / 1000.0

    def set_sample_shape(
        self, ws, mat_dict, alpha, beta, gamma, ellipsoid_stl
    ):
        """
        Set the ellipsoid sample shape and material on a workspace.

        Loads the mesh built by :meth:`write_ellipsoid_stl` via
        Mantid's ``LoadSampleShape`` (rotated by alpha/beta/gamma about
        X, Y, Z in that order), then sets the material separately.

        ``LoadSampleShape`` requires a workspace with a real instrument
        (specifically a "sample holder" component) -- a bare
        ``LeanElasticPeak`` workspace has none, so the mesh is instead
        loaded onto a throwaway scratch workspace (which does have a
        default instrument) and its resulting ``Sample`` (shape) is
        copied onto ``ws`` via ``setSample``.

        Parameters
        ----------
        ws : str
            Name of the workspace to set the sample on.
        mat_dict : dict
            Material dictionary from :meth:`get_material_dict`.
        alpha, beta, gamma : float
            Rotation angles in degrees for the ellipsoid mesh's
            ``LoadSampleShape`` orientation.
        ellipsoid_stl : str
            Path to the ellipsoid STL file from
            :meth:`write_ellipsoid_stl`.
        """
        scratch = "_{}_ellipsoid_scratch".format(ws)
        CreateSampleWorkspace(OutputWorkspace=scratch)
        LoadSampleShape(
            InputWorkspace=scratch,
            Filename=ellipsoid_stl,
            Scale="mm",
            XDegrees=alpha,
            YDegrees=beta,
            ZDegrees=gamma,
            OutputWorkspace=scratch,
        )
        mtd[ws].setSample(mtd[scratch].sample())
        DeleteWorkspace(Workspace=scratch)
        SetSample(InputWorkspace=ws, Material=mat_dict)

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
        return {
            "ChemicalFormula": chemical_formula,
            "ZParameter": z_parameter,
            "UnitCellVolume": volume,
        }

    def generate_hkl_list(self, d_min=0.7):
        """
        Generate every allowed reflection and its d-spacing down to d_min.

        Unlike :meth:`generate_F2` (the Factors tab, which reduces to
        symmetry-unique reflections via ``getUniqueHKLsUsingFilter``),
        this returns every symmetry-equivalent allowed reflection via
        Mantid's ``ReflectionGenerator.getHKLsUsingFilter`` -- the
        absorption/transmission prediction is per-reflection (each
        equivalent has its own Q direction and therefore its own
        goniometer setting and absorption path), so the full set is
        needed, not just one representative per family.

        Absorption is centrosymmetric though: reversing (h,k,l) to
        (-h,-k,-l) swaps the incident/outgoing beam directions through
        the sample, and an ellipsoid (like every shape supported here)
        is itself inversion-symmetric about its center, so
        T(hkl) == T(-hkl) exactly. Only one reflection from each such
        Friedel pair is kept (whichever has the first nonzero index
        positive), halving the simulation work for no loss of
        information.

        Parameters
        ----------
        d_min : float, optional
            Minimum d-spacing (default 0.7).

        Returns
        -------
        hkls : (N, 3) ndarray
            Miller indices of every allowed reflection, one per
            centrosymmetric (Friedel) pair.
        ds : (N,) ndarray
            d-spacing of each reflection.
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        generator = ReflectionGenerator(cryst_struct)

        sf_filt = ReflectionConditionFilter.StructureFactor

        unit_cell = cryst_struct.getUnitCell()

        d_max = np.max([unit_cell.a(), unit_cell.b(), unit_cell.c()])

        hkls = generator.getHKLsUsingFilter(d_min, d_max, sf_filt)

        ds = np.array(generator.getDValues(hkls))
        hkls = np.array([[hkl[0], hkl[1], hkl[2]] for hkl in hkls])

        first_nonzero = np.argmax(hkls != 0, axis=1)
        keep = hkls[np.arange(len(hkls)), first_nonzero] > 0

        return hkls[keep], ds[keep]

    def get_UB_from_vectors(self, u_vector, v_vector):
        """
        Build a sample-orientation UB matrix from two crystallographic
        directions, independent of the model's own ``self.UB``.

        Distinct from :meth:`get_euler_angles` (which orients the
        *shape* mesh relative to whatever orientation the crystal
        already has): this defines the crystal's *own* orientation for
        the absorption prediction -- ``u_vector`` is aligned with the
        local z axis and, together with ``v_vector``, fixes an
        orthonormal frame U, independent of the arbitrary Busing-Levy
        default in :meth:`calculate_UB`. B comes directly from the
        lattice metric tensor, not from ``self.UB``.

        Parameters
        ----------
        u_vector : 3-element 1d array-like
            Crystallographic direction to align with the local z axis.
        v_vector : 3-element 1d array-like
            Crystallographic direction used together with `u_vector` to
            define the orientation plane.

        Returns
        -------
        UB : (3, 3) ndarray
            Orientation matrix ``U @ B``, or None if the vectors are
            collinear.
        """
        w_vector = np.cross(u_vector, v_vector)

        if np.linalg.norm(w_vector) == 0:
            return None

        cryst_struct = mtd["crystal"].sample().getCrystalStructure()
        uc = cryst_struct.getUnitCell()
        G = uc.getG()
        G_star = np.linalg.inv(G)
        B = scipy.linalg.cholesky(G_star, lower=False)

        u = np.dot(B, u_vector)
        v = np.dot(B, v_vector)

        u /= np.linalg.norm(u)

        w = np.cross(u, v)
        w /= np.linalg.norm(w)

        v = np.cross(w, u)

        U = np.column_stack([v, w, u])

        return U @ B

    def get_transform_from_UB(self, UB):
        """
        Normalized orientation matrix (unit a*/b*/c* Cartesian columns)
        for a given UB matrix.

        Same normalization as the base model's ``get_transform``, but
        operating on a caller-supplied ``UB`` (e.g. from
        :meth:`get_UB_from_vectors`) instead of ``self.UB`` -- used to
        orient the a*/b*/c* arrows drawn next to the absorption sample
        without touching the model's own stored orientation (which the
        Structure tab's unit-cell view depends on).

        Parameters
        ----------
        UB : (3, 3) ndarray
            Orientation matrix.

        Returns
        -------
        T : (3, 3) ndarray
            ``UB`` with each column normalized to unit length.
        """
        T = np.array(UB, dtype=float)
        return T / np.linalg.norm(T, axis=0)

    def predict_transmission(
        self,
        hkls,
        ds,
        wavelength,
        shape_params,
        mat_dict,
        alpha,
        beta,
        gamma,
        UB,
    ):
        """
        Predict the transmission and absorption-weighted path length
        (T-bar) of every given reflection for a monochromatic rotation
        experiment.

        Builds a ``LeanElasticPeak`` peaks workspace with one peak per
        hkl (``Q = 2*pi*UB*hkl``, using the given ``UB``), each given
        its own unique run number (each peak is an independent
        simulated "setting"), computes a goniometer setting for each
        via Mantid's ``HFIRCalculateGoniometer`` (constant wavelength,
        vertical-axis rotation -- no user-specified goniometer axes are
        needed), drops any reflection whose goniometer matrix comes
        back NaN (not reachable by a single vertical-axis rotation at
        this wavelength -- kept in the result as NaN rows rather than
        silently dropped), sets the ellipsoid sample shape/material,
        then runs Mantid's ``AddAbsorptionWeightedPathLengths`` (Monte
        Carlo) to get T-bar per peak and derives the transmission
        ``T = exp(-mu * Tbar)``.

        Parameters
        ----------
        hkls : (N, 3) ndarray
            Miller indices, from :meth:`generate_hkl_list`.
        ds : (N,) ndarray
            d-spacing of each reflection, from :meth:`generate_hkl_list`.
        wavelength : float
            Incident wavelength, in Angstrom.
        shape_params : list of float
            Ellipsoid thickness, width, and height, in mm.
        mat_dict : dict
            Material dictionary from :meth:`get_material_dict`.
        alpha, beta, gamma : float
            Shape orientation Euler angles from :meth:`get_euler_angles`
            (shape U/V vectors).
        UB : (3, 3) ndarray
            Sample orientation matrix used for ``Q = 2*pi*UB*hkl``, from
            :meth:`get_UB_from_vectors` (sample U/V vectors), or
            ``self.UB`` if the sample U/V vectors weren't set.

        Returns
        -------
        hkls : (M, 3) ndarray
            Miller indices, sorted by decreasing d-spacing then
            increasing h, k, l.
        ds : (M,) ndarray
            d-spacing of each reflection.
        Ts : (M,) ndarray
            Transmission of each reflection (NaN if unreachable at this
            wavelength).
        Tbars : (M,) ndarray
            Absorption-weighted path length (cm) of each reflection
            (NaN if unreachable at this wavelength).
        volume : float
            Ellipsoid volume, in cm^3.
        """
        ws = self.PEAKS_WORKSPACE

        CreatePeaksWorkspace(
            OutputType="LeanElasticPeak", NumberOfPeaks=0, OutputWorkspace=ws
        )
        peaks = mtd[ws]

        for i, (h, k, l) in enumerate(hkls):
            Q = 2 * np.pi * UB @ np.array([h, k, l])
            pk = peaks.createPeakQSample([Q[0], Q[1], Q[2]])
            pk.setHKL(h, k, l)
            pk.setRunNumber(i + 1)
            peaks.addPeak(pk)

        HFIRCalculateGoniometer(Workspace=ws, Wavelength=wavelength)

        bad = [
            i
            for i in range(peaks.getNumberPeaks())
            if np.any(np.isnan(peaks.getPeak(i).getGoniometerMatrix()))
        ]

        hkls_ok = np.delete(hkls, bad, axis=0)
        ds_ok = np.delete(ds, bad)
        hkls_bad = hkls[bad]

        if bad:
            peaks.removePeaks(bad)

        if peaks.getNumberPeaks() == 0:
            return None

        ellipsoid_stl, ellipsoid_volume = self.write_ellipsoid_stl(
            shape_params
        )

        try:
            self.set_sample_shape(
                ws, mat_dict, alpha, beta, gamma, ellipsoid_stl
            )
        finally:
            os.remove(ellipsoid_stl)

        AddAbsorptionWeightedPathLengths(InputWorkspace=ws)

        mat = mtd[ws].sample().getMaterial()
        n = mat.numberDensityEffective
        mu = n * (mat.absorbXSection(wavelength) + mat.totalScatterXSection())

        Ts, Tbars = [], []
        for i in range(peaks.getNumberPeaks()):
            tbar = peaks.getPeak(i).getAbsorptionWeightedPathLength()
            Tbars.append(tbar)
            Ts.append(np.exp(-mu * tbar))
        Ts, Tbars = np.array(Ts), np.array(Tbars)

        nan = np.full(len(hkls_bad), np.nan)
        hkls_all = np.vstack([hkls_ok, hkls_bad]) if len(hkls_bad) else hkls_ok
        Ts_all = np.concatenate([Ts, nan])
        Tbars_all = np.concatenate([Tbars, nan])
        ds_all = np.concatenate([ds_ok, ds[bad]]) if len(hkls_bad) else ds_ok

        order = np.lexsort(
            (hkls_all[:, 2], hkls_all[:, 1], hkls_all[:, 0], -ds_all)
        )

        return (
            hkls_all[order],
            ds_all[order],
            Ts_all[order],
            Tbars_all[order],
            ellipsoid_volume,
        )

    def get_absorption_dict(self, wavelength, volume=None):
        """
        Compute absorption and scattering parameters for the sample material.

        Parameters
        ----------
        wavelength : float
            Incident wavelength, in Angstrom (the absorption cross
            section is wavelength-dependent).
        volume : float, optional
            Sample volume (cm^3), for shapes whose Mantid ``Sample``
            object can't report its own volume -- in practice only the
            ellipsoid (a ``MeshObject``, which exposes no ``volume()``
            method), whose volume is instead computed from the PyVista
            mesh in :meth:`write_ellipsoid_stl`. If None, the volume is
            read directly from the sample shape (CSG shapes only).

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
        mat = mtd[self.PEAKS_WORKSPACE].sample().getMaterial()

        sigma_a = mat.absorbXSection(wavelength)
        sigma_s = mat.totalScatterXSection()

        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective
        N = mat.totalAtoms

        if volume is None:
            shape = mtd[self.PEAKS_WORKSPACE].sample().getShape()
            V = abs(shape.volume() * 100**3)
        else:
            V = volume

        rho = (n / N) / 0.6022 * M
        m = rho * V

        mu_s = n * sigma_s
        mu_a = n * sigma_a

        return {
            "sigma_a": sigma_a,
            "sigma_s": sigma_s,
            "mu_a": mu_a,
            "mu_s": mu_s,
            "N": N,
            "M": M,
            "n": n,
            "rho": rho,
            "V": V,
            "m": m,
        }

    def sample_mesh(self):
        """
        Return the triangulated mesh of the sample shape for visualization.

        Returns
        -------
        mesh : ndarray
            Array of triangle vertices describing the sample shape mesh,
            scaled from meters to centimeters.
        """
        shape = mtd[self.PEAKS_WORKSPACE].sample().getShape()

        return shape.getMesh() * 100

    def get_incident_path_length(self, mesh, UB, u_vector):
        """
        Sample radius along the incident beam direction.

        Matches ``AddAbsorptionWeightedPathLengths``'s single-path
        formula (see its C++ source): the total single-path length is
        the sample's radius along the *outgoing* direction (per point,
        already used directly as ``|vertex|`` when coloring the mesh)
        plus a *constant* term -- the sample's radius along the fixed
        *incident* beam direction. Found by looking up the mesh's own
        vertex closest to the beam direction (rather than an
        independent analytic/rotation-matrix calculation), so it is
        guaranteed self-consistent with the mesh actually being drawn.

        Parameters
        ----------
        mesh : (N, 3, 3) ndarray
            Triangle vertex coordinates, from :meth:`sample_mesh`.
        UB : (3, 3) ndarray
            Sample orientation matrix, from :meth:`get_UB_from_vectors`
            (or ``self.UB``).
        u_vector : 3-element 1d array-like
            Beam-direction crystallographic direction (the sample U
            vector).

        Returns
        -------
        r_incident : float
            Sample radius (cm) along the incident beam direction.
        """
        beam_dir = UB @ np.asarray(u_vector, dtype=float)
        beam_dir /= np.linalg.norm(beam_dir)

        vertices = mesh.reshape(-1, 3)
        directions = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        best = np.argmax(directions @ beam_dir)

        return np.linalg.norm(vertices[best])

    def load_UB_from_file(self, filename):
        """
        Load a UB matrix from an ISAW UB file, for the Simulator tab.

        Loaded onto a throwaway scratch workspace (``LoadIsawUB``
        requires an existing sample), independent of the ``"crystal"``
        workspace's own UB from :meth:`calculate_UB`, so switching the
        Simulator to a loaded UB never disturbs the Structure/Factors/
        Absorption tabs' orientation.

        Parameters
        ----------
        filename : str
            Path to the ISAW UB matrix file.

        Returns
        -------
        UB : (3, 3) ndarray
            Orientation matrix loaded from `filename`.
        """
        scratch = "_simulator_UB_scratch"
        CreateSampleWorkspace(OutputWorkspace=scratch)
        LoadIsawUB(InputWorkspace=scratch, Filename=filename)
        UB = mtd[scratch].sample().getOrientedLattice().getUB().copy()
        DeleteWorkspace(Workspace=scratch)

        return UB

    def load_simulator_response(self, instrument):
        """
        Load an instrument's calibrated bank response and background-rate
        workspaces for the Simulator tab.

        Parameters
        ----------
        instrument : str
            Instrument name ("TOPAZ", "MANDI", or "CORELLI"), used to
            look up the ``count_rate.nxs``/``background_count_rate.nxs``
            paths in ``config.instruments.beamlines``.
        """
        sim = beamlines[instrument]["Simulator"]

        LoadNexus(
            Filename=sim["CountRate"],
            OutputWorkspace=self.SIM_RESPONSE_WORKSPACE,
        )
        LoadNexus(
            Filename=sim["Background"],
            OutputWorkspace=self.SIM_BACKGROUND_WORKSPACE,
        )

    def get_instrument_d_min(self, instrument):
        """
        Default minimum d-spacing for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument name.

        Returns
        -------
        d_min : float
            Instrument-default minimum d-spacing (Å).
        """
        return beamlines[instrument]["MinD"]

    def get_goniometer_axes(self, instrument, omega, chi, phi):
        """
        Build ``SetGoniometer`` Axis0-2 strings for fixed numeric angles.

        Substitutes the given numeric angles in place of the motor-name
        token of each configured axis string (the same convention used
        for scan-orientation goniometer settings elsewhere in
        NeuXtalViz, e.g. the Experiment Planner), so the axis rotation
        vectors and sense come from the instrument configuration while
        the angle itself is a literal value rather than a sample log.

        Parameters
        ----------
        instrument : str
            Instrument name.
        omega, chi, phi : float
            Goniometer angles, in degrees.

        Returns
        -------
        axes : list of str
            ``SetGoniometer`` Axis0, Axis1, Axis2 argument strings.
        """
        axes_cfg = beamlines[instrument]["Goniometers"]
        angles = (omega, chi, phi)

        axes = []
        for cfg, angle in zip(axes_cfg, angles):
            _, x, y, z, sense = cfg.split(",")
            axes.append(",".join([str(angle), x, y, z, sense]))

        return axes

    def predict_simulator_peaks(self, instrument, UB, omega, chi, phi, d_min):
        """
        Predict the reflections observable by an instrument at a fixed
        goniometer setting.

        Clones the already-loaded response workspace (which carries
        the instrument's real detector geometry, from
        :meth:`load_simulator_response`) so Mantid's ``PredictPeaks``
        can determine which reflections actually land on a detector
        element at this orientation and wavelength band, rather than
        just which are geometrically allowed by the lattice. The
        clone's sample is also given the loaded CIF's actual
        ``CrystalStructure`` (not just its UB), so ``PredictPeaks``
        can compute each reflection's structure factor squared
        directly (via ``CalculateStructureFactors=True``, read back
        as ``peak.getIntensity()``) instead of a separate
        ``ReflectionGenerator`` pass.

        Parameters
        ----------
        instrument : str
            Instrument name.
        UB : (3, 3) ndarray
            Sample orientation matrix at the zero-goniometer setting.
        omega, chi, phi : float
            Goniometer angles, in degrees.
        d_min : float
            Minimum d-spacing (Å).

        Returns
        -------
        peaks : mantid.dataobjects.PeaksWorkspace
            Predicted peaks, with detector ID, wavelength, d-spacing,
            HKL, and structure factor squared (``getIntensity()``) set
            for each. Systematically absent reflections come back with
            F² approximately (not exactly) zero -- callers must filter
            on an epsilon (see :attr:`F2_EPSILON`), not exact equality.
        """
        ws = "_{}_input".format(self.SIM_PEAKS_WORKSPACE)

        CloneWorkspace(
            InputWorkspace=self.SIM_RESPONSE_WORKSPACE, OutputWorkspace=ws
        )

        cs = mtd["crystal"].sample().getCrystalStructure()
        mtd[ws].sample().setCrystalStructure(cs)

        SetUB(Workspace=ws, UB=UB)

        axes = self.get_goniometer_axes(instrument, omega, chi, phi)
        SetGoniometer(
            Workspace=ws, Axis0=axes[0], Axis1=axes[1], Axis2=axes[2]
        )

        a, b, c = self.get_lattice_constants()[0:3]
        d_max = max(a, b, c)

        wl_min, wl_max = beamlines[instrument]["Wavelength"]

        PredictPeaks(
            InputWorkspace=ws,
            MinDSpacing=d_min,
            MaxDSpacing=d_max,
            WavelengthMin=wl_min,
            WavelengthMax=wl_max,
            ReflectionCondition="Primitive",
            CalculateStructureFactors=True,
            OutputWorkspace=self.SIM_PEAKS_WORKSPACE,
        )

        DeleteWorkspace(Workspace=ws)

        return mtd[self.SIM_PEAKS_WORKSPACE]

    def spectrum_for_detector(self, ws, det_id):
        """
        Spectrum (workspace) index of a response/background workspace
        containing a given detector.

        Uses Mantid's own detector-ID-to-workspace-index map rather
        than matching bank names between the peak's instrument and the
        workspace's vertical-axis labels -- the calibration file's
        axis label convention isn't guaranteed (in practice, generated
        ``count_rate.nxs``/``background_count_rate.nxs`` files have
        used a plain spectrum-number ``SpectraAxis``, not a bank-name
        ``TextAxis``), but each spectrum's own detector grouping is
        always authoritative, so this works regardless of what (if
        anything) the axis labels say.

        Parameters
        ----------
        ws : str
            Workspace name.
        det_id : int
            Detector ID.

        Returns
        -------
        spectrum_index : int or None
            Spectrum index containing `det_id`, or None if no spectrum
            of `ws` contains it.
        """
        indices = mtd[ws].getIndicesFromDetectorIDs([det_id])

        return indices[0] if indices else None

    def response_value(self, ws, spectrum_index, wavelength):
        """
        Calibrated response/background density at a given wavelength.

        ``count_rate.nxs``/``background_count_rate.nxs`` store Y as a
        distribution (density per Angstrom, not a per-bin integrated
        count), so the matching bin's value is used directly with no
        bin-width division.

        Parameters
        ----------
        ws : str
            Workspace name.
        spectrum_index : int
            Spectrum (workspace) index, from
            :meth:`spectrum_for_detector`.
        wavelength : float
            Wavelength, in Angstrom.

        Returns
        -------
        value : float
            Response/background density at `wavelength` (0 if outside
            the calibrated wavelength range).
        error : float
            One-sigma uncertainty of `value`.
        """
        x = mtd[ws].readX(spectrum_index)
        y = mtd[ws].readY(spectrum_index)
        e = mtd[ws].readE(spectrum_index)

        if wavelength < x[0] or wavelength >= x[-1]:
            return 0.0, 0.0

        bin_index = np.searchsorted(x, wavelength, side="right") - 1
        bin_index = min(max(bin_index, 0), len(y) - 1)

        return y[bin_index], e[bin_index]

    def simulate_intensities(
        self,
        instrument,
        d_min,
        UB,
        omega,
        chi,
        phi,
        shape_params,
        mat_dict,
        shape_angles,
        counting_time,
        n_sigma=3.0,
        k_diffraction=1.0,
    ):
        """
        Predict observable reflections and their expected counts and
        I/sigma for a fixed goniometer setting on a calibrated
        instrument.

        Combines the instrument's calibrated bank response/background
        (loaded via :meth:`load_simulator_response`) with the
        structure factor of every reflection ``PredictPeaks`` finds
        observable, and the sample's Monte Carlo absorption (via the
        same ellipsoid-shape machinery as the Absorption tab, applied
        directly to the predicted-peaks workspace since it already
        carries the real goniometer/detector geometry -- no separate
        vertical-axis solve is needed here, unlike
        :meth:`predict_transmission`).

        The Lorentz factor uses the standard TOF Laue convention
        ``L = lambda^4 / sin(theta)^2 = 4 d^2 lambda^2`` (via Bragg's
        law), matching the correction applied in the opposite direction
        when extracting |F|^2 from observed SNS single-crystal data
        (e.g. Mantid's ``AnvredCorrection``/``LorentzCorrection``).

        The background ROI (solid angle and wavelength width) is a
        placeholder built directly from the instrument's divergence
        sigmas (``DivergenceParams``) rather than a full resolution
        ellipsoid -- to be replaced once the
        ``garnet.reduction.resolution.ResolutionEllipsoid`` model is
        wired in. Likewise, rocking-curve coverage and extinction are
        both assumed to be 1 (single static setting, not a scan), and
        the full sample volume is assumed illuminated (no beam
        footprint calculation).

        Parameters
        ----------
        instrument : str
            Instrument name.
        d_min : float
            Minimum d-spacing (Å).
        UB : (3, 3) ndarray
            Sample orientation matrix at the zero-goniometer setting.
        omega, chi, phi : float
            Goniometer angles, in degrees.
        shape_params : list of float
            Ellipsoid thickness, width, and height, in mm.
        mat_dict : dict
            Material dictionary from :meth:`get_material_dict`.
        shape_angles : tuple of float
            Ellipsoid shape orientation Euler angles, from
            :meth:`get_euler_angles`.
        counting_time : float
            Requested counting time, in minutes.
        n_sigma : float, optional
            Angular/wavelength half-width, in divergence sigmas, of
            the placeholder background ROI (default 3).
        k_diffraction : float, optional
            Overall normalization constant absorbing any remaining
            unit-convention difference between the calculated |F|^2
            and the calibrated response's "barn" unit -- tune against
            a measured crystal standard (default 1).

        Returns
        -------
        hkls : (N, 3) ndarray
            Miller indices, sorted by decreasing d-spacing.
        ds : (N,) ndarray
            d-spacing (Å) of each reflection.
        lambdas : (N,) ndarray
            Wavelength (Å) of each reflection.
        F2s : (N,) ndarray
            Squared structure factor of each reflection.
        Is : (N,) ndarray
            Predicted integrated counts.
        IsigmaIs : (N,) ndarray
            Predicted I/sigma.
        volume : float
            Ellipsoid volume, in cm^3.

        Returns None if no reflection is both geometrically observable
        and has a nonzero structure factor.
        """
        peaks = self.predict_simulator_peaks(
            instrument, UB, omega, chi, phi, d_min
        )

        n_peaks = peaks.getNumberPeaks()
        if n_peaks == 0:
            return None

        hkls = np.empty((n_peaks, 3))
        F2s = np.empty(n_peaks)
        for i in range(n_peaks):
            peak = peaks.getPeak(i)
            hkls[i] = peak.getH(), peak.getK(), peak.getL()
            F2s[i] = peak.getIntensity()

        bad = np.flatnonzero(F2s <= self.F2_EPSILON).tolist()
        if bad:
            peaks.removePeaks(bad)
            keep = np.flatnonzero(F2s > self.F2_EPSILON)
            hkls = hkls[keep]
            F2s = F2s[keep]

        if peaks.getNumberPeaks() == 0:
            return None

        ellipsoid_stl, ellipsoid_volume = self.write_ellipsoid_stl(
            shape_params
        )
        try:
            self.set_sample_shape(
                self.SIM_PEAKS_WORKSPACE,
                mat_dict,
                *shape_angles,
                ellipsoid_stl,
            )
        finally:
            os.remove(ellipsoid_stl)

        AddAbsorptionWeightedPathLengths(
            InputWorkspace=self.SIM_PEAKS_WORKSPACE
        )

        mat = peaks.sample().getMaterial()
        n = mat.numberDensityEffective

        cell_volume = self.get_unit_cell_volume()
        N_cell = (ellipsoid_volume * 1e24) / cell_volume

        div = beamlines[instrument].get("DivergenceParams", {})
        sigma_alpha_f = np.radians(div.get("sigma_alpha_f", 0.5))
        sigma_beta_f = np.radians(div.get("sigma_beta_f", 0.5))
        sigma_dl_mod = div.get("sigma_dl_mod", 0.01)

        q_eff = counting_time * self.CHARGE_PER_MINUTE

        ds, lambdas, Is, IsigmaIs = [], [], [], []
        keep_rows = []

        for i in range(peaks.getNumberPeaks()):
            peak = peaks.getPeak(i)

            d = peak.getDSpacing()
            lam = peak.getWavelength()

            det_id = peak.getDetectorID()
            response_spectrum = self.spectrum_for_detector(
                self.SIM_RESPONSE_WORKSPACE, det_id
            )
            if response_spectrum is None:
                continue

            R, _ = self.response_value(
                self.SIM_RESPONSE_WORKSPACE, response_spectrum, lam
            )
            if R <= 0:
                continue

            B = 0.0
            background_spectrum = self.spectrum_for_detector(
                self.SIM_BACKGROUND_WORKSPACE, det_id
            )
            if background_spectrum is not None:
                B, _ = self.response_value(
                    self.SIM_BACKGROUND_WORKSPACE, background_spectrum, lam
                )

            mu = n * (mat.absorbXSection(lam) + mat.totalScatterXSection())
            tbar = peak.getAbsorptionWeightedPathLength()
            T = np.exp(-mu * tbar)

            L = 4 * d**2 * lam**2

            S = k_diffraction * N_cell * F2s[i] * L * T

            I = q_eff * R * S

            d_omega_roi = (2 * n_sigma * sigma_alpha_f) * (
                2 * n_sigma * sigma_beta_f
            )
            d_lambda_roi = 2 * n_sigma * sigma_dl_mod * lam

            b = q_eff * B * d_omega_roi * d_lambda_roi

            variance = I + 2 * b
            IsigmaI = I / np.sqrt(variance) if variance > 0 else 0.0

            keep_rows.append(i)
            ds.append(d)
            lambdas.append(lam)
            Is.append(I)
            IsigmaIs.append(IsigmaI)

        if not keep_rows:
            return None

        hkls = hkls[keep_rows]
        F2s = F2s[keep_rows]
        ds = np.array(ds)
        lambdas = np.array(lambdas)
        Is = np.array(Is)
        IsigmaIs = np.array(IsigmaIs)

        order = np.argsort(ds)[::-1]

        return (
            hkls[order],
            ds[order],
            lambdas[order],
            F2s[order],
            Is[order],
            IsigmaIs[order],
            ellipsoid_volume,
        )

    def get_crystal_system(self):
        """
        Get the crystal system of the current crystal structure.

        Returns
        -------
        crystal_system : str
            Name of the crystal system (e.g. ``"Cubic"``).
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        pg = cryst_struct.getSpaceGroup().getPointGroup()

        return pg.getCrystalSystem().name

    def get_lattice_system(self):
        """
        Get the lattice system of the current crystal structure.

        Returns
        -------
        lattice_system : str
            Name of the lattice system (e.g. ``"Rhombohedral"``).
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        pg = cryst_struct.getSpaceGroup().getPointGroup()

        return pg.getLatticeSystem().name

    def get_point_group_name(self):
        """
        Get the point group name of the current crystal structure.

        Returns
        -------
        point_group_name : str
            Name of the point group (e.g. ``"2/m (unique axis b)"``).
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        pg = cryst_struct.getSpaceGroup().getPointGroup()

        return pg.getName()

    def get_space_group(self):
        """
        Get the space group of the current crystal structure.

        Returns
        -------
        space_group : str
            Space group formatted as ``"{number}: {symbol}"``.
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        sg = cryst_struct.getSpaceGroup()

        no = sg.getNumber()
        symbol = SpaceGroupFactory.subscribedSpaceGroupSymbols(no)[0]

        return "{}: {}".format(no, symbol)

    def get_setting(self):
        """
        Get the current space group setting (Hermann-Mauguin symbol).

        Returns
        -------
        setting : str
            Hermann-Mauguin symbol of the current space group setting.
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        return cryst_struct.getSpaceGroup().getHMSymbol()

    def get_lattice_constants(self):
        """
        Get the lattice constants of the current unit cell.

        Returns
        -------
        params : tuple of float
            Lattice parameters ``(a, b, c, alpha, beta, gamma)``.
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        uc = cryst_struct.getUnitCell()

        params = uc.a(), uc.b(), uc.c(), uc.alpha(), uc.beta(), uc.gamma()

        return params

    def get_unit_cell_volume(self):
        """
        Get the volume of the current unit cell.

        Returns
        -------
        volume : float
            Unit cell volume.
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        return cryst_struct.getUnitCell().volume()

    def get_scatterers(self):
        """
        Get the list of scatterers of the current crystal structure.

        Returns
        -------
        scatterers : list of list
            List of scatterers, each formatted as
            ``[atom, x, y, z, occupancy, Uiso]``, where `atom` is a str
            and the remaining values are floats.
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        scatterers = cryst_struct.getScatterers()

        scatterers = [atm.split(" ") for atm in list(scatterers)]

        scatterers = [
            [val if val.isalpha() else float(val) for val in scatterer]
            for scatterer in scatterers
        ]

        return scatterers

    def get_chemical_formula_z_parameter(self):
        """
        Get the chemical formula and Z parameter of the crystal structure.

        Groups scatterers by atom type, accounting for symmetry
        multiplicity and site occupancy, to build the reduced chemical
        formula and the number of formula units per unit cell.

        Returns
        -------
        chemical_formula : str
            Chemical formula with per-atom subscripts, hyphen-separated
            for multi-element formulas.
        Z : int
            Number of formula units per unit cell (greatest common
            divisor of the per-atom site multiplicities).
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        sg = cryst_struct.getSpaceGroup()

        scatterers = self.get_scatterers()

        atom_dict = {}

        for scatterer in scatterers:
            atom, x, y, z, occ, Uiso = scatterer
            n = len(sg.getEquivalentPositions([x, y, z]))
            if atom_dict.get(atom) is None:
                atom_dict[atom] = [n], [occ]
            else:
                ns, occs = atom_dict[atom]
                ns.append(n)
                occs.append(occ)
                atom_dict[atom] = ns, occs

        chemical_formula = []

        n_atm = []
        n_wgt = []

        for key in atom_dict.keys():
            ns, occs = atom_dict[key]
            n_atm.append(np.sum(ns))
            n_wgt.append(np.sum(np.multiply(ns, occs)))
            if key.isalpha():
                chemical_formula.append(key + "{:.3g}")
            else:
                chemical_formula.append("(" + key + ")" + "{:.3g}")

        Z = np.gcd.reduce(n_atm)
        n = np.divide(n_wgt, Z)

        chemical_formula = "-".join(chemical_formula).format(*n)

        return chemical_formula, Z

    def generate_atom_positions(self):
        """
        Generate symmetry-equivalent atom positions within the unit cell.

        For each scatterer, applies the space group's equivalent
        positions, wraps into the unit cell, extends to unit cell
        corners/edges/faces so atoms on cell boundaries are included, and
        transforms fractional coordinates into Cartesian coordinates.

        Returns
        -------
        atom_dict : dict
            Dictionary keyed by atom symbol, where each value is a tuple
            ``(coordinates, occupancies, indices)``: `coordinates` is a
            list of 3-element Cartesian coordinates, `occupancies` is a
            list of the corresponding site occupancies, and `indices` is
            a list of the originating scatterer index for each position.
        """
        scatterers = self.get_scatterers()

        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        sg = cryst_struct.getSpaceGroup()

        A = self.get_unit_cell_transform()

        atom_dict = {}

        corners = (
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        )

        for ind, scatterer in enumerate(scatterers):
            atom, x, y, z, occ, U = scatterer

            xyz = np.array(sg.getEquivalentPositions([x, y, z]))

            xyz = np.mod(xyz, 1)

            xyz = np.row_stack([xyz + corner for corner in corners])

            xyz = xyz[np.all(xyz <= 1, axis=1)]

            r_xyz = np.einsum("ij,kj->ki", A, xyz).tolist()
            r_occ = np.full(len(xyz), float(occ)).tolist()
            r_ind = np.full(len(xyz), ind).tolist()

            if atom_dict.get(atom) is None:
                atom_dict[atom] = r_xyz, r_occ, r_ind
            else:
                R_xyz, R_occ, R_ind = atom_dict[atom]
                R_xyz += r_xyz
                R_occ += r_occ
                R_ind += r_ind
                atom_dict[atom] = R_xyz, R_occ, R_ind

        return atom_dict

    def get_unit_cell_transform(self):
        """
        Get the transformation matrix from fractional to Cartesian coordinates.

        Returns
        -------
        A : numpy.ndarray
            3x3 upper-triangular transformation matrix obtained from the
            Cholesky decomposition of the unit cell metric tensor.
        """
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        uc = cryst_struct.getUnitCell()

        G = uc.getG()

        A = scipy.linalg.cholesky(G, lower=False)

        return A

    def constrain_parameters(self):
        """
        Determine which lattice parameters are fixed by the lattice system.

        Returns
        -------
        params : list of bool
            Six flags, one per lattice parameter in the order
            ``[a, b, c, alpha, beta, gamma]``, where True indicates the
            parameter is constrained (dependent on `a`/`alpha`) and
            should be disabled for editing.
        """
        params = np.array([False] * 6)

        lattice_system = self.get_lattice_system()

        if lattice_system == "Cubic":
            params[1:6] = True
        elif lattice_system == "Rhombohedral":
            params[1:3] = True
            params[4:6] = True
        elif lattice_system == "Hexagonal" or lattice_system == "Tetragonal":
            params[1] = True
            params[3:6] = True
        elif lattice_system == "Orthorhombic":
            params[3:6] = True
        elif lattice_system == "Monoclinic":
            if "unique axis b" in self.get_point_group_name():
                params[3] = True
                params[5] = True
            else:
                params[3:4] = True

        return params.tolist()

    def update_parameters(self, params):
        """
        Symmetrize lattice parameters according to the current lattice system.

        Copies dependent parameters (e.g. `b`, `c` from `a` for cubic
        systems) so that the returned parameters are consistent with the
        lattice system's constraints.

        Parameters
        ----------
        params : array-like
            Lattice parameters ``[a, b, c, alpha, beta, gamma]``.

        Returns
        -------
        params : list of float
            Lattice parameters with dependent values overwritten to match
            their independent counterparts per the lattice system.
        """
        params = np.array(params)

        lattice_system = self.get_lattice_system()

        if lattice_system == "Cubic":
            params[1:3] = params[0]
        elif lattice_system == "Rhombohedral":
            params[1:3] = params[0]
            params[4:6] = params[3]
        elif lattice_system == "Hexagonal" or lattice_system == "Tetragonal":
            params[1] = params[0]

        return params.tolist()

    def get_periodic_table(self, atom):
        """
        Create a periodic table model for isotope selection of an element.

        Parameters
        ----------
        atom : str
            Element or isotope symbol to initialize the periodic table
            model with.

        Returns
        -------
        model : PeriodicTableModel
            Periodic table model for selecting an isotope of `atom`.
        """
        return PeriodicTableModel(atom)
