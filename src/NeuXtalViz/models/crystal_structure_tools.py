from mantid.kernel import V3D

from mantid.geometry import (
    CrystalStructure,
    ReflectionGenerator,
    ReflectionConditionFilter,
    PointGroup,
    PointGroupFactory,
    SpaceGroupFactory,
)

from mantid.simpleapi import (
    CreateSampleWorkspace,
    LoadCIF,
    SaveINS,
    SetSample,
    SetUB,
    mtd,
)

import numpy as np
import scipy.linalg

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
