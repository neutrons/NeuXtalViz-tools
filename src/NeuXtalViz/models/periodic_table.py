from mantid.kernel import Atom

from NeuXtalViz.config.atoms import isotopes, names


class PeriodicTableModel:
    """
    Model holding the currently selected periodic table element/isotope.

    Parameters
    ----------
    atom : str
        Chemical symbol of the initially selected element (e.g. ``"H"``).

    Attributes
    ----------
    value : str
        Chemical symbol of the currently selected element.

    """

    def __init__(self, atom):
        """
        Initialize the model with the given atom selection.

        Parameters
        ----------
        atom : str
            Chemical symbol of the initially selected element.

        """

        self.value = atom

    def get_atom_model(self, atm):
        """
        Create an :class:`AtomModel` for the given element symbol.

        Parameters
        ----------
        atm : str
            Chemical symbol of the element (e.g. ``"H"``).

        Returns
        -------
        atom_model : AtomModel
            Model containing isotope and name data for the element.

        """

        return AtomModel(atm)


class AtomModel:
    """
    Model providing isotope, neutron scattering, and identity data for an element.

    Parameters
    ----------
    atm : str, optional
        Chemical symbol of the element. Default is ``"H"``.

    Attributes
    ----------
    atm : str
        Chemical symbol of the element.
    isotopes : list
        Mass numbers of the isotopes available for this element.
    name : str
        Full element name.
    atom_dict : dict
        Isotope-specific atomic data, populated by :meth:`generate_data`.
    neutron_dict : dict
        Isotope-specific neutron scattering data, populated by
        :meth:`generate_data`.

    """

    def __init__(self, atm="H"):
        """
        Initialize the atom model with symbol, isotopes, and name.

        Parameters
        ----------
        atm : str, optional
            Chemical symbol of the element. Default is ``"H"``.

        """

        self.atm, self.isotopes = atm, isotopes.get(atm)

        self.name = names[atm]

    def get_symbol_name(self):
        """
        Return the chemical symbol and full name of the element.

        Returns
        -------
        atm : str
            Chemical symbol of the element.
        name : str
            Full element name.

        """

        return self.atm, self.name

    def get_isotope_numbers(self):
        """
        Return the available isotope mass numbers for the element.

        Returns
        -------
        isotopes : list
            Mass numbers of the isotopes available for this element.

        """

        return self.isotopes

    def generate_data(self, iso):
        """
        Compute atomic and neutron scattering data for a given isotope.

        Populates the `atom_dict` and `neutron_dict` attributes.

        Parameters
        ----------
        iso : int
            Mass number of the isotope to look up.

        """

        atom = Atom(self.atm, iso)

        self.atom_dict = {
            "mass_number": atom.a_number,
            "abundance": atom.abundance,
            "mass": atom.mass,
            "z": atom.z_number,
        }

        neutron = atom.neutron()

        self.neutron_dict = {
            "sigma_coh": neutron["coh_scatt_xs"],
            "sigma_inc": neutron["inc_scatt_xs"],
            "sigma_tot": neutron["tot_scatt_xs"],
            "sigma_abs": neutron["abs_xs"],
            "b_coh_re": neutron["coh_scatt_length_real"],
            "b_coh_im": neutron["coh_scatt_length_img"],
            "b_inc_re": neutron["inc_scatt_length_real"],
            "b_inc_im": neutron["inc_scatt_length_img"],
        }
