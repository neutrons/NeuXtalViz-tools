import numpy as np
import qtawesome as qta

from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QButtonGroup,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QGridLayout,
)

from qtpy.QtCore import Qt, Signal

from NeuXtalViz.config.atoms import indexing, groups, isotopes

colors = {
    "Transition Metals": "#A1C9F4",  # blue
    "Alkaline Earth Metals": "#FFB482",  # orange
    "Nonmetals": "#8DE5A1",  # green
    "Alkali Metals": "#FF9F9B",  # red
    "Lanthanides": "#D0BBFF",  # purple
    "Metalloids": "#DEBB9B",  # brown
    "Actinides": "#FAB0E4",  # pink
    "Other Metals": "#CFCFCF",  # gray
    "Halogens": "#FFFEA3",  # yellow
    "Noble Gases": "#B9F2F0",  # cyan
}


class PeriodicTableView(QWidget):
    """
    View for displaying and selecting elements from the periodic table in
    NeuXtalViz.

    Provides a user interface for element selection, highlighting, and
    displaying element properties.
    """

    selection = Signal(str)
    """Signal(str): Emitted with the selected element symbol on close."""

    def __init__(self, parent=None):
        """
        Initialize the periodic table view and build the button grid.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None.
        """
        super().__init__(parent)

        layout = QHBoxLayout()

        table = self.__init_table()

        layout.addLayout(table)

        self.setLayout(layout)

        self.value = "H"

    def __init_table(self):
        """
        Build the grid of row/column labels and element buttons.

        Creates a mutually-exclusive button group with one button per
        element, positioned according to `indexing`, colored by group
        membership using `colors`, and disabled for elements with no
        isotope data.

        Returns
        -------
        table : QGridLayout
            Layout containing the periodic table row/column labels and
            element buttons.
        """
        table = QGridLayout()

        for row in range(7):
            label = QLabel(str(row + 1))
            table.addWidget(label, row + 1, 0, Qt.AlignCenter)

        for col in range(18):
            label = QLabel(str(col + 1))
            table.addWidget(label, 0, col + 1, Qt.AlignCenter)

        self.atom_buttons = QButtonGroup()
        self.atom_buttons.setExclusive(True)

        for key in indexing.keys():
            row, col = indexing[key]
            button = QPushButton(key, self)
            button.setFixedSize(50, 50)
            group = groups.get(key)
            if group is not None:
                color = colors[group]
                button.setStyleSheet("background-color: {}".format(color))
            if isotopes.get(key) is None:
                button.setDisabled(True)
            self.atom_buttons.addButton(button)
            table.addWidget(button, row, col)

        return table

    def get_atom_view(self):
        """
        Create a new atom/isotope selection view.

        Returns
        -------
        atom_view : AtomView
            Newly constructed atom detail/isotope selection widget.
        """
        return AtomView()

    def connect_atoms(self, atom_info):
        """
        Connect element button clicks to a handler.

        Parameters
        ----------
        atom_info : callable
            Function called with the clicked element's symbol when an
            element button is clicked.
        """
        self.atom_info = atom_info

        self.atom_buttons.buttonClicked.connect(self.show_atom_dialog)

    def show_atom_dialog(self, button):
        """
        Invoke the atom info callback for the clicked element button.

        Parameters
        ----------
        button : QPushButton
            Button that was clicked, whose text is the element symbol.

        Returns
        -------
        result : object
            Return value of the `atom_info` callback connected via
            :meth:`connect_atoms`.
        """
        return self.atom_info(button.text())

    def closeEvent(self, event):
        """
        Emit the current selection when the widget is closed.

        Parameters
        ----------
        event : QCloseEvent
            Close event triggering the shutdown; accepted unconditionally.
        """
        self.selection.emit(self.value)
        event.accept()

    def connect_selected(self, value):
        """
        Connect a callback to the selection signal.

        Parameters
        ----------
        value : callable
            Function called with the selected element symbol when the
            selection signal is emitted.
        """
        self.selection.connect(value)


class AtomView(QWidget):
    """
    View for displaying isotope and neutron scattering data for an element.

    Provides a user interface for selecting an isotope and displaying the
    corresponding atomic mass, abundance, and neutron scattering
    parameters.
    """

    selection = Signal(str)
    """Signal(str): Emitted with the selected isotope symbol on close."""

    def __init__(self):
        """Initialize the atom view and build its widgets and layout."""
        super().__init__()

        card = QGridLayout()

        self.z_label = QLabel("1")
        self.symbol_label = QLabel("H")
        self.name_label = QLabel("Hydrogen")
        self.mass_label = QLabel("1.007825")
        self.abundance_label = QLabel("99.9885")

        self.sigma_coh_label = QLabel("")
        self.sigma_inc_label = QLabel("")
        self.sigma_tot_label = QLabel("")
        self.b_coh_label = QLabel("")
        self.b_inc_label = QLabel("")

        self.isotope_combo = QComboBox(self)

        self.select_button = QPushButton("Use Isotope", self)
        self.select_button.setIcon(qta.icon("fa6s.atom"))

        card.addWidget(self.z_label, 0, 0, Qt.AlignCenter)
        card.addWidget(self.isotope_combo, 0, 1, 1, 2)
        card.addWidget(self.symbol_label, 1, 1, 1, 2, Qt.AlignCenter)
        card.addWidget(self.name_label, 2, 1, 1, 2, Qt.AlignCenter)
        card.addWidget(self.mass_label, 3, 1, 1, 2, Qt.AlignCenter)

        card.addWidget(self.abundance_label, 0, 3)
        card.addWidget(self.sigma_tot_label, 1, 3)
        card.addWidget(self.sigma_coh_label, 2, 3)
        card.addWidget(self.sigma_inc_label, 3, 3)
        card.addWidget(self.b_coh_label, 2, 4)
        card.addWidget(self.b_inc_label, 3, 4)
        card.addWidget(self.select_button, 0, 4)

        self.setLayout(card)

    def closeEvent(self, event):
        """
        Emit the current isotope selection when the widget is closed.

        Parameters
        ----------
        event : QCloseEvent
            Close event triggering the shutdown; accepted unconditionally.
        """
        self.selection.emit(self.get_selection())
        event.accept()

    def connect_selected(self, value):
        """
        Connect a callback to the selection signal.

        Parameters
        ----------
        value : callable
            Function called with the selected isotope symbol when the
            selection signal is emitted.
        """
        self.selection.connect(value)

    def connect_isotopes(self, update_info):
        """
        Connect a callback to isotope combo box changes.

        Parameters
        ----------
        update_info : callable
            Function called when the selected isotope combo box index
            changes.
        """
        self.isotope_combo.currentIndexChanged.connect(update_info)

    def connect_selection(self, use_isotope):
        """
        Connect a callback to the "Use Isotope" button.

        Parameters
        ----------
        use_isotope : callable
            Function called when the select button is clicked.
        """
        self.select_button.clicked.connect(use_isotope)

    def set_symbol_name(self, symbol, name):
        """
        Set the displayed element symbol and name.

        Parameters
        ----------
        symbol : str
            Chemical symbol of the element.
        name : str
            Full name of the element.
        """
        self.symbol_label.setText(symbol)
        self.name_label.setText(name)

    def get_selection(self):
        """
        Return the currently displayed element symbol.

        Returns
        -------
        isotope : str
            Chemical symbol shown in the symbol label.
        """
        isotope = self.symbol_label.text()  # +self.isotope_combo.currentText()

        return isotope

    def set_isotope_numbers(self, numbers):
        """
        Populate the isotope combo box with the given mass numbers.

        Parameters
        ----------
        numbers : list or None
            Mass numbers of the available isotopes. If None, the combo
            box is left empty.
        """
        self.isotope_combo.clear()
        if numbers is not None:
            self.isotope_combo.addItems(np.array(numbers).astype(str).tolist())

    def get_isotope(self):
        """
        Return the mass number of the currently selected isotope.

        Returns
        -------
        iso : int or None
            Mass number of the selected isotope, 0 if no specific
            isotope is selected (empty combo box text), or None if the
            combo box text itself is None.
        """
        iso = self.isotope_combo.currentText()
        if iso is not None:
            return 0 if iso == "" else int(iso)

    def set_atom_parameters(self, atom, scatt):
        """
        Update the displayed atomic and neutron scattering parameters.

        Parameters
        ----------
        atom : dict
            Atomic data with keys ``"z"``, ``"mass"``, and
            ``"abundance"``.
        scatt : dict
            Neutron scattering data with keys ``"sigma_coh"``,
            ``"sigma_inc"``, ``"sigma_tot"``, ``"b_coh_re"``,
            ``"b_coh_im"``, ``"b_inc_re"``, and ``"b_inc_im"``.
        """
        self.z_label.setText(str(atom["z"]))
        self.mass_label.setText(str(atom["mass"]))
        self.abundance_label.setText(str(atom["abundance"]))

        self.sigma_coh_label.setText("σ(coh) = {}".format(scatt["sigma_coh"]))
        self.sigma_inc_label.setText("σ(inc) = {}".format(scatt["sigma_inc"]))
        self.sigma_tot_label.setText("σ(tot) = {}".format(scatt["sigma_tot"]))

        self.b_coh_label.setText(
            "b(coh) = {}+{}i".format(scatt["b_coh_re"], scatt["b_coh_im"])
        )
        self.b_inc_label.setText(
            "b(inc) = {}+{}i".format(scatt["b_inc_re"], scatt["b_inc_im"])
        )
