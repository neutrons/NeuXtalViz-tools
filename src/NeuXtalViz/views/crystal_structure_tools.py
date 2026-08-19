import os

import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QLabel,
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QTabWidget,
    QFileDialog,
)

from qtpy.QtGui import QDoubleValidator
from qtpy.QtCore import Qt

import pyvista as pv

import matplotlib.colors

from NeuXtalViz.config.atoms import colors, radii
from NeuXtalViz.views.periodic_table import PeriodicTableView
from NeuXtalViz.views.base_view import NeuXtalVizWidget

import qtawesome as qta


class CrystalStructureView(NeuXtalVizWidget):
    """
    View for visualizing and editing crystal structures in NeuXtalViz.

    Provides user interface elements for entering lattice parameters,
    selecting crystal system and space group, loading/saving structure
    files, and visualizing atomic positions and structure factors.
    """

    def __init__(self, parent=None):
        """
        Initialize the crystal structure view and build its tabs.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (default None).

        """

        super().__init__(parent)

        self.tab_widget = QTabWidget(self)

        self.structure_tab()
        self.factors_tab()
        self.absorption_tab()
        self.simulator_tab()

        self.layout().addWidget(self.tab_widget, stretch=1)

    def structure_tab(self):
        """
        Build the "Structure" tab layout and widgets.

        Constructs the lattice parameter fields, crystal system/space
        group/setting combo boxes, CIF load/INS save buttons, the atom
        site table, atom entry fields, and chemical formula display, and
        adds them to the tab widget.

        """

        struct_tab = QWidget()
        self.tab_widget.addTab(struct_tab, "Structure")

        structure_layout = QVBoxLayout()

        crystal_layout = QHBoxLayout()
        parameters_layout = QGridLayout()

        self.a_line = QLineEdit()
        self.b_line = QLineEdit()
        self.c_line = QLineEdit()

        self.alpha_line = QLineEdit()
        self.beta_line = QLineEdit()
        self.gamma_line = QLineEdit()

        self.a_line.setPlaceholderText("Å")
        self.b_line.setPlaceholderText("Å")
        self.c_line.setPlaceholderText("Å")
        self.alpha_line.setPlaceholderText("°")
        self.beta_line.setPlaceholderText("°")
        self.gamma_line.setPlaceholderText("°")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.1, 1000, 4, notation=notation)

        self.a_line.setValidator(validator)
        self.b_line.setValidator(validator)
        self.c_line.setValidator(validator)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(10, 170, 4, notation=notation)

        self.alpha_line.setValidator(validator)
        self.beta_line.setValidator(validator)
        self.gamma_line.setValidator(validator)

        a_label = QLabel("a")
        b_label = QLabel("b")
        c_label = QLabel("c")

        alpha_label = QLabel("α")
        beta_label = QLabel("β")
        gamma_label = QLabel("γ")

        angstrom_label = QLabel("Å")
        degree_label = QLabel("°")

        parameters_layout.addWidget(a_label, 0, 0)
        parameters_layout.addWidget(self.a_line, 0, 1)
        parameters_layout.addWidget(b_label, 0, 2)
        parameters_layout.addWidget(self.b_line, 0, 3)
        parameters_layout.addWidget(c_label, 0, 4)
        parameters_layout.addWidget(self.c_line, 0, 5)
        parameters_layout.addWidget(angstrom_label, 0, 6)
        parameters_layout.addWidget(alpha_label, 1, 0)
        parameters_layout.addWidget(self.alpha_line, 1, 1)
        parameters_layout.addWidget(beta_label, 1, 2)
        parameters_layout.addWidget(self.beta_line, 1, 3)
        parameters_layout.addWidget(gamma_label, 1, 4)
        parameters_layout.addWidget(self.gamma_line, 1, 5)
        parameters_layout.addWidget(degree_label, 1, 6)

        self.crystal_system_combo = QComboBox(self)
        self.crystal_system_combo.addItem("Triclinic")
        self.crystal_system_combo.addItem("Monoclinic")
        self.crystal_system_combo.addItem("Orthorhombic")
        self.crystal_system_combo.addItem("Tetragonal")
        self.crystal_system_combo.addItem("Trigonal")
        self.crystal_system_combo.addItem("Hexagonal")
        self.crystal_system_combo.addItem("Cubic")

        self.space_group_combo = QComboBox(self)
        self.setting_combo = QComboBox(self)

        self.crystal_system_combo.setEnabled(False)
        self.space_group_combo.setEnabled(False)
        self.setting_combo.setEnabled(False)

        self.auto_scale_dropdown(self.crystal_system_combo)
        self.auto_scale_dropdown(self.space_group_combo)
        self.auto_scale_dropdown(self.setting_combo)

        self.load_CIF_button = QPushButton("Load CIF", self)
        self.load_CIF_button.setIcon(qta.icon("fa6s.folder-open"))

        self.save_INS_button = QPushButton("Save INS", self)
        self.save_INS_button.setIcon(qta.icon("fa6s.floppy-disk"))

        crystal_layout.addWidget(self.crystal_system_combo)
        crystal_layout.addWidget(self.space_group_combo)
        crystal_layout.addWidget(self.setting_combo)
        crystal_layout.addWidget(self.load_CIF_button)
        crystal_layout.addWidget(self.save_INS_button)

        structure_layout.addLayout(crystal_layout)
        structure_layout.addLayout(parameters_layout)

        stretch = QHeaderView.Stretch

        self.atm_table = QTableWidget()

        self.atm_table.setRowCount(0)
        self.atm_table.setColumnCount(6)

        self.atm_table.horizontalHeader().setSectionResizeMode(stretch)
        self.atm_table.setHorizontalHeaderLabels(
            ["atm", "x", "y", "z", "occ", "U"]
        )
        self.atm_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.atm_table.setSelectionBehavior(QTableWidget.SelectRows)

        structure_layout.addWidget(self.atm_table)

        scatterer_layout = QHBoxLayout()

        self.atm_button = QPushButton("", self)
        self.atm_button.setIcon(qta.icon("fa6s.atom"))

        self.x_line = QLineEdit()
        self.y_line = QLineEdit()
        self.z_line = QLineEdit()
        self.occ_line = QLineEdit()
        self.Uiso_line = QLineEdit()

        self.x_line.setPlaceholderText("x")
        self.y_line.setPlaceholderText("y")
        self.z_line.setPlaceholderText("z")
        self.occ_line.setPlaceholderText("1.0")
        self.Uiso_line.setPlaceholderText("Uiso")

        validator = QDoubleValidator(-1, 1, 4, notation=notation)

        self.x_line.setValidator(validator)
        self.y_line.setValidator(validator)
        self.z_line.setValidator(validator)

        validator = QDoubleValidator(0, 1, 4, notation=notation)

        self.occ_line.setValidator(validator)

        validator = QDoubleValidator(0, 100, 4, notation=notation)

        self.Uiso_line.setValidator(validator)

        scatterer_layout.addWidget(self.atm_button)
        scatterer_layout.addWidget(self.x_line)
        scatterer_layout.addWidget(self.y_line)
        scatterer_layout.addWidget(self.z_line)
        scatterer_layout.addWidget(self.occ_line)
        scatterer_layout.addWidget(self.Uiso_line)

        sample_layout = QHBoxLayout()

        self.chem_line = QLineEdit()
        self.Z_line = QLineEdit()
        self.V_line = QLineEdit()

        self.chem_line.setReadOnly(True)
        self.Z_line.setReadOnly(True)
        self.V_line.setReadOnly(True)

        self.chem_line.setPlaceholderText("Chemical formula")
        self.Z_line.setPlaceholderText("Z")
        self.V_line.setPlaceholderText("Ω (Å³)")

        Z_label = QLabel("Z")
        V_label = QLabel("Ω")
        uc_vol_label = QLabel("Å^3")

        sample_layout.addWidget(self.chem_line)
        sample_layout.addWidget(Z_label)
        sample_layout.addWidget(self.Z_line)
        sample_layout.addWidget(V_label)
        sample_layout.addWidget(self.V_line)
        sample_layout.addWidget(uc_vol_label)

        structure_layout.addLayout(scatterer_layout)
        structure_layout.addLayout(sample_layout)

        struct_tab.setLayout(structure_layout)

        self.a_line.setToolTip(
            "Lattice parameter a (Å). Enter a positive real value."
        )
        self.b_line.setToolTip(
            "Lattice parameter b (Å). Enter a positive real value."
        )
        self.c_line.setToolTip(
            "Lattice parameter c (Å). Enter a positive real value."
        )
        self.alpha_line.setToolTip(
            "Lattice angle α (degrees). Enter a value between 10 and 170."
        )
        self.beta_line.setToolTip(
            "Lattice angle β (degrees). Enter a value between 10 and 170."
        )
        self.gamma_line.setToolTip(
            "Lattice angle γ (degrees). Enter a value between 10 and 170."
        )
        self.crystal_system_combo.setToolTip("Select the crystal system.")
        self.space_group_combo.setToolTip("Select the space group.")
        self.setting_combo.setToolTip("Select the space group setting.")
        self.load_CIF_button.setToolTip(
            "Load a crystal structure from a CIF file."
        )
        self.save_INS_button.setToolTip(
            "Save the current structure as an INS file."
        )
        self.atm_table.setToolTip("Table of atomic positions and parameters.")
        self.x_line.setToolTip(
            "Fractional x coordinate for the selected atom."
        )
        self.y_line.setToolTip(
            "Fractional y coordinate for the selected atom."
        )
        self.z_line.setToolTip(
            "Fractional z coordinate for the selected atom."
        )
        self.occ_line.setToolTip("Occupancy for the selected atom (0 to 1).")
        self.Uiso_line.setToolTip(
            "Isotropic displacement parameter U (0 to 100)."
        )
        self.chem_line.setToolTip("Chemical formula of the unit cell.")
        self.Z_line.setToolTip("Number of formula units per unit cell (Z).")
        self.V_line.setToolTip("Unit cell volume (Å³).")

    def factors_tab(self):
        """
        Build the "Factors" tab layout and widgets.

        Constructs the minimum d-spacing field and calculate button, the
        table of calculated structure factors, and the individual hkl
        entry fields with their calculate button, and adds them to the
        tab widget.

        """

        fact_tab = QWidget()
        self.tab_widget.addTab(fact_tab, "Factors")

        factors_layout = QVBoxLayout()

        calculate_layout = QHBoxLayout()

        dmin_label = QLabel("d(min)")
        angstrom_label = QLabel("Å")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.1, 1000, 4, notation=notation)

        self.dmin_line = QLineEdit()
        self.dmin_line.setValidator(validator)
        self.dmin_line.setPlaceholderText("d-min (Å)")

        self.calculate_button = QPushButton("Calculate", self)
        self.calculate_button.setIcon(qta.icon("fa6s.calculator"))

        calculate_layout.addWidget(dmin_label)
        calculate_layout.addWidget(self.dmin_line)
        calculate_layout.addWidget(angstrom_label)
        calculate_layout.addStretch(1)
        calculate_layout.addWidget(self.calculate_button)

        stretch = QHeaderView.Stretch

        self.f2_table = QTableWidget()

        self.f2_table.setRowCount(0)
        self.f2_table.setColumnCount(5)

        self.f2_table.horizontalHeader().setSectionResizeMode(stretch)
        self.f2_table.setHorizontalHeaderLabels(["h", "k", "l", "d", "F²"])
        self.f2_table.setEditTriggers(QTableWidget.NoEditTriggers)

        indivdual_layout = QHBoxLayout()

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.h_line = QLineEdit()
        self.k_line = QLineEdit()
        self.l_line = QLineEdit()

        self.h_line.setValidator(validator)
        self.k_line.setValidator(validator)
        self.l_line.setValidator(validator)

        self.h_line.setPlaceholderText("h")
        self.k_line.setPlaceholderText("k")
        self.l_line.setPlaceholderText("l")

        self.individual_button = QPushButton("Calculate", self)
        self.individual_button.setIcon(qta.icon("fa6s.calculator"))

        hkl_label = QLabel("hkl")

        indivdual_layout.addWidget(hkl_label)
        indivdual_layout.addWidget(self.h_line)
        indivdual_layout.addWidget(self.k_line)
        indivdual_layout.addWidget(self.l_line)
        indivdual_layout.addStretch(1)
        indivdual_layout.addWidget(self.individual_button)

        factors_layout.addLayout(calculate_layout)
        factors_layout.addWidget(self.f2_table)
        factors_layout.addLayout(indivdual_layout)

        fact_tab.setLayout(factors_layout)

        self.dmin_line.setToolTip(
            "Minimum d-spacing (Å) for structure factor calculation."
        )
        self.calculate_button.setToolTip(
            "Calculate structure factors for all reflections."
        )
        self.f2_table.setToolTip("Table of calculated structure factors (F²).")
        self.h_line.setToolTip(
            "h index for individual structure factor calculation."
        )
        self.k_line.setToolTip(
            "k index for individual structure factor calculation."
        )
        self.l_line.setToolTip(
            "l index for individual structure factor calculation."
        )
        self.individual_button.setToolTip(
            "Calculate structure factor for the specified hkl."
        )

    def absorption_tab(self):
        """
        Build the "Absorption" tab layout and widgets.

        Constructs the ellipsoid dimension fields, wavelength and
        d-min fields with a calculate button, the transmission results
        table, the beam-/shape-orientation vector fields, the
        CIF-derived (read-only) material fields, and the
        absorption/scattering info panel, and adds them to the tab
        widget.
        """

        abs_tab = QWidget()
        self.tab_widget.addTab(abs_tab, "Absorption")

        abs_layout = QVBoxLayout()

        notation = QDoubleValidator.StandardNotation

        # --- Ellipsoid dimensions --------------------------------------
        validator = QDoubleValidator(0, 100, 5, notation=notation)

        param1_label = QLabel("Thickness", self)
        param2_label = QLabel("Width", self)
        param3_label = QLabel("Height", self)

        unit_label = QLabel("mm", self)

        self.abs_param1_line = QLineEdit("1.0")
        self.abs_param2_line = QLineEdit("1.0")
        self.abs_param3_line = QLineEdit("1.0")

        self.abs_param1_line.setValidator(validator)
        self.abs_param2_line.setValidator(validator)
        self.abs_param3_line.setValidator(validator)

        shape_layout = QHBoxLayout()
        shape_layout.addWidget(param1_label)
        shape_layout.addWidget(self.abs_param1_line)
        shape_layout.addWidget(param2_label)
        shape_layout.addWidget(self.abs_param2_line)
        shape_layout.addWidget(param3_label)
        shape_layout.addWidget(self.abs_param3_line)
        shape_layout.addWidget(unit_label)
        shape_layout.addStretch(1)

        # --- Wavelength / d-min / Calculate (same row as shape) --------
        wavelength_label = QLabel("λ", self)
        dmin_label = QLabel("d(min)", self)
        angstrom_label = QLabel("Å", self)

        self.abs_wavelength_line = QLineEdit("1.5", self)
        self.abs_wavelength_line.setValidator(
            QDoubleValidator(0.1, 20, 4, notation=notation)
        )

        self.abs_dmin_line = QLineEdit(self)
        self.abs_dmin_line.setValidator(
            QDoubleValidator(0.1, 1000, 4, notation=notation)
        )
        self.abs_dmin_line.setPlaceholderText("d-min (Å)")

        self.abs_calculate_button = QPushButton("Calculate", self)
        self.abs_calculate_button.setIcon(qta.icon("fa6s.calculator"))

        shape_layout.addWidget(wavelength_label)
        shape_layout.addWidget(self.abs_wavelength_line)
        shape_layout.addWidget(dmin_label)
        shape_layout.addWidget(self.abs_dmin_line)
        shape_layout.addWidget(angstrom_label)
        shape_layout.addWidget(self.abs_calculate_button)

        # --- Results table ---------------------------------------------
        stretch = QHeaderView.Stretch

        self.abs_table = QTableWidget()
        self.abs_table.setRowCount(0)
        self.abs_table.setColumnCount(6)
        self.abs_table.horizontalHeader().setSectionResizeMode(stretch)
        self.abs_table.setHorizontalHeaderLabels(
            ["h", "k", "l", "d", "T", "T-bar"]
        )
        self.abs_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # --- Sample orientation (defines the crystal's own U matrix) ----
        a_star_label1 = QLabel("a*", self)
        b_star_label1 = QLabel("b*", self)
        c_star_label1 = QLabel("c*", self)

        sample_orient_label = QLabel("Sample Orientation", self)
        sample_u_label = QLabel("Beam Direction:", self)
        sample_v_label = QLabel("In-plane Direction:", self)

        self.abs_sample_hu_line = QLineEdit("0")
        self.abs_sample_ku_line = QLineEdit("0")
        self.abs_sample_lu_line = QLineEdit("1")

        self.abs_sample_hv_line = QLineEdit("1")
        self.abs_sample_kv_line = QLineEdit("0")
        self.abs_sample_lv_line = QLineEdit("0")

        sample_orient_layout = QGridLayout()

        sample_orient_layout.addWidget(
            sample_orient_label, 0, 0, Qt.AlignCenter
        )
        sample_orient_layout.addWidget(a_star_label1, 0, 1, Qt.AlignCenter)
        sample_orient_layout.addWidget(b_star_label1, 0, 2, Qt.AlignCenter)
        sample_orient_layout.addWidget(c_star_label1, 0, 3, Qt.AlignCenter)

        sample_orient_layout.addWidget(sample_u_label, 1, 0)
        sample_orient_layout.addWidget(self.abs_sample_hu_line, 1, 1)
        sample_orient_layout.addWidget(self.abs_sample_ku_line, 1, 2)
        sample_orient_layout.addWidget(self.abs_sample_lu_line, 1, 3)
        sample_orient_layout.addWidget(sample_v_label, 2, 0)
        sample_orient_layout.addWidget(self.abs_sample_hv_line, 2, 1)
        sample_orient_layout.addWidget(self.abs_sample_kv_line, 2, 2)
        sample_orient_layout.addWidget(self.abs_sample_lv_line, 2, 3)

        # --- Shape orientation (orients the shape mesh) ------------------
        a_star_label2 = QLabel("a*", self)
        b_star_label2 = QLabel("b*", self)
        c_star_label2 = QLabel("c*", self)

        shape_orient_label = QLabel("Shape Orientation", self)
        shape_u_label = QLabel("Along Thickness:", self)
        shape_v_label = QLabel("In-plane Lateral:", self)

        self.abs_shape_hu_line = QLineEdit("0")
        self.abs_shape_ku_line = QLineEdit("0")
        self.abs_shape_lu_line = QLineEdit("1")

        self.abs_shape_hv_line = QLineEdit("1")
        self.abs_shape_kv_line = QLineEdit("0")
        self.abs_shape_lv_line = QLineEdit("0")

        shape_orient_layout = QGridLayout()

        shape_orient_layout.addWidget(shape_orient_label, 0, 0, Qt.AlignCenter)
        shape_orient_layout.addWidget(a_star_label2, 0, 1, Qt.AlignCenter)
        shape_orient_layout.addWidget(b_star_label2, 0, 2, Qt.AlignCenter)
        shape_orient_layout.addWidget(c_star_label2, 0, 3, Qt.AlignCenter)

        shape_orient_layout.addWidget(shape_u_label, 1, 0)
        shape_orient_layout.addWidget(self.abs_shape_hu_line, 1, 1)
        shape_orient_layout.addWidget(self.abs_shape_ku_line, 1, 2)
        shape_orient_layout.addWidget(self.abs_shape_lu_line, 1, 3)
        shape_orient_layout.addWidget(shape_v_label, 2, 0)
        shape_orient_layout.addWidget(self.abs_shape_hv_line, 2, 1)
        shape_orient_layout.addWidget(self.abs_shape_kv_line, 2, 2)
        shape_orient_layout.addWidget(self.abs_shape_lv_line, 2, 3)

        # --- Material (read-only, auto-filled from the loaded CIF) ------
        material_layout = QHBoxLayout()

        self.abs_chem_line = QLineEdit()
        self.abs_Z_line = QLineEdit()
        self.abs_V_line = QLineEdit()

        self.abs_chem_line.setReadOnly(True)
        self.abs_Z_line.setReadOnly(True)
        self.abs_V_line.setReadOnly(True)

        self.abs_chem_line.setPlaceholderText("Chemical formula")
        self.abs_Z_line.setPlaceholderText("Z")
        self.abs_V_line.setPlaceholderText("Ω (Å³)")

        Z_label = QLabel("Z")
        V_label = QLabel("Ω")
        uc_vol_label = QLabel("Å^3")

        material_layout.addWidget(self.abs_chem_line)
        material_layout.addWidget(Z_label)
        material_layout.addWidget(self.abs_Z_line)
        material_layout.addWidget(V_label)
        material_layout.addWidget(self.abs_V_line)
        material_layout.addWidget(uc_vol_label)

        # --- Absorption / scattering info panel (read-only) --------------
        cryst_layout = QGridLayout()

        scattering_label = QLabel("Scattering", self)
        absorption_label = QLabel("Absorption", self)

        sigma_label = QLabel("σ", self)
        mu_label = QLabel("μ", self)

        sigma_unit_label = QLabel("barn", self)
        mu_unit_label = QLabel("1/cm", self)

        self.abs_sigma_a_line = QLineEdit()
        self.abs_sigma_s_line = QLineEdit()

        self.abs_mu_a_line = QLineEdit()
        self.abs_mu_s_line = QLineEdit()

        self.abs_sigma_a_line.setReadOnly(True)
        self.abs_sigma_s_line.setReadOnly(True)
        self.abs_mu_a_line.setReadOnly(True)
        self.abs_mu_s_line.setReadOnly(True)

        cryst_layout.addWidget(scattering_label, 0, 1, Qt.AlignCenter)
        cryst_layout.addWidget(absorption_label, 0, 2, Qt.AlignCenter)

        cryst_layout.addWidget(sigma_label, 1, 0)
        cryst_layout.addWidget(self.abs_sigma_a_line, 1, 1)
        cryst_layout.addWidget(self.abs_sigma_s_line, 1, 2)
        cryst_layout.addWidget(sigma_unit_label, 1, 3)

        cryst_layout.addWidget(mu_label, 2, 0)
        cryst_layout.addWidget(self.abs_mu_a_line, 2, 1)
        cryst_layout.addWidget(self.abs_mu_s_line, 2, 2)
        cryst_layout.addWidget(mu_unit_label, 2, 3)

        N_label = QLabel("N", self)
        M_label = QLabel("M", self)
        n_label = QLabel("n", self)
        rho_label = QLabel("rho", self)
        v_label = QLabel("V", self)
        m_label = QLabel("m", self)

        N_unit_label = QLabel("atoms", self)
        M_unit_label = QLabel("g/mol", self)
        n_unit_label = QLabel("1/Å^3", self)
        rho_unit_label = QLabel("g/cm^3", self)
        v_unit_label = QLabel("cm^3", self)
        m_unit_label = QLabel("g", self)

        self.abs_N_line = QLineEdit()
        self.abs_M_line = QLineEdit()
        self.abs_n_line = QLineEdit()
        self.abs_rho_line = QLineEdit()
        self.abs_v_line = QLineEdit()
        self.abs_m_line = QLineEdit()

        for w in (
            self.abs_N_line,
            self.abs_M_line,
            self.abs_n_line,
            self.abs_rho_line,
            self.abs_v_line,
            self.abs_m_line,
        ):
            w.setReadOnly(True)

        cryst_layout.addWidget(N_label, 3, 0)
        cryst_layout.addWidget(self.abs_N_line, 3, 1)
        cryst_layout.addWidget(N_unit_label, 3, 2)

        cryst_layout.addWidget(M_label, 4, 0)
        cryst_layout.addWidget(self.abs_M_line, 4, 1)
        cryst_layout.addWidget(M_unit_label, 4, 2)

        cryst_layout.addWidget(n_label, 5, 0)
        cryst_layout.addWidget(self.abs_n_line, 5, 1)
        cryst_layout.addWidget(n_unit_label, 5, 2)

        cryst_layout.addWidget(rho_label, 6, 0)
        cryst_layout.addWidget(self.abs_rho_line, 6, 1)
        cryst_layout.addWidget(rho_unit_label, 6, 2)

        cryst_layout.addWidget(v_label, 7, 0)
        cryst_layout.addWidget(self.abs_v_line, 7, 1)
        cryst_layout.addWidget(v_unit_label, 7, 2)

        cryst_layout.addWidget(m_label, 8, 0)
        cryst_layout.addWidget(self.abs_m_line, 8, 1)
        cryst_layout.addWidget(m_unit_label, 8, 2)

        abs_layout.addLayout(shape_layout)
        abs_layout.addWidget(self.abs_table)
        abs_layout.addLayout(sample_orient_layout)
        abs_layout.addLayout(shape_orient_layout)
        abs_layout.addLayout(material_layout)
        abs_layout.addLayout(cryst_layout)

        abs_tab.setLayout(abs_layout)

        self.abs_param1_line.setToolTip("Set the ellipsoid thickness (mm).")
        self.abs_param2_line.setToolTip("Set the ellipsoid width (mm).")
        self.abs_param3_line.setToolTip("Set the ellipsoid height (mm).")
        self.abs_wavelength_line.setToolTip(
            "Incident wavelength (Å) for the monochromatic rotation "
            "experiment."
        )
        self.abs_dmin_line.setToolTip(
            "Minimum d-spacing (Å) for the predicted reflections."
        )
        self.abs_calculate_button.setToolTip(
            "Predict the goniometer setting and transmission/absorption "
            "for every reflection out to d-min."
        )
        self.abs_table.setToolTip(
            "Predicted transmission (T) and absorption-weighted path "
            "length (T-bar, cm) per reflection, sorted by d-spacing."
        )
        self.abs_sample_hu_line.setToolTip(
            "h-index of the crystallographic direction to align with "
            "the beam direction (independent of the shape orientation)."
        )
        self.abs_sample_ku_line.setToolTip(
            "k-index of the beam direction vector."
        )
        self.abs_sample_lu_line.setToolTip(
            "l-index of the beam direction vector."
        )
        self.abs_sample_hv_line.setToolTip(
            "h-index of the in-plane direction vector."
        )
        self.abs_sample_kv_line.setToolTip(
            "k-index of the in-plane direction vector."
        )
        self.abs_sample_lv_line.setToolTip(
            "l-index of the in-plane direction vector."
        )
        self.abs_shape_hu_line.setToolTip(
            "h-index of the crystallographic direction along the "
            "shape's thickness."
        )
        self.abs_shape_ku_line.setToolTip("k-index of the shape U vector.")
        self.abs_shape_lu_line.setToolTip("l-index of the shape U vector.")
        self.abs_shape_hv_line.setToolTip("h-index of the shape V vector.")
        self.abs_shape_kv_line.setToolTip("k-index of the shape V vector.")
        self.abs_shape_lv_line.setToolTip("l-index of the shape V vector.")
        self.abs_chem_line.setToolTip(
            "Chemical formula, auto-filled from the loaded CIF."
        )
        self.abs_Z_line.setToolTip(
            "Z parameter, auto-filled from the loaded CIF."
        )
        self.abs_V_line.setToolTip(
            "Unit cell volume (Å^3), auto-filled from the loaded CIF."
        )
        self.abs_sigma_a_line.setToolTip("Absorption cross-section (barn).")
        self.abs_sigma_s_line.setToolTip("Scattering cross-section (barn).")
        self.abs_mu_a_line.setToolTip("Linear absorption coefficient (1/cm).")
        self.abs_mu_s_line.setToolTip("Linear scattering coefficient (1/cm).")
        self.abs_N_line.setToolTip("Number of atoms in the unit cell.")
        self.abs_M_line.setToolTip("Molar mass of the material (g/mol).")
        self.abs_n_line.setToolTip("Effective number density (1/Å^3).")
        self.abs_rho_line.setToolTip("Mass density (g/cm^3).")
        self.abs_v_line.setToolTip("Sample volume (cm^3).")
        self.abs_m_line.setToolTip("Sample mass (g).")

    def simulator_tab(self):
        """
        Build the "Simulator" tab layout and widgets.

        Constructs the instrument selector and d-min field, the UB
        load/clear controls and beam-/in-plane-direction u/v vector
        fields (used only when no UB has been loaded from file), the
        goniometer angle fields, the counting time field and calculate
        button, and the results table, and adds them to the tab
        widget. The sample shape and its orientation are not
        duplicated here -- they are read directly from the Absorption
        tab (:meth:`get_absorption_shape_constants`,
        :meth:`get_absorption_shape_vectors`) when calculating.
        """

        sim_tab = QWidget()
        self.tab_widget.addTab(sim_tab, "Simulator")

        sim_layout = QVBoxLayout()

        notation = QDoubleValidator.StandardNotation

        # --- Instrument / d-min -----------------------------------------
        instrument_label = QLabel("Instrument", self)
        dmin_label = QLabel("d(min)", self)
        angstrom_label = QLabel("Å", self)

        self.sim_instrument_combo = QComboBox(self)
        self.sim_instrument_combo.addItem("TOPAZ")
        self.sim_instrument_combo.addItem("MANDI")
        self.sim_instrument_combo.addItem("CORELLI")

        self.auto_scale_dropdown(self.sim_instrument_combo)

        self.sim_dmin_line = QLineEdit(self)
        self.sim_dmin_line.setValidator(
            QDoubleValidator(0.1, 1000, 4, notation=notation)
        )
        self.sim_dmin_line.setPlaceholderText("d-min (Å)")

        instrument_layout = QHBoxLayout()
        instrument_layout.addWidget(instrument_label)
        instrument_layout.addWidget(self.sim_instrument_combo)
        instrument_layout.addWidget(dmin_label)
        instrument_layout.addWidget(self.sim_dmin_line)
        instrument_layout.addWidget(angstrom_label)
        instrument_layout.addStretch(1)

        # --- UB load/clear ------------------------------------------------
        self.sim_load_UB_button = QPushButton("Load UB", self)
        self.sim_load_UB_button.setIcon(qta.icon("fa6s.folder-open"))

        self.sim_clear_UB_button = QPushButton("Clear UB", self)
        self.sim_clear_UB_button.setIcon(qta.icon("fa6s.xmark"))

        self.sim_UB_line = QLineEdit(self)
        self.sim_UB_line.setReadOnly(True)
        self.sim_UB_line.setPlaceholderText(
            "No UB loaded -- using u/v vectors below"
        )

        UB_layout = QHBoxLayout()
        UB_layout.addWidget(self.sim_load_UB_button)
        UB_layout.addWidget(self.sim_clear_UB_button)
        UB_layout.addWidget(self.sim_UB_line)

        # --- Sample orientation (u/v vectors, same convention as the
        # Absorption tab's sample-orientation fields) --------------------
        a_star_label = QLabel("a*", self)
        b_star_label = QLabel("b*", self)
        c_star_label = QLabel("c*", self)

        orient_label = QLabel("Sample Orientation", self)
        u_label = QLabel("Beam Direction:", self)
        v_label = QLabel("In-plane Direction:", self)

        self.sim_hu_line = QLineEdit("0")
        self.sim_ku_line = QLineEdit("0")
        self.sim_lu_line = QLineEdit("1")

        self.sim_hv_line = QLineEdit("1")
        self.sim_kv_line = QLineEdit("0")
        self.sim_lv_line = QLineEdit("0")

        orient_layout = QGridLayout()

        orient_layout.addWidget(orient_label, 0, 0, Qt.AlignCenter)
        orient_layout.addWidget(a_star_label, 0, 1, Qt.AlignCenter)
        orient_layout.addWidget(b_star_label, 0, 2, Qt.AlignCenter)
        orient_layout.addWidget(c_star_label, 0, 3, Qt.AlignCenter)

        orient_layout.addWidget(u_label, 1, 0)
        orient_layout.addWidget(self.sim_hu_line, 1, 1)
        orient_layout.addWidget(self.sim_ku_line, 1, 2)
        orient_layout.addWidget(self.sim_lu_line, 1, 3)
        orient_layout.addWidget(v_label, 2, 0)
        orient_layout.addWidget(self.sim_hv_line, 2, 1)
        orient_layout.addWidget(self.sim_kv_line, 2, 2)
        orient_layout.addWidget(self.sim_lv_line, 2, 3)

        # --- Goniometer ----------------------------------------------------
        gon_label = QLabel("Goniometer", self)
        omega_label = QLabel("ω", self)
        chi_label = QLabel("χ", self)
        phi_label = QLabel("φ", self)
        degree_label = QLabel("°", self)

        validator = QDoubleValidator(-360, 360, 4, notation=notation)

        self.sim_omega_line = QLineEdit("0")
        self.sim_chi_line = QLineEdit("0")
        self.sim_phi_line = QLineEdit("0")

        self.sim_omega_line.setValidator(validator)
        self.sim_chi_line.setValidator(validator)
        self.sim_phi_line.setValidator(validator)

        # --- Counting time --------------------------------------------------
        time_label = QLabel("Counting Time", self)
        minute_label = QLabel("min", self)

        self.sim_time_line = QLineEdit("2")
        self.sim_time_line.setValidator(
            QDoubleValidator(0.01, 10000, 4, notation=notation)
        )

        self.sim_calculate_button = QPushButton("Calculate", self)
        self.sim_calculate_button.setIcon(qta.icon("fa6s.calculator"))

        gon_layout = QHBoxLayout()
        gon_layout.addWidget(gon_label)
        gon_layout.addWidget(omega_label)
        gon_layout.addWidget(self.sim_omega_line)
        gon_layout.addWidget(chi_label)
        gon_layout.addWidget(self.sim_chi_line)
        gon_layout.addWidget(phi_label)
        gon_layout.addWidget(self.sim_phi_line)
        gon_layout.addWidget(degree_label)
        gon_layout.addWidget(time_label)
        gon_layout.addWidget(self.sim_time_line)
        gon_layout.addWidget(minute_label)
        gon_layout.addStretch(1)
        gon_layout.addWidget(self.sim_calculate_button)

        # --- Results table ---------------------------------------------
        stretch = QHeaderView.Stretch

        self.sim_table = QTableWidget()
        self.sim_table.setRowCount(0)
        self.sim_table.setColumnCount(8)
        self.sim_table.horizontalHeader().setSectionResizeMode(stretch)
        self.sim_table.setHorizontalHeaderLabels(
            ["h", "k", "l", "d", "λ", "F²", "I", "I/σ"]
        )
        self.sim_table.setEditTriggers(QTableWidget.NoEditTriggers)

        sim_layout.addLayout(instrument_layout)
        sim_layout.addLayout(UB_layout)
        sim_layout.addLayout(orient_layout)
        sim_layout.addLayout(gon_layout)
        sim_layout.addWidget(self.sim_table)

        sim_tab.setLayout(sim_layout)

        self.sim_instrument_combo.setToolTip(
            "Instrument to simulate (determines the wavelength band, "
            "goniometer convention, and calibrated bank response)."
        )
        self.sim_dmin_line.setToolTip(
            "Minimum d-spacing (Å) for the predicted reflections "
            "(defaults to the instrument's nominal minimum)."
        )
        self.sim_load_UB_button.setToolTip(
            "Load a UB matrix from an ISAW UB file, overriding the "
            "u/v vectors below."
        )
        self.sim_clear_UB_button.setToolTip(
            "Clear the loaded UB matrix and use the u/v vectors below "
            "instead."
        )
        self.sim_UB_line.setToolTip(
            "Path of the currently loaded UB file, if any."
        )
        self.sim_hu_line.setToolTip(
            "h-index of the crystallographic direction to align with "
            "the beam direction (ignored if a UB file is loaded)."
        )
        self.sim_ku_line.setToolTip("k-index of the beam direction vector.")
        self.sim_lu_line.setToolTip("l-index of the beam direction vector.")
        self.sim_hv_line.setToolTip(
            "h-index of the in-plane direction vector."
        )
        self.sim_kv_line.setToolTip(
            "k-index of the in-plane direction vector."
        )
        self.sim_lv_line.setToolTip(
            "l-index of the in-plane direction vector."
        )
        self.sim_omega_line.setToolTip("Goniometer ω angle (degrees).")
        self.sim_chi_line.setToolTip("Goniometer χ angle (degrees).")
        self.sim_phi_line.setToolTip("Goniometer φ angle (degrees).")
        self.sim_time_line.setToolTip("Requested counting time (minutes).")
        self.sim_calculate_button.setToolTip(
            "Predict observable reflections and their expected counts "
            "and I/σ for the current instrument, orientation, "
            "goniometer setting, and counting time."
        )
        self.sim_table.setToolTip(
            "Predicted reflections sorted by decreasing d-spacing: "
            "Miller indices, d-spacing (Å), wavelength (Å), squared "
            "structure factor, expected integrated counts, and I/σ."
        )

    def connect_save_INS(self, save_INS):
        """
        Connect the save INS button to a handler.

        Parameters
        ----------
        save_INS : callable
            Slot invoked when the save INS button is clicked.

        """

        self.save_INS_button.clicked.connect(save_INS)

    def connect_group_generator(self, generate_groups):
        """
        Connect crystal system selection to a handler.

        Parameters
        ----------
        generate_groups : callable
            Slot invoked when the crystal system combo box is activated.

        """

        self.crystal_system_combo.activated.connect(generate_groups)

    def connect_setting_generator(self, generate_settings):
        """
        Connect space group selection to a handler.

        Parameters
        ----------
        generate_settings : callable
            Slot invoked when the space group combo box is activated.

        """

        self.space_group_combo.activated.connect(generate_settings)

    def connect_F2_calculator(self, calculate_F2):
        """
        Connect the structure factor calculate button to a handler.

        Parameters
        ----------
        calculate_F2 : callable
            Slot invoked when the calculate button is clicked.

        """

        self.calculate_button.clicked.connect(calculate_F2)

    def connect_hkl_calculator(self, calculate_hkl):
        """
        Connect the individual hkl calculate button to a handler.

        Parameters
        ----------
        calculate_hkl : callable
            Slot invoked when the individual calculate button is
            clicked.

        """

        self.individual_button.clicked.connect(calculate_hkl)

    def connect_calculate_absorption(self, calculate_absorption):
        """
        Connect the absorption calculate button to a handler.

        Parameters
        ----------
        calculate_absorption : callable
            Slot invoked when the absorption calculate button is
            clicked.
        """

        self.abs_calculate_button.clicked.connect(calculate_absorption)

    def connect_row_highligter(self, highlight_row):
        """
        Connect atom table row selection to a handler.

        Parameters
        ----------
        highlight_row : callable
            Slot invoked when the selected item in the atom table
            changes.

        """

        self.atm_table.itemSelectionChanged.connect(highlight_row)

    def connect_lattice_parameters(self, update_parameters):
        """
        Connect lattice parameter edits to a handler.

        Parameters
        ----------
        update_parameters : callable
            Slot invoked when editing finishes on any of the a, b, c,
            alpha, beta, or gamma line edits.

        """

        self.a_line.editingFinished.connect(update_parameters)
        self.b_line.editingFinished.connect(update_parameters)
        self.c_line.editingFinished.connect(update_parameters)
        self.alpha_line.editingFinished.connect(update_parameters)
        self.beta_line.editingFinished.connect(update_parameters)
        self.gamma_line.editingFinished.connect(update_parameters)

    def connect_atom_table(self, set_atom_table):
        """
        Connect atom site field edits to a handler.

        Parameters
        ----------
        set_atom_table : callable
            Slot invoked when editing finishes on any of the x, y, z,
            occupancy, or Uiso line edits for the current atom.

        """

        self.x_line.editingFinished.connect(set_atom_table)
        self.y_line.editingFinished.connect(set_atom_table)
        self.z_line.editingFinished.connect(set_atom_table)
        self.occ_line.editingFinished.connect(set_atom_table)
        self.Uiso_line.editingFinished.connect(set_atom_table)

    def connect_load_CIF(self, load_CIF):
        """
        Connect the load CIF button to a handler.

        Parameters
        ----------
        load_CIF : callable
            Slot invoked when the load CIF button is clicked.

        """

        self.load_CIF_button.clicked.connect(load_CIF)

    def connect_select_isotope(self, select_isotope):
        """
        Connect the atom/isotope button to a handler.

        Parameters
        ----------
        select_isotope : callable
            Slot invoked when the atom selection button is clicked.

        """

        self.atm_button.clicked.connect(select_isotope)

    def draw_cell(self, A):
        """
        Draw the unit cell as a wireframe box in the 3D view.

        Parameters
        ----------
        A : 3x3 array-like
            Transformation matrix mapping the unit cube to the unit
            cell edges.

        """

        T = np.eye(4)
        T[:3, :3] = A

        mesh = pv.Box(bounds=(0, 1, 0, 1, 0, 1), level=0, quads=True)
        mesh.transform(T, inplace=True)

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

    def load_CIF_file_dialog(self):
        """
        Open a file dialog to select a CIF file to load.

        Returns
        -------
        filename : str
            Path to the selected ``.cif`` file, or an empty string if
            the dialog was cancelled.

        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getOpenFileName(
            self,
            "Load CIF file",
            self._get_file_dialog_dir(),
            "CIF files (*.cif)",
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def save_INS_file_dialog(self):
        """
        Open a file dialog to select a destination INS file.

        Returns
        -------
        filename : str
            Path to the selected ``.ins`` file, or an empty string if
            the dialog was cancelled.

        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self,
            "Save INS file",
            self._get_file_dialog_dir(),
            "INS files (*.ins)",
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def get_crystal_system(self):
        """
        Current crystal system selection.

        Returns
        -------
        crystal_system : str
            Selected crystal system name (e.g. "Triclinic", "Cubic").

        """

        return self.crystal_system_combo.currentText()

    def set_crystal_system(self, crystal_system):
        """
        Select a crystal system in the combo box.

        Parameters
        ----------
        crystal_system : str
            Name of the crystal system to select.

        """

        index = self.crystal_system_combo.findText(crystal_system)
        if index >= 0:
            self.crystal_system_combo.setCurrentIndex(index)

    def update_space_groups(self, nos):
        """
        Repopulate the space group combo box.

        Parameters
        ----------
        nos : iterable of str
            Space group identifiers to add to the combo box.

        """

        self.space_group_combo.clear()
        for no in nos:
            self.space_group_combo.addItem(no)

    def get_space_group(self):
        """
        Current space group selection.

        Returns
        -------
        space_group : str
            Selected space group identifier.

        """

        return self.space_group_combo.currentText()

    def set_space_group(self, space_group):
        """
        Select a space group in the combo box.

        Parameters
        ----------
        space_group : str
            Space group identifier to select.

        """

        index = self.space_group_combo.findText(space_group)
        if index >= 0:
            self.space_group_combo.setCurrentIndex(index)

    def update_settings(self, settings):
        """
        Repopulate the space group setting combo box.

        Parameters
        ----------
        settings : iterable of str
            Space group setting identifiers to add to the combo box.

        """

        self.setting_combo.clear()
        for setting in settings:
            self.setting_combo.addItem(setting)

    def get_setting(self):
        """
        Current space group setting selection.

        Returns
        -------
        setting : str
            Selected space group setting identifier.

        """

        return self.setting_combo.currentText()

    def set_setting(self, setting):
        """
        Select a space group setting in the combo box.

        Parameters
        ----------
        setting : str
            Space group setting identifier to select.

        """

        index = self.setting_combo.findText(setting)
        if index >= 0:
            self.setting_combo.setCurrentIndex(index)

    def set_lattice_constants(self, params):
        """
        Populate the lattice parameter fields.

        Parameters
        ----------
        params : list of float
            Lattice constants a, b, c (Å) and angles alpha, beta, gamma
            (degrees), in that order.

        """

        self.a_line.setText("{:.4f}".format(params[0]))
        self.b_line.setText("{:.4f}".format(params[1]))
        self.c_line.setText("{:.4f}".format(params[2]))

        self.alpha_line.setText("{:.4f}".format(params[3]))
        self.beta_line.setText("{:.4f}".format(params[4]))
        self.gamma_line.setText("{:.4f}".format(params[5]))

    def get_lattice_constants(self):
        """
        Lattice parameter values entered by the user.

        Returns
        -------
        params : list of float or None
            Lattice constants a, b, c (Å) and angles alpha, beta, gamma
            (degrees), or None if any field has invalid input.

        """

        params = (
            self.a_line,
            self.b_line,
            self.c_line,
            self.alpha_line,
            self.beta_line,
            self.gamma_line,
        )

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return [float(param.text()) for param in params]

    def set_unit_cell_volume(self, vol):
        """
        Populate the unit cell volume field.

        Parameters
        ----------
        vol : float
            Unit cell volume (Å^3).

        """

        self.V_line.setText("{:.4f}".format(vol))

    def set_scatterers(self, scatterers):
        """
        Repopulate the atom site table.

        Parameters
        ----------
        scatterers : list
            List of atom site entries, each as accepted by
            :meth:`set_scatterer`.

        """

        self.atm_table.clearSelection()
        self.atm_table.setRowCount(0)
        self.atm_table.setRowCount(len(scatterers))

        for row, scatterer in enumerate(scatterers):
            self.set_scatterer(row, scatterer)

    def set_scatterer(self, row, scatterer):
        """
        Populate a row of the atom site table.

        Parameters
        ----------
        row : int
            Row index in the atom site table.
        scatterer : list
            Atom label followed by x, y, z, occupancy, and Uiso values,
            in that order.

        """

        atm, *xyz, occ, Uiso = scatterer
        xyz = ["{:.4f}".format(val) for val in xyz]
        occ = "{:.4f}".format(occ)
        Uiso = "{:.4f}".format(Uiso)
        self.atm_table.setItem(row, 0, QTableWidgetItem(atm))
        self.atm_table.setItem(row, 1, QTableWidgetItem(xyz[0]))
        self.atm_table.setItem(row, 2, QTableWidgetItem(xyz[1]))
        self.atm_table.setItem(row, 3, QTableWidgetItem(xyz[2]))
        self.atm_table.setItem(row, 4, QTableWidgetItem(occ))
        self.atm_table.setItem(row, 5, QTableWidgetItem(Uiso))

    def get_scatterer(self):
        """
        Atom site values for the currently selected table row.

        Returns
        -------
        scatterer : list or None
            Atom label and x, y, z, occupancy, and Uiso values for the
            selected row, or None if no row is selected.

        """

        row = self.atm_table.currentRow()
        if row is not None:
            return self.get_atom_site(row)

    def get_atom_site(self, row):
        """
        Atom site values from a specific table row.

        Parameters
        ----------
        row : int
            Row index in the atom site table.

        Returns
        -------
        scatterer : list
            Atom label (str) followed by x, y, z, occupancy, and Uiso
            values (float) for the given row.

        """

        atm = self.atm_table.item(row, 0).text()
        x = self.atm_table.item(row, 1).text()
        y = self.atm_table.item(row, 2).text()
        z = self.atm_table.item(row, 3).text()
        occ = self.atm_table.item(row, 4).text()
        Uiso = self.atm_table.item(row, 5).text()
        scatterer = [atm, *[float(val) for val in [x, y, z, occ, Uiso]]]

        return scatterer

    def get_scatterers(self):
        """
        Atom site values for all rows of the atom site table.

        Returns
        -------
        scatterers : list of list
            Atom site values (label, x, y, z, occupancy, Uiso) for each
            row of the table.

        """

        n = self.atm_table.rowCount()

        scatterers = []
        for row in range(n):
            scatterer = self.get_atom_site(row)
            scatterers.append(scatterer)

        return scatterers

    def set_isotope(self, isotope):
        """
        Set the isotope/atom button label.

        Parameters
        ----------
        isotope : str
            Isotope or element label to display on the button.

        """

        self.atm_button.setText(isotope)

    def get_isotope(self):
        """
        Current isotope/atom button label.

        Returns
        -------
        isotope : str
            Isotope or element label currently shown on the button.

        """

        return self.atm_button.text()

    def set_atom(self, scatterer):
        """
        Populate the atom entry fields.

        Parameters
        ----------
        scatterer : list
            Atom label followed by x, y, z, occupancy, and Uiso values,
            in that order.

        """

        self.atm_button.setText(scatterer[0])
        self.x_line.setText(str(scatterer[1]))
        self.y_line.setText(str(scatterer[2]))
        self.z_line.setText(str(scatterer[3]))
        self.occ_line.setText(str(scatterer[4]))
        self.Uiso_line.setText(str(scatterer[5]))

    def set_atom_table(self):
        """
        Apply the atom entry fields to the currently selected row.

        Reads the x, y, z, occupancy, and Uiso line edits and, if they
        all contain valid input and a row is selected, writes the
        values, together with the current isotope button label, into
        that row of the atom site table.

        """

        row = self.atm_table.currentRow()

        params = (
            self.x_line,
            self.y_line,
            self.z_line,
            self.occ_line,
            self.Uiso_line,
        )

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params and row is not None:
            scatterer = [
                self.atm_button.text(),
                *[float(param.text()) for param in params],
            ]

            self.set_scatterer(row, scatterer)

    def set_formula_z(self, chemical_formula, z_parameter):
        """
        Populate the chemical formula and Z fields.

        Parameters
        ----------
        chemical_formula : str
            Chemical formula of the unit cell.
        z_parameter : int or float
            Number of formula units per unit cell (Z).

        """

        self.chem_line.setText(chemical_formula)
        self.Z_line.setText(str(z_parameter))

    def get_minimum_d_spacing(self):
        """
        Minimum d-spacing entered by the user.

        Returns
        -------
        d_min : float or None
            Minimum d-spacing (Å) for structure factor calculation, or
            None if the field has invalid input.

        """

        if self.dmin_line.hasAcceptableInput():
            return float(self.dmin_line.text())

    def constrain_parameters(self, const):
        """
        Enable or disable lattice parameter fields based on constraints.

        Parameters
        ----------
        const : list of bool
            Flags indicating, for each of a, b, c, alpha, beta, and
            gamma, whether the corresponding field should be disabled
            (fixed) due to the crystal system's symmetry constraints.

        """

        params = (
            self.a_line,
            self.b_line,
            self.c_line,
            self.alpha_line,
            self.beta_line,
            self.gamma_line,
        )

        for fixed, param in zip(const, params):
            param.setDisabled(fixed)

    def add_atoms(self, atom_dict):
        """
        Draw atom sites as spheres in the 3D view.

        Merges symmetry-equivalent atoms that fall on the same fractional
        coordinate (within rounding) into a single occupancy-weighted
        sphere whose color, radius, and opacity reflect the blended
        contributions, then enables block picking so atoms can be
        selected in the 3D view.

        Parameters
        ----------
        atom_dict : dict
            Mapping of atom/element label to a tuple of (coordinates,
            occupancies, indices), where ``coordinates`` is an iterable
            of fractional (x, y, z) positions, ``occupancies`` is an
            iterable of site occupancy factors, and ``indices`` is an
            iterable of row indices into the atom site table
            corresponding to each coordinate.

        """

        self.plotter.clear_actors()

        T = np.eye(4)
        geoms = []
        self.indexing = {}
        # Track original per-block colors so highlight can toggle
        # between the base color and the highlight color.
        self._block_colors = {}

        sphere = pv.Icosphere(radius=1, nsub=2)

        site_info = {}

        for atom, (coordinates, opacities, indices) in atom_dict.items():
            base_color = colors[atom]
            base_radius = radii[atom][0]
            base_rgb = np.array(matplotlib.colors.to_rgb(base_color))

            for coord, occ, ind in zip(coordinates, opacities, indices):
                occ = float(occ)
                key = tuple(round(c, 4) for c in coord)

                if key not in site_info:
                    site_info[key] = {
                        "coord": np.array(coord, dtype=float),
                        "rgb_sum": occ * base_rgb,
                        "radius_sum": occ * base_radius,
                        "occ_total": occ,
                        "best_index": ind,
                        "best_occ": occ,
                    }
                else:
                    info = site_info[key]
                    info["rgb_sum"] += occ * base_rgb
                    info["radius_sum"] += occ * base_radius
                    info["occ_total"] += occ
                    if occ > info["best_occ"]:
                        info["best_occ"] = occ
                        info["best_index"] = ind

        block_keys = []
        for block_idx, (key, info) in enumerate(site_info.items()):
            occ_total = info["occ_total"]

            if occ_total <= 0.0:
                continue

            coord = info["coord"]
            radius = info["radius_sum"] / occ_total if occ_total > 0 else 0.0

            T[0, 0] = T[1, 1] = T[2, 2] = radius
            T[:3, 3] = coord
            atm = sphere.copy().transform(T, inplace=True)
            geoms.append(atm)

            self.indexing[len(block_keys)] = info["best_index"]
            block_keys.append(key)

        multiblock = pv.MultiBlock(geoms)

        _, mapper = self.plotter.add_composite(
            multiblock,
            smooth_shading=True,
            show_scalar_bar=False,
        )

        self.mapper = mapper

        for i, key in enumerate(block_keys, start=1):
            info = site_info[key]
            occ_total = info["occ_total"]

            if occ_total > 0.0:
                rgb = info["rgb_sum"] / occ_total
            else:
                rgb = info["rgb_sum"]

            alpha = max(0.0, min(1.0, occ_total))

            try:
                self.mapper.block_attr[i].color = tuple(rgb)
                self.mapper.block_attr[i].opacity = float(alpha)
                # Cache the original color for this block index.
                self._block_colors[i] = self.mapper.block_attr[i].color
            except Exception:
                continue

        self.plotter.enable_block_picking(callback=self.highlight, side="left")
        self.plotter.enable_block_picking(
            callback=self.highlight, side="right"
        )

        self.reset_view()

    def highlight(self, index, dataset):
        """Toggle highlight color while preserving original atom color.

        Callback for block picking on the 3D view: recolors the picked
        atom block pink (or restores its original color if already
        highlighted) and synchronizes the atom table selection.

        Parameters
        ----------
        index : int
            Picked block index (1-based) within the composite mesh's
            block attributes.
        dataset : pyvista.DataSet
            Picked dataset block, as provided by the block-picking
            callback (unused).

        """

        current_color = self.mapper.block_attr[index].color
        base_color = self._block_colors.get(index, current_color)

        self.atm_table.clearSelection()

        if current_color == "pink":
            # Turn off highlight: restore the original color.
            self.mapper.block_attr[index].color = base_color
            return

        # Turn on highlight: switch to pink and select the row.
        self.mapper.block_attr[index].color = "pink"

        ind = self.indexing[index - 1]

        selected = self.atm_table.selectedIndexes()
        if selected:
            selected_row = selected[0].row()
            if selected_row == ind:
                return
        self.atm_table.selectRow(ind)

    def set_factors(self, hkls, ds, F2s):
        """
        Repopulate the structure factor table.

        Parameters
        ----------
        hkls : iterable of array-like
            Miller indices (h, k, l) for each reflection.
        ds : iterable of float
            d-spacing (Å) for each reflection.
        F2s : iterable of float
            Squared structure factor for each reflection.

        """

        self.f2_table.setRowCount(0)
        self.f2_table.setRowCount(len(hkls))

        for row, (hkl, d, F2) in enumerate(zip(hkls, ds, F2s)):
            hkl = ["{:.0f}".format(val) for val in hkl]
            d = "{:.4f}".format(d)
            F2 = "{:.2f}".format(F2)
            self.f2_table.setItem(row, 0, QTableWidgetItem(hkl[0]))
            self.f2_table.setItem(row, 1, QTableWidgetItem(hkl[1]))
            self.f2_table.setItem(row, 2, QTableWidgetItem(hkl[2]))
            self.f2_table.setItem(row, 3, QTableWidgetItem(d))
            self.f2_table.setItem(row, 4, QTableWidgetItem(F2))

    def get_hkl(self):
        """
        Individual hkl indices entered by the user.

        Returns
        -------
        hkl : list of float or None
            Miller indices (h, k, l), or None if any field has invalid
            input.

        """

        params = self.h_line, self.k_line, self.l_line

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return [float(param.text()) for param in params]

    def set_equivalents(self, hkls, d, F2):
        """
        Repopulate the structure factor table with symmetry equivalents.

        Unlike :meth:`set_factors`, all rows share the same d-spacing
        and squared structure factor since they are symmetry equivalents
        of a single reflection.

        Parameters
        ----------
        hkls : iterable of array-like
            Miller indices (h, k, l) for each symmetry-equivalent
            reflection.
        d : float
            d-spacing (Å) shared by all the equivalent reflections.
        F2 : float
            Squared structure factor shared by all the equivalent
            reflections.

        """

        self.f2_table.setRowCount(0)
        self.f2_table.setRowCount(len(hkls))

        d = "{:.4f}".format(d)
        F2 = "{:.2f}".format(F2)

        for row, hkl in enumerate(hkls):
            hkl = ["{:.0f}".format(val) for val in hkl]
            self.f2_table.setItem(row, 0, QTableWidgetItem(hkl[0]))
            self.f2_table.setItem(row, 1, QTableWidgetItem(hkl[1]))
            self.f2_table.setItem(row, 2, QTableWidgetItem(hkl[2]))
            self.f2_table.setItem(row, 3, QTableWidgetItem(d))
            self.f2_table.setItem(row, 4, QTableWidgetItem(F2))

    def get_periodic_table(self):
        """
        Create a periodic table dialog view.

        Returns
        -------
        view : NeuXtalViz.views.periodic_table.PeriodicTableView
            New periodic table view instance for isotope selection.

        """

        return PeriodicTableView()

    def set_absorption_shape_constants(self, params):
        """
        Populate the ellipsoid dimension fields.

        Parameters
        ----------
        params : list of float
            Thickness, width, and height values (mm), in that order.
        """

        self.abs_param1_line.setText("{:.2f}".format(params[0]))
        self.abs_param2_line.setText("{:.2f}".format(params[1]))
        self.abs_param3_line.setText("{:.2f}".format(params[2]))

    def get_absorption_shape_constants(self):
        """
        Ellipsoid dimension values entered by the user.

        Returns
        -------
        params : list of float or None
            Thickness, width, and height values (mm), or None if any
            field has invalid input.
        """

        params = (
            self.abs_param1_line,
            self.abs_param2_line,
            self.abs_param3_line,
        )

        if all(param.hasAcceptableInput() for param in params):
            return [float(param.text()) for param in params]

    def get_wavelength(self):
        """
        Incident wavelength entered by the user, if valid.

        Returns
        -------
        wavelength : float or None
            Wavelength (Å), or None if the field does not currently
            contain acceptable input.
        """

        if self.abs_wavelength_line.hasAcceptableInput():
            return float(self.abs_wavelength_line.text())

    def get_absorption_d_min(self):
        """
        Minimum d-spacing entered by the user, if valid.

        Returns
        -------
        d_min : float or None
            Minimum d-spacing (Å), or None if the field does not
            currently contain acceptable input.
        """

        if self.abs_dmin_line.hasAcceptableInput():
            return float(self.abs_dmin_line.text())

    def get_absorption_sample_vectors(self):
        """
        Beam-/in-plane-direction vectors entered by the user.

        These define the crystal's own orientation for the absorption
        prediction, independent of :meth:`get_absorption_shape_vectors`
        (which orients the shape mesh).

        Returns
        -------
        u_vector : list of float
            Reciprocal lattice indices (h, k, l) of the beam direction.
        v_vector : list of float
            Reciprocal lattice indices (h, k, l) of the in-plane
            direction.
        """

        params = (
            self.abs_sample_hu_line,
            self.abs_sample_ku_line,
            self.abs_sample_lu_line,
            self.abs_sample_hv_line,
            self.abs_sample_kv_line,
            self.abs_sample_lv_line,
        )

        if all(param.hasAcceptableInput() for param in params):
            vals = [float(param.text()) for param in params]
            return vals[0:3], vals[3:6]

    def get_absorption_shape_vectors(self):
        """
        Shape-orientation U/V vectors entered by the user.

        Returns
        -------
        u_vector : list of float
            Reciprocal lattice indices (h, k, l) of the direction along
            the sample thickness (the shape's face normal).
        v_vector : list of float
            Reciprocal lattice indices (h, k, l) of an in-plane lateral
            direction.
        """

        params = (
            self.abs_shape_hu_line,
            self.abs_shape_ku_line,
            self.abs_shape_lu_line,
            self.abs_shape_hv_line,
            self.abs_shape_kv_line,
            self.abs_shape_lv_line,
        )

        if all(param.hasAcceptableInput() for param in params):
            vals = [float(param.text()) for param in params]
            return vals[0:3], vals[3:6]

    def set_material_display(self, chemical_formula, z_parameter, volume):
        """
        Populate the (read-only) absorption-tab material fields.

        Parameters
        ----------
        chemical_formula : str
            Chemical formula, as returned by the model's
            ``get_chemical_formula_z_parameter``.
        z_parameter : float
            Number of formula units per unit cell.
        volume : float
            Unit cell volume (Å^3).
        """

        self.abs_chem_line.setText(chemical_formula)
        self.abs_Z_line.setText(str(z_parameter))
        self.abs_V_line.setText("{:.4f}".format(volume))

    def set_absorption_parameters(self, abs_dict):
        """
        Populate the scattering/absorption and material property fields.

        Parameters
        ----------
        abs_dict : dict
            Dictionary with keys "sigma_a", "sigma_s", "mu_a", "mu_s",
            "N", "M", "n", "rho", "V", and "m" giving the absorption and
            scattering cross sections, linear coefficients, number of
            atoms, molar mass, number density, mass density, volume,
            and mass, respectively.
        """

        self.abs_sigma_a_line.setText("{:.4f}".format(abs_dict["sigma_a"]))
        self.abs_sigma_s_line.setText("{:.4f}".format(abs_dict["sigma_s"]))

        self.abs_mu_a_line.setText("{:.4f}".format(abs_dict["mu_a"]))
        self.abs_mu_s_line.setText("{:.4f}".format(abs_dict["mu_s"]))

        self.abs_N_line.setText("{:.4f}".format(abs_dict["N"]))
        self.abs_M_line.setText("{:.4f}".format(abs_dict["M"]))
        self.abs_n_line.setText("{:.4f}".format(abs_dict["n"]))
        self.abs_rho_line.setText("{:.4f}".format(abs_dict["rho"]))
        self.abs_v_line.setText("{:.4f}".format(abs_dict["V"]))
        self.abs_m_line.setText("{:.4f}".format(abs_dict["m"]))

    def set_absorption_results(self, hkls, ds, Ts, Tbars):
        """
        Repopulate the absorption results table.

        Parameters
        ----------
        hkls : iterable of array-like
            Miller indices (h, k, l) for each reflection.
        ds : iterable of float
            d-spacing (Å) for each reflection.
        Ts : iterable of float
            Transmission for each reflection (NaN if unreachable at the
            chosen wavelength).
        Tbars : iterable of float
            Absorption-weighted path length (cm) for each reflection
            (NaN if unreachable at the chosen wavelength).
        """

        self.abs_table.setRowCount(0)
        self.abs_table.setRowCount(len(hkls))

        for row, (hkl, d, T, tbar) in enumerate(zip(hkls, ds, Ts, Tbars)):
            h, k, l = ["{:.0f}".format(v) for v in hkl]
            d_str = "{:.4f}".format(d)
            T_str = "N/A" if np.isnan(T) else "{:.4f}".format(T)
            tbar_str = "N/A" if np.isnan(tbar) else "{:.4f}".format(tbar)
            for col, val in enumerate([h, k, l, d_str, T_str, tbar_str]):
                self.abs_table.setItem(row, col, QTableWidgetItem(val))

    def add_absorption_sample(self, mesh, T, mu, r_incident):
        """
        Draw the absorption sample mesh in the 3D view.

        Clears the current scene, adds the sample as a collection of
        triangles colored by the transmission from the incident beam
        entering the sample out to each surface point
        (``exp(-mu*(r_incident + r))``, with a colorbar) -- the same
        two-leg path used by the model's ``predict_transmission``, an
        easy, physically meaningful way to show which faces of the
        shape are most absorbing without hiding the shape itself
        behind a separate plot. Also adds a*/b*/c* arrows attached to
        the sample (colored red/green/blue, oriented by ``T``, scaled
        to the sample's own size), and resets the view.

        Parameters
        ----------
        mesh : iterable of array-like
            Triangle vertex coordinates describing the sample surface
            mesh, in cm.
        T : (3, 3) array-like
            Orientation matrix (unit a*/b*/c* Cartesian columns) from
            the model's ``get_transform_from_UB``.
        mu : float
            Linear attenuation coefficient (1/cm, ``mu_a + mu_s`` from
            the model's ``get_absorption_dict``) used to color the
            sample surface by local transmission.
        r_incident : float
            Sample radius (cm) along the incident beam direction, from
            the model's ``get_incident_path_length`` -- the constant
            leg of the path added to each surface point's own outgoing
            radius before computing transmission.
        """

        self.plotter.clear_actors()

        triangles = []
        for triangle in mesh:
            tri = pv.Triangle(triangle)
            radius = np.linalg.norm(triangle, axis=1)
            tri.point_data["Transmission"] = np.exp(
                -mu * (r_incident + radius)
            )
            triangles.append(tri)

        multiblock = pv.MultiBlock(triangles)

        self.plotter.add_composite(
            multiblock,
            scalars="Transmission",
            cmap="gist_rainbow",
            clim=[0, 1],
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args={"title": "T"},
        )

        self.plotter.add_legend_scale(
            corner_offset_factor=2,
            bottom_border_offset=50,
            top_border_offset=50,
            left_border_offset=100,
            right_border_offset=100,
            legend_visibility=True,
            xy_label_mode=False,
        )

        T = np.array(T)
        vertices = np.asarray(mesh).reshape(-1, 3)
        extent = np.linalg.norm(vertices, axis=1).max()
        length = 1.5 * extent

        labels = ["a*", "b*", "c*"]
        colors = ["r", "g", "b"]
        points = []
        for i in range(3):
            direction = T[:, i]
            arrow = pv.Arrow(
                start=[0.0, 0.0, 0.0], direction=direction, scale=length
            )
            self.plotter.add_mesh(arrow, color=colors[i], smooth_shading=True)
            points.append(direction / np.linalg.norm(direction) * length)

        self.plotter.add_point_labels(
            points,
            labels,
            text_color="k",
            font_size=20,
            shape=None,
            always_visible=True,
            show_points=False,
        )

        self.reset_view()

    def connect_instrument_selector(self, select_instrument):
        """
        Connect the Simulator instrument combo box to a handler.

        Parameters
        ----------
        select_instrument : callable
            Slot invoked when the instrument combo box is activated.
        """

        self.sim_instrument_combo.activated.connect(select_instrument)

    def connect_load_UB(self, load_UB):
        """
        Connect the Simulator "Load UB" button to a handler.

        Parameters
        ----------
        load_UB : callable
            Slot invoked when the load UB button is clicked.
        """

        self.sim_load_UB_button.clicked.connect(load_UB)

    def connect_clear_UB(self, clear_UB):
        """
        Connect the Simulator "Clear UB" button to a handler.

        Parameters
        ----------
        clear_UB : callable
            Slot invoked when the clear UB button is clicked.
        """

        self.sim_clear_UB_button.clicked.connect(clear_UB)

    def connect_calculate_simulator(self, calculate_simulator):
        """
        Connect the Simulator calculate button to a handler.

        Parameters
        ----------
        calculate_simulator : callable
            Slot invoked when the Simulator calculate button is
            clicked.
        """

        self.sim_calculate_button.clicked.connect(calculate_simulator)

    def load_UB_file_dialog(self):
        """
        Open a file dialog to select an ISAW UB matrix file to load.

        Returns
        -------
        filename : str
            Path to the selected UB file, or an empty string if the
            dialog was cancelled.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getOpenFileName(
            self,
            "Load UB file",
            self._get_file_dialog_dir(),
            "UB files (*.mat *.ub)",
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def get_simulator_instrument(self):
        """
        Currently selected Simulator instrument.

        Returns
        -------
        instrument : str
            Selected instrument name ("TOPAZ", "MANDI", or "CORELLI").
        """

        return self.sim_instrument_combo.currentText()

    def set_simulator_d_min(self, d_min):
        """
        Populate the Simulator d-min field.

        Parameters
        ----------
        d_min : float
            Minimum d-spacing (Å).
        """

        self.sim_dmin_line.setText("{:.4f}".format(d_min))

    def get_simulator_d_min(self):
        """
        Minimum d-spacing entered by the user, if valid.

        Returns
        -------
        d_min : float or None
            Minimum d-spacing (Å), or None if the field does not
            currently contain acceptable input.
        """

        if self.sim_dmin_line.hasAcceptableInput():
            return float(self.sim_dmin_line.text())

    def set_UB_status(self, filename):
        """
        Show or clear the loaded UB filename, enabling/disabling the
        u/v vector fields accordingly.

        Parameters
        ----------
        filename : str or None
            Path of the loaded UB file, or None if no UB is loaded
            (the u/v vector fields are then re-enabled).
        """

        self.sim_UB_line.setText(filename if filename else "")

        enabled = not filename

        for line in (
            self.sim_hu_line,
            self.sim_ku_line,
            self.sim_lu_line,
            self.sim_hv_line,
            self.sim_kv_line,
            self.sim_lv_line,
        ):
            line.setEnabled(enabled)

    def get_simulator_UB_filename(self):
        """
        Path of the currently loaded UB file.

        Returns
        -------
        filename : str
            Path of the loaded UB file, or an empty string if none is
            loaded.
        """

        return self.sim_UB_line.text()

    def get_simulator_vectors(self):
        """
        Beam-/in-plane-direction vectors entered by the user.

        Returns
        -------
        u_vector : list of float
            Reciprocal lattice indices (h, k, l) of the beam
            direction.
        v_vector : list of float
            Reciprocal lattice indices (h, k, l) of the in-plane
            direction.
        """

        params = (
            self.sim_hu_line,
            self.sim_ku_line,
            self.sim_lu_line,
            self.sim_hv_line,
            self.sim_kv_line,
            self.sim_lv_line,
        )

        if all(param.hasAcceptableInput() for param in params):
            vals = [float(param.text()) for param in params]
            return vals[0:3], vals[3:6]

    def get_simulator_goniometer(self):
        """
        Goniometer angles entered by the user, if valid.

        Returns
        -------
        angles : list of float or None
            Omega, chi, and phi angles (degrees), or None if any field
            has invalid input.
        """

        params = self.sim_omega_line, self.sim_chi_line, self.sim_phi_line

        if all(param.hasAcceptableInput() for param in params):
            return [float(param.text()) for param in params]

    def get_simulator_counting_time(self):
        """
        Counting time entered by the user, if valid.

        Returns
        -------
        counting_time : float or None
            Counting time (minutes), or None if the field does not
            currently contain acceptable input.
        """

        if self.sim_time_line.hasAcceptableInput():
            return float(self.sim_time_line.text())

    def set_simulator_results(self, hkls, ds, lambdas, F2s, Is, IsigmaIs):
        """
        Repopulate the Simulator results table.

        Parameters
        ----------
        hkls : iterable of array-like
            Miller indices (h, k, l) for each reflection.
        ds : iterable of float
            d-spacing (Å) for each reflection.
        lambdas : iterable of float
            Wavelength (Å) for each reflection.
        F2s : iterable of float
            Squared structure factor for each reflection.
        Is : iterable of float
            Predicted integrated counts for each reflection.
        IsigmaIs : iterable of float
            Predicted I/σ for each reflection.
        """

        self.sim_table.setRowCount(0)
        self.sim_table.setRowCount(len(hkls))

        rows = zip(hkls, ds, lambdas, F2s, Is, IsigmaIs)
        for row, (hkl, d, lam, F2, I, IsigmaI) in enumerate(rows):
            values = [
                "{:.0f}".format(hkl[0]),
                "{:.0f}".format(hkl[1]),
                "{:.0f}".format(hkl[2]),
                "{:.4f}".format(d),
                "{:.4f}".format(lam),
                "{:.2f}".format(F2),
                "{:.2f}".format(I),
                "{:.2f}".format(IsigmaI),
            ]
            for col, val in enumerate(values):
                self.sim_table.setItem(row, col, QTableWidgetItem(val))
