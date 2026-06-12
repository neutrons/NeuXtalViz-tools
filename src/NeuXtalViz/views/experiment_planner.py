import os

from qtpy.QtWidgets import (
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QTabWidget,
    QFileDialog,
)

from qtpy.QtGui import QDoubleValidator, QIntValidator
from qtpy.QtCore import Qt, Signal

import numpy as np

import matplotlib.pyplot as plt
import pyvista as pv

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (
    FormatStrFormatter,
    PercentFormatter,
)
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import Axes, GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import (
    ExtremeFinderSimple,
    MaxNLocator,
)
from NeuXtalViz.views.base_view import NeuXtalVizWidget

import qtawesome as qta


class ExperimentView(NeuXtalVizWidget):
    """
    View for experiment planning and peak visualization in NeuXtalViz.

    Provides user interface elements for experiment setup, coverage
    analysis, peak calculation, and plan management.
    """

    roi_ready = Signal(float, float)
    sel_ready = Signal(float, float)
    viz_ready = Signal()
    harm_ready = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.tab_widget = QTabWidget(self)

        self.coverage_tab()
        self.peak_tab()
        self.mesh_tab()

        self.layout().addWidget(self.tab_widget, stretch=1)
        self.plan_table.itemChanged.connect(self.handle_item_changed)
        self.goniometer_table.itemChanged.connect(self.update_limits)
        self.mesh_table.itemChanged.connect(self.calculate_mesh_step)

        self.hkl = None
        self.scale = None

        self.hkl_alt = None
        self.scale_alt = None

        self.hkl_laue = None
        self.scale_laue = None

    def coverage_tab(self):
        cov_tab = QWidget()
        self.tab_widget.addTab(cov_tab, "Coverage")

        planner_layout = QVBoxLayout()

        self.instrument_combo = QComboBox(self)
        self.instrument_combo.addItem("TOPAZ")
        self.instrument_combo.addItem("MANDI")
        self.instrument_combo.addItem("CORELLI")
        self.instrument_combo.addItem("SNAP")
        self.instrument_combo.addItem("IMAGINE")
        self.instrument_combo.addItem("WAND²")
        self.instrument_combo.addItem("DEMAND")
        self.instrument_combo.setToolTip(
            "Select the instrument for the experiment."
        )

        self.auto_scale_dropdown(self.instrument_combo)

        self.mode_combo = QComboBox(self)
        self.mode_combo.setToolTip(
            "Select the goniometer mode for the experiment."
        )

        self.crystal_combo = QComboBox(self)
        self.crystal_combo.setToolTip(
            "Select the crystal system for the sample."
        )
        self.point_group_combo = QComboBox(self)
        self.point_group_combo.setToolTip(
            "Select the point group for the sample."
        )
        self.lattice_centering_combo = QComboBox(self)
        self.lattice_centering_combo.setToolTip(
            "Select the lattice centering for the sample."
        )

        self.auto_scale_dropdown(self.mode_combo)
        self.auto_scale_dropdown(self.crystal_combo)
        self.auto_scale_dropdown(self.point_group_combo)
        self.auto_scale_dropdown(self.lattice_centering_combo)

        self.crystal_combo.addItem("Triclinic")
        self.crystal_combo.addItem("Monoclinic")
        self.crystal_combo.addItem("Orthorhombic")
        self.crystal_combo.addItem("Tetragonal")
        self.crystal_combo.addItem("Trigonal/Rhombohedral")
        self.crystal_combo.addItem("Trigonal/Hexagonal")
        self.crystal_combo.addItem("Hexagonal")
        self.crystal_combo.addItem("Cubic")
        self.auto_scale_dropdown(self.crystal_combo)

        self.load_UB_button = QPushButton("Load UB", self)
        self.load_UB_button.setToolTip(
            "Load a UB matrix from a file for orientation."
        )
        self.load_UB_button.setIcon(qta.icon("fa6s.folder-open"))
        self.reset_button = QPushButton("Recalculate Coverage", self)
        self.reset_button.setToolTip("Recalculate the coverage.")
        self.reset_button.setIcon(qta.icon("fa6s.calculator"))
        self.instrument_button = QPushButton("Show Instrument", self)
        self.instrument_button.setToolTip("Show instrument.")
        self.instrument_button.setIcon(qta.icon("fa6s.eye"))
        self.optimize_button = QPushButton("Optimize Coverage", self)
        self.optimize_button.setToolTip(
            "Optimize the coverage of the experiment plan."
        )
        self.optimize_button.setIcon(qta.icon("fa6s.wand-magic-sparkles"))
        self.delete_button = QPushButton("Delete Highlighted", self)
        self.delete_button.setToolTip(
            "Delete the selected orientations from the plan."
        )
        self.delete_button.setIcon(qta.icon("fa6s.trash"))
        self.highlight_button = QPushButton("Highlight All", self)
        self.highlight_button.setToolTip(
            "Highlight all orientations in the plan table."
        )
        self.highlight_button.setIcon(qta.icon("fa6s.highlighter"))
        self.update_button = QPushButton("Update Highlighted", self)
        self.update_button.setToolTip(
            "Update the selected orientations with the current settings."
        )
        self.update_button.setIcon(qta.icon("fa6s.file-pen"))

        self.count_combo = QComboBox(self)
        self.count_combo.setToolTip(
            "Select the counting method for the experiment."
        )
        self.auto_scale_dropdown(self.count_combo)
        self.count_line = QLineEdit("1.0")
        self.count_line.setToolTip(
            "Set the counting value for the experiment (e.g., time or monitor)."
        )
        self.title_line = QLineEdit("Scan Title")
        self.title_line.setToolTip("Set the title for the scan or experiment.")

        self.move_up_button = QPushButton("Move Up", self)
        self.move_up_button.setToolTip(
            "Move selected orientation up in the plan."
        )
        self.move_up_button.setIcon(qta.icon("fa6s.square-caret-up"))
        self.move_down_button = QPushButton("Move Down", self)
        self.move_down_button.setToolTip(
            "Move selected orientation down in the plan."
        )
        self.move_down_button.setIcon(qta.icon("fa6s.square-caret-down"))

        notation = QDoubleValidator.StandardNotation
        validator = QDoubleValidator(0.001, 10000, 5, notation=notation)

        self.count_line.setValidator(validator)

        self.save_plan_button = QPushButton("Save CSV", self)
        self.save_plan_button.setToolTip(
            "Save the current experiment plan as a CSV file."
        )
        self.save_plan_button.setIcon(qta.icon("fa6s.file-csv"))

        self.wl_min_line = QLineEdit("0.4")
        self.wl_min_line.setToolTip(
            "Set the minimum wavelength (λ) in Ångströms."
        )
        self.wl_max_line = QLineEdit("3.5")
        self.wl_max_line.setToolTip(
            "Set the maximum wavelength (λ) in Ångströms."
        )

        self.d_min_line = QLineEdit("0.7")
        self.d_min_line.setToolTip("Set the minimum d-spacing in Ångströms.")

        wl_label = QLabel("λ:")
        d_min_label = QLabel("d(min):")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.2, 10, 5, notation=notation)

        self.wl_min_line.setValidator(validator)
        self.wl_max_line.setValidator(validator)

        validator = QDoubleValidator(0.4, 10, 5, notation=notation)

        self.d_min_line.setValidator(validator)

        angstrom_label = QLabel("Å")

        settings_label = QLabel("Settings")
        self.settings_line = QLineEdit("10")
        self.settings_line.setToolTip(
            "Set the number of settings for the experiment plan."
        )

        validator = QIntValidator(1, 1000)

        self.settings_line.setValidator(validator)

        resize = QHeaderView.Stretch

        self.goniometer_table = QTableWidget()
        self.goniometer_table.setToolTip(
            "Table of goniometer angles and their limits."
        )

        self.goniometer_table.setRowCount(0)
        self.goniometer_table.setColumnCount(3)

        labels = ["Motor", "Min", "Max"]

        self.goniometer_table.horizontalHeader().setStretchLastSection(True)
        self.goniometer_table.horizontalHeader().setSectionResizeMode(resize)
        self.goniometer_table.setHorizontalHeaderLabels(labels)

        self.motor_table = QTableWidget()
        self.motor_table.setToolTip("Table of calibration and motor values.")

        self.motor_table.setRowCount(0)
        self.motor_table.setColumnCount(2)

        self.plan_table = QTableWidget()
        self.plan_table.setToolTip(
            "Table of planned orientations and settings."
        )
        self.plan_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.plan_table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        labels = ["Motor", "Value"]

        self.motor_table.horizontalHeader().setStretchLastSection(True)
        self.motor_table.horizontalHeader().setSectionResizeMode(resize)
        self.motor_table.setHorizontalHeaderLabels(labels)

        self.save_experiment_button = QPushButton("Save Experiment", self)
        self.save_experiment_button.setToolTip(
            "Save the current experiment as a NeXus file."
        )
        self.save_experiment_button.setIcon(qta.icon("fa6s.floppy-disk"))
        self.load_experiment_button = QPushButton("Load Experiment", self)
        self.load_experiment_button.setToolTip(
            "Load an experiment from a NeXus file."
        )
        self.load_experiment_button.setIcon(qta.icon("fa6s.folder-open"))

        settings_layout = QHBoxLayout()

        settings_layout.addWidget(self.load_UB_button)
        settings_layout.addWidget(self.crystal_combo)
        settings_layout.addWidget(self.point_group_combo)
        settings_layout.addWidget(self.lattice_centering_combo)

        params_layout = QHBoxLayout()

        params_layout.addWidget(self.instrument_combo)
        params_layout.addWidget(wl_label)
        params_layout.addWidget(self.wl_min_line)
        params_layout.addWidget(self.wl_max_line)
        params_layout.addWidget(d_min_label)
        params_layout.addWidget(self.d_min_line)
        params_layout.addWidget(angstrom_label)

        details_layout = QGridLayout()

        details_layout.addLayout(settings_layout, 0, 0)
        details_layout.addLayout(params_layout, 1, 0)
        details_layout.addWidget(self.reset_button, 0, 1)
        details_layout.addWidget(self.instrument_button, 1, 1)
        details_layout.addWidget(self.load_experiment_button, 0, 2)
        details_layout.addWidget(self.save_experiment_button, 1, 2)

        result_layout = QVBoxLayout()

        self.values_tab = QTabWidget()

        goniometer_tab = QWidget()
        motor_tab = QWidget()
        plan_tab = QWidget()

        goniometer_layout = QVBoxLayout()
        motor_layout = QVBoxLayout()
        plan_layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch(1)

        planning_layout = QHBoxLayout()
        planning_layout.addWidget(self.title_line)
        planning_layout.addWidget(self.count_combo)
        planning_layout.addWidget(self.count_line)
        planning_layout.addStretch(1)
        planning_layout.addWidget(settings_label)
        planning_layout.addWidget(self.settings_line)
        planning_layout.addWidget(self.optimize_button)

        save_layout = QHBoxLayout()
        save_layout.addWidget(self.delete_button)
        save_layout.addWidget(self.highlight_button)
        save_layout.addWidget(self.update_button)
        save_layout.addWidget(self.move_up_button)
        save_layout.addWidget(self.move_down_button)
        save_layout.addStretch(1)
        save_layout.addWidget(self.save_plan_button)

        goniometer_layout.addLayout(mode_layout)
        goniometer_layout.addWidget(self.goniometer_table)

        cal_layout = QGridLayout()

        self.cal_line = QLineEdit("")
        self.cal_line.setToolTip("Path to detector calibration file.")

        self.gon_line = QLineEdit("")
        self.gon_line.setToolTip("Path to goniometer calibration file.")

        self.mask_line = QLineEdit("")
        self.mask_line.setToolTip("Path to detector mask file.")

        self.cal_browse_button = QPushButton("Detector", self)
        self.cal_browse_button.setToolTip(
            "Browse for detector calibration file."
        )
        self.gon_browse_button = QPushButton("Goniometer", self)
        self.gon_browse_button.setToolTip(
            "Browse for goniometer calibration file."
        )
        self.mask_browse_button = QPushButton("Mask", self)
        self.mask_browse_button.setToolTip("Browse for detector mask file.")

        browse_icon = qta.icon("fa6s.folder-open")
        self.cal_browse_button.setIcon(browse_icon)
        self.gon_browse_button.setIcon(browse_icon)
        self.mask_browse_button.setIcon(browse_icon)

        cal_layout.addWidget(self.cal_line, 0, 0)
        cal_layout.addWidget(self.cal_browse_button, 0, 1)

        cal_layout.addWidget(self.gon_line, 1, 0)
        cal_layout.addWidget(self.gon_browse_button, 1, 1)

        cal_layout.addWidget(self.mask_line, 2, 0)
        cal_layout.addWidget(self.mask_browse_button, 2, 1)

        motor_layout.addLayout(cal_layout)
        motor_layout.addWidget(self.motor_table)

        plan_layout.addLayout(planning_layout)
        plan_layout.addWidget(self.plan_table)
        plan_layout.addLayout(save_layout)
        plan_layout.setStretch(1, 2)
        plan_layout.setStretch(2, 1)

        goniometer_tab.setLayout(goniometer_layout)
        motor_tab.setLayout(motor_layout)
        plan_tab.setLayout(plan_layout)

        self.values_tab.addTab(goniometer_tab, "Goniometers")
        self.values_tab.addTab(motor_tab, "Calibration/Motors")
        self.values_tab.addTab(plan_tab, "Plan")

        result_layout.addWidget(self.values_tab)

        self.color_combo = QComboBox(self)
        self.color_combo.addItem("Sphere")
        self.color_combo.addItem("Redundancy")
        self.color_combo.setToolTip(
            "Select the color scheme for coverage visualization."
        )

        self.auto_scale_dropdown(self.color_combo)

        self.h_max_line = QLineEdit("")
        self.k_max_line = QLineEdit("")
        self.l_max_line = QLineEdit("")

        self.h_max_line.setToolTip("Maximum h index")
        self.k_max_line.setToolTip("Maximum k index")
        self.l_max_line.setToolTip("Maximum l index")

        self.h_max_line.setReadOnly(True)
        self.k_max_line.setReadOnly(True)
        self.l_max_line.setReadOnly(True)

        data_layout = QHBoxLayout()

        h_max_label = QLabel("h(max):")
        k_max_label = QLabel("k(max):")
        l_max_label = QLabel("l(max):")

        data_layout.addWidget(self.color_combo)
        data_layout.addWidget(h_max_label)
        data_layout.addWidget(self.h_max_line)
        data_layout.addWidget(k_max_label)
        data_layout.addWidget(self.k_max_line)
        data_layout.addWidget(l_max_label)
        data_layout.addWidget(self.l_max_line)

        result_layout.addLayout(data_layout)

        self.canvas_cov = FigureCanvas(
            Figure(constrained_layout=True, figsize=(6.4, 4.8))
        )

        self.canvas_cum = FigureCanvas(
            Figure(constrained_layout=True, figsize=(6.4, 4.8))
        )

        results_tab = QTabWidget()
        coverage_tab = QWidget()
        cumulative_tab = QWidget()

        coverage_layout = QVBoxLayout()
        cumulative_layout = QVBoxLayout()

        coverage_layout.addWidget(NavigationToolbar2QT(self.canvas_cov, self))
        coverage_layout.addWidget(self.canvas_cov)

        cumulative_layout.addWidget(
            NavigationToolbar2QT(self.canvas_cum, self)
        )
        cumulative_layout.addWidget(self.canvas_cum)

        fig = self.canvas_cov.figure

        self.ax_cov = fig.subplots(3, 1, sharex=True)
        self.ax_cov[2].set_xlabel("Resolution Shell [Å]")
        self.ax_cov[0].set_ylabel("Completeness")
        self.ax_cov[1].set_ylabel("Redundancy")
        self.ax_cov[2].set_ylabel("Unique")

        fig = self.canvas_cum.figure

        self.ax_cum = fig.subplots(3, 1, sharex=True)
        self.ax_cum[2].set_xlabel("Orientation Number")
        self.ax_cum[0].set_ylabel("Completeness")
        self.ax_cum[1].set_ylabel("Redundancy")
        self.ax_cum[2].set_ylabel("Unique")

        coverage_tab.setLayout(coverage_layout)
        cumulative_tab.setLayout(cumulative_layout)

        results_tab.addTab(coverage_tab, "Resolution")
        results_tab.addTab(cumulative_tab, "Cumulative")

        planner_layout.addLayout(details_layout)
        planner_layout.addLayout(result_layout)
        planner_layout.addWidget(results_tab)

        cov_tab.setLayout(planner_layout)

    def mesh_tab(self):
        inst_tab = QWidget()
        self.tab_widget.addTab(inst_tab, "Mesh")

        # ── shared slice controls ──────────────────────────────────────────

        self.mesh_symmetry_box = QCheckBox("Use Symmetry", self)
        self.mesh_symmetry_box.setToolTip("Apply symmetry to mesh.")
        self.mesh_symmetry_box.setChecked(False)

        self.coverage_mesh_button = QPushButton("Calculate Mesh", self)
        self.coverage_mesh_button.setToolTip(
            "Calculate slice from the mesh scan."
        )
        self.coverage_mesh_button.setIcon(qta.icon("fa6s.calculator"))

        self.coverage_plan_button = QPushButton("Calculate Plan", self)
        self.coverage_plan_button.setToolTip("Calculate slice from the plan.")
        self.coverage_plan_button.setIcon(qta.icon("fa6s.calculator"))

        self.slice_combo = QComboBox(self)
        self.slice_combo.addItem("Axis 1/2")
        self.slice_combo.addItem("Axis 1/3")
        self.slice_combo.addItem("Axis 2/3")
        self.slice_combo.setCurrentIndex(0)
        self.slice_combo.setToolTip("Select the axes for the slice view.")
        self.auto_scale_dropdown(self.slice_combo)

        slice_label = QLabel("Slice:", self)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-10, 10, 5, notation=notation)

        self.slice_line = QLineEdit("0.0")
        self.slice_line.setValidator(validator)
        self.slice_line.setToolTip("Enter the slice position value.")

        validator = QDoubleValidator(0.0001, 100, 5, notation=notation)

        slice_thickness_label = QLabel("Thickness:", self)

        self.slice_thickness_line = QLineEdit("0.1")
        self.slice_thickness_line.setValidator(validator)
        self.slice_thickness_line.setToolTip(
            "Enter the slice thickness value."
        )

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.coverage_plan_button)
        control_layout.addWidget(self.slice_combo)
        control_layout.addWidget(slice_label)
        control_layout.addWidget(self.slice_line)
        control_layout.addWidget(slice_thickness_label)
        control_layout.addWidget(self.slice_thickness_line)
        control_layout.addWidget(self.mesh_symmetry_box)

        self.mesh_definition_tabs = QTabWidget(self)

        angles_tab = QWidget()

        self.mesh_table = QTableWidget()
        self.mesh_table.setToolTip("Table for mesh scan angles and limits.")

        self.mesh_table.setRowCount(0)
        self.mesh_table.setColumnCount(5)
        self.mesh_table.blockSignals(True)

        labels = ["Motor", "Min", "Max", "Angles", "Step"]

        resize = QHeaderView.Stretch

        self.mesh_table.horizontalHeader().setStretchLastSection(True)
        self.mesh_table.horizontalHeader().setSectionResizeMode(resize)
        self.mesh_table.setHorizontalHeaderLabels(labels)

        self.mesh_button = QPushButton("Add Mesh", self)
        self.mesh_button.setToolTip("Add a mesh scan to the experiment plan.")
        self.mesh_button.setIcon(qta.icon("fa6s.square-plus"))

        angles_btn_layout = QHBoxLayout()
        angles_btn_layout.addStretch()
        angles_btn_layout.addWidget(self.coverage_mesh_button)
        angles_btn_layout.addWidget(self.mesh_button)

        angles_layout = QVBoxLayout()
        angles_layout.addWidget(self.mesh_table)
        angles_layout.addLayout(angles_btn_layout)
        angles_tab.setLayout(angles_layout)

        plane_tab = QWidget()

        notation = QDoubleValidator.StandardNotation
        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        u_label = QLabel("u:", self)
        u_label.setToolTip("First in-plane hkl vector.")
        v_label = QLabel("v:", self)
        v_label.setToolTip("Second in-plane hkl vector.")
        h_pl_label = QLabel("h", self)
        k_pl_label = QLabel("k", self)
        l_pl_label = QLabel("l", self)

        self.plane_u1_line = QLineEdit("1")
        self.plane_u2_line = QLineEdit("0")
        self.plane_u3_line = QLineEdit("0")
        self.plane_v1_line = QLineEdit("0")
        self.plane_v2_line = QLineEdit("1")
        self.plane_v3_line = QLineEdit("0")

        for w in [
            self.plane_u1_line,
            self.plane_u2_line,
            self.plane_u3_line,
            self.plane_v1_line,
            self.plane_v2_line,
            self.plane_v3_line,
        ]:
            w.setValidator(validator)

        self.plane_u1_line.setToolTip("h component of the u vector.")
        self.plane_u2_line.setToolTip("k component of the u vector.")
        self.plane_u3_line.setToolTip("l component of the u vector.")
        self.plane_v1_line.setToolTip("h component of the v vector.")
        self.plane_v2_line.setToolTip("k component of the v vector.")
        self.plane_v3_line.setToolTip("l component of the v vector.")

        plane_grid = QGridLayout()
        plane_grid.addWidget(h_pl_label, 0, 1, Qt.AlignCenter)
        plane_grid.addWidget(k_pl_label, 0, 2, Qt.AlignCenter)
        plane_grid.addWidget(l_pl_label, 0, 3, Qt.AlignCenter)
        plane_grid.addWidget(u_label, 1, 0, Qt.AlignCenter)
        plane_grid.addWidget(v_label, 2, 0, Qt.AlignCenter)
        plane_grid.addWidget(self.plane_u1_line, 1, 1)
        plane_grid.addWidget(self.plane_u2_line, 1, 2)
        plane_grid.addWidget(self.plane_u3_line, 1, 3)
        plane_grid.addWidget(self.plane_v1_line, 2, 1)
        plane_grid.addWidget(self.plane_v2_line, 2, 2)
        plane_grid.addWidget(self.plane_v3_line, 2, 3)

        notation = QDoubleValidator.StandardNotation
        pos_validator = QDoubleValidator(1.0, 360.0, 2, notation=notation)
        int_validator = QIntValidator(2, 3600)

        max_angle_label = QLabel("Coverage [°]:", self)
        max_angle_label.setToolTip(
            "Total angular range to sweep about the plane normal."
        )
        self.plane_max_angle_line = QLineEdit("360")
        self.plane_max_angle_line.setValidator(pos_validator)
        self.plane_max_angle_line.setToolTip(
            "Angular coverage in degrees (1–360)."
        )

        n_steps_label = QLabel("# Angles:", self)
        n_steps_label.setToolTip(
            "Number of orientations to sample across the coverage range."
        )
        self.plane_n_steps_line = QLineEdit("60")
        self.plane_n_steps_line.setValidator(int_validator)
        self.plane_n_steps_line.setToolTip("Number of orientations to sample.")

        coverage_layout = QHBoxLayout()
        coverage_layout.addWidget(max_angle_label)
        coverage_layout.addWidget(self.plane_max_angle_line)
        coverage_layout.addWidget(n_steps_label)
        coverage_layout.addWidget(self.plane_n_steps_line)

        self.coverage_plane_button = QPushButton("Calculate Plane", self)
        self.coverage_plane_button.setToolTip(
            "Calculate coverage slice for this scattering plane."
        )
        self.coverage_plane_button.setIcon(qta.icon("fa6s.calculator"))

        self.plane_button = QPushButton("Add Plane", self)
        self.plane_button.setToolTip(
            "Add scattering-plane orientations to the experiment plan."
        )
        self.plane_button.setIcon(qta.icon("fa6s.square-plus"))

        plane_btn_layout = QHBoxLayout()
        plane_btn_layout.addStretch()
        plane_btn_layout.addWidget(self.coverage_plane_button)
        plane_btn_layout.addWidget(self.plane_button)

        plane_layout = QVBoxLayout()
        plane_layout.addLayout(plane_grid)
        plane_layout.addLayout(coverage_layout)
        plane_layout.addLayout(plane_btn_layout)
        plane_layout.addStretch()
        plane_tab.setLayout(plane_layout)

        self.mesh_definition_tabs.addTab(angles_tab, "Define Angles")
        self.mesh_definition_tabs.addTab(plane_tab, "Define Plane")

        # ── canvas & projection matrix (shared) ───────────────────────────

        self.canvas_slice = FigureCanvas(Figure(figsize=[12.8, 12.8]))
        self.cb_slice = None

        self.fig_slice = self.canvas_slice.figure
        self.ax_slice = self.fig_slice.subplots(1, 1)

        convert_to_hkl_params_layout = QGridLayout()

        notation = QDoubleValidator.StandardNotation
        validator = QDoubleValidator(-10, 10, 5, notation=notation)

        self.U1_line = QLineEdit("1")
        self.U2_line = QLineEdit("0")
        self.U3_line = QLineEdit("0")

        self.V1_line = QLineEdit("0")
        self.V2_line = QLineEdit("1")
        self.V3_line = QLineEdit("0")

        self.W1_line = QLineEdit("0")
        self.W2_line = QLineEdit("0")
        self.W3_line = QLineEdit("1")

        self.U1_line.setValidator(validator)
        self.U2_line.setValidator(validator)
        self.U3_line.setValidator(validator)

        self.V1_line.setValidator(validator)
        self.V2_line.setValidator(validator)
        self.V3_line.setValidator(validator)

        self.W1_line.setValidator(validator)
        self.W2_line.setValidator(validator)
        self.W3_line.setValidator(validator)

        ax1_label = QLabel("1:")
        ax2_label = QLabel("2:")
        ax3_label = QLabel("3:")

        h_label = QLabel("h")
        k_label = QLabel("k")
        l_label = QLabel("l")

        convert_to_hkl_params_layout.addWidget(h_label, 0, 1, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(k_label, 0, 2, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(l_label, 0, 3, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(ax1_label, 1, 0, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(ax2_label, 2, 0, Qt.AlignCenter)
        convert_to_hkl_params_layout.addWidget(ax3_label, 3, 0, Qt.AlignCenter)

        convert_to_hkl_params_layout.addWidget(self.U1_line, 1, 1)
        convert_to_hkl_params_layout.addWidget(self.V1_line, 2, 1)
        convert_to_hkl_params_layout.addWidget(self.W1_line, 3, 1)

        convert_to_hkl_params_layout.addWidget(self.U2_line, 1, 2)
        convert_to_hkl_params_layout.addWidget(self.V2_line, 2, 2)
        convert_to_hkl_params_layout.addWidget(self.W2_line, 3, 2)

        convert_to_hkl_params_layout.addWidget(self.U3_line, 1, 3)
        convert_to_hkl_params_layout.addWidget(self.V3_line, 2, 3)
        convert_to_hkl_params_layout.addWidget(self.W3_line, 3, 3)

        # ── assemble outer layout ──────────────────────────────────────────

        mesh_layout = QVBoxLayout()
        mesh_layout.addWidget(self.mesh_definition_tabs)
        mesh_layout.addLayout(control_layout)
        mesh_layout.addWidget(NavigationToolbar2QT(self.canvas_slice, self))
        mesh_layout.addWidget(self.canvas_slice)
        mesh_layout.addLayout(convert_to_hkl_params_layout)

        inst_tab.setLayout(mesh_layout)

    def peak_tab(self):
        inst_tab = QWidget()
        self.tab_widget.addTab(inst_tab, "Peaks")

        peak_layout = QVBoxLayout()

        calculator_layout = QGridLayout()

        h_label = QLabel("h", self)
        h_label.setToolTip("Miller index h for the peak.")
        k_label = QLabel("k", self)
        k_label.setToolTip("Miller index k for the peak.")
        l_label = QLabel("l", self)
        l_label.setToolTip("Miller index l for the peak.")

        peak_1_label = QLabel("1:", self)
        peak_1_label.setToolTip("First peak indices.")
        peak_2_label = QLabel("2:", self)
        peak_2_label.setToolTip("Second peak indices.")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.h1_line = QLineEdit()
        self.h1_line.setToolTip("Enter the h index for the first peak.")
        self.k1_line = QLineEdit()
        self.k1_line.setToolTip("Enter the k index for the first peak.")
        self.l1_line = QLineEdit()
        self.l1_line.setToolTip("Enter the l index for the first peak.")

        self.h2_line = QLineEdit()
        self.h2_line.setToolTip("Enter the h index for the second peak.")
        self.k2_line = QLineEdit()
        self.k2_line.setToolTip("Enter the k index for the second peak.")
        self.l2_line = QLineEdit()
        self.l2_line.setToolTip("Enter the l index for the second peak.")

        self.h1_line.setValidator(validator)
        self.k1_line.setValidator(validator)
        self.l1_line.setValidator(validator)

        self.h2_line.setValidator(validator)
        self.k2_line.setValidator(validator)
        self.l2_line.setValidator(validator)

        gamma_label = QLabel("γ [°]", self)
        gamma_label.setToolTip("Gamma angle (horizontal) in degrees.")
        nu_label = QLabel("ν [°]", self)
        nu_label.setToolTip("Nu angle (vertical) in degrees.")
        intersect_label = QLabel("λ [Å]", self)
        intersect_label.setToolTip("Wavelength in Ångström.")
        d_label = QLabel("d [Å]", self)
        d_label.setToolTip("Interplanar d-spacing in Ångström.")

        self.horizontal_line = QLineEdit()
        self.horizontal_line.setToolTip(
            "Horizontal angle (γ) for the selected peak."
        )
        self.vertical_line = QLineEdit()
        self.vertical_line.setToolTip(
            "Vertical angle (ν) for the selected peak."
        )
        self.intersect_line = QLineEdit()
        self.intersect_line.setToolTip("Wavelength (λ) for the selected peak.")

        self.d_spacing_line = QLineEdit()
        self.d_spacing_line.setToolTip(
            "Interplanar d-spacing (d) for the selected peak."
        )

        self.horizontal_line.setReadOnly(True)
        self.vertical_line.setReadOnly(True)
        self.intersect_line.setReadOnly(True)
        self.d_spacing_line.setReadOnly(True)

        self.horizontal_alt_line = QLineEdit()
        self.horizontal_alt_line.setToolTip(
            "Horizontal angle (γ) for the alternate peak."
        )
        self.vertical_alt_line = QLineEdit()
        self.vertical_alt_line.setToolTip(
            "Vertical angle (ν) for the alternate peak."
        )
        self.intersect_alt_line = QLineEdit()
        self.intersect_alt_line.setToolTip(
            "Wavelength (λ) for the alternate peak."
        )

        self.d_spacing_alt_line = QLineEdit()
        self.d_spacing_alt_line.setToolTip(
            "Interplanar d-spacing (d) for the alternate peak."
        )

        self.horizontal_alt_line.setReadOnly(True)
        self.vertical_alt_line.setReadOnly(True)
        self.intersect_alt_line.setReadOnly(True)
        self.d_spacing_alt_line.setReadOnly(True)

        self.calculate_single_button = QPushButton("Individual", self)
        self.calculate_single_button.setToolTip(
            "Calculate instrument angles for the first peak."
        )
        self.calculate_single_button.setIcon(qta.icon("fa6s.calculator"))
        self.calculate_double_button = QPushButton("Simultaneous", self)
        self.calculate_double_button.setToolTip(
            "Calculate instrument angles for both peaks simultaneously."
        )
        self.calculate_double_button.setIcon(qta.icon("fa6s.calculator"))

        self.calculate_single_alt_button = QPushButton("Individual", self)
        self.calculate_single_alt_button.setToolTip(
            "Calculate instrument angles for the second peak."
        )
        self.calculate_single_alt_button.setIcon(qta.icon("fa6s.calculator"))

        self.equivalents_box = QCheckBox("Equivalents", self)
        self.equivalents_box.setToolTip(
            "Allow equivalent reflections in the calculation."
        )
        self.equivalents_box.setChecked(False)

        calculator_layout.addWidget(h_label, 0, 1, Qt.AlignCenter)
        calculator_layout.addWidget(k_label, 0, 2, Qt.AlignCenter)
        calculator_layout.addWidget(l_label, 0, 3, Qt.AlignCenter)

        calculator_layout.addWidget(peak_1_label, 1, 0)
        calculator_layout.addWidget(self.h1_line, 1, 1)
        calculator_layout.addWidget(self.k1_line, 1, 2)
        calculator_layout.addWidget(self.l1_line, 1, 3)
        calculator_layout.addWidget(self.calculate_single_button, 1, 4)
        calculator_layout.addWidget(self.equivalents_box, 1, 5)

        calculator_layout.addWidget(peak_2_label, 2, 0)
        calculator_layout.addWidget(self.h2_line, 2, 1)
        calculator_layout.addWidget(self.k2_line, 2, 2)
        calculator_layout.addWidget(self.l2_line, 2, 3)
        calculator_layout.addWidget(self.calculate_single_alt_button, 2, 4)
        calculator_layout.addWidget(self.calculate_double_button, 2, 5)

        calculator_layout.addWidget(gamma_label, 0, 6, Qt.AlignCenter)
        calculator_layout.addWidget(nu_label, 0, 7, Qt.AlignCenter)
        calculator_layout.addWidget(intersect_label, 0, 8, Qt.AlignCenter)
        calculator_layout.addWidget(d_label, 0, 9, Qt.AlignCenter)

        calculator_layout.addWidget(self.horizontal_line, 1, 6)
        calculator_layout.addWidget(self.horizontal_alt_line, 2, 6)

        calculator_layout.addWidget(self.vertical_line, 1, 7)
        calculator_layout.addWidget(self.vertical_alt_line, 2, 7)

        calculator_layout.addWidget(self.intersect_line, 1, 8)
        calculator_layout.addWidget(self.intersect_alt_line, 2, 8)

        calculator_layout.addWidget(self.d_spacing_line, 1, 9)
        calculator_layout.addWidget(self.d_spacing_alt_line, 2, 9)

        peak_layout.addLayout(calculator_layout)

        self.canvas_inst = FigureCanvas(Figure(constrained_layout=True))
        self.canvas_laue = FigureCanvas(Figure(constrained_layout=True))

        view_tab = QTabWidget()

        coverage_layout = QVBoxLayout()
        laue_layout = QVBoxLayout()

        coverage_tab = QWidget()
        laue_tab = QWidget()

        coverage_layout.addWidget(NavigationToolbar2QT(self.canvas_inst, self))
        coverage_layout.addWidget(self.canvas_inst)
        laue_layout.addWidget(NavigationToolbar2QT(self.canvas_laue, self))
        laue_layout.addWidget(self.canvas_laue)

        coverage_tab.setLayout(coverage_layout)
        laue_tab.setLayout(laue_layout)

        view_tab.addTab(coverage_tab, "Coverage View")
        view_tab.addTab(laue_tab, "Laue View")

        peak_layout.addWidget(view_tab)

        self.fig_inst = self.canvas_inst.figure
        self.ax_band, self.ax_inst = self.fig_inst.subplots(
            2, 1, height_ratios=[1, 2]
        )
        self.ax_inst.clear()
        self.ax_inst.invert_xaxis()
        self.ax_band.set_yticks([])
        self.ax_band.spines[["left", "right", "top"]].set_visible(False)
        self.ax_band.tick_params(axis="y", left=False, labelleft=False)

        self.cb_inst = None
        self.cb_inst_alt = None

        self.fig_laue = self.canvas_laue.figure
        self.ax_harm, self.ax_laue = self.fig_laue.subplots(
            2, 1, height_ratios=[1, 2]
        )
        self.ax_laue.clear()
        self.ax_laue.invert_xaxis()
        self.ax_harm.set_yticks([])
        self.ax_harm.spines[["left", "right", "top"]].set_visible(False)
        self.ax_harm.tick_params(axis="y", left=False, labelleft=False)

        self.cb_laue = None

        orientation_layout = QHBoxLayout()

        self.combined_box = QCheckBox("Combine All", self)
        self.combined_box.setToolTip(
            "Show all combined peaks or individual orientations."
        )
        self.combined_box.setChecked(True)

        self.add_button = QPushButton("Add Orientation", self)
        self.add_button.setToolTip("Add the current orientation to the plan.")
        self.add_button.setIcon(qta.icon("fa6s.square-plus"))

        self.angles_line = QLineEdit()
        self.angles_line.setToolTip("Goniometer angles for the selected peak.")
        self.angles_line.setReadOnly(True)

        self.comment_line = QLineEdit()
        self.comment_line.setToolTip("Selected peak(s).")
        self.angles_line.setReadOnly(True)

        settings_label = QLabel("Settings:", self)
        angles_label = QLabel("Goniometer:", self)

        self.angles_combo = QComboBox(self)
        self.angles_combo.setToolTip("Select an orientation from the list.")
        self.auto_scale_dropdown(self.angles_combo)

        orientation_layout.addWidget(settings_label)
        orientation_layout.addWidget(self.angles_combo)
        orientation_layout.addWidget(self.combined_box)
        orientation_layout.addStretch(1)
        orientation_layout.addWidget(angles_label)
        orientation_layout.addWidget(self.angles_line)
        orientation_layout.addWidget(self.comment_line)
        orientation_layout.addWidget(self.add_button)

        peak_layout.addLayout(orientation_layout)

        stretch = QHeaderView.Stretch

        self.peaks_table = QTableWidget()
        self.peaks_table.setToolTip(
            "Table of calculated peaks and their properties."
        )
        self.peaks_table.setRowCount(0)
        self.peaks_table.setColumnCount(5)
        self.peaks_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        header = ["h", "k", "l", "d", "λ"]

        self.peaks_table.horizontalHeader().setSectionResizeMode(stretch)
        self.peaks_table.setHorizontalHeaderLabels(header)
        self.peaks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.peaks_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.peaks_table.setSortingEnabled(False)

        peak_layout.addWidget(self.peaks_table)

        inst_tab.setLayout(peak_layout)

    def set_default_symmetry(self):
        self.crystal_combo.setCurrentIndex(0)
        self.point_group_combo.setCurrentIndex(1)
        self.lattice_centering_combo.setCurrentIndex(0)

    def connect_convert_mesh_to_hkl(self, convert_to_hkl):
        self.coverage_mesh_button.clicked.connect(convert_to_hkl)

    def connect_convert_plan_to_hkl(self, convert_to_hkl):
        self.coverage_plan_button.clicked.connect(convert_to_hkl)

    def connect_slice_thickness_line(self, update_slice):
        self.slice_thickness_line.editingFinished.connect(update_slice)

    def connect_slice_line(self, update_slice):
        self.slice_line.editingFinished.connect(update_slice)

    def connect_slice_combo(self, update_slice):
        self.slice_combo.currentIndexChanged.connect(update_slice)

    def connect_combined(self, update_combined):
        self.combined_box.toggled.connect(update_combined)

    def connect_peak_table(self, update_table):
        self.angles_combo.activated.connect(update_table)

    def connect_add_orientation(self, add_orientation):
        self.add_button.clicked.connect(add_orientation)

    def connect_delete_angles(self, delete_angles):
        self.delete_button.clicked.connect(delete_angles)

    def connect_highlight_angles(self, highlight_angles):
        self.highlight_button.clicked.connect(highlight_angles)

    def connect_update(self, update):
        self.update_button.clicked.connect(update)

    def connect_calculate_single(self, calculate_single):
        self.calculate_single_button.clicked.connect(calculate_single)

    def connect_calculate_double(self, calculate_double):
        self.calculate_double_button.clicked.connect(calculate_double)

    def connect_calculate_single_alt(self, calculate_single):
        self.calculate_single_alt_button.clicked.connect(calculate_single)

    def connect_switch_crystal(self, switch_crystal):
        self.crystal_combo.activated.connect(switch_crystal)

    def connect_switch_point_group(self, switch_group):
        self.point_group_combo.activated.connect(switch_group)

    def connect_switch_lattice_centering(self, switch_centering):
        self.lattice_centering_combo.activated.connect(switch_centering)

    def connect_switch_instrument(self, switch_instrument):
        self.instrument_combo.activated.connect(switch_instrument)

    def connect_update_goniometer(self, update_goniometer):
        self.mode_combo.activated.connect(update_goniometer)

    def connect_optimize(self, optimize):
        self.optimize_button.clicked.connect(optimize)

    def connect_mesh(self, mesh):
        self.mesh_button.clicked.connect(mesh)

    def connect_load_UB(self, load_UB):
        self.load_UB_button.clicked.connect(load_UB)

    def connect_calculate_plane(self, calculate_plane):
        self.coverage_plane_button.clicked.connect(calculate_plane)

    def connect_add_plane(self, add_plane):
        self.plane_button.clicked.connect(add_plane)

    def get_plane_hkl_1(self):
        try:
            return [
                float(self.plane_u1_line.text()),
                float(self.plane_u2_line.text()),
                float(self.plane_u3_line.text()),
            ]
        except ValueError:
            return None

    def get_plane_hkl_2(self):
        try:
            return [
                float(self.plane_v1_line.text()),
                float(self.plane_v2_line.text()),
                float(self.plane_v3_line.text()),
            ]
        except ValueError:
            return None

    def get_plane_max_angle(self):
        try:
            return float(self.plane_max_angle_line.text())
        except ValueError:
            return 360.0

    def get_plane_n_steps(self):
        try:
            return int(self.plane_n_steps_line.text())
        except ValueError:
            return 360

    def connect_reset(self, reset):
        self.reset_button.clicked.connect(reset)

    def connect_show_instrument(self, show):
        self.instrument_button.clicked.connect(show)

    def connect_save_CSV(self, save_CSV):
        self.save_plan_button.clicked.connect(save_CSV)

    def connect_load_experiment(self, load_experiment):
        self.load_experiment_button.clicked.connect(load_experiment)

    def connect_save_experiment(self, save_experiment):
        self.save_experiment_button.clicked.connect(save_experiment)

    def connect_wavelength(self, update_wavelength):
        self.wl_min_line.editingFinished.connect(update_wavelength)

    def connect_move_up(self, move_up):
        self.move_up_button.clicked.connect(move_up)

    def connect_move_down(self, move_down):
        self.move_down_button.clicked.connect(move_down)

    def connect_load_mask(self, load_mask):
        self.mask_browse_button.clicked.connect(load_mask)

    def connect_load_detector(self, load_detector_cal):
        self.cal_browse_button.clicked.connect(load_detector_cal)

    def connect_load_goniometer(self, load_goniometer_cal):
        self.gon_browse_button.clicked.connect(load_goniometer_cal)

    def connect_peak_row_highlighter(self, highlight_row):
        self.peaks_table.itemSelectionChanged.connect(highlight_row)

    def connect_color_scheme(self, update_color_scheme):
        self.color_combo.currentIndexChanged.connect(update_color_scheme)

    def connect_hkl_limits(self, update_hkl_limits):
        self.d_min_line.editingFinished.connect(update_hkl_limits)

    def get_color_scheme(self):
        return self.color_combo.currentText()

    def set_hkl_limits(self, h_max, k_max, l_max):
        self.h_max_line.setText(str(h_max))
        self.k_max_line.setText(str(k_max))
        self.l_max_line.setText(str(l_max))

    def get_detector_calibration(self):
        return self.cal_line.text()

    def set_detector_calibration(self, filename):
        return self.cal_line.setText(filename)

    def get_goniometer_calibration(self):
        return self.gon_line.text()

    def set_goniometer_calibration(self, filename):
        return self.gon_line.setText(filename)

    def get_mask(self):
        return self.mask_line.text()

    def set_mask(self, filename):
        self.mask_line.setText(filename)

    def load_detector_cal_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Calibration files (*.DetCal *.detcal *.xml)"

        filename, _ = file_dialog.getOpenFileName(
            self,
            "Load calibration file",
            path or self._get_file_dialog_dir(),
            file_filters,
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def load_goniometer_cal_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Goniometer calibration files (*.xml)"

        filename, _ = file_dialog.getOpenFileName(
            self,
            "Load calibration file",
            path or self._get_file_dialog_dir(),
            file_filters,
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def load_mask_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Mask files (*.xml)"

        filename, _ = file_dialog.getOpenFileName(
            self,
            "Load mask file",
            path or self._get_file_dialog_dir(),
            file_filters,
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def load_UB_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getOpenFileName(
            self,
            "Load UB file",
            path or self._get_file_dialog_dir(),
            "UB files (*.mat)",
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def save_CSV_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self,
            "Save peaks file",
            path or self._get_file_dialog_dir(),
            "Experiment files (*.csv)",
            options=options,
        )

        if filename is not None:
            if not filename.endswith(".csv"):
                filename += ".csv"
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def load_experiment_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getOpenFileName(
            self,
            "Load experiment file",
            path or self._get_file_dialog_dir(),
            "Experiment files (*.nxs)",
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def save_experiment_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self,
            "Save experiment file",
            path or self._get_file_dialog_dir(),
            "Experiment files (*.nxs)",
            options=options,
        )

        if filename is not None:
            if not filename.endswith(".nxs"):
                filename += ".nxs"
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def get_d_min(self):
        if self.d_min_line.hasAcceptableInput():
            return float(self.d_min_line.text())

    def set_d_min(self, d_min):
        self.d_min_line.setText(str(d_min))

    def get_crystal_system(self):
        return self.crystal_combo.currentText()

    def set_crystal_system(self, crystal_system):
        index = self.crystal_combo.findText(crystal_system)
        if index >= 0:
            self.crystal_combo.blockSignals(True)
            self.crystal_combo.setCurrentIndex(index)
            self.crystal_combo.blockSignals(False)

    def set_point_groups(self, groups):
        self.point_group_combo.clear()
        for group in groups:
            self.point_group_combo.addItem(group)
        self.auto_scale_dropdown(self.point_group_combo)

    def get_point_group(self):
        return self.point_group_combo.currentText()

    def set_point_group(self, point_group):
        index = self.point_group_combo.findText(point_group)
        if index >= 0:
            self.point_group_combo.blockSignals(True)
            self.point_group_combo.setCurrentIndex(index)
            self.point_group_combo.blockSignals(False)

    def set_lattice_centerings(self, centerings):
        self.lattice_centering_combo.clear()
        for centering in centerings:
            self.lattice_centering_combo.addItem(centering)
        self.auto_scale_dropdown(self.lattice_centering_combo)

    def get_lattice_centering(self):
        return self.lattice_centering_combo.currentText()

    def set_lattice_centering(self, lattice_centering):
        index = self.lattice_centering_combo.findText(lattice_centering)
        if index >= 0:
            self.lattice_centering_combo.blockSignals(True)
            self.lattice_centering_combo.setCurrentIndex(index)
            self.lattice_centering_combo.blockSignals(False)

    def get_mode(self):
        return self.mode_combo.currentText()

    def set_mode(self, mode):
        index = self.mode_combo.findText(mode)
        if index >= 0:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(False)

    def set_modes(self, modes):
        self.mode_combo.clear()
        for mode in modes:
            self.mode_combo.addItem(mode)
        self.auto_scale_dropdown(self.mode_combo)

    def set_counting_options(self, options):
        self.count_combo.clear()
        for option in options:
            self.count_combo.addItem(option)
        self.auto_scale_dropdown(self.count_combo)

    def get_counting_options(self):
        return [
            self.count_combo.itemText(i)
            for i in range(self.count_combo.count())
        ]

    def get_counting_index(self):
        return self.count_combo.currentIndex()

    def get_count_value(self):
        if self.count_line.hasAcceptableInput():
            return float(self.count_line.text())

    def set_peak_list(self, rows):
        self.angles_combo.blockSignals(True)
        self.angles_combo.clear()
        self.angles_combo.addItem("Missing")
        for row in range(rows):
            self.angles_combo.addItem("#" + (str(row + 1)))
        self.angles_combo.setCurrentIndex(self.angles_combo.count() - 1)
        self.angles_combo.blockSignals(False)
        self.auto_scale_dropdown(self.angles_combo)

    def get_peak_list(self):
        val = self.angles_combo.currentText()
        if val is not None:
            if val.split(":")[0].lstrip("#").isdigit():
                return int(val.split(":")[0].lstrip("#")) - 1
            else:
                return -1

    def set_wavelength(self, wavelength):
        self.wl_min_line.blockSignals(True)
        self.wl_max_line.blockSignals(True)
        if type(wavelength) is list:
            self.wl_min_line.setText(str(wavelength[0]))
            self.wl_max_line.setText(str(wavelength[1]))
            self.wl_max_line.setEnabled(True)
        else:
            self.wl_min_line.setText(str(wavelength))
            self.wl_max_line.setText(str(wavelength))
            self.wl_max_line.setReadOnly(True)
        self.wl_min_line.blockSignals(False)
        self.wl_max_line.blockSignals(False)

    def get_wavelength(self):
        params = self.wl_min_line, self.wl_max_line

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return [float(param.text()) for param in params]

    def update_wavelength(self, lamda_min):
        if not self.wl_max_line.isEnabled():
            self.wl_max_line.setText(str(lamda_min))

    def update_tables(self, title, goniometers, motors):
        self.goniometer_table.clearContents()
        self.goniometer_table.setRowCount(0)
        self.goniometer_table.setRowCount(len(goniometers))
        self.goniometer_table.blockSignals(True)

        free = []
        for row, gon in enumerate(goniometers):
            angle, amin, amax = gon
            amin, amax = str(amin), str(amax)
            self.goniometer_table.setItem(row, 0, QTableWidgetItem(angle))
            self.goniometer_table.setItem(row, 1, QTableWidgetItem(amin))
            self.goniometer_table.setItem(row, 2, QTableWidgetItem(amax))
            item = self.goniometer_table.item(row, 0)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if float(amin) == float(amax):
                for j in [1, 2]:
                    item = self.goniometer_table.item(row, j)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            else:
                free.append(angle)
        self.goniometer_table.blockSignals(False)

        self.motor_table.setRowCount(0)
        self.motor_table.setRowCount(len(motors))

        for row, mot in enumerate(motors):
            setting, val = mot
            val = str(val)
            self.motor_table.setItem(row, 0, QTableWidgetItem(setting))
            self.motor_table.setItem(row, 1, QTableWidgetItem(val))
            item = self.motor_table.item(row, 0)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        self.plan_table.blockSignals(True)
        self.plan_table.clearContents()
        self.plan_table.setRowCount(0)
        self.plan_table.setColumnCount(0)
        self.plan_table.setColumnCount(len(free) + 5)

        clean = [item.split(":")[-1] for item in free]

        css = ["Comment", "Wait For", "Value", "Use"]

        labels = [title.split(":")[-1]] + clean + css
        long_labels = [title] + free + css

        resize = QHeaderView.Stretch

        self.plan_table.horizontalHeader().setSectionResizeMode(resize)
        self.plan_table.setHorizontalHeaderLabels(labels)

        for col, long_text in enumerate(long_labels):
            item = self.plan_table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(long_text)
                item.setData(Qt.UserRole, long_text)

        self.plan_table.blockSignals(False)

        self.mesh_table.clearContents()
        self.mesh_table.setRowCount(0)
        self.mesh_table.setRowCount(len(free))

        self.mesh_table.blockSignals(True)
        row = 0
        for gon in goniometers:
            angle, amin, amax = gon
            if not float(amin) == float(amax):
                amin, amax = str(amin), str(amax)
                self.mesh_table.setItem(row, 0, QTableWidgetItem(angle))
                self.mesh_table.setItem(row, 1, QTableWidgetItem(amin))
                self.mesh_table.setItem(row, 2, QTableWidgetItem(amax))
                self.mesh_table.setItem(row, 3, QTableWidgetItem("1"))

                min_val = float(amin)
                max_val = float(amax)
                angle_val = 1.0
                step = (
                    (max_val - min_val) / (angle_val - 1)
                    if angle_val > 1
                    else 0.0
                )
                step_item = QTableWidgetItem("{:.2f}".format(step))
                step_item.setFlags(step_item.flags() & ~Qt.ItemIsEditable)
                self.mesh_table.setItem(row, 4, step_item)

                row += 1
        self.mesh_table.blockSignals(False)

    def update_limits(self, item):
        text = item.text()
        row, col = item.row(), item.column()

        angle = self.goniometer_table.item(row, 0).text()

        rows = self.mesh_table.rowCount()

        if (
            text.lstrip("-").replace(".", "").isnumeric()
            and len(text.split(".")) <= 2
        ):
            value = float(text)
            if value.is_integer():
                value = int(value)
            for row in range(rows):
                if angle == self.mesh_table.item(row, 0).text():
                    text = QTableWidgetItem(str(value))
                    self.mesh_table.setItem(row, col, text)

    def calculate_mesh_step(self, item):
        """Automatically calculate step size based on min, max, and angle."""
        row = item.row()
        col = item.column()

        if col not in [1, 2, 3]:
            return

        min_item = self.mesh_table.item(row, 1)
        max_item = self.mesh_table.item(row, 2)
        angles_item = self.mesh_table.item(row, 3)

        if not all([min_item, max_item, angles_item]):
            return

        min_text = min_item.text().strip()
        max_text = max_item.text().strip()
        angles_text = angles_item.text().strip()

        if not all([min_text, max_text, angles_text]):
            return

        min_val = float(min_text)
        max_val = float(max_text)
        angles_val = float(angles_text)

        if angles_val <= 0:
            return

        step = (
            (max_val - min_val) / (angles_val - 1) if angles_val > 1 else 0.0
        )

        self.mesh_table.blockSignals(True)
        step_item = QTableWidgetItem("{:.2f}".format(step))
        step_item.setFlags(step_item.flags() & ~Qt.ItemIsEditable)
        self.mesh_table.setItem(row, 4, step_item)
        self.mesh_table.blockSignals(False)

    def get_mesh_angles(self):
        rows = self.mesh_table.rowCount()

        all_angles = self.get_all_angles()
        n = len(all_angles)

        limits = self.get_goniometer_limits()
        angles = [1] * n

        for row in range(rows):
            ind = all_angles.index(self.mesh_table.item(row, 0).text())
            limits[ind][0] = float(self.mesh_table.item(row, 1).text())
            limits[ind][1] = float(self.mesh_table.item(row, 2).text())
            angles[ind] = int(float(self.mesh_table.item(row, 3).text()))
        return limits, angles

    def get_plan_angles(self):
        col = self.plan_table.columnCount() - 1

        all_angles = self.get_all_angles()
        n = len(all_angles)

        limits = self.get_goniometer_limits()
        angles = [1] * n

        settings = []
        for row in range(self.get_number_of_orientations()):
            setting = self.get_angle_setting(row)
            item = self.plan_table.item(row, col)
            use = item.checkState() == Qt.Checked
            if use:
                k = 0
                values = angles.copy()
                for j, angle in enumerate(limits):
                    if np.isclose(angle[1], angle[0]):
                        values[j] = float(angle[0])
                    else:
                        values[j] = float(setting[k])
                        k += 1
                settings.append(values)

        return settings

    def get_angles_to_delete(self):
        self.plan_table.blockSignals(True)
        self.plan_table.setUpdatesEnabled(False)
        self.plan_table.setSortingEnabled(False)
        rows = self.get_selected_plan_rows()
        return rows

    def get_selected_plan_rows(self):
        return sorted(
            set(index.row() for index in self.plan_table.selectedIndexes())
        )

    def delete_angles(self, rows):
        self.plan_table.blockSignals(True)
        self.plan_table.setUpdatesEnabled(False)
        self.plan_table.setSortingEnabled(False)
        self.plan_table.clearSelection()

        for row in sorted(rows, reverse=True):
            self.plan_table.removeRow(row)

        self.set_peak_list(self.get_number_of_orientations())

        self.plan_table.setSortingEnabled(True)
        self.plan_table.setUpdatesEnabled(True)
        self.plan_table.blockSignals(False)

    def get_selected_angle(self):
        self.plan_table.blockSignals(True)
        self.plan_table.setUpdatesEnabled(False)
        self.plan_table.setSortingEnabled(False)
        rows = self.get_selected_plan_rows()
        if len(rows) == 1:
            return rows[0]
        else:
            self.plan_table.blockSignals(False)
            self.plan_table.setUpdatesEnabled(True)
            self.plan_table.setSortingEnabled(True)

    def swap_angles(self, rows):
        if rows[0] == rows[1]:
            self.plan_table.setSortingEnabled(True)
            self.plan_table.setUpdatesEnabled(True)
            self.plan_table.blockSignals(False)
            return

        self.plan_table.blockSignals(True)
        self.plan_table.setUpdatesEnabled(False)
        self.plan_table.setSortingEnabled(False)
        self.plan_table.clearSelection()

        cols = self.plan_table.columnCount()

        for c in range(cols):
            w1 = self.plan_table.cellWidget(rows[0], c)
            w2 = self.plan_table.cellWidget(rows[1], c)

            if w1 is not None or w2 is not None:
                if isinstance(w1, QComboBox):
                    index1 = w1.currentIndex()
                else:
                    index1 = None
                if isinstance(w2, QComboBox):
                    index2 = w2.currentIndex()
                else:
                    index2 = None

                self.plan_table.removeCellWidget(rows[0], c)
                self.plan_table.removeCellWidget(rows[1], c)

                if w2 is not None and isinstance(w2, QComboBox):
                    combobox = QComboBox()
                    for i in range(w2.count()):
                        combobox.addItem(w2.itemText(i))
                    if index2 is not None:
                        combobox.setCurrentIndex(index2)
                    self.plan_table.setCellWidget(rows[0], c, combobox)

                if w1 is not None and isinstance(w1, QComboBox):
                    combobox = QComboBox()
                    for i in range(w1.count()):
                        combobox.addItem(w1.itemText(i))
                    if index1 is not None:
                        combobox.setCurrentIndex(index1)
                    self.plan_table.setCellWidget(rows[1], c, combobox)
            else:
                item1 = self.plan_table.takeItem(rows[0], c)
                item2 = self.plan_table.takeItem(rows[1], c)

                if item1 is None:
                    item1 = QTableWidgetItem("")
                if item2 is None:
                    item2 = QTableWidgetItem("")

                self.plan_table.setItem(rows[0], c, item2)
                self.plan_table.setItem(rows[1], c, item1)

        self.plan_table.setCurrentCell(
            rows[1], self.plan_table.currentColumn()
        )

        self.plan_table.setSortingEnabled(True)
        self.plan_table.setUpdatesEnabled(True)
        self.plan_table.blockSignals(False)

    def highlight_angles(self):
        self.plan_table.selectAll()
        self.plan_table.setFocus()

    def get_title(self):
        return self.title_line.text()

    def update_counting(self):
        self.plan_table.blockSignals(True)
        self.plan_table.setUpdatesEnabled(False)
        self.plan_table.setSortingEnabled(False)

        title = self.get_title()
        index = self.get_counting_index()
        value = self.get_count_value()

        col = self.plan_table.columnCount() - 3

        rows = self.get_selected_plan_rows()
        for row in rows:
            if title is not None:
                item = QTableWidgetItem(title)
                self.plan_table.setItem(row, 0, item)
            if index is not None:
                widget = self.plan_table.cellWidget(row, col)
                if isinstance(widget, QComboBox):
                    widget.setCurrentIndex(index)
            if value is not None:
                item = QTableWidgetItem("{:.3f}".format(value))
                self.plan_table.setItem(row, col + 1, item)

        self.plan_table.setUpdatesEnabled(True)
        self.plan_table.setSortingEnabled(True)
        self.plan_table.blockSignals(False)

        return rows

    def get_all_angles(self):
        rows = self.goniometer_table.rowCount()

        angles = [
            self.goniometer_table.item(row, 0).text() for row in range(rows)
        ]

        return angles

    def get_free_angles(self):
        cols = self.plan_table.columnCount() - 5

        angles = [
            self.plan_table.horizontalHeaderItem(i + 1).data(Qt.UserRole)
            for i in range(cols)
        ]

        return angles

    def get_number_of_orientations(self):
        return self.plan_table.rowCount()

    def get_orientations_to_use(self):
        col = self.plan_table.columnCount() - 1

        use = []
        for row in range(self.get_number_of_orientations()):
            item = self.plan_table.item(row, col)
            use.append(item.checkState() == Qt.Checked)

        return use

    def get_all_titles(self):
        title = []
        for row in range(self.get_number_of_orientations()):
            item = self.plan_table.item(row, 0).text()
            title.append(item)

        return title

    def get_all_values(self):
        col = self.plan_table.columnCount() - 2

        value = []
        for row in range(self.get_number_of_orientations()):
            item = self.plan_table.item(row, col).text()
            item = float(item) if item.replace(".", "").isnumeric() else 0.0
            value.append(item)

        return value

    def get_all_countings(self):
        col = self.plan_table.columnCount() - 3

        count = []
        for row in range(self.get_number_of_orientations()):
            widget = self.plan_table.cellWidget(row, col)
            if isinstance(widget, QComboBox):
                count.append(widget.currentText())
            else:
                count.append("")

        return count

    def get_all_comments(self):
        col = self.plan_table.columnCount() - 4

        comment = []
        for row in range(self.get_number_of_orientations()):
            comment.append(self.plan_table.item(row, col).text())

        return comment

    def get_all_settings(self):
        settings = []
        for row in range(self.get_number_of_orientations()):
            setting = self.get_angle_setting(row)
            settings.append(setting)

        return settings

    def get_settings(self):
        if self.settings_line.hasAcceptableInput():
            return int(self.settings_line.text())

    def get_optimized_settings(self):
        col = self.plan_table.columnCount() - 5

        opt = []
        for row in range(self.get_number_of_orientations()):
            item = self.plan_table.item(row, col + 1)
            opt.append(item.text() == "CrystalPlan")

        return opt

    def get_angle_setting(self, row):
        cols = self.plan_table.columnCount() - 5

        setting = []
        for col in range(cols):
            setting.append(float(self.plan_table.item(row, col + 1).text()))

        return setting

    def add_orientations(self, title, comment, angles_list):
        rows = self.get_number_of_orientations()
        self.plan_table.blockSignals(True)
        self.plan_table.setUpdatesEnabled(False)
        self.plan_table.setSortingEnabled(False)
        self.plan_table.setRowCount(rows + len(angles_list))

        for i, angles in enumerate(angles_list):

            row = rows + i

            col = 0

            item = QTableWidgetItem(title)
            self.plan_table.setItem(row, col, item)
            col += 1

            for angle in angles:
                item = QTableWidgetItem("{:.1f}".format(angle))
                self.plan_table.setItem(row, col, item)
                col += 1

            self.plan_table.setItem(row, col, QTableWidgetItem(comment))
            col += 1

            combobox = QComboBox()
            options = self.get_counting_options()
            for option in options:
                combobox.addItem(option)
            index = self.get_counting_index()
            if index is not None:
                combobox.setCurrentIndex(index)
            self.plan_table.setCellWidget(row, col, combobox)
            col += 1

            val = self.get_count_value()
            if val is not None:
                item = QTableWidgetItem("{:.3f}".format(val))
                self.plan_table.setItem(row, col, item)
            col += 1

            flags = (
                Qt.ItemIsSelectable | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
            )

            checkbox = QTableWidgetItem("")
            checkbox.setText("")
            checkbox.setFlags(flags)
            checkbox.setCheckState(Qt.Checked)
            self.plan_table.setItem(row, col, checkbox)

        self.plan_table.setSortingEnabled(False)
        self.plan_table.setUpdatesEnabled(True)
        self.plan_table.blockSignals(False)
        self.set_peak_list(self.get_number_of_orientations())

    def add_settings(self, titles, settings, comments, counts, values, use):
        self.plan_table.setUpdatesEnabled(False)
        self.plan_table.setSortingEnabled(False)
        self.plan_table.blockSignals(True)
        self.plan_table.clearContents()
        self.plan_table.setRowCount(len(use))

        for row, angles in enumerate(settings):
            col = 0

            item = QTableWidgetItem(titles[row])
            self.plan_table.setItem(row, col, item)
            col += 1

            for angle in angles:
                item = QTableWidgetItem("{:.1f}".format(angle))
                self.plan_table.setItem(row, col, item)
                col += 1

            self.plan_table.setItem(row, col, QTableWidgetItem(comments[row]))
            col += 1

            combobox = QComboBox()
            options = self.get_counting_options()
            for option in options:
                combobox.addItem(option)
            if counts[row] in options:
                index = options.index(counts[row])
                combobox.setCurrentIndex(index)

            self.plan_table.setCellWidget(row, col, combobox)
            col += 1

            item = QTableWidgetItem("{:.3f}".format(values[row]))
            self.plan_table.setItem(row, col, item)
            col += 1

            flags = (
                Qt.ItemIsSelectable | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
            )

            checkbox = QTableWidgetItem("")
            checkbox.setText("")
            checkbox.setFlags(flags)
            checkbox.setCheckState(Qt.Checked if use[row] else Qt.Unchecked)
            self.plan_table.setItem(row, col, checkbox)

        self.plan_table.setUpdatesEnabled(True)
        self.plan_table.setSortingEnabled(False)
        self.plan_table.blockSignals(False)

        self.set_peak_list(self.get_number_of_orientations())

    def handle_item_changed(self, item):
        self.plan_table.blockSignals(True)
        self.plan_table.setUpdatesEnabled(False)
        self.plan_table.setSortingEnabled(False)

        col = item.column()

        if col == self.plan_table.columnCount() - 1:
            rows = self.get_selected_plan_rows()
            if item.row() not in rows:
                rows.append(item.row())

            state = item.checkState()
            for row in rows:
                checkbox = self.plan_table.item(row, col)
                if checkbox is not None:
                    checkbox.setCheckState(state)

            self.viz_ready.emit()

        self.plan_table.setUpdatesEnabled(True)
        self.plan_table.setSortingEnabled(True)
        self.plan_table.blockSignals(False)

    def connect_visualization_ready(self, visualize):
        self.viz_ready.connect(visualize)

    def get_instrument(self):
        return self.instrument_combo.currentText()

    def set_instrument(self, instrument):
        index = self.instrument_combo.findText(instrument)
        if index >= 0:
            self.instrument_combo.blockSignals(True)
            self.instrument_combo.setCurrentIndex(index)
            self.instrument_combo.blockSignals(False)

    def get_motors(self):
        logs = {}
        for row in range(self.motor_table.rowCount()):
            setting = self.motor_table.item(row, 0).text()
            logs[setting] = float(self.motor_table.item(row, 1).text())

        return logs

    def set_motors(self, values):
        for row, value in enumerate(values):
            self.motor_table.setItem(row, 1, self.set_item_value(str(value)))

    def get_goniometer_limits(self):
        limits = []
        for row in range(self.goniometer_table.rowCount()):
            amin = float(self.goniometer_table.item(row, 1).text())
            amax = float(self.goniometer_table.item(row, 2).text())
            limits.append([amin, amax])

        return limits

    def set_goniometer_limits(self, limits):
        for row, limit in enumerate(limits):
            amin, amax = str(limit[0]), str(limit[1])
            self.goniometer_table.setItem(row, 1, self.set_item_value(amin))
            self.goniometer_table.setItem(row, 2, self.set_item_value(amax))

    def add_instrument(self, inst_dict):
        self.clear_scene()

        points = inst_dict["points"]
        faces = inst_dict["faces"]
        rx, ry, rz = inst_dict["radius"]

        mesh = pv.PolyData(points, faces)

        self.plotter.add_mesh(mesh, render_lines_as_tubes=True)

        mesh = pv.Line(pointa=(-rx, 0, 0), pointb=(rx, 0, 0), resolution=1)

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        mesh = pv.Line(pointa=(0, -ry, 0), pointb=(0, ry, 0), resolution=1)

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        mesh = pv.Line(pointa=(0, 0, -rz), pointb=(0, 0, rz), resolution=1)

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        arrow = pv.Arrow(start=[0, 0, -rz], direction=[0, 0, rz], scale="auto")
        self.plotter.add_mesh(arrow, color="pink", smooth_shading=True)

        self.plotter.add_legend_scale(
            corner_offset_factor=2,
            bottom_border_offset=50,
            top_border_offset=50,
            left_border_offset=100,
            right_border_offset=100,
            legend_visibility=True,
            xy_label_mode=False,
        )

        self.plotter.enable_depth_peeling()

        self.reset_view()

    def add_peaks(self, peak_dict):
        self.clear_scene()

        if peak_dict is None:
            self.reset_view()
            return None

        coords = np.array(peak_dict["coords"], dtype=float)
        colors = np.array(peak_dict["colors"])

        points = pv.PolyData(coords)
        points["colors"] = colors

        size = 5 if peak_dict["type"] == "filtered" else 10
        spheres = False if peak_dict["type"] == "missing" else True

        self.plotter.add_mesh(
            points,
            scalars="colors",
            rgb=True,
            smooth_shading=True,
            point_size=size,
            render_points_as_spheres=spheres,
        )

        self.plotter.enable_depth_peeling()

        coords = np.array(peak_dict["axis_coords"])
        colors = np.array(peak_dict["axis_colors"])

        for i in range(3):
            arrow = pv.Arrow(
                start=[0, 0, 0], direction=coords[i], scale="auto"
            )
            self.plotter.add_mesh(arrow, color=colors[i], smooth_shading=True)

        radius = 0.2 * np.sqrt(np.min(np.sum(coords**2, axis=1)))
        sphere = pv.Sphere(radius=radius)

        self.plotter.add_mesh(sphere, color="w", smooth_shading=True)

        Q_max = 2 * np.pi / peak_dict["axis_limit"]

        mesh = pv.Line(
            pointa=(-Q_max, 0, 0), pointb=(Q_max, 0, 0), resolution=1
        )

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        mesh = pv.Line(
            pointa=(0, -Q_max, 0), pointb=(0, Q_max, 0), resolution=1
        )

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        mesh = pv.Line(
            pointa=(0, 0, -Q_max), pointb=(0, 0, Q_max), resolution=1
        )

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        self.reset_view()

    def update_peaks_table(self, peaks):
        self.peaks_table.blockSignals(True)
        self.peaks_table.clearSelection()
        self.peaks_table.setRowCount(0)
        self.peaks_table.setRowCount(len(peaks))

        for row, peak in enumerate(peaks):
            self.set_peak(row, peak)

        self.peaks_table.blockSignals(False)

    def set_peak(self, row, peak):
        h, k, l, d, lamda = peak
        h = "{:.3f}".format(h)
        k = "{:.3f}".format(k)
        l = "{:.3f}".format(l)
        d = "{:.4f}".format(d)
        lamda = "{:.4f}".format(lamda)
        self.peaks_table.setItem(row, 0, self.set_item_value(h, row))
        self.peaks_table.setItem(row, 1, self.set_item_value(k, row))
        self.peaks_table.setItem(row, 2, self.set_item_value(l, row))
        self.peaks_table.setItem(row, 3, self.set_item_value(d, row))
        self.peaks_table.setItem(row, 4, self.set_item_value(lamda, row))

    def set_item_value(self, value, row=0):
        item = QTableWidgetItem()
        item.setData(Qt.DisplayRole, float(value))
        item.setData(Qt.UserRole, row)
        return item

    def get_input_hkls(self):
        params_1 = self.h1_line, self.k1_line, self.l1_line
        params_2 = self.h2_line, self.k2_line, self.l2_line

        valid_params = all([param.hasAcceptableInput() for param in params_1])

        if valid_params:
            params_1 = [float(param.text()) for param in params_1]
        else:
            params_1 = None

        valid_params = all([param.hasAcceptableInput() for param in params_2])

        if valid_params:
            params_2 = [float(param.text()) for param in params_2]
        else:
            params_2 = None

        return params_1, params_2

    def plot_statistics(self, sym, asym, cumsym, cumasym):
        self.ax_cov[0].clear()
        self.ax_cov[1].clear()
        self.ax_cov[2].clear()

        self.ax_cum[0].clear()
        self.ax_cum[1].clear()
        self.ax_cum[2].clear()

        color = plt.get_cmap("tab20").colors

        width = 1 / 3

        shel, comp, mult, refl = sym

        x = np.arange(len(shel))

        self.ax_cov[0].bar(x, comp, width, color=color[0], label="Symmetry")
        self.ax_cov[1].bar(x, mult, width, color=color[2], label="Symmetry")
        self.ax_cov[2].bar(x, refl, width, color=color[4], label="Symmetry")

        shel, comp, mult, refl = asym

        self.ax_cov[0].bar(
            x + width, comp, width, color=color[1], label="Asymmetry"
        )
        self.ax_cov[1].bar(
            x + width, mult, width, color=color[3], label="Asymmetry"
        )
        self.ax_cov[2].bar(
            x + width, refl, width, color=color[5], label="Asymmetry"
        )

        self.ax_cov[0].legend(shadow=True)
        self.ax_cov[1].legend(shadow=True)
        self.ax_cov[2].legend(shadow=True)

        self.ax_cov[0].set_ylim(0, 100)
        self.ax_cov[1].set_ylim(1, self.ax_cov[1].get_ylim()[1])

        self.ax_cov[0].minorticks_on()
        self.ax_cov[1].minorticks_on()
        self.ax_cov[2].minorticks_on()

        self.ax_cov[2].set_xlabel("Resolution Shell [Å]")
        self.ax_cov[0].set_ylabel("Completeness")
        self.ax_cov[1].set_ylabel("Redundancy")
        self.ax_cov[2].set_ylabel("Unique")

        self.ax_cov[0].yaxis.set_major_formatter(PercentFormatter(100))

        self.ax_cov[0].set_xticks(x + width, shel)
        self.ax_cov[1].set_xticks(x + width, shel)
        self.ax_cov[2].set_xticks(x + width, shel)

        self.canvas_cov.draw_idle()

        x, comp, mult, refl = cumsym

        self.ax_cum[0].plot(
            x, comp, "-o", color=color[0], label="Symmetry", clip_on=False
        )
        self.ax_cum[1].plot(
            x, mult, "-o", color=color[2], label="Symmetry", clip_on=False
        )
        self.ax_cum[2].plot(
            x, refl, "-o", color=color[4], label="Symmetry", clip_on=False
        )

        x, comp, mult, refl = cumasym

        self.ax_cum[0].plot(
            x, comp, "-o", color=color[1], label="Asymmetry", clip_on=False
        )
        self.ax_cum[1].plot(
            x, mult, "-o", color=color[3], label="Asymmetry", clip_on=False
        )
        self.ax_cum[2].plot(
            x, refl, "-o", color=color[5], label="Asymmetry", clip_on=False
        )

        self.ax_cum[0].legend(shadow=True)
        self.ax_cum[1].legend(shadow=True)
        self.ax_cum[2].legend(shadow=True)

        self.ax_cum[0].set_ylim(0, 100)
        self.ax_cum[1].set_ylim(1, self.ax_cum[1].get_ylim()[1])

        self.ax_cum[0].minorticks_on()
        self.ax_cum[1].minorticks_on()
        self.ax_cum[2].minorticks_on()

        self.ax_cum[2].set_xlabel("Orientation Number")
        self.ax_cum[0].set_ylabel("Completeness")
        self.ax_cum[1].set_ylabel("Redundancy")
        self.ax_cum[2].set_ylabel("Unique")

        if len(x) >= 2:
            self.ax_cum[2].xaxis.get_major_locator().set_params(integer=True)

        self.canvas_cum.draw_idle()

    def _plot_instrument_background(self, inst_background):
        if inst_background is None:
            return

        img = inst_background["img"]
        xedges = inst_background["xedges"]
        yedges = inst_background["yedges"]

        self.ax_inst.pcolormesh(
            xedges,
            yedges,
            img.T,
            shading="flat",
            cmap=ListedColormap(["white", "lightgray"]),
            vmin=0,
            vmax=1,
            rasterized=True,
            zorder=0,
        )

    def plot_instrument(self, inst_background, gamma, nu, lamda):
        if self.cb_inst is not None:
            self.cb_inst.remove()
            self.cb_inst = None

        if self.cb_inst_alt is not None:
            self.cb_inst_alt.remove()
            self.cb_inst_alt = None

        x = self.get_wavelength()

        self.bragg_band = 0
        self.bragg_band_alt = None
        self.ax_band.clear()
        self.ax_band.minorticks_on()
        self.ax_band.set_xlabel(r"$\lambda$ [Å]")
        self.ax_band.set_xlim(*x)
        self.ax_band.set_ylim(0, 1)
        self.ax_band.set_yticks([])
        self.ax_band.spines[["left", "right", "top"]].set_visible(False)
        self.ax_band.tick_params(axis="y", left=False, labelleft=False)
        self.ax_band.format_coord = self.__format_band_coord

        self.ax_inst.clear()
        self.ax_inst.invert_xaxis()

        self._plot_instrument_background(inst_background)

        self.im = self.ax_inst.scatter(
            gamma, nu, c=lamda, marker="o", rasterized=True
        )

        self.ax_inst.set_aspect(1)
        self.ax_inst.minorticks_on()

        self.ax_inst.set_xlabel(r"$\gamma$")
        self.ax_inst.set_ylabel(r"$\nu$")

        fmt_str_form = FormatStrFormatter(r"$%d^\circ$")

        self.ax_inst.xaxis.set_major_formatter(fmt_str_form)
        self.ax_inst.yaxis.set_major_formatter(fmt_str_form)

        self.ax_inst.format_coord = self.__format_inst_coord

        if len(lamda) > 0:
            self.cb_inst = self.fig_inst.colorbar(
                self.im, ax=self.ax_inst, orientation="horizontal"
            )
            self.cb_inst.minorticks_on()
            self.cb_inst.ax.set_xlabel(r"$\lambda$ [Å]")

        self.fig_inst.canvas.mpl_connect(
            "button_press_event", self.on_press_inst
        )

        self.canvas_inst.draw_idle()

    def plot_instrument_alternate(
        self,
        inst_background,
        gamma_1,
        nu_1,
        lamda_1,
        gamma_2,
        nu_2,
        lamda_2,
    ):
        if self.cb_inst is not None:
            self.cb_inst.remove()
            self.cb_inst = None

        if self.cb_inst_alt is not None:
            self.cb_inst_alt.remove()
            self.cb_inst_alt = None

        x = self.get_wavelength()

        self.bragg_band = 0
        self.bragg_band_alt = 0
        self.ax_band.clear()
        self.ax_band.minorticks_on()
        self.ax_band.set_xlabel(r"$\lambda$ [Å]")
        self.ax_band.set_xlim(*x)
        self.ax_band.set_ylim(0, 1)
        self.ax_band.set_yticks([])
        self.ax_band.spines[["left", "right", "top"]].set_visible(False)
        self.ax_band.tick_params(axis="y", left=False, labelleft=False)
        self.ax_band.format_coord = self.__format_band_coord

        self.ax_inst.clear()
        self.ax_inst.invert_xaxis()

        self._plot_instrument_background(inst_background)

        self.im = self.ax_inst.scatter(
            gamma_1, nu_1, c=lamda_1, marker="o", cmap="GnBu", rasterized=True
        )

        self.im_alt = self.ax_inst.scatter(
            gamma_2, nu_2, c=lamda_2, marker="o", cmap="RdPu", rasterized=True
        )

        self.ax_inst.set_aspect(1)
        self.ax_inst.minorticks_on()

        self.ax_inst.set_xlabel(r"$\gamma$")
        self.ax_inst.set_ylabel(r"$\nu$")

        fmt_str_form = FormatStrFormatter(r"$%d^\circ$")

        self.ax_inst.xaxis.set_major_formatter(fmt_str_form)
        self.ax_inst.yaxis.set_major_formatter(fmt_str_form)

        self.ax_inst.format_coord = self.__format_inst_coord

        if len(lamda_2) > 0:
            self.cb_inst_alt = self.fig_inst.colorbar(
                self.im_alt, ax=self.ax_inst, orientation="horizontal"
            )
            self.cb_inst_alt.minorticks_on()

        if len(lamda_1) > 0:
            self.cb_inst = self.fig_inst.colorbar(
                self.im, ax=self.ax_inst, orientation="horizontal"
            )
            self.cb_inst.minorticks_on()

        if len(lamda_2) > 0:
            self.cb_inst_alt.ax.set_xlabel(r"$\lambda$ [Å]")
        elif len(lamda_1) > 0:
            self.cb_inst.ax.set_xlabel(r"$\lambda$ [Å]")

        self.fig_inst.canvas.mpl_connect(
            "button_press_event", self.on_press_inst
        )

        self.canvas_inst.draw_idle()

    def get_intersect(self):
        if self.intersect_line.hasAcceptableInput():
            return float(self.intersect_line.text())

    def set_intersect(self, val):
        self.intersect_line.setText(str(round(val, 2)))

    def get_intersect_alternate(self):
        if self.intersect_alt_line.hasAcceptableInput():
            if self.intersect_alt_line.text() != "":
                return float(self.intersect_alt_line.text())

    def set_intersect_alternate(self, val):
        value = str(round(val, 2)) if val is not None else ""
        self.intersect_alt_line.setText(value)

    def get_horizontal(self):
        if self.horizontal_line.hasAcceptableInput():
            return float(self.horizontal_line.text())

    def get_vertical(self):
        if self.vertical_line.hasAcceptableInput():
            return float(self.vertical_line.text())

    def get_d(self):
        if self.d_spacing_line.hasAcceptableInput():
            return float(self.d_spacing_line.text())

    def set_horizontal(self, val):
        self.horizontal_line.setText(str(round(val, 2)))

    def set_vertical(self, val):
        self.vertical_line.setText(str(round(val, 2)))

    def set_d(self, val):
        self.d_spacing_line.setText(str(round(val, 4)))

    def get_horizontal_alternate(self):
        if self.horizontal_alt_line.hasAcceptableInput():
            if self.horizontal_alt_line.text() != "":
                return float(self.horizontal_alt_line.text())

    def get_vertical_alternate(self):
        if self.vertical_alt_line.hasAcceptableInput():
            if self.vertical_alt_line.text() != "":
                return float(self.vertical_alt_line.text())

    def get_d_alternate(self):
        if self.d_spacing_alt_line.hasAcceptableInput():
            if self.d_spacing_alt_line.text() != "":
                return float(self.d_spacing_alt_line.text())

    def set_horizontal_alternate(self, val):
        value = str(round(val, 2)) if val is not None else ""
        self.horizontal_alt_line.setText(value)

    def set_vertical_alternate(self, val):
        value = str(round(val, 2)) if val is not None else ""
        self.vertical_alt_line.setText(value)

    def set_d_alternate(self, val):
        value = str(round(val, 4)) if val is not None else ""
        self.d_spacing_alt_line.setText(value)

    def set_angles(self, values):
        ang = "(" + ", ".join(np.array(values).astype(str)) + ")"

        self.angles_line.setText(ang)

    def set_comment(self, values):
        self.comment_line.setText(str(values))

    def get_comment(self):
        return self.comment_line.text()

    def get_angles(self):
        ang = self.angles_line.text()
        ang = ang.strip("(").strip(")").split(",")

        return [float(val) for val in ang if val != ""]

    def on_press_inst(self, event):
        if (
            event.inaxes == self.ax_inst
            and self.fig_inst.canvas.toolbar.mode == ""
        ):
            horz, vert = event.xdata, event.ydata

            self.roi_ready.emit(horz, vert)

    def plot_harmonics(self, hkls, lamdas):
        for line in self.ax_band.lines[:]:
            line.remove()
        for text in self.ax_band.texts:
            text.remove()

        for hkl, lamda in zip(hkls, lamdas):
            self.ax_band.axvline(x=lamda, color="C0", linestyle="--")
            if hkl is not None:
                self.ax_band.text(
                    lamda,
                    0.95,
                    f"({hkl[0]:.0f} {hkl[1]:.0f} {hkl[2]:.0f})",
                    rotation=90,
                    verticalalignment="top",
                    horizontalalignment="right",
                    transform=self.ax_band.get_xaxis_transform(),
                    color="k",
                )
                self.hkl = hkl
                self.scale = lamda
                self.hkl_alt = hkl
                self.scale_alt = lamda

        self.canvas_inst.draw_idle()

    def plot_harmonics_alternate(self, hkls, lamdas):
        for hkl, lamda in zip(hkls, lamdas):
            self.ax_band.axvline(x=lamda, color="C1", linestyle="--")
            if hkl is not None:
                self.ax_band.text(
                    lamda,
                    0.05,
                    f"({hkl[0]:.0f} {hkl[1]:.0f} {hkl[2]:.0f})",
                    rotation=90,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    transform=self.ax_band.get_xaxis_transform(),
                    color="k",
                )
                self.hkl_alt = hkl
                self.scale_alt = lamda

        self.canvas_inst.draw_idle()

    def update_inst(self):
        for line in self.ax_inst.lines[:]:
            line.remove()

        xmin, xmax = self.ax_inst.get_xlim()
        ymin, ymax = self.ax_inst.get_ylim()

        horz, vert = self.get_horizontal(), self.get_vertical()

        horz_alt = self.get_horizontal_alternate()
        vert_alt = self.get_vertical_alternate()

        for line in self.ax_band.lines[:]:
            line.remove()

        self.harm_ready.emit()

        self.bragg_band = self.__scale_inst(horz, vert)
        self.bragg_band_alt = self.__scale_inst(horz_alt, vert_alt)

        if horz_alt is None and vert_alt is None:
            self.ax_inst.axvline(x=horz, color="k", linestyle="--")
            self.ax_inst.axhline(y=vert, color="k", linestyle="--")
        else:

            horz_min = 0 if horz > horz_alt else (horz - xmin) / (xmax - xmin)
            horz_max = 1 if horz < horz_alt else (horz - xmin) / (xmax - xmin)

            vert_min = 0 if vert < vert_alt else (vert - ymin) / (ymax - ymin)
            vert_max = 1 if vert > vert_alt else (vert - ymin) / (ymax - ymin)

            horz_alt_min = (
                0 if horz_alt > horz else (horz_alt - xmin) / (xmax - xmin)
            )
            horz_alt_max = (
                1 if horz_alt < horz else (horz_alt - xmin) / (xmax - xmin)
            )

            vert_alt_min = (
                0 if vert_alt < vert else (vert_alt - ymin) / (ymax - ymin)
            )
            vert_alt_max = (
                1 if vert_alt > vert else (vert_alt - ymin) / (ymax - ymin)
            )

            self.ax_inst.axvline(
                x=horz, color="k", linestyle="--", ymin=vert_min, ymax=vert_max
            )
            self.ax_inst.axhline(
                y=vert, color="k", linestyle="--", xmin=horz_min, xmax=horz_max
            )

            self.ax_inst.axvline(
                x=horz_alt,
                color="k",
                linestyle=":",
                ymin=vert_alt_min,
                ymax=vert_alt_max,
            )
            self.ax_inst.axhline(
                y=vert_alt,
                color="k",
                linestyle=":",
                xmin=horz_alt_min,
                xmax=horz_alt_max,
            )

        self.canvas_inst.draw_idle()

    def connect_roi_ready(self, lookup):
        self.roi_ready.connect(lookup)

    def connect_harmonic_ready(self, lookup):
        self.harm_ready.connect(lookup)

    def use_equivalents(self):
        return self.equivalents_box.isChecked()

    def use_symmetry(self):
        return self.symmetry_box.isChecked()

    def draw_all(self):
        return self.combined_box.isChecked()

    def plot_laue(self, gamma_laue, nu_laue, gamma, nu, lamda, d):
        if self.cb_laue is not None:
            self.cb_laue.remove()
            self.cb_laue = None

        x = self.get_wavelength()

        self.bragg_harm = 0
        self.ax_harm.clear()
        self.ax_harm.minorticks_on()
        self.ax_harm.set_xlabel(r"$\lambda$ [Å]")
        self.ax_harm.set_xlim(*x)
        self.ax_harm.set_ylim(0, 1)
        self.ax_harm.set_yticks([])
        self.ax_harm.spines[["left", "right", "top"]].set_visible(False)
        self.ax_harm.tick_params(axis="y", left=False, labelleft=False)
        self.ax_harm.format_coord = self.__format_harm_coord

        self.ax_laue.clear()
        self.ax_laue.invert_xaxis()

        self.ax_laue.scatter(
            gamma_laue, nu_laue, color="lightgray", marker="o", rasterized=True
        )

        sort = np.argsort(d)

        self.im = self.ax_laue.scatter(
            gamma[sort],
            nu[sort],
            c=d[sort],
            marker="o",
            rasterized=True,
            cmap="turbo",
        )

        self.ax_laue.set_aspect(1)
        self.ax_laue.minorticks_on()

        self.ax_laue.set_xlabel(r"$\gamma$")
        self.ax_laue.set_ylabel(r"$\nu$")

        fmt_str_form = FormatStrFormatter(r"$%d^\circ$")

        self.ax_laue.xaxis.set_major_formatter(fmt_str_form)
        self.ax_laue.yaxis.set_major_formatter(fmt_str_form)

        self.ax_laue.format_coord = self.__format_inst_coord

        if len(lamda) > 0:
            self.cb_laue = self.fig_laue.colorbar(
                self.im, ax=self.ax_laue, orientation="horizontal"
            )
            self.cb_laue.minorticks_on()
            self.cb_laue.ax.set_xlabel(r"$d$ [Å]")

        self.fig_laue.canvas.mpl_connect(
            "button_press_event", self.on_press_laue
        )

        self.canvas_laue.draw_idle()

    def on_press_laue(self, event):
        if (
            event.inaxes == self.ax_laue
            and self.fig_laue.canvas.toolbar.mode == ""
        ):
            horz, vert = event.xdata, event.ydata
            self.sel_ready.emit(horz, vert)

    def update_laue(self, horz, vert, lamdas, hkl, lamda_0):
        for line in self.ax_laue.lines[:]:
            line.remove()

        self.ax_laue.axvline(x=horz, color="k", linestyle="--")
        self.ax_laue.axhline(y=vert, color="k", linestyle="--")

        for line in self.ax_harm.lines[:]:
            line.remove()
        for text in self.ax_harm.texts:
            text.remove()

        self.bragg_harm = self.__scale_inst(horz, vert)

        for lamda in lamdas:
            self.ax_harm.axvline(x=lamda, color="k", linestyle="--")
        self.hkl_laue = hkl
        self.scale_laue = lamda_0

        self.canvas_laue.draw_idle()

    def connect_selection_ready(self, lookup):
        self.sel_ready.connect(lookup)

    def highlight_peak(self, row):
        self.peaks_table.blockSignals(True)
        self.peaks_table.clearSelection()
        self.peaks_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        for r in range(self.peaks_table.rowCount()):
            if self.peaks_table.item(r, 0).data(Qt.UserRole) == row:
                self.peaks_table.selectRow(r)
        self.peaks_table.blockSignals(False)

    def get_peak(self):
        row = self.peaks_table.currentRow()
        return self.peaks_table.item(row, 0).data(Qt.UserRole)

    def get_projection_matrix(self):
        params = (
            self.U1_line,
            self.U2_line,
            self.U3_line,
            self.V1_line,
            self.V2_line,
            self.V3_line,
            self.W1_line,
            self.W2_line,
            self.W3_line,
        )
        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            params = [float(param.text()) for param in params]

            return params

    def get_slice_value(self):
        if self.slice_line.hasAcceptableInput():
            return float(self.slice_line.text())

    def get_slice_thickness(self):
        if self.slice_thickness_line.hasAcceptableInput():
            return float(self.slice_thickness_line.text())

    def get_slice(self):
        return self.slice_combo.currentText()

    def use_symmetry_mesh(self):
        return self.mesh_symmetry_box.isChecked()

    def update_slice(self, slice_dict):
        x = slice_dict["x"]
        y = slice_dict["y"]

        labels = slice_dict["labels"]
        title = slice_dict["title"]
        signal = slice_dict["signal"]

        self.z = slice_dict["z"]
        self.W = slice_dict["W"]

        T = slice_dict["transform"]
        aspect = slice_dict["aspect"]

        transform = Affine2D(T)

        self.T_inv = np.linalg.inv(T)

        self.ax_slice.remove()

        if self.cb_slice is not None:
            self.cb_slice.remove()
            self.cb_slice = None

        extreme_finder = ExtremeFinderSimple(20, 20)

        grid_locator1 = MaxNLocator(nbins=10)
        grid_locator2 = MaxNLocator(nbins=10)

        grid_locator1.set_params(integer=True)
        grid_locator2.set_params(integer=True)

        grid_helper = GridHelperCurveLinear(
            transform,
            extreme_finder=extreme_finder,
            grid_locator1=grid_locator1,
            grid_locator2=grid_locator2,
        )

        self.ax_slice = self.fig_slice.add_subplot(
            1, 1, 1, axes_class=Axes, grid_helper=grid_helper
        )

        self.ax_slice.set_aspect(aspect)

        trans = transform + self.ax_slice.transData

        im = self.ax_slice.pcolormesh(
            x,
            y,
            signal,
            vmin=1,
            cmap="turbo",
            shading="flat",
            transform=trans,
            rasterized=True,
        )

        self.ax_slice.set_xlabel(labels[0])
        self.ax_slice.set_ylabel(labels[1])

        self.im = im
        self.vmin, self.vmax = self.im.norm.vmin, self.im.norm.vmax

        self.ax_slice.set_title(title)
        self.ax_slice.grid(True)

        self.cb_slice = self.fig_slice.colorbar(self.im, ax=self.ax_slice)
        self.cb_slice.minorticks_on()
        self.cb_slice.ax.set_ylabel("Redundancy")

        self.canvas_slice.draw_idle()

        self.ax_slice.format_coord = self.__format_hkl_coord

    def __format_hkl_coord(self, x, y):
        x, y, _ = np.dot(self.T_inv, [x, y, 1])
        h, k, l = np.dot(self.W, [x, y, self.z])
        return "hkl = ({:.3f}, {:.3f}, {:.3f})".format(h, k, l)

    def __format_inst_coord(self, x, y):
        return "γ = {:.1f}°, ν = {:.1f}°".format(x, y)

    def __format_band_coord(self, x, _y):
        wl = "λ = {:.3f} Å, ".format(x)
        if self.bragg_band_alt is not None:
            if self.hkl is not None or self.hkl_alt is not None:
                hkl = self.hkl / x * self.scale
                hkl_alt = self.hkl_alt / x * self.scale_alt
                hkl1 = "hkl₁ = ({:.2g} {:.2g} {:.2g}), ".format(*hkl)
                hkl2 = "hkl₂ = ({:.2g} {:.2g} {:.2g}), ".format(*hkl_alt)
            else:
                hkl1 = ""
                hkl2 = ""
            d1 = "d₁ = {:.3f} Å, ".format(self.bragg_band * x)
            d2 = "d₂ = {:.3f} Å".format(self.bragg_band_alt * x)
            return wl + hkl1 + d1 + hkl2 + d2
        else:
            if self.hkl is not None:
                hkl = self.hkl / x * self.scale
                hkl = "hkl = ({:.2g} {:.2g} {:.2g}), ".format(*hkl)
            else:
                hkl = ""
            d = "d = {:.3f} Å".format(self.bragg_band * x)
            return wl + hkl + d

    def __format_harm_coord(self, x, _y):
        wl = "λ = {:.3f} Å, ".format(x)
        if self.hkl_laue is not None:
            hkl = self.hkl_laue / x * self.scale_laue
            hkl = "hkl = ({:.2g} {:.2g} {:.2g}), ".format(*hkl)
        else:
            hkl = ""
        d = "d = {:.3f} Å".format(self.bragg_harm * x)

        return wl + hkl + d

    def __scale_inst(self, horz, vert):
        if horz is not None or vert is not None:
            horz = np.deg2rad(horz)
            vert = np.deg2rad(vert)
            return 1 / np.sqrt(2 * (1 - np.cos(horz) * np.cos(vert)))
