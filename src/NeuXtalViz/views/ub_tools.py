from cProfile import label
import re

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib import colors as mcolors

from qtpy.QtWidgets import (
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QLabel,
    QComboBox,
    QPushButton,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QTabWidget,
    QFileDialog,
    QFrame,
    QSlider,
    QAbstractItemView,
    QMessageBox,
)

from qtpy.QtGui import (
    QDoubleValidator,
    QIntValidator,
    QRegularExpressionValidator,
)
from qtpy.QtCore import (
    Qt,
    Signal,
    QEvent,
    QRegularExpression,
    QItemSelectionModel,
    QTimer,
)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator as MplMaxNLocator

from mpl_toolkits.axisartist import Axes, GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import (
    ExtremeFinderSimple,
    MaxNLocator,
)

from NeuXtalViz.views.base_view import NeuXtalVizWidget

import qtawesome as qta

cmaps = {
    "Sequential": "viridis",
    "Binary": "binary",
    "Diverging": "bwr",
    "Rainbow": "turbo",
    "Modified": "modified",
}


class UBView(NeuXtalVizWidget):
    """
    View for UB matrix and peak indexing tools in NeuXtalViz.

    Provides user interface elements for UB calculation, peak finding,
    indexing, prediction, integration, and filtering. Tool tips are added
    to all major widgets and controls for improved usability.
    """

    roi_ready = Signal()
    scan_ready = Signal()
    index_ready = Signal()
    slice_ready = Signal(float, float, float)

    def __init__(self, parent=None):
        """
        Initialize the UB view and build all tool tabs.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None.
        """

        super().__init__(parent)

        self.tab_widget = QTabWidget(self)

        self._slice_smin = 0.0
        self._slice_smax = 1.0
        self._slice_step = 0.1
        self._slice_steps = 10

        self.parameters_tab()
        self.table_tab()
        self.verify_tab()
        self.modulation_tab()
        self.alignment_tab()

        self.layout().addWidget(self.tab_widget, stretch=1)

        self.last_highlight = None
        self._highlight_actor = None
        self._peak_centers = None
        self._inst_click_cid = None
        self._scan_click_cid = None
        self._slice_click_cid = None
        self._roi_lines = None
        self.slice_im = None
        self.inst_im = None
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
        self._last_slice_view = None
        self._last_slice_labels = None
        self._last_inst_view = None

    def parameters_tab(self):
        """
        Build the "Parameters" tab containing Q/peaks/UB I/O controls.

        Assembles the convert-to-Q, peaks, and UB sub-tabs along with
        save/load buttons and status labels, and adds the resulting tab
        to the main tab widget.
        """

        ub_peaks_tab = QWidget()
        self.tab_widget.addTab(ub_peaks_tab, "Parameters")

        ub_layout = QVBoxLayout()

        self.save_q_button = QPushButton("Save Q", self)
        self.save_q_button.setToolTip("Save the Q workspace to a file.")
        self.save_q_button.setIcon(qta.icon("fa6s.floppy-disk"))
        self.load_q_button = QPushButton("Load Q", self)
        self.load_q_button.setToolTip("Load a Q-sample workspace from a file.")
        self.load_q_button.setIcon(qta.icon("fa6s.folder-open"))

        self.save_peaks_button = QPushButton("Save Peaks", self)
        self.save_peaks_button.setToolTip("Save the peaks table to a file.")
        self.save_peaks_button.setIcon(qta.icon("fa6s.floppy-disk"))
        self.load_peaks_button = QPushButton("Load Peaks", self)
        self.load_peaks_button.setToolTip("Load a peaks table from a file.")
        self.load_peaks_button.setIcon(qta.icon("fa6s.folder-open"))

        self.save_ub_button = QPushButton("Save UB", self)
        self.save_ub_button.setToolTip("Save the UB matrix to a file.")
        self.save_ub_button.setIcon(qta.icon("fa6s.floppy-disk"))
        self.load_ub_button = QPushButton("Load UB", self)
        self.load_ub_button.setToolTip("Load a UB matrix from a file.")
        self.load_ub_button.setIcon(qta.icon("fa6s.folder-open"))

        self.q_label = QLabel("No Q-sample")
        self.peaks_label = QLabel("No peaks table")
        self.ub_label = QLabel("No UB matrix")

        self.q_label.setStyleSheet("color: red;")
        self.peaks_label.setStyleSheet("color: red;")
        self.ub_label.setStyleSheet("color: red;")

        convert_frame = QFrame()
        convert_frame.setFrameShape(QFrame.Shape.HLine)
        convert_frame.setFrameShadow(QFrame.Shadow.Sunken)

        peaks_frame = QFrame()
        peaks_frame.setFrameShape(QFrame.Shape.HLine)
        peaks_frame.setFrameShadow(QFrame.Shadow.Sunken)

        ubS_frame = QFrame()
        ubS_frame.setFrameShape(QFrame.Shape.HLine)
        ubS_frame.setFrameShadow(QFrame.Shadow.Sunken)

        convert_io_layout = QHBoxLayout()

        convert_io_layout.addWidget(self.q_label)
        convert_io_layout.addStretch(1)
        convert_io_layout.addWidget(self.save_q_button)
        convert_io_layout.addWidget(self.load_q_button)

        peaks_io_layout = QHBoxLayout()

        peaks_io_layout.addWidget(self.peaks_label)
        peaks_io_layout.addStretch(1)
        peaks_io_layout.addWidget(self.save_peaks_button)
        peaks_io_layout.addWidget(self.load_peaks_button)

        ub_io_layout = QHBoxLayout()

        ub_io_layout.addWidget(self.ub_label)
        ub_io_layout.addStretch(1)
        ub_io_layout.addWidget(self.save_ub_button)
        ub_io_layout.addWidget(self.load_ub_button)

        convert_tab = self.__init_convert_tab()
        self.peaks_tab = self.__init_peaks_tab()
        self.ub_tab = self.__init_ub_tab()
        values_tab = self.__init_values_tab()

        ub_layout.addWidget(convert_tab)
        ub_layout.addLayout(convert_io_layout)
        ub_layout.addWidget(convert_frame)

        ub_layout.addWidget(self.peaks_tab)
        ub_layout.addLayout(peaks_io_layout)
        ub_layout.addWidget(peaks_frame)

        ub_layout.addWidget(self.ub_tab)
        ub_layout.addLayout(ub_io_layout)
        ub_layout.addWidget(ubS_frame)

        ub_layout.addWidget(values_tab)

        ub_peaks_tab.setLayout(ub_layout)

    def __init_values_tab(self):
        """
        Build the sub-tab widget for lattice, orientation, and modulation values.

        Creates the "Lattice Parameters", "Sample Orientation", and
        "Modulation Parameters" sub-tabs, wiring up their line edits,
        validators, and set/search buttons.

        Returns
        -------
        values_tab : QTabWidget
            Tab widget containing the lattice parameters, sample
            orientation, and modulation parameters sub-tabs.
        """

        values_tab = QTabWidget()

        parameters_tab = QWidget()
        orientation_tab = QWidget()
        satellite_tab = QWidget()

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

        pattern = r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?(\(\d+\))?$$"
        regex = QRegularExpression(pattern)
        validator = QRegularExpressionValidator(regex)

        self.a_line.setValidator(validator)
        self.b_line.setValidator(validator)
        self.c_line.setValidator(validator)

        self.alpha_line.setValidator(validator)
        self.beta_line.setValidator(validator)
        self.gamma_line.setValidator(validator)

        a_label = QLabel("a:")
        b_label = QLabel("b:")
        c_label = QLabel("c:")

        alpha_label = QLabel("α:")
        beta_label = QLabel("β:")
        gamma_label = QLabel("γ:")

        angstrom_label = QLabel("Å")
        degree_label = QLabel("°")

        parameters_layout = QGridLayout()

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

        self.set_ub_button = QPushButton("Set UB", self)
        self.set_ub_button.setToolTip("Set the UB matrix.")
        self.set_ub_button.setIcon(qta.icon("fa6s.file-pen"))

        self.set_scattering_plane_ub_button = QPushButton(
            "Search U from Scattering Plane", self
        )
        self.set_scattering_plane_ub_button.setToolTip(
            "Calculate the UB matrix from the scattering plane."
        )
        self.set_scattering_plane_ub_button.setIcon(
            qta.icon("fa6s.paper-plane")
        )

        set_layout = QHBoxLayout()
        set_layout.addWidget(self.conventional_button)
        set_layout.addWidget(self.set_scattering_plane_ub_button)
        set_layout.addWidget(self.set_ub_button)
        set_layout.addStretch(1)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-5, 5, 4, notation=notation)

        self.dh1_line = QLineEdit("0.0")
        self.dk1_line = QLineEdit("0.0")
        self.dl1_line = QLineEdit("0.0")

        self.dh2_line = QLineEdit("0.0")
        self.dk2_line = QLineEdit("0.0")
        self.dl2_line = QLineEdit("0.0")

        self.dh3_line = QLineEdit("0.0")
        self.dk3_line = QLineEdit("0.0")
        self.dl3_line = QLineEdit("0.0")

        self.dh1_line.setValidator(validator)
        self.dk1_line.setValidator(validator)
        self.dl1_line.setValidator(validator)

        self.dh2_line.setValidator(validator)
        self.dk2_line.setValidator(validator)
        self.dl2_line.setValidator(validator)

        self.dh3_line.setValidator(validator)
        self.dk3_line.setValidator(validator)
        self.dl3_line.setValidator(validator)

        self.max_order_line = QLineEdit("0")

        mod_vec1_label = QLabel("1:")
        mod_vec2_label = QLabel("2:")
        mod_vec3_label = QLabel("3:")

        dh_label = QLabel("Δh")
        dk_label = QLabel("Δk")
        dl_label = QLabel("Δl")

        max_order_label = QLabel("Max Order")

        self.cross_box = QCheckBox("Cross Terms", self)
        self.cross_box.setChecked(False)

        satellite_layout = QGridLayout()

        satellite_layout.addWidget(dh_label, 0, 1, Qt.AlignCenter)
        satellite_layout.addWidget(dk_label, 0, 2, Qt.AlignCenter)
        satellite_layout.addWidget(dl_label, 0, 3, Qt.AlignCenter)
        satellite_layout.addWidget(max_order_label, 0, 4, Qt.AlignCenter)
        satellite_layout.addWidget(mod_vec1_label, 1, 0)
        satellite_layout.addWidget(self.dh1_line, 1, 1)
        satellite_layout.addWidget(self.dk1_line, 1, 2)
        satellite_layout.addWidget(self.dl1_line, 1, 3)
        satellite_layout.addWidget(self.max_order_line, 1, 4)
        satellite_layout.addWidget(mod_vec2_label, 2, 0)
        satellite_layout.addWidget(self.dh2_line, 2, 1)
        satellite_layout.addWidget(self.dk2_line, 2, 2)
        satellite_layout.addWidget(self.dl2_line, 2, 3)
        satellite_layout.addWidget(self.cross_box, 2, 4)
        satellite_layout.addWidget(mod_vec3_label, 3, 0)
        satellite_layout.addWidget(self.dh3_line, 3, 1)
        satellite_layout.addWidget(self.dk3_line, 3, 2)
        satellite_layout.addWidget(self.dl3_line, 3, 3)

        x_label = QLabel("x (In-Plane):")
        y_label = QLabel("y (Vertical):")
        z_label = QLabel("z (Beam):")

        a_star_label = QLabel("a*")
        b_star_label = QLabel("b*")
        c_star_label = QLabel("c*")

        self.uh_line = QLineEdit()
        self.uk_line = QLineEdit()
        self.ul_line = QLineEdit()

        self.vh_line = QLineEdit()
        self.vk_line = QLineEdit()
        self.vl_line = QLineEdit()

        self.wh_line = QLineEdit()
        self.wk_line = QLineEdit()
        self.wl_line = QLineEdit()

        self.uh_line.setPlaceholderText("h")
        self.uk_line.setPlaceholderText("k")
        self.ul_line.setPlaceholderText("l")
        self.vh_line.setPlaceholderText("h")
        self.vk_line.setPlaceholderText("k")
        self.vl_line.setPlaceholderText("l")
        self.wh_line.setPlaceholderText("h")
        self.wk_line.setPlaceholderText("k")
        self.wl_line.setPlaceholderText("l")

        self.uh_line.setReadOnly(False)
        self.uk_line.setReadOnly(False)
        self.ul_line.setReadOnly(False)

        self.vh_line.setReadOnly(False)
        self.vk_line.setReadOnly(False)
        self.vl_line.setReadOnly(False)

        self.wh_line.setReadOnly(False)
        self.wk_line.setReadOnly(False)
        self.wl_line.setReadOnly(False)

        self.wh_line.setEnabled(False)
        self.wk_line.setEnabled(False)
        self.wl_line.setEnabled(False)

        self.wh_line.setToolTip(
            "Vertical direction, derived from the in-plane and beam directions."
        )
        self.wk_line.setToolTip(
            "Vertical direction, derived from the in-plane and beam directions."
        )
        self.wl_line.setToolTip(
            "Vertical direction, derived from the in-plane and beam directions."
        )

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 190, 4, notation=notation)

        self.uh_line.setValidator(validator)
        self.uk_line.setValidator(validator)
        self.ul_line.setValidator(validator)

        self.vh_line.setValidator(validator)
        self.vk_line.setValidator(validator)
        self.vl_line.setValidator(validator)

        self.wh_line.setValidator(validator)
        self.wk_line.setValidator(validator)
        self.wl_line.setValidator(validator)

        orientation_layout = QGridLayout()

        orientation_layout.addWidget(a_star_label, 0, 1, Qt.AlignCenter)
        orientation_layout.addWidget(b_star_label, 0, 2, Qt.AlignCenter)
        orientation_layout.addWidget(c_star_label, 0, 3, Qt.AlignCenter)
        orientation_layout.addWidget(y_label, 1, 0)
        orientation_layout.addWidget(self.wh_line, 1, 1)
        orientation_layout.addWidget(self.wk_line, 1, 2)
        orientation_layout.addWidget(self.wl_line, 1, 3)
        orientation_layout.addWidget(z_label, 2, 0)
        orientation_layout.addWidget(self.uh_line, 2, 1)
        orientation_layout.addWidget(self.uk_line, 2, 2)
        orientation_layout.addWidget(self.ul_line, 2, 3)
        orientation_layout.addWidget(x_label, 3, 0)
        orientation_layout.addWidget(self.vh_line, 3, 1)
        orientation_layout.addWidget(self.vk_line, 3, 2)
        orientation_layout.addWidget(self.vl_line, 3, 3)

        lattice_layout = QVBoxLayout()
        directions_layout = QVBoxLayout()
        modulation_layout = QVBoxLayout()

        lattice_layout.addLayout(parameters_layout)
        lattice_layout.addStretch(1)
        lattice_layout.addLayout(set_layout)

        directions_layout.addLayout(orientation_layout)
        directions_layout.addStretch(1)

        modulation_layout.addLayout(satellite_layout)
        modulation_layout.addStretch(1)

        parameters_tab.setLayout(lattice_layout)
        orientation_tab.setLayout(directions_layout)
        satellite_tab.setLayout(modulation_layout)

        values_tab.addTab(parameters_tab, "Lattice Parameters")
        values_tab.addTab(orientation_tab, "Sample Orientation")
        values_tab.addTab(satellite_tab, "Modulation Parameters")

        return values_tab

    def __init_convert_tab(self):
        """
        Build the "Convert To Q" sub-tab.

        Creates the controls for instrument selection, IPTS/experiment/run
        entry, calibration file browsing, wavelength/d-spacing limits,
        and the convert-to-Q action buttons.

        Returns
        -------
        convert_tab : QTabWidget
            Tab widget containing the "Convert To Q" sub-tab.
        """

        convert_tab = QTabWidget()

        convert_to_q_tab = QWidget()
        convert_to_q_tab_layout = QVBoxLayout()

        experiment_params_layout = QHBoxLayout()
        wavelength_params_layout = QHBoxLayout()
        instrument_params_layout = QGridLayout()

        self.instrument_combo = QComboBox(self)
        self.instrument_combo.addItem("TOPAZ")
        self.instrument_combo.addItem("MANDI")
        self.instrument_combo.addItem("CORELLI")
        self.instrument_combo.addItem("SNAP")
        self.instrument_combo.addItem("WAND²")
        self.instrument_combo.addItem("DEMAND")
        self.instrument_combo.addItem("SXD")
        self.instrument_combo.addItem("WISH")
        self.instrument_combo.setToolTip(
            "Select the instrument for data conversion."
        )
        self.auto_scale_dropdown(self.instrument_combo)

        self.live_box = QCheckBox("Live", self)
        self.live_box.setEnabled(False)
        self.live_box.setToolTip(
            "Stream live event data (TOPAZ, CORELLI, MANDI only)."
        )

        ipts_label = QLabel("IPTS:")
        exp_label = QLabel("Experiment:")
        run_label = QLabel("Runs:")
        filter_time_label = QLabel("Time Stop [s]:")
        angstrom_label = QLabel("Å")

        exp_label.hide()

        validator = QIntValidator(1, 1000000000, self)

        self.runs_line = QLineEdit("")
        self.runs_line.setPlaceholderText("e.g. 12345,12346 or 12345:12400")

        self.ipts_line = QLineEdit("")
        self.ipts_line.setValidator(validator)
        self.ipts_line.setToolTip("Enter the IPTS number for the experiment.")
        self.ipts_line.setPlaceholderText("IPTS number")

        self.exp_line = QLineEdit("")
        self.exp_line.setValidator(validator)
        self.exp_line.setToolTip("Enter the experiment number.")
        self.exp_line.setPlaceholderText("Experiment number")
        self.exp_line.hide()

        self.cal_line = QLineEdit("")
        self.cal_line.setPlaceholderText("Detector calibration file")
        self.tube_line = QLineEdit("")
        self.tube_line.setPlaceholderText("Tube calibration file")

        self.wl_min_line = QLineEdit("0.3")
        self.wl_min_line.setToolTip("Minimum wavelength for conversion.")
        self.wl_max_line = QLineEdit("3.5")
        self.wl_max_line.setToolTip("Maximum wavelength for conversion.")

        wl_label = QLabel("λ:")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.2, 10, 5, notation=notation)

        self.wl_min_line.setValidator(validator)
        self.wl_max_line.setValidator(validator)

        d_min_label = QLabel("d(min):", self)

        self.convert_min_d_line = QLineEdit("0.7")
        self.convert_min_d_line.setValidator(validator)
        self.convert_min_d_line.setToolTip("Minimum d-spacing for conversion.")

        validator = QIntValidator(1, 1000, self)

        self.filter_time_line = QLineEdit("")
        self.filter_time_line.setValidator(validator)
        self.filter_time_line.setPlaceholderText("Max time (s)")
        self.filter_time_line.setToolTip(
            "Maximum time (s) for filtering events."
        )

        self.cal_browse_button = QPushButton("Detector", self)
        self.tube_browse_button = QPushButton("Tube", self)

        browse_icon = qta.icon("fa6s.folder-open")
        self.cal_browse_button.setIcon(browse_icon)
        self.tube_browse_button.setIcon(browse_icon)

        experiment_params_layout.addWidget(self.instrument_combo)
        experiment_params_layout.addWidget(self.live_box)
        experiment_params_layout.addWidget(ipts_label)
        experiment_params_layout.addWidget(self.ipts_line)

        experiment_params_layout.addWidget(run_label)
        experiment_params_layout.addWidget(self.runs_line)

        wavelength_params_layout.addWidget(wl_label)
        wavelength_params_layout.addWidget(self.wl_min_line)
        wavelength_params_layout.addWidget(self.wl_max_line)
        wavelength_params_layout.addWidget(angstrom_label)

        instrument_params_layout.addWidget(self.cal_line, 1, 0)
        instrument_params_layout.addWidget(self.cal_browse_button, 1, 1)
        instrument_params_layout.addWidget(self.tube_line, 2, 0)
        instrument_params_layout.addWidget(self.tube_browse_button, 2, 1)

        self.convert_to_q_button = QPushButton("Convert", self)
        self.convert_to_q_button.setToolTip("Convert raw data to Q workspace.")
        self.convert_to_q_button.setIcon(
            qta.icon("fa6s.arrow-right-arrow-left")
        )
        self.reload_convert_to_q_button = QPushButton("Convert + Reload", self)
        self.reload_convert_to_q_button.setToolTip(
            "Force reloading raw workspaces before converting to Q."
        )
        self.reload_convert_to_q_button.setIcon(qta.icon("fa6s.arrows-rotate"))

        self.lorentz_box = QCheckBox("Lorentz Correction", self)
        self.lorentz_box.setChecked(True)
        self.lorentz_box.setToolTip(
            "Apply Lorentz correction during conversion."
        )

        convert_to_q_action_layout = QHBoxLayout()
        convert_to_q_action_layout.addWidget(self.convert_to_q_button)
        convert_to_q_action_layout.addWidget(self.reload_convert_to_q_button)
        convert_to_q_action_layout.addWidget(self.lorentz_box)
        convert_to_q_action_layout.addWidget(filter_time_label)
        convert_to_q_action_layout.addWidget(self.filter_time_line)
        convert_to_q_action_layout.addStretch(1)
        convert_to_q_action_layout.addWidget(d_min_label)
        convert_to_q_action_layout.addWidget(self.convert_min_d_line)
        convert_to_q_action_layout.addLayout(wavelength_params_layout)

        convert_to_q_tab_layout.addLayout(experiment_params_layout)
        convert_to_q_tab_layout.addLayout(instrument_params_layout)
        convert_to_q_tab_layout.addStretch(1)
        convert_to_q_tab_layout.addLayout(convert_to_q_action_layout)

        convert_to_q_tab.setLayout(convert_to_q_tab_layout)

        convert_tab.addTab(convert_to_q_tab, "Convert To Q")

        return convert_tab

    def __init_peaks_tab(self):
        """
        Build the peaks processing sub-tab widget.

        Creates the "Find Peaks", "Index Peaks", "Predict Peaks",
        "Integrate Peaks", and "Filter Peaks" sub-tabs with their
        respective parameter controls and action buttons.

        Returns
        -------
        peaks_tab : QTabWidget
            Tab widget containing the peak finding, indexing,
            prediction, integration, and filtering sub-tabs.
        """

        peaks_tab = QTabWidget()

        find_tab = QWidget()
        find_tab_layout = QVBoxLayout()

        max_peaks_label = QLabel("Max Peaks:")
        min_distance_label = QLabel("Min Distance:")
        max_spacing_label = QLabel("Max Spacing:")
        peak_width_label = QLabel("Peak Width:")
        density_threshold_label = QLabel("Min Density:")
        find_edge_label = QLabel("Edge Pixels:")
        max_peaks_unit_label = QLabel("#")
        distance_unit_label = QLabel("Å⁻¹")
        peak_width_unit_label = QLabel("Å⁻¹")
        angstrom_unit_label = QLabel("Å")
        density_unit_label = QLabel("×")
        find_edge_unit_label = QLabel("#")
        self.aluminum_box = QCheckBox("Avoid Aluminum", self)
        self.aluminum_box.setChecked(True)
        self.copper_box = QCheckBox("Avoid Copper", self)
        self.copper_box.setChecked(False)
        self.iron_box = QCheckBox("Avoid Iron", self)
        self.iron_box.setChecked(False)

        validator = QIntValidator(10, 10000, self)

        self.max_peaks_line = QLineEdit("100")
        self.max_peaks_line.setValidator(validator)
        self.max_peaks_line.setToolTip("Maximum number of peaks to find.")

        validator = QIntValidator(1, 100000, self)

        self.density_threshold_line = QLineEdit("100")
        self.density_threshold_line.setValidator(validator)
        self.density_threshold_line.setToolTip(
            "Minimum density threshold for peak finding, as a "
            "multiplicative factor over the mean signal density."
        )

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.01, 10, 4, notation=notation)

        self.min_distance_line = QLineEdit("0.20")
        self.min_distance_line.setValidator(validator)
        self.min_distance_line.setToolTip(
            "Minimum distance between peaks (Å⁻¹)."
        )

        validator = QDoubleValidator(0.1, 100, 4, notation=notation)

        self.max_spacing_line = QLineEdit("31.46")
        self.max_spacing_line.setValidator(validator)
        self.max_spacing_line.setToolTip("Maximum d-spacing for peaks (Å).")

        validator = QDoubleValidator(0.001, 10, 4, notation=notation)

        self.peak_width_line = QLineEdit("0.1")
        self.peak_width_line.setValidator(validator)
        self.peak_width_line.setToolTip(
            "Shared half-width for removing aluminum, copper, and iron peaks (Å⁻¹)."
        )

        validator = QIntValidator(0, 64, self)

        self.find_edge_line = QLineEdit("0")
        self.find_edge_line.setValidator(validator)
        self.find_edge_line.setToolTip(
            "Number of edge pixels to exclude from peak finding."
        )

        find_params_layout = QGridLayout()

        find_params_layout.addWidget(max_peaks_label, 0, 0)
        find_params_layout.addWidget(self.max_peaks_line, 0, 1)
        find_params_layout.addWidget(max_peaks_unit_label, 0, 2)
        find_params_layout.addWidget(min_distance_label, 0, 3)
        find_params_layout.addWidget(self.min_distance_line, 0, 4)
        find_params_layout.addWidget(distance_unit_label, 0, 5)
        find_params_layout.addWidget(peak_width_label, 0, 6)
        find_params_layout.addWidget(self.peak_width_line, 0, 7)
        find_params_layout.addWidget(peak_width_unit_label, 0, 8)

        find_params_layout.addWidget(density_threshold_label, 1, 0)
        find_params_layout.addWidget(self.density_threshold_line, 1, 1)
        find_params_layout.addWidget(density_unit_label, 1, 2)
        find_params_layout.addWidget(max_spacing_label, 1, 3)
        find_params_layout.addWidget(self.max_spacing_line, 1, 4)
        find_params_layout.addWidget(angstrom_unit_label, 1, 5)
        find_params_layout.addWidget(find_edge_label, 1, 6)
        find_params_layout.addWidget(self.find_edge_line, 1, 7)
        find_params_layout.addWidget(find_edge_unit_label, 1, 8)

        self.find_button = QPushButton("Find", self)
        self.find_button.setIcon(qta.icon("fa6s.magnifying-glass"))
        self.find_button.setToolTip("Find peaks in the Q workspace.")

        find_action_layout = QHBoxLayout()
        find_action_layout.addWidget(self.find_button)
        find_action_layout.addWidget(self.aluminum_box)
        find_action_layout.addWidget(self.copper_box)
        find_action_layout.addWidget(self.iron_box)
        find_action_layout.addStretch(1)

        find_tab_layout.addLayout(find_params_layout)
        find_tab_layout.addStretch(1)
        find_tab_layout.addLayout(find_action_layout)

        find_tab.setLayout(find_tab_layout)

        index_tab = QWidget()
        index_tab_layout = QVBoxLayout()

        index_tolerance_label = QLabel("Tolerance:")
        index_tolerance_unit_label = QLabel("r.l.u.")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.01, 1, 5, notation=notation)

        self.index_sat_box = QCheckBox("Index Satellite", self)
        self.index_sat_box.setChecked(False)

        self.index_tolerance_line = QLineEdit("0.1")
        self.index_tolerance_line.setValidator(validator)
        self.index_tolerance_line.setToolTip("Tolerance for indexing peaks.")

        self.index_sat_tolerance_line = QLineEdit("0.1")
        self.index_sat_tolerance_line.setValidator(validator)
        self.index_sat_tolerance_line.setToolTip(
            "Tolerance for satellite peak indexing."
        )
        self.index_sat_tolerance_line.setEnabled(
            self.index_sat_box.isChecked()
        )
        self.index_sat_box.toggled.connect(
            self.index_sat_tolerance_line.setEnabled
        )

        index_params_layout = QGridLayout()

        index_params_layout.addWidget(index_tolerance_label, 0, 0)
        index_params_layout.addWidget(self.index_tolerance_line, 0, 1)
        index_params_layout.addWidget(self.index_sat_tolerance_line, 0, 2)
        index_params_layout.addWidget(index_tolerance_unit_label, 0, 3)
        index_params_layout.addWidget(self.index_sat_box, 0, 4)
        index_params_layout.setColumnStretch(5, 1)

        self.round_box = QCheckBox("Round hkl", self)
        self.round_box.setChecked(True)
        self.round_box.setToolTip("Round hkl values to nearest integer.")

        self.index_button = QPushButton("Index", self)
        self.index_button.setIcon(qta.icon("fa6s.list-ol"))
        self.index_button.setToolTip("Index the peaks using the current UB.")

        index_action_layout = QHBoxLayout()
        index_action_layout.addWidget(self.index_button)
        index_action_layout.addWidget(self.round_box)
        index_action_layout.addStretch(1)

        index_tab_layout.addLayout(index_params_layout)
        index_tab_layout.addStretch(1)
        index_tab_layout.addLayout(index_action_layout)

        index_tab.setLayout(index_tab_layout)

        centering_label = QLabel("Centering:")

        self.centering_combo = QComboBox(self)
        self.centering_combo.addItem("P")
        self.centering_combo.addItem("I")
        self.centering_combo.addItem("F")
        self.centering_combo.addItem("R(obv)")
        self.centering_combo.addItem("R(rev)")
        self.centering_combo.addItem("A")
        self.centering_combo.addItem("B")
        self.centering_combo.addItem("C")
        self.centering_combo.addItem("H")
        self.centering_combo.setToolTip(
            "Select lattice centering for peak prediction."
        )
        self.auto_scale_dropdown(self.centering_combo)

        min_d_unit_label = QLabel("Å")
        predict_edge_unit_label = QLabel("#")

        min_d_label = QLabel("Min d-spacing:")
        predict_edge_label = QLabel("Edge Pixels:")

        self.predict_sat_box = QCheckBox("Predict Satellite", self)
        self.predict_sat_box.setChecked(False)
        self.predict_sat_box.setToolTip("Enable satellite peak prediction.")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.4, 100, 3, notation=notation)

        self.min_d_line = QLineEdit("0.7")
        self.min_d_line.setValidator(validator)
        self.min_d_line.setToolTip("Minimum d-spacing for peak prediction.")

        self.min_sat_d_line = QLineEdit("1.0")
        self.min_sat_d_line.setValidator(validator)
        self.min_sat_d_line.setToolTip(
            "Minimum d-spacing for satellite peaks."
        )
        self.min_sat_d_line.setEnabled(self.predict_sat_box.isChecked())
        self.predict_sat_box.toggled.connect(self.min_sat_d_line.setEnabled)

        validator = QIntValidator(0, 64, self)

        self.predict_edge_line = QLineEdit("0")
        self.predict_edge_line.setValidator(validator)
        self.predict_edge_line.setToolTip(
            "Number of edge pixels to exclude from prediction."
        )

        predict_tab = QWidget()
        predict_tab_layout = QVBoxLayout()

        predict_params_layout = QGridLayout()

        predict_params_layout.addWidget(centering_label, 0, 0)
        predict_params_layout.addWidget(self.centering_combo, 0, 1)
        predict_params_layout.addWidget(self.predict_sat_box, 0, 2)
        predict_params_layout.addWidget(min_d_label, 1, 0)
        predict_params_layout.addWidget(self.min_d_line, 1, 1)
        predict_params_layout.addWidget(self.min_sat_d_line, 1, 2)
        predict_params_layout.addWidget(min_d_unit_label, 1, 3)
        predict_params_layout.addWidget(predict_edge_label, 1, 4)
        predict_params_layout.addWidget(self.predict_edge_line, 1, 5)
        predict_params_layout.addWidget(predict_edge_unit_label, 1, 6)

        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setIcon(qta.icon("fa6s.bullseye"))
        self.predict_button.setToolTip(
            "Predict peak positions based on UB and centering."
        )

        predict_action_layout = QHBoxLayout()
        predict_action_layout.addWidget(self.predict_button)
        predict_action_layout.addStretch(1)

        predict_tab_layout.addLayout(predict_params_layout)
        predict_tab_layout.addStretch(1)
        predict_tab_layout.addLayout(predict_action_layout)

        predict_tab.setLayout(predict_tab_layout)

        self.centroid_box = QCheckBox("Centroid", self)
        self.centroid_box.setChecked(True)
        self.centroid_box.setToolTip("Use centroid for peak integration.")

        self.adaptive_box = QCheckBox("Adaptive Envelope", self)
        self.adaptive_box.setChecked(True)
        self.adaptive_box.setToolTip(
            "Enable adaptive envelope for integration."
        )

        radius_label = QLabel("Radius:")
        inner_label = QLabel("Inner Factor:")
        outer_label = QLabel("Outer Factor:")
        radius_unit_label = QLabel("Å⁻¹")
        inner_unit_label = QLabel("×")
        outer_unit_label = QLabel("×")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0, 1, 3, notation=notation)

        self.radius_line = QLineEdit("0.25")
        self.radius_line.setValidator(validator)
        self.radius_line.setToolTip("Peak integration radius (Å⁻¹).")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(1, 3, 3, notation=notation)

        self.inner_line = QLineEdit("1.5")
        self.inner_line.setValidator(validator)
        self.inner_line.setToolTip("Inner factor for background shell.")

        self.outer_line = QLineEdit("2")
        self.outer_line.setValidator(validator)
        self.outer_line.setToolTip("Outer factor for background shell.")

        integrate_tab = QWidget()
        integrate_tab_layout = QVBoxLayout()

        integrate_params_layout = QGridLayout()

        integrate_params_layout.addWidget(radius_label, 0, 0)
        integrate_params_layout.addWidget(self.radius_line, 0, 1)
        integrate_params_layout.addWidget(radius_unit_label, 0, 2)
        integrate_params_layout.addWidget(inner_label, 0, 3)
        integrate_params_layout.addWidget(self.inner_line, 0, 4)
        integrate_params_layout.addWidget(inner_unit_label, 0, 5)
        integrate_params_layout.addWidget(outer_label, 0, 6)
        integrate_params_layout.addWidget(self.outer_line, 0, 7)
        integrate_params_layout.addWidget(outer_unit_label, 0, 8)

        self.integrate_button = QPushButton("Integrate", self)
        self.integrate_button.setIcon(qta.icon("fa6s.chart-area"))
        self.integrate_button.setToolTip("Integrate peak intensities.")

        integrate_action_layout = QHBoxLayout()
        integrate_action_layout.addWidget(self.integrate_button)
        integrate_action_layout.addWidget(self.centroid_box)
        integrate_action_layout.addWidget(self.adaptive_box)
        integrate_action_layout.addStretch(1)

        integrate_tab_layout.addLayout(integrate_params_layout)
        integrate_tab_layout.addStretch(1)
        integrate_tab_layout.addLayout(integrate_action_layout)

        integrate_tab.setLayout(integrate_tab_layout)

        self.filter_combo = QComboBox(self)
        self.filter_combo.addItem("I/σ")
        self.filter_combo.addItem("d")
        self.filter_combo.addItem("λ")
        self.filter_combo.addItem("Q")
        self.filter_combo.addItem("h^2+k^2+l^2")
        self.filter_combo.addItem("m^2+n^2+p^2")
        self.filter_combo.addItem("Run #")

        self.auto_scale_dropdown(self.filter_combo)

        self.filter_description_label = QLabel(self)
        self.filter_description_label.setToolTip(
            "Describes the currently selected peak filter variable."
        )
        self.filter_description_label.setWordWrap(False)
        self.filter_combo.currentTextChanged.connect(
            self.update_filter_description_label
        )
        self.update_filter_description_label(self.filter_combo.currentText())

        self.comparison_combo = QComboBox(self)
        self.comparison_combo.addItem(">")
        self.comparison_combo.addItem("<")
        self.comparison_combo.addItem(">=")
        self.comparison_combo.addItem("<=")
        self.comparison_combo.addItem("=")
        self.comparison_combo.addItem("!=")

        self.auto_scale_dropdown(self.comparison_combo)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-1e6, 1e6, 3, notation=notation)

        self.filter_line = QLineEdit("0")
        self.filter_line.setValidator(validator)
        self.filter_line.setToolTip("Value for filtering peaks.")

        filter_tab = QWidget()
        filter_tab_layout = QVBoxLayout()

        filter_params_layout = QHBoxLayout()

        filter_params_layout.addWidget(self.filter_combo)
        filter_params_layout.addWidget(self.comparison_combo)
        filter_params_layout.addWidget(self.filter_line)
        filter_params_layout.addStretch(1)
        filter_params_layout.addWidget(self.filter_description_label)

        self.filter_button = QPushButton("Filter", self)
        self.filter_button.setIcon(qta.icon("fa6s.filter"))
        self.filter_button.setToolTip("Apply filter to peaks table.")

        self.undo_filter_button = QPushButton("Undo", self)
        self.undo_filter_button.setIcon(qta.icon("fa6s.rotate-left"))
        self.undo_filter_button.setToolTip(
            "Restore the peaks table from the state before the last filter."
        )
        self.undo_filter_button.setEnabled(False)

        filter_action_layout = QHBoxLayout()
        filter_action_layout.addWidget(self.filter_button)
        filter_action_layout.addWidget(self.undo_filter_button)
        filter_action_layout.addStretch(1)

        filter_tab_layout.addLayout(filter_params_layout)
        filter_tab_layout.addStretch(1)
        filter_tab_layout.addLayout(filter_action_layout)

        filter_tab.setLayout(filter_tab_layout)

        peaks_tab.addTab(find_tab, "Find Peaks")
        peaks_tab.addTab(index_tab, "Index Peaks")
        peaks_tab.addTab(predict_tab, "Predict Peaks")
        peaks_tab.addTab(integrate_tab, "Integrate Peaks")
        peaks_tab.addTab(filter_tab, "Filter Peaks")

        return peaks_tab

    def __init_ub_tab(self):
        """
        Build the UB matrix sub-tab widget.

        Creates the "Calculate UB", "Transform UB", and "Refine UB"
        sub-tabs, including the cell search table, lattice
        transformation matrix inputs, and refinement options.

        Returns
        -------
        ub_tab : QTabWidget
            Tab widget containing the calculate, transform, and refine
            UB sub-tabs.
        """

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.01, 1, 5, notation=notation)

        ub_tab = QTabWidget()

        calculate_tolerance_label = QLabel("Tolerance:")
        calculate_tolerance_unit_label = QLabel("r.l.u.")

        self.calculate_tolerance_line = QLineEdit("0.1")
        self.calculate_tolerance_line.setValidator(validator)
        self.calculate_tolerance_line.setToolTip(
            "Tolerance for UB calculation."
        )

        max_scalar_error_label = QLabel("Max Scalar Error:")

        self.max_scalar_error_line = QLineEdit("0.2")
        self.max_scalar_error_line.setValidator(validator)
        self.max_scalar_error_line.setToolTip(
            "Maximum scalar error for cell search."
        )

        calculate_tab = QWidget()
        calculate_tab_layout = QVBoxLayout()

        calculate_params_layout = QHBoxLayout()

        calculate_params_layout.addWidget(calculate_tolerance_label)
        calculate_params_layout.addWidget(self.calculate_tolerance_line)
        calculate_params_layout.addWidget(calculate_tolerance_unit_label)
        calculate_params_layout.addStretch(1)
        calculate_params_layout.addWidget(max_scalar_error_label)
        calculate_params_layout.addWidget(self.max_scalar_error_line)

        self.conventional_button = QPushButton(
            "Find UB with Lattice Parameters", self
        )
        self.conventional_button.setToolTip("Calculate UB from cell guess.")
        self.conventional_button.setIcon(qta.icon("fa6s.ruler"))

        self.niggli_button = QPushButton("Find Primitive Cell", self)
        self.niggli_button.setToolTip("Calculate primitive (Niggli) cell.")
        self.niggli_button.setIcon(qta.icon("fa6s.bone"))
        self.select_button = QPushButton("Select Conventional Cell", self)
        self.select_button.setToolTip("Select the highlighted cell.")
        self.select_button.setIcon(qta.icon("fa6s.square-check"))

        self.form_line = QLineEdit("")
        self.form_line.setReadOnly(True)
        self.form_line.setToolTip("Form number of the selected cell.")

        form_label = QLabel("Form:")

        min_const_label = QLabel("Min(a,b,c) [Å]:")
        max_const_label = QLabel("Max(a,b,c) [Å]:")

        self.min_const_line = QLineEdit("5")
        self.max_const_line = QLineEdit("15")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.1, 1000, 4, notation=notation)

        self.min_const_line.setValidator(validator)
        self.max_const_line.setValidator(validator)
        self.min_const_line.setToolTip(
            "Minimum lattice constant for cell search."
        )
        self.max_const_line.setToolTip(
            "Maximum lattice constant for cell search."
        )

        const_layout = QHBoxLayout()
        const_layout.addWidget(min_const_label)
        const_layout.addWidget(self.min_const_line)
        const_layout.addWidget(max_const_label)
        const_layout.addWidget(self.max_const_line)

        calculate_action_layout = QHBoxLayout()
        calculate_action_layout.addLayout(const_layout)
        calculate_action_layout.addWidget(self.niggli_button)
        calculate_action_layout.addStretch(1)
        calculate_action_layout.addWidget(form_label)
        calculate_action_layout.addWidget(self.form_line)
        calculate_action_layout.addWidget(self.select_button)

        stretch = QHeaderView.Stretch

        self.cell_table = QTableWidget()
        self.cell_table.setRowCount(0)
        self.cell_table.setColumnCount(9)

        header = ["Error", "Bravais", "a", "b", "c", "α", "β", "γ", "V"]

        self.cell_table.horizontalHeader().setSectionResizeMode(stretch)
        self.cell_table.setHorizontalHeaderLabels(header)
        self.cell_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cell_table.setSelectionBehavior(QTableWidget.SelectRows)
        # self.cell_table.setSortingEnabled(True)

        calculate_tab_layout.addLayout(calculate_params_layout)
        calculate_tab_layout.addWidget(self.cell_table)
        calculate_tab_layout.addLayout(calculate_action_layout)

        calculate_tab.setLayout(calculate_tab_layout)

        transform_tolerance_label = QLabel("Tolerance:")
        transform_tolerance_unit_label = QLabel("r.l.u.")

        self.transform_tolerance_line = QLineEdit("0.1")
        self.transform_tolerance_line.setValidator(validator)
        self.transform_tolerance_line.setToolTip(
            "Tolerance for lattice transformation."
        )

        self.lattice_combo = QComboBox(self)
        self.lattice_combo.addItem("Triclinic")
        self.lattice_combo.addItem("Monoclinic")
        self.lattice_combo.addItem("Orthorhombic")
        self.lattice_combo.addItem("Tetragonal")
        self.lattice_combo.addItem("Rhombohedral")
        self.lattice_combo.addItem("Hexagonal")
        self.lattice_combo.addItem("Cubic")
        self.lattice_combo.setToolTip(
            "Select lattice system for transformation."
        )

        self.symmetry_combo = QComboBox(self)
        self.symmetry_combo.addItem("x,y,z")
        self.symmetry_combo.addItem("-x,-y,-z")

        self.auto_scale_dropdown(self.lattice_combo)
        self.auto_scale_dropdown(self.symmetry_combo)
        self.symmetry_combo.setToolTip(
            "Select symmetry operation for transformation."
        )

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-10, 10, 5, notation=notation)

        self.T11_line = QLineEdit("1")
        self.T12_line = QLineEdit("0")
        self.T13_line = QLineEdit("0")

        self.T21_line = QLineEdit("0")
        self.T22_line = QLineEdit("1")
        self.T23_line = QLineEdit("0")

        self.T31_line = QLineEdit("0")
        self.T32_line = QLineEdit("0")
        self.T33_line = QLineEdit("1")

        self.T11_line.setValidator(validator)
        self.T12_line.setValidator(validator)
        self.T13_line.setValidator(validator)

        self.T21_line.setValidator(validator)
        self.T22_line.setValidator(validator)
        self.T23_line.setValidator(validator)

        self.T31_line.setValidator(validator)
        self.T32_line.setValidator(validator)
        self.T33_line.setValidator(validator)

        hp_label = QLabel("h′:")
        kp_label = QLabel("k′:")
        lp_label = QLabel("l′:")

        h_label = QLabel("h")
        k_label = QLabel("k")
        l_label = QLabel("l")

        transform_tab = QWidget()
        transform_tab_layout = QVBoxLayout()

        transform_params_layout = QHBoxLayout()

        transform_params_layout.addWidget(transform_tolerance_label)
        transform_params_layout.addWidget(self.transform_tolerance_line)
        transform_params_layout.addWidget(transform_tolerance_unit_label)
        transform_params_layout.addWidget(self.lattice_combo)
        transform_params_layout.addWidget(self.symmetry_combo)

        transform_matrix_layout = QGridLayout()

        transform_matrix_layout.addWidget(h_label, 0, 1, Qt.AlignCenter)
        transform_matrix_layout.addWidget(k_label, 0, 2, Qt.AlignCenter)
        transform_matrix_layout.addWidget(l_label, 0, 3, Qt.AlignCenter)
        transform_matrix_layout.addWidget(hp_label, 1, 0)
        transform_matrix_layout.addWidget(self.T11_line, 1, 1)
        transform_matrix_layout.addWidget(self.T12_line, 1, 2)
        transform_matrix_layout.addWidget(self.T13_line, 1, 3)
        transform_matrix_layout.addWidget(kp_label, 2, 0)
        transform_matrix_layout.addWidget(self.T21_line, 2, 1)
        transform_matrix_layout.addWidget(self.T22_line, 2, 2)
        transform_matrix_layout.addWidget(self.T23_line, 2, 3)
        transform_matrix_layout.addWidget(lp_label, 3, 0)
        transform_matrix_layout.addWidget(self.T31_line, 3, 1)
        transform_matrix_layout.addWidget(self.T32_line, 3, 2)
        transform_matrix_layout.addWidget(self.T33_line, 3, 3)

        self.transform_button = QPushButton("Transform", self)
        self.transform_button.setIcon(qta.icon("fa6s.arrow-right-arrow-left"))
        self.transform_button.setToolTip("Apply the lattice transformation.")

        transform_action_layout = QHBoxLayout()
        transform_action_layout.addWidget(self.transform_button)
        transform_action_layout.addStretch(1)
        transform_action_layout.addLayout(transform_params_layout)

        transform_tab_layout.addLayout(transform_matrix_layout)
        transform_tab_layout.addStretch(1)
        transform_tab_layout.addLayout(transform_action_layout)

        transform_tab.setLayout(transform_tab_layout)

        refine_tolerance_label = QLabel("Tolerance:")
        refine_tolerance_unit_label = QLabel("r.l.u.")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.01, 1, 5, notation=notation)

        self.refine_tolerance_line = QLineEdit("0.1")
        self.refine_tolerance_line.setValidator(validator)
        self.refine_tolerance_line.setToolTip("Tolerance for UB refinement.")

        self.optimize_combo = QComboBox(self)
        self.optimize_combo.addItem("Unconstrained")
        self.optimize_combo.addItem("Constrained")
        self.optimize_combo.addItem("Triclinic")
        self.optimize_combo.addItem("Monoclinic")
        self.optimize_combo.addItem("Orthorhombic")
        self.optimize_combo.addItem("Tetragonal")
        self.optimize_combo.addItem("Rhombohedral")
        self.optimize_combo.addItem("Hexagonal")
        self.optimize_combo.addItem("Cubic")
        self.optimize_combo.setToolTip(
            "Select refinement constraints or lattice system."
        )
        self.auto_scale_dropdown(self.optimize_combo)

        self.refine_constraint_label = QLabel(self)
        self.refine_constraint_label.setToolTip(
            "Displays the lattice constraints applied by the selected refinement option."
        )
        self.refine_constraint_label.setWordWrap(False)

        self.optimize_combo.currentTextChanged.connect(
            self.update_refine_constraint_label
        )
        self.update_refine_constraint_label(self.optimize_combo.currentText())

        refine_tab = QWidget()
        refine_tab_layout = QVBoxLayout()

        refine_params_layout = QHBoxLayout()
        refine_params_layout.addWidget(refine_tolerance_label)
        refine_params_layout.addWidget(self.refine_tolerance_line)
        refine_params_layout.addWidget(refine_tolerance_unit_label)
        refine_params_layout.addWidget(self.optimize_combo)
        refine_params_layout.addStretch(1)
        refine_params_layout.addWidget(self.refine_constraint_label)

        self.refine_button = QPushButton("Refine", self)
        self.refine_button.setIcon(qta.icon("fa6s.wand-magic-sparkles"))
        self.refine_button.setToolTip("Refine the UB matrix.")

        refine_action_layout = QHBoxLayout()
        refine_action_layout.addWidget(self.refine_button)
        refine_action_layout.addStretch(1)

        refine_tab_layout.addLayout(refine_params_layout)
        refine_tab_layout.addStretch(1)
        refine_tab_layout.addLayout(refine_action_layout)

        refine_tab.setLayout(refine_tab_layout)

        ub_tab.addTab(calculate_tab, "Calculate UB")
        ub_tab.addTab(transform_tab, "Transform UB")
        ub_tab.addTab(refine_tab, "Refine UB")

        return ub_tab

    def table_tab(self):
        """
        Build the "Peaks" tab.

        Creates the d-spacing/angle calculator between two peaks, the
        peaks table, and the extended peak-information panel (hkl
        indices, satellite indices, run/bank/row/col, intensity, and
        goniometer angles), then adds the tab to the main tab widget.
        """

        peaks_table_tab = QWidget()
        self._peaks_tab = peaks_table_tab
        self.tab_widget.addTab(peaks_table_tab, "Peaks")

        peaks_layout = QVBoxLayout()

        calculator_layout = QGridLayout()

        h_label = QLabel("h", self)
        k_label = QLabel("k", self)
        l_label = QLabel("l", self)

        peak_1_label = QLabel("1:", self)
        peak_2_label = QLabel("2:", self)
        highlight_1_label = QLabel("1:", self)
        highlight_2_label = QLabel("2:", self)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.h1_line = QLineEdit()
        self.k1_line = QLineEdit()
        self.l1_line = QLineEdit()

        self.h2_line = QLineEdit()
        self.k2_line = QLineEdit()
        self.l2_line = QLineEdit()

        self.h1_line.setValidator(validator)
        self.k1_line.setValidator(validator)
        self.l1_line.setValidator(validator)

        self.h2_line.setValidator(validator)
        self.k2_line.setValidator(validator)
        self.l2_line.setValidator(validator)

        self.h1_line.setPlaceholderText("h")
        self.k1_line.setPlaceholderText("k")
        self.l1_line.setPlaceholderText("l")
        self.h2_line.setPlaceholderText("h")
        self.k2_line.setPlaceholderText("k")
        self.l2_line.setPlaceholderText("l")

        self.highlight_1_line = QLineEdit()
        self.highlight_2_line = QLineEdit()

        self.highlight_1_line.setEnabled(False)
        self.highlight_2_line.setEnabled(False)

        self.highlight_1_line.setToolTip(
            "Index of the first highlighted peak."
        )
        self.highlight_2_line.setToolTip(
            "Index of the second highlighted peak."
        )

        d_label = QLabel("d [Å]", self)

        phi_label = QLabel("φ [°]", self)

        self.d1_line = QLineEdit()
        self.d2_line = QLineEdit()
        self.phi_line = QLineEdit()

        self.d1_line.setReadOnly(True)
        self.d2_line.setReadOnly(True)
        self.phi_line.setReadOnly(True)

        self.highlight_d1_line = QLineEdit()
        self.highlight_d2_line = QLineEdit()
        self.highlight_phi_line = QLineEdit()

        self.highlight_d1_line.setReadOnly(True)
        self.highlight_d2_line.setReadOnly(True)
        self.highlight_phi_line.setReadOnly(True)

        self.calculate_button = QPushButton("Calculate", self)
        self.calculate_button.setToolTip(
            "Calculate d-spacings and angle between peaks."
        )
        self.calculate_button.setIcon(qta.icon("fa6s.calculator"))

        self.highlight_1_button = QPushButton("Add Highlighted", self)
        self.highlight_1_button.setIcon(qta.icon("fa6s.square-plus"))
        self.highlight_2_button = QPushButton("Add Highlighted", self)
        self.highlight_2_button.setIcon(qta.icon("fa6s.square-plus"))

        self.calculate_highlight_button = QPushButton("Calculate", self)
        self.calculate_highlight_button.setToolTip(
            "Calculate the highlighted peaks."
        )
        self.calculate_highlight_button.setIcon(qta.icon("fa6s.calculator"))

        self.delete_peak_button = QPushButton("Delete Highlighted", self)
        self.delete_peak_button.setToolTip(
            "Delete the currently highlighted peak from the peaks table."
        )
        self.delete_peak_button.setIcon(qta.icon("fa6s.trash"))

        calculator_layout.addWidget(h_label, 0, 1, Qt.AlignCenter)
        calculator_layout.addWidget(k_label, 0, 2, Qt.AlignCenter)
        calculator_layout.addWidget(l_label, 0, 3, Qt.AlignCenter)
        calculator_layout.addWidget(d_label, 0, 4, Qt.AlignCenter)
        calculator_layout.addWidget(phi_label, 0, 5, Qt.AlignCenter)

        calculator_layout.addWidget(peak_1_label, 1, 0)
        calculator_layout.addWidget(self.h1_line, 1, 1)
        calculator_layout.addWidget(self.k1_line, 1, 2)
        calculator_layout.addWidget(self.l1_line, 1, 3)
        calculator_layout.addWidget(self.d1_line, 1, 4)
        calculator_layout.addWidget(self.phi_line, 1, 5)

        calculator_layout.addWidget(peak_2_label, 2, 0)
        calculator_layout.addWidget(self.h2_line, 2, 1)
        calculator_layout.addWidget(self.k2_line, 2, 2)
        calculator_layout.addWidget(self.l2_line, 2, 3)
        calculator_layout.addWidget(self.d2_line, 2, 4)
        calculator_layout.addWidget(self.calculate_button, 2, 5)

        Q1_label = QLabel("Q₁(x,y,z) [Å⁻¹]:", self)
        Q2_label = QLabel("Q₂(x,y,z) [Å⁻¹]:", self)

        highlight_1_layout = QHBoxLayout()
        highlight_2_layout = QHBoxLayout()

        highlight_1_layout.addWidget(self.highlight_1_button)
        highlight_1_layout.addWidget(Q1_label)
        highlight_1_layout.addWidget(self.highlight_1_line)

        highlight_2_layout.addWidget(self.highlight_2_button)
        highlight_2_layout.addWidget(Q2_label)
        highlight_2_layout.addWidget(self.highlight_2_line)

        calculator_layout.addWidget(highlight_1_label, 3, 0)
        calculator_layout.addLayout(highlight_1_layout, 3, 1, 1, 3)
        calculator_layout.addWidget(self.highlight_d1_line, 3, 4)
        calculator_layout.addWidget(self.highlight_phi_line, 3, 5)

        calculator_layout.addWidget(highlight_2_label, 4, 0)
        calculator_layout.addLayout(highlight_2_layout, 4, 1, 1, 3)
        calculator_layout.addWidget(self.highlight_d2_line, 4, 4)
        calculator_layout.addWidget(self.calculate_highlight_button, 4, 5)

        stretch = QHeaderView.Stretch

        self.peaks_table = QTableWidget()
        self.peaks_table.setRowCount(0)
        self.peaks_table.setColumnCount(8)

        header = ["h", "k", "l", "d", "λ", "I", "I/σ", "#"]

        self.peaks_table.horizontalHeader().setSectionResizeMode(stretch)
        self.peaks_table.setHorizontalHeaderLabels(header)
        self.peaks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.peaks_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.peaks_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.peaks_table.setSortingEnabled(True)
        self.peaks_table.setToolTip(
            "Table of indexed peaks and their properties."
        )

        extended_info = QGridLayout()

        d_label = QLabel("d [Å]:", self)
        lambda_label = QLabel("λ [Å]:", self)
        self.gonio_label = QLabel("ω,χ,φ [°]:", self)
        two_theta_label = QLabel("2θ [°]:", self)
        horz_vert_label = QLabel("γ,ν [°]:", self)
        run_label = QLabel("Run #", self)
        bank_label = QLabel("Bank #", self)
        row_label = QLabel("Row #", self)
        col_label = QLabel("Col #", self)

        self.d_line = QLineEdit()
        self.lambda_line = QLineEdit()
        self.gonio_line = QLineEdit()
        self.two_theta_line = QLineEdit()
        self.horz_vert_line = QLineEdit()
        self.run_line = QLineEdit()
        self.bank_line = QLineEdit()
        self.row_line = QLineEdit()
        self.col_line = QLineEdit()

        self.d_line.setReadOnly(True)
        self.lambda_line.setReadOnly(True)
        self.gonio_line.setReadOnly(True)
        self.two_theta_line.setReadOnly(True)
        self.horz_vert_line.setReadOnly(True)
        self.run_line.setReadOnly(True)
        self.bank_line.setReadOnly(True)
        self.row_line.setReadOnly(True)
        self.col_line.setReadOnly(True)

        self.intensity_line = QLineEdit()
        self.sigma_line = QLineEdit()

        self.intensity_line.setReadOnly(True)
        self.sigma_line.setReadOnly(True)

        intensity_label = QLabel("I: ", self)
        sigma_label = QLabel("± σ:", self)

        extended_info.addWidget(run_label, 0, 0)
        extended_info.addWidget(self.run_line, 0, 1)
        extended_info.addWidget(bank_label, 0, 2)
        extended_info.addWidget(self.bank_line, 0, 3)
        extended_info.addWidget(row_label, 0, 4)
        extended_info.addWidget(self.row_line, 0, 5)
        extended_info.addWidget(col_label, 0, 6)
        extended_info.addWidget(self.col_line, 0, 7)

        extended_info.addWidget(intensity_label, 1, 0)
        extended_info.addWidget(self.intensity_line, 1, 1)
        extended_info.addWidget(sigma_label, 1, 2)
        extended_info.addWidget(self.sigma_line, 1, 3)
        extended_info.addWidget(d_label, 1, 4)
        extended_info.addWidget(self.d_line, 1, 5)
        extended_info.addWidget(lambda_label, 1, 6)
        extended_info.addWidget(self.lambda_line, 1, 7)

        extended_info.addWidget(self.gonio_label, 2, 0)
        extended_info.addWidget(self.gonio_line, 2, 1, 1, 3)
        extended_info.addWidget(two_theta_label, 2, 4)
        extended_info.addWidget(self.two_theta_line, 2, 5)
        extended_info.addWidget(horz_vert_label, 2, 6)
        extended_info.addWidget(self.horz_vert_line, 2, 7)

        hkl_info = QHBoxLayout()
        peak_info = QGridLayout()

        left_label = QLabel("(", self)
        left_comma_label = QLabel(",", self)
        right_comma_label = QLabel(",", self)
        right_label = QLabel(")", self)

        index_label = QLabel("Indexed:", self)
        total_label = QLabel("Total:", self)

        self.index_line = QLineEdit("0")
        self.total_line = QLineEdit("0")

        self.index_line.setReadOnly(True)
        self.total_line.setReadOnly(True)

        int_h_label = QLabel("h", self)
        int_k_label = QLabel("k", self)
        int_l_label = QLabel("l", self)

        int_m_label = QLabel("m", self)
        int_n_label = QLabel("n", self)
        int_p_label = QLabel("p", self)

        self.h_line = QLineEdit()
        self.k_line = QLineEdit()
        self.l_line = QLineEdit()

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.h_line.setValidator(validator)
        self.k_line.setValidator(validator)
        self.l_line.setValidator(validator)

        self.h_line.setPlaceholderText("h")
        self.k_line.setPlaceholderText("k")
        self.l_line.setPlaceholderText("l")

        validator = QIntValidator(-1000000000, 1000000000, self)

        self.int_h_line = QLineEdit()
        self.int_k_line = QLineEdit()
        self.int_l_line = QLineEdit()

        self.int_h_line.setValidator(validator)
        self.int_k_line.setValidator(validator)
        self.int_l_line.setValidator(validator)

        self.int_h_line.setPlaceholderText("H")
        self.int_k_line.setPlaceholderText("K")
        self.int_l_line.setPlaceholderText("L")

        self.int_m_line = QLineEdit()
        self.int_n_line = QLineEdit()
        self.int_p_line = QLineEdit()

        self.int_m_line.setValidator(validator)
        self.int_n_line.setValidator(validator)
        self.int_p_line.setValidator(validator)

        self.int_m_line.setPlaceholderText("m")
        self.int_n_line.setPlaceholderText("n")
        self.int_p_line.setPlaceholderText("p")

        hkl_info.addWidget(self.delete_peak_button)
        hkl_info.addWidget(left_label)
        hkl_info.addWidget(self.h_line)
        hkl_info.addWidget(left_comma_label)
        hkl_info.addWidget(self.k_line)
        hkl_info.addWidget(right_comma_label)
        hkl_info.addWidget(self.l_line)
        hkl_info.addWidget(right_label)
        hkl_info.addWidget(index_label)
        hkl_info.addWidget(self.index_line)
        hkl_info.addWidget(total_label)
        hkl_info.addWidget(self.total_line)

        peak_info.addWidget(int_h_label, 0, 0, Qt.AlignCenter)
        peak_info.addWidget(int_k_label, 0, 1, Qt.AlignCenter)
        peak_info.addWidget(int_l_label, 0, 2, Qt.AlignCenter)
        peak_info.addWidget(int_m_label, 0, 3, Qt.AlignCenter)
        peak_info.addWidget(int_n_label, 0, 4, Qt.AlignCenter)
        peak_info.addWidget(int_p_label, 0, 5, Qt.AlignCenter)

        peak_info.addWidget(self.int_h_line, 1, 0)
        peak_info.addWidget(self.int_k_line, 1, 1)
        peak_info.addWidget(self.int_l_line, 1, 2)
        peak_info.addWidget(self.int_m_line, 1, 3)
        peak_info.addWidget(self.int_n_line, 1, 4)
        peak_info.addWidget(self.int_p_line, 1, 5)

        peaks_layout.addLayout(calculator_layout)
        peaks_layout.addWidget(self.peaks_table)
        peaks_layout.addLayout(hkl_info)
        peaks_layout.addLayout(peak_info)
        peaks_layout.addLayout(extended_info)

        peaks_table_tab.setLayout(peaks_layout)

    def verify_tab(self):
        """
        Build the "Views" tab containing the slice and detector views.

        Creates the "Slice View" and "Detector View" sub-tabs and adds
        the resulting tab widget to the main tab widget.
        """

        self.inspect_verify_tab = QTabWidget()
        self.tab_widget.addTab(self.inspect_verify_tab, "Views")

        inspect_tab = self.__init_inspect_tab()
        verify_tab = self.__init_verify_tab()

        self.inspect_verify_tab.addTab(inspect_tab, "Slice View")
        self.inspect_verify_tab.addTab(verify_tab, "Detector View")

    def __init_inspect_tab(self):
        """
        Build the "Slice View" sub-tab.

        Creates the HKL projection axis inputs, slice position/thickness/
        width controls, colormap/color-limit/scale selectors, the slice
        slider, and the matplotlib canvas used to display HKL slices.

        Returns
        -------
        convert_to_hkl_tab : QWidget
            Widget containing the slice view controls and plot canvas.
        """

        convert_to_hkl_tab = QWidget()
        convert_to_hkl_tab_layout = QVBoxLayout()

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

        self.convert_to_hkl_button = QPushButton("Convert", self)
        self.convert_to_hkl_button.setToolTip("Convert HKL to Q space.")
        self.convert_to_hkl_button.setIcon(
            qta.icon("fa6s.arrow-right-arrow-left")
        )

        self.slice_add_peak_box = QCheckBox("Add Peak", self)
        self.slice_add_peak_box.setToolTip(
            "Enable adding a peak to the peaks workspace when clicking the slice view."
        )

        self.clim_combo = QComboBox(self)
        self.clim_combo.addItem("Min/Max")
        self.clim_combo.addItem("μ±3×σ")
        self.clim_combo.addItem("Q₃/Q₁±1.5×IQR")
        self.clim_combo.setCurrentIndex(1)
        self.clim_combo.setToolTip("Select color limit adjustment method.")

        self.cbar_combo = QComboBox(self)

        self.cbar_combo.addItem("Sequential")
        self.cbar_combo.addItem("Rainbow")
        self.cbar_combo.addItem("Binary")
        self.cbar_combo.addItem("Diverging")
        self.cbar_combo.addItem("Modified")
        self.cbar_combo.setCurrentIndex(2)
        self.cbar_combo.setToolTip("Select color map for the slice view.")
        self.auto_scale_dropdown(self.filter_combo)
        self.auto_scale_dropdown(self.comparison_combo)

        self.slice_combo = QComboBox(self)
        self.slice_combo.addItem("Axis 1/2")
        self.slice_combo.addItem("Axis 1/3")
        self.slice_combo.addItem("Axis 2/3")
        self.slice_combo.setCurrentIndex(0)
        self.slice_combo.setToolTip("Select the axes for the slice view.")

        self.auto_scale_dropdown(self.clim_combo)
        self.auto_scale_dropdown(self.cbar_combo)
        self.auto_scale_dropdown(self.slice_combo)

        slice_label = QLabel("Slice:", self)

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

        validator = QDoubleValidator(0.005, 0.5, 5, notation=notation)

        slice_width_label = QLabel("Width:", self)

        self.slice_width_line = QLineEdit("0.05")
        self.slice_width_line.setValidator(validator)
        self.slice_width_line.setToolTip("Enter the slice width value.")

        self.slice_scale_combo = QComboBox(self)
        self.slice_scale_combo.addItem("Linear")
        self.slice_scale_combo.addItem("Log")
        self.slice_scale_combo.setToolTip(
            "Select the scale for the slice view."
        )
        self.auto_scale_dropdown(self.slice_scale_combo)

        validator = QDoubleValidator(-1e32, 1e32, 6, notation=notation)

        self.vmin_line = QLineEdit("")
        self.vmax_line = QLineEdit("")
        self.slice_auto_limits_box = QCheckBox("Auto Limits")
        self.slice_auto_limits_box.setChecked(True)
        self.slice_auto_limits_box.setToolTip(
            "Automatically reset slice limits on redraw. Uncheck to reuse the current limits."
        )
        self.slice_auto_zoom_box = QCheckBox("Auto Zoom")
        self.slice_auto_zoom_box.setChecked(True)
        self.slice_auto_zoom_box.setToolTip(
            "Automatically reset slice zoom to full extent on redraw. Uncheck to retain current zoom when possible."
        )
        self.slice_auto_zoom_box.toggled.connect(
            self._handle_slice_auto_zoom_toggle
        )

        self.vmin_line.setValidator(validator)
        self.vmax_line.setValidator(validator)
        self.vmin_line.setToolTip("Set the minimum value for the colorbar.")
        self.vmax_line.setToolTip("Set the maximum value for the colorbar.")

        convert_to_hkl_action_layout = QHBoxLayout()
        convert_to_hkl_action_layout.addWidget(self.convert_to_hkl_button)
        convert_to_hkl_action_layout.addWidget(self.slice_add_peak_box)
        convert_to_hkl_action_layout.addWidget(self.slice_combo)
        convert_to_hkl_action_layout.addWidget(slice_label)
        convert_to_hkl_action_layout.addWidget(self.slice_line)
        convert_to_hkl_action_layout.addWidget(slice_thickness_label)
        convert_to_hkl_action_layout.addWidget(self.slice_thickness_line)
        convert_to_hkl_action_layout.addWidget(slice_width_label)
        convert_to_hkl_action_layout.addWidget(self.slice_width_line)

        convert_to_hkl_view_layout = QHBoxLayout()
        convert_to_hkl_view_layout.addWidget(self.cbar_combo)
        convert_to_hkl_view_layout.addWidget(self.clim_combo)
        convert_to_hkl_view_layout.addWidget(self.slice_scale_combo)
        convert_to_hkl_view_layout.addWidget(self.slice_auto_limits_box)
        convert_to_hkl_view_layout.addWidget(self.slice_auto_zoom_box)
        convert_to_hkl_view_layout.addWidget(QLabel("V Min:", self))
        convert_to_hkl_view_layout.addWidget(self.vmin_line)
        convert_to_hkl_view_layout.addWidget(QLabel("V Max:", self))
        convert_to_hkl_view_layout.addWidget(self.vmax_line)

        convert_to_hkl_tab_layout.addLayout(convert_to_hkl_params_layout)
        convert_to_hkl_tab_layout.addStretch(1)
        convert_to_hkl_tab_layout.addLayout(convert_to_hkl_action_layout)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setToolTip(
            "Drag or use arrow keys to move the slice position."
        )
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self._slice_steps)
        self.slice_slider.valueChanged.connect(self._on_slice_slider_changed)
        self.slice_slider.sliderReleased.connect(self._slice_slider_released)
        self.slice_slider.installEventFilter(self)

        self.canvas_slice = FigureCanvas(Figure(figsize=[12.8, 12.8]))
        self.canvas_slice.setFocusPolicy(Qt.StrongFocus)
        self.canvas_slice.installEventFilter(self)

        self.ax_xint = None
        self.ax_yint = None
        self.cb_slice = None
        self.cb_inst = None

        self.fig_slice = self.canvas_slice.figure

        self.ax_slice = self.fig_slice.subplots(1, 1)

        slice_layout = QVBoxLayout()
        slice_layout.addWidget(self.slice_slider)
        slice_layout.addWidget(NavigationToolbar2QT(self.canvas_slice, self))
        slice_layout.addWidget(self.canvas_slice)

        convert_to_hkl_tab_layout.addLayout(slice_layout)
        convert_to_hkl_tab_layout.addLayout(convert_to_hkl_view_layout)

        convert_to_hkl_tab.setLayout(convert_to_hkl_tab_layout)

        return convert_to_hkl_tab

    def __init_verify_tab(self):
        """
        Build the "Detector View" sub-tab.

        Creates the data source selector, d-spacing/hkl check controls,
        color scale/colormap selectors, the horizontal/vertical detector
        angle and ROI controls, and the matplotlib canvases for the
        detector and scan views.

        Returns
        -------
        instrument_tab : QWidget
            Widget containing the detector view controls and plot canvases.
        """

        instrument_tab = QWidget()
        instrument_tab_layout = QVBoxLayout()

        notation = QDoubleValidator.StandardNotation

        self.data_combo = QComboBox(self)

        d_min_label = QLabel("d(min):", self)
        d_max_label = QLabel("d(max):", self)

        validator = QDoubleValidator(0, float("inf"), 5, notation=notation)

        self.d_min_line = QLineEdit("0")
        self.d_min_line.setValidator(validator)
        self.d_min_line.setToolTip("Minimum d-spacing for verification.")

        self.d_max_line = QLineEdit("inf")
        self.d_max_line.setValidator(validator)
        self.d_max_line.setToolTip("Maximum d-spacing for verification.")

        self.check_h_line = QLineEdit()
        self.check_k_line = QLineEdit()
        self.check_l_line = QLineEdit()

        self.check_h_line.setPlaceholderText("h")
        self.check_k_line.setPlaceholderText("k")
        self.check_l_line.setPlaceholderText("l")

        self.check_hkl_button = QPushButton("Check hkl", self)
        self.check_hkl_button.setToolTip(
            "Check the validity of the hkl indices."
        )

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.check_h_line.setValidator(validator)
        self.check_k_line.setValidator(validator)
        self.check_l_line.setValidator(validator)

        self.vlim_combo = QComboBox(self)
        self.vlim_combo.addItem("Min/Max")
        self.vlim_combo.addItem("μ±3×σ")
        self.vlim_combo.addItem("Q₃/Q₁±1.5×IQR")
        self.vlim_combo.setCurrentIndex(1)
        self.vlim_combo.setToolTip("Select color limit adjustment method.")

        self.vbar_combo = QComboBox(self)
        self.vbar_combo.addItem("Sequential")
        self.vbar_combo.addItem("Rainbow")
        self.vbar_combo.addItem("Binary")
        self.vbar_combo.addItem("Diverging")
        self.vbar_combo.addItem("Modified")
        self.vbar_combo.setCurrentIndex(0)
        self.vbar_combo.setToolTip("Select color map for the slice view.")

        self.instrument_scale_combo = QComboBox(self)
        self.instrument_scale_combo.addItem("Linear")
        self.instrument_scale_combo.addItem("Log")
        self.instrument_scale_combo.setToolTip(
            "Select the scale for the instrument view."
        )

        validator = QDoubleValidator(-1e32, 1e32, 6, notation=notation)

        self.inst_vmin_line = QLineEdit("")
        self.inst_vmax_line = QLineEdit("")
        self.instrument_auto_limits_box = QCheckBox("Auto Limits")
        self.instrument_auto_limits_box.setChecked(True)
        self.instrument_auto_limits_box.setToolTip(
            "Automatically reset instrument limits on redraw. Uncheck to reuse the current limits."
        )
        self.instrument_auto_zoom_box = QCheckBox("Auto Zoom")
        self.instrument_auto_zoom_box.setChecked(True)
        self.instrument_auto_zoom_box.setToolTip(
            "Automatically reset instrument zoom to full extent on redraw. Uncheck to retain current zoom when possible."
        )
        self.instrument_auto_zoom_box.toggled.connect(
            self._handle_instrument_auto_zoom_toggle
        )

        self.inst_vmin_line.setValidator(validator)
        self.inst_vmax_line.setValidator(validator)
        self.inst_vmin_line.setToolTip(
            "Set the minimum value for the instrument color range."
        )
        self.inst_vmax_line.setToolTip(
            "Set the maximum value for the instrument color range."
        )

        self.auto_scale_dropdown(self.data_combo)
        self.auto_scale_dropdown(self.vlim_combo)
        self.auto_scale_dropdown(self.vbar_combo)
        self.auto_scale_dropdown(self.instrument_scale_combo)

        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_combo)
        data_layout.addStretch(1)
        data_layout.addWidget(self.vlim_combo)
        data_layout.addWidget(self.vbar_combo)
        data_layout.addWidget(self.instrument_scale_combo)
        data_layout.addWidget(self.instrument_auto_limits_box)
        data_layout.addWidget(self.instrument_auto_zoom_box)
        data_layout.addWidget(QLabel("V Min:", self))
        data_layout.addWidget(self.inst_vmin_line)
        data_layout.addWidget(QLabel("V Max:", self))
        data_layout.addWidget(self.inst_vmax_line)

        vertical_label = QLabel("Vertical Angle:", self)
        horizontal_label = QLabel("Horizontal Angle:", self)

        vertical_roi_label = QLabel("ROI:", self)
        horizontal_roi_label = QLabel("ROI:", self)

        validator = QDoubleValidator(-180, 180, 5, notation=notation)

        self.vertical_line = QLineEdit("0")
        self.vertical_line.setValidator(validator)
        self.vertical_line.setToolTip("Vertical angle for the detector view.")

        self.horizontal_line = QLineEdit("0")
        self.horizontal_line.setValidator(validator)
        self.horizontal_line.setToolTip(
            "Horizontal angle for the detector view."
        )

        validator = QDoubleValidator(0, 180, 5, notation=notation)

        self.vertical_roi_line = QLineEdit("2")
        self.vertical_roi_line.setValidator(validator)
        self.vertical_roi_line.setToolTip(
            "Vertical ROI size for the detector view."
        )

        self.horizontal_roi_line = QLineEdit("2")
        self.horizontal_roi_line.setValidator(validator)
        self.horizontal_roi_line.setToolTip(
            "Horizontal ROI size for the detector view."
        )

        angle_layout = QHBoxLayout()
        angle_layout.addWidget(horizontal_label)
        angle_layout.addWidget(self.horizontal_line)
        angle_layout.addWidget(horizontal_roi_label)
        angle_layout.addWidget(self.horizontal_roi_line)
        angle_layout.addWidget(vertical_label)
        angle_layout.addWidget(self.vertical_line)
        angle_layout.addWidget(vertical_roi_label)
        angle_layout.addWidget(self.vertical_roi_line)
        angle_layout.addStretch(1)
        angle_layout.addWidget(d_min_label)
        angle_layout.addWidget(self.d_min_line)
        angle_layout.addWidget(d_max_label)
        angle_layout.addWidget(self.d_max_line)

        self.add_peak_button = QPushButton("Add Peak", self)
        self.add_peak_button.setToolTip("Add a peak to the list.")
        self.add_peak_button.setIcon(qta.icon("fa6s.square-plus"))

        self.save_roi_mask_button = QPushButton("Save ROI Mask", self)
        self.save_roi_mask_button.setToolTip(
            "Save the current instrument ROI box as a detector mask XML file."
        )
        self.save_roi_mask_button.setIcon(qta.icon("fa6s.floppy-disk"))

        self.diffraction_label = QLabel("Axis:", self)
        self.inst_gonio_label = QLabel("ω,χ,φ [°]:", self)

        validator = QDoubleValidator(
            -float("inf"), float("inf"), 5, notation=notation
        )

        self.diffraction_line = QLineEdit("0")
        self.inst_gonio_line = QLineEdit()
        self.diffraction_line.setValidator(validator)
        self.diffraction_line.setToolTip("Diffraction angle or wavelength.")
        self.inst_gonio_line.setReadOnly(True)

        peak_layout = QHBoxLayout()
        peak_layout.addWidget(self.check_hkl_button)
        peak_layout.addWidget(self.check_h_line)
        peak_layout.addWidget(self.check_k_line)
        peak_layout.addWidget(self.check_l_line)
        peak_layout.addWidget(self.diffraction_label)
        peak_layout.addWidget(self.diffraction_line)
        peak_layout.addWidget(self.inst_gonio_label)
        peak_layout.addWidget(self.inst_gonio_line)
        peak_layout.addStretch(1)
        peak_layout.addWidget(self.save_roi_mask_button)
        peak_layout.addWidget(self.add_peak_button)

        self.canvas_inst = FigureCanvas(Figure(constrained_layout=True))
        self.canvas_scan = FigureCanvas(Figure(constrained_layout=True))

        self.fig_inst = self.canvas_inst.figure
        self.fig_scan = self.canvas_scan.figure

        self.ax_inst = self.fig_inst.subplots(1, 1)
        self.ax_scan = self.fig_scan.subplots(1, 1)

        view_layout = QVBoxLayout()

        view_layout.addLayout(data_layout)
        view_layout.addWidget(NavigationToolbar2QT(self.canvas_inst, self))
        view_layout.addWidget(self.canvas_inst)

        view_layout.addLayout(angle_layout)
        view_layout.addWidget(NavigationToolbar2QT(self.canvas_scan, self))
        view_layout.addWidget(self.canvas_scan)

        view_layout.addLayout(peak_layout)

        instrument_tab_layout.addLayout(view_layout)

        instrument_tab.setLayout(instrument_tab_layout)

        return instrument_tab

    def modulation_tab(self):
        """
        Build the "Modulation" tab.

        Creates the DBSCAN-style clustering controls (maximum distance,
        minimum samples), the cluster results table, and the matplotlib
        canvas showing the h00/0k0/00l offset distributions, then adds
        the tab to the main tab widget.
        """

        mod_tab = QWidget()
        self.tab_widget.addTab(mod_tab, "Modulation")

        modulation_layout = QVBoxLayout()

        self.cluster_button = QPushButton("Cluster", self)
        self.cluster_button.setToolTip("Cluster the selected peaks.")
        self.cluster_button.setIcon(qta.icon("fa6s.circle-nodes"))

        self.param_eps_line = QLineEdit("0.025")
        self.param_min_line = QLineEdit("15")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.0001, 10, 5, notation=notation)

        self.param_eps_line.setValidator(validator)

        validator = QIntValidator(1, 1000)

        self.param_min_line.setValidator(validator)

        self.cluster_table = QTableWidget()

        self.cluster_table.setRowCount(0)
        self.cluster_table.setColumnCount(3)

        self.cluster_table.horizontalHeader().setStretchLastSection(True)
        self.cluster_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.cluster_table.setHorizontalHeaderLabels(["h", "k", "l"])

        cluster_layout = QVBoxLayout()
        params_layout = QHBoxLayout()

        dist_label = QLabel("Maximum distance:", self)
        samp_label = QLabel("Minimum samples:", self)

        params_layout.addWidget(self.cluster_button)
        params_layout.addWidget(dist_label)
        params_layout.addWidget(self.param_eps_line)
        params_layout.addWidget(QLabel("r.l.u.", self))
        params_layout.addWidget(samp_label)
        params_layout.addWidget(self.param_min_line)
        params_layout.addWidget(QLabel("#", self))

        cluster_layout.addLayout(params_layout)
        cluster_layout.addWidget(self.cluster_table)

        plot_layout = QVBoxLayout()

        self.canvas_clust = FigureCanvas(Figure(tight_layout=True))

        plot_layout.addWidget(NavigationToolbar2QT(self.canvas_clust, self))
        plot_layout.addWidget(self.canvas_clust)

        fig = self.canvas_clust.figure

        self.ax_clust = fig.subplots(3, 1, sharex=True, sharey=True)

        for i in range(3):
            self.ax_clust[i].set_xlim(-1, 1)
            self.ax_clust[i].set_ylim(1, 100)
            self.ax_clust[i].minorticks_on()
            self.ax_clust[i].set_yscale("log")

        self.ax_clust[0].set_xlabel("$[h00]$")
        self.ax_clust[1].set_xlabel("$[0k0]$")
        self.ax_clust[2].set_xlabel("$[00l]$")

        modulation_layout.addLayout(cluster_layout)
        modulation_layout.addLayout(plot_layout)

        mod_tab.setLayout(modulation_layout)

    def alignment_tab(self):
        """
        Build the "Alignment" tab.

        Creates the run selector, manual goniometer yaw/pitch/roll
        correction inputs, the calculate-alignment button, and the
        matplotlib canvas comparing UB-predicted and observed peak
        positions, then adds the tab to the main tab widget.
        """

        align_tab = QWidget()
        self.tab_widget.addTab(align_tab, "Alignment")

        alignment_layout = QVBoxLayout()

        run_label = QLabel("Run #:")
        yaw_label = QLabel("Yaw (y):")
        pitch_label = QLabel("Pitch (x):")
        roll_label = QLabel("Roll (z):")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-360, 360, 3, notation=notation)

        self.align_yaw_line = QLineEdit("0")
        self.align_pitch_line = QLineEdit("0")
        self.align_roll_line = QLineEdit("0")

        self.align_yaw_line.setValidator(validator)
        self.align_pitch_line.setValidator(validator)
        self.align_roll_line.setValidator(validator)

        self.align_yaw_line.setToolTip(
            "Manual goniometer yaw correction in degrees about y."
        )
        self.align_pitch_line.setToolTip(
            "Manual goniometer pitch correction in degrees about x."
        )
        self.align_roll_line.setToolTip(
            "Manual goniometer roll correction in degrees about z."
        )

        self.alignment_run_combo = QComboBox(self)
        self.alignment_run_combo.addItem("None")
        self.alignment_run_combo.setEnabled(False)
        self.alignment_run_combo.setToolTip(
            "Run number used for the alignment comparison."
        )
        self.auto_scale_dropdown(self.alignment_run_combo)

        self.calculate_alignment_button = QPushButton(
            "Calculate Alignment", self
        )
        self.calculate_alignment_button.setToolTip(
            "Compare UB-predicted Q positions to observed peaks for one run."
        )
        self.calculate_alignment_button.setIcon(
            qta.icon("fa6s.ruler-combined")
        )
        self.calculate_alignment_button.setEnabled(False)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(run_label)
        controls_layout.addWidget(self.alignment_run_combo)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(yaw_label)
        controls_layout.addWidget(self.align_yaw_line)
        controls_layout.addWidget(pitch_label)
        controls_layout.addWidget(self.align_pitch_line)
        controls_layout.addWidget(roll_label)
        controls_layout.addWidget(self.align_roll_line)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(self.calculate_alignment_button)
        controls_layout.addStretch(1)

        plot_layout = QVBoxLayout()

        self.canvas_align = FigureCanvas(Figure(tight_layout=True))

        plot_layout.addWidget(NavigationToolbar2QT(self.canvas_align, self))
        plot_layout.addWidget(self.canvas_align)

        fig = self.canvas_align.figure

        self.ax_align = fig.subplots(3, 1, sharex=True, sharey=True)

        self.clear_alignment_plot()

        alignment_layout.addLayout(controls_layout)
        alignment_layout.addLayout(plot_layout)

        align_tab.setLayout(alignment_layout)

    def connect_cluster(self, cluster):
        """
        Connect a callback to the cluster button click.

        Parameters
        ----------
        cluster : callable
            Slot invoked to cluster the selected peaks.
        """

        self.cluster_button.clicked.connect(cluster)

    def connect_calculate_alignment(self, calculate_alignment):
        """
        Connect a callback to the calculate-alignment button click.

        Parameters
        ----------
        calculate_alignment : callable
            Slot invoked to compute the alignment comparison.
        """

        self.calculate_alignment_button.clicked.connect(calculate_alignment)

    def connect_h_index(self, update_index):
        """
        Connect a callback to editing-finished on the fractional h index field.

        Parameters
        ----------
        update_index : callable
            Slot invoked when editing of the h field finishes.
        """

        self.h_line.editingFinished.connect(update_index)

    def connect_k_index(self, update_index):
        """
        Connect a callback to editing-finished on the fractional k index field.

        Parameters
        ----------
        update_index : callable
            Slot invoked when editing of the k field finishes.
        """

        self.k_line.editingFinished.connect(update_index)

    def connect_l_index(self, update_index):
        """
        Connect a callback to editing-finished on the fractional l index field.

        Parameters
        ----------
        update_index : callable
            Slot invoked when editing of the l field finishes.
        """

        self.l_line.editingFinished.connect(update_index)

    def connect_integer_h_index(self, update_index):
        """
        Connect a callback to editing-finished on the integer H index field.

        Parameters
        ----------
        update_index : callable
            Slot invoked when editing of the integer H field finishes.
        """

        self.int_h_line.editingFinished.connect(update_index)

    def connect_integer_k_index(self, update_index):
        """
        Connect a callback to editing-finished on the integer K index field.

        Parameters
        ----------
        update_index : callable
            Slot invoked when editing of the integer K field finishes.
        """

        self.int_k_line.editingFinished.connect(update_index)

    def connect_integer_l_index(self, update_index):
        """
        Connect a callback to editing-finished on the integer L index field.

        Parameters
        ----------
        update_index : callable
            Slot invoked when editing of the integer L field finishes.
        """

        self.int_l_line.editingFinished.connect(update_index)

    def connect_integer_m_index(self, update_index):
        """
        Connect a callback to editing-finished on the integer m satellite index field.

        Parameters
        ----------
        update_index : callable
            Slot invoked when editing of the integer m field finishes.
        """

        self.int_m_line.editingFinished.connect(update_index)

    def connect_integer_n_index(self, update_index):
        """
        Connect a callback to editing-finished on the integer n satellite index field.

        Parameters
        ----------
        update_index : callable
            Slot invoked when editing of the integer n field finishes.
        """

        self.int_n_line.editingFinished.connect(update_index)

    def connect_integer_p_index(self, update_index):
        """
        Connect a callback to editing-finished on the integer p satellite index field.

        Parameters
        ----------
        update_index : callable
            Slot invoked when editing of the integer p field finishes.
        """

        self.int_p_line.editingFinished.connect(update_index)

    def connect_data_combo(self, update_inst_data):
        """
        Connect a callback to the instrument data source combo box.

        Parameters
        ----------
        update_inst_data : callable
            Slot invoked when the selected data source changes.
        """

        self.data_combo.currentIndexChanged.connect(update_inst_data)

    def connect_add_peak(self, add_peak):
        """
        Connect a callback to the "Add Peak" button click.

        Parameters
        ----------
        add_peak : callable
            Slot invoked to add a new peak to the peaks workspace.
        """

        self.add_peak_button.clicked.connect(add_peak)

    def connect_save_roi_mask(self, save_roi_mask):
        """
        Connect a callback to the "Save ROI Mask" button click.

        Parameters
        ----------
        save_roi_mask : callable
            Slot invoked to save the current instrument ROI as a mask file.
        """

        self.save_roi_mask_button.clicked.connect(save_roi_mask)

    def connect_delete_peak(self, delete_peak):
        """
        Connect a callback to the "Delete Highlighted" button click.

        Parameters
        ----------
        delete_peak : callable
            Slot invoked to delete the currently highlighted peak.
        """

        self.delete_peak_button.clicked.connect(delete_peak)

    def connect_check_hkl(self, check_hkl):
        """
        Connect a callback to the "Check hkl" button click.

        Parameters
        ----------
        check_hkl : callable
            Slot invoked to validate the entered hkl indices.
        """

        self.check_hkl_button.clicked.connect(check_hkl)

    def connect_diffraction(self, update_inst_data):
        """
        Connect a callback to editing-finished on the diffraction angle field.

        Parameters
        ----------
        update_inst_data : callable
            Slot invoked when editing of the diffraction field finishes.
        """

        self.diffraction_line.editingFinished.connect(update_inst_data)

    def connect_d_min(self, update_inst_data):
        """
        Connect a callback to editing-finished on the detector view minimum d-spacing field.

        Parameters
        ----------
        update_inst_data : callable
            Slot invoked when editing of the d(min) field finishes.
        """

        self.d_min_line.editingFinished.connect(update_inst_data)

    def connect_d_max(self, update_inst_data):
        """
        Connect a callback to editing-finished on the detector view maximum d-spacing field.

        Parameters
        ----------
        update_inst_data : callable
            Slot invoked when editing of the d(max) field finishes.
        """

        self.d_max_line.editingFinished.connect(update_inst_data)

    def connect_horizontal(self, update_inst_data):
        """
        Connect a callback to editing-finished on the horizontal angle field.

        Parameters
        ----------
        update_inst_data : callable
            Slot invoked when editing of the horizontal angle field finishes.
        """

        self.horizontal_line.editingFinished.connect(update_inst_data)

    def connect_vertical(self, update_inst_data):
        """
        Connect a callback to editing-finished on the vertical angle field.

        Parameters
        ----------
        update_inst_data : callable
            Slot invoked when editing of the vertical angle field finishes.
        """

        self.vertical_line.editingFinished.connect(update_inst_data)

    def connect_horizontal_roi(self, update_inst_data):
        """
        Connect a callback to editing-finished on the horizontal ROI size field.

        Parameters
        ----------
        update_inst_data : callable
            Slot invoked when editing of the horizontal ROI field finishes.
        """

        self.horizontal_roi_line.editingFinished.connect(update_inst_data)

    def connect_vertical_roi(self, update_inst_data):
        """
        Connect a callback to editing-finished on the vertical ROI size field.

        Parameters
        ----------
        update_inst_data : callable
            Slot invoked when editing of the vertical ROI field finishes.
        """

        self.vertical_roi_line.editingFinished.connect(update_inst_data)

    def connect_convert_to_hkl(self, convert_to_hkl):
        """
        Connect a callback to the "Convert" button in the slice view tab.

        Parameters
        ----------
        convert_to_hkl : callable
            Slot invoked to convert Q-sample data to an HKL workspace.
        """

        self.convert_to_hkl_button.clicked.connect(convert_to_hkl)

    def connect_browse_calibration(self, load_detector_cal):
        """
        Connect a callback to the detector calibration browse button.

        Parameters
        ----------
        load_detector_cal : callable
            Slot invoked to open a file dialog and load a detector
            calibration file.
        """

        self.cal_browse_button.clicked.connect(load_detector_cal)

    def connect_browse_tube(self, load_tube_cal):
        """
        Connect a callback to the tube calibration browse button.

        Parameters
        ----------
        load_tube_cal : callable
            Slot invoked to open a file dialog and load a tube
            calibration file.
        """

        self.tube_browse_button.clicked.connect(load_tube_cal)

    def connect_convert_Q(self, convert_Q):
        """
        Connect a callback to the "Convert" button click for Q conversion.

        Parameters
        ----------
        convert_Q : callable
            Slot invoked to convert raw data to a Q-sample workspace.
        """

        self.convert_to_q_button.clicked.connect(convert_Q)

    def connect_live_toggle(self, toggle_live):
        """
        Connect a callback to the "Live" checkbox toggling.

        Parameters
        ----------
        toggle_live : callable
            Slot invoked with the new checkbox state (bool) when the
            "Live" checkbox is toggled.
        """

        self.live_box.toggled.connect(toggle_live)

    def connect_reload_convert_Q(self, convert_Q):
        """
        Connect a callback to the "Convert + Reload" button click.

        Parameters
        ----------
        convert_Q : callable
            Slot invoked to force-reload raw data and convert to a
            Q-sample workspace.
        """

        self.reload_convert_to_q_button.clicked.connect(convert_Q)

    def connect_find_peaks(self, find_peaks):
        """
        Connect a callback to the "Find" peaks button click.

        Parameters
        ----------
        find_peaks : callable
            Slot invoked to search for peaks in the Q workspace.
        """

        self.find_button.clicked.connect(find_peaks)

    def connect_find_distance(self, update):
        """
        Connect a callback to editing-finished on the minimum peak distance field.

        Parameters
        ----------
        update : callable
            Slot invoked when editing of the minimum distance field finishes.
        """

        self.min_distance_line.editingFinished.connect(update)

    def connect_find_spacing(self, update):
        """
        Connect a callback to editing-finished on the maximum d-spacing field.

        Parameters
        ----------
        update : callable
            Slot invoked when editing of the maximum spacing field finishes.
        """

        self.max_spacing_line.editingFinished.connect(update)

    def connect_find_conventional(self, find_conventional):
        """
        Connect a callback to the "Find UB with Lattice Parameters" button click.

        Parameters
        ----------
        find_conventional : callable
            Slot invoked to search for a conventional cell matching the
            given lattice parameters.
        """

        self.conventional_button.clicked.connect(find_conventional)

    def connect_find_niggli(self, find_niggli):
        """
        Connect a callback to the "Find Primitive Cell" button click.

        Parameters
        ----------
        find_niggli : callable
            Slot invoked to search for the reduced (Niggli) cell.
        """

        self.niggli_button.clicked.connect(find_niggli)

    def connect_select_form(self, select_form):
        """
        Connect a callback to the "Select Conventional Cell" button click.

        Parameters
        ----------
        select_form : callable
            Slot invoked to select the highlighted cell in the cell table.
        """

        self.select_button.clicked.connect(select_form)

    def connect_set_UB(self, set_UB):
        """
        Connect a callback to the "Set UB" button click.

        Parameters
        ----------
        set_UB : callable
            Slot invoked to set the UB matrix from the entered lattice
            parameters and orientation.
        """

        self.set_ub_button.clicked.connect(set_UB)

    def connect_set_UB_from_scattering_plane(self, set_UB):
        """
        Connect a callback to the "Search U from Scattering Plane" button click.

        Parameters
        ----------
        set_UB : callable
            Slot invoked to compute the UB matrix from the scattering
            plane vectors.
        """

        self.set_scattering_plane_ub_button.clicked.connect(set_UB)

    def connect_convert_HKL(self, convert_HKL):
        """
        Connect a callback to the "Convert" button click for HKL conversion.

        Parameters
        ----------
        convert_HKL : callable
            Slot invoked to convert Q-sample data to HKL.
        """

        self.convert_to_hkl_button.clicked.connect(convert_HKL)

    def connect_switch_instrument(self, switch_instrument):
        """
        Connect a callback to the instrument selection combo box.

        Parameters
        ----------
        switch_instrument : callable
            Slot invoked when the user selects a different instrument.
        """

        self.instrument_combo.activated.connect(switch_instrument)

    def connect_wavelength(self, update_wavelength):
        """
        Connect a callback to editing-finished on the minimum wavelength field.

        Parameters
        ----------
        update_wavelength : callable
            Slot invoked when editing of the wavelength field finishes.
        """

        self.wl_min_line.editingFinished.connect(update_wavelength)

    def connect_load_Q(self, load_Q):
        """
        Connect a callback to the "Load Q" button click.

        Parameters
        ----------
        load_Q : callable
            Slot invoked to load a Q-sample workspace from a file.
        """

        self.load_q_button.clicked.connect(load_Q)

    def connect_save_Q(self, save_Q):
        """
        Connect a callback to the "Save Q" button click.

        Parameters
        ----------
        save_Q : callable
            Slot invoked to save the Q-sample workspace to a file.
        """

        self.save_q_button.clicked.connect(save_Q)

    def connect_load_peaks(self, load_peaks):
        """
        Connect a callback to the "Load Peaks" button click.

        Parameters
        ----------
        load_peaks : callable
            Slot invoked to load a peaks table from a file.
        """

        self.load_peaks_button.clicked.connect(load_peaks)

    def connect_save_peaks(self, save_peaks):
        """
        Connect a callback to the "Save Peaks" button click.

        Parameters
        ----------
        save_peaks : callable
            Slot invoked to save the peaks table to a file.
        """

        self.save_peaks_button.clicked.connect(save_peaks)

    def connect_load_UB(self, load_UB):
        """
        Connect a callback to the "Load UB" button click.

        Parameters
        ----------
        load_UB : callable
            Slot invoked to load a UB matrix from a file.
        """

        self.load_ub_button.clicked.connect(load_UB)

    def connect_save_UB(self, save_UB):
        """
        Connect a callback to the "Save UB" button click.

        Parameters
        ----------
        save_UB : callable
            Slot invoked to save the UB matrix to a file.
        """

        self.save_ub_button.clicked.connect(save_UB)

    def connect_lattice_transform(self, lattice_transform):
        """
        Connect a callback to the lattice system combo box.

        Parameters
        ----------
        lattice_transform : callable
            Slot invoked when the selected lattice system changes.
        """

        self.lattice_combo.currentIndexChanged.connect(lattice_transform)

    def connect_symmetry_transform(self, symmetry_transform):
        """
        Connect a callback to the symmetry operation combo box.

        Parameters
        ----------
        symmetry_transform : callable
            Slot invoked when the selected symmetry operation changes.
        """

        self.symmetry_combo.currentIndexChanged.connect(symmetry_transform)

    def connect_transform_UB(self, transform_UB):
        """
        Connect a callback to the "Transform" button click.

        Parameters
        ----------
        transform_UB : callable
            Slot invoked to apply the lattice transformation matrix to
            the UB matrix.
        """

        self.transform_button.clicked.connect(transform_UB)

    def connect_optimize_UB(self, optimize_UB):
        """
        Connect a callback to the "Refine" button click.

        Parameters
        ----------
        optimize_UB : callable
            Slot invoked to refine (optimize) the UB matrix.
        """

        self.refine_button.clicked.connect(optimize_UB)

    def connect_index_peaks(self, index_peaks):
        """
        Connect a callback to the "Index" button click.

        Parameters
        ----------
        index_peaks : callable
            Slot invoked to index the peaks table using the current UB.
        """

        self.index_button.clicked.connect(index_peaks)

    def connect_predict_peaks(self, predict_peaks):
        """
        Connect a callback to the "Predict" button click.

        Parameters
        ----------
        predict_peaks : callable
            Slot invoked to predict peak positions from the UB matrix.
        """

        self.predict_button.clicked.connect(predict_peaks)

    def connect_filter_peaks(self, filter_peaks):
        """
        Connect a callback to the "Filter" button click.

        Parameters
        ----------
        filter_peaks : callable
            Slot invoked to filter the peaks table by the selected
            variable, comparison, and value.
        """

        self.filter_button.clicked.connect(filter_peaks)

    def connect_undo_filter_peaks(self, undo_filter_peaks):
        """
        Connect a callback to the "Undo" filter button click.

        Parameters
        ----------
        undo_filter_peaks : callable
            Slot invoked to restore the peaks table to its state before
            the last filter operation.
        """

        self.undo_filter_button.clicked.connect(undo_filter_peaks)

    def set_undo_filter_enabled(self, enabled):
        """
        Enable or disable the "Undo" filter button.

        Parameters
        ----------
        enabled : bool
            True to enable the undo button, False to disable it.
        """

        self.undo_filter_button.setEnabled(enabled)

    def connect_integrate_peaks(self, integrate_peaks):
        """
        Connect a callback to the "Integrate" button click.

        Parameters
        ----------
        integrate_peaks : callable
            Slot invoked to integrate peak intensities.
        """

        self.integrate_button.clicked.connect(integrate_peaks)

    def connect_calculate_peaks(self, calculate_peaks):
        """
        Connect a callback to the peak calculator's "Calculate" button click.

        Parameters
        ----------
        calculate_peaks : callable
            Slot invoked to calculate d-spacings and angle between the
            two entered hkl values.
        """

        self.calculate_button.clicked.connect(calculate_peaks)

    def connect_calculate_highlight(self, calculate_highlight):
        """
        Connect a callback to the highlighted-peaks "Calculate" button click.

        Parameters
        ----------
        calculate_highlight : callable
            Slot invoked to calculate d-spacings and angle between the
            two highlighted peaks.
        """

        self.calculate_highlight_button.clicked.connect(calculate_highlight)

    def connect_highlight_1(self, add):
        """
        Connect a callback to the first "Add Highlighted" button click.

        Parameters
        ----------
        add : callable
            Slot invoked to add the currently highlighted peak as peak 1.
        """

        self.highlight_1_button.clicked.connect(add)

    def connect_highlight_2(self, add):
        """
        Connect a callback to the second "Add Highlighted" button click.

        Parameters
        ----------
        add : callable
            Slot invoked to add the currently highlighted peak as peak 2.
        """

        self.highlight_2_button.clicked.connect(add)

    def connect_peak_row_highligter(self, highlight_row):
        """
        Connect a callback to peaks table selection changes.

        Parameters
        ----------
        highlight_row : callable
            Slot invoked when the selected row(s) in the peaks table change.
        """

        self.peaks_table.itemSelectionChanged.connect(highlight_row)

    def connect_cell_row_highligter(self, highlight_row):
        """
        Connect a callback to cell table selection changes.

        Parameters
        ----------
        highlight_row : callable
            Slot invoked when the selected row(s) in the cell table change.
        """

        self.cell_table.itemSelectionChanged.connect(highlight_row)

    def connect_select_cell(self, select_cell):
        """
        Connect a callback to the cell selection button click.

        Parameters
        ----------
        select_cell : callable
            Slot invoked to select the highlighted cell.
        """

        self.select_button.clicked.connect(select_cell)

    def connect_clim_combo(self, update_clim):
        """
        Connect a callback to the slice view color-limit method combo box.

        Parameters
        ----------
        update_clim : callable
            Slot invoked when the selected color-limit method changes.
        """

        self.clim_combo.currentIndexChanged.connect(update_clim)

    def connect_cbar_combo(self, update_cbar):
        """
        Connect a callback to the slice view colormap combo box.

        Parameters
        ----------
        update_cbar : callable
            Slot invoked when the selected colormap changes.
        """

        self.cbar_combo.currentIndexChanged.connect(update_cbar)

    def connect_slice_width_line(self, update_slice):
        """
        Connect a callback to editing-finished on the slice width field.

        Parameters
        ----------
        update_slice : callable
            Slot invoked when editing of the slice width field finishes.
        """

        self.slice_width_line.editingFinished.connect(update_slice)

    def connect_slice_thickness_line(self, update_slice):
        """
        Connect a callback to editing-finished on the slice thickness field.

        Parameters
        ----------
        update_slice : callable
            Slot invoked when editing of the slice thickness field finishes.
        """

        self.slice_thickness_line.editingFinished.connect(update_slice)

    def connect_slice_line(self, update_slice):
        """
        Connect a callback to editing-finished on the slice position field.

        Parameters
        ----------
        update_slice : callable
            Slot invoked when editing of the slice position field finishes.
        """

        self.slice_line.editingFinished.connect(update_slice)

    def connect_slice_slider(self, reslice):
        """
        Register the callback invoked when the slice slider is released.

        Parameters
        ----------
        reslice : callable
            Slot invoked to redraw the slice view once slider dragging ends.
        """

        self._slice_reslice = reslice

    def _slice_slider_released(self):
        """
        Invoke the registered reslice callback when the slice slider is released.
        """

        if hasattr(self, "_slice_reslice"):
            self._slice_reslice()

    def eventFilter(self, obj, event):
        """
        Filter key press/release events for the slice canvas and slider.

        Handles left/right arrow key presses on the slice canvas by
        stepping the slice slider, and triggers a reslice when the
        arrow key is released.

        Parameters
        ----------
        obj : QObject
            The object that generated the event.
        event : QEvent
            The event being filtered.

        Returns
        -------
        handled : bool
            True if the event was handled and should be filtered out,
            otherwise the result of the base class event filter.
        """

        if event.type() == QEvent.KeyPress:
            if obj is self.canvas_slice and event.key() in (
                Qt.Key_Left,
                Qt.Key_Right,
            ):
                step = -1 if event.key() == Qt.Key_Left else 1
                self.slice_slider.setValue(
                    max(
                        self.slice_slider.minimum(),
                        min(
                            self.slice_slider.maximum(),
                            self.slice_slider.value() + step,
                        ),
                    )
                )
                return True
        elif event.type() == QEvent.KeyRelease:
            if event.key() in (Qt.Key_Left, Qt.Key_Right) and obj in (
                self.canvas_slice,
                self.slice_slider,
            ):
                if hasattr(self, "_slice_reslice"):
                    self._slice_reslice()
                return True
        return super().eventFilter(obj, event)

    @staticmethod
    def _nice_step(span, n_bins):
        """
        Compute a "nice" round step size for dividing a numeric span into bins.

        Rounds the raw step (span / n_bins) to the nearest value from
        {1, 2, 5, 10} times a power of ten, similar to common axis
        tick-spacing algorithms.

        Parameters
        ----------
        span : float
            The total numeric range to be divided.
        n_bins : int
            The desired approximate number of bins.

        Returns
        -------
        step : float
            A rounded step size close to span / n_bins.
        """

        raw = span / max(n_bins, 1)
        if raw <= 0:
            return 0.1
        exp = int(np.floor(np.log10(raw)))
        mult = raw / 10**exp
        if mult < 1.5:
            m = 1.0
        elif mult < 3.5:
            m = 2.0
        elif mult < 7.5:
            m = 5.0
        else:
            m = 10.0
        return m * 10**exp

    def _setup_slice_slider(self, smin, smax):
        """
        Configure the slice slider range and step size for a symmetric extent.

        Rescales the given bounds to a symmetric range about zero, picks
        a "nice" step size, and updates the slider's minimum/maximum and
        current position to match the current slice value.

        Parameters
        ----------
        smin : float
            Lower bound of the slice axis extent.
        smax : float
            Upper bound of the slice axis extent.
        """

        extent = max(abs(smin), abs(smax)) or 1.0
        smin, smax = -extent, extent
        span = smax - smin
        n_bins = round(span / 0.1) if span > 0 else 10
        step = self._nice_step(span, n_bins)
        n_min = int(np.floor(np.round(smin / step, 6)))
        n_max = int(np.ceil(np.round(smax / step, 6)))
        self._slice_step = step
        self._slice_smin = n_min * step
        self._slice_smax = n_max * step
        self._slice_steps = max(n_max - n_min, 1)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self._slice_steps)
        val = self.get_slice_value()
        if val is not None:
            pos = int(round((val - self._slice_smin) / step))
            self.slice_slider.setValue(max(0, min(self._slice_steps, pos)))
        self.slice_slider.blockSignals(False)

    def _on_slice_slider_changed(self, pos):
        """
        Update the slice position field when the slice slider value changes.

        Parameters
        ----------
        pos : int
            Current integer position of the slice slider.
        """

        val = self._slice_smin + pos * self._slice_step
        decimals = max(0, -int(np.floor(np.log10(self._slice_step))))
        self.slice_line.blockSignals(True)
        self.slice_line.setText(str(round(val, decimals)))
        self.slice_line.blockSignals(False)

    def connect_slice_scale_combo(self, update_slice):
        """
        Connect a callback to the slice view scale (linear/log) combo box.

        Parameters
        ----------
        update_slice : callable
            Slot invoked when the selected scale changes.
        """

        self.slice_scale_combo.currentIndexChanged.connect(update_slice)

    def connect_vmin_line(self, update_vals):
        """
        Connect a callback to editing-finished on the slice view minimum color value field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the V Min field finishes.
        """

        self.vmin_line.editingFinished.connect(update_vals)

    def connect_vmax_line(self, update_vals):
        """
        Connect a callback to editing-finished on the slice view maximum color value field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the V Max field finishes.
        """

        self.vmax_line.editingFinished.connect(update_vals)

    def connect_slice_auto_limits(self, update_limits):
        """
        Connect a callback to the slice view "Auto Limits" checkbox.

        Parameters
        ----------
        update_limits : callable
            Slot invoked when the auto-limits checkbox is toggled.
        """

        self.slice_auto_limits_box.toggled.connect(update_limits)

    def _reset_slice_to_zero(self):
        """
        Reset the slice position field to "0.0" without emitting signals.
        """

        self.slice_line.blockSignals(True)
        self.slice_line.setText("0.0")
        self.slice_line.blockSignals(False)

    def setup_slice_slider(self, z_min, z_max):
        """
        Configure the slice slider for the given axis extent.

        Parameters
        ----------
        z_min : float
            Lower bound of the slice axis extent.
        z_max : float
            Upper bound of the slice axis extent.
        """

        self._setup_slice_slider(z_min, z_max)

    def connect_slice_combo(self, update_slice, update_extent=None):
        """
        Connect callbacks to the slice axis-pair combo box.

        Resetting the slice position to zero is always connected first,
        followed optionally by an extent-update callback, and finally
        the slice redraw callback.

        Parameters
        ----------
        update_slice : callable
            Slot invoked to redraw the slice after the axis pair changes.
        update_extent : callable, optional
            Slot invoked to update the slice extent/slider range after
            the axis pair changes, by default None.
        """

        self.slice_combo.currentIndexChanged.connect(self._reset_slice_to_zero)
        if update_extent is not None:
            self.slice_combo.currentIndexChanged.connect(update_extent)
        self.slice_combo.currentIndexChanged.connect(update_slice)

    def connect_vlim_combo(self, update_clim):
        """
        Connect a callback to the detector view color-limit method combo box.

        Parameters
        ----------
        update_clim : callable
            Slot invoked when the selected color-limit method changes.
        """

        self.vlim_combo.currentIndexChanged.connect(update_clim)

    def connect_vbar_combo(self, update_cbar):
        """
        Connect a callback to the detector view colormap combo box.

        Parameters
        ----------
        update_cbar : callable
            Slot invoked when the selected colormap changes.
        """

        self.vbar_combo.currentIndexChanged.connect(update_cbar)

    def connect_instrument_scale_combo(self, update_view):
        """
        Connect a callback to the detector view scale (linear/log) combo box.

        Parameters
        ----------
        update_view : callable
            Slot invoked when the selected scale changes.
        """

        self.instrument_scale_combo.currentIndexChanged.connect(update_view)

    def connect_inst_vmin_line(self, update_vals):
        """
        Connect a callback to editing-finished on the detector view minimum color value field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the instrument V Min field finishes.
        """

        self.inst_vmin_line.editingFinished.connect(update_vals)

    def connect_inst_vmax_line(self, update_vals):
        """
        Connect a callback to editing-finished on the detector view maximum color value field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the instrument V Max field finishes.
        """

        self.inst_vmax_line.editingFinished.connect(update_vals)

    def connect_instrument_auto_limits(self, update_limits):
        """
        Connect a callback to the detector view "Auto Limits" checkbox.

        Parameters
        ----------
        update_limits : callable
            Slot invoked when the instrument auto-limits checkbox is toggled.
        """

        self.instrument_auto_limits_box.toggled.connect(update_limits)

    def set_Q_status(self, status):
        """
        Update the Q-sample status label text and color.

        Parameters
        ----------
        status : int
            Status code: 0 for no Q-sample, 1 for files missing, 2 for
            files existing and calculating, any other value for ready.
        """

        if status == 0:
            self.q_label.setText("No Q-sample")
            self.q_label.setStyleSheet("color: red;")
        elif status == 1:
            self.q_label.setText("Files do not exist")
            self.q_label.setStyleSheet("color: orange;")
        elif status == 2:
            self.q_label.setText("Files exist, calculating...")
            self.q_label.setStyleSheet("color: blue;")
        else:
            self.q_label.setText("Q-sample ready")
            self.q_label.setStyleSheet("color: green;")

    def set_peaks_status(self, status):
        """
        Update the peaks table status label text and color.

        Parameters
        ----------
        status : int
            Status code: 0 for no peaks table, 1 for peaks not indexed,
            2 for peaks indexed.
        """

        if status == 0:
            self.peaks_label.setText("No peaks table")
            self.peaks_label.setStyleSheet("color: red;")
        elif status == 1:
            self.peaks_label.setText("Peaks not indexed")
            self.peaks_label.setStyleSheet("color: blue;")
        elif status == 2:
            self.peaks_label.setText("Peaks indexed")
            self.peaks_label.setStyleSheet("color: green;")

    def set_UB_status(self, status):
        """
        Update the UB matrix status label text and color.

        Parameters
        ----------
        status : int
            Status code: 0 for no UB matrix, 1 for UB matrix available.
        """

        if status == 0:
            self.ub_label.setText("No UB matrix")
            self.ub_label.setStyleSheet("color: red;")
        elif status == 1:
            self.ub_label.setText("UB matrix available")
            self.ub_label.setStyleSheet("color: green;")

    def set_peak_goniometer_axes(self, names):
        """
        Set the goniometer axis name label in the peaks tab.

        Parameters
        ----------
        names : str
            Comma-separated goniometer axis names to display.
        """

        self.gonio_label.setText("{} [°]:".format(names))

    def set_instrument_goniometer_axes(self, names):
        """
        Set the goniometer axis name label in the detector view tab, if present.

        Parameters
        ----------
        names : str
            Comma-separated goniometer axis names to display.
        """

        if hasattr(self, "inst_gonio_label"):
            self.inst_gonio_label.setText("{} [°]:".format(names))

    def set_peak_goniometer_setting(self, angles):
        """
        Set the goniometer angle readout in the peaks tab.

        Parameters
        ----------
        angles : tuple of float or None
            The (omega, chi, phi) goniometer angles in degrees, or None
            to clear the field.
        """

        if angles is None:
            self.gonio_line.clear()
        else:
            self.gonio_line.setText("({:.2f}, {:.2f}, {:.2f})".format(*angles))

    def set_instrument_goniometer_setting(self, angles):
        """
        Set the goniometer angle readout in the detector view tab, if present.

        Parameters
        ----------
        angles : tuple of float or None
            The (omega, chi, phi) goniometer angles in degrees, or None
            to clear the field.
        """

        if hasattr(self, "inst_gonio_line"):
            if angles is None:
                self.inst_gonio_line.clear()
            else:
                self.inst_gonio_line.setText(
                    "({:.2f}, {:.2f}, {:.2f})".format(*angles)
                )

    def set_goniometer_axes(self, names):
        """
        Set the goniometer axis name labels in both the peaks and detector view tabs.

        Parameters
        ----------
        names : str
            Comma-separated goniometer axis names to display.
        """

        self.set_peak_goniometer_axes(names)
        self.set_instrument_goniometer_axes(names)

    def set_goniometer_setting(self, angles):
        """
        Set the goniometer angle readouts in both the peaks and detector view tabs.

        Parameters
        ----------
        angles : tuple of float or None
            The (omega, chi, phi) goniometer angles in degrees, or None
            to clear the fields.
        """

        self.set_peak_goniometer_setting(angles)
        self.set_instrument_goniometer_setting(angles)

    def update_slice_color(self):
        """
        Recompute and apply slice colorbar limits from the current color-bar slider values.
        """

        if self.cb_slice is not None:
            min_slider, max_slider = self.get_color_bar_values()

            vmin = self.vmin + (self.vmax - self.vmin) * min_slider / 100
            vmax = self.vmin + (self.vmax - self.vmin) * max_slider / 100

            self.update_colorbar_vlims(vmin, vmax)

    def update_colorbar_vlims(self, vmin, vmax):
        """
        Apply new color limits to the slice image and colorbar.

        Parameters
        ----------
        vmin : float
            Minimum color value.
        vmax : float
            Maximum color value.
        """

        if self.cb_slice is not None and self.slice_im is not None:
            self.set_vmin_value(vmin)
            self.set_vmax_value(vmax)

            self.slice_im.set_clim(vmin=vmin, vmax=vmax)
            self.cb_slice.update_normal(self.slice_im)
            self.cb_slice.minorticks_on()

            self.canvas_slice.draw_idle()

    def update_instrument_colorbar_vlims(self, vmin, vmax):
        """
        Apply new color limits to the detector view image.

        Parameters
        ----------
        vmin : float
            Minimum color value.
        vmax : float
            Maximum color value.
        """

        if self.inst_im is not None:
            self.set_inst_vmin_value(vmin)
            self.set_inst_vmax_value(vmax)

            self.inst_im.set_clim(vmin=vmin, vmax=vmax)

            self.canvas_inst.draw_idle()

    def _create_norm(self, scale, vmin, vmax):
        """
        Create a matplotlib color normalization for the given scale and limits.

        Parameters
        ----------
        scale : str
            Color scale name, "log" for logarithmic or any other value
            for linear normalization.
        vmin : float
            Minimum color value.
        vmax : float
            Maximum color value.

        Returns
        -------
        norm : matplotlib.colors.Normalize or matplotlib.colors.LogNorm
            The normalization object for the requested scale.
        """

        scale = scale.lower()

        if scale == "log":
            vmin = max(vmin, np.finfo(float).tiny)
            return mcolors.LogNorm(vmin=vmin, vmax=vmax)

        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def update_slice_display(self, cmap_key, scale, vmin, vmax):
        """
        Update the colormap, normalization, and colorbar of the slice plot.

        Parameters
        ----------
        cmap_key : str
            Key into the ``cmaps`` lookup for the colormap to apply.
        scale : str
            Color scale name, "log" for logarithmic or any other value
            for linear normalization.
        vmin : float
            Minimum color value.
        vmax : float
            Maximum color value.
        """

        if self.slice_im is None:
            return

        self.set_vmin_value(vmin)
        self.set_vmax_value(vmax)

        self.slice_im.set_cmap(cmaps[cmap_key])
        self.slice_im.set_norm(self._create_norm(scale, vmin, vmax))

        if self.cb_slice is not None:
            self.cb_slice.update_normal(self.slice_im)
            self.cb_slice.minorticks_on()

        self.canvas_slice.draw_idle()

    def update_instrument_display(self, cmap_key, scale, vmin, vmax):
        """
        Update the colormap and normalization of the instrument view plot.

        Parameters
        ----------
        cmap_key : str
            Key into the ``cmaps`` lookup for the colormap to apply.
        scale : str
            Color scale name, "log" for logarithmic or any other value
            for linear normalization.
        vmin : float
            Minimum color value.
        vmax : float
            Maximum color value.
        """

        if self.inst_im is None:
            return

        self.set_inst_vmin_value(vmin)
        self.set_inst_vmax_value(vmax)

        self.inst_im.set_cmap(cmaps[cmap_key])
        self.inst_im.set_norm(self._create_norm(scale, vmin, vmax))

        self.canvas_inst.draw_idle()

    def get_color_bar_values(self):
        """
        Return the default color bar percentile limits.

        Returns
        -------
        vmin, vmax : int
            Lower and upper percentile limits, fixed at 0 and 100.
        """

        return 0, 100

    def load_detector_cal_dialog(self, path=""):
        """
        Open a dialog to select a detector calibration file to load.

        Parameters
        ----------
        path : str, optional
            Initial directory or filename shown in the dialog.

        Returns
        -------
        filename : str
            Path to the selected calibration file, or an empty string
            if the dialog was cancelled.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Calibration files (*.DetCal *.detcal *.xml)"

        filename, _ = file_dialog.getOpenFileName(
            self, "Load calibration file", path, file_filters, options=options
        )

        return filename

    def load_tube_cal_dialog(self, path=""):
        """
        Open a dialog to select a tube calibration file to load.

        Parameters
        ----------
        path : str, optional
            Initial directory or filename shown in the dialog.

        Returns
        -------
        filename : str
            Path to the selected tube calibration file, or an empty
            string if the dialog was cancelled.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Tube files (*.h5 *.nxs)"

        filename, _ = file_dialog.getOpenFileName(
            self, "Load tube file", path, file_filters, options=options
        )

        return filename

    def load_Q_file_dialog(self, path=""):
        """
        Open a dialog to select a Q-space NeXus file to load.

        Parameters
        ----------
        path : str, optional
            Initial directory or filename shown in the dialog.

        Returns
        -------
        filename : str
            Path to the selected Q file, or an empty string if the
            dialog was cancelled.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getOpenFileName(
            self, "Load Q file", path, "Q files (*.nxs)", options=options
        )

        return filename

    def save_Q_file_dialog(self, path=""):
        """
        Open a dialog to choose a filename for saving Q-space data.

        Appends a ``.nxs`` extension if the chosen filename lacks one.

        Parameters
        ----------
        path : str, optional
            Initial directory or filename shown in the dialog.

        Returns
        -------
        filename : str
            Path to save the Q file to, or an empty string if the
            dialog was cancelled.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self, "Save Q file", path, "Q files (*.nxs)", options=options
        )

        if filename is not None:
            if not filename.endswith(".nxs"):
                filename += ".nxs"

        return filename

    def load_peaks_file_dialog(self, path=""):
        """
        Open a dialog to select a peaks file to load.

        Parameters
        ----------
        path : str, optional
            Initial directory or filename shown in the dialog.

        Returns
        -------
        filename : str
            Path to the selected peaks file, or an empty string if
            the dialog was cancelled.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load peaks file",
            path,
            "Peaks files (*.nxs *.peaks *.integrate)",
            options=options,
        )

        return filename

    def save_peaks_file_dialog(self, path=""):
        """
        Open a dialog to choose a filename for saving peaks data.

        Appends a ``.nxs`` extension if the chosen filename lacks one.

        Parameters
        ----------
        path : str, optional
            Initial directory or filename shown in the dialog.

        Returns
        -------
        filename : str
            Path to save the peaks file to, or an empty string if the
            dialog was cancelled.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self,
            "Save peaks file",
            path,
            "Peaks files (*.nxs *.peaks *.integrate)",
            options=options,
        )

        if filename is not None:
            if not filename.endswith(".nxs"):
                filename += ".nxs"

        return filename

    def load_UB_file_dialog(self, path=""):
        """
        Open a dialog to select a UB matrix file to load.

        Parameters
        ----------
        path : str, optional
            Initial directory or filename shown in the dialog.

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
            self, "Load UB file", path, "UB files (*.mat)", options=options
        )

        return filename

    def save_UB_file_dialog(self, path=""):
        """
        Open a dialog to choose a filename for saving a UB matrix.

        Appends a ``.mat`` extension if the chosen filename lacks one.

        Parameters
        ----------
        path : str, optional
            Initial directory or filename shown in the dialog.

        Returns
        -------
        filename : str
            Path to save the UB file to, or an empty string if the
            dialog was cancelled.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self, "Save UB file", path, "UB files (*.mat)", options=options
        )

        if filename is not None:
            if not filename.endswith(".mat"):
                filename += ".mat"

        return filename

    def save_mask_file_dialog(self, path=""):
        """
        Open a dialog to choose a filename for saving a detector mask.

        Appends a ``.xml`` extension if the chosen filename lacks one.

        Parameters
        ----------
        path : str, optional
            Initial directory or filename shown in the dialog.

        Returns
        -------
        filename : str
            Path to save the mask file to, or an empty string if the
            dialog was cancelled.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self,
            "Save detector mask file",
            path,
            "Mask files (*.xml)",
            options=options,
        )

        if filename is not None:
            if not filename.endswith(".xml"):
                filename += ".xml"

        return filename

    def set_data_list(self, values):
        """
        Populate the data selection combo box with numbered entries.

        Parameters
        ----------
        values : list or None
            Labels for each dataset. Each entry is shown prefixed
            with its 1-based row number. If None, the combo box is
            simply cleared.
        """

        self.data_combo.blockSignals(True)
        self.data_combo.clear()
        if values is not None:
            for row in range(len(values)):
                self.data_combo.addItem("{}: {}".format(row + 1, values[row]))
            self.data_combo.setCurrentIndex(0)
            self.auto_scale_dropdown(self.data_combo)
        self.data_combo.blockSignals(False)

    def get_data_list(self):
        """
        Get the currently selected dataset index.

        Returns
        -------
        index : int
            Zero-based index of the selected dataset, parsed from the
            "row: label" text of the combo box.
        """

        val = self.data_combo.currentText()
        if len(val) >= 1:
            return int(val.split(":")[0]) - 1

    def set_wavelength(self, wavelength):
        """
        Set the wavelength line edit(s) from a single value or a range.

        Parameters
        ----------
        wavelength : float or list
            A single monochromatic wavelength, or a two-element
            [min, max] range for a polychromatic (Laue/TOF) beam. When
            a list is given the maximum wavelength field is enabled;
            otherwise it is set equal to the minimum and disabled.
        """

        if type(wavelength) is list:
            self.wl_min_line.setText(str(wavelength[0]))
            self.wl_max_line.setText(str(wavelength[1]))
            self.wl_max_line.setEnabled(True)
        else:
            self.wl_min_line.setText(str(wavelength))
            self.wl_max_line.setText(str(wavelength))
            self.wl_max_line.setEnabled(False)

    def get_wavelength(self):
        """
        Get the wavelength range from the min/max line edits.

        Returns
        -------
        wavelength : list
            Two-element [min, max] wavelength in angstroms, or None
            if either field contains invalid input.
        """

        params = self.wl_min_line, self.wl_max_line

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return [float(param.text()) for param in params]

    def update_wavelength(self, lamda_min):
        """
        Mirror the minimum wavelength into the maximum field if locked.

        Parameters
        ----------
        lamda_min : float
            Minimum wavelength value to propagate.
        """

        if not self.wl_max_line.isEnabled():
            self.wl_max_line.setText(str(lamda_min))

    def get_instrument(self):
        """
        Get the currently selected instrument.

        Returns
        -------
        instrument : str
            Name of the selected instrument.
        """

        return self.instrument_combo.currentText()

    def set_instrument(self, instrument):
        """
        Select an instrument in the instrument combo box.

        Parameters
        ----------
        instrument : str
            Name of the instrument to select. No change is made if
            the name is not found.
        """

        idx = self.instrument_combo.findText(instrument)
        if idx >= 0:
            self.instrument_combo.setCurrentIndex(idx)

    def get_live(self):
        """
        Check whether the "Live" checkbox is checked.

        Returns
        -------
        bool
            True if live-data mode is selected.
        """

        return self.live_box.isChecked()

    def set_live_enabled(self, enabled):
        """
        Enable or disable the "Live" checkbox.

        Parameters
        ----------
        enabled : bool
            True to allow live-data mode (TOPAZ/CORELLI/MANDI only);
            False disables and unchecks it.
        """

        self.live_box.setEnabled(enabled)
        if not enabled:
            self.live_box.setChecked(False)

    def set_live_checked(self, checked):
        """
        Set the "Live" checkbox state without user interaction.

        Parameters
        ----------
        checked : bool
            New checkbox state.
        """

        self.live_box.setChecked(checked)

    def show_live_idle_warning(self, seconds):
        """
        Warn that live data has run unattended and offer to keep it going.

        Shown after many live-data ticks have landed with no user
        interaction. Blocks (modally) for up to `seconds`, updating the
        displayed countdown once a second, then closes on its own if
        the user hasn't responded.

        Parameters
        ----------
        seconds : int
            Countdown duration in seconds before auto-closing.

        Returns
        -------
        keep_going : bool
            True if the user clicked "Keep Live Data Running" before
            the countdown ran out; False if it ran out (or the user
            chose to stop).
        """

        box = QMessageBox(self)
        box.setWindowTitle("Live Data Still Running")
        box.setIcon(QMessageBox.Warning)
        keep_button = box.addButton(
            "Keep Live Data Running", QMessageBox.AcceptRole
        )
        box.addButton("Stop Live Data", QMessageBox.RejectRole)
        box.setDefaultButton(keep_button)

        remaining = [seconds]

        def message():
            return (
                "Live data has been running unattended.\n\n"
                "It will stop automatically in {} s unless you choose "
                "to keep it running.".format(remaining[0])
            )

        def tick():
            remaining[0] -= 1
            if remaining[0] <= 0:
                timer.stop()
                box.close()
            else:
                box.setText(message())

        box.setText(message())

        timer = QTimer(box)
        timer.timeout.connect(tick)
        timer.start(1000)

        box.exec_()
        timer.stop()

        return box.clickedButton() is keep_button

    def set_convert_button_text(self, text):
        """
        Set the label of the "Convert" button.

        Parameters
        ----------
        text : str
            New button text, e.g. "Convert", "Start Live", "Stop Live".
        """

        self.convert_to_q_button.setText(text)

    def set_run_entry_enabled(self, enabled):
        """
        Enable or disable the IPTS/run-number entry and reload button.

        Disabled while live-data mode is active, since live streaming
        does not use an explicit IPTS/run selection.

        Parameters
        ----------
        enabled : bool
            True to enable the fields, False to disable them.
        """

        self.ipts_line.setEnabled(enabled)
        self.runs_line.setEnabled(enabled)
        self.reload_convert_to_q_button.setEnabled(enabled)

    def update_diffraction_label(self, mono):
        """
        Update the label describing the diffraction axis units.

        Parameters
        ----------
        mono : bool
            True if the instrument is monochromatic, in which case
            the label reads "Angle:"; otherwise it reads "Spacing:".
        """

        text = "Spacing:" if not mono else "Angle:"

        self.diffraction_label.setText(text)

    def clear_run_info(self, filepath):
        """
        Reset run/calibration fields and enable widgets for a facility.

        Parameters
        ----------
        filepath : str
            Path or identifier used to detect the facility/instrument,
            e.g. containing "exp", "SNS", or "CORELLI". Determines
            which experiment number, time filter, and calibration
            controls are enabled.
        """

        self.exp_line.setText("")
        self.cal_line.setText("")
        self.tube_line.setText("")

        if "exp" in filepath:
            self.exp_line.setEnabled(True)
        else:
            self.exp_line.setEnabled(False)

        if "SNS" in filepath:
            self.filter_time_line.setEnabled(True)
            self.filter_time_line.setText("")
            self.cal_line.setEnabled(True)
            self.cal_browse_button.setEnabled(True)
            self.tube_line.setEnabled(False)
            self.tube_browse_button.setEnabled(False)
            self.reload_convert_to_q_button.setEnabled(True)
            if "CORELLI" in filepath:
                self.tube_line.setEnabled(True)
                self.tube_browse_button.setEnabled(True)
        else:
            self.filter_time_line.setEnabled(False)
            self.filter_time_line.setText("")
            self.cal_line.setEnabled(False)
            self.cal_browse_button.setEnabled(False)
            self.tube_line.setEnabled(False)
            self.tube_browse_button.setEnabled(False)
            self.reload_convert_to_q_button.setEnabled(False)

    def get_tube_calibration(self):
        """
        Get the tube calibration file path.

        Returns
        -------
        filename : str
            Path to the tube calibration file.
        """

        return self.tube_line.text()

    def get_detector_calibration(self):
        """
        Get the detector calibration file path.

        Returns
        -------
        filename : str
            Path to the detector calibration file.
        """

        return self.cal_line.text()

    def set_tube_calibration(self, filename):
        """
        Set the tube calibration file path.

        Parameters
        ----------
        filename : str
            Path to the tube calibration file.
        """

        return self.tube_line.setText(filename)

    def set_detector_calibration(self, filename):
        """
        Set the detector calibration file path.

        Parameters
        ----------
        filename : str
            Path to the detector calibration file.
        """

        return self.cal_line.setText(filename)

    def get_IPTS(self):
        """
        Get the IPTS proposal number.

        Returns
        -------
        ipts : str
            IPTS number, or None if the field contains invalid input.
        """

        if self.ipts_line.hasAcceptableInput():
            return self.ipts_line.text()

    def get_experiment(self):
        """
        Get the experiment number.

        Returns
        -------
        experiment : str
            Experiment number, or None if the field contains invalid
            input.
        """

        if self.exp_line.hasAcceptableInput():
            return self.exp_line.text()

    def runs_string_to_list(self, runs_str):
        """
        Convert runs string to list using regex validation.
        Return None for invalid formats.

        Parameters
        ----------
        runs_str : str
            Condensed notation for run numbers.

        Returns
        -------
        runs : list or None
            Integer run numbers or None if the input is invalid.

        """

        pattern = r"^(\d+(?::\d+(?:;\d+)?)?)(,\d+(?::\d+(?:;\d+)?)?)*$"
        if not re.match(pattern, runs_str):
            return None

        runs = []
        ranges = runs_str.split(",")

        for part in ranges:
            if ":" in part:
                range_part, *skip_part = part.split(";")
                start, end = map(int, range_part.split(":"))
                skip = int(skip_part[0]) if skip_part else 1

                if start > end or skip <= 0:
                    return None

                runs.extend(range(start, end + 1, skip))
            else:
                runs.append(int(part))

        return runs

    def get_runs(self):
        """
        Get the list of run numbers from the runs line edit.

        Returns
        -------
        runs : list or None
            Integer run numbers parsed from the condensed notation, or
            None if the input is invalid.
        """

        run_str = self.runs_line.text()

        return self.runs_string_to_list(run_str)

    def get_lorentz(self):
        """
        Get whether the Lorentz correction should be applied.

        Returns
        -------
        lorentz : bool
            True if the Lorentz correction checkbox is checked.
        """

        return self.lorentz_box.isChecked()

    def get_time_stop(self):
        """
        Get the time filter cutoff.

        Returns
        -------
        time_stop : str
            Time filter stop value, or None if the field contains
            invalid input.
        """

        if self.filter_time_line.hasAcceptableInput():
            return self.filter_time_line.text()

    def get_convert_min_d(self):
        """
        Get the minimum d-spacing used when converting to Q-space.

        Returns
        -------
        min_d : float
            Minimum d-spacing in angstroms, or None if the field
            contains invalid input.
        """

        if self.convert_min_d_line.hasAcceptableInput():
            return float(self.convert_min_d_line.text())

    def set_convert_min_d(self, min_d):
        """
        Set the minimum d-spacing used when converting to Q-space.

        Parameters
        ----------
        min_d : float
            Minimum d-spacing in angstroms.
        """

        self.convert_min_d_line.setText(str(min_d))

    def add_Q_viz(self, Q_dict):
        """
        Render the Q-space volume and, if available, indexed peaks.

        Clears the current scene, adds a volumetric rendering of the
        reciprocal-space signal along with axis lines, and if peak
        ellipsoid transforms are present, renders each peak as a
        sphere/ellipsoid colored by intensity (when integrated) or by
        HKL indexing status, with block picking enabled to allow
        selecting peaks interactively.

        Parameters
        ----------
        Q_dict : dict
            Dictionary describing the Q-space volume and peaks, with
            keys such as "signal", "spacing", "min_lim", "max_lim"
            for the volume, and "transforms", "intensities",
            "indexings", "numbers" for the peak ellipsoids.
        """

        self.clear_scene()

        signal = Q_dict.get("signal")
        spacing = Q_dict.get("spacing")
        min_lim = Q_dict.get("min_lim")
        max_lim = Q_dict.get("max_lim")

        if (
            signal is None
            or spacing is None
            or min_lim is None
            or max_lim is None
        ):
            return

        grid = pv.ImageData(
            spacing=spacing, dimensions=signal.shape, origin=min_lim
        )

        grid["scalars"] = signal.T.flatten()

        actor = self.plotter.add_volume(
            grid,
            opacity="linear",
            show_scalar_bar=False,
            cmap="binary",
            culling=True,
        )
        actor.prop.interpolation_type = "nearest"

        transforms = Q_dict.get("transforms")
        intensities = Q_dict.get("intensities")
        indexings = Q_dict.get("indexings")
        numbers = Q_dict.get("numbers")

        params = [transforms, intensities, indexings, numbers]

        integrate = np.any(intensities)

        mesh = pv.Line(
            pointa=(min_lim[0], 0, 0), pointb=(max_lim[0], 0, 0), resolution=1
        )

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        mesh = pv.Line(
            pointa=(0, min_lim[1], 0), pointb=(0, max_lim[1], 0), resolution=1
        )

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        mesh = pv.Line(
            pointa=(0, 0, min_lim[2]), pointb=(0, 0, max_lim[2]), resolution=1
        )

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        few = len(numbers) < 100
        if all([elem is not None for elem in params]) and len(numbers) > 0:
            if few:
                sphere = pv.Icosphere(radius=1, nsub=0)
            else:
                sphere = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))

            transforms = params[0]
            self._peak_centers = np.array([T[:3, 3] for T in transforms])

            geoms = []
            for i, (T, I, ind, no) in enumerate(zip(*params)):
                ellipsoid = sphere.copy(deep=False).transform(T, inplace=True)
                color = I if integrate else ind
                ellipsoid["scalars"] = np.full(sphere.n_cells, color)
                ellipsoid.point_data["peak_id"] = np.full(sphere.n_points, i)
                geoms.append(ellipsoid)

            # A single merged mesh instead of pv.MultiBlock + add_composite:
            # the composite mapper's per-block bookkeeping is what's slow
            # here (hundreds to thousands of peaks), not the geometry
            # itself -- merging avoids it while keeping full per-peak
            # ellipsoid shape/orientation and coloring.
            merged = pv.merge(geoms)

            mu = np.nanmean(intensities)
            sigma = np.nanstd(intensities)

            cmap = "viridis" if integrate else ["lightblue", "lightgreen"]
            n_colors = 16 if integrate else 2
            clim = [mu - 3 * sigma, mu + 3 * sigma] if integrate else [0, 1]

            self.plotter.add_mesh(
                merged,
                scalars="scalars",
                color=None,
                log_scale=False,
                style="wireframe" if few else "points",
                culling=True,
                cmap=cmap,
                clim=clim,
                n_colors=n_colors,
                point_size=10,
                render_points_as_spheres=True,
                show_scalar_bar=False,
                smooth_shading=False,
            )

            def get_picked_peak(*args, **kwargs):
                plotter = self.plotter
                x, y = plotter.mouse_position
                loc = plotter.iren.get_event_subplot_loc()
                index = plotter.renderers.loc_to_index(loc)
                renderer = plotter.renderers[index]

                picker = pv._vtk.vtkPointPicker()
                picker.Pick(x, y, 0, renderer)

                dataset = picker.GetDataSet()
                point_id = picker.GetPointId()
                if dataset is None or point_id < 0:
                    return

                peak_ids = dataset.GetPointData().GetArray("peak_id")
                if peak_ids is None:
                    return

                self.highlight(int(peak_ids.GetValue(point_id)))

            self.plotter.track_click_position(
                callback=get_picked_peak, viewport=True, side="left"
            )
            self.plotter.track_click_position(
                callback=get_picked_peak, viewport=True, side="right"
            )

            self.last_highlight = None
            if self._highlight_actor is not None:
                self.plotter.remove_actor(self._highlight_actor)
                self._highlight_actor = None

        self.reset_scene()

    def _clear_highlight_actor(self):
        """Remove any existing highlight actor from the scene."""

        if self._highlight_actor is not None:
            self.plotter.remove_actor(self._highlight_actor)
            self._highlight_actor = None

    def _set_highlight_actor(self, centers):
        """Add highlight markers at the given peak centers."""

        camera_pos = self.plotter.camera_position

        self._clear_highlight_actor()

        if len(centers) > 0:
            sphere = pv.PolyData(np.asarray(centers, dtype=float))
            actor = self.plotter.add_mesh(
                sphere,
                color="pink",
                style="points",
                point_size=10,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )
            self._highlight_actor = actor

        self.plotter.camera_position = camera_pos

    def highlight(self, ind):
        """Select a peak in the table from a 3D view click."""

        rows = self.peaks_table.rowCount()
        for row in range(rows):
            item = self.peaks_table.item(row, 7)
            if item is None:
                continue
            peak_no = item.text()
            if peak_no.isnumeric() and ind == int(peak_no) - 1:
                self.tab_widget.setCurrentWidget(self._peaks_tab)
                index_item = self.peaks_table.model().index(row, 0)
                self.peaks_table.selectionModel().select(
                    index_item,
                    QItemSelectionModel.ClearAndSelect
                    | QItemSelectionModel.Rows,
                )
                self.peaks_table.scrollTo(
                    index_item, QAbstractItemView.PositionAtCenter
                )
                break

    def highlight_peak(self, index):
        """Highlight a peak given its block index (from table/logic)."""

        self.highlight_peaks([index])

    def highlight_peaks(self, indices):
        """Highlight multiple peaks given their block indices."""

        centers = []
        for index in indices:
            if self._peak_centers is not None and index - 1 < len(
                self._peak_centers
            ):
                centers.append(self._peak_centers[index - 1])

        self._set_highlight_actor(centers)
        self.last_highlight = indices[-1] if len(indices) > 0 else None

    def clear_peak_selection(self):
        """Clear the peaks table selection and any highlight marker."""

        self.peaks_table.blockSignals(True)
        self.peaks_table.clearSelection()
        self.peaks_table.blockSignals(False)
        self._clear_highlight_actor()
        self.last_highlight = None

    def get_peaks(self):
        """
        Get the peak numbers currently selected in the peaks table.

        Returns
        -------
        peaks : list
            Sorted, unique zero-based peak indices for the selected
            rows.
        """

        peaks = []
        for model_index in self.peaks_table.selectionModel().selectedRows():
            item = self.peaks_table.item(model_index.row(), 7)
            if item is not None:
                peak_no = item.text()
                if peak_no.isnumeric():
                    peaks.append(int(peak_no) - 1)

        return sorted(set(peaks))

    def set_sample_directions(self, params):
        """
        Set the u, v, w sample orientation direction line edits.

        Parameters
        ----------
        params : tuple
            Three-element (v, w, u) sequence, each a length-3 vector
            of HKL components for the corresponding sample direction.
        """

        v, w, u = params

        self.uh_line.setText("{}".format(u[0]))
        self.uk_line.setText("{}".format(u[1]))
        self.ul_line.setText("{}".format(u[2]))

        self.vh_line.setText("{}".format(v[0]))
        self.vk_line.setText("{}".format(v[1]))
        self.vl_line.setText("{}".format(v[2]))

        self.wh_line.setText("{}".format(w[0]))
        self.wk_line.setText("{}".format(w[1]))
        self.wl_line.setText("{}".format(w[2]))

    def get_sample_directions(self):
        """
        Get the u and v sample orientation direction vectors.

        Returns
        -------
        params : list
            Six HKL components, [uh, uk, ul, vh, vk, vl], as floats,
            or None if any field contains invalid input.
        """

        params = (
            self.uh_line,
            self.uk_line,
            self.ul_line,
            self.vh_line,
            self.vk_line,
            self.vl_line,
        )

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return [float(param.text()) for param in params]

    def format_with_error(self, value, error):
        """
        Format a value with its uncertainty in compact parenthetical notation.

        The value and error are rounded to the decimal place implied
        by the error's order of magnitude, e.g. 3.14159 with error
        0.02 becomes "3.14(2)".

        Parameters
        ----------
        value : float
            Central value to format.
        error : float
            Uncertainty on the value. If not positive, the value is
            returned unformatted with no error digits.

        Returns
        -------
        formatted_str : str
            Value with the trailing error digit(s) in parentheses.
        """

        if error <= 0:
            return f"{value}"

        error_order = int(np.floor(np.log10(error)))

        decimal_places = max(0, -error_order)

        rounded_value = round(value, decimal_places)
        rounded_error = round(error, decimal_places)

        error_digits = int(round(rounded_error * (10**decimal_places)))

        formatted_str = f"{rounded_value:.{decimal_places}f}({error_digits})"
        return formatted_str

    def set_lattice_constants(self, params, errors):
        """
        Set the lattice constant fields, each formatted with its error.

        Parameters
        ----------
        params : list
            Six lattice parameters [a, b, c, alpha, beta, gamma].
        errors : list
            Uncertainties corresponding to each lattice parameter.
        """

        self.a_line.setText(self.format_with_error(params[0], errors[0]))
        self.b_line.setText(self.format_with_error(params[1], errors[1]))
        self.c_line.setText(self.format_with_error(params[2], errors[2]))

        self.alpha_line.setText(self.format_with_error(params[3], errors[3]))
        self.beta_line.setText(self.format_with_error(params[4], errors[4]))
        self.gamma_line.setText(self.format_with_error(params[5], errors[5]))

    def get_lattice_constants(self):
        """
        Get the six lattice constants from their line edits.

        Returns
        -------
        params : list
            [a, b, c, alpha, beta, gamma] as floats, parsed by
            stripping any trailing "(error)" notation, or None if any
            field contains invalid input.
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
            return [float(param.text().split("(")[0]) for param in params]

    def get_min_max_constants(self):
        """
        Get the minimum and maximum lattice constant search bounds.

        Returns
        -------
        params : list
            Two-element [min, max] lattice constant bound in
            angstroms, or None if either field contains invalid
            input.
        """

        params = self.min_const_line, self.max_const_line

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return [float(param.text()) for param in params]

    def get_find_peaks_parameters(self):
        """
        Get the density threshold and maximum peak count for peak finding.

        Returns
        -------
        params : list
            Two-element [density_threshold, max_peaks] as integers,
            or None if either field contains invalid input.
        """

        params = self.density_threshold_line, self.max_peaks_line

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return [int(param.text()) for param in params]

    def get_find_peaks_distance(self):
        """
        Get the minimum distance between found peaks.

        Returns
        -------
        distance : float
            Minimum peak separation, or None if the field contains
            invalid input.
        """

        param = self.min_distance_line

        if param.hasAcceptableInput():
            return float(param.text())

    def get_find_peaks_spacing(self):
        """
        Get the maximum d-spacing for peak finding.

        Returns
        -------
        spacing : float
            Maximum d-spacing in angstroms, or None if the field
            contains invalid input.
        """

        param = self.max_spacing_line

        if param.hasAcceptableInput():
            return float(param.text())

    def set_find_peaks_distance(self, val):
        """
        Set the minimum distance between found peaks.

        Parameters
        ----------
        val : float
            Minimum peak separation, displayed with two decimal
            places.
        """

        self.min_distance_line.setText("{:.2f}".format(val))

    def set_find_peaks_spacing(self, val):
        """
        Set the maximum d-spacing for peak finding.

        Parameters
        ----------
        val : float
            Maximum d-spacing in angstroms, displayed with two
            decimal places.
        """

        self.max_spacing_line.setText("{:.2f}".format(val))

    def get_find_peaks_edge(self):
        """
        Get the number of detector edge pixels to exclude during peak finding.

        Returns
        -------
        edge : int
            Number of edge pixels to exclude, or None if the field
            contains invalid input.
        """

        param = self.find_edge_line

        if param.hasAcceptableInput():
            return int(param.text())

    def get_peak_width(self):
        """
        Get the peak width parameter used for peak finding.

        Returns
        -------
        width : float
            Peak width, or None if the field contains invalid input.
        """

        param = self.peak_width_line

        if param.hasAcceptableInput():
            return float(param.text())

    def get_avoid_aluminum(self):
        """
        Get whether aluminum powder-ring regions should be excluded.

        Returns
        -------
        avoid : bool
            True if the aluminum exclusion checkbox is checked.
        """

        return self.aluminum_box.isChecked()

    def get_avoid_copper(self):
        """
        Get whether copper powder-ring regions should be excluded.

        Returns
        -------
        avoid : bool
            True if the copper exclusion checkbox is checked.
        """

        return self.copper_box.isChecked()

    def get_avoid_iron(self):
        """
        Get whether iron powder-ring regions should be excluded.

        Returns
        -------
        avoid : bool
            True if the iron exclusion checkbox is checked.
        """

        return self.iron_box.isChecked()

    def get_calculate_UB_tol(self):
        """
        Get the tolerance used when calculating the UB matrix.

        Returns
        -------
        tol : float
            HKL indexing tolerance, or None if the field contains
            invalid input.
        """

        param = self.calculate_tolerance_line

        if param.hasAcceptableInput():
            return float(param.text())

    def get_lattice_transform(self):
        """
        Get the selected lattice/Bravais system for symmetry transforms.

        Returns
        -------
        cell : str
            Name of the selected lattice system.
        """

        return self.lattice_combo.currentText()

    def get_symmetry_symbol(self):
        """
        Get the selected symmetry (point/space group) symbol.

        Returns
        -------
        symbol : str
            Selected symmetry symbol.
        """

        return self.symmetry_combo.currentText()

    def update_symmetry_symbols(self, symbols):
        """
        Repopulate the symmetry combo box with available symbols.

        Parameters
        ----------
        symbols : list
            Symmetry symbol strings to show, typically the keys of
            the lattice transform dictionary for the selected system.
        """

        self.symmetry_combo.clear()
        for symbol in symbols:
            self.symmetry_combo.addItem(symbol)
        self.auto_scale_dropdown(self.symmetry_combo)

    def get_transform_matrix(self):
        """
        Get the 3x3 lattice transformation matrix from its line edits.

        Returns
        -------
        params : list
            Nine matrix elements, row-major [T11, T12, T13, T21, T22,
            T23, T31, T32, T33], as floats, or None if any field
            contains invalid input.
        """

        params = (
            self.T11_line,
            self.T12_line,
            self.T13_line,
            self.T21_line,
            self.T22_line,
            self.T23_line,
            self.T31_line,
            self.T32_line,
            self.T33_line,
        )
        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            params = [float(param.text()) for param in params]

            return params

    def get_projection_matrix(self):
        """
        Get the 3x3 projection matrix from its line edits.

        Returns
        -------
        params : list
            Nine matrix elements, row-major [U1, U2, U3, V1, V2, V3,
            W1, W2, W3], as floats, or None if any field contains
            invalid input.
        """

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

    def set_transform_matrix(self, params):
        """
        Set the 3x3 lattice transformation matrix line edits.

        Parameters
        ----------
        params : array_like
            3x3 transformation matrix; values are displayed rounded
            to the nearest integer.
        """

        self.T11_line.setText("{:.0f}".format(params[0][0]))
        self.T12_line.setText("{:.0f}".format(params[0][1]))
        self.T13_line.setText("{:.0f}".format(params[0][2]))

        self.T21_line.setText("{:.0f}".format(params[1][0]))
        self.T22_line.setText("{:.0f}".format(params[1][1]))
        self.T23_line.setText("{:.0f}".format(params[1][2]))

        self.T31_line.setText("{:.0f}".format(params[2][0]))
        self.T32_line.setText("{:.0f}".format(params[2][1]))
        self.T33_line.setText("{:.0f}".format(params[2][2]))

    def get_transform_UB_tol(self):
        """
        Get the tolerance used when applying a UB transformation.

        Returns
        -------
        tol : float
            HKL indexing tolerance, or None if the field contains
            invalid input.
        """

        param = self.transform_tolerance_line

        if param.hasAcceptableInput():
            return float(param.text())

    def get_refine_UB_option(self):
        """
        Get the selected UB refinement constraint option.

        Returns
        -------
        option : str
            Name of the selected refinement option, e.g.
            "Unconstrained", "Constrained", or a Bravais lattice
            system such as "Cubic".
        """

        return self.optimize_combo.currentText()

    def update_refine_constraint_label(self, option):
        """
        Update the label describing the constraints of a refinement option.

        Parameters
        ----------
        option : str
            Selected UB refinement option, used to look up the
            corresponding constraint description.
        """

        labels = {
            "Unconstrained": "No lattice constraints.",
            "Constrained": "Fix current a, b, c, α, β, γ; refine orientation only.",
            "Triclinic": "No lattice constraints.",
            "Monoclinic": "α=γ=90",
            "Orthorhombic": "α=β=γ=90",
            "Tetragonal": "a=b, α=β=γ=90",
            "Rhombohedral": "a=b=c, α=β=γ",
            "Hexagonal": "a=b, α=β=90, γ=120",
            "Cubic": "a=b=c, α=β=γ=90",
        }

        text = labels.get(option, "")
        self.refine_constraint_label.setText(text)

    def get_refine_UB_tol(self):
        """
        Get the tolerance used when refining the UB matrix.

        Returns
        -------
        tol : float
            HKL indexing tolerance, or None if the field contains
            invalid input.
        """

        param = self.refine_tolerance_line

        if param.hasAcceptableInput():
            return float(param.text())

    def get_modulatation_offsets(self):
        """
        Get the three modulation vectors for satellite peak indexing.

        Returns
        -------
        params : list
            Nine components, [dh1, dk1, dl1, dh2, dk2, dl2, dh3, dk3,
            dl3], as floats describing up to three modulation
            vectors, or None if any field contains invalid input.
        """

        params = (
            self.dh1_line,
            self.dk1_line,
            self.dl1_line,
            self.dh2_line,
            self.dk2_line,
            self.dl2_line,
            self.dh3_line,
            self.dk3_line,
            self.dl3_line,
        )

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            params = [float(param.text()) for param in params]

            return params

    def set_modulatation_offsets(self, params):
        """
        Set the three modulation vector line edits for satellite peaks.

        Parameters
        ----------
        params : array_like
            Three modulation vectors, each a length-3 sequence of
            [dh, dk, dl] components, displayed to three decimal
            places.
        """

        self.dh1_line.setText("{:.3f}".format(params[0][0]))
        self.dk1_line.setText("{:.3f}".format(params[0][1]))
        self.dl1_line.setText("{:.3f}".format(params[0][2]))

        self.dh2_line.setText("{:.3f}".format(params[1][0]))
        self.dk2_line.setText("{:.3f}".format(params[1][1]))
        self.dl2_line.setText("{:.3f}".format(params[1][2]))

        self.dh3_line.setText("{:.3f}".format(params[2][0]))
        self.dk3_line.setText("{:.3f}".format(params[2][1]))
        self.dl3_line.setText("{:.3f}".format(params[2][2]))

    def get_max_order_cross_terms(self):
        """
        Get the maximum modulation order and whether cross terms are used.

        Returns
        -------
        max_order : int
            Maximum modulation order for satellite peak indexing.
        cross_terms : bool
            True if cross terms between modulation vectors should be
            included.

        Returns None if the maximum order field contains invalid
        input.
        """

        param = self.max_order_line

        if param.hasAcceptableInput():
            return int(param.text()), self.cross_box.isChecked()

    def get_max_scalar_error(self):
        """
        Get the maximum scalar error for Niggli cell determination.

        Returns
        -------
        error : float
            Maximum allowed scalar error, or None if the field
            contains invalid input.
        """

        param = self.max_scalar_error_line

        if param.hasAcceptableInput():
            return float(param.text())

    def get_form_number(self):
        """
        Get the selected Niggli cell form number.

        Returns
        -------
        form : int
            Form number of the selected conventional cell, or None if
            the field is empty.
        """

        form = self.form_line.text()

        if form != "":
            return int(form)

    def get_index_peaks_parameters(self):
        """
        Get the HKL indexing tolerances for fundamental and satellite peaks.

        Returns
        -------
        tol : float
            Indexing tolerance for fundamental peaks.
        sat_tol : float
            Indexing tolerance for satellite peaks. Falls back to
            ``tol`` if the satellite tolerance field is invalid.

        Returns None if the fundamental tolerance field contains
        invalid input.
        """

        param = self.index_tolerance_line

        sat_param = self.index_sat_tolerance_line

        if param.hasAcceptableInput():
            tol = float(param.text())

            if sat_param.hasAcceptableInput():
                sat_tol = float(sat_param.text())
            else:
                sat_tol = tol

            return tol, sat_tol

    def get_index_satellite_peaks(self):
        """
        Get whether satellite peak indexing is enabled.

        Returns
        -------
        index_sat : bool
            True if the satellite peak indexing checkbox is checked.
        """

        return self.index_sat_box.isChecked()

    def get_index_peaks_round(self):
        """
        Get whether indexed HKL values should be rounded to integers.

        Returns
        -------
        round_hkl : bool
            True if the round HKL checkbox is checked.
        """

        return self.round_box.isChecked()

    def get_predict_peaks_centering(self):
        """
        Get the selected lattice centering used for peak prediction.

        Returns
        -------
        centering : str
            Selected centering symbol.
        """

        return self.centering_combo.currentText()

    def get_predict_peaks_parameters(self):
        """
        Get the minimum d-spacing for predicting fundamental and satellite peaks.

        Returns
        -------
        d_min : float
            Minimum d-spacing for fundamental peaks.
        sat_d_min : float
            Minimum d-spacing for satellite peaks. Falls back to
            ``d_min`` if the satellite field is invalid.

        Returns None if the fundamental minimum d-spacing field
        contains invalid input.
        """

        param = self.min_d_line

        sat_param = self.min_sat_d_line

        if param.hasAcceptableInput():
            d_min = float(param.text())

            if sat_param.hasAcceptableInput():
                sat_d_min = float(sat_param.text())
            else:
                sat_d_min = d_min

            return d_min, sat_d_min

    def get_predict_peaks_edge(self):
        """
        Get the number of detector edge pixels to exclude during peak prediction.

        Returns
        -------
        edge : int
            Number of edge pixels to exclude, or None if the field
            contains invalid input.
        """

        param = self.predict_edge_line

        if param.hasAcceptableInput():
            return int(param.text())

    def get_predict_satellite_peaks(self):
        """
        Get whether satellite peak prediction is enabled.

        Returns
        -------
        predict_sat : bool
            True if the satellite peak prediction checkbox is
            checked.
        """

        return self.predict_sat_box.isChecked()

    def get_integrate_peaks_parameters(self):
        """
        Get the peak and background shell radii for integration.

        Returns
        -------
        params : list
            Three-element [radius, inner, outer] radii, or None if
            any field contains invalid input.
        """

        params = self.radius_line, self.inner_line, self.outer_line

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            params = [float(param.text()) for param in params]

            return params

    def get_centroid(self):
        """
        Get whether peak centroiding is enabled during integration.

        Returns
        -------
        centroid : bool
            True if the centroid checkbox is checked.
        """

        return self.centroid_box.isChecked()

    def get_ellipsoid(self):
        """
        Get whether adaptive ellipsoidal peak integration is enabled.

        Returns
        -------
        ellipsoid : bool
            True if the adaptive ellipsoid checkbox is checked.
        """

        return self.adaptive_box.isChecked()

    def get_filter_variable(self):
        """
        Get the selected peak property to filter by.

        Returns
        -------
        variable : str
            Selected filter variable, e.g. "I/sigma", "d", "Q".
        """

        return self.filter_combo.currentText()

    def update_filter_description_label(self, option):
        """
        Update the label describing the selected peak filter variable.

        Parameters
        ----------
        option : str
            Selected filter variable, used to look up its description.
        """

        descriptions = {
            "I/σ": "Signal-to-noise ratio of each peak.",
            "d": "Peak d-spacing in angstroms.",
            "λ": "Peak wavelength in angstroms.",
            "Q": "Magnitude of the scattering vector.",
            "h^2+k^2+l^2": "Squared magnitude of the integer HKL index.",
            "m^2+n^2+p^2": "Squared magnitude of the satellite MNP index.",
            "Run #": "Run number associated with each peak.",
        }

        text = descriptions.get(option, "")
        self.filter_description_label.setText(text)

    def get_filter_comparison(self):
        """
        Get the selected comparison operator for peak filtering.

        Returns
        -------
        comparison : str
            Selected comparison operator, e.g. ">", "<", "=".
        """

        return self.comparison_combo.currentText()

    def get_filter_value(self):
        """
        Get the threshold value for peak filtering.

        Returns
        -------
        value : float
            Filter threshold value, or None if the field contains
            invalid input.
        """

        param = self.filter_line

        if param.hasAcceptableInput():
            return float(param.text())

    def update_peaks_table(self, peaks):
        """
        Repopulate the peaks table from a list of peak dictionaries.

        Clears the current selection and highlight, resizes the table to
        match `peaks`, fills each row via `set_peak`, and updates the
        indexed/total peak count labels.

        Parameters
        ----------
        peaks : list of dict
            Peak information dictionaries, one per row, as produced by
            the model (see `set_peak`).
        """
        self.peaks_table.blockSignals(True)
        self.peaks_table.setSortingEnabled(False)
        self.peaks_table.clearSelection()
        self._clear_highlight_actor()
        self.last_highlight = None
        self.peaks_table.setRowCount(0)
        self.peaks_table.setRowCount(len(peaks))

        ind, tot = 0, 0
        for row, peak in enumerate(peaks):
            self.set_peak(row, peak)
            ind += peak["ind"]
            tot += 1

        self.index_line.setText("{}".format(ind))
        self.total_line.setText("{}".format(tot))

        self.peaks_table.blockSignals(False)
        self.peaks_table.setSortingEnabled(True)

    def set_peak(self, row, peak):
        """
        Populate one row of the peaks table with a peak's summary values.

        Parameters
        ----------
        row : int
            Row index in `peaks_table` to fill.
        peak : dict
            Peak information dictionary containing ``"hkl"``,
            ``"d_spacing"``, ``"wavelength"``, ``"intensity"``,
            ``"signal_to_noise"``, and ``"peak_no"`` entries.
        """
        hkl = peak["hkl"]
        d = peak["d_spacing"]
        lamda = peak["wavelength"]
        intens = peak["intensity"]
        signal_to_noise = peak["signal_to_noise"]
        peak_no = peak["peak_no"]
        h, k, l = hkl
        h = "{:.3f}".format(h)
        k = "{:.3f}".format(k)
        l = "{:.3f}".format(l)
        d = "{:.4f}".format(d)
        lamda = "{:.4f}".format(lamda)
        intens = "{:.2e}".format(intens)
        signal_to_noise = "{:.2f}".format(signal_to_noise)
        peak_no = str(peak_no + 1)
        self.peaks_table.setItem(row, 0, self.set_item_value(h))
        self.peaks_table.setItem(row, 1, self.set_item_value(k))
        self.peaks_table.setItem(row, 2, self.set_item_value(l))
        self.peaks_table.setItem(row, 3, self.set_item_value(d))
        self.peaks_table.setItem(row, 4, self.set_item_value(lamda))
        self.peaks_table.setItem(row, 5, self.set_item_value(intens))
        self.peaks_table.setItem(row, 6, self.set_item_value(signal_to_noise))
        self.peaks_table.setItem(row, 7, self.set_item_value(peak_no))

    def set_item_value(self, value):
        """
        Create a numeric, sortable table item.

        Parameters
        ----------
        value : str or float
            Value to store, converted to float so the table sorts the
            column numerically instead of lexicographically.

        Returns
        -------
        item : QTableWidgetItem
            Table item with `value` stored via `Qt.DisplayRole`.
        """
        item = QTableWidgetItem()
        item.setData(Qt.DisplayRole, float(value))
        return item

    def clear_niggli_info(self):
        """
        Clear the reduced-cell table and the selected form label.
        """
        self.cell_table.clearSelection()
        self.cell_table.setRowCount(0)
        self.form_line.setText("")

    def update_cell_table(self, cells):
        """
        Repopulate the reduced-cell table from a list of candidate cells.

        Parameters
        ----------
        cells : list
            Candidate reduced cells, one per row, as produced by the
            model (see `set_cell`).
        """
        self.cell_table.clearSelection()
        self.cell_table.setRowCount(0)
        self.cell_table.setRowCount(len(cells))

        for row, cell in enumerate(cells):
            self.set_cell(row, cell)

    def set_cell(self, row, cell):
        """
        Populate one row of the reduced-cell table with a candidate cell.

        Parameters
        ----------
        row : int
            Row index in `cell_table` to fill.
        cell : tuple
            ``(form, error, bl, params)`` where `form` is the Niggli
            form number, `error` is the fit error, `bl` is a sequence of
            Bravais lattice symbols, and `params` is
            ``(a, b, c, alpha, beta, gamma, vol)``.
        """
        form, error, bl, params = cell
        a, b, c, alpha, beta, gamma, vol = params
        error = "{:.4f}".format(error)
        bravais = " ".join(bl)
        a = "{:.2f}".format(a)
        b = "{:.2f}".format(b)
        c = "{:.2f}".format(c)
        alpha = "{:.1f}".format(alpha)
        beta = "{:.1f}".format(beta)
        gamma = "{:.1f}".format(gamma)
        vol = "{:.0f}".format(vol)
        self.cell_table.setVerticalHeaderItem(row, QTableWidgetItem(str(form)))
        self.cell_table.setItem(row, 0, QTableWidgetItem(error))
        self.cell_table.setItem(row, 1, QTableWidgetItem(bravais))
        self.cell_table.setItem(row, 2, QTableWidgetItem(a))
        self.cell_table.setItem(row, 3, QTableWidgetItem(b))
        self.cell_table.setItem(row, 4, QTableWidgetItem(c))
        self.cell_table.setItem(row, 5, QTableWidgetItem(alpha))
        self.cell_table.setItem(row, 6, QTableWidgetItem(beta))
        self.cell_table.setItem(row, 7, QTableWidgetItem(gamma))
        self.cell_table.setItem(row, 8, QTableWidgetItem(vol))

    def get_form(self):
        """
        Get the Niggli form number of the selected reduced-cell row.

        Returns
        -------
        form : int
            Form number shown in the vertical header of the currently
            selected `cell_table` row, or None if no row is selected.
        """
        row = self.cell_table.currentRow()
        if row is not None:
            item = int(self.cell_table.verticalHeaderItem(row).text())
            return item

    def set_cell_form(self, form):
        """
        Set the form-number input field.

        Parameters
        ----------
        form : int
            Niggli cell form number to display.
        """
        self.form_line.setText(str(form))

    def get_peak(self):
        """
        Get the index of the peak selected in the peaks table.

        Returns
        -------
        index : int
            Zero-based peak index corresponding to the selected row, or
            None if no row is selected or the peak number is missing.
        """
        row = self.peaks_table.currentRow()
        if row >= 0:
            peak_no = self.peaks_table.item(row, 7)
            if peak_no is not None:
                peak_no = peak_no.text()
                if peak_no.isnumeric():
                    return int(peak_no) - 1

    def set_peak_info(self, peak):
        """
        Populate the peak-detail fields with a single peak's information.

        Parameters
        ----------
        peak : dict
            Peak information dictionary containing ``"hkl"``,
            ``"d_spacing"``, ``"wavelength"``, ``"intensity"``,
            ``"sigma"``, ``"int_hkl"``, ``"int_mnp"``, ``"run_number"``,
            ``"bank"``, ``"row"``, ``"col"``, ``"two_theta"``,
            ``"gamma"``, ``"nu"``, and ``"angles"`` entries.
        """
        hkl = peak["hkl"]
        d = peak["d_spacing"]
        lamda = peak["wavelength"]
        intens = peak["intensity"]
        sigma = peak["sigma"]
        int_hkl = peak["int_hkl"]
        int_mnp = peak["int_mnp"]
        run = peak["run_number"]
        bank = peak["bank"]
        row = peak["row"]
        col = peak["col"]
        two_theta = peak["two_theta"]
        gamma = peak["gamma"]
        nu = peak["nu"]
        gonio = peak["angles"]

        self.set_indices(hkl, int_hkl, int_mnp)

        self.intensity_line.setText("{:.2e}".format(intens))
        self.sigma_line.setText("{:.2e}".format(sigma))

        self.lambda_line.setText("{:.4f}".format(lamda))
        self.d_line.setText("{:.4f}".format(d))

        self.run_line.setText(str(run))
        self.bank_line.setText(str(bank))
        self.row_line.setText(str(row))
        self.col_line.setText(str(col))

        self.two_theta_line.setText("{:.2f}".format(two_theta))
        self.horz_vert_line.setText("({:.1f}, {:.1f})".format(gamma, nu))
        self.set_peak_goniometer_setting(gonio)

    def update_table_index(self, ind, hkl):
        """
        Update the HKL columns of a single peaks-table row after reindexing.

        Parameters
        ----------
        ind : int
            Zero-based peak index whose row should be updated.
        hkl : tuple
            Updated ``(h, k, l)`` Miller indices to display.
        """
        rows = self.peaks_table.rowCount()
        for row in range(rows):
            peak_no = self.peaks_table.item(row, 7).text()
            if peak_no.isnumeric():
                if ind == int(peak_no) - 1:
                    h, k, l = hkl
                    h = "{:.3f}".format(h)
                    k = "{:.3f}".format(k)
                    l = "{:.3f}".format(l)
                    self.peaks_table.setItem(row, 0, QTableWidgetItem(h))
                    self.peaks_table.setItem(row, 1, QTableWidgetItem(k))
                    self.peaks_table.setItem(row, 2, QTableWidgetItem(l))

    def set_indices(self, hkl, int_hkl, int_mnp):
        """
        Set the fractional and integer HKL/MNP index fields.

        Signals on the index fields are blocked while updating so this
        does not trigger reindexing callbacks.

        Parameters
        ----------
        hkl : tuple
            Fractional ``(H, K, L)`` Miller indices.
        int_hkl : tuple
            Integer ``(h, k, l)`` Miller indices.
        int_mnp : tuple
            Integer ``(m, n, p)`` satellite indices.
        """
        H, K, L = hkl

        h, k, l = int_hkl
        m, n, p = int_mnp

        self.h_line.blockSignals(True)
        self.k_line.blockSignals(True)
        self.l_line.blockSignals(True)

        self.int_h_line.blockSignals(True)
        self.int_k_line.blockSignals(True)
        self.int_l_line.blockSignals(True)

        self.int_m_line.blockSignals(True)
        self.int_n_line.blockSignals(True)
        self.int_p_line.blockSignals(True)

        self.h_line.setText("{:.3f}".format(H))
        self.k_line.setText("{:.3f}".format(K))
        self.l_line.setText("{:.3f}".format(L))

        self.int_h_line.setText("{:.0f}".format(h))
        self.int_k_line.setText("{:.0f}".format(k))
        self.int_l_line.setText("{:.0f}".format(l))

        self.int_m_line.setText("{:.0f}".format(m))
        self.int_n_line.setText("{:.0f}".format(n))
        self.int_p_line.setText("{:.0f}".format(p))

        self.h_line.blockSignals(False)
        self.k_line.blockSignals(False)
        self.l_line.blockSignals(False)

        self.int_h_line.blockSignals(False)
        self.int_k_line.blockSignals(False)
        self.int_l_line.blockSignals(False)

        self.int_m_line.blockSignals(False)
        self.int_n_line.blockSignals(False)
        self.int_p_line.blockSignals(False)

    def get_indices(self):
        """
        Get the fractional and integer HKL/MNP indices from the index fields.

        Returns
        -------
        hkl : list
            Fractional ``[H, K, L]`` Miller indices.
        int_hkl : list
            Integer ``[h, k, l]`` Miller indices.
        int_mnp : list
            Integer ``[m, n, p]`` satellite indices.

        Returns None instead if any index field contains invalid input.
        """
        params_hkl = self.h_line, self.k_line, self.l_line
        params_int_hkl = self.int_h_line, self.int_k_line, self.int_l_line
        params_int_mnp = self.int_m_line, self.int_n_line, self.int_p_line

        params = params_hkl + params_int_hkl + params_int_mnp

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            hkl = [float(param.text()) for param in params_hkl]
            int_hkl = [int(param.text()) for param in params_int_hkl]
            int_mnp = [int(param.text()) for param in params_int_mnp]
            return hkl, int_hkl, int_mnp

    def connect_hand_index_peak(self, reindex):
        """
        Connect a callback to the `index_ready` signal.

        Parameters
        ----------
        reindex : callable
            Slot invoked when `index_ready` is emitted, i.e. when a peak
            has been reindexed by hand.
        """
        self.index_ready.connect(reindex)

    def get_input_hkls(self):
        """
        Get the two manually entered HKL triples used for angle/d-spacing
        calculations.

        Returns
        -------
        params_1 : list or None
            ``[h, k, l]`` for the first plane, or None if the fields
            contain invalid input.
        params_2 : list or None
            ``[h, k, l]`` for the second plane, or None if the fields
            contain invalid input.
        """
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

    def set_d_phi(self, d_1, d_2, phi_12):
        """
        Set the calculated d-spacing and angle fields for the two input HKLs.

        Parameters
        ----------
        d_1 : float or None
            D-spacing of the first plane, or None to clear the field.
        d_2 : float or None
            D-spacing of the second plane, or None to clear the field.
        phi_12 : float or None
            Angle between the two planes in degrees, or None to clear
            the field.
        """
        if d_1 is not None:
            self.d1_line.setText("{:.4f}".format(d_1))
        else:
            self.d1_line.setText("")
        if d_2 is not None:
            self.d2_line.setText("{:.4f}".format(d_2))
        else:
            self.d2_line.setText("")
        if phi_12 is not None:
            self.phi_line.setText("{:.4f}".format(phi_12))
        else:
            self.phi_line.setText("")

    def get_highlight(self):
        """
        Get the Q vectors of the two highlighted peaks.

        Returns
        -------
        peak_1 : list of float
            Q vector ``[Qx, Qy, Qz]`` of the first highlighted peak.
        peak_2 : list of float
            Q vector ``[Qx, Qy, Qz]`` of the second highlighted peak.

        Returns None instead if either highlighted-peak field does not
        contain exactly three comma-separated values.
        """
        peak_1 = self.highlight_1_line.text().split(",")
        peak_2 = self.highlight_2_line.text().split(",")

        if len(peak_1) == 3 and len(peak_2) == 3:
            peak_1 = [float(val) for val in peak_1]
            peak_2 = [float(val) for val in peak_2]

            return peak_1, peak_2

    def add_highlight_1(self, peak):
        """
        Display the first highlighted peak's Q vector and d-spacing.

        Parameters
        ----------
        peak : dict
            Peak information dictionary containing ``"Q"`` and
            ``"d_spacing"`` entries.
        """
        Q = peak["Q"]
        d = peak["d_spacing"]
        self.highlight_1_line.setText("{:.4f}, {:.4f}, {:.4f}".format(*Q))
        self.highlight_d1_line.setText("{:.4f}".format(d))

    def add_highlight_2(self, peak):
        """
        Display the second highlighted peak's Q vector and d-spacing.

        Parameters
        ----------
        peak : dict
            Peak information dictionary containing ``"Q"`` and
            ``"d_spacing"`` entries.
        """
        Q = peak["Q"]
        d = peak["d_spacing"]
        self.highlight_2_line.setText("{:.4f}, {:.4f}, {:.4f}".format(*Q))
        self.highlight_d2_line.setText("{:.4f}".format(d))

    def set_highlight_phi(self, phi):
        """
        Set the calculated angle between the two highlighted peaks.

        Parameters
        ----------
        phi : float
            Angle between the two highlighted peaks' Q vectors, in
            degrees.
        """
        self.highlight_phi_line.setText("{:.4f}".format(phi))

    def get_diffraction(self):
        """
        Get the diffraction coordinate for the current detector position.

        Returns
        -------
        val : float
            Diffraction value (e.g. time-of-flight or wavelength), or
            None if the field contains invalid input.
        """
        if self.diffraction_line.hasAcceptableInput():
            return float(self.diffraction_line.text())

    def set_diffraction(self, val):
        """
        Set the diffraction coordinate for the current detector position.

        Parameters
        ----------
        val : float
            Diffraction value (e.g. time-of-flight or wavelength).
        """
        self.diffraction_line.setText(str(round(val, 3)))

    def get_d_min(self):
        """
        Get the minimum d-spacing for the instrument/ROI view.

        Returns
        -------
        val : float
            Minimum d-spacing in angstroms, or None if the field
            contains invalid input.
        """
        if self.d_min_line.hasAcceptableInput():
            return float(self.d_min_line.text())

    def set_d_min(self, val):
        """
        Set the minimum d-spacing for the instrument/ROI view.

        Parameters
        ----------
        val : float
            Minimum d-spacing in angstroms.
        """
        self.d_min_line.setText(str(round(val, 1)))

    def get_d_max(self):
        """
        Get the maximum d-spacing for the instrument/ROI view.

        Returns
        -------
        val : float
            Maximum d-spacing in angstroms (may be ``inf``), or None if
            the field contains invalid input.
        """
        text = self.d_max_line.text()

        if self.d_max_line.hasAcceptableInput() or text == "inf":
            return float(text)

    def get_horizontal(self):
        """
        Get the horizontal detector coordinate.

        Returns
        -------
        val : float
            Horizontal detector position, or None if the field contains
            invalid input.
        """
        if self.horizontal_line.hasAcceptableInput():
            return float(self.horizontal_line.text())

    def get_vertical(self):
        """
        Get the vertical detector coordinate.

        Returns
        -------
        val : float
            Vertical detector position, or None if the field contains
            invalid input.
        """
        if self.vertical_line.hasAcceptableInput():
            return float(self.vertical_line.text())

    def set_horizontal(self, val):
        """
        Set the horizontal detector coordinate.

        Parameters
        ----------
        val : float
            Horizontal detector position.
        """
        self.horizontal_line.setText(str(round(val, 2)))

    def set_vertical(self, val):
        """
        Set the vertical detector coordinate.

        Parameters
        ----------
        val : float
            Vertical detector position.
        """
        self.vertical_line.setText(str(round(val, 2)))

    def get_horizontal_roi(self):
        """
        Get the horizontal region-of-interest half-width.

        Returns
        -------
        val : float
            Horizontal ROI half-width, or None if the field contains
            invalid input.
        """
        if self.horizontal_roi_line.hasAcceptableInput():
            return float(self.horizontal_roi_line.text())

    def get_vertical_roi(
        self,
    ):
        """
        Get the vertical region-of-interest half-width.

        Returns
        -------
        val : float
            Vertical ROI half-width, or None if the field contains
            invalid input.
        """
        if self.vertical_roi_line.hasAcceptableInput():
            return float(self.vertical_roi_line.text())

    def get_check_hkl(self):
        """
        Get the hkl to check the detector position for.

        Returns
        -------
        hkl : list of float
            ``[h, k, l]`` Miller indices, or None if any field contains
            invalid input.
        """
        params = self.check_h_line, self.check_k_line, self.check_l_line

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            hkl = [float(param.text()) for param in params]
            return hkl

    def set_check_hkl(self, h, k, l):
        """
        Set the hkl to check the detector position for.

        Parameters
        ----------
        h, k, l : float
            Miller indices to display in the check-hkl fields.
        """
        self.check_h_line.setText(str(round(h, 4)))
        self.check_k_line.setText(str(round(k, 4)))
        self.check_l_line.setText(str(round(l, 4)))

    def update_instrument_view(self, inst_view):
        """
        Redraw the instrument detector (gamma/nu) image.

        Preserves the current zoom unless instrument auto-zoom is
        enabled. Updates the instrument color-limit fields, clears and
        redraws the instrument axes with the new image, and reconnects
        the cursor coordinate formatter.

        Parameters
        ----------
        inst_view : dict
            Instrument view data containing ``"img"``, ``"xedges"``,
            ``"yedges"``, ``"vmin"``, and ``"vmax"`` entries.
        """
        prev_xlim = (
            self.ax_inst.get_xlim() if self.inst_im is not None else None
        )
        prev_ylim = (
            self.ax_inst.get_ylim() if self.inst_im is not None else None
        )

        img = inst_view["img"]
        xedges = inst_view["xedges"]
        yedges = inst_view["yedges"]
        vmin = inst_view["vmin"]
        vmax = inst_view["vmax"]

        self.inst_vmin_line.blockSignals(True)
        self.inst_vmax_line.blockSignals(True)
        self.set_inst_vmin_value(vmin)
        self.set_inst_vmax_value(vmax)
        self.inst_vmin_line.blockSignals(False)
        self.inst_vmax_line.blockSignals(False)

        if self.cb_inst is not None:
            self.cb_inst.remove()
            self.cb_inst = None

        self.ax_inst.clear()
        self.ax_inst.invert_xaxis()

        scale = self.get_instrument_scale()
        cmap = cmaps[self.get_instrument_colormap()]

        self.inst_im = self.ax_inst.pcolormesh(
            xedges,
            yedges,
            img.T,
            shading="flat",
            norm=self._create_norm(scale, vmin, vmax),
            cmap=cmap,
            rasterized=True,
        )

        self.ax_inst.set_aspect(1)
        self.ax_inst.minorticks_on()

        self.ax_inst.set_xlabel(r"$\gamma$")
        self.ax_inst.set_ylabel(r"$\nu$")

        fmt_str_form = FormatStrFormatter(r"$%d^\circ$")

        self.ax_inst.xaxis.set_major_formatter(fmt_str_form)
        self.ax_inst.yaxis.set_major_formatter(fmt_str_form)

        # self.cb_inst = self.fig_inst.colorbar(self.im, ax=self.ax_inst)
        # self.cb_inst.minorticks_on()

        if (
            not self.get_instrument_auto_zoom()
            and prev_xlim is not None
            and prev_ylim is not None
        ):
            self.ax_inst.set_xlim(prev_xlim)
            self.ax_inst.set_ylim(prev_ylim)

        self.canvas_inst.draw_idle()

        self.ax_inst.format_coord = self.__format_inst_coord

        self._last_inst_view = inst_view

    def update_roi_view(self, roi_view):
        """
        Draw or update the region-of-interest box on the instrument view.

        Reuses existing ROI guide lines if present, otherwise creates
        them, then reconnects the instrument click handler.

        Parameters
        ----------
        roi_view : dict
            ROI position data containing ``"horz"``, ``"vert"``,
            ``"horz_roi"``, and ``"vert_roi"`` entries.
        """
        horz = roi_view["horz"]
        vert = roi_view["vert"]
        horz_roi = roi_view["horz_roi"]
        vert_roi = roi_view["vert_roi"]

        if (
            self._roi_lines is not None
            and self._roi_lines[0] in self.ax_inst.lines
        ):
            vl1, vl2, hl1, hl2 = self._roi_lines
            vl1.set_xdata([horz - horz_roi, horz - horz_roi])
            vl2.set_xdata([horz + horz_roi, horz + horz_roi])
            hl1.set_ydata([vert - vert_roi, vert - vert_roi])
            hl2.set_ydata([vert + vert_roi, vert + vert_roi])
        else:
            vl1 = self.ax_inst.axvline(
                x=horz - horz_roi, color="k", linestyle="--"
            )
            vl2 = self.ax_inst.axvline(
                x=horz + horz_roi, color="k", linestyle="--"
            )
            hl1 = self.ax_inst.axhline(
                y=vert - vert_roi, color="k", linestyle="--"
            )
            hl2 = self.ax_inst.axhline(
                y=vert + vert_roi, color="k", linestyle="--"
            )
            self._roi_lines = [vl1, vl2, hl1, hl2]

        self.canvas_inst.draw_idle()

        self.inst_roi = {"roi": (horz_roi, vert_roi)}

        if self._inst_click_cid is not None:
            self.fig_inst.canvas.mpl_disconnect(self._inst_click_cid)
        self._inst_click_cid = self.fig_inst.canvas.mpl_connect(
            "button_press_event", self.on_press_inst
        )

    def update_scan_view(self, roi_view):
        """
        Redraw the 1D diffraction scan (ROI-integrated intensity) plot.

        Reconnects the scan click handler after drawing so the user can
        pick a new diffraction position.

        Parameters
        ----------
        roi_view : dict
            Scan data containing ``"x"`` (scan coordinate), ``"y"``
            (integrated intensity), ``"val"`` (current diffraction
            value), and ``"label"`` (``"d"`` for d-spacing or otherwise
            two-theta) entries.
        """
        x = roi_view["x"]
        y = roi_view["y"]
        val = roi_view["val"]
        label = roi_view["label"]

        self.ax_scan.clear()

        self.ax_scan.errorbar(x, y, yerr=np.sqrt(y), fmt=".", color="C0")
        self.ax_scan.plot(x, y, color="C1")
        # self.ax_scan.set_yscale('log')
        self.line_scan = self.ax_scan.axvline(x=val, color="k", linestyle="--")
        self.ax_scan.minorticks_on()

        if label == "d":
            xlabel = r"$d$ [Å]"
        else:
            xlabel = r"$\vartheta$ [°]"

        self.ax_scan.set_xlabel(xlabel)

        self.canvas_scan.draw_idle()

        if self._scan_click_cid is not None:
            self.fig_scan.canvas.mpl_disconnect(self._scan_click_cid)
        self._scan_click_cid = self.fig_scan.canvas.mpl_connect(
            "button_press_event", self.on_press_scan
        )

    def on_press_scan(self, event):
        """
        Matplotlib button-press handler that picks a diffraction value.

        Fires only if the click is inside the scan axes and no toolbar
        navigation mode (zoom/pan) is active. Updates the diffraction
        field and the scan-plot marker line to the clicked position and
        emits `scan_ready` to request a refresh of the instrument view.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse button-press event.
        """
        if (
            event.inaxes == self.ax_scan
            and self.fig_scan.canvas.toolbar.mode == ""
        ):
            val = event.xdata

            self.diffraction_line.blockSignals(True)

            self.set_diffraction(val)

            self.diffraction_line.blockSignals(False)

            self.line_scan.set_xdata([val])

            self.canvas_scan.draw_idle()

            self.scan_ready.emit()

    def on_press_inst(self, event):
        """
        Matplotlib button-press handler that picks a detector position.

        Fires only if the click is inside the instrument axes and no
        toolbar navigation mode (zoom/pan) is active. Updates the
        horizontal/vertical position fields and the ROI guide lines to
        the clicked position and emits `roi_ready` to request a
        refresh of the ROI-integrated scan.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse button-press event.
        """
        if (
            event.inaxes == self.ax_inst
            and self.fig_inst.canvas.toolbar.mode == ""
        ):
            horz_roi, vert_roi = self.inst_roi["roi"]

            horz, vert = event.xdata, event.ydata

            self.horizontal_line.blockSignals(True)
            self.vertical_line.blockSignals(True)

            self.set_horizontal(horz)
            self.set_vertical(vert)

            self.horizontal_line.blockSignals(False)
            self.vertical_line.blockSignals(False)

            if (
                self._roi_lines is not None
                and self._roi_lines[0] in self.ax_inst.lines
            ):
                vl1, vl2, hl1, hl2 = self._roi_lines
                vl1.set_xdata([horz - horz_roi, horz - horz_roi])
                vl2.set_xdata([horz + horz_roi, horz + horz_roi])
                hl1.set_ydata([vert - vert_roi, vert - vert_roi])
                hl2.set_ydata([vert + vert_roi, vert + vert_roi])
            else:
                vl1 = self.ax_inst.axvline(
                    x=horz - horz_roi, color="k", linestyle="--"
                )
                vl2 = self.ax_inst.axvline(
                    x=horz + horz_roi, color="k", linestyle="--"
                )
                hl1 = self.ax_inst.axhline(
                    y=vert - vert_roi, color="k", linestyle="--"
                )
                hl2 = self.ax_inst.axhline(
                    y=vert + vert_roi, color="k", linestyle="--"
                )
                self._roi_lines = [vl1, vl2, hl1, hl2]

            self.canvas_inst.draw_idle()

            self.roi_ready.emit()

    def connect_roi_ready(self, replot):
        """
        Connect a handler to the `roi_ready` signal.

        Parameters
        ----------
        replot : callable
            Slot invoked when `roi_ready` is emitted, i.e. when the
            region of interest has been repositioned by clicking on
            the instrument view.
        """
        self.roi_ready.connect(replot)

    def connect_scan_ready(self, replot):
        """
        Connect a handler to the `scan_ready` signal.

        Parameters
        ----------
        replot : callable
            Slot invoked when `scan_ready` is emitted, i.e. when the
            diffraction value has been repositioned by clicking on the
            scan view.
        """
        self.scan_ready.connect(replot)

    def connect_slice_ready(self, add_peak):
        """
        Connect a handler to the `slice_ready` signal.

        Parameters
        ----------
        add_peak : callable
            Slot invoked when `slice_ready` is emitted, i.e. when the
            user picks an hkl position on the slice plot to add as a
            peak.
        """
        self.slice_ready.connect(add_peak)

    def get_slice_value(self):
        """
        Get the out-of-plane slice coordinate.

        Returns
        -------
        val : float
            Slice position, or None if the field contains invalid
            input.
        """
        if self.slice_line.hasAcceptableInput():
            return float(self.slice_line.text())

    def get_slice_thickness(self):
        """
        Get the out-of-plane slice integration thickness.

        Returns
        -------
        val : float
            Slice thickness, or None if the field contains invalid
            input.
        """
        if self.slice_thickness_line.hasAcceptableInput():
            return float(self.slice_thickness_line.text())

    def get_slice_width(self):
        """
        Get the in-plane bin width for the slice view.

        Returns
        -------
        val : float
            Slice bin width, or None if the field contains invalid
            input.
        """
        if self.slice_width_line.hasAcceptableInput():
            return float(self.slice_width_line.text())

    def enable_slice_peak_add(self):
        """
        Get whether clicking the slice plot should add a peak.

        Returns
        -------
        enabled : bool
            True if the "add peak from slice" checkbox is checked.
        """
        return self.slice_add_peak_box.isChecked()

    def get_clim_clip_type(self):
        """
        Get the selected color-limit clipping method for the slice view.

        Returns
        -------
        ctype : str
            Selected clipping method, e.g. "μ±3×σ" or "Q₃/Q₁±1.5×IQR".
        """
        return self.clim_combo.currentText()

    def get_slice(self):
        """
        Get the selected slice plane.

        Returns
        -------
        plane : str
            Selected slice plane, e.g. "Axis 1/2".
        """
        return self.slice_combo.currentText()

    def get_slice_scale(self):
        """
        Get the selected color scale for the slice view.

        Returns
        -------
        scale : str
            Selected scale in lowercase, e.g. "linear" or "log".
        """
        return self.slice_scale_combo.currentText().lower()

    def get_inst_vmin_value(self):
        """
        Get the minimum color-scale value for the instrument view.

        Returns
        -------
        val : float
            Minimum display value, or None if the field contains
            invalid input.
        """
        if self.inst_vmin_line.hasAcceptableInput():
            return float(self.inst_vmin_line.text())

    def get_inst_vmax_value(self):
        """
        Get the maximum color-scale value for the instrument view.

        Returns
        -------
        val : float
            Maximum display value, or None if the field contains
            invalid input.
        """
        if self.inst_vmax_line.hasAcceptableInput():
            return float(self.inst_vmax_line.text())

    def get_instrument_auto_limits(self):
        """
        Get whether the instrument view color limits are auto-computed.

        Returns
        -------
        enabled : bool
            True if automatic color-limit scaling is enabled.
        """
        return self.instrument_auto_limits_box.isChecked()

    def get_instrument_auto_zoom(self):
        """
        Get whether the instrument view auto-zooms to fit new data.

        Returns
        -------
        enabled : bool
            True if automatic zoom is enabled.
        """
        return self.instrument_auto_zoom_box.isChecked()

    def set_inst_vmin_value(self, val):
        """
        Set the minimum color-scale value for the instrument view.

        Parameters
        ----------
        val : float
            Minimum display value.
        """
        self.inst_vmin_line.setText(str(round(val, 5)))

    def set_inst_vmax_value(self, val):
        """
        Set the maximum color-scale value for the instrument view.

        Parameters
        ----------
        val : float
            Maximum display value.
        """
        self.inst_vmax_line.setText(str(round(val, 5)))

    def clear_inst_vlims(self):
        """
        Clear the instrument view minimum/maximum color-scale fields.
        """
        self.inst_vmin_line.clear()
        self.inst_vmax_line.clear()

    def get_vmin_value(self):
        """
        Get the minimum color-scale value for the slice view.

        Returns
        -------
        val : float
            Minimum display value, or None if the field contains
            invalid input.
        """
        if self.vmin_line.hasAcceptableInput():
            return float(self.vmin_line.text())

    def get_vmax_value(self):
        """
        Get the maximum color-scale value for the slice view.

        Returns
        -------
        val : float
            Maximum display value, or None if the field contains
            invalid input.
        """
        if self.vmax_line.hasAcceptableInput():
            return float(self.vmax_line.text())

    def get_slice_auto_limits(self):
        """
        Get whether the slice view color limits are auto-computed.

        Returns
        -------
        enabled : bool
            True if automatic color-limit scaling is enabled.
        """
        return self.slice_auto_limits_box.isChecked()

    def get_slice_auto_zoom(self):
        """
        Get whether the slice view auto-zooms to fit new data.

        Returns
        -------
        enabled : bool
            True if automatic zoom is enabled.
        """
        return self.slice_auto_zoom_box.isChecked()

    def _handle_slice_auto_zoom_toggle(self, checked):
        """
        Redraw the slice view when slice auto-zoom is toggled on.

        Parameters
        ----------
        checked : bool
            New checked state of the slice auto-zoom checkbox.
        """
        if checked and self._last_slice_view is not None:
            self.update_slice(self._last_slice_view)

    def _handle_instrument_auto_zoom_toggle(self, checked):
        """
        Redraw the instrument view when instrument auto-zoom is toggled on.

        Parameters
        ----------
        checked : bool
            New checked state of the instrument auto-zoom checkbox.
        """
        if checked and self._last_inst_view is not None:
            self.update_instrument_view(self._last_inst_view)

    def set_vmin_value(self, val):
        """
        Set the minimum color-scale value for the slice view.

        Parameters
        ----------
        val : float
            Minimum display value.
        """
        self.vmin_line.setText(str(round(val, 5)))

    def set_vmax_value(self, val):
        """
        Set the maximum color-scale value for the slice view.

        Parameters
        ----------
        val : float
            Maximum display value.
        """
        self.vmax_line.setText(str(round(val, 5)))

    def get_colormap(self):
        """
        Get the selected colormap for the slice view.

        Returns
        -------
        cmap : str
            Selected colormap name.
        """
        return self.cbar_combo.currentText()

    def get_instrument_colormap(self):
        """
        Get the selected colormap for the instrument view.

        Returns
        -------
        cmap : str
            Selected colormap name.
        """
        return self.vbar_combo.currentText()

    def get_vlim_clip_type(self):
        """
        Get the selected color-limit clipping method for the instrument view.

        Returns
        -------
        ctype : str
            Selected clipping method, e.g. "μ±3×σ" or "Q₃/Q₁±1.5×IQR".
        """
        return self.vlim_combo.currentText()

    def get_instrument_scale(self):
        """
        Get the selected color scale for the instrument view.

        Returns
        -------
        scale : str
            Selected scale in lowercase, e.g. "linear" or "log".
        """
        return self.instrument_scale_combo.currentText().lower()

    def __format_inst_coord(self, x, y):
        """
        Format instrument-view display coordinates as a gamma/nu string.

        Used as the matplotlib Axes `format_coord` callback to show the
        detector angles under the cursor in the instrument plot's
        status readout.

        Parameters
        ----------
        x : float
            X display coordinate (gamma, in degrees) on the instrument
            axes.
        y : float
            Y display coordinate (nu, in degrees) on the instrument
            axes.

        Returns
        -------
        coord_str : str
            Formatted string of the form ``"γ = x°, ν = y°"`` with
            values to one decimal place.
        """
        return "γ = {:.1f}°, ν = {:.1f}°".format(x, y)

    def __format_hkl_coord(self, x, y):
        """
        Format slice-plot display coordinates as an HKL string.

        Used as the matplotlib Axes `format_coord` callback to show the
        crystallographic HKL indices under the cursor in the slice
        plot's status readout.

        Parameters
        ----------
        x : float
            X display coordinate on the slice axes.
        y : float
            Y display coordinate on the slice axes.

        Returns
        -------
        coord_str : str
            Formatted string of the form
            ``"hkl = (h, k, l)"`` with values to three decimal places.
        """
        x, y, _ = np.dot(self.T_inv, [x, y, 1])
        h, k, l = np.dot(self.W, [x, y, self.z])
        return "hkl = ({:.3f}, {:.3f}, {:.3f})".format(h, k, l)

    def update_slice(self, slice_dict):
        """
        Redraw the 2D reciprocal-space slice plot from new slice data.

        Rebuilds the slice axes with a curvilinear grid helper so that
        non-orthogonal reciprocal-lattice axes are displayed correctly,
        redraws the colorbar, preserves the current zoom (unless slice
        auto-zoom is enabled or the axis labels changed), and
        reconnects the click handler and cursor coordinate formatter.
        Also (re)configures the out-of-plane slice slider if depth
        limits are provided.

        Parameters
        ----------
        slice_dict : dict
            Slice data containing ``"x"``, ``"y"``, ``"labels"``,
            ``"title"``, ``"signal"``, ``"z"`` (out-of-plane
            coordinate), ``"W"`` (index transform), ``"transform"``
            (plot affine transform), ``"aspect"``, optional
            ``"vmin"``/``"vmax"`` display limits, and optional
            ``"z_min"``/``"z_max"`` slider bounds.
        """
        prev_xlim = (
            self.ax_slice.get_xlim() if self.slice_im is not None else None
        )
        prev_ylim = (
            self.ax_slice.get_ylim() if self.slice_im is not None else None
        )

        cmap = cmaps[self.get_colormap()]

        x = slice_dict["x"]
        y = slice_dict["y"]

        labels = slice_dict["labels"]
        title = slice_dict["title"]
        signal = slice_dict["signal"]
        self.z = slice_dict["z"]
        self.W = slice_dict["W"]

        scale = self.get_slice_scale()

        vmin = slice_dict.get("vmin")
        vmax = slice_dict.get("vmax")

        if vmin is None or vmax is None:
            finite = signal[np.isfinite(signal)]
            if finite.size > 0:
                vmin = np.nanmin(finite)
                vmax = np.nanmax(finite)
            else:
                vmin, vmax = (0.1, 1) if scale == "log" else (0, 1)

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

        self.slice_im = self.ax_slice.pcolormesh(
            x,
            y,
            signal,
            norm=self._create_norm(scale, vmin, vmax),
            cmap=cmap,
            shading="flat",
            transform=trans,
            rasterized=True,
        )

        self.ax_slice.set_xlabel(labels[0])
        self.ax_slice.set_ylabel(labels[1])

        self.vmin, self.vmax = self.slice_im.norm.vmin, self.slice_im.norm.vmax

        self.vmin_line.blockSignals(True)
        self.vmax_line.blockSignals(True)
        self.set_vmin_value(self.vmin)
        self.set_vmax_value(self.vmax)
        self.vmin_line.blockSignals(False)
        self.vmax_line.blockSignals(False)

        self.ax_slice.set_title(title)
        self.ax_slice.grid(True)

        self.cb_slice = self.fig_slice.colorbar(
            self.slice_im, ax=self.ax_slice
        )
        self.cb_slice.minorticks_on()

        labels_key = tuple(labels)
        major_change = (
            self._last_slice_labels is not None
            and self._last_slice_labels != labels_key
        )

        if (
            not self.get_slice_auto_zoom()
            and prev_xlim is not None
            and prev_ylim is not None
            and not major_change
        ):
            self.ax_slice.set_xlim(prev_xlim)
            self.ax_slice.set_ylim(prev_ylim)

        self.canvas_slice.draw_idle()

        self.ax_slice.format_coord = self.__format_hkl_coord
        if self._slice_click_cid is not None:
            self.fig_slice.canvas.mpl_disconnect(self._slice_click_cid)
        self._slice_click_cid = self.fig_slice.canvas.mpl_connect(
            "button_press_event", self.on_press_slice
        )

        self._last_slice_view = slice_dict
        self._last_slice_labels = labels_key

        z_min = slice_dict.get("z_min")
        z_max = slice_dict.get("z_max")
        if z_min is not None and z_max is not None and z_max > z_min:
            self._setup_slice_slider(z_min, z_max)

    def on_press_slice(self, event):
        """
        Matplotlib button-press handler that picks an hkl on the slice plot.

        Fires only if the click is inside the slice axes and no toolbar
        navigation mode (zoom/pan) is active. Converts the clicked
        display coordinates to hkl, updates the check-hkl fields, and,
        if peak-adding from the slice is enabled, emits `slice_ready`
        to request that a peak be added at that hkl.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse button-press event.
        """
        if (
            event.inaxes == self.ax_slice
            and self.fig_slice.canvas.toolbar.mode == ""
        ):
            if event.xdata is None or event.ydata is None:
                return

            x, y, _ = np.dot(self.T_inv, [event.xdata, event.ydata, 1])
            h, k, l = np.dot(self.W, [x, y, self.z])

            self.set_check_hkl(h, k, l)

            if self.enable_slice_peak_add():
                self.slice_ready.emit(h, k, l)

    def update_cluster_table(self, peak_info):
        """
        Repopulate the cluster table with satellite centroid coordinates.

        Parameters
        ----------
        peak_info : dict
            Clustering result containing a ``"satellites"`` entry, an
            (N, 3) array of fractional satellite modulation vector
            centroids, one row per cluster.
        """
        centroids = peak_info["satellites"].round(3).astype(str)

        self.cluster_table.setRowCount(0)
        self.cluster_table.setRowCount(len(centroids))

        for row, centroid in enumerate(centroids):
            self.cluster_table.setItem(row, 0, QTableWidgetItem(centroid[0]))
            self.cluster_table.setItem(row, 1, QTableWidgetItem(centroid[1]))
            self.cluster_table.setItem(row, 2, QTableWidgetItem(centroid[2]))

    def get_cluster_parameters(self):
        """
        Get the DBSCAN clustering parameters for satellite peak clustering.

        Returns
        -------
        eps : float
            Maximum neighborhood distance between two samples.
        min_samples : int
            Minimum number of samples in a neighborhood for a point to
            be considered a core point.

        Returns None instead if either field contains invalid input.
        """
        params = [self.param_eps_line, self.param_min_line]
        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return float(self.param_eps_line.text()), int(
                self.param_min_line.text()
            )

    def update_alignment_runs(self, peaks):
        """
        Repopulate the alignment run combo box from the indexed peaks.

        Enables the run combo box and "calculate alignment" button only
        if at least one indexed peak with a run number is present;
        otherwise disables them and shows "None".

        Parameters
        ----------
        peaks : list of dict
            Peak information dictionaries, each optionally containing
            ``"ind"`` (indexed flag) and ``"run_number"`` entries.
        """
        current = self.alignment_run_combo.currentText()

        runs = sorted(
            {
                int(peak["run_number"])
                for peak in peaks
                if peak.get("ind") and peak.get("run_number") is not None
            }
        )

        self.alignment_run_combo.blockSignals(True)
        self.alignment_run_combo.clear()

        if len(runs) > 0:
            for run in runs:
                self.alignment_run_combo.addItem(str(run))

            index = self.alignment_run_combo.findText(current)
            self.alignment_run_combo.setCurrentIndex(0 if index < 0 else index)
            self.alignment_run_combo.setEnabled(True)
            self.calculate_alignment_button.setEnabled(True)
        else:
            self.alignment_run_combo.addItem("None")
            self.alignment_run_combo.setCurrentIndex(0)
            self.alignment_run_combo.setEnabled(False)
            self.calculate_alignment_button.setEnabled(False)

        self.auto_scale_dropdown(self.alignment_run_combo)
        self.alignment_run_combo.blockSignals(False)

    def get_alignment_run(self):
        """
        Get the run number selected for goniometer alignment.

        Returns
        -------
        run_number : int
            Selected run number, or None if no run is selected (e.g.
            the combo box shows "None" or free-form text).
        """
        text = self.alignment_run_combo.currentText().strip()

        if re.fullmatch(r"-?\d+", text):
            return int(text)

    def get_alignment_tilts(self):
        """
        Get the goniometer tilt angles for alignment calculation.

        Returns
        -------
        tilts : tuple of float
            ``(yaw, pitch, roll)`` tilt angles.

        Returns None instead if any tilt field contains invalid input.
        """
        params = [
            self.align_yaw_line,
            self.align_pitch_line,
            self.align_roll_line,
        ]
        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return tuple(float(param.text()) for param in params)

    def clear_alignment_plot(self):
        """
        Clear the 3D alignment view and reset the observed/predicted
        hkl histogram axes.
        """
        self.plotter.clear_actors()

        labels = ["$h$", "$k$", "$l$"]

        for axis, label in zip(self.ax_align, labels):
            axis.clear()
            axis.set_xlim(-1, 1)
            axis.set_ylim(1, 100)
            axis.minorticks_on()
            axis.set_yscale("log")
            axis.xaxis.set_major_locator(MplMaxNLocator(integer=True))
            axis.set_xlabel(label)
            axis.axvline(0, color="0.5", linestyle="--", linewidth=1)

        self.canvas_align.draw_idle()

    def add_alignment_peaks(self, alignment_dict):
        """
        Draw observed-vs-predicted Q vectors for goniometer alignment.

        Clears the current alignment plot, histograms the observed hkl
        deviations along each of the h/k/l axes, and renders the 3D
        observed and predicted Q vector point clouds with connecting
        axis wireframes for comparison.

        Parameters
        ----------
        alignment_dict : dict
            Alignment information containing ``"observed"`` and
            ``"predicted"`` Q vector arrays, and ``"observed_hkl"``
            hkl deviations, as produced by the model.
        """
        observed = np.asarray(alignment_dict["observed"])
        predicted = np.asarray(alignment_dict["predicted"])
        observed_hkl = np.asarray(alignment_dict["observed_hkl"])
        observed_color = "lightblue"
        predicted_color = "pink"
        bin_width = 0.05

        self.clear_alignment_plot()

        if observed.size == 0 or predicted.size == 0 or observed_hkl.size == 0:
            return

        limit = np.nanmax(np.abs(observed_hkl))
        limit = max(bin_width, np.ceil(limit / bin_width) * bin_width)
        bins = np.arange(-limit, limit + 1.5 * bin_width, bin_width)

        for i, axis in enumerate(self.ax_align):
            counts, _ = np.histogram(observed_hkl[:, i], bins=bins)
            axis.stairs(counts, bins, color="C{}".format(i))
            axis.set_xlim(-limit, limit)

            positive = counts[counts > 0]
            ymax = 10 if positive.size == 0 else max(10, positive.max() * 1.25)
            axis.set_ylim(1, ymax)

        self.canvas_align.draw_idle()

        coords = np.vstack([observed, predicted])
        min_lim = np.nanmin(coords, axis=0)
        max_lim = np.nanmax(coords, axis=0)

        if np.allclose(min_lim, max_lim):
            min_lim = min_lim - 1
            max_lim = max_lim + 1
        else:
            span = max_lim - min_lim
            pad = np.where(span > 0, 0.05 * span, 1.0)
            min_lim = min_lim - pad
            max_lim = max_lim + pad

        mesh = pv.Line(
            pointa=(min_lim[0], 0, 0), pointb=(max_lim[0], 0, 0), resolution=1
        )
        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        mesh = pv.Line(
            pointa=(0, min_lim[1], 0), pointb=(0, max_lim[1], 0), resolution=1
        )
        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        mesh = pv.Line(
            pointa=(0, 0, min_lim[2]), pointb=(0, 0, max_lim[2]), resolution=1
        )
        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        self.plotter.add_mesh(
            pv.PolyData(np.asarray(observed, dtype=float)),
            color=observed_color,
            smooth_shading=True,
            point_size=12,
            render_points_as_spheres=True,
        )
        self.plotter.add_mesh(
            pv.PolyData(np.asarray(predicted, dtype=float)),
            color=predicted_color,
            smooth_shading=True,
            point_size=10,
            render_points_as_spheres=True,
        )
        self.plotter.add_legend(
            [["Observed", observed_color], ["Predicted", predicted_color]],
            loc="lower right",
            bcolor="w",
            face=None,
        )
        self.plotter.enable_depth_peeling()
        self.reset_view()

    def add_cluster_peaks(self, peak_dict):
        """
        Draw clustered satellite peak positions and their unit-cell images.

        Clears the current cluster plot, then for each DBSCAN cluster
        replicates its points across neighboring unit cells (via the
        lattice translation vectors), keeps only the images that fall
        within the reference cell, and renders them as a 3D point
        cloud. Unclustered (noise) points are drawn in black; clustered
        points are colored by cluster and histogrammed along each of
        the h/k/l axes. Also draws the reference unit-cell wireframe
        and its coordinate planes.

        Parameters
        ----------
        peak_dict : dict
            Clustering result containing ``"coordinates"`` (peak
            positions), ``"clusters"`` (cluster label per peak, with
            negative labels for noise), ``"translation"`` (lattice
            translation vectors), ``"transform"`` (cell orientation
            matrix), and ``"inverse"`` (its inverse), as produced by
            the model.
        """
        self.plotter.clear_actors()

        for i in range(3):
            self.ax_clust[i].clear()

        bins = np.linspace(-1.025, 1.025, 42)

        coordinates = np.array(peak_dict["coordinates"])
        clusters = np.array(peak_dict["clusters"])

        vectors = peak_dict["translation"]
        T = peak_dict["transform"]
        T_inv = peak_dict["inverse"]

        translations = np.array(
            np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        ).T.reshape(-1, 3)

        offsets = np.dot(translations, vectors)

        multiblock = pv.MultiBlock()

        for uni in np.unique(clusters):
            coords = coordinates[clusters == uni]
            coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, 3)
            delta = (T_inv @ coords.T).T
            mask = (np.abs(delta) < 1).all(axis=1)
            coords = coords[mask]
            delta = delta[mask]
            points = pv.PolyData(np.asarray(coords, dtype=float))
            if uni >= 0:
                color = "C{}".format(uni)
                multiblock[color] = points
                if len(delta) > 0:
                    h, _ = np.histogram(delta[:, 0], bins=bins)
                    k, _ = np.histogram(delta[:, 1], bins=bins)
                    l, _ = np.histogram(delta[:, 2], bins=bins)
                    self.ax_clust[0].stairs(h, bins, color=color)
                    self.ax_clust[1].stairs(k, bins, color=color)
                    self.ax_clust[2].stairs(l, bins, color=color)
            else:
                self.plotter.add_mesh(
                    points,
                    color="k",
                    smooth_shading=True,
                    point_size=5,
                    render_points_as_spheres=True,
                )

        for i in range(3):
            self.ax_clust[i].minorticks_on()
            self.ax_clust[i].set_yscale("log")
            self.ax_clust[i].xaxis.set_major_locator(
                MplMaxNLocator(integer=True)
            )

        self.ax_clust[0].set_xlabel("$[h00]$")
        self.ax_clust[1].set_xlabel("$[0k0]$")
        self.ax_clust[2].set_xlabel("$[00l]$")

        self.canvas_clust.draw_idle()

        _, mapper = self.plotter.add_composite(
            multiblock,
            multi_colors=True,
            smooth_shading=True,
            point_size=10,
            render_points_as_spheres=True,
        )

        prop_cycle = plt.rcParams["axes.prop_cycle"]

        cmap = prop_cycle.by_key()["color"]

        colors = []
        for i in range(1, len(mapper.block_attr)):
            colors.append(cmap[i - 1])
            mapper.block_attr[i].color = cmap[i - 1]

        legend = [["C{}".format(i), color] for i, color in enumerate(colors)]

        A = np.eye(4)
        A[:3, :3] = T

        mesh = pv.Box(bounds=(-1, 1, -1, 1, -1, 1), level=0, quads=True)
        mesh.transform(A, inplace=True)

        self.plotter.add_mesh(
            mesh, color="k", style="wireframe", render_lines_as_tubes=True
        )

        for point in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            mesh = pv.Line(pointa=-np.array(point), pointb=point, resolution=1)
            mesh.transform(A, inplace=True)

            self.plotter.add_mesh(
                mesh, color="k", style="wireframe", render_lines_as_tubes=True
            )

        pointsa = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        pointsb = [(-1, 1), (1, 1), (1, -1), (-1, -1)]

        for i in range(4):
            a, b = pointsa[i], pointsb[i]

            mesh = pv.Line(
                pointa=(a[0], a[1], 0), pointb=(b[0], b[1], 0), resolution=1
            )

            mesh.transform(A, inplace=True)

            self.plotter.add_mesh(
                mesh, color="k", style="wireframe", render_lines_as_tubes=True
            )

            mesh = pv.Line(
                pointa=(a[0], 0, a[1]), pointb=(b[0], 0, b[1]), resolution=1
            )

            mesh.transform(A, inplace=True)

            self.plotter.add_mesh(
                mesh, color="k", style="wireframe", render_lines_as_tubes=True
            )

            mesh = pv.Line(
                pointa=(0, a[0], a[1]), pointb=(0, b[0], b[1]), resolution=1
            )

            mesh.transform(A, inplace=True)

            self.plotter.add_mesh(
                mesh, color="k", style="wireframe", render_lines_as_tubes=True
            )

        self.plotter.add_legend(
            legend, loc="lower right", bcolor="w", face=None
        )

        self.plotter.enable_depth_peeling()

        self.reset_view()
