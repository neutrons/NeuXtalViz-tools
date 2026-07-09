import os
import pathlib
import re

import matplotlib.pyplot as plt

from qtpy.QtWidgets import (
    QWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QFileDialog,
    QPlainTextEdit,
    QApplication,
)

from qtpy.QtGui import (
    QDoubleValidator,
    QFont,
    QColor,
    QPalette,
    QPixmap,
    QPainter,
    QIcon,
)
from qtpy.QtCore import Qt, Signal, QSettings

import qdarkstyle
from qdarkstyle.light.palette import LightPalette as _LightPalette
from qdarkstyle.dark.palette import DarkPalette as _DarkPalette

import numpy as np
import pyvista as pv

pv.set_plot_theme("document")

from pyvistaqt import QtInteractor

from NeuXtalViz.views.utilities import Worker, ThreadPool

import qtawesome as qta

_MPL_STYLE_BY_THEME = {
    "default": "default",
    "document": "default",
    "dark": "dark_background",
    "paraview": "Solarize_Light2",
}


class NeuXtalVizWidget(QWidget):
    """
    Base widget for all NeuXtalViz views, providing shared functionality
    and interface for user-facing widgets.

    This widget assembles the common 3D visualization panel shared by
    all feature-specific views: a PyVista/VTK plotter, camera and view
    controls, axis and reciprocal/real lattice toggles, theming
    controls, an information console, and a status/progress bar with a
    stop button. Feature-specific views subclass or embed this widget
    and add their own controls alongside it.

    Attributes
    ----------
    log_output : Signal(str)
        Signal emitted to relay log/console text.
    cam_ready : Signal()
        Signal emitted whenever the camera state has been updated.
    """

    log_output = Signal(str)
    cam_ready = Signal()

    def __init__(self, parent=None):
        """
        Build the shared visualization panel and its controls.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None.
        """

        super().__init__(parent)
        self._worker_running = False

        self.proj_box = QCheckBox("Enable Parallel Projection", self)
        self.proj_box.setChecked(True)
        self.proj_box.setToolTip("Toggle parallel projection for the 3D view.")
        self.proj_box.clicked.connect(self.change_projection)

        self.reset_button = QPushButton("Reset View", self)
        self.reset_button.setToolTip(
            "Reset the view to the default orientation."
        )
        self.reset_button.clicked.connect(self.reset_view)
        self.reset_button.setIcon(qta.icon("fa6s.house"))

        self.camera_button = QPushButton("Reset Camera", self)
        self.camera_button.setToolTip("Reset the camera position.")
        self.camera_button.clicked.connect(self.reset_camera)
        self.camera_button.setIcon(qta.icon("fa6s.camera"))

        self.recip_box = QCheckBox("Toggle Reciprocal Lattice", self)
        self.recip_box.setChecked(True)
        self.recip_box.setToolTip("Show or hide reciprocal lattice vectors.")

        self.axes_box = QCheckBox("Show Axes", self)
        self.axes_box.setChecked(True)
        self.axes_box.setToolTip(
            "Show or hide the coordinate axes in the plot."
        )
        self.axes_box.clicked.connect(self.show_axes)

        self.cons_box = QCheckBox("Expand Console", self)
        self.cons_box.setChecked(True)
        self.cons_box.setToolTip("Show or hide console output.")
        self.cons_box.stateChanged.connect(self.toggle_console)

        self.save_button = QPushButton("Save Screenshot", self)
        self.save_button.setToolTip(
            "Save a screenshot of the current 3D view."
        )
        self.save_button.setIcon(qta.icon("fa6s.floppy-disk"))

        self.theme_combo = QComboBox(self)
        self.theme_combo.addItem("default")
        self.theme_combo.addItem("document")
        self.theme_combo.addItem("dark")
        self.theme_combo.addItem("paraview")
        _app = QApplication.instance()
        _is_dark = bool(_app.property("ui_dark")) if _app else False
        self.theme_combo.setCurrentText("dark" if _is_dark else "default")
        self.theme_combo.setToolTip(
            'Select 3D view theme. Selecting "dark" also switches the '
            "application to dark mode."
        )
        self.theme_combo.currentIndexChanged.connect(self.update_theme)
        self.auto_scale_dropdown(self.theme_combo)

        self.frame = QFrame()

        self.plotter = QtInteractor(self.frame)

        layout = QHBoxLayout()
        vis_layout = QVBoxLayout()

        camera_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        middle_layout = QVBoxLayout()

        left_layout.addWidget(self.save_button)
        left_layout.addWidget(self.reset_button)
        left_layout.addWidget(self.camera_button)
        left_layout.addWidget(self.theme_combo)

        right_layout.addWidget(self.recip_box)
        right_layout.addWidget(self.axes_box)
        right_layout.addWidget(self.proj_box)
        right_layout.addWidget(self.cons_box)

        view_tab = self.__init_view_tab()

        middle_layout.addWidget(view_tab)

        camera_layout.addLayout(left_layout)
        camera_layout.addLayout(middle_layout)
        camera_layout.addLayout(right_layout)

        vis_layout.addLayout(camera_layout)
        vis_layout.addWidget(self.plotter.interactor)
        vis_layout.setStretch(1, 1)

        info_tab = self.__init_info_tab()

        vis_layout.addWidget(info_tab)

        self.stop_button = QPushButton("Stop Process", self)
        self.stop_button.setToolTip("Stop the current processes.")

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready!")
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.addPermanentWidget(self.stop_button)

        self.console = QPlainTextEdit(readOnly=True)

        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(10)
        self.console.setFont(font)

        vis_layout.addWidget(self.console)
        vis_layout.addWidget(self.status_bar)

        layout.addLayout(vis_layout, stretch=1)

        self.setLayout(layout)

        self.camera_position = None
        self.T = None

        self.threadpool = ThreadPool()

        self.plotter.enable_parallel_projection()

    @staticmethod
    def stop_processing(stop_event):
        """
        Check if a stop has been requested.

        Convenience method for worker tasks to check if they should terminate early.

        Parameters
        ----------
        stop_event : threading.Event or None
            The stop event to check.

        Returns
        -------
        stop : bool
            True if stop was requested, False otherwise.
        """
        return stop_event is not None and stop_event.is_set()

    def connect_stop(self, stop):
        """
        Connect stop button to presenter callback.

        Parameters
        ----------
        stop : callable
            Callback function to handle stop requests.
        """
        self.stop_button.clicked.connect(stop)

    def auto_scale_dropdown(self, combo):
        """Autoscale a combobox width to fit text plus any icons/checks.

        This keeps the drop-down (and closed state) wide enough for the
        longest item label while leaving extra room for icons or check
        indicators drawn on the left-hand side. Also assigns a
        representative icon to each item (based on item text, e.g. a
        gem icon for "TOPAZ") before measuring the required width.

        Parameters
        ----------
        combo : QComboBox
            Combo box to resize and decorate with icons.
        """

        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        fm = combo.fontMetrics()
        max_width = 0

        digit = all(
            [combo.itemText(i).isdigit() for i in range(combo.count())]
        )

        for i in range(combo.count()):
            text = combo.itemText(i)
            icon = qta.icon("fa6s.hashtag" if digit else "fa6s.minus")
            space_group_match = re.match(r"^(\d+):", text)
            if text == "TOPAZ":
                icon = qta.icon("fa6s.gem")
            elif text == "CORELLI":
                icon = qta.icon("fa6s.scissors")
            elif text == "MANDI":
                icon = qta.icon("fa6s.dna")
            elif text == "WAND²":
                icon = qta.icon("fa6s.wand-magic")
            elif text == "DEMAND":
                icon = qta.icon("fa6s.magnet")
            elif text == "SNAP":
                icon = qta.icon("fa6s.weight-scale")
            elif text == "IMAGINE":
                icon = qta.icon("fa6s.lightbulb")
            elif space_group_match:
                no = space_group_match.group(1)
                pixmap = QPixmap(64, 64)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setFont(QFont("Arial", 32))
                painter.drawText(
                    pixmap.rect(), Qt.AlignmentFlag.AlignCenter, no
                )
                painter.end()
                icon = QIcon(pixmap)
            combo.setItemIcon(i, icon)

        for i in range(combo.count()):
            text = combo.itemText(i)
            text_width = fm.horizontalAdvance(text)

            icon = combo.itemIcon(i)
            icon_width = 0
            if not icon.isNull():
                size = icon.actualSize(combo.iconSize())
                icon_width = size.width() + 8

            max_width = max(max_width, text_width + icon_width)

        if max_width:
            padding = 40
            combo.setMinimumWidth(max_width + padding)

    def __init_view_tab(self):
        """
        Build the view-control tab widget.

        Creates the "Direction View" (axis-aligned and reciprocal/real
        lattice snap buttons), "Manual View" (typed [hkl]/[uvw] view
        and up-direction entry), and "Rotate View" (roll/elevate/
        azimuth stepping and camera orientation readout) sub-tabs, and
        wires their internal signal connections.

        Returns
        -------
        view_tab : QTabWidget
            Tab widget containing the direction, manual, and rotate
            view-control tabs.
        """

        view_tab = QTabWidget()

        self.view_combo = QComboBox(self)
        self.view_combo.addItem("[hkl]")
        self.view_combo.addItem("[uvw]")
        self.view_combo.setToolTip("Select axis notation for view direction.")
        self.view_combo.currentIndexChanged.connect(self.update_labels)

        self.viewup_combo = QComboBox(self)
        self.viewup_combo.addItem("[hkl]")
        self.viewup_combo.addItem("[uvw]")
        self.viewup_combo.setToolTip("Select axis notation for up direction.")
        self.viewup_combo.currentIndexChanged.connect(self.update_labels)
        self.auto_scale_dropdown(self.view_combo)
        self.auto_scale_dropdown(self.viewup_combo)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.axis1_line = QLineEdit()
        self.axis2_line = QLineEdit()
        self.axis3_line = QLineEdit()

        self.axis1_line.setValidator(validator)
        self.axis2_line.setValidator(validator)
        self.axis3_line.setValidator(validator)
        self.axis1_line.setToolTip(
            "First component of the view direction (e.g., h or u)"
        )
        self.axis2_line.setToolTip(
            "Second component of the view direction (e.g., k or v)"
        )
        self.axis3_line.setToolTip(
            "Third component of the view direction (e.g., l or w)"
        )

        self.axis1_line.setPlaceholderText("h")
        self.axis2_line.setPlaceholderText("k")
        self.axis3_line.setPlaceholderText("l")

        self.axis1_label = QLabel("h", self)
        self.axis2_label = QLabel("k", self)
        self.axis3_label = QLabel("l", self)

        self.axisup1_line = QLineEdit()
        self.axisup2_line = QLineEdit()
        self.axisup3_line = QLineEdit()

        self.axisup1_line.setValidator(validator)
        self.axisup2_line.setValidator(validator)
        self.axisup3_line.setValidator(validator)
        self.axisup1_line.setToolTip(
            "First component of the up direction (e.g., h or u)"
        )
        self.axisup2_line.setToolTip(
            "Second component of the up direction (e.g., k or v)"
        )
        self.axisup3_line.setToolTip(
            "Third component of the up direction (e.g., l or w)"
        )

        self.axisup1_line.setPlaceholderText("h")
        self.axisup2_line.setPlaceholderText("k")
        self.axisup3_line.setPlaceholderText("l")

        self.axisup1_label = QLabel("h", self)
        self.axisup2_label = QLabel("k", self)
        self.axisup3_label = QLabel("l", self)

        self.manual_button = QPushButton("View Axis", self)
        self.manual_button.setToolTip(
            "Set the view direction using the specified axis components."
        )
        self.manualup_button = QPushButton("View Up Axis", self)
        self.manualup_button.setToolTip(
            "Set the up direction using the specified axis components."
        )

        self.manual_button.setIcon(qta.icon("fa6s.right-long"))
        self.manualup_button.setIcon(qta.icon("fa6s.up-long"))

        self.px_button = QPushButton("+Qx", self)
        self.px_button.setToolTip("View along the +Qx direction.")
        self.py_button = QPushButton("+Qy", self)
        self.py_button.setToolTip("View along the +Qy direction.")
        self.pz_button = QPushButton("+Qz", self)
        self.pz_button.setToolTip("View along the +Qz direction.")

        self.mx_button = QPushButton("-Qx", self)
        self.mx_button.setToolTip("View along the -Qx direction.")
        self.my_button = QPushButton("-Qy", self)
        self.my_button.setToolTip("View along the -Qy direction.")
        self.mz_button = QPushButton("-Qz", self)
        self.mz_button.setToolTip("View along the -Qz direction.")

        self.px_button.clicked.connect(self.view_yz)
        self.py_button.clicked.connect(self.view_zx)
        self.pz_button.clicked.connect(self.view_xy)

        self.mx_button.clicked.connect(self.view_zy)
        self.my_button.clicked.connect(self.view_xz)
        self.mz_button.clicked.connect(self.view_yx)

        self.px_button.setIcon(qta.icon("fa6s.right-long"))
        self.py_button.setIcon(qta.icon("fa6s.right-long"))
        self.pz_button.setIcon(qta.icon("fa6s.right-long"))

        self.mx_button.setIcon(qta.icon("fa6s.right-long"))
        self.my_button.setIcon(qta.icon("fa6s.right-long"))
        self.mz_button.setIcon(qta.icon("fa6s.right-long"))

        self.a_star_button = QPushButton("a*", self)
        self.a_star_button.setToolTip(
            "Align view along the a* reciprocal axis."
        )
        self.b_star_button = QPushButton("b*", self)
        self.b_star_button.setToolTip(
            "Align view along the b* reciprocal axis."
        )
        self.c_star_button = QPushButton("c*", self)
        self.c_star_button.setToolTip(
            "Align view along the c* reciprocal axis."
        )

        self.a_button = QPushButton("a", self)
        self.a_button.setToolTip("Align view along the a real axis.")
        self.b_button = QPushButton("b", self)
        self.b_button.setToolTip("Align view along the b real axis.")
        self.c_button = QPushButton("c", self)
        self.c_button.setToolTip("Align view along the c real axis.")

        directions_layout = QGridLayout()
        manual_layout = QGridLayout()

        directions_tab = QWidget()
        manual_tab = QWidget()

        directions_layout.addWidget(self.px_button, 0, 0)
        directions_layout.addWidget(self.py_button, 0, 1)
        directions_layout.addWidget(self.pz_button, 0, 2)
        directions_layout.addWidget(self.a_star_button, 0, 3)
        directions_layout.addWidget(self.b_star_button, 0, 4)
        directions_layout.addWidget(self.c_star_button, 0, 5)

        directions_layout.addWidget(self.mx_button, 1, 0)
        directions_layout.addWidget(self.my_button, 1, 1)
        directions_layout.addWidget(self.mz_button, 1, 2)
        directions_layout.addWidget(self.a_button, 1, 3)
        directions_layout.addWidget(self.b_button, 1, 4)
        directions_layout.addWidget(self.c_button, 1, 5)

        manual_layout.addWidget(self.axis1_label, 0, 0, Qt.AlignCenter)
        manual_layout.addWidget(self.axis2_label, 0, 1, Qt.AlignCenter)
        manual_layout.addWidget(self.axis3_label, 0, 2, Qt.AlignCenter)

        manual_layout.addWidget(self.axis1_line, 1, 0)
        manual_layout.addWidget(self.axis2_line, 1, 1)
        manual_layout.addWidget(self.axis3_line, 1, 2)

        manual_layout.addWidget(self.view_combo, 0, 3)
        manual_layout.addWidget(self.manual_button, 1, 3)

        manual_layout.addWidget(self.axisup1_label, 0, 4, Qt.AlignCenter)
        manual_layout.addWidget(self.axisup2_label, 0, 5, Qt.AlignCenter)
        manual_layout.addWidget(self.axisup3_label, 0, 6, Qt.AlignCenter)

        manual_layout.addWidget(self.axisup1_line, 1, 4)
        manual_layout.addWidget(self.axisup2_line, 1, 5)
        manual_layout.addWidget(self.axisup3_line, 1, 6)

        manual_layout.addWidget(self.viewup_combo, 0, 7)
        manual_layout.addWidget(self.manualup_button, 1, 7)

        snap_layout = QVBoxLayout()
        values_layout = QVBoxLayout()

        snap_layout.addLayout(directions_layout)
        snap_layout.addStretch(1)

        values_layout.addLayout(manual_layout)
        values_layout.addStretch(1)

        directions_tab.setLayout(snap_layout)
        manual_tab.setLayout(values_layout)

        rotate_tab = QWidget()
        rotate_layout = QGridLayout()

        rotate_label = QLabel("Step [°]", self)
        self.rotate_step_line = QLineEdit(self)
        self.rotate_step_line.setText("5.0")

        rotate_validator = QDoubleValidator.StandardNotation
        rotate_step_validator = QDoubleValidator(
            -360.0, 360.0, 2, notation=rotate_validator
        )
        self.rotate_step_line.setValidator(rotate_step_validator)
        self.rotate_step_line.setToolTip(
            "Rotation step in degrees about the current viewing axis."
        )

        self.rotate_ccw_button = QPushButton("Roll CCW", self)
        self.rotate_ccw_button.setToolTip(
            "Roll the view counter‑clockwise about the viewing axis."
        )
        self.rotate_ccw_button.setIcon(qta.icon("fa6s.rotate-left"))

        self.rotate_cw_button = QPushButton("Roll CW", self)
        self.rotate_cw_button.setToolTip(
            "Roll the view clockwise about the viewing axis."
        )
        self.rotate_cw_button.setIcon(qta.icon("fa6s.rotate-right"))

        self.elev_up_button = QPushButton("Elevate Up", self)
        self.elev_up_button.setToolTip("Tilt the camera up by the step angle.")
        self.elev_up_button.setIcon(qta.icon("fa6s.arrow-up"))

        self.elev_down_button = QPushButton("Elevate Down", self)
        self.elev_down_button.setToolTip(
            "Tilt the camera down by the step angle."
        )
        self.elev_down_button.setIcon(qta.icon("fa6s.arrow-down"))

        self.az_left_button = QPushButton("Azimuth Left", self)
        self.az_left_button.setToolTip(
            "Rotate the camera left (azimuth) by the step angle."
        )
        self.az_left_button.setIcon(qta.icon("fa6s.arrow-left"))

        self.az_right_button = QPushButton("Azimuth Right", self)
        self.az_right_button.setToolTip(
            "Rotate the camera right (azimuth) by the step angle."
        )
        self.az_right_button.setIcon(qta.icon("fa6s.arrow-right"))

        rotate_layout.addWidget(self.rotate_ccw_button, 0, 0)
        rotate_layout.addWidget(self.rotate_cw_button, 1, 0)
        rotate_layout.addWidget(self.elev_up_button, 0, 1)
        rotate_layout.addWidget(self.elev_down_button, 1, 1)
        rotate_layout.addWidget(self.az_left_button, 0, 2)
        rotate_layout.addWidget(self.az_right_button, 1, 2)
        rotate_layout.addWidget(rotate_label, 0, 3, Qt.AlignCenter)
        rotate_layout.addWidget(self.rotate_step_line, 1, 3)

        camera_label = QLabel("Camera [°]", self)
        self.camera_pos_line = QLineEdit(self)
        self.camera_pos_line.setReadOnly(True)
        self.camera_pos_line.setToolTip(
            "Current camera roll, elevation, and azimuth in degrees."
        )

        rotate_layout.addWidget(camera_label, 0, 4)
        rotate_layout.addWidget(self.camera_pos_line, 1, 4)

        control_layout = QVBoxLayout()
        control_layout.addLayout(rotate_layout)
        control_layout.addStretch(1)

        rotate_tab.setLayout(control_layout)

        view_tab.addTab(directions_tab, "Direction View")
        view_tab.addTab(manual_tab, "Manual View")
        view_tab.addTab(rotate_tab, "Rotate View")

        # Apply colored styling to a/b/c and a*/b*/c* buttons
        self._init_axis_icons()

        return view_tab

    def __init_info_tab(self):
        """
        Build the lattice/orientation info tab widget.

        Creates the "Lattice Parameters" (read-only a, b, c, alpha,
        beta, gamma) and "Sample Orientation" (read-only u and v
        vectors) sub-tabs displaying the current oriented lattice.

        Returns
        -------
        info_tab : QTabWidget
            Tab widget containing the lattice parameters and sample
            orientation display tabs.
        """

        info_tab = QTabWidget()

        ub_a_label = QLabel("a:", self)
        ub_b_label = QLabel("b:", self)
        ub_c_label = QLabel("c:", self)
        ub_alpha_label = QLabel("α:", self)
        ub_beta_label = QLabel("β:", self)
        ub_gamma_label = QLabel("γ:", self)
        ub_u_label = QLabel("u:", self)
        ub_v_label = QLabel("v:", self)

        ub_angstrom_label = QLabel("Å")
        ub_degree_label = QLabel("°")

        self.ub_a_line = QLineEdit()
        self.ub_b_line = QLineEdit()
        self.ub_c_line = QLineEdit()
        self.ub_alpha_line = QLineEdit()
        self.ub_beta_line = QLineEdit()
        self.ub_gamma_line = QLineEdit()
        self.ub_u1_line = QLineEdit()
        self.ub_u2_line = QLineEdit()
        self.ub_u3_line = QLineEdit()
        self.ub_v1_line = QLineEdit()
        self.ub_v2_line = QLineEdit()
        self.ub_v3_line = QLineEdit()

        self.ub_a_line.setReadOnly(True)
        self.ub_b_line.setReadOnly(True)
        self.ub_c_line.setReadOnly(True)
        self.ub_alpha_line.setReadOnly(True)
        self.ub_beta_line.setReadOnly(True)
        self.ub_gamma_line.setReadOnly(True)
        self.ub_u1_line.setReadOnly(True)
        self.ub_u2_line.setReadOnly(True)
        self.ub_u3_line.setReadOnly(True)
        self.ub_v1_line.setReadOnly(True)
        self.ub_v2_line.setReadOnly(True)
        self.ub_v3_line.setReadOnly(True)

        lattice_layout = QGridLayout()
        orientation_layout = QGridLayout()

        lattice_tab = QWidget()
        orientation_tab = QWidget()

        lattice_layout.addWidget(ub_a_label, 0, 0)
        lattice_layout.addWidget(self.ub_a_line, 0, 1)
        lattice_layout.addWidget(ub_b_label, 0, 2)
        lattice_layout.addWidget(self.ub_b_line, 0, 3)
        lattice_layout.addWidget(ub_c_label, 0, 4)
        lattice_layout.addWidget(self.ub_c_line, 0, 5)
        lattice_layout.addWidget(ub_angstrom_label, 0, 6)

        lattice_layout.addWidget(ub_alpha_label, 1, 0)
        lattice_layout.addWidget(self.ub_alpha_line, 1, 1)
        lattice_layout.addWidget(ub_beta_label, 1, 2)
        lattice_layout.addWidget(self.ub_beta_line, 1, 3)
        lattice_layout.addWidget(ub_gamma_label, 1, 4)
        lattice_layout.addWidget(self.ub_gamma_line, 1, 5)
        lattice_layout.addWidget(ub_degree_label, 1, 6)

        orientation_layout.addWidget(ub_u_label, 0, 0)
        orientation_layout.addWidget(self.ub_u1_line, 0, 1)
        orientation_layout.addWidget(self.ub_u2_line, 0, 2)
        orientation_layout.addWidget(self.ub_u3_line, 0, 3)
        orientation_layout.addWidget(ub_v_label, 1, 0)
        orientation_layout.addWidget(self.ub_v1_line, 1, 1)
        orientation_layout.addWidget(self.ub_v2_line, 1, 2)
        orientation_layout.addWidget(self.ub_v3_line, 1, 3)

        lattice_tab.setLayout(lattice_layout)
        orientation_tab.setLayout(orientation_layout)

        info_tab.addTab(lattice_tab, "Lattice Parameters")
        info_tab.addTab(orientation_tab, "Sample Orientation")

        return info_tab

    def append_to_console(self, text):
        """
        Append a line of text to the console output widget.

        Parameters
        ----------
        text : str
            Text to append to the console.
        """

        self.console.appendPlainText(text)

    def _init_axis_icons(self):
        """Assign subtle colored borders to axis and Q-direction buttons.

        a / a* = red, b / b* = green, c / c* = blue (with a
        paraview-specific swap), and Qx/Qy/Qz use softened NeuXtalViz
        brand colors. Existing button style sheets are preserved and
        only the border styling is updated.
        """

        theme = (
            self.theme_combo.currentText()
            if hasattr(self, "theme_combo")
            else "default"
        )

        # Lattice-axis colors
        red = QColor("red")
        green = QColor("green")
        blue = QColor("blue")
        yellow = QColor("yellow")

        a_color = red
        b_color = yellow if theme == "paraview" else green
        c_color = green if theme == "paraview" else blue

        # Real-space axes: solid arrows
        self.a_button.setIcon(qta.icon("fa6s.right-long", color=a_color))
        self.b_button.setIcon(qta.icon("fa6s.right-long", color=b_color))
        self.c_button.setIcon(qta.icon("fa6s.right-long", color=c_color))

        # Reciprocal axes: outlined arrows to distinguish a*/b*/c*
        self.a_star_button.setIcon(qta.icon("fa6s.right-long", color=a_color))
        self.b_star_button.setIcon(qta.icon("fa6s.right-long", color=b_color))
        self.c_star_button.setIcon(qta.icon("fa6s.right-long", color=c_color))

    def toggle_console(self, state):
        """
        Show or hide the console output widget.

        Connected to the "Expand Console" checkbox's ``stateChanged``
        signal.

        Parameters
        ----------
        state : int or Qt.CheckState
            Checkbox state; truthy (checked) shows the console,
            falsy (unchecked) hides it.
        """

        self.console.setVisible(bool(state))

    def start_worker_pool(self, worker):
        """
        Connect a worker's output to the console and start it.

        Wires ``worker.signals.output`` to :meth:`append_to_console`
        so any captured stdout/stderr/logging text produced while the
        task runs is streamed live to the console, then hands the
        worker off to the shared :class:`ThreadPool` for execution.

        Parameters
        ----------
        worker : NeuXtalViz.views.utilities.Worker
            Worker to run, typically created via :meth:`worker` with
            its ``result``/``finished``/``progress`` signals already
            connected by the presenter.
        """

        worker.signals.output.connect(self.append_to_console)
        self.threadpool.start_worker_pool(worker)

    def worker(self, task):
        """
        Wrap a callable as a background :class:`Worker` task.

        The returned worker calls ``task`` on a thread-pool thread,
        automatically injecting ``progress`` and ``stop_event``
        keyword arguments so the task can report progress and check
        for cooperative cancellation.

        Parameters
        ----------
        task : callable
            Function (or `functools.partial`-bound method) to run in
            the background, e.g. a presenter processing method.

        Returns
        -------
        worker : NeuXtalViz.views.utilities.Worker
            Worker object wrapping ``task``, ready to have its
            signals connected and be passed to
            :meth:`start_worker_pool`.
        """

        return Worker(task)

    def set_info(self, status):
        """
        Update status information.

        Parameters
        ----------
        status : str
            Information.

        """

        self.status_bar.showMessage(status)

    def set_step(self, progress):
        """
        Update progress step.

        Parameters
        ----------
        progress : int, str
            Step or status.

        """

        if type(progress) is int:
            self.progress_bar.setFormat("%p%")
            self.progress_bar.setValue(progress)
        else:
            self.progress_bar.setFormat(progress)
            self.progress_bar.setValue(0)

    def set_oriented_lattice_parameters(
        self, a, b, c, alpha, beta, gamma, u, v
    ):
        """
        Update the oriented lattice paramters.

        Parameters
        ----------
        a, b, c : float
            Lattice constants.
        alpha, beta, gamma : float
            Lattice angles.
        u, v : 3-element 1d array-like
            Sample orientation vectors.

        """

        self.ub_a_line.setText("{:.5f}".format(a))
        self.ub_b_line.setText("{:.5f}".format(b))
        self.ub_c_line.setText("{:.5f}".format(c))
        self.ub_alpha_line.setText("{:.3f}".format(alpha))
        self.ub_beta_line.setText("{:.3f}".format(beta))
        self.ub_gamma_line.setText("{:.3f}".format(gamma))
        self.ub_u1_line.setText("{:.4f}".format(u[0]))
        self.ub_u2_line.setText("{:.4f}".format(u[1]))
        self.ub_u3_line.setText("{:.4f}".format(u[2]))
        self.ub_v1_line.setText("{:.4f}".format(v[0]))
        self.ub_v2_line.setText("{:.4f}".format(v[1]))
        self.ub_v3_line.setText("{:.4f}".format(v[2]))

    def connect_manual_axis(self, view_manual):
        """
        Manual axis view connection.

        Parameters
        ----------
        view_manual : function
            Manual axis view handler.

        """

        self.manual_button.clicked.connect(view_manual)

    def connect_manual_up_axis(self, view_manual):
        """
        Manual axis up view connection.

        Parameters
        ----------
        view_manual : function
            Manual axis up view handler.

        """

        self.manualup_button.clicked.connect(view_manual)

    def connect_reciprocal_axes(self, view_a_star, view_b_star, view_c_star):
        """
        Reciprocal axes view connections.

        Parameters
        ----------
        view_a_star : function
            :math:`a^\ast`-axis view handler.
        view_b_star : function
            :math:`b^\ast`-axis view handler.
        view_c_star : function
            :math:`c^\ast`-axis view handler.

        """

        self.a_star_button.clicked.connect(view_a_star)
        self.b_star_button.clicked.connect(view_b_star)
        self.c_star_button.clicked.connect(view_c_star)

    def connect_real_axes(self, view_a, view_b, view_c):
        """
        Real axes view connections.

        Parameters
        ----------
        view_a : function
            :math:`a`-axis view handler.
        view_b : function
            :math:`b`-axis view handler.
        view_c : function
            :math:`c`-axis view handler.

        """

        self.a_button.clicked.connect(view_a)
        self.b_button.clicked.connect(view_b)
        self.c_button.clicked.connect(view_c)

    def connect_rotate(self, cww, cw):
        """Connect Roll to presenter handler.

        Parameters
        ----------
        cww : function
            Handler for rolling counter-clockwise.
        cw : function
            Handler for rolling clockwise.
        """

        self.rotate_ccw_button.clicked.connect(cww)
        self.rotate_cw_button.clicked.connect(cw)

    def connect_elev(self, up, down):
        """Connect Elevate to presenter handler.

        Parameters
        ----------
        up : function
            Handler for tilting the camera up.
        down : function
            Handler for tilting the camera down.
        """

        self.elev_up_button.clicked.connect(up)
        self.elev_down_button.clicked.connect(down)

    def connect_az(self, left, right):
        """Connect Azimuth to presenter handler.

        Parameters
        ----------
        left : function
            Handler for rotating the camera left (azimuth).
        right : function
            Handler for rotating the camera right (azimuth).
        """

        self.az_left_button.clicked.connect(left)
        self.az_right_button.clicked.connect(right)

    def connect_save_screenshot(self, save_screenshot):
        """
        Screenshot connection.

        Parameters
        ----------
        save_screenshot : function
            Screenshot handler.

        """

        self.save_button.clicked.connect(save_screenshot)

    def connect_reciprocal_real_compass(self, change_lattice):
        """
        Reciprocal/real axis compass

        Parameters
        ----------
        change_lattice : function
            Lattice handler.

        """

        self.recip_box.clicked.connect(change_lattice)

    def change_projection(self):
        """
        Enable or disable parallel projection.

        """

        if self.proj_box.isChecked():
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()

    def reset_view(self, negative=False):
        """
        Reset the view to the default isometric orientation.

        Also caches the resulting camera position and emits
        ``cam_ready``.

        Parameters
        ----------
        negative : bool, optional
            If True, view from the opposite isometric direction, by
            default False.

        """

        self.plotter.reset_camera()
        self.plotter.view_isometric(negative=negative)
        self.camera_position = self.plotter.camera_position
        self.cam_ready.emit()

    def reset_camera(self):
        """
        Reset the camera.

        """

        self.plotter.reset_camera()
        self.cam_ready.emit()

    def clear_scene(self):
        """
        Clear plane widgets and actors from the plotter.

        Uses ``clear_actors()`` (not ``clear()``) so that scene lights
        are preserved and shading is not flattened. Also caches the
        current camera position (if one is already stored) so it can
        be restored later by :meth:`reset_scene`.
        """

        self.plotter.clear_plane_widgets()
        self.plotter.clear_actors()

        if self.camera_position is not None:
            self.camera_position = self.plotter.camera_position

    def reset_scene(self):
        """
        Restore the cached camera position, or reset the view.

        If a camera position was previously cached (see
        :meth:`clear_scene`/:meth:`reset_view`), it is reapplied to
        the plotter; otherwise the view is reset to the default
        isometric orientation via :meth:`reset_view`. Emits
        ``cam_ready`` when done.
        """

        if self.camera_position is not None:
            self.plotter.camera_position = self.camera_position
        else:
            self.reset_view()
        self.cam_ready.emit()

    def save_screenshot(self, filename):
        """
        Save plotter screenshot.

        Parameters
        ----------
        filename : str
            Filename with *.png extension.

        """

        self.plotter.screenshot(filename)

    _FILE_DIALOG_MAX_DEPTH = 3
    """int: Maximum number of path components remembered for the
    screenshot save-file dialog's starting directory."""

    def _get_file_dialog_dir(self):
        """
        Retrieve the last-used directory for the screenshot save dialog.

        Returns
        -------
        directory : str
            Previously remembered directory path, or an empty string
            if none has been stored yet.
        """

        settings = QSettings("NeuXtalViz", "NeuXtalViz")
        return settings.value("file_dialog_last_dir", "")

    def _remember_file_dialog_dir(self, path):
        """
        Persist a directory as the last-used screenshot dialog location.

        The path is truncated to at most
        ``_FILE_DIALOG_MAX_DEPTH + 1`` components before being stored,
        to avoid saving overly long paths.

        Parameters
        ----------
        path : str
            Directory path to remember. If falsy, nothing is stored.
        """

        if not path:
            return
        parts = pathlib.Path(path).parts
        if len(parts) > self._FILE_DIALOG_MAX_DEPTH + 1:
            path = str(pathlib.Path(*parts[: self._FILE_DIALOG_MAX_DEPTH + 1]))
        settings = QSettings("NeuXtalViz", "NeuXtalViz")
        settings.setValue("file_dialog_last_dir", path)

    def save_screenshot_file_dialog(self):
        """
        Prompt the user to choose a PNG file path for a screenshot.

        Opens a "Save PNG file" dialog seeded with the last
        remembered directory, and remembers the chosen directory for
        next time.

        Returns
        -------
        filename : str
            Selected file path, or an empty string if the user
            cancelled the dialog.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self,
            "Save PNG file",
            self._get_file_dialog_dir(),
            "PNG files (*.png)",
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def set_transform(self, T):
        """
        Apply a transform to the axes.

        Parameters
        ----------
        T : 3x3 2d array
            Trasformation matrix.

        """

        if self.axes_show():
            if T is not None:
                self.T = T
                self.show_axes()

    def reciprocal_lattice(self):
        """
        State of reciprocal lattice vectors.

        Returns
        -------
        checked : bool
            True if the reciprocal lattice toggle is checked (axes
            should be labeled a*/b*/c*), False otherwise (a/b/c).

        """

        return self.recip_box.isChecked()

    def show_axes(self):
        """
        Show or hide the coordinate axes actor in the plotter.

        If the axes checkbox is unchecked, the axes are hidden.
        Otherwise, if a transform :attr:`T` has been set, the axes
        actor is (re)drawn with labels chosen according to
        :meth:`reciprocal_lattice` (a*/b*/c* vs a/b/c) and oriented
        using the current transform matrix.
        """

        if not self.axes_show():
            self.plotter.hide_axes()
        elif self.T is not None:
            t = pv._vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    t.SetElement(i, j, self.T[i, j])
            if self.reciprocal_lattice():
                actor = self.plotter.add_axes(
                    xlabel="a*", ylabel="b*", zlabel="c*"
                )

            else:
                actor = self.plotter.add_axes(
                    xlabel="a", ylabel="b", zlabel="c"
                )
            actor.SetUserMatrix(t)

    def axes_show(self):
        """
        State of axes.

        Returns
        -------
        checked : bool
            True if the "Show Axes" checkbox is checked, False
            otherwise.

        """

        return self.axes_box.isChecked()

    def view_vector(self, vecs):
        """
        Set the camera according to given vector(s).

        Parameters
        ----------
        vecs : list of 2 or single 3 element 1d array-like
            Camera direction and optional upward vector.

        """

        if len(vecs) == 2:
            vec = np.cross(vecs[0], vecs[1])
            self.plotter.view_vector(vecs[0], vec)
        else:
            self.plotter.view_vector(vecs)

        self.cam_ready.emit()

    def view_up_vector(self, vec):
        """
        Set the camera according to given vector(s).

        Parameters
        ----------
        vec : 3 element 1d array-like
            Camera up direction.

        """

        self.plotter.set_viewup(vec)
        self.cam_ready.emit()

    def get_camera_state(self):
        """Return the current camera position, focal point, and up vectors.

        Returns
        -------
        position : 3-element ndarray of float
            Camera position in Cartesian coordinates.
        focal_point : 3-element ndarray of float
            Camera focal point in Cartesian coordinates.
        up : 3-element ndarray of float
            Camera up direction in Cartesian coordinates.
        """

        cam = self.plotter.camera
        position = np.array(cam.position, dtype=float)
        focal_point = np.array(cam.focal_point, dtype=float)
        up = np.array(cam.up, dtype=float)
        return position, focal_point, up

    def set_camera_state(self, position, focal_point, up):
        """Apply a new camera state to the underlying plotter and refresh.

        Re-renders the plotter and emits ``cam_ready`` afterward.

        Parameters
        ----------
        position : 3-element 1d array-like
            Camera position in Cartesian coordinates.
        focal_point : 3-element 1d array-like
            Camera focal point in Cartesian coordinates.
        up : 3-element 1d array-like
            Camera up direction in Cartesian coordinates.
        """

        cam = self.plotter.camera
        cam.position = tuple(position)
        cam.focal_point = tuple(focal_point)
        cam.up = tuple(up)

        self.plotter.render()
        self.cam_ready.emit()

    def get_rotate_step(self):
        """Return the rotation step in degrees from the rotate tab.

        Returns
        -------
        step : float
            Rotation step in degrees entered in the rotate tab, or
            5.0 if the current text is not a valid value.
        """

        if self.rotate_step_line.hasAcceptableInput():
            return float(self.rotate_step_line.text())
        else:
            return 5.0

    def connect_camera_ready(self, calculate):
        """
        Connect a callback to the ``cam_ready`` signal.

        The ``cam_ready`` signal is emitted whenever the camera state
        changes (e.g. after a view/up vector change or a camera
        reset). Presenters typically connect this to a handler that
        recomputes and displays roll/elevation/azimuth.

        Parameters
        ----------
        calculate : callable
            Slot to invoke whenever the camera is ready/updated.
        """

        self.cam_ready.connect(calculate)

    def get_camera_roll_direction(self):
        """
        Get the current camera roll angle and view direction.

        Returns
        -------
        roll : float
            Camera roll angle in degrees.
        direction : 3-element tuple of float
            Camera view direction vector.
        """

        return self.plotter.camera.roll, self.plotter.camera.direction

    def update_camera_display(self, roll, elevation, azimuth):
        """Update the read-only camera orientation display, if present.

        This shows the camera's roll, elevation, azimuth in degrees.

        Parameters
        ----------
        roll : float
            Camera roll angle in degrees.
        elevation : float
            Camera elevation angle in degrees.
        azimuth : float
            Camera azimuth angle in degrees.
        """

        text = "{:.1f},{:.1f},{:.1f}".format(roll, elevation, azimuth)

        self.camera_pos_line.setText(text)

    _MPL_CANVAS_NAMES = (
        "canvas_slice",
        "canvas_cut",
        "canvas_inst",
        "canvas_scan",
        "canvas_clust",
        "canvas_align",
    )
    """tuple of str: Matplotlib canvas attribute names to refresh in
    :meth:`update_theme`. Different tool views define different
    subsets of these; missing ones are skipped."""

    def update_theme(self):
        """
        Apply the selected PyVista/matplotlib plot theme to the view.

        Reads the current selection from the theme combo box,
        applies it as the global PyVista plot theme, updates the
        plotter background, re-renders, refreshes the axes actor, and
        refreshes the axis button icon colors to match the new theme.
        Selecting the "dark" theme also switches the whole application
        to a dark Qt stylesheet; any other theme selection switches
        back to the light stylesheet. Also applies a corresponding
        matplotlib style ("default" for "default"/"document", "dark"
        -> "dark_background", "paraview" -> "Solarize_Light2") and
        re-styles any matplotlib canvases already drawn on this view.
        """

        theme = self.theme_combo.currentText()
        pv.set_plot_theme(theme)
        bg = pv.global_theme.background
        self.plotter.set_background(bg)
        self.plotter.render()
        # call twice to update colors
        self.show_axes()
        self.show_axes()
        # Refresh axis button colors to match the active theme.
        self._init_axis_icons()

        is_dark = theme == "dark"
        app = QApplication.instance()
        qt_api = os.environ.get("QT_API")
        kwargs = {"qt_api": qt_api} if qt_api else {}
        app.setStyleSheet(
            qdarkstyle.load_stylesheet(
                **kwargs, palette=_DarkPalette if is_dark else _LightPalette
            )
        )
        app.setProperty("ui_dark", is_dark)

        plt.style.use(_MPL_STYLE_BY_THEME.get(theme, "default"))
        self._refresh_mpl_canvases()

    def _refresh_mpl_canvases(self):
        """
        Re-apply the current matplotlib style to already-drawn canvases.

        ``matplotlib.pyplot.style.use`` only affects figures/axes
        created afterward; existing ``Figure``/``Axes`` objects keep
        the colors they were created with. This re-applies the
        properties that make a style switch visible (face colors,
        spines, ticks, labels, and legends) to every matplotlib canvas
        already built on this view (see :attr:`_MPL_CANVAS_NAMES`),
        then redraws it.
        """

        face = plt.rcParams["figure.facecolor"]
        axes_face = plt.rcParams["axes.facecolor"]
        edge = plt.rcParams["axes.edgecolor"]
        label_color = plt.rcParams["axes.labelcolor"]
        tick_color = plt.rcParams["xtick.color"]
        text_color = plt.rcParams["text.color"]

        for name in self._MPL_CANVAS_NAMES:
            canvas = getattr(self, name, None)
            if canvas is None:
                continue
            figure = canvas.figure
            figure.set_facecolor(face)
            for ax in figure.axes:
                ax.set_facecolor(axes_face)
                for spine in ax.spines.values():
                    spine.set_color(edge)
                ax.tick_params(colors=tick_color, labelcolor=label_color)
                ax.xaxis.label.set_color(label_color)
                ax.yaxis.label.set_color(label_color)
                ax.title.set_color(text_color)
                legend = ax.get_legend()
                if legend is not None:
                    legend.get_frame().set_facecolor(axes_face)
                    legend.get_frame().set_edgecolor(edge)
            canvas.draw_idle()

    def update_labels(self):
        """
        Change the axes labels between Miller and fractional notation.

        """

        axes_type = self.view_combo.currentText()

        if axes_type == "[hkl]":
            self.axis1_label.setText("h")
            self.axis2_label.setText("k")
            self.axis3_label.setText("l")
        else:
            self.axis1_label.setText("u")
            self.axis2_label.setText("v")
            self.axis3_label.setText("w")

        axesup_type = self.viewup_combo.currentText()

        if axesup_type == "[hkl]":
            self.axisup1_label.setText("h")
            self.axisup2_label.setText("k")
            self.axisup3_label.setText("l")
        else:
            self.axisup1_label.setText("u")
            self.axisup2_label.setText("v")
            self.axisup3_label.setText("w")

    def get_manual_axis_indices(self):
        """
        Indices of manually entered direction components.

        Returns
        -------
        axes_type : str, [hkl] or [uvw]
            Miller index or fractional coordinate.
        ind : 3-element 1d array-like
            Indices.
        None
            Returned (implicitly) instead of the tuple above if any
            of the three axis line edits currently holds invalid
            input.

        """

        axes_type = self.view_combo.currentText()

        axes = [self.axis1_line, self.axis2_line, self.axis3_line]
        valid_axes = all([axis.hasAcceptableInput() for axis in axes])

        if valid_axes:
            axis1 = float(self.axis1_line.text())
            axis2 = float(self.axis2_line.text())
            axis3 = float(self.axis3_line.text())

            ind = np.array([axis1, axis2, axis3])

            return axes_type, ind

    def get_manual_axis_up_indices(self):
        """
        Indices of manually entered direction up components.

        Returns
        -------
        axes_type : str, [hkl] or [uvw]
            Miller index or fractional coordinate.
        ind : 3-element 1d array-like
            Indices.
        None
            Returned (implicitly) instead of the tuple above if any
            of the three up-axis line edits currently holds invalid
            input.

        """

        axes_type = self.viewup_combo.currentText()

        axes = [self.axisup1_line, self.axisup2_line, self.axisup3_line]
        valid_axes = all([axis.hasAcceptableInput() for axis in axes])

        if valid_axes:
            axis1 = float(self.axisup1_line.text())
            axis2 = float(self.axisup2_line.text())
            axis3 = float(self.axisup3_line.text())

            ind = np.array([axis1, axis2, axis3])

            return axes_type, ind

    def view_xy(self):
        """
        View :math:`xy`-plane.

        """

        self.plotter.view_vector([0, 0, 1], [0, 1, 0])

    def view_yz(self):
        """
        View :math:`yz`-plane.

        """

        self.plotter.view_vector([1, 0, 0], [0, 1, 0])

    def view_zx(self):
        """
        View :math:`zx`-plane.

        """

        self.plotter.view_vector([0, 1, 0], [0, 0, 1])

    def view_yx(self):
        """
        View :math:`yx`-plane.

        """

        self.plotter.view_vector([0, 0, -1], [0, 1, 0])

    def view_zy(self):
        """
        View :math:`zy`-plane.

        """

        self.plotter.view_vector([-1, 0, 0], [0, 1, 0])

    def view_xz(self):
        """
        View :math:`xz`-plane.

        """

        self.plotter.view_vector([0, -1, 0], [0, 0, 1])

    def set_position(self, pos):
        """
        Set the position.

        Parameters
        ----------
        pos : 3-element 1d array-like
            Coordinate position.

        """

        self.plotter.set_position(pos, reset=True)

    def stop_processes(self):
        """
        Stop all running worker processes.
        """
        self.threadpool.stop_all_workers()
        self.append_to_console("Stop requested for all running processes...\n")
        self.status_bar.showMessage("Stopping processes...")
