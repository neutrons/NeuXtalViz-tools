import os

from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QTabWidget,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QCheckBox,
    QSlider,
    QListWidget,
    QMessageBox,
    QInputDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSpinBox,
)

from qtpy.QtGui import QDoubleValidator
from qtpy.QtCore import Qt, Signal, QEvent

import numpy as np
import pyvista as pv
from matplotlib import colors as mcolors

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D

from mpl_toolkits.axisartist import Axes, GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import (
    ExtremeFinderSimple,
    MaxNLocator,
)

from NeuXtalViz.views.base_view import NeuXtalVizWidget
from NeuXtalViz.config import colormap

import qtawesome as qta

colormap.add_modified()

cmaps = {
    "Sequential": "viridis",
    "Binary": "binary",
    "Diverging": "bwr",
    "Rainbow": "turbo",
    "Modified": "modified",
}

# V/U-shaped custom transfer functions:
_z = np.abs(np.linspace(-1, 1, 256))
_linear_symmetric_opacity = _z * 255
_sigmoid_symmetric_opacity = 0.5 * (1 - np.cos(np.pi * _z)) * 255
_geom_symmetric_opacity = np.concatenate(
    [np.geomspace(255, 1e-6, 128), np.geomspace(1e-6, 255, 128)]
)

opacities = {
    "None": 1.0,
    "Linear Ascending": "linear",
    "Linear Descending": "linear_r",
    "Geometric Ascending": "geom",
    "Geometric Descending": "geom_r",
    "Sigmoid Ascending": "sigmoid",
    "Sigmoid Descending": "sigmoid_r",
    "Linear Symmetric": _linear_symmetric_opacity,
    "Sigmoid Symmetric": _sigmoid_symmetric_opacity,
    "Geometric Symmetric": _geom_symmetric_opacity,
}


class _SliceNavigationToolbar(NavigationToolbar2QT):
    """
    Slice-plot toolbar that flags interactive use of the "Home" button.

    The toolbar's actions are bound to ``self.home`` (etc.) at
    construction time, so intercepting the click requires overriding
    the method itself rather than reassigning it on the instance
    afterward. ``view._slice_home_clicked`` is set for the duration of
    the reset so :meth:`VolumeSlicerView.slice_limits` can tell it
    apart from an interactive pan/zoom.

    ``update_slice`` rebuilds ``ax_slice`` from scratch on every
    redraw (required for its curvilinear grid), which orphans
    matplotlib's own navigation stack -- it is keyed to specific
    ``Axes`` instances, so ``NavigationToolbar2.home()`` silently does
    nothing once any redraw has happened since the stack was last
    populated. To keep "Home" reliable regardless, this also directly
    resets the view to the full data extent via the view's own
    "Auto Zoom" logic rather than depending on that stack.
    """

    def __init__(self, canvas, parent, view):
        super().__init__(canvas, parent)
        self._view = view

    def home(self, *args, **kwargs):
        view = self._view
        view._slice_home_clicked = True
        try:
            super().home(*args, **kwargs)
        finally:
            view._slice_home_clicked = False

        view.auto_zoom_box.blockSignals(True)
        view.auto_zoom_box.setChecked(True)
        view.auto_zoom_box.blockSignals(False)
        view._handle_auto_zoom_toggle(True)


class VolumeSlicerView(NeuXtalVizWidget):
    """
    View for the volume slicer tool.

    Provides the UI for loading a NeXus MDHisto workspace, rendering it as
    an interactive 3D slice plane, slicing it into a 2D plane, and cutting
    a 1D line profile through that slice, together with controls for
    scaling, colormaps, opacity, and display/color limits.

    Attributes
    ----------
    slice_ready : qtpy.QtCore.Signal
        Emitted when the slice position/parameters are ready to be
        recomputed (e.g. after dragging the 3D slice plane or slider).
    cut_ready : qtpy.QtCore.Signal
        Emitted when the cut position/parameters are ready to be
        recomputed (e.g. after dragging the line cut on the slice plot).
    """

    slice_ready = Signal()
    cut_ready = Signal()

    def __init__(self, parent=None):
        """
        Initialize the volume slicer view and its internal state.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (default None).
        """
        super().__init__(parent)

        self.tab_widget = QTabWidget(self)

        self.slice_tab()
        self.transform_tab()

        self.layout().addWidget(self.tab_widget, stretch=1)

        self.reset_slice_cut()

        self._cx = None
        self._sx = None
        self._sy = None
        self.space = "reciprocal"
        self._slice_home_clicked = False
        self.slice_im = None
        self.xlim = None
        self.ylim = None
        self._cut_lines = None
        self._slice_zoom_xlim = None
        self._slice_zoom_ylim = None
        self._volume_limits = None
        self._volume_nbins = None

    def slice_tab(self):
        """
        Build the "Slice" tab, its widgets, and its layouts.

        Creates the volume/opacity/colormap controls, slice and cut
        position/thickness controls, axis limit fields, the 2D slice and
        1D cut matplotlib canvases, and wires up the toggle/auto-zoom
        checkboxes.
        """
        slice_tab = QWidget()
        self.tab_widget.addTab(slice_tab, "Slice")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.container = QWidget()
        self.container.setVisible(False)

        plots_layout = QVBoxLayout()
        slice_params_layout = QHBoxLayout()
        view_params_layout = QGridLayout()
        collabsible_layout = QVBoxLayout(self.container)
        cut_params_layout = QHBoxLayout()
        draw_layout = QHBoxLayout()

        self.vol_scale_combo = QComboBox(self)
        self.vol_scale_combo.addItem("Linear")
        self.vol_scale_combo.addItem("Log")
        self.vol_scale_combo.setCurrentIndex(0)
        self.vol_scale_combo.setToolTip(
            "Select the scaling for the volume data (Linear or Logarithmic)."
        )
        self.auto_scale_dropdown(self.vol_scale_combo)

        self.opacity_combo = QComboBox(self)
        self.opacity_combo.addItem("None")
        self.opacity_combo.addItem("Linear Ascending")
        self.opacity_combo.addItem("Linear Descending")
        self.opacity_combo.addItem("Linear Symmetric")
        self.opacity_combo.addItem("Geometric Ascending")
        self.opacity_combo.addItem("Geometric Descending")
        self.opacity_combo.addItem("Geometric Symmetric")
        self.opacity_combo.addItem("Sigmoid Ascending")
        self.opacity_combo.addItem("Sigmoid Descending")
        self.opacity_combo.addItem("Sigmoid Symmetric")
        self.opacity_combo.setCurrentIndex(0)
        self.opacity_combo.setToolTip(
            "Choose the opacity mapping for the 3D slice planes. "
            '"None" (default) is fully opaque everywhere, with no '
            'value-based transparency. "Ascending" is transparent at '
            'the minimum and opaque at the maximum ("Descending" the '
            'reverse). The "Symmetric" options are instead fully '
            "opaque at both the minimum and maximum and transparent "
            "in the middle -- useful for signed data such as a "
            'delta-PDF result -- with "Linear" a sharp V-shape, '
            '"Sigmoid" a smooth ease in/out, and "Geometric" staying '
            "low across most of the range before rising sharply only "
            "very near the extremes."
        )
        self.auto_scale_dropdown(self.opacity_combo)

        self.clim_combo = QComboBox(self)
        self.clim_combo.addItem("Min/Max")
        self.clim_combo.addItem("μ±3×σ")
        self.clim_combo.addItem("Q₃/Q₁±1.5×IQR")
        self.clim_combo.setCurrentIndex(2)
        self.clim_combo.setToolTip(
            "Choose the method for setting color limits for the slice."
        )
        self.auto_scale_dropdown(self.clim_combo)

        self.vlim_combo = QComboBox(self)
        self.vlim_combo.addItem("Min/Max")
        self.vlim_combo.addItem("μ±3×σ")
        self.vlim_combo.addItem("Q₃/Q₁±1.5×IQR")
        self.vlim_combo.setCurrentIndex(2)
        self.vlim_combo.setToolTip(
            "Choose the method for setting value limits for the cut."
        )
        self.auto_scale_dropdown(self.vlim_combo)

        self.cbar_combo = QComboBox(self)
        self.cbar_combo.addItem("Sequential")
        self.cbar_combo.addItem("Rainbow")
        self.cbar_combo.addItem("Binary")
        self.cbar_combo.addItem("Modified")
        self.cbar_combo.addItem("Diverging")
        self.cbar_combo.setToolTip(
            "Select the colormap for the slice visualization. "
            '"Diverging" is a good fit for signed data centered about '
            "zero, e.g. a 3D-ΔPDF result."
        )
        self.auto_scale_dropdown(self.cbar_combo)

        self.load_NXS_button = QPushButton("Load NXS", self)
        self.load_NXS_button.setToolTip(
            "Load a NeXus (NXS) file for volume slicing."
        )
        self.load_NXS_button.setIcon(qta.icon("fa6s.folder-open"))

        self.workspace_combo = QComboBox(self)
        self.workspace_combo.setToolTip(
            "The workspace currently being sliced/cut. Load new "
            "workspaces, rename/delete them, and derive new ones "
            '(arithmetic, Bragg punch, KAREN, 3D-ΔPDF) from the "3D-ΔPDF" '
            "tab."
        )
        self.auto_scale_dropdown(self.workspace_combo)

        self.redraw_workspace_button = QPushButton("Redraw", self)
        self.redraw_workspace_button.setIcon(qta.icon("fa6s.arrows-rotate"))
        self.redraw_workspace_button.setToolTip(
            "Redraw the current workspace. Use this if its data "
            "changed underneath it -- e.g. after re-running Punch, "
            "Filter, Blur, or Transform with the same output name "
            "while it's the active workspace."
        )

        draw_layout.addWidget(self.vol_scale_combo)
        draw_layout.addWidget(self.opacity_combo)
        draw_layout.addWidget(self.clim_combo)
        draw_layout.addWidget(self.cbar_combo)
        draw_layout.addWidget(self.load_NXS_button)
        draw_layout.addWidget(self.workspace_combo)
        draw_layout.addWidget(self.redraw_workspace_button)

        self.slice_combo = QComboBox(self)
        self.slice_combo.addItem("Axis 1/2")
        self.slice_combo.addItem("Axis 1/3")
        self.slice_combo.addItem("Axis 2/3")
        self.slice_combo.setCurrentIndex(0)
        self.slice_combo.setToolTip("Select the plane for slicing the volume.")
        self.auto_scale_dropdown(self.slice_combo)

        self.cut_combo = QComboBox(self)
        self.cut_combo.addItem("Axis 1")
        self.cut_combo.addItem("Axis 2")
        self.cut_combo.setCurrentIndex(0)
        self.cut_combo.setToolTip(
            "Select the axis for cutting through the slice."
        )
        self.auto_scale_dropdown(self.cut_combo)

        slice_label = QLabel("Slice:", self)
        cut_label = QLabel("Cut:", self)

        self.slice_line = QLineEdit("0.0")
        self.slice_line.setValidator(validator)
        self.slice_line.setToolTip(
            "Set the position of the slice along the selected plane."
        )

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setToolTip("Drag to move the slice position.")
        self._slice_smin = 0.0
        self._slice_smax = 1.0
        self._slice_step = 0.001
        self._slice_steps = 1000
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self._slice_steps)
        self.slice_slider.valueChanged.connect(self._on_slice_slider_changed)
        self.slice_slider.sliderReleased.connect(self.slice_ready)
        self.slice_slider.installEventFilter(self)

        self.cut_line = QLineEdit("0.0")
        self.cut_line.setValidator(validator)
        self.cut_line.setToolTip(
            "Set the position of the cut along the selected axis."
        )

        validator = QDoubleValidator(0.0001, 100, 5, notation=notation)

        slice_thickness_label = QLabel("Thickness:", self)
        cut_thickness_label = QLabel("Thickness:", self)

        self.slice_thickness_line = QLineEdit("0.1")
        self.slice_thickness_line.setValidator(validator)
        self.slice_thickness_line.setToolTip("Set the thickness of the slice.")
        self.cut_thickness_line = QLineEdit("0.5")
        self.cut_thickness_line.setValidator(validator)
        self.cut_thickness_line.setToolTip("Set the thickness of the cut.")

        self.slice_scale_combo = QComboBox(self)
        self.slice_scale_combo.addItem("Linear")
        self.slice_scale_combo.addItem("Log")
        self.slice_scale_combo.addItem("Symlog")
        self.slice_scale_combo.setToolTip(
            "Select the scale for the slice plot (Linear, Logarithmic, "
            'or "Symlog": linear near zero, logarithmic away from it -- '
            "a good fit for signed data such as a 3D-ΔPDF result)."
        )
        self.auto_scale_dropdown(self.slice_scale_combo)

        self.cut_scale_combo = QComboBox(self)
        self.cut_scale_combo.addItem("Linear")
        self.cut_scale_combo.addItem("Log")
        self.cut_scale_combo.addItem("Symlog")
        self.cut_scale_combo.setToolTip(
            "Select the scale for the cut plot (Linear, Logarithmic, "
            "or Symlog)."
        )
        self.auto_scale_dropdown(self.cut_scale_combo)

        self.vmin_line = QLineEdit("")
        self.vmax_line = QLineEdit("")
        self.auto_limits_box = QCheckBox("Auto Limits")
        self.auto_limits_box.setChecked(True)
        self.auto_limits_box.setToolTip(
            "Automatically reset slice limits on redraw. Uncheck to reuse the current limits."
        )
        self.auto_zoom_box = QCheckBox("Auto Zoom")
        self.auto_zoom_box.setChecked(True)
        self.auto_zoom_box.setToolTip(
            "Automatically reset zoom to full extent on redraw. Uncheck to retain zoom when possible."
        )

        self.xmin_line = QLineEdit("")
        self.xmax_line = QLineEdit("")

        self.ymin_line = QLineEdit("")
        self.ymax_line = QLineEdit("")

        self.vmin_line.setPlaceholderText("auto")
        self.vmax_line.setPlaceholderText("auto")
        self.xmin_line.setPlaceholderText("auto")
        self.xmax_line.setPlaceholderText("auto")
        self.ymin_line.setPlaceholderText("auto")
        self.ymax_line.setPlaceholderText("auto")

        validator = QDoubleValidator(-1e32, 1e32, 6, notation=notation)

        # Scientific notation (e.g. "1e-4") is common for diffuse-scattering
        # and 3D-ΔPDF color limits, which can span many orders of magnitude.
        vlim_validator = QDoubleValidator(
            -1e32, 1e32, 6, notation=QDoubleValidator.ScientificNotation
        )
        self.vmin_line.setValidator(vlim_validator)
        self.vmax_line.setValidator(vlim_validator)
        self.vmin_line.setToolTip("Set the minimum value for the colorbar.")
        self.vmax_line.setToolTip("Set the maximum value for the colorbar.")

        self.xmin_line.setValidator(validator)
        self.xmax_line.setValidator(validator)
        self.xmin_line.setToolTip("Set the minimum value for the X axis.")
        self.xmax_line.setToolTip("Set the maximum value for the X axis.")

        self.ymin_line.setValidator(validator)
        self.ymax_line.setValidator(validator)
        self.ymin_line.setToolTip("Set the minimum value for the Y axis.")
        self.ymax_line.setToolTip("Set the maximum value for the Y axis.")

        xmin_label = QLabel("X Min:", self)
        xmax_label = QLabel("X Max:", self)

        ymin_label = QLabel("Y Min:", self)
        ymax_label = QLabel("Y Max:", self)

        vmin_label = QLabel("V Min:", self)
        vmax_label = QLabel("V Max:", self)

        self.save_slice_button = QPushButton("Save Slice", self)
        self.save_slice_button.setToolTip(
            "Save the current slice as a CSV file."
        )
        self.save_slice_button.setIcon(qta.icon("fa6s.floppy-disk"))
        self.save_cut_button = QPushButton("Save Cut", self)
        self.save_cut_button.setToolTip("Save the current cut as a CSV file.")
        self.save_cut_button.setIcon(qta.icon("fa6s.floppy-disk"))

        self.toggle_line_box = QCheckBox("Show Line Cut")
        self.toggle_line_box.setChecked(False)
        self.toggle_line_box.setToolTip(
            "Show or hide the line cut overlay on the slice plot."
        )

        self.symmetric_zero_box = QCheckBox("Symmetric About Zero")
        self.symmetric_zero_box.setChecked(False)
        self.symmetric_zero_box.setToolTip(
            "Force the colorbar limits to be symmetric about zero "
            "(vmax = -vmin). Useful for signed data such as a "
            "3D-ΔPDF result."
        )

        self.symmetric_xy_box = QCheckBox("Symmetric X/Y")
        self.symmetric_xy_box.setChecked(False)
        self.symmetric_xy_box.setToolTip(
            "Keep the X/Y axis limits symmetric about zero. While "
            "checked, editing X Min or X Max sets the other to its "
            "negative (and likewise for Y Min/Y Max)."
        )

        slice_params_layout.addWidget(self.slice_combo)
        slice_params_layout.addWidget(slice_label)
        slice_params_layout.addWidget(self.slice_line)
        slice_params_layout.addWidget(slice_thickness_label)
        slice_params_layout.addWidget(self.slice_thickness_line)
        slice_params_layout.addWidget(self.auto_limits_box)
        slice_params_layout.addWidget(self.auto_zoom_box)
        slice_params_layout.addWidget(self.slice_scale_combo)
        slice_params_layout.addWidget(self.save_slice_button)

        view_params_layout.addWidget(xmin_label, 1, 0)
        view_params_layout.addWidget(self.xmin_line, 1, 1)
        view_params_layout.addWidget(xmax_label, 0, 0)
        view_params_layout.addWidget(self.xmax_line, 0, 1)
        view_params_layout.addWidget(ymin_label, 1, 2)
        view_params_layout.addWidget(self.ymin_line, 1, 3)
        view_params_layout.addWidget(ymax_label, 0, 2)
        view_params_layout.addWidget(self.ymax_line, 0, 3)

        view_params_layout.addWidget(self.vlim_combo, 0, 4)
        view_params_layout.addWidget(self.symmetric_xy_box, 1, 4)
        view_params_layout.addWidget(vmin_label, 1, 5)
        view_params_layout.addWidget(self.vmin_line, 1, 6)
        view_params_layout.addWidget(vmax_label, 0, 5)
        view_params_layout.addWidget(self.vmax_line, 0, 6)
        view_params_layout.addWidget(self.symmetric_zero_box, 0, 7)
        view_params_layout.addWidget(self.toggle_line_box, 1, 7)

        cut_params_layout.addWidget(self.cut_combo)
        cut_params_layout.addWidget(cut_label)
        cut_params_layout.addWidget(self.cut_line)
        cut_params_layout.addWidget(cut_thickness_label)
        cut_params_layout.addWidget(self.cut_thickness_line)
        cut_params_layout.addStretch(1)
        cut_params_layout.addWidget(self.cut_scale_combo)
        cut_params_layout.addWidget(self.save_cut_button)

        plots_layout.addLayout(draw_layout)

        self.canvas_slice = FigureCanvas(Figure(constrained_layout=True))
        self.canvas_slice.setFocusPolicy(Qt.StrongFocus)
        self.canvas_slice.installEventFilter(self)
        self.canvas_cut = FigureCanvas(
            Figure(constrained_layout=True, figsize=(6.4, 3.2))
        )

        image_layout = QHBoxLayout()
        line_layout = QHBoxLayout()

        fig_2d_layout = QVBoxLayout()
        fig_1d_layout = QVBoxLayout()

        self.slice_toolbar = _SliceNavigationToolbar(
            self.canvas_slice, self, self
        )
        fig_2d_layout.addWidget(self.slice_toolbar)
        fig_2d_layout.addWidget(self.canvas_slice)

        fig_1d_layout.addWidget(NavigationToolbar2QT(self.canvas_cut, self))
        fig_1d_layout.addWidget(self.canvas_cut)

        image_layout.addLayout(fig_2d_layout)

        line_layout.addLayout(fig_1d_layout)

        plots_layout.addLayout(image_layout)
        plots_layout.addWidget(self.slice_slider)
        plots_layout.addLayout(slice_params_layout)
        plots_layout.addLayout(view_params_layout)

        collabsible_layout.addLayout(line_layout)
        collabsible_layout.addLayout(cut_params_layout)

        plots_layout.addWidget(self.container)

        self.fig_slice = self.canvas_slice.figure
        self.fig_cut = self.canvas_cut.figure

        self.ax_slice = self.fig_slice.subplots(1, 1)
        self.ax_cut = self.fig_cut.subplots(1, 1)

        self.cb = None

        slice_tab.setLayout(plots_layout)

        self.toggle_line_box.toggled.connect(self.toggle_container)
        self.auto_zoom_box.toggled.connect(self._handle_auto_zoom_toggle)

    def transform_tab(self):
        """
        Build the "3D-ΔPDF" tab, its widgets, and its layouts.

        Creates the workspace management list (rename/delete), the
        arithmetic panel (``a*ws_a - b*ws_b``), the crystal-system/
        space-group cascade, the Bragg-punch controls, the blur
        controls, and the 3D-ΔPDF calculation controls.
        """
        transform_tab = QWidget()
        self.tab_widget.addTab(transform_tab, "Transform")

        notation = QDoubleValidator.StandardNotation
        coeff_validator = QDoubleValidator(-1e6, 1e6, 6, notation=notation)
        q_validator = QDoubleValidator(0, 1e6, 6, notation=notation)

        layout = QVBoxLayout()

        # --- Workspace management -----------------------------------
        manage_layout = QVBoxLayout()

        self.workspace_list_widget = QListWidget(self)
        self.workspace_list_widget.setToolTip(
            "All loaded/derived workspaces. Select one to rename, "
            "delete, or save it."
        )

        manage_button_layout = QHBoxLayout()
        self.rename_workspace_button = QPushButton("Rename", self)
        self.rename_workspace_button.setToolTip(
            "Rename the selected workspace."
        )
        self.rename_workspace_button.setIcon(qta.icon("fa6s.pen"))
        self.delete_workspace_button = QPushButton("Delete", self)
        self.delete_workspace_button.setToolTip(
            "Delete the selected workspace."
        )
        self.delete_workspace_button.setIcon(qta.icon("fa6s.trash"))
        self.save_workspace_button = QPushButton("Save", self)
        self.save_workspace_button.setToolTip(
            "Save the selected workspace as an MD (.nxs) file."
        )
        self.save_workspace_button.setIcon(qta.icon("fa6s.floppy-disk"))
        manage_button_layout.addWidget(self.rename_workspace_button)
        manage_button_layout.addWidget(self.delete_workspace_button)
        manage_button_layout.addWidget(self.save_workspace_button)
        manage_button_layout.addStretch(1)

        manage_layout.addWidget(self.workspace_list_widget)
        manage_layout.addLayout(manage_button_layout)

        # --- Arithmetic ------------------------------------------------
        combine_layout = QHBoxLayout()

        self.combine_ws_a_combo = QComboBox(self)
        self.combine_ws_a_combo.setToolTip("Workspace A.")
        self.combine_coeff_a_line = QLineEdit("1.0", self)
        self.combine_coeff_a_line.setValidator(coeff_validator)
        self.combine_coeff_a_line.setToolTip("Coefficient a.")

        self.combine_ws_b_combo = QComboBox(self)
        self.combine_ws_b_combo.setToolTip("Workspace B.")
        self.combine_coeff_b_line = QLineEdit("1.0", self)
        self.combine_coeff_b_line.setValidator(coeff_validator)
        self.combine_coeff_b_line.setToolTip("Coefficient b.")

        self.combine_output_line = QLineEdit(self)
        self.combine_output_line.setPlaceholderText("output name")
        self.combine_output_line.setToolTip(
            "Display name for the combined result."
        )

        self.combine_button = QPushButton("Subtract", self)
        self.combine_button.setToolTip(
            "Compute a×A − b×B and register the result as a "
            "new workspace. The two workspaces must share common "
            "binning."
        )
        self.combine_button.setIcon(qta.icon("fa6s.calculator"))

        combine_layout.addWidget(self.combine_coeff_a_line)
        combine_layout.addWidget(QLabel("×", self))
        combine_layout.addWidget(self.combine_ws_a_combo)
        combine_layout.addWidget(QLabel("-", self))
        combine_layout.addWidget(self.combine_coeff_b_line)
        combine_layout.addWidget(QLabel("×", self))
        combine_layout.addWidget(self.combine_ws_b_combo)
        combine_layout.addWidget(QLabel("=", self))
        combine_layout.addWidget(self.combine_output_line)
        combine_layout.addWidget(self.combine_button)

        # --- Crystal system / space group / setting ---------------------
        crystal_layout = QHBoxLayout()

        self.pdf_crystal_system_combo = QComboBox(self)
        self.pdf_crystal_system_combo.addItem("Triclinic")
        self.pdf_crystal_system_combo.addItem("Monoclinic")
        self.pdf_crystal_system_combo.addItem("Orthorhombic")
        self.pdf_crystal_system_combo.addItem("Tetragonal")
        self.pdf_crystal_system_combo.addItem("Trigonal")
        self.pdf_crystal_system_combo.addItem("Hexagonal")
        self.pdf_crystal_system_combo.addItem("Cubic")
        self.pdf_crystal_system_combo.setToolTip(
            "Select the crystal system for the Bragg punch."
        )

        self.pdf_space_group_combo = QComboBox(self)
        self.pdf_space_group_combo.setToolTip(
            "Select the space group for the Bragg punch."
        )

        self.auto_scale_dropdown(self.pdf_crystal_system_combo)
        self.auto_scale_dropdown(self.pdf_space_group_combo)

        crystal_layout.addWidget(self.pdf_crystal_system_combo)
        crystal_layout.addWidget(self.pdf_space_group_combo)
        crystal_layout.addStretch(1)

        # --- Bragg punch / KAREN / Blur / Calculate 3D-ΔPDF -------------
        # One shared grid so the input combo, fields, units, "->", output
        # name, and button line up in columns across all four steps.
        steps_layout = QGridLayout()

        self.punch_input_combo = QComboBox(self)
        self.punch_input_combo.setToolTip("Workspace to punch.")

        self.punch_output_line = QLineEdit("punched", self)
        self.punch_output_line.setToolTip(
            "Display name for the punched result."
        )

        self.punch_q_size_line = QLineEdit("0.15", self)
        self.punch_q_size_line.setValidator(q_validator)
        self.punch_q_size_line.setToolTip(
            "Radius (Å⁻¹) of the region punched out around "
            "each allowed reflection."
        )

        self.punch_q_inner_line = QLineEdit("0.5", self)
        self.punch_q_inner_line.setValidator(q_validator)
        self.punch_q_inner_line.setToolTip(
            "Inner radius (Å⁻¹) below which low-Q signal " "is removed."
        )

        self.punch_outlier_line = QLineEdit("1.5", self)
        self.punch_outlier_line.setValidator(coeff_validator)
        self.punch_outlier_line.setToolTip(
            "Outlier scale factor (unitless) applied to each "
            "reflection's local interquartile range (IQR): voxels "
            "within the punch radius are only removed if they fall "
            "outside [Q1 - factor*IQR, Q3 + factor*IQR] for that "
            "local region (Tukey's fences; 1.5 is the conventional "
            "default), so genuine diffuse scattering near a Bragg "
            "peak is kept."
        )

        crystal_layout.addWidget(QLabel("Outlier:", self))
        crystal_layout.addWidget(self.punch_outlier_line)

        self.run_punch_button = QPushButton("Punch", self)
        self.run_punch_button.setToolTip(
            "Punch out statistical outliers (local IQR-based) within "
            "an ellipsoidal region around each allowed reflection, "
            "and remove the low-Q region, producing a new, "
            "separately inspectable workspace."
        )
        self.run_punch_button.setIcon(qta.icon("fa6s.bullseye"))

        steps_layout.addWidget(self.punch_input_combo, 0, 0)
        steps_layout.addWidget(QLabel("Size:", self), 0, 1)
        steps_layout.addWidget(self.punch_q_size_line, 0, 2)
        steps_layout.addWidget(QLabel("Inner:", self), 0, 3)
        steps_layout.addWidget(self.punch_q_inner_line, 0, 4)
        steps_layout.addWidget(QLabel("Å⁻¹", self), 0, 5)
        steps_layout.addWidget(QLabel("→", self), 0, 6)
        steps_layout.addWidget(self.punch_output_line, 0, 7)
        steps_layout.addWidget(self.run_punch_button, 0, 8)

        self.karen_input_combo = QComboBox(self)
        self.karen_input_combo.setToolTip("Workspace to filter with KAREN.")

        self.karen_output_line = QLineEdit("filtered", self)
        self.karen_output_line.setToolTip(
            "Display name for the KAREN-filtered result."
        )

        self.karen_width_line = QLineEdit("0.1", self)
        self.karen_width_line.setValidator(q_validator)
        self.karen_width_line.setToolTip(
            "Moving-window size (Å⁻¹) for the median/MAD outlier "
            "filter -- the same meaning as the blur step's size below."
        )

        self.karen_z_score_line = QLineEdit("3", self)
        self.karen_z_score_line.setValidator(q_validator)
        self.karen_z_score_line.setToolTip(
            "Outlier cutoff, in estimated standard deviations "
            "(1.4826*MAD) from the local median (3 is Mantid's "
            "DeltaPDF3D default)."
        )

        self.run_karen_button = QPushButton("Filter", self)
        self.run_karen_button.setToolTip(
            "Replace Bragg-peak/local outliers with a robust "
            "median-based estimate (Mantid's KAREN method), producing "
            "a new, separately inspectable workspace. An alternative "
            "to Punch + Blur that needs no separate fill step."
        )
        self.run_karen_button.setIcon(qta.icon("fa6s.filter"))

        steps_layout.addWidget(self.karen_input_combo, 1, 0)
        steps_layout.addWidget(QLabel("Z-score:", self), 1, 1)
        steps_layout.addWidget(self.karen_z_score_line, 1, 2)
        steps_layout.addWidget(QLabel("Width:", self), 1, 3)
        steps_layout.addWidget(self.karen_width_line, 1, 4)
        steps_layout.addWidget(QLabel("Å⁻¹", self), 1, 5)
        steps_layout.addWidget(QLabel("→", self), 1, 6)
        steps_layout.addWidget(self.karen_output_line, 1, 7)
        steps_layout.addWidget(self.run_karen_button, 1, 8)

        self.blur_input_combo = QComboBox(self)
        self.blur_input_combo.setToolTip(
            "Workspace to blur (typically a Bragg-punch result)."
        )

        self.blur_output_line = QLineEdit("blurred", self)
        self.blur_output_line.setToolTip(
            "Display name for the blurred result."
        )

        self.blur_q_blur_line = QLineEdit("0.05", self)
        self.blur_q_blur_line.setValidator(q_validator)
        self.blur_q_blur_line.setToolTip(
            "Gaussian blur size (Å⁻¹) used to fill in "
            "punched/cut regions before the transform."
        )

        self.run_blur_button = QPushButton("Blur", self)
        self.run_blur_button.setToolTip(
            "NaN-Gaussian-blur the gaps closed, producing a new, "
            "separately inspectable workspace."
        )
        self.run_blur_button.setIcon(qta.icon("fa6s.blender"))

        steps_layout.addWidget(self.blur_input_combo, 2, 0)
        steps_layout.addWidget(QLabel("Blur:", self), 2, 3)
        steps_layout.addWidget(self.blur_q_blur_line, 2, 4)
        steps_layout.addWidget(QLabel("Å⁻¹", self), 2, 5)
        steps_layout.addWidget(QLabel("→", self), 2, 6)
        steps_layout.addWidget(self.blur_output_line, 2, 7)
        steps_layout.addWidget(self.run_blur_button, 2, 8)

        self.pdf_input_combo = QComboBox(self)
        self.pdf_input_combo.setToolTip(
            "Workspace to transform (typically a blurred result)."
        )

        self.pdf_output_line = QLineEdit("transformed", self)
        self.pdf_output_line.setToolTip("Display name for the 3D-ΔPDF result.")

        self.pdf_q_outer_line = QLineEdit("5", self)
        self.pdf_q_outer_line.setValidator(q_validator)
        self.pdf_q_outer_line.setToolTip(
            "Padding extent (Å⁻¹) applied before the "
            "Fourier transform to real space."
        )

        self.pdf_window_combo = QComboBox(self)
        self.pdf_window_combo.addItem("None")
        self.pdf_window_combo.addItem("Lorch")
        self.pdf_window_combo.addItem("Hann")
        self.pdf_window_combo.setToolTip(
            "Apodization window applied before the Fourier transform, "
            "to suppress series-termination ripples from the padded "
            'cutoff. "Lorch" is the conventional choice for PDF data.'
        )
        self.auto_scale_dropdown(self.pdf_window_combo)

        self.calculate_pdf_button = QPushButton("Transform", self)
        self.calculate_pdf_button.setToolTip(
            "Run the pad/FFT step, producing a real-space 3D-ΔPDF " "result."
        )
        self.calculate_pdf_button.setIcon(qta.icon("fa6s.wave-square"))

        steps_layout.addWidget(self.pdf_input_combo, 3, 0)
        steps_layout.addWidget(QLabel("Window:", self), 3, 1)
        steps_layout.addWidget(self.pdf_window_combo, 3, 2)
        steps_layout.addWidget(QLabel("Outer:", self), 3, 3)
        steps_layout.addWidget(self.pdf_q_outer_line, 3, 4)
        steps_layout.addWidget(QLabel("Å⁻¹", self), 3, 5)
        steps_layout.addWidget(QLabel("→", self), 3, 6)
        steps_layout.addWidget(self.pdf_output_line, 3, 7)
        steps_layout.addWidget(self.calculate_pdf_button, 3, 8)

        layout.addLayout(manage_layout)
        layout.addLayout(combine_layout)
        layout.addLayout(crystal_layout)
        layout.addLayout(steps_layout)
        layout.addLayout(self._bond_network_layout())
        layout.addStretch(1)

        transform_tab.setLayout(layout)

    def _bond_network_layout(self):
        """
        Build the bond-network viewer section of the "3D-ΔPDF" tab.

        Creates the "Load CIF" button/filename display, the checkable
        atom-site table, the supercell-extent spin boxes, the bond
        tolerance field, and the "Plot Bond Network" button.

        Returns
        -------
        layout : QVBoxLayout
            Layout containing the bond-network viewer controls.
        """
        layout = QVBoxLayout()

        header_layout = QHBoxLayout()

        self.load_bond_cif_button = QPushButton("Load CIF...", self)
        self.load_bond_cif_button.setToolTip(
            "Load a crystal structure (CIF) to build a supercell bond "
            "network, drawn in the shared 3D view."
        )
        self.load_bond_cif_button.setIcon(qta.icon("fa6s.folder-open"))

        self.bond_cif_file_label = QLabel("No file loaded", self)

        header_layout.addWidget(self.load_bond_cif_button)
        header_layout.addWidget(self.bond_cif_file_label)
        header_layout.addStretch(1)

        self.bond_atom_table = QTableWidget(self)
        self.bond_atom_table.setColumnCount(6)
        self.bond_atom_table.setHorizontalHeaderLabels(
            ["Use", "Label", "Element", "x", "y", "z"]
        )
        self.bond_atom_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.bond_atom_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.bond_atom_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.bond_atom_table.setToolTip(
            "Asymmetric-unit sites read from the loaded CIF. Uncheck a "
            "row to exclude that site -- and all of its "
            "symmetry-equivalent copies -- from the supercell."
        )
        self.bond_atom_table.setMaximumHeight(150)

        extent_layout = QHBoxLayout()

        self.bond_nx_spin = QSpinBox(self)
        self.bond_ny_spin = QSpinBox(self)
        self.bond_nz_spin = QSpinBox(self)

        axis_names = ("a", "b", "c")
        for spin, axis in zip(
            (self.bond_nx_spin, self.bond_ny_spin, self.bond_nz_spin),
            axis_names,
        ):
            spin.setMinimum(1)
            spin.setMaximum(10)
            spin.setValue(1)
            spin.setToolTip(
                "Number of unit cells to repeat along {} in each "
                "direction (±N, giving 2N+1 cells total along this "
                "axis) -- an odd cell count is what keeps the bond "
                "set centrosymmetric.".format(axis)
            )

        extent_layout.addWidget(QLabel("Nx:", self))
        extent_layout.addWidget(self.bond_nx_spin)
        extent_layout.addWidget(QLabel("Ny:", self))
        extent_layout.addWidget(self.bond_ny_spin)
        extent_layout.addWidget(QLabel("Nz:", self))
        extent_layout.addWidget(self.bond_nz_spin)

        self.bond_tolerance_line = QLineEdit("1e-3", self)
        self.bond_tolerance_line.setValidator(
            QDoubleValidator(
                0.0, 1.0, 6, notation=QDoubleValidator.ScientificNotation
            )
        )
        self.bond_tolerance_line.setToolTip(
            "Fractional-coordinate tolerance used to collapse "
            "separation vectors that are effectively the same "
            "crystallographic bond. Covalent radii are not used here "
            "-- this tool samples 3D-ΔPDF correlation strength at a "
            "separation vector, not chemical bonding."
        )

        self.plot_bond_network_button = QPushButton("Plot Bond Network", self)
        self.plot_bond_network_button.setIcon(qta.icon("fa6s.diagram-project"))
        self.plot_bond_network_button.setToolTip(
            "Build the supercell and bond network from the enabled "
            "sites and draw it in the 3D view."
        )

        extent_layout.addWidget(QLabel("Tolerance:", self))
        extent_layout.addWidget(self.bond_tolerance_line)
        extent_layout.addWidget(self.plot_bond_network_button)

        self.bond_pdf_workspace_combo = QComboBox(self)
        self.bond_pdf_workspace_combo.setToolTip(
            "3D-ΔPDF (real-space) workspace to sample at each bond's "
            "separation vector and color bonds by (blue/white/red) --"
            " independent of whichever workspace is active in the "
            "Slice tab."
        )
        self.auto_scale_dropdown(self.bond_pdf_workspace_combo)

        extent_layout.addWidget(QLabel("Workspace:", self))
        extent_layout.addWidget(self.bond_pdf_workspace_combo)

        layout.addLayout(header_layout)
        layout.addWidget(self.bond_atom_table)
        layout.addLayout(extent_layout)

        return layout

    def toggle_container(self, state):
        """
        Show or hide the collapsible cut section and its line overlay.

        Parameters
        ----------
        state : bool
            Whether the collapsible container (cut plot and controls)
            should be visible, and the alpha state passed to
            `update_lines` for the line-cut overlay.
        """
        self.container.setVisible(state)
        self.update_lines(state)

    def connect_save_slice(self, save_slice):
        """
        Connect a handler to the "Save Slice" button click.

        Parameters
        ----------
        save_slice : callable
            Slot invoked when the "Save Slice" button is clicked.
        """
        self.save_slice_button.clicked.connect(save_slice)

    def connect_save_cut(self, save_cut):
        """
        Connect a handler to the "Save Cut" button click.

        Parameters
        ----------
        save_cut : callable
            Slot invoked when the "Save Cut" button is clicked.
        """
        self.save_cut_button.clicked.connect(save_cut)

    def connect_vol_scale_combo(self, update_vol):
        """
        Connect a handler to changes in the volume scale combo box.

        Parameters
        ----------
        update_vol : callable
            Slot invoked when the volume scale selection changes.
        """
        self.vol_scale_combo.currentIndexChanged.connect(update_vol)

    def connect_opacity_combo(self, update_opacity):
        """
        Connect a handler to changes in the opacity mapping combo box.

        Parameters
        ----------
        update_opacity : callable
            Slot invoked when the opacity mapping selection changes.
        """
        self.opacity_combo.currentIndexChanged.connect(update_opacity)

    def connect_clim_combo(self, update_clim):
        """
        Connect a handler to changes in the 3D volume color-limit combo box.

        Parameters
        ----------
        update_clim : callable
            Slot invoked when the volume color-limit method selection
            changes.
        """
        self.clim_combo.currentIndexChanged.connect(update_clim)

    def connect_vlim_combo(self, update_clim):
        """
        Connect a handler to changes in the 2D slice value-limit combo box.

        Parameters
        ----------
        update_clim : callable
            Slot invoked when the slice value-limit method selection
            changes.
        """
        self.vlim_combo.currentIndexChanged.connect(update_clim)

    def connect_cbar_combo(self, update_cbar):
        """
        Connect a handler to changes in the colormap combo box.

        Parameters
        ----------
        update_cbar : callable
            Slot invoked when the colormap selection changes.
        """
        self.cbar_combo.currentIndexChanged.connect(update_cbar)

    def connect_symmetric_zero(self, update_symmetric_zero):
        """
        Connect a handler to toggling of the "Symmetric About Zero" box.

        Parameters
        ----------
        update_symmetric_zero : callable
            Slot invoked when the checkbox is toggled.
        """
        self.symmetric_zero_box.toggled.connect(update_symmetric_zero)

    def connect_symmetric_xy(self, update_symmetric_xy):
        """
        Connect a handler to toggling of the "Symmetric X/Y" box.

        Parameters
        ----------
        update_symmetric_xy : callable
            Slot invoked when the checkbox is toggled.
        """
        self.symmetric_xy_box.toggled.connect(update_symmetric_xy)

    def connect_workspace_combo(self, activate_workspace):
        """
        Connect a handler to changes in the active-workspace combo box.

        Parameters
        ----------
        activate_workspace : callable
            Slot invoked when a different loaded/derived workspace is
            selected for slicing/cutting.
        """
        self.workspace_combo.currentIndexChanged.connect(activate_workspace)

    def connect_redraw_workspace(self, redraw_workspace):
        """
        Connect a handler to the "Redraw" button click.

        Parameters
        ----------
        redraw_workspace : callable
            Slot invoked when the "Redraw" button is clicked.
        """
        self.redraw_workspace_button.clicked.connect(redraw_workspace)

    def connect_delete_workspace(self, delete_workspace):
        """
        Connect a handler to the "Delete" button click.

        Parameters
        ----------
        delete_workspace : callable
            Slot invoked when the "Delete" button is clicked.
        """
        self.delete_workspace_button.clicked.connect(delete_workspace)

    def connect_rename_workspace(self, rename_workspace):
        """
        Connect a handler to the "Rename" button click.

        Parameters
        ----------
        rename_workspace : callable
            Slot invoked when the "Rename" button is clicked.
        """
        self.rename_workspace_button.clicked.connect(rename_workspace)

    def connect_save_workspace(self, save_workspace):
        """
        Connect a handler to the "Save" button click.

        Parameters
        ----------
        save_workspace : callable
            Slot invoked when the "Save" button is clicked.
        """
        self.save_workspace_button.clicked.connect(save_workspace)

    def connect_combine_workspaces(self, combine_workspaces):
        """
        Connect a handler to the "Subtract" button click.

        Parameters
        ----------
        combine_workspaces : callable
            Slot invoked when the "Subtract" button is clicked.
        """
        self.combine_button.clicked.connect(combine_workspaces)

    def connect_pdf_crystal_system_combo(self, update_space_groups):
        """
        Connect a handler to changes in the Transform tab's crystal-system
        combo box.

        Parameters
        ----------
        update_space_groups : callable
            Slot invoked when the crystal-system selection changes.
        """
        self.pdf_crystal_system_combo.activated.connect(update_space_groups)

    def connect_run_bragg_punch(self, run_bragg_punch):
        """
        Connect a handler to the "Punch" button click.

        Parameters
        ----------
        run_bragg_punch : callable
            Slot invoked when the "Punch" button is clicked.
        """
        self.run_punch_button.clicked.connect(run_bragg_punch)

    def connect_run_karen(self, run_karen):
        """
        Connect a handler to the "Filter" button click.

        Parameters
        ----------
        run_karen : callable
            Slot invoked when the "Filter" button is clicked.
        """
        self.run_karen_button.clicked.connect(run_karen)

    def connect_run_blur(self, run_blur):
        """
        Connect a handler to the "Blur" button click.

        Parameters
        ----------
        run_blur : callable
            Slot invoked when the "Blur" button is clicked.
        """
        self.run_blur_button.clicked.connect(run_blur)

    def connect_calculate_pdf(self, calculate_pdf):
        """
        Connect a handler to the "Transform" button click.

        Parameters
        ----------
        calculate_pdf : callable
            Slot invoked when the "Transform" button is clicked.
        """
        self.calculate_pdf_button.clicked.connect(calculate_pdf)

    def connect_load_bond_cif(self, load_bond_cif):
        """
        Connect a handler to the bond-network viewer's "Load CIF" button.

        Parameters
        ----------
        load_bond_cif : callable
            Slot invoked when the "Load CIF..." button is clicked.
        """
        self.load_bond_cif_button.clicked.connect(load_bond_cif)

    def connect_plot_bond_network(self, plot_bond_network):
        """
        Connect a handler to the "Plot Bond Network" button click.

        Parameters
        ----------
        plot_bond_network : callable
            Slot invoked when the "Plot Bond Network" button is
            clicked.
        """
        self.plot_bond_network_button.clicked.connect(plot_bond_network)

    def load_bond_cif_file_dialog(self):
        """
        Prompt the user for a CIF file path to load for the
        bond-network viewer.

        Returns
        -------
        filename : str
            Selected file path, or an empty string if the dialog was
            cancelled.
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

    def set_bond_cif_filename(self, filename):
        """
        Update the bond-network viewer's loaded-filename display.

        Parameters
        ----------
        filename : str
            Path of the loaded CIF file, or a falsy value to show the
            "no file loaded" placeholder.
        """
        text = os.path.basename(filename) if filename else "No file loaded"
        self.bond_cif_file_label.setText(text)

    def set_bond_atom_table(self, sites):
        """
        Populate the bond-network viewer's atom-site table.

        Parameters
        ----------
        sites : list of dict
            One entry per asymmetric-unit site, as returned by
            `VolumeSlicerModel.load_bond_cif` (keys ``"label"``,
            ``"element"``, ``"x"``, ``"y"``, ``"z"``). Every row
            starts checked (enabled).
        """
        self.bond_atom_table.setRowCount(0)
        self.bond_atom_table.setRowCount(len(sites))

        for row, site in enumerate(sites):
            checkbox = QCheckBox()
            checkbox.setChecked(True)

            cell_widget = QWidget()
            cell_layout = QHBoxLayout(cell_widget)
            cell_layout.addWidget(checkbox)
            cell_layout.setAlignment(Qt.AlignCenter)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            self.bond_atom_table.setCellWidget(row, 0, cell_widget)

            values = [
                site["label"],
                site["element"],
                "{:.4f}".format(site["x"]),
                "{:.4f}".format(site["y"]),
                "{:.4f}".format(site["z"]),
            ]
            for col, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.bond_atom_table.setItem(row, col, item)

    def get_bond_site_enabled(self):
        """
        Get the checked/unchecked state of every row in the atom-site table.

        Returns
        -------
        enabled : list of bool
            One flag per site, in table order.
        """
        enabled = []
        for row in range(self.bond_atom_table.rowCount()):
            cell_widget = self.bond_atom_table.cellWidget(row, 0)
            checkbox = (
                cell_widget.findChild(QCheckBox) if cell_widget else None
            )
            enabled.append(checkbox.isChecked() if checkbox else True)
        return enabled

    def get_bond_supercell_extent(self):
        """
        Get the requested supercell extent along each axis.

        Returns
        -------
        nx, ny, nz : int
            Number of unit cells to show along a, b, c -- see
            `VolumeSlicerModel.build_bond_network` for how this maps
            to translations.
        """
        return (
            self.bond_nx_spin.value(),
            self.bond_ny_spin.value(),
            self.bond_nz_spin.value(),
        )

    def get_bond_tolerance(self):
        """
        Get the fractional-coordinate bond-matching tolerance.

        Returns
        -------
        tolerance : float
            Value of the tolerance field, or 1e-3 if it does not
            currently hold valid input.
        """
        if self.bond_tolerance_line.hasAcceptableInput():
            return float(self.bond_tolerance_line.text())
        return 1e-3

    def get_bond_pdf_workspace(self):
        """
        Get the display name selected in the ΔPDF-workspace combo.

        Returns
        -------
        display_name : str or None
            Currently selected display name, or None if the combo is
            empty (no real-space workspace registered yet).
        """
        text = self.bond_pdf_workspace_combo.currentText()
        return text if text else None

    def update_bond_pdf_workspace_combo(self, display_names):
        """
        Repopulate the ΔPDF-workspace combo used for bond coloring.

        Parameters
        ----------
        display_names : list of str
            Display names of registered real-space (3D-ΔPDF)
            workspaces only -- see
            `VolumeSlicerModel.workspace_registry`.
        """
        current = self.bond_pdf_workspace_combo.currentText()

        self.bond_pdf_workspace_combo.blockSignals(True)
        self.bond_pdf_workspace_combo.clear()
        self.bond_pdf_workspace_combo.addItems(display_names)
        if current in display_names:
            self.bond_pdf_workspace_combo.setCurrentIndex(
                display_names.index(current)
            )
        self.bond_pdf_workspace_combo.blockSignals(False)
        self.auto_scale_dropdown(self.bond_pdf_workspace_combo)

    def add_bond_network(self, geometry, bond_values=None):
        """
        Draw the supercell unit-cell edges in the shared 3D view, plus
        one point per bond location colored by its sampled ΔPDF
        correlation strength (if given).

        The supercell's individual atoms are deliberately not
        rendered -- for a supercell with more than a few hundred
        atoms, that's both unnecessary for this tool's purpose (seeing
        the correlation-colored bond points) and, empirically, far
        slower to build than the bond points themselves. Bonds
        themselves are drawn as a true point cloud (a single
        `pv.PolyData`, GPU point-sprite rendered via
        `render_points_as_spheres`), the same pattern used by
        `experiment_planner.add_peaks` -- not one glyphed mesh per
        point, which is far slower for the hundreds of thousands of
        bonds a modest supercell can produce.

        Parameters
        ----------
        geometry : dict
            Dictionary as returned by
            `VolumeSlicerModel.build_bond_network`: ``"bond_vectors"``
            ((N, 3) array of Cartesian separation vectors, each
            measured from an implicit origin) and ``"edges"`` (list
            of (p0, p1) Cartesian line segments for the repeated
            unit-cell wireframe).
        bond_values : array-like, optional
            Signed ΔPDF value per bond (from
            `VolumeSlicerModel.sample_bond_correlations`), same
            length/order as ``geometry["bond_vectors"]``. A point is
            drawn at each bond vector's own position, colored with a
            diverging (blue/white/red) colormap symmetric about zero
            and a matching V-shaped opacity (near-zero/white values
            rendered more translucent, the extremes fully opaque;
            NaN values drawn gray). If not given, no bond markers are
            drawn at all -- just the unit-cell wireframe.
        """
        self.clear_scene()

        bond_vectors = geometry["bond_vectors"]
        edges = geometry["edges"]

        if edges:
            points = []
            lines = []
            for a, b in edges:
                idx = len(points)
                points.append(a)
                points.append(b)
                lines.append([2, idx, idx + 1])

            mesh = pv.PolyData(np.array(points), lines=np.hstack(lines))
            self.plotter.add_mesh(
                mesh, color="black", line_width=1, opacity=0.3
            )

        if (
            len(bond_vectors)
            and bond_values is not None
            and len(bond_values) == len(bond_vectors)
        ):
            values = np.asarray(bond_values, dtype=float)
            finite = values[np.isfinite(values)]
            lim = np.max(np.abs(finite)) if len(finite) else 1.0
            if not (lim > 0):
                lim = 1.0

            point_cloud = pv.PolyData(np.asarray(bond_vectors))
            point_cloud.point_data["correlation"] = values

            self.plotter.add_mesh(
                point_cloud,
                scalars="correlation",
                cmap="bwr",
                clim=[-lim, lim],
                opacity=_sigmoid_symmetric_opacity,
                nan_color="gray",
                nan_opacity=0.05,
                point_size=8,
                render_points_as_spheres=True,
                smooth_shading=True,
                show_scalar_bar=True,
                scalar_bar_args={"title": "ΔPDF"},
            )

        self.reset_scene()

    def connect_slice_thickness_line(self, update_slice):
        """
        Connect a handler to editing of the slice thickness field.

        Parameters
        ----------
        update_slice : callable
            Slot invoked when editing of the slice thickness line finishes.
        """
        self.slice_thickness_line.editingFinished.connect(update_slice)

    def connect_cut_thickness_line(self, update_cut):
        """
        Connect a handler to editing of the cut thickness field.

        Parameters
        ----------
        update_cut : callable
            Slot invoked when editing of the cut thickness line finishes.
        """
        self.cut_thickness_line.editingFinished.connect(update_cut)

    def connect_slice_line(self, update_slice):
        """
        Connect a handler to editing of the slice position field.

        Parameters
        ----------
        update_slice : callable
            Slot invoked when editing of the slice position line finishes.
        """
        self.slice_line.editingFinished.connect(update_slice)

    def connect_cut_line(self, update_cut):
        """
        Connect a handler to editing of the cut position field.

        Parameters
        ----------
        update_cut : callable
            Slot invoked when editing of the cut position line finishes.
        """
        self.cut_line.editingFinished.connect(update_cut)

    def connect_slice_scale_combo(self, update_slice):
        """
        Connect a handler to changes in the slice display scale combo box.

        Parameters
        ----------
        update_slice : callable
            Slot invoked when the slice scale (Linear/Log) selection
            changes.
        """
        self.slice_scale_combo.currentIndexChanged.connect(update_slice)

    def connect_cut_scale_combo(self, update_cut):
        """
        Connect a handler to changes in the cut display scale combo box.

        Parameters
        ----------
        update_cut : callable
            Slot invoked when the cut scale (Linear/Log) selection changes.
        """
        self.cut_scale_combo.currentIndexChanged.connect(update_cut)

    def _reset_slice_to_zero(self):
        """
        Reset the slice position field to zero and re-init the slider.

        Blocks signals while resetting the text so the change does not
        trigger a redraw, then rebuilds the slider range for the newly
        selected slice plane using the cached volume limits/bin counts.
        """
        self.slice_line.blockSignals(True)
        self.slice_line.setText("0.0")
        self.slice_line.blockSignals(False)
        if self._volume_limits is not None:
            ind = [2, 1, 0][self.slice_combo.currentIndex()]
            self._setup_slice_slider(
                self._volume_limits[ind][0],
                self._volume_limits[ind][1],
                self._volume_nbins[ind],
            )

    def connect_slice_combo(self, update_slice):
        """
        Connect handlers to changes in the slice plane combo box.

        Also connects an internal handler that resets the slice position
        to zero and reconfigures the slider whenever the plane changes.

        Parameters
        ----------
        update_slice : callable
            Slot invoked when the slice plane selection changes.
        """
        self.slice_combo.currentIndexChanged.connect(self._reset_slice_to_zero)
        self.slice_combo.currentIndexChanged.connect(update_slice)

    def connect_cut_combo(self, update_cut):
        """
        Connect a handler to changes in the cut axis combo box.

        Parameters
        ----------
        update_cut : callable
            Slot invoked when the cut axis selection changes.
        """
        self.cut_combo.currentIndexChanged.connect(update_cut)

    def connect_min_slider(self, update_colorbar):
        """
        No-op placeholder for connecting a minimum colorbar slider.

        This view does not expose a minimum colorbar slider widget; the
        method exists to satisfy the interface expected by the presenter.

        Parameters
        ----------
        update_colorbar : callable
            Slot that would be invoked on slider changes (unused).
        """
        pass

    def connect_max_slider(self, update_colorbar):
        """
        No-op placeholder for connecting a maximum colorbar slider.

        This view does not expose a maximum colorbar slider widget; the
        method exists to satisfy the interface expected by the presenter.

        Parameters
        ----------
        update_colorbar : callable
            Slot that would be invoked on slider changes (unused).
        """
        pass

    def connect_vmin_line(self, update_vals):
        """
        Connect a handler to editing of the colorbar minimum field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the vmin line finishes.
        """
        self.vmin_line.editingFinished.connect(update_vals)

    def connect_vmax_line(self, update_vals):
        """
        Connect a handler to editing of the colorbar maximum field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the vmax line finishes.
        """
        self.vmax_line.editingFinished.connect(update_vals)

    def connect_auto_limits(self, update_limits):
        """
        Connect a handler to toggling of the "Auto Limits" checkbox.

        Parameters
        ----------
        update_limits : callable
            Slot invoked when the auto limits checkbox is toggled.
        """
        self.auto_limits_box.toggled.connect(update_limits)

    def connect_auto_zoom(self, update_zoom):
        """
        Connect a handler to toggling of the "Auto Zoom" checkbox.

        Parameters
        ----------
        update_zoom : callable
            Slot invoked when the auto zoom checkbox is toggled.
        """
        self.auto_zoom_box.toggled.connect(update_zoom)

    def connect_xmin_line(self, update_vals):
        """
        Connect a handler to editing of the X-axis minimum field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the xmin line finishes.
        """
        self.xmin_line.editingFinished.connect(update_vals)

    def connect_xmax_line(self, update_vals):
        """
        Connect a handler to editing of the X-axis maximum field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the xmax line finishes.
        """
        self.xmax_line.editingFinished.connect(update_vals)

    def connect_ymin_line(self, update_vals):
        """
        Connect a handler to editing of the Y-axis minimum field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the ymin line finishes.
        """
        self.ymin_line.editingFinished.connect(update_vals)

    def connect_ymax_line(self, update_vals):
        """
        Connect a handler to editing of the Y-axis maximum field.

        Parameters
        ----------
        update_vals : callable
            Slot invoked when editing of the ymax line finishes.
        """
        self.ymax_line.editingFinished.connect(update_vals)

    def save_file_dialog(self):
        """
        Prompt the user for a CSV file path to save to.

        Returns
        -------
        filename : str
            Selected file path, or an empty string if the dialog was
            cancelled.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self,
            "Save csv file",
            self._get_file_dialog_dir(),
            "CSV files (*.csv)",
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def update_colorbar_min(self):
        """
        No-op placeholder for updating the minimum colorbar slider.

        This view does not expose a minimum colorbar slider widget; the
        method exists to satisfy the interface expected elsewhere.
        """
        pass

    def update_colorbar_max(self):
        """
        No-op placeholder for updating the maximum colorbar slider.

        This view does not expose a maximum colorbar slider widget; the
        method exists to satisfy the interface expected elsewhere.
        """
        pass

    def update_slice_color(self):
        """
        Recompute and apply colorbar limits from the color-bar slider values.

        Uses `get_color_bar_values` (percentages of the [vmin, vmax] range)
        to derive new display limits and applies them via
        `update_colorbar_vlims`. Does nothing if the colorbar has not been
        created yet.
        """
        if self.cb is not None:
            min_slider, max_slider = self.get_color_bar_values()

            vmin = self.vmin + (self.vmax - self.vmin) * min_slider / 100
            vmax = self.vmin + (self.vmax - self.vmin) * max_slider / 100

            self.update_colorbar_vlims(vmin, vmax)

    def update_colorbar_vlims(self, vmin, vmax):
        """
        Apply new color limits to the slice image and colorbar.

        Updates the vmin/vmax display fields, sets the new color limits on
        the slice image, refreshes the colorbar, and redraws the canvas.
        Does nothing if the colorbar or slice image has not been created
        yet.

        Parameters
        ----------
        vmin : float
            New lower color limit.
        vmax : float
            New upper color limit.
        """
        if self.cb is not None and self.slice_im is not None:
            self.set_vmin_value(vmin)
            self.set_vmax_value(vmax)

            self.slice_im.set_clim(vmin=vmin, vmax=vmax)
            self.cb.update_normal(self.slice_im)
            self.cb.minorticks_on()

            self.canvas_slice.draw_idle()

    def _create_norm(self, scale, vmin, vmax):
        """
        Build a matplotlib color normalization for the given scale.

        Parameters
        ----------
        scale : str
            Display scale (case-insensitive): 'log' for logarithmic,
            'symlog' for symmetric-log (linear near zero, logarithmic
            away from it -- a good fit for signed data such as a
            delta-PDF result), or any other value for linear
            normalization.
        vmin : float
            Lower color limit. Clamped to the smallest positive finite
            float when `scale` is 'log'.
        vmax : float
            Upper color limit.

        Returns
        -------
        norm : matplotlib.colors.Normalize, matplotlib.colors.LogNorm,
        or matplotlib.colors.SymLogNorm
            Normalization object for the slice image.
        """
        scale = scale.lower()

        if scale == "log":
            vmin = max(vmin, np.finfo(float).tiny)
            return mcolors.LogNorm(vmin=vmin, vmax=vmax)

        if scale == "symlog":
            # Linear threshold auto-set to 1% of the larger display
            # limit's magnitude -- a simple default so this doesn't
            # need its own manual-override field (unlike vmin/vmax).
            linthresh = max(abs(vmin), abs(vmax), np.finfo(float).tiny) * 0.01
            return mcolors.SymLogNorm(
                linthresh=linthresh, vmin=vmin, vmax=vmax
            )

        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def update_slice_display(self, cmap_key, scale, vmin, vmax):
        """
        Update the slice image's colormap, normalization, and colorbar.

        Does nothing if no slice image has been drawn yet.

        Parameters
        ----------
        cmap_key : str
            Key into the module-level `cmaps` mapping identifying the
            colormap to apply (e.g. 'Sequential', 'Rainbow').
        scale : str
            Display scale, 'log' or 'linear', passed to `_create_norm`.
        vmin : float
            Lower display/color limit.
        vmax : float
            Upper display/color limit.
        """
        if self.slice_im is None:
            return

        self.set_vmin_value(vmin)
        self.set_vmax_value(vmax)

        self.slice_im.set_cmap(cmaps[cmap_key])
        self.slice_im.set_norm(self._create_norm(scale, vmin, vmax))

        if self.cb is not None:
            self.cb.update_normal(self.slice_im)
            self.cb.minorticks_on()

        self.canvas_slice.draw_idle()

    def get_color_bar_values(self):
        """
        Get the colorbar slider percentages used by `update_slice_color`.

        This view does not expose interactive colorbar sliders, so a
        fixed full-range value is always returned.

        Returns
        -------
        min_slider : int
            Lower slider percentage (always 0).
        max_slider : int
            Upper slider percentage (always 100).
        """
        return 0, 100

    def reset_slider(self):
        """
        No-op placeholder for resetting a colorbar slider.

        This view does not expose colorbar slider widgets; the method
        exists to satisfy the interface expected elsewhere.
        """
        pass

    def connect_load_NXS(self, load_NXS):
        """
        Connect a handler to the "Load NXS" button click.

        Parameters
        ----------
        load_NXS : callable
            Slot invoked when the "Load NXS" button is clicked.
        """
        self.load_NXS_button.clicked.connect(load_NXS)

    def load_NXS_file_dialog(self):
        """
        Prompt the user for a NeXus (.nxs) file path to load.

        Returns
        -------
        filename : str
            Selected file path, or an empty string if the dialog was
            cancelled.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getOpenFileName(
            self,
            "Load NXS file",
            self._get_file_dialog_dir(),
            "NXS files (*.nxs)",
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def save_md_file_dialog(self, default_name=""):
        """
        Prompt the user for a NeXus (.nxs) file path to save an MD
        workspace to.

        Parameters
        ----------
        default_name : str, optional
            Display name to pre-fill as the suggested filename.

        Returns
        -------
        filename : str
            Selected file path, or an empty string if the dialog was
            cancelled.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        start_dir = self._get_file_dialog_dir()
        start = (
            os.path.join(start_dir, default_name + ".nxs")
            if start_dir
            else default_name + ".nxs"
        )

        filename, _ = file_dialog.getSaveFileName(
            self,
            "Save MD workspace",
            start,
            "NXS files (*.nxs)",
            options=options,
        )

        if filename:
            self._remember_file_dialog_dir(os.path.dirname(filename))

        return filename

    def add_histo(self, histo_dict, norm, value):
        """
        Render 3 axis-aligned slice planes for the loaded histogram.

        Builds a PyVista `ImageData` grid from the histogram signal and
        cuts it with `ImageData.slice_index` along each of the 3 axes
        (a plain, non-interactive geometric cut -- no plane widget),
        colored/scaled with the current opacity/colormap/scale
        settings, configures the bounding-box axes with the
        crystallographic transform, and resets the camera to fit the
        scene. Also (re)configures the slice position slider range for
        the newly selected plane.

        Two of the three planes sit at index 0 (the origin); the third
        -- whichever axis is currently selected -- sits at `value`,
        the position driven by the slice slider/line edit. This gives
        the same tri-planar context as PyVista's `slice_orthogonal`
        example, but built from `slice_index` instead: `slice_index`
        extracts the requested point layer directly by index (an O(1)
        structured-grid lookup), whereas `slice`/`slice_orthogonal` run
        VTK's generic cutter over the *entire* volume for every plane
        -- about 90x slower on a 201^3 volume in local testing, and
        3 planes means paying that 3 times over. A previous version of
        this used PyVista's `add_mesh_slice`, which additionally builds
        an interactive plane-widget (handles, outline, VTK interactor
        observers) around the cut -- and since every reslice already
        tears down and rebuilds the whole scene from scratch (see
        `clear_scene`), that widget setup/teardown cost was paid on
        every single slider tick for no benefit (nothing here supports
        dragging a plane directly in the 3D view).

        Parameters
        ----------
        histo_dict : dict
            Histogram information dictionary (as returned by the model's
            `get_histo_info`) containing 'signal', 'labels', 'min_lim',
            'max_lim', 'spacing', 'projection', 'transform', and 'scales'.
        norm : array-like
            Normal vector identifying the selected slice plane in
            crystallographic axis space (e.g. [0, 0, 1]); mutated in
            place to build the slice plane origin.
        value : float
            Position along `norm` at which to place the selected slice
            plane; the other two planes stay at the origin (index 0).
        """
        opacity = opacities[self.get_opacity()]

        log_scale = True if self.get_vol_scale() == "Log" else False

        cmap = cmaps[self.get_colormap()]

        self.clear_scene()

        self.norm = np.array(norm).copy()
        origin = norm
        origin[np.abs(origin).tolist().index(1)] = value

        signal = histo_dict["signal"]
        labels = histo_dict["labels"]

        min_lim = histo_dict["min_lim"]
        max_lim = histo_dict["max_lim"]
        spacing = histo_dict["spacing"]

        P = histo_dict["projection"]
        T = histo_dict["transform"]
        S = histo_dict["scales"]

        grid = pv.ImageData()

        grid.dimensions = np.array(signal.shape) + 1

        grid.origin = min_lim
        grid.spacing = spacing

        min_bnd = min_lim * S
        max_bnd = max_lim * S

        bounds = np.array([[min_bnd[i], max_bnd[i]] for i in [0, 1, 2]])
        limits = np.array([[min_lim[i], max_lim[i]] for i in [0, 1, 2]])

        ind = np.abs(self.norm).tolist().index(1)
        self._volume_limits = limits
        self._volume_nbins = signal.shape
        self._setup_slice_slider(
            limits[ind][0], limits[ind][1], signal.shape[ind]
        )

        b = pv._vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                b.SetElement(i, j, P[i, j])

        grid.cell_data["scalars"] = signal.flatten(order="F")

        clim = [np.nanmin(signal), np.nanmax(signal)]

        if not np.all(np.isfinite(clim)):
            clim = [0.1, 10]

        point_index = np.clip(
            np.round((np.array(origin) - min_lim) / spacing).astype(int),
            0,
            np.array(signal.shape),
        )

        slices = [
            grid.slice_index(i=point_index[0]),
            grid.slice_index(j=point_index[1]),
            grid.slice_index(k=point_index[2]),
        ]

        for i, slice_mesh in enumerate(slices):
            self.plotter.add_mesh(
                slice_mesh,
                opacity=opacity,
                log_scale=log_scale,
                clim=clim,
                show_scalar_bar=(i == 0),
                cmap=cmap,
                user_matrix=b,
            )

        actor = self.plotter.show_grid(
            xtitle=labels[0],
            ytitle=labels[1],
            ztitle=labels[2],
            font_size=8,
            minor_ticks=True,
        )

        actor.SetAxisBaseForX(*T[:, 0])
        actor.SetAxisBaseForY(*T[:, 1])
        actor.SetAxisBaseForZ(*T[:, 2])

        actor.bounds = bounds.ravel()
        actor.SetXAxisRange(limits[0])
        actor.SetYAxisRange(limits[1])
        actor.SetZAxisRange(limits[2])

        vmin0, vmax0, n0, fmt0 = (
            *limits[0],
            actor.n_xlabels,
            actor.x_label_format,
        )
        vmin1, vmax1, n1, fmt1 = (
            *limits[1],
            actor.n_ylabels,
            actor.y_label_format,
        )
        vmin2, vmax2, n2, fmt2 = (
            *limits[2],
            actor.n_zlabels,
            actor.z_label_format,
        )

        axis0_label = pv.plotting.cube_axes_actor.make_axis_labels(
            vmin=vmin0, vmax=vmax0, n=n0, fmt=fmt0
        )
        axis1_label = pv.plotting.cube_axes_actor.make_axis_labels(
            vmin=vmin1, vmax=vmax1, n=n1, fmt=fmt1
        )
        axis2_label = pv.plotting.cube_axes_actor.make_axis_labels(
            vmin=vmin2, vmax=vmax2, n=n2, fmt=fmt2
        )

        actor.SetAxisLabels(0, axis0_label)
        actor.SetAxisLabels(1, axis1_label)
        actor.SetAxisLabels(2, axis2_label)

        self.reset_scene()

    def connect_slice_ready(self, reslice):
        """
        Connect a handler to the `slice_ready` signal.

        Parameters
        ----------
        reslice : callable
            Slot invoked when `slice_ready` is emitted.
        """
        self.slice_ready.connect(reslice)

    @staticmethod
    def _transform_bbox(T, xmin, ymin, xmax, ymax):
        """
        Map an axis-aligned bounding box through an affine transform.

        Transforming only the two diagonal corners (xmin, ymin) and
        (xmax, ymax) is only valid for a transform with no shear --
        for a skewed/non-orthogonal crystallographic axis transform
        (e.g. a hexagonal or monoclinic lattice), that maps the
        rectangle to a parallelogram whose true bounding box depends
        on all four corners, not just two. Using only two corners
        silently crops most of the intended view for a large shear.

        Parameters
        ----------
        T : np.ndarray
            3x3 affine transform (as used for ``Affine2D``/``self.T``
            or its inverse ``self.T_inv``).
        xmin, ymin, xmax, ymax : float
            Bounding box corners before the transform.

        Returns
        -------
        xmin, xmax, ymin, ymax : float
            Bounding box of the transformed corners.
        """
        corners = np.array(
            [
                [xmin, ymin, 1],
                [xmax, ymin, 1],
                [xmin, ymax, 1],
                [xmax, ymax, 1],
            ]
        )
        tx, ty, _ = T @ corners.T
        return tx.min(), tx.max(), ty.min(), ty.max()

    def __format_axis_coord(self, x, y):
        """
        Format slice-plot display coordinates as an HKL or UVW string.

        Used as the matplotlib Axes `format_coord` callback to show the
        crystallographic indices under the cursor in the slice plot's
        status readout: HKL for a reciprocal-space workspace, UVW for a
        real-space (delta-PDF) one.

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
            ``"hkl = (h, k, l)"`` or ``"uvw = (u, v, w)"`` with values
            to three decimal places.
        """
        x, y, _ = np.dot(self.T_inv, [x, y, 1])
        i, j, k = np.dot(self.W, [x, y, self.z])
        if self.space == "real":
            return "uvw = ({:.3f}, {:.3f}, {:.3f})".format(i, j, k)
        return "hkl = ({:.3f}, {:.3f}, {:.3f})".format(i, j, k)

    def add_slice(self, slice_dict):
        """
        Draw the 2D slice plot from newly computed slice data.

        Rebuilds the slice axes with a curvilinear grid helper so that
        the (possibly non-orthogonal) crystallographic axes are drawn
        correctly, plots the signal with `pcolormesh`, updates/creates
        the colorbar, restores or resets the zoom/pan limits depending on
        the "Auto Zoom" setting and prior view state, and reconnects the
        axis limit-change callbacks used to keep the min/max fields in
        sync.

        Parameters
        ----------
        slice_dict : dict
            Slice information dictionary (as returned by the model's
            `get_slice_info`) containing 'x', 'y', 'labels', 'title',
            'signal', 'z', 'W', 'space', 'transform', 'aspect', and
            optionally 'vmin'/'vmax'.
        """
        prev_xlim = (
            self.ax_slice.get_xlim() if self.slice_im is not None else None
        )
        prev_ylim = (
            self.ax_slice.get_ylim() if self.slice_im is not None else None
        )

        if self._cx is not None:
            self.ax_cut.callbacks.disconnect(self._cx)
        if self._sx is not None:
            self.ax_slice.callbacks.disconnect(self._sx)
        if self._sy is not None:
            self.ax_slice.callbacks.disconnect(self._sy)

        self.xmin_line.blockSignals(True)
        self.xmax_line.blockSignals(True)
        self.ymin_line.blockSignals(True)
        self.ymax_line.blockSignals(True)

        cmap = cmaps[self.get_colormap()]

        x = slice_dict["x"]
        y = slice_dict["y"]

        labels = slice_dict["labels"]
        title = slice_dict["title"]
        signal = slice_dict["signal"]

        self.z = slice_dict["z"]
        self.W = slice_dict["W"]
        self.space = slice_dict.get("space", "reciprocal")

        scale = self.get_slice_scale()

        vmin = np.nanmin(signal)
        vmax = np.nanmax(signal)

        if np.isclose(vmax, vmin) or not np.isfinite([vmin, vmax]).all():
            vmin, vmax = (0.1, 1) if scale == "log" else (0, 1)

        T = slice_dict["transform"]
        aspect = slice_dict["aspect"]

        self.T_inv = np.linalg.inv(T)
        self.T = T

        transform = Affine2D(T)
        self.transform = transform

        self.xlim = np.array([x.min(), x.max()])
        self.ylim = np.array([y.min(), y.max()])

        if self.cb is not None:
            self.cb.remove()
            self.cb = None

        self.ax_slice.remove()

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
            norm=self._create_norm(
                scale,
                slice_dict.get("vmin", vmin),
                slice_dict.get("vmax", vmax),
            ),
            cmap=cmap,
            shading="flat",
            transform=trans,
            rasterized=True,
        )

        self.trans = trans

        self.ax_slice.set_xlabel(labels[0])
        self.ax_slice.set_ylabel(labels[1])

        self.vmin, self.vmax = self.slice_im.norm.vmin, self.slice_im.norm.vmax

        self.ax_slice.set_title(title)

        self.cb = self.fig_slice.colorbar(self.slice_im, ax=self.ax_slice)
        self.cb.minorticks_on()

        self.ax_slice.format_coord = self.__format_axis_coord

        self.vmin, self.vmax = self.slice_im.norm.vmin, self.slice_im.norm.vmax

        self.set_vmin_value(self.vmin)
        self.set_vmax_value(self.vmax)

        have_limits = all(
            val is not None
            for val in (
                self.get_xmin_value(),
                self.get_xmax_value(),
                self.get_ymin_value(),
                self.get_ymax_value(),
            )
        )

        if (
            self.get_auto_zoom()
            or prev_xlim is None
            or prev_ylim is None
            or not have_limits
        ):
            self.set_xmin_value(self.xlim[0])
            self.set_xmax_value(self.xlim[1])
            self.set_ymin_value(self.ylim[0])
            self.set_ymax_value(self.ylim[1])

            xmin, ymin = self.get_xmin_value(), self.get_ymin_value()
            xmax, ymax = self.get_xmax_value(), self.get_ymax_value()

            xmin, xmax, ymin, ymax = self._transform_bbox(
                self.T, xmin, ymin, xmax, ymax
            )

            self.ax_slice.set_xlim([xmin, xmax])
            self.ax_slice.set_ylim([ymin, ymax])
        else:
            self.ax_slice.set_xlim(prev_xlim)
            self.ax_slice.set_ylim(prev_ylim)

            self._slice_zoom_xlim = prev_xlim
            self._slice_zoom_ylim = prev_ylim

        self.canvas_slice.draw_idle()

        self._cx = self.ax_cut.callbacks.connect(
            "xlim_changed", self.cut_limits
        )
        self._sx = self.ax_slice.callbacks.connect(
            "xlim_changed", self.slice_limits
        )
        self._sy = self.ax_slice.callbacks.connect(
            "ylim_changed", self.slice_limits
        )

        self.xmin_line.blockSignals(False)
        self.xmax_line.blockSignals(False)
        self.ymin_line.blockSignals(False)
        self.ymax_line.blockSignals(False)

    def update_lines(self, alpha):
        """
        Set the transparency of the line-cut overlay on the slice plot.

        Parameters
        ----------
        alpha : bool or float
            Alpha value applied to the cut overlay lines (0/False hides
            them, 1/True shows them fully opaque).
        """
        lines = (
            self._cut_lines
            if self._cut_lines is not None
            else self.ax_slice.get_lines()
        )
        for line in lines:
            line.set_alpha(alpha)
        self.canvas_slice.draw_idle()

    def add_cut(self, cut_dict):
        """
        Draw the 1D cut plot and its overlay lines on the slice plot.

        Updates or creates the pair of dashed lines on the slice image
        marking the extent of the cut band, redraws the 1D error-bar cut
        plot, restores the appropriate axis limits, and (re)connects the
        mouse callbacks that allow interactively dragging the cut lines
        as well as the axis limit-change callbacks. Does nothing if no
        slice has been drawn yet.

        Parameters
        ----------
        cut_dict : dict
            Cut information dictionary (as returned by the model's
            `get_cut_info`) containing 'x', 'y', 'e', 'value', 'label',
            and 'title'.
        """
        if self.xlim is None or self.ylim is None or self.slice_im is None:
            return

        if self._cx is not None:
            self.ax_cut.callbacks.disconnect(self._cx)
        if self._sx is not None:
            self.ax_slice.callbacks.disconnect(self._sx)
        if self._sy is not None:
            self.ax_slice.callbacks.disconnect(self._sy)

        self.xmin_line.blockSignals(True)
        self.xmax_line.blockSignals(True)
        self.ymin_line.blockSignals(True)
        self.ymax_line.blockSignals(True)

        x = cut_dict["x"]
        y = cut_dict["y"]
        e = cut_dict["e"]

        val = cut_dict["value"]

        label = cut_dict["label"]
        title = cut_dict["title"]

        scale = self.get_cut_scale()

        line_cut = self.get_cut()

        xlim = self.xlim
        ylim = self.ylim

        thick = self.get_cut_thickness()

        delta = 0 if thick is None else thick / 2

        if line_cut == "Axis 2":
            l0 = [val - delta, val - delta], ylim
            l1 = [val + delta, val + delta], ylim
            direction = "vertical"
        else:
            l0 = xlim, [val - delta, val - delta]
            l1 = xlim, [val + delta, val + delta]
            direction = "horizontal"

        l = self.toggle_line_box.isChecked()

        if (
            self._cut_lines is not None
            and self._cut_lines[0] in self.ax_slice.get_lines()
        ):
            self._cut_lines[0].set_xdata(l0[0])
            self._cut_lines[0].set_ydata(l0[1])
            self._cut_lines[0].set_alpha(l)
            self._cut_lines[1].set_xdata(l1[0])
            self._cut_lines[1].set_ydata(l1[1])
            self._cut_lines[1].set_alpha(l)
        else:
            (ln0,) = self.ax_slice.plot(
                *l0, "w--", lw=1, alpha=l, transform=self.trans
            )
            (ln1,) = self.ax_slice.plot(
                *l1, "w--", lw=1, alpha=l, transform=self.trans
            )
            self._cut_lines = [ln0, ln1]

        self.ax_cut.clear()

        self.ax_cut.errorbar(x, y, e)
        self.ax_cut.set_xlabel(label)
        self.ax_cut.set_yscale(scale)
        self.ax_cut.set_title(title)
        self.ax_cut.minorticks_on()

        self.ax_cut.xaxis.get_major_locator().set_params(integer=True)

        if line_cut == "Axis 1":
            lims = [self.get_xmin_value(), self.get_xmax_value()]
        else:
            lims = [self.get_ymin_value(), self.get_ymax_value()]

        self.ax_cut.set_xlim(lims)

        self.canvas_cut.draw_idle()

        self.canvas_slice.draw_idle()

        self.linecut = {
            "is_dragging": False,
            "line_cut": (xlim, ylim, delta, direction),
        }

        self.fig_slice.canvas.mpl_connect("button_press_event", self.on_press)

        self.fig_slice.canvas.mpl_connect(
            "button_release_event", self.on_release
        )

        self.fig_slice.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

        self._cx = self.ax_cut.callbacks.connect(
            "xlim_changed", self.cut_limits
        )
        self._sx = self.ax_slice.callbacks.connect(
            "xlim_changed", self.slice_limits
        )
        self._sy = self.ax_slice.callbacks.connect(
            "ylim_changed", self.slice_limits
        )

        self.xmin_line.blockSignals(False)
        self.xmax_line.blockSignals(False)
        self.ymin_line.blockSignals(False)
        self.ymax_line.blockSignals(False)

    def on_press(self, event):
        """
        Matplotlib button-press handler that begins dragging the cut line.

        Starts a drag only if the click is inside the slice axes, no
        toolbar navigation mode (zoom/pan) is active, and the line-cut
        overlay is currently shown.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse button-press event.
        """
        if (
            event.inaxes == self.ax_slice
            and self.fig_slice.canvas.toolbar.mode == ""
            and self.toggle_line_box.isChecked()
        ):
            self.linecut["is_dragging"] = True

    def on_release(self, event):
        """
        Matplotlib button-release handler that ends dragging the cut line.

        Clears the dragging flag and emits `cut_ready` to request that the
        cut be recomputed at its final dragged position.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse button-release event.
        """
        self.linecut["is_dragging"] = False

        self.cut_ready.emit()

    def connect_cut_ready(self, recut):
        """
        Connect a handler to the `cut_ready` signal.

        Parameters
        ----------
        recut : callable
            Slot invoked when `cut_ready` is emitted.
        """
        self.cut_ready.connect(recut)

    def on_motion(self, event):
        """
        Matplotlib motion-notify handler that drags the cut line overlay.

        While dragging, converts the cursor position from display space
        into crystallographic axis coordinates, updates the cut position
        field (without re-triggering its edit signal), and moves the
        overlay lines to follow the cursor.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse motion event.
        """
        if self.linecut["is_dragging"] and event.inaxes == self.ax_slice:
            xlim, ylim, delta, direction = self.linecut["line_cut"]

            x, y, _ = np.dot(self.T_inv, [event.xdata, event.ydata, 1])

            self.cut_line.blockSignals(True)

            if direction == "vertical":
                l0 = [x - delta, x - delta], ylim
                l1 = [x + delta, x + delta], ylim
                self.set_cut_value(x)
            else:
                l0 = xlim, [y - delta, y - delta]
                l1 = xlim, [y + delta, y + delta]
                self.set_cut_value(y)

            self.cut_line.blockSignals(False)

            if (
                self._cut_lines is not None
                and self._cut_lines[0] in self.ax_slice.get_lines()
            ):
                self._cut_lines[0].set_xdata(l0[0])
                self._cut_lines[0].set_ydata(l0[1])
                self._cut_lines[1].set_xdata(l1[0])
                self._cut_lines[1].set_ydata(l1[1])
            else:
                (ln0,) = self.ax_slice.plot(
                    *l0, "w--", linewidth=1, transform=self.trans
                )
                (ln1,) = self.ax_slice.plot(
                    *l1, "w--", linewidth=1, transform=self.trans
                )
                self._cut_lines = [ln0, ln1]

            self.canvas_slice.draw_idle()

    def get_vol_scale(self):
        """
        Get the currently selected 3D volume display scale.

        Returns
        -------
        scale : str
            'Linear' or 'Log', as displayed in the volume scale combo box.
        """
        return self.vol_scale_combo.currentText()

    def get_opacity(self):
        """
        Get the currently selected opacity mapping type.

        Returns
        -------
        opacity : str
            One of 'Linear Ascending', 'Linear Descending', 'Geometric
            Ascending', 'Geometric Descending', 'Sigmoid Ascending',
            'Sigmoid Descending', 'Linear Symmetric', 'Sigmoid
            Symmetric', or 'Geometric Symmetric'.
        """
        return self.opacity_combo.currentText()

    def get_colormap(self):
        """
        Get the currently selected colormap name.

        Returns
        -------
        cmap_key : str
            Key into the module-level `cmaps` mapping (e.g. 'Sequential',
            'Rainbow', 'Binary', 'Diverging', 'Modified').
        """
        return self.cbar_combo.currentText()

    def set_colormap(self, cmap_key):
        """
        Set the currently selected colormap.

        Parameters
        ----------
        cmap_key : str
            Key into the module-level `cmaps` mapping (e.g. 'Diverging').
        """
        index = self.cbar_combo.findText(cmap_key)
        if index >= 0:
            self.cbar_combo.setCurrentIndex(index)

    def get_symmetric_zero(self):
        """
        Get whether the slice colorbar limits should be symmetric about zero.

        Returns
        -------
        symmetric_zero : bool
            True if the "Symmetric About Zero" box is checked.
        """
        return self.symmetric_zero_box.isChecked()

    def set_symmetric_zero(self, checked):
        """
        Set whether the slice colorbar limits should be symmetric about zero.

        Parameters
        ----------
        checked : bool
            Whether to check the "Symmetric About Zero" box.
        """
        self.symmetric_zero_box.setChecked(checked)

    def get_symmetric_xy(self):
        """
        Get whether the X/Y axis limits should be kept symmetric about zero.

        Returns
        -------
        symmetric_xy : bool
            True if the "Symmetric X/Y" box is checked.
        """
        return self.symmetric_xy_box.isChecked()

    def set_symmetric_xy(self, checked):
        """
        Set whether the X/Y axis limits should be kept symmetric about zero.

        Parameters
        ----------
        checked : bool
            Whether to check the "Symmetric X/Y" box.
        """
        self.symmetric_xy_box.setChecked(checked)

    def get_active_workspace(self):
        """
        Get the display name of the currently selected workspace.

        Returns
        -------
        display_name : str or None
            Display name selected in the active-workspace combo box, or
            None if no workspace is loaded.
        """
        text = self.workspace_combo.currentText()
        return text if text else None

    def update_workspace_combos(self, display_names, active_display_name):
        """
        Repopulate the active-workspace combo box.

        Parameters
        ----------
        display_names : list of str
            Display names of all currently loaded/derived workspaces.
        active_display_name : str or None
            Display name to select after repopulating, if present.
        """
        self.workspace_combo.blockSignals(True)
        self.workspace_combo.clear()
        self.workspace_combo.addItems(display_names)
        if active_display_name in display_names:
            self.workspace_combo.setCurrentIndex(
                display_names.index(active_display_name)
            )
        self.workspace_combo.blockSignals(False)
        self.auto_scale_dropdown(self.workspace_combo)

        for combo in (
            self.combine_ws_a_combo,
            self.combine_ws_b_combo,
            self.punch_input_combo,
            self.karen_input_combo,
            self.blur_input_combo,
            self.pdf_input_combo,
        ):
            self._repopulate_preserving_selection(combo, display_names)

    def _repopulate_preserving_selection(self, combo, display_names):
        """
        Repopulate a workspace-name combo box, preserving its selection.

        Parameters
        ----------
        combo : QComboBox
            Combo box listing workspace display names.
        display_names : list of str
            Display names of all currently loaded/derived workspaces.
        """
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(display_names)
        if current in display_names:
            combo.setCurrentIndex(display_names.index(current))
        combo.blockSignals(False)
        self.auto_scale_dropdown(combo)

    def update_workspace_list(self, display_names):
        """
        Repopulate the workspace management list (rename/delete).

        Parameters
        ----------
        display_names : list of str
            Display names of all currently loaded/derived workspaces.
        """
        current = self.get_selected_workspace_for_management()
        self.workspace_list_widget.clear()
        self.workspace_list_widget.addItems(display_names)
        if current in display_names:
            items = self.workspace_list_widget.findItems(
                current, Qt.MatchExactly
            )
            if items:
                self.workspace_list_widget.setCurrentItem(items[0])

    def get_selected_workspace_for_management(self):
        """
        Get the display name currently selected in the management list.

        Returns
        -------
        display_name : str or None
            Selected display name, or None if no item is selected.
        """
        item = self.workspace_list_widget.currentItem()
        return item.text() if item is not None else None

    def confirm_delete(self, display_name):
        """
        Prompt the user to confirm deleting a workspace.

        Parameters
        ----------
        display_name : str
            Display name of the workspace to be deleted.

        Returns
        -------
        confirmed : bool
            True if the user confirmed the deletion.
        """
        answer = QMessageBox.question(
            self,
            "Delete Workspace",
            'Delete workspace "{}"? This cannot be undone.'.format(
                display_name
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def prompt_rename(self, old_name):
        """
        Prompt the user for a new display name for a workspace.

        Parameters
        ----------
        old_name : str
            Current display name of the workspace being renamed.

        Returns
        -------
        new_name : str or None
            New display name, or None if the user cancelled or entered
            an empty name.
        """
        new_name, ok = QInputDialog.getText(
            self, "Rename Workspace", "New name:", QLineEdit.Normal, old_name
        )
        if not ok or not new_name:
            return None
        return new_name

    def prompt_load_name(self, default_name):
        """
        Prompt the user for a display name for a newly loaded workspace.

        Parameters
        ----------
        default_name : str
            Suggested display name, pre-filled in the prompt (the
            sequential ``"data"``/``"data_2"``/... name that would
            otherwise be auto-assigned).

        Returns
        -------
        display_name : str or None
            The confirmed display name, or None if the user cancelled
            or cleared the field (in which case the load should be
            aborted).
        """
        display_name, ok = QInputDialog.getText(
            self,
            "Load NXS",
            "Workspace name:",
            QLineEdit.Normal,
            default_name,
        )
        if not ok or not display_name:
            return None
        return display_name

    def show_error(self, title, message):
        """
        Show an error dialog.

        Parameters
        ----------
        title : str
            Dialog title.
        message : str
            Error message to display.
        """
        QMessageBox.critical(self, title, message)

    def get_combine_ws_a(self):
        """
        Get the display name selected as workspace A for arithmetic.

        Returns
        -------
        display_name : str or None
            Selected display name, or None if no workspace is loaded.
        """
        text = self.combine_ws_a_combo.currentText()
        return text if text else None

    def get_combine_ws_b(self):
        """
        Get the display name selected as workspace B for arithmetic.

        Returns
        -------
        display_name : str or None
            Selected display name, or None if no workspace is loaded.
        """
        text = self.combine_ws_b_combo.currentText()
        return text if text else None

    def get_combine_coeff_a(self):
        """
        Get the coefficient ``a`` for the combine operation, if valid.

        Returns
        -------
        coeff : float or None
            Coefficient parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.combine_coeff_a_line.hasAcceptableInput():
            return float(self.combine_coeff_a_line.text())

    def get_combine_coeff_b(self):
        """
        Get the coefficient ``b`` for the combine operation, if valid.

        Returns
        -------
        coeff : float or None
            Coefficient parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.combine_coeff_b_line.hasAcceptableInput():
            return float(self.combine_coeff_b_line.text())

    def get_combine_output_name(self):
        """
        Get the requested display name for the combine result.

        Returns
        -------
        output_name : str or None
            Text entered in the output-name field, or None if empty.
        """
        text = self.combine_output_line.text().strip()
        return text if text else None

    def get_pdf_crystal_system(self):
        """
        Get the crystal system selected in the Transform tab.

        Returns
        -------
        system : str
            Name of the selected crystal system.
        """
        return self.pdf_crystal_system_combo.currentText()

    def get_pdf_space_group(self):
        """
        Get the space group selected in the Transform tab.

        Returns
        -------
        space_group : str or None
            Space group formatted as ``"{number}: {symbol}"``, or None
            if the combo box is currently empty.
        """
        text = self.pdf_space_group_combo.currentText()
        return text if text else None

    def update_pdf_space_groups(self, space_groups):
        """
        Repopulate the Transform tab's space-group combo box.

        Parameters
        ----------
        space_groups : list of str
            Space groups formatted as ``"{number}: {symbol}"``.
        """
        self.pdf_space_group_combo.clear()
        self.pdf_space_group_combo.addItems(space_groups)
        self.auto_scale_dropdown(self.pdf_space_group_combo)

    def get_punch_input_workspace(self):
        """
        Get the display name selected as input to the Bragg punch.

        Returns
        -------
        display_name : str or None
            Selected display name, or None if no workspace is loaded.
        """
        text = self.punch_input_combo.currentText()
        return text if text else None

    def get_punch_output_name(self):
        """
        Get the requested display name for the Bragg-punch result.

        Returns
        -------
        output_name : str or None
            Text entered in the output-name field, or None if empty.
        """
        text = self.punch_output_line.text().strip()
        return text if text else None

    def get_punch_q_size(self):
        """
        Get the Bragg-punch radius, if valid.

        Returns
        -------
        Q_size : float or None
            Punch radius (Å⁻¹) parsed from the field, or None if the
            field does not currently contain acceptable input.
        """
        if self.punch_q_size_line.hasAcceptableInput():
            return float(self.punch_q_size_line.text())

    def get_punch_q_inner(self):
        """
        Get the Bragg-punch step's inner-cut radius, if valid.

        Returns
        -------
        Q_inner : float or None
            Inner-cut radius (Å⁻¹) parsed from the field, or None if
            the field does not currently contain acceptable input.
        """
        if self.punch_q_inner_line.hasAcceptableInput():
            return float(self.punch_q_inner_line.text())

    def get_punch_outlier(self):
        """
        Get the Bragg-punch step's IQR outlier scale factor, if valid.

        Returns
        -------
        outlier : float or None
            Tukey's-fences scale factor parsed from the field, or None
            if the field does not currently contain acceptable input.
        """
        if self.punch_outlier_line.hasAcceptableInput():
            return float(self.punch_outlier_line.text())

    def get_karen_input_workspace(self):
        """
        Get the display name selected as input to the KAREN filter.

        Returns
        -------
        display_name : str or None
            Selected display name, or None if no workspace is loaded.
        """
        text = self.karen_input_combo.currentText()
        return text if text else None

    def get_karen_output_name(self):
        """
        Get the requested display name for the KAREN-filtered result.

        Returns
        -------
        output_name : str or None
            Text entered in the output-name field, or None if empty.
        """
        text = self.karen_output_line.text().strip()
        return text if text else None

    def get_karen_width(self):
        """
        Get the KAREN step's moving-window size, if valid.

        Returns
        -------
        width : float or None
            Window size (Å⁻¹) parsed from the field, or None if the
            field does not currently contain acceptable input.
        """
        if self.karen_width_line.hasAcceptableInput():
            return float(self.karen_width_line.text())

    def get_karen_z_score(self):
        """
        Get the KAREN step's outlier cutoff, if valid.

        Returns
        -------
        z_score : float or None
            Outlier cutoff (estimated standard deviations from the
            local median) parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.karen_z_score_line.hasAcceptableInput():
            return float(self.karen_z_score_line.text())

    def get_blur_input_workspace(self):
        """
        Get the display name selected as input to the blur step.

        Returns
        -------
        display_name : str or None
            Selected display name, or None if no workspace is loaded.
        """
        text = self.blur_input_combo.currentText()
        return text if text else None

    def get_blur_output_name(self):
        """
        Get the requested display name for the blurred result.

        Returns
        -------
        output_name : str or None
            Text entered in the output-name field, or None if empty.
        """
        text = self.blur_output_line.text().strip()
        return text if text else None

    def get_blur_q_blur(self):
        """
        Get the blur step's NaN-Gaussian blur size, if valid.

        Returns
        -------
        Q_blur : float or None
            Blur size (Å⁻¹) parsed from the field, or None if the
            field does not currently contain acceptable input.
        """
        if self.blur_q_blur_line.hasAcceptableInput():
            return float(self.blur_q_blur_line.text())

    def get_pdf_input_workspace(self):
        """
        Get the display name selected as input to the 3D-ΔPDF transform.

        Returns
        -------
        display_name : str or None
            Selected display name, or None if no workspace is loaded.
        """
        text = self.pdf_input_combo.currentText()
        return text if text else None

    def get_pdf_output_name(self):
        """
        Get the requested display name for the 3D-ΔPDF result.

        Returns
        -------
        output_name : str or None
            Text entered in the output-name field, or None if empty.
        """
        text = self.pdf_output_line.text().strip()
        return text if text else None

    def get_pdf_q_outer(self):
        """
        Get the 3D-ΔPDF padding extent, if valid.

        Returns
        -------
        Q_outer : float or None
            Padding extent (Å⁻¹) parsed from the field, or None if the
            field does not currently contain acceptable input.
        """
        if self.pdf_q_outer_line.hasAcceptableInput():
            return float(self.pdf_q_outer_line.text())

    def get_pdf_window(self):
        """
        Get the selected 3D-ΔPDF apodization window.

        Returns
        -------
        window : str
            One of ``"None"``, ``"Lorch"``, or ``"Hann"``.
        """
        return self.pdf_window_combo.currentText()

    def get_slice_value(self):
        """
        Get the current slice position value, if valid.

        Returns
        -------
        value : float or None
            Slice position parsed from the slice line edit, or None if
            the field does not currently contain acceptable input.
        """
        if self.slice_line.hasAcceptableInput():
            return float(self.slice_line.text())

    def get_cut_value(self):
        """
        Get the current cut position value, if valid.

        Returns
        -------
        value : float or None
            Cut position parsed from the cut line edit, or None if the
            field does not currently contain acceptable input.
        """
        if self.cut_line.hasAcceptableInput():
            return float(self.cut_line.text())

    def eventFilter(self, obj, event):
        """
        Qt event filter handling arrow-key stepping of the slice slider.

        While the slice canvas has focus, Left/Right key presses step the
        slice slider by one position; the corresponding key release emits
        `slice_ready` to trigger a re-slice. All other events are passed
        to the base class implementation.

        Parameters
        ----------
        obj : QObject
            The object the event is being filtered for (the slice canvas
            or the slice slider).
        event : QEvent
            The Qt event being processed.

        Returns
        -------
        handled : bool
            True if the event was handled here, otherwise the result of
            the base class `eventFilter`.
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
                self.slice_ready.emit()
                return True
        return super().eventFilter(obj, event)

    @staticmethod
    def _nice_step(span, n_bins):
        """
        Compute a "nice" round step size for the slice slider.

        Chooses a step of the form ``m * 10**exp`` with ``m`` in
        {1, 2, 5, 10} that is close to `span` divided evenly across
        `n_bins`, so that slider positions land on human-friendly values.

        Parameters
        ----------
        span : float
            Total span the slider must cover.
        n_bins : int
            Number of histogram bins along the slider's axis, used as the
            target number of steps.

        Returns
        -------
        step : float
            Rounded step size, at least as small as 0.01 if `span` is
            non-positive.
        """
        raw = span / max(n_bins, 1)
        if raw <= 0:
            return 0.01
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

    def _setup_slice_slider(self, smin, smax, n_bins):
        """
        Configure the slice slider's range and step for a new volume axis.

        Rescales the slider to a symmetric range around zero based on the
        larger of `smin`/`smax` in magnitude, snapped to a "nice" step
        size from `_nice_step`, and repositions the slider handle to
        match the current slice value if one is set.

        Parameters
        ----------
        smin : float
            Minimum extent of the current slice axis.
        smax : float
            Maximum extent of the current slice axis.
        n_bins : int
            Number of histogram bins along the current slice axis.
        """
        extent = max(abs(smin), abs(smax)) or 1.0
        smin, smax = -extent, extent
        span = smax - smin
        step = self._nice_step(span, n_bins)
        # Snap to step multiples so integer values are exactly reachable
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
        Update the slice position field when the slider handle moves.

        Parameters
        ----------
        pos : int
            New slider position (in slider steps).
        """
        val = self._slice_smin + pos * self._slice_step
        decimals = max(0, -int(np.floor(np.log10(self._slice_step))))
        self.slice_line.blockSignals(True)
        self.slice_line.setText(str(round(val, decimals)))
        self.slice_line.blockSignals(False)

    def set_slice_value(self, val):
        """
        Set the slice position field and move the slider to match.

        Parameters
        ----------
        val : float
            New slice position value.
        """
        self.slice_line.setText(str(round(val, 4)))
        if self._slice_smax != self._slice_smin:
            pos = int(round((val - self._slice_smin) / self._slice_step))
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(max(0, min(self._slice_steps, pos)))
            self.slice_slider.blockSignals(False)

    def set_cut_value(self, val):
        """
        Set the cut position field.

        Parameters
        ----------
        val : float
            New cut position value.
        """
        self.cut_line.setText(str(round(val, 4)))

    def get_slice_thickness(self):
        """
        Get the current slice thickness value, if valid.

        Returns
        -------
        thickness : float or None
            Slice thickness parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.slice_thickness_line.hasAcceptableInput():
            return float(self.slice_thickness_line.text())

    def get_cut_thickness(self):
        """
        Get the current cut thickness value, if valid.

        Returns
        -------
        thickness : float or None
            Cut thickness parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.cut_thickness_line.hasAcceptableInput():
            return float(self.cut_thickness_line.text())

    def set_slice_thickness(self, val):
        """
        Set the slice thickness field.

        Parameters
        ----------
        val : float
            New slice thickness value.
        """
        self.slice_thickness_line.setText(str(val))

    def set_cut_thickness(self, val):
        """
        Set the cut thickness field.

        Parameters
        ----------
        val : float
            New cut thickness value.
        """
        self.cut_thickness_line.setText(str(val))

    def get_clim_clip_type(self):
        """
        Get the currently selected color-limit clipping method for the
        3D volume display.

        Returns
        -------
        clip_type : str
            One of 'Min/Max', 'μ±3×σ', or 'Q₃/Q₁±1.5×IQR'.
        """
        return self.clim_combo.currentText()

    def get_vlim_clip_type(self):
        """
        Get the currently selected value-limit clipping method for the
        2D slice display.

        Returns
        -------
        clip_type : str
            One of 'Min/Max', 'μ±3×σ', or 'Q₃/Q₁±1.5×IQR'.
        """
        return self.vlim_combo.currentText()

    def get_slice(self):
        """
        Get the currently selected slice plane.

        Returns
        -------
        slice_plane : str
            One of 'Axis 1/2', 'Axis 1/3', or 'Axis 2/3'.
        """
        return self.slice_combo.currentText()

    def get_cut(self):
        """
        Get the currently selected line-cut axis.

        Returns
        -------
        line_cut : str
            'Axis 1' or 'Axis 2'.
        """
        return self.cut_combo.currentText()

    def get_slice_scale(self):
        """
        Get the currently selected slice display scale, lower-cased.

        Returns
        -------
        scale : str
            'linear', 'log', or 'symlog'.
        """
        return self.slice_scale_combo.currentText().lower()

    def get_cut_scale(self):
        """
        Get the currently selected cut display scale, lower-cased.

        Returns
        -------
        scale : str
            'linear' or 'log'.
        """
        return self.cut_scale_combo.currentText().lower()

    def get_vmin_value(self):
        """
        Get the current colorbar minimum value, if valid.

        Returns
        -------
        vmin : float or None
            Colorbar minimum parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.vmin_line.hasAcceptableInput():
            return float(self.vmin_line.text())

    def get_vmax_value(self):
        """
        Get the current colorbar maximum value, if valid.

        Returns
        -------
        vmax : float or None
            Colorbar maximum parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.vmax_line.hasAcceptableInput():
            return float(self.vmax_line.text())

    def get_auto_limits(self):
        """
        Get whether automatic display limit calculation is enabled.

        Returns
        -------
        auto_limits : bool
            True if the "Auto Limits" checkbox is checked.
        """
        return self.auto_limits_box.isChecked()

    def get_auto_zoom(self):
        """
        Get whether automatic zoom reset on redraw is enabled.

        Returns
        -------
        auto_zoom : bool
            True if the "Auto Zoom" checkbox is checked.
        """
        return self.auto_zoom_box.isChecked()

    def set_vmin_value(self, val):
        """
        Set the colorbar minimum field.

        Parameters
        ----------
        val : float
            New colorbar minimum value.
        """
        self.vmin_line.setText(str(round(val, 5)))

    def set_vmax_value(self, val):
        """
        Set the colorbar maximum field.

        Parameters
        ----------
        val : float
            New colorbar maximum value.
        """
        self.vmax_line.setText(str(round(val, 5)))

    def set_vmax_from_vmin(self):
        """
        Mirror the minimum field's text into the maximum field, negated.

        Used for "Symmetric About Zero": flips the sign of vmin's raw
        text rather than reformatting the numeric value, so a typed
        notation (e.g. "1e-4") is preserved instead of flip-flopping
        to a different format (e.g. "0.0001") each time either field
        is edited.

        Parameters
        ----------
        None
        """
        self.vmax_line.setText(
            self._negate_numeric_text(self.vmin_line.text())
        )

    def set_vmin_from_vmax(self):
        """
        Mirror the maximum field's text into the minimum field, negated.

        See :meth:`set_vmax_from_vmin` -- preserves vmax's typed
        notation instead of reformatting it.

        Parameters
        ----------
        None
        """
        self.vmin_line.setText(
            self._negate_numeric_text(self.vmax_line.text())
        )

    @staticmethod
    def _negate_numeric_text(text):
        """Flip the leading sign of a numeric string, keeping its notation."""
        text = text.strip()
        if text.startswith("-"):
            return text[1:]
        if text.startswith("+"):
            return "-" + text[1:]
        return "-" + text

    def set_xmax_from_xmin(self):
        """
        Mirror the X minimum field's text into the X maximum field, negated.

        Used for "Symmetric X/Y" -- see :meth:`set_vmax_from_vmin`.
        """
        self.xmax_line.setText(
            self._negate_numeric_text(self.xmin_line.text())
        )

    def set_xmin_from_xmax(self):
        """
        Mirror the X maximum field's text into the X minimum field, negated.

        Used for "Symmetric X/Y" -- see :meth:`set_vmax_from_vmin`.
        """
        self.xmin_line.setText(
            self._negate_numeric_text(self.xmax_line.text())
        )

    def set_ymax_from_ymin(self):
        """
        Mirror the Y minimum field's text into the Y maximum field, negated.

        Used for "Symmetric X/Y" -- see :meth:`set_vmax_from_vmin`.
        """
        self.ymax_line.setText(
            self._negate_numeric_text(self.ymin_line.text())
        )

    def set_ymin_from_ymax(self):
        """
        Mirror the Y maximum field's text into the Y minimum field, negated.

        Used for "Symmetric X/Y" -- see :meth:`set_vmax_from_vmin`.
        """
        self.ymin_line.setText(
            self._negate_numeric_text(self.ymax_line.text())
        )

    def get_xmin_value(self):
        """
        Get the current X-axis minimum value, if valid.

        Returns
        -------
        xmin : float or None
            X-axis minimum parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.xmin_line.hasAcceptableInput():
            return float(self.xmin_line.text())

    def get_xmax_value(self):
        """
        Get the current X-axis maximum value, if valid.

        Returns
        -------
        xmax : float or None
            X-axis maximum parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.xmax_line.hasAcceptableInput():
            return float(self.xmax_line.text())

    def set_xmin_value(self, val):
        """
        Set the X-axis minimum field.

        Parameters
        ----------
        val : float
            New X-axis minimum value.
        """
        self.xmin_line.setText(str(round(val, 4)))

    def set_xmax_value(self, val):
        """
        Set the X-axis maximum field.

        Parameters
        ----------
        val : float
            New X-axis maximum value.
        """
        self.xmax_line.setText(str(round(val, 4)))

    def get_ymin_value(self):
        """
        Get the current Y-axis minimum value, if valid.

        Returns
        -------
        ymin : float or None
            Y-axis minimum parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.ymin_line.hasAcceptableInput():
            return float(self.ymin_line.text())

    def get_ymax_value(self):
        """
        Get the current Y-axis maximum value, if valid.

        Returns
        -------
        ymax : float or None
            Y-axis maximum parsed from the field, or None if the field
            does not currently contain acceptable input.
        """
        if self.ymax_line.hasAcceptableInput():
            return float(self.ymax_line.text())

    def set_ymin_value(self, val):
        """
        Set the Y-axis minimum field.

        Parameters
        ----------
        val : float
            New Y-axis minimum value.
        """
        self.ymin_line.setText(str(round(val, 4)))

    def set_ymax_value(self, val):
        """
        Set the Y-axis maximum field.

        Parameters
        ----------
        val : float
            New Y-axis maximum value.
        """
        self.ymax_line.setText(str(round(val, 4)))

    def reset_slice_cut(self):
        """
        Clear the X/Y axis limit fields and any cached zoom state.

        Blocks signals while clearing the fields so no update handlers
        fire, then clears the remembered slice zoom limits so the next
        slice draw recomputes the view extent from scratch.
        """
        self.xmin_line.blockSignals(True)
        self.xmax_line.blockSignals(True)
        self.ymin_line.blockSignals(True)
        self.ymax_line.blockSignals(True)
        self.xmin_line.setText("")
        self.xmax_line.setText("")
        self.ymin_line.setText("")
        self.ymax_line.setText("")
        self.xmin_line.blockSignals(False)
        self.xmax_line.blockSignals(False)
        self.ymin_line.blockSignals(False)
        self.ymax_line.blockSignals(False)
        self._slice_zoom_xlim = None
        self._slice_zoom_ylim = None

    def _handle_auto_zoom_toggle(self, checked):
        """
        Reset the view to the full slice extent when auto zoom is enabled.

        Parameters
        ----------
        checked : bool
            New checked state of the "Auto Zoom" checkbox. When True (and
            a slice has already been drawn), the X/Y limit fields and
            slice plot view are reset to the full data extent.
        """
        if (
            checked
            and self.slice_im is not None
            and self.xlim is not None
            and self.ylim is not None
        ):
            self.set_xmin_value(self.xlim[0])
            self.set_xmax_value(self.xlim[1])
            self.set_ymin_value(self.ylim[0])
            self.set_ymax_value(self.ylim[1])
            self.set_slice_lim(self.xlim, self.ylim)

    def slice_limits(self, ax):
        """
        Matplotlib callback that syncs the X/Y fields to the slice zoom.

        Invoked when the slice axes' x/y limits change (e.g. via toolbar
        pan/zoom); converts the new display-space limits back into
        crystallographic axis coordinates and writes them into the
        min/max fields without re-triggering their edit signals. Also
        toggles "Auto Zoom": resetting the view via the toolbar's
        "Home" button re-enables it, while any other interactive
        pan/zoom disables it.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The slice axes whose limits changed.
        """
        self.auto_zoom_box.blockSignals(True)
        self.auto_zoom_box.setChecked(self._slice_home_clicked)
        self.auto_zoom_box.blockSignals(False)

        self.xmin_line.blockSignals(True)
        self.xmax_line.blockSignals(True)
        self.ymin_line.blockSignals(True)
        self.ymax_line.blockSignals(True)
        slice_xlim = ax.get_xlim()
        slice_ylim = ax.get_ylim()
        xmin, xmax = slice_xlim
        ymin, ymax = slice_ylim
        xmin, xmax, ymin, ymax = self._transform_bbox(
            self.T_inv, xmin, ymin, xmax, ymax
        )
        self.set_xmin_value(xmin)
        self.set_xmax_value(xmax)
        self.set_ymin_value(ymin)
        self.set_ymax_value(ymax)
        self.xmin_line.blockSignals(False)
        self.xmax_line.blockSignals(False)
        self.ymin_line.blockSignals(False)
        self.ymax_line.blockSignals(False)

    def cut_limits(self, ax):
        """
        Matplotlib callback that syncs the X/Y fields to the cut zoom.

        Invoked when the cut axes' x limits change (e.g. via toolbar
        pan/zoom); writes the new limits into the X or Y min/max fields
        depending on which axis is currently being cut, without
        re-triggering their edit signals.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The cut axes whose limits changed.
        """
        self.xmin_line.blockSignals(True)
        self.xmax_line.blockSignals(True)
        self.ymin_line.blockSignals(True)
        self.ymax_line.blockSignals(True)
        cut_lim = ax.get_xlim()
        xmin, xmax = cut_lim
        line_cut = self.get_cut()
        if line_cut == "Axis 1":
            self.set_xmin_value(xmin)
            self.set_xmax_value(xmax)
        else:
            self.set_ymin_value(xmin)
            self.set_ymax_value(xmax)
        self.xmin_line.blockSignals(False)
        self.xmax_line.blockSignals(False)
        self.ymin_line.blockSignals(False)
        self.ymax_line.blockSignals(False)

    def set_slice_lim(self, xlim, ylim):
        """
        Apply new X/Y limits to the slice plot and remember them as the
        current zoom state.

        Converts the given crystallographic-axis limits into display
        space via the current transform before applying them. Does
        nothing if no slice has been drawn yet (no colorbar).

        Parameters
        ----------
        xlim : sequence of float
            New (xmin, xmax) limits in crystallographic axis coordinates.
        ylim : sequence of float
            New (ymin, ymax) limits in crystallographic axis coordinates.
        """
        if self.cb is not None:
            xmin, xmax = xlim
            ymin, ymax = ylim
            xmin, xmax, ymin, ymax = self._transform_bbox(
                self.T, xmin, ymin, xmax, ymax
            )

            # Disconnect while setting limits programmatically so this
            # doesn't get misread by slice_limits as an interactive
            # toolbar pan/zoom (which toggles "Auto Zoom" off).
            if self._sx is not None:
                self.ax_slice.callbacks.disconnect(self._sx)
            if self._sy is not None:
                self.ax_slice.callbacks.disconnect(self._sy)

            self.ax_slice.set_xlim(xmin, xmax)
            self.ax_slice.set_ylim(ymin, ymax)

            self._sx = self.ax_slice.callbacks.connect(
                "xlim_changed", self.slice_limits
            )
            self._sy = self.ax_slice.callbacks.connect(
                "ylim_changed", self.slice_limits
            )

            self._slice_zoom_xlim = self.ax_slice.get_xlim()
            self._slice_zoom_ylim = self.ax_slice.get_ylim()
            self.canvas_slice.draw_idle()

    def set_cut_lim(self, lim):
        """
        Apply new X-axis limits to the cut plot.

        Does nothing if no slice has been drawn yet (no colorbar).

        Parameters
        ----------
        lim : sequence of float
            New (min, max) limits for the cut plot's X axis.
        """
        if self.cb is not None:
            self.ax_cut.set_xlim(*lim)
            self.canvas_cut.draw_idle()
