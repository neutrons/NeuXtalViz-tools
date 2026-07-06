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

opacities = {
    "Linear": {"Low->High": "linear", "High->Low": "linear_r"},
    "Geometric": {"Low->High": "geom", "High->Low": "geom_r"},
    "Sigmoid": {"Low->High": "sigmoid", "High->Low": "sigmoid_r"},
}


class VolumeSlicerView(NeuXtalVizWidget):
    """
    View for the volume slicer tool.

    Provides the UI for loading a NeXus MDHisto workspace, rendering it as
    a 3D clipped volume, slicing it into a 2D plane, and cutting a 1D line
    profile through that slice, together with controls for scaling,
    colormaps, opacity, and display/color limits.

    Attributes
    ----------
    slice_ready : qtpy.QtCore.Signal
        Emitted when the slice position/parameters are ready to be
        recomputed (e.g. after dragging the 3D clip plane or slider).
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

        self.slicer_tab()

        self.layout().addWidget(self.tab_widget, stretch=1)

        self.reset_slice_cut()

        self._cx = None
        self._sx = None
        self._sy = None
        self.slice_im = None
        self.xlim = None
        self.ylim = None
        self._cut_lines = None
        self._slice_zoom_xlim = None
        self._slice_zoom_ylim = None
        self._volume_limits = None
        self._volume_nbins = None

    def slicer_tab(self):
        """
        Build the "Slicer" tab, its widgets, and its layouts.

        Creates the volume/opacity/colormap controls, slice and cut
        position/thickness controls, axis limit fields, the 2D slice and
        1D cut matplotlib canvases, and wires up the toggle/auto-zoom
        checkboxes.
        """
        slice_tab = QWidget()
        self.tab_widget.addTab(slice_tab, "Slicer")

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
        self.opacity_combo.addItem("Linear")
        self.opacity_combo.addItem("Geometric")
        self.opacity_combo.addItem("Sigmoid")
        self.opacity_combo.setCurrentIndex(0)
        self.opacity_combo.setToolTip(
            "Choose the opacity mapping for the volume rendering."
        )
        self.auto_scale_dropdown(self.opacity_combo)

        self.range_combo = QComboBox(self)
        self.range_combo.addItem("Low->High")
        self.range_combo.addItem("High->Low")
        self.range_combo.setCurrentIndex(0)
        self.range_combo.setToolTip(
            "Set the direction of the opacity or color range."
        )
        self.auto_scale_dropdown(self.range_combo)

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
        self.cbar_combo.addItem("Diverging")
        self.cbar_combo.addItem("Modified")
        self.cbar_combo.setToolTip(
            "Select the colormap for the slice visualization."
        )
        self.auto_scale_dropdown(self.cbar_combo)

        self.load_NXS_button = QPushButton("Load NXS", self)
        self.load_NXS_button.setToolTip(
            "Load a NeXus (NXS) file for volume slicing."
        )
        self.load_NXS_button.setIcon(qta.icon("fa6s.folder-open"))

        draw_layout.addWidget(self.vol_scale_combo)
        draw_layout.addWidget(self.opacity_combo)
        draw_layout.addWidget(self.range_combo)
        draw_layout.addWidget(self.clim_combo)
        draw_layout.addWidget(self.cbar_combo)
        draw_layout.addWidget(self.load_NXS_button)

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
        self.slice_scale_combo.setToolTip(
            "Select the scale for the slice plot (Linear or Logarithmic)."
        )
        self.auto_scale_dropdown(self.slice_scale_combo)

        self.cut_scale_combo = QComboBox(self)
        self.cut_scale_combo.addItem("Linear")
        self.cut_scale_combo.addItem("Log")
        self.cut_scale_combo.setToolTip(
            "Select the scale for the cut plot (Linear or Logarithmic)."
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

        self.vmin_line.setValidator(validator)
        self.vmax_line.setValidator(validator)
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

        view_params_layout.addWidget(self.toggle_line_box, 1, 4)
        view_params_layout.addWidget(self.vlim_combo, 0, 4)
        view_params_layout.addWidget(vmin_label, 1, 5)
        view_params_layout.addWidget(self.vmin_line, 1, 6)
        view_params_layout.addWidget(vmax_label, 0, 5)
        view_params_layout.addWidget(self.vmax_line, 0, 6)

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

        fig_2d_layout.addWidget(NavigationToolbar2QT(self.canvas_slice, self))
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

    def connect_range_combo(self, update_range):
        """
        Connect a handler to changes in the opacity/color range combo box.

        Parameters
        ----------
        update_range : callable
            Slot invoked when the range direction selection changes.
        """
        self.range_combo.currentIndexChanged.connect(update_range)

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
            Display scale, 'log' (case-insensitive) for logarithmic
            normalization or any other value for linear normalization.
        vmin : float
            Lower color limit. Clamped to the smallest positive finite
            float when `scale` is 'log'.
        vmax : float
            Upper color limit.

        Returns
        -------
        norm : matplotlib.colors.Normalize or matplotlib.colors.LogNorm
            Normalization object for the slice image.
        """
        scale = scale.lower()

        if scale == "log":
            vmin = max(vmin, np.finfo(float).tiny)
            return mcolors.LogNorm(vmin=vmin, vmax=vmax)

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

    def add_histo(self, histo_dict, normal, norm, value):
        """
        Render the 3D clipped volume for the loaded histogram.

        Builds a PyVista `ImageData` grid from the histogram signal,
        applies the current opacity/colormap/scale settings, adds a
        clipped volume actor with a movable clip plane at the given
        origin/normal, configures the bounding-box axes with the
        crystallographic transform, and resets the camera to fit the
        scene. Also (re)configures the slice position slider range for
        the newly selected plane and registers a callback so that
        interactively moving the clip plane updates the slice position.

        Parameters
        ----------
        histo_dict : dict
            Histogram information dictionary (as returned by the model's
            `get_histo_info`) containing 'signal', 'labels', 'min_lim',
            'max_lim', 'spacing', 'projection', 'transform', and 'scales'.
        normal : numpy.ndarray
            Plane normal vector (in Cartesian/plot space) used to orient
            the volume clip plane.
        norm : array-like
            Normal vector identifying the selected slice plane in
            crystallographic axis space (e.g. [0, 0, 1]); mutated in
            place to build the clip plane origin.
        value : float
            Position along `norm` at which to place the initial clip
            plane / slice.
        """
        opacity = opacities[self.get_opacity()][self.get_range()]

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

        a = pv._vtk.vtkMatrix3x3()
        b = pv._vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                a.SetElement(i, j, T[i, j])
                b.SetElement(i, j, P[i, j])

        grid.cell_data["scalars"] = signal.flatten(order="F")

        normal /= np.linalg.norm(normal)

        origin = np.dot(P, origin)

        clim = [np.nanmin(signal), np.nanmax(signal)]

        if not np.all(np.isfinite(clim)):
            clim = [0.1, 10]

        self.clip = self.plotter.add_volume_clip_plane(
            grid,
            opacity=opacity,
            log_scale=log_scale,
            clim=clim,
            normal=normal,
            origin=origin,
            origin_translation=False,
            show_scalar_bar=True,
            normal_rotation=False,
            cmap=cmap,
            user_matrix=b,
        )

        prop = self.clip.GetOutlineProperty()
        prop.SetOpacity(0)

        prop = self.clip.GetEdgesProperty()
        prop.SetOpacity(0)

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

        self.clip.AddObserver("InteractionEvent", self.interaction_callback)

        self.P_inv = np.linalg.inv(P)

    def interaction_callback(self, caller, event):
        """
        VTK observer callback fired while the 3D clip plane is dragged.

        Converts the clip plane's current origin back into slice-position
        units, updates the slice position field without re-triggering its
        edit signal, and emits `slice_ready` to request a re-slice.

        Parameters
        ----------
        caller : vtkImplicitPlaneWidget or similar
            The VTK widget/actor that raised the interaction event; its
            `GetOrigin()` is used to obtain the new clip plane origin.
        event : str
            The VTK event name (unused, required by the observer
            signature).
        """
        orig = caller.GetOrigin()

        ind = np.abs(self.norm).tolist().index(1)

        value = np.dot(self.P_inv, orig)[ind]

        self.slice_line.blockSignals(True)
        self.set_slice_value(value)
        self.slice_line.blockSignals(False)

        self.slice_ready.emit()

    def update_clip(self, origin=None, normal=None):
        """
        Update the 3D volume clip plane's origin and/or normal.

        Parameters
        ----------
        origin : array-like, optional
            New origin point for the clip plane (default None, meaning
            unchanged).
        normal : array-like, optional
            New normal vector for the clip plane (default None, meaning
            unchanged).
        """
        if origin is not None:
            self.clip.SetOrigin(*origin)
        if normal is not None:
            self.clip.SetNormal(*normal)

        self.plotter.update()

    def connect_slice_ready(self, reslice):
        """
        Connect a handler to the `slice_ready` signal.

        Parameters
        ----------
        reslice : callable
            Slot invoked when `slice_ready` is emitted.
        """
        self.slice_ready.connect(reslice)

    def __format_axis_coord(self, x, y):
        """
        Format slice-plot display coordinates as an HKL string.

        Used as the matplotlib Axes `format_coord` callback to show the
        crystallographic HKL indices under the cursor in the slice plot's
        status readout.

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
            'signal', 'z', 'W', 'transform', 'aspect', and optionally
            'vmin'/'vmax'.
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

            xmin, ymin, _ = np.dot(self.T, [xmin, ymin, 1])
            xmax, ymax, _ = np.dot(self.T, [xmax, ymax, 1])

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
            One of 'Linear', 'Geometric', or 'Sigmoid'.
        """
        return self.opacity_combo.currentText()

    def get_range(self):
        """
        Get the currently selected opacity/color range direction.

        Returns
        -------
        range_dir : str
            'Low->High' or 'High->Low'.
        """
        return self.range_combo.currentText()

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
            'linear' or 'log'.
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
        min/max fields without re-triggering their edit signals.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The slice axes whose limits changed.
        """
        self.xmin_line.blockSignals(True)
        self.xmax_line.blockSignals(True)
        self.ymin_line.blockSignals(True)
        self.ymax_line.blockSignals(True)
        slice_xlim = ax.get_xlim()
        slice_ylim = ax.get_ylim()
        xmin, xmax = slice_xlim
        ymin, ymax = slice_ylim
        xmin, ymin, _ = np.dot(self.T_inv, [xmin, ymin, 1])
        xmax, ymax, _ = np.dot(self.T_inv, [xmax, ymax, 1])
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
            xmin, ymin, _ = np.dot(self.T, [xmin, ymin, 1])
            xmax, ymax, _ = np.dot(self.T, [xmax, ymax, 1])
            self.ax_slice.set_xlim(xmin, xmax)
            self.ax_slice.set_ylim(ymin, ymax)
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
