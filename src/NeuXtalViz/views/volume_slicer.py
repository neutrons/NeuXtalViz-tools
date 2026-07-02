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
    slice_ready = Signal()
    cut_ready = Signal()

    def __init__(self, parent=None):
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
        self.container.setVisible(state)
        self.update_lines(state)

    def connect_save_slice(self, save_slice):
        self.save_slice_button.clicked.connect(save_slice)

    def connect_save_cut(self, save_cut):
        self.save_cut_button.clicked.connect(save_cut)

    def connect_vol_scale_combo(self, update_vol):
        self.vol_scale_combo.currentIndexChanged.connect(update_vol)

    def connect_opacity_combo(self, update_opacity):
        self.opacity_combo.currentIndexChanged.connect(update_opacity)

    def connect_range_combo(self, update_range):
        self.range_combo.currentIndexChanged.connect(update_range)

    def connect_clim_combo(self, update_clim):
        self.clim_combo.currentIndexChanged.connect(update_clim)

    def connect_vlim_combo(self, update_clim):
        self.vlim_combo.currentIndexChanged.connect(update_clim)

    def connect_cbar_combo(self, update_cbar):
        self.cbar_combo.currentIndexChanged.connect(update_cbar)

    def connect_slice_thickness_line(self, update_slice):
        self.slice_thickness_line.editingFinished.connect(update_slice)

    def connect_cut_thickness_line(self, update_cut):
        self.cut_thickness_line.editingFinished.connect(update_cut)

    def connect_slice_line(self, update_slice):
        self.slice_line.editingFinished.connect(update_slice)

    def connect_cut_line(self, update_cut):
        self.cut_line.editingFinished.connect(update_cut)

    def connect_slice_scale_combo(self, update_slice):
        self.slice_scale_combo.currentIndexChanged.connect(update_slice)

    def connect_cut_scale_combo(self, update_cut):
        self.cut_scale_combo.currentIndexChanged.connect(update_cut)

    def _reset_slice_to_zero(self):
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
        self.slice_combo.currentIndexChanged.connect(self._reset_slice_to_zero)
        self.slice_combo.currentIndexChanged.connect(update_slice)

    def connect_cut_combo(self, update_cut):
        self.cut_combo.currentIndexChanged.connect(update_cut)

    def connect_min_slider(self, update_colorbar):
        pass

    def connect_max_slider(self, update_colorbar):
        pass

    def connect_vmin_line(self, update_vals):
        self.vmin_line.editingFinished.connect(update_vals)

    def connect_vmax_line(self, update_vals):
        self.vmax_line.editingFinished.connect(update_vals)

    def connect_auto_limits(self, update_limits):
        self.auto_limits_box.toggled.connect(update_limits)

    def connect_auto_zoom(self, update_zoom):
        self.auto_zoom_box.toggled.connect(update_zoom)

    def connect_xmin_line(self, update_vals):
        self.xmin_line.editingFinished.connect(update_vals)

    def connect_xmax_line(self, update_vals):
        self.xmax_line.editingFinished.connect(update_vals)

    def connect_ymin_line(self, update_vals):
        self.ymin_line.editingFinished.connect(update_vals)

    def connect_ymax_line(self, update_vals):
        self.ymax_line.editingFinished.connect(update_vals)

    def save_file_dialog(self):
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
        pass

    def update_colorbar_max(self):
        pass

    def update_slice_color(self):
        if self.cb is not None:
            min_slider, max_slider = self.get_color_bar_values()

            vmin = self.vmin + (self.vmax - self.vmin) * min_slider / 100
            vmax = self.vmin + (self.vmax - self.vmin) * max_slider / 100

            self.update_colorbar_vlims(vmin, vmax)

    def update_colorbar_vlims(self, vmin, vmax):
        if self.cb is not None and self.slice_im is not None:
            self.set_vmin_value(vmin)
            self.set_vmax_value(vmax)

            self.slice_im.set_clim(vmin=vmin, vmax=vmax)
            self.cb.update_normal(self.slice_im)
            self.cb.minorticks_on()

            self.canvas_slice.draw_idle()

    def _create_norm(self, scale, vmin, vmax):
        scale = scale.lower()

        if scale == "log":
            vmin = max(vmin, np.finfo(float).tiny)
            return mcolors.LogNorm(vmin=vmin, vmax=vmax)

        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def update_slice_display(self, cmap_key, scale, vmin, vmax):
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
        return 0, 100

    def reset_slider(self):
        pass

    def connect_load_NXS(self, load_NXS):
        self.load_NXS_button.clicked.connect(load_NXS)

    def load_NXS_file_dialog(self):
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
        orig = caller.GetOrigin()

        ind = np.abs(self.norm).tolist().index(1)

        value = np.dot(self.P_inv, orig)[ind]

        self.slice_line.blockSignals(True)
        self.set_slice_value(value)
        self.slice_line.blockSignals(False)

        self.slice_ready.emit()

    def update_clip(self, origin=None, normal=None):
        if origin is not None:
            self.clip.SetOrigin(*origin)
        if normal is not None:
            self.clip.SetNormal(*normal)

        self.plotter.update()

    def connect_slice_ready(self, reslice):
        self.slice_ready.connect(reslice)

    def __format_axis_coord(self, x, y):
        x, y, _ = np.dot(self.T_inv, [x, y, 1])
        h, k, l = np.dot(self.W, [x, y, self.z])
        return "hkl = ({:.3f}, {:.3f}, {:.3f})".format(h, k, l)

    def add_slice(self, slice_dict):
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
        lines = (
            self._cut_lines
            if self._cut_lines is not None
            else self.ax_slice.get_lines()
        )
        for line in lines:
            line.set_alpha(alpha)
        self.canvas_slice.draw_idle()

    def add_cut(self, cut_dict):
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
        if (
            event.inaxes == self.ax_slice
            and self.fig_slice.canvas.toolbar.mode == ""
            and self.toggle_line_box.isChecked()
        ):
            self.linecut["is_dragging"] = True

    def on_release(self, event):
        self.linecut["is_dragging"] = False

        self.cut_ready.emit()

    def connect_cut_ready(self, recut):
        self.cut_ready.connect(recut)

    def on_motion(self, event):
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
        return self.vol_scale_combo.currentText()

    def get_opacity(self):
        return self.opacity_combo.currentText()

    def get_range(self):
        return self.range_combo.currentText()

    def get_colormap(self):
        return self.cbar_combo.currentText()

    def get_slice_value(self):
        if self.slice_line.hasAcceptableInput():
            return float(self.slice_line.text())

    def get_cut_value(self):
        if self.cut_line.hasAcceptableInput():
            return float(self.cut_line.text())

    def eventFilter(self, obj, event):
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
        val = self._slice_smin + pos * self._slice_step
        decimals = max(0, -int(np.floor(np.log10(self._slice_step))))
        self.slice_line.blockSignals(True)
        self.slice_line.setText(str(round(val, decimals)))
        self.slice_line.blockSignals(False)

    def set_slice_value(self, val):
        self.slice_line.setText(str(round(val, 4)))
        if self._slice_smax != self._slice_smin:
            pos = int(round((val - self._slice_smin) / self._slice_step))
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(max(0, min(self._slice_steps, pos)))
            self.slice_slider.blockSignals(False)

    def set_cut_value(self, val):
        self.cut_line.setText(str(round(val, 4)))

    def get_slice_thickness(self):
        if self.slice_thickness_line.hasAcceptableInput():
            return float(self.slice_thickness_line.text())

    def get_cut_thickness(self):
        if self.cut_thickness_line.hasAcceptableInput():
            return float(self.cut_thickness_line.text())

    def set_slice_thickness(self, val):
        self.slice_thickness_line.setText(str(val))

    def set_cut_thickness(self, val):
        self.cut_thickness_line.setText(str(val))

    def get_clim_clip_type(self):
        return self.clim_combo.currentText()

    def get_vlim_clip_type(self):
        return self.vlim_combo.currentText()

    def get_slice(self):
        return self.slice_combo.currentText()

    def get_cut(self):
        return self.cut_combo.currentText()

    def get_slice_scale(self):
        return self.slice_scale_combo.currentText().lower()

    def get_cut_scale(self):
        return self.cut_scale_combo.currentText().lower()

    def get_vmin_value(self):
        if self.vmin_line.hasAcceptableInput():
            return float(self.vmin_line.text())

    def get_vmax_value(self):
        if self.vmax_line.hasAcceptableInput():
            return float(self.vmax_line.text())

    def get_auto_limits(self):
        return self.auto_limits_box.isChecked()

    def get_auto_zoom(self):
        return self.auto_zoom_box.isChecked()

    def set_vmin_value(self, val):
        self.vmin_line.setText(str(round(val, 5)))

    def set_vmax_value(self, val):
        self.vmax_line.setText(str(round(val, 5)))

    def get_xmin_value(self):
        if self.xmin_line.hasAcceptableInput():
            return float(self.xmin_line.text())

    def get_xmax_value(self):
        if self.xmax_line.hasAcceptableInput():
            return float(self.xmax_line.text())

    def set_xmin_value(self, val):
        self.xmin_line.setText(str(round(val, 4)))

    def set_xmax_value(self, val):
        self.xmax_line.setText(str(round(val, 4)))

    def get_ymin_value(self):
        if self.ymin_line.hasAcceptableInput():
            return float(self.ymin_line.text())

    def get_ymax_value(self):
        if self.ymax_line.hasAcceptableInput():
            return float(self.ymax_line.text())

    def set_ymin_value(self, val):
        self.ymin_line.setText(str(round(val, 4)))

    def set_ymax_value(self, val):
        self.ymax_line.setText(str(round(val, 4)))

    def reset_slice_cut(self):
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
        if self.cb is not None:
            self.ax_cut.set_xlim(*lim)
            self.canvas_cut.draw_idle()
