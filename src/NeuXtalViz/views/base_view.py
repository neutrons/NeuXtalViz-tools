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
)

from qtpy.QtGui import QDoubleValidator, QFont
from PyQt5.QtCore import Qt, pyqtSignal

import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

from NeuXtalViz.views.utilities import Worker, ThreadPool


class NeuXtalVizWidget(QWidget):
    """
    Base widget for all NeuXtalViz views, providing shared functionality
    and interface for user-facing widgets.
    """

    log_output = pyqtSignal(str)

    def __init__(self, parent=None):
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

        self.camera_button = QPushButton("Reset Camera", self)
        self.camera_button.setToolTip("Reset the camera position.")
        self.camera_button.clicked.connect(self.reset_camera)

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

        self.theme_combo = QComboBox(self)
        self.theme_combo.addItem("default")
        self.theme_combo.addItem("document")
        self.theme_combo.addItem("dark")
        self.theme_combo.addItem("paraview")
        self.theme_combo.setToolTip("Select theme.")
        self.theme_combo.currentIndexChanged.connect(self.update_theme)

        self.frame = QFrame()

        self.plotter = QtInteractor(self.frame)

        layout = QHBoxLayout()
        vis_layout = QVBoxLayout()

        camera_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # left_layout.addStretch(1)
        left_layout.addWidget(self.save_button)
        left_layout.addWidget(self.reset_button)
        left_layout.addWidget(self.camera_button)
        left_layout.addWidget(self.theme_combo)

        # right_layout.addStretch(1)
        right_layout.addWidget(self.recip_box)
        right_layout.addWidget(self.axes_box)
        right_layout.addWidget(self.proj_box)
        right_layout.addWidget(self.cons_box)

        view_tab = self.__init_view_tab()

        camera_layout.addLayout(left_layout)
        camera_layout.addWidget(view_tab)
        camera_layout.addLayout(right_layout)

        vis_layout.addLayout(camera_layout)
        vis_layout.addWidget(self.plotter.interactor)
        vis_layout.setStretch(1, 1)

        info_tab = self.__init_info_tab()

        vis_layout.addWidget(info_tab)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready!")
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

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

    def __init_view_tab(self):
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

        view_tab.addTab(directions_tab, "Direction View")
        view_tab.addTab(manual_tab, "Manual View")

        return view_tab

    def __init_info_tab(self):
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
        self.console.appendPlainText(text)

    def toggle_console(self, state):
        self.console.setVisible(bool(state))

    def toggle_console(self, state):
        self.console.setVisible(bool(state))

    def start_worker_pool(self, worker):
        """
        Create a worker pool and connect output to console.
        """
        worker.signals.output.connect(self.append_to_console)
        self.threadpool.start_worker_pool(worker)

    def worker(self, task):
        """
        Worker task.

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
        Manual axis upv iew connection.

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
        Reset the view.

        """

        self.plotter.reset_camera()
        self.plotter.view_isometric(negative)
        self.camera_position = self.plotter.camera_position

    def reset_camera(self):
        """
        Reset the camera.

        """

        self.plotter.reset_camera()

    def clear_scene(self):
        self.plotter.clear_plane_widgets()
        self.plotter.clear_actors()

        if self.camera_position is not None:
            self.camera_position = self.plotter.camera_position

    def reset_scene(self):
        if self.camera_position is not None:
            self.plotter.camera_position = self.camera_position
        else:
            self.reset_view()

    def save_screenshot(self, filename):
        """
        Save plotter screenshot.

        Parameters
        ----------
        filename : str
            Filename with *.png extension.

        """

        self.plotter.screenshot(filename)

    def save_screenshot_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self, "Save PNG file", "", "PNG files (*.png)", options=options
        )

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

        """

        return self.recip_box.isChecked()

    def show_axes(self):
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

    def view_up_vector(self, vec):
        """
        Set the camera according to given vector(s).

        Parameters
        ----------
        vec : 3 element 1d array-like
            Camera up direction and optional upward vector.

        """

        self.plotter.set_viewup(vec)

    def update_theme(self):
        theme = self.theme_combo.currentText()
        pv.set_plot_theme(theme)
        bg = pv.global_theme.background
        self.plotter.set_background(bg)
        self.plotter.render()
        # call twice to update colors
        self.show_axes()
        self.show_axes()

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
