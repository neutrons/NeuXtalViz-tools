"""
NeuXtalVizPresenter: Presenter class for connecting the view and model in NeuXtalViz.

This module defines the NeuXtalVizPresenter class, which acts as the mediator between the user interface (view) and the underlying data/model logic. It handles user actions, updates the view based on model state, and manages status/progress reporting for the main visualization window.

Classes
-------
NeuXtalVizPresenter
    Presenter for the main visualization window, connecting view events to model logic and updating the UI accordingly.
"""


class NeuXtalVizPresenter:
    """
    Presenter for the main visualization window in NeuXtalViz.

    Connects user interface events to model logic, updates the view based on model state, and manages status and progress reporting.
    """

    def __init__(self, view, model):
        """
        Initialize the presenter and connect view signals to presenter methods.

        Parameters
        ----------
        view : object
            The view/UI instance.
        model : object
            The model instance containing data and logic.
        """

        self.view = view
        self.model = model

        self.view.connect_stop(self.stop_workers)

        self.view.connect_manual_axis(self.view_manual)
        self.view.connect_manual_up_axis(self.view_up_manual)

        self.view.connect_reciprocal_axes(
            self.view_bc_star, self.view_ca_star, self.view_ab_star
        )

        self.view.connect_real_axes(self.view_bc, self.view_ca, self.view_ab)

        self.view.connect_save_screenshot(self.save_screenshot)
        self.view.connect_reciprocal_real_compass(self.change_lattice)

        self.view.connect_rotate(self.rotate_view_ccw, self.rotate_view_cw)
        self.view.connect_elev(self.elevate_view_up, self.elevate_view_down)
        self.view.connect_az(self.azimuth_view_left, self.azimuth_view_right)

        self.view.connect_camera_ready(self.calculate_camera_parameters)

    def update_status(self, status):
        """
        Update status information in the view.

        Parameters
        ----------
        status : str
            Status message to display.
        """

        self.view.set_info(status)

    def update_progress(self, progress):
        """
        Update progress step in the view.

        Parameters
        ----------
        progress : int, str
            Progress step or value.
        """

        self.view.set_step(progress)

    def update_invalid(self):
        """
        Indicate invalid parameters to the user and reset progress.
        """

        self.update_status("Invalid parameters.")
        self.update_progress("Error.")

    def update_complete(self, status="Complete!"):
        """
        Indicate completion to the user and reset progress.

        Parameters
        ----------
        status : str, optional
            Completion message (default is "Complete!").
        """

        self.update_status(status)
        self.update_progress("Done")

    def update_processing(self, status="Processing...", progress=1):
        """
        Indicate processing state to the user and update progress.

        Parameters
        ----------
        status : str, optional
            Processing message (default is "Processing...").
        progress : int, optional
            Progress step or value (default is 1).
        """

        self.update_status(status)
        self.update_progress(progress)

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

    def stop_workers(self):
        """
        Stop all running workers and update UI.
        """
        self.view.stop_processes()
        self.update_stopped()

    def update_stopped(self):
        """
        Indicate that processing was stopped by the user.
        """

        self.update_status("Stop!")
        self.update_progress("Stopped.")

    def update_oriented_lattice(self):
        """
        Update the oriented lattice parameter display in the view.
        """

        ol = self.model.get_oriented_lattice_parameters()
        if ol is not None:
            self.view.set_oriented_lattice_parameters(*ol)

    def change_lattice(self):
        """
        Enable or disable reciprocal lattice display in the view.
        """

        T = self.model.get_transform(self.view.reciprocal_lattice())

        self.view.set_transform(T)

    def save_screenshot(self):
        """
        Save a screenshot of the current view to a file.
        """

        filename = self.view.save_screenshot_file_dialog()

        if filename:
            self.view.save_screenshot(filename)

    def view_manual(self):
        """
        Set the view to a manually specified axis direction.
        """

        indices = self.view.get_manual_axis_indices()

        if indices is not None:
            vec = self.model.get_vector(*indices)
            if vec is not None:
                self.view.view_vector(vec)

    def view_up_manual(self):
        """
        Set the view's up direction to a manually specified axis.
        """

        indices = self.view.get_manual_axis_up_indices()

        if indices is not None:
            vec = self.model.get_vector(*indices)
            if vec is not None:
                self.view.view_up_vector(vec)

    def view_ab_star(self):
        """
        Set the view to the c-axis direction (reciprocal lattice).
        """

        vecs = self.model.ab_star_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_bc_star(self):
        """
        Set the view to the a-axis direction (reciprocal lattice).
        """

        vecs = self.model.bc_star_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_ca_star(self):
        """
        Set the view to the b-axis direction (reciprocal lattice).
        """

        vecs = self.model.ca_star_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_ab(self):
        """
        Set the view to the c* direction (real lattice).
        """

        vecs = self.model.ab_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_bc(self):
        """
        Set the view to the a* direction (real lattice).
        """

        vecs = self.model.bc_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_ca(self):
        """
        Set the view to the b* direction (real lattice).
        """

        vecs = self.model.ca_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def _apply_camera_rotation(self, rotate_func, angle_deg):
        """Helper to fetch, rotate, and apply camera state via the model."""

        position, fp, up = self.view.get_camera_state()
        new_pos, new_fp, new_up = rotate_func(position, fp, up, angle_deg)
        self.view.set_camera_state(new_pos, new_fp, new_up)

    def rotate_view_ccw(self):
        """Roll the view counter‑clockwise about the viewing axis."""

        angle = self.view.get_rotate_step()
        self._apply_camera_rotation(self.model.rotate_camera_roll, angle)

    def rotate_view_cw(self):
        """Roll the view clockwise about the viewing axis."""

        angle = self.view.get_rotate_step()
        self._apply_camera_rotation(self.model.rotate_camera_roll, -angle)

    def elevate_view_up(self):
        """Tilt the camera up by the configured step angle."""

        angle = self.view.get_rotate_step()
        self._apply_camera_rotation(self.model.rotate_camera_elevation, angle)

    def elevate_view_down(self):
        """Tilt the camera down by the configured step angle."""

        angle = self.view.get_rotate_step()
        self._apply_camera_rotation(self.model.rotate_camera_elevation, -angle)

    def azimuth_view_left(self):
        """Rotate the camera left (azimuth) by the configured step angle."""

        angle = self.view.get_rotate_step()
        self._apply_camera_rotation(self.model.rotate_camera_azimuth, -angle)

    def azimuth_view_right(self):
        """Rotate the camera right (azimuth) by the configured step angle."""

        angle = self.view.get_rotate_step()
        self._apply_camera_rotation(self.model.rotate_camera_azimuth, angle)

    def calculate_camera_parameters(self):
        roll, dir = self.view.get_camera_roll_direction()
        roll, elevation, azimuth = self.model.calculate_camera_parameters(
            dir, roll
        )
        self.view.update_camera_display(roll, elevation, azimuth)

        _, _, up = self.view.get_camera_state()
        view_hkl = self.model.get_indices("[hkl]", dir)
        up_hkl = self.model.get_indices("[hkl]", up)
        self.view.update_camera_info(dir, up, view_hkl, up_hkl)
