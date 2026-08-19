from NeuXtalViz.presenters.base_presenter import NeuXtalVizPresenter

import functools


class Experiment(NeuXtalVizPresenter):
    """
    Presenter for the experiment planner in NeuXtalViz.

    Connects the experiment-planner view's signals to model logic for
    goniometer/instrument setup, peak coverage calculation and
    visualization, plane/mesh scan generation, coverage optimization,
    and experiment plan save/load.
    """

    def __init__(self, view, model):
        """
        Wire up the experiment planner view signals and initialize state.

        Parameters
        ----------
        view : object
            The experiment planner view/UI instance.
        model : object
            The experiment planner model instance containing data and
            logic.
        """

        super(Experiment, self).__init__(view, model)

        self.view.connect_load_UB(self.load_UB)
        self.view.connect_reset(self.add_settings)
        self.view.connect_show_instrument(self.show_instrument)
        self.view.connect_switch_instrument(self.switch_instrument)
        self.view.connect_update_goniometer(self.update_goniometer)
        self.view.connect_switch_crystal(self.switch_crystal)
        self.view.connect_switch_point_group(self.switch_group)
        self.view.connect_switch_lattice_centering(self.switch_centering)
        self.view.connect_wavelength(self.update_wavelength)
        self.view.connect_hkl_limits(self.update_hkl_limits)
        self.view.connect_optimize(self.optimize_coverage)
        self.view.connect_mesh(self.mesh_scan)
        self.view.connect_calculate_plane(self.calculate_plane)
        self.view.connect_add_plane(self.plane_scan)
        self.view.connect_calculate_single(self.calculate_single)
        self.view.connect_calculate_double(self.calculate_double)
        self.view.connect_calculate_single_alt(self.calculate_single_alt)
        self.view.connect_add_orientation(self.add_orientation)
        self.view.connect_delete_angles(self.delete_angles)
        self.view.connect_save_CSV(self.save_CSV)
        self.view.connect_save_experiment(self.save_experiment)
        self.view.connect_load_experiment(self.load_experiment)
        self.view.connect_combined(self.visualize)
        self.view.connect_color_scheme(self.visualize)
        self.view.connect_peak_table(self.update_peaks)
        self.view.connect_load_mask(self.load_mask)
        self.view.connect_load_detector(self.load_detector)
        self.view.connect_load_goniometer(self.load_goniometer)
        self.view.connect_convert_mesh_to_hkl(self.convert_mesh_to_hkl)
        self.view.connect_convert_plan_to_hkl(self.convert_plan_to_hkl)

        self.view.connect_slice_combo(self.reslice)
        self.view.connect_slice_thickness_line(self.reslice)
        self.view.connect_slice_line(self.reslice)

        self.view.connect_roi_ready(self.lookup_angle)
        self.view.connect_selection_ready(self.select_peak)
        self.view.connect_visualization_ready(self.visualize)
        self.view.connect_harmonic_ready(self.calculate_harmonics)

        self.view.connect_update(self.view.update_counting)
        self.view.connect_highlight_angles(self.view.highlight_angles)
        self.view.connect_peak_row_highlighter(self.highlight_peak)
        self.view.connect_move_up(self.move_orientation_up)
        self.view.connect_move_down(self.move_orientation_down)

        self.draw_idle = True

        self.switch_instrument()
        self.switch_crystal()

        self.view.set_default_symmetry()

        self.mesh = True
        self.convert_idle = True
        self.slice_only = False

    def load_detector(self):
        """
        Load detector calibration file and set it in the view.

        """
        inst = self.view.get_instrument()
        path = self.model.get_calibration_file_path(inst)
        filename = self.view.load_detector_cal_dialog(path)

        if filename:
            self.view.set_detector_calibration(filename)

    def load_goniometer(self):
        """
        Load goniometer calibration file and set it in the view.

        """
        inst = self.view.get_instrument()
        path = self.model.get_calibration_file_path(inst)
        filename = self.view.load_goniometer_cal_dialog(path)

        if filename:
            self.view.set_goniometer_calibration(filename)

    def load_mask(self):
        """
        Load detector mask file and set it in the view.

        """
        inst = self.view.get_instrument()
        path = self.model.get_calibration_file_path(inst)
        filename = self.view.load_mask_dialog(path)

        if filename:
            self.view.set_mask(filename)

    def load_UB(self):
        """
        Load UB matrix from file and update the view and model.

        """
        inst = self.view.get_instrument()
        path = self.model.get_instrument_directory(inst)
        filename = self.view.load_UB_file_dialog(path)

        if filename:
            self.model.load_UB(filename)

            self.update_oriented_lattice()

            self.view.set_transform(self.model.get_transform())

            self.update_hkl_limits()

    def switch_instrument(self):
        """
        Switch instrument and update all related view parameters.

        """
        instrument = self.view.get_instrument()

        wavelength = self.model.get_wavelength(instrument)
        motors = self.model.get_motors(instrument)
        modes = self.model.get_modes(instrument)
        goniometers = self.model.get_goniometers(instrument, modes[0])
        options = self.model.get_counting_options(instrument)
        title = self.model.get_scan_log(instrument)

        self.view.set_modes(modes)
        self.view.set_wavelength(wavelength)
        self.view.update_tables(title, goniometers, motors)
        self.view.set_counting_options(options)

        self.model.remove_instrument()

    def switch_crystal(self):
        """
        Update available point groups when the crystal system changes.

        Slot for the crystal system combo box's activation signal.
        Fetches the point groups compatible with the selected crystal
        system, updates the view, and refreshes the lattice centering
        list via :meth:`switch_group`.
        """

        cs = self.view.get_crystal_system()

        point_groups = self.model.get_crystal_system_point_groups(cs)

        self.view.set_point_groups(point_groups)

        self.switch_group()

    def switch_group(self):
        """
        Update available lattice centerings when the point group changes.

        Slot for the point group combo box's activation signal. Fetches
        the lattice centerings compatible with the selected point group,
        updates the view, and triggers a re-visualization.
        """

        pg = self.view.get_point_group()

        centerings = self.model.get_point_group_centering(pg)

        self.view.set_lattice_centerings(centerings)

        self.visualize()

    def switch_centering(self):
        """
        Re-visualize the coverage when the lattice centering changes.

        Slot for the lattice centering combo box's activation signal.
        """

        self.visualize()

    def update_hkl_limits(self):
        """
        Recompute and display HKL limits from the current d-min and UB.

        Slot for the d-min line edit's ``editingFinished`` signal (and
        called after loading a UB matrix). Does nothing if no d-min is
        set or no UB matrix has been loaded.
        """

        d_min = self.view.get_d_min()
        if d_min is not None and self.model.has_UB():
            hkl_limits = self.model.calculate_hkl_limits(d_min)
            self.view.set_hkl_limits(*hkl_limits)

    def update_goniometer(self):
        """
        Refresh the goniometer/motor tables for the selected mode.

        Slot for the goniometer mode combo box's activation signal.
        """

        instrument = self.view.get_instrument()
        mode = self.view.get_mode()

        goniometers = self.model.get_goniometers(instrument, mode)
        motors = self.model.get_motors(instrument)
        title = self.model.get_scan_log(instrument)

        self.view.update_tables(title, goniometers, motors)

    def update_wavelength(self):
        """
        Update the derived wavelength display from the minimum wavelength.

        Slot for the minimum wavelength line edit's ``editingFinished``
        signal.
        """

        wl_min, _ = self.view.get_wavelength()
        self.view.update_wavelength(wl_min)

    def show_instrument(self):
        """
        Launch a background worker to render the instrument geometry.

        Slot for the "Show Instrument" button. Gathers the current
        instrument/motor/calibration settings from the view and runs
        :meth:`show_instrument_process` in a worker thread, updating the
        view with the result via :meth:`show_instrument_complete` and
        refreshing the HKL limits when finished.
        """

        instrument = self.view.get_instrument()
        motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()

        worker = self.view.worker(
            functools.partial(
                self.show_instrument_process,
                instrument=instrument,
                motors=motors,
                cal=cal,
                gon_cal=gon_cal,
                mask=mask,
            )
        )
        worker.connect_result(self.show_instrument_complete)
        worker.connect_finished(self.update_hkl_limits)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def show_instrument_complete(self, result):
        """
        Add the computed instrument geometry to the 3D view.

        Parameters
        ----------
        result : dict or None
            Instrument mesh dictionary from
            :meth:`show_instrument_process` (as returned by
            ``model.extract_instrument_view``), or None if the
            calculation was cancelled or failed.
        """

        if result is not None:
            self.view.add_instrument(result)

    def show_instrument_process(
        self,
        progress,
        stop_event=None,
        instrument=None,
        motors=None,
        cal=None,
        gon_cal=None,
        mask=None,
    ):
        """
        Worker task: initialize the instrument and extract its geometry.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        instrument : str, optional
            Instrument identifier.
        motors : list, optional
            Auxiliary motor ``(name, value)`` settings.
        cal : str, optional
            Detector calibration file path.
        gon_cal : str, optional
            Goniometer calibration file path.
        mask : str, optional
            Detector mask file path.

        Returns
        -------
        inst_dict : dict or None
            Instrument mesh dictionary with ``"points"``, ``"faces"``,
            and ``"radius"`` keys, or None if the worker was stopped
            early.
        """

        if self.stop_processing(stop_event):
            return None

        progress("Initializing instrument", 5)

        if self.stop_processing(stop_event):
            return None

        self.model.initialize_instrument(
            instrument, motors, cal, gon_cal, mask
        )

        progress("Calculating instrument view.", 5)

        if self.stop_processing(stop_event):
            return None

        inst_dict = self.model.extract_instrument_view()

        progress("Instrument view calculated!", 0)

        return inst_dict

    def create_instrument(self):
        """
        Initialize the instrument workspace with the current view settings.

        Reads the instrument, motors, and calibration/mask files from
        the view and initializes the model's instrument workspace
        synchronously (not run in a background worker).
        """

        instrument = self.view.get_instrument()
        motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()

        self.model.initialize_instrument(
            instrument, motors, cal, gon_cal, mask
        )

    def calculate_single(self):
        """
        Calculate peak coverage for the first (primary) input HKL.

        Slot for the "Calculate Single" button. Selects the primary
        HKL input and delegates to :meth:`calculate_single_hkl`.
        """

        self.alt_hkl = False
        self.calculate_single_hkl()

    def calculate_single_alt(self):
        """
        Calculate peak coverage for the second (alternate) input HKL.

        Slot for the alternate "Calculate Single" button. Selects the
        alternate HKL input and delegates to :meth:`calculate_single_hkl`.
        """

        self.alt_hkl = True
        self.calculate_single_hkl()

    def calculate_single_hkl(self):
        """
        Launch a background worker to calculate a single reflection's
        detector coverage across all goniometer settings.

        Gathers the HKL, wavelength, symmetry, and instrument settings
        from the view and runs :meth:`calculate_single_process` in a
        worker thread, plotting the result via
        :meth:`calculate_single_complete` when finished.
        """

        hkl_1, hkl_2 = self.view.get_input_hkls()
        wavelength = self.view.get_wavelength()
        equiv = self.view.use_equivalents()
        pg = self.view.get_point_group()
        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        limits = self.view.get_goniometer_limits()
        instr_motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()

        worker = self.view.worker(
            functools.partial(
                self.calculate_single_process,
                hkl_1=hkl_1,
                hkl_2=hkl_2,
                wavelength=wavelength,
                equiv=equiv,
                pg=pg,
                instrument=instrument,
                mode=mode,
                limits=limits,
                instr_motors=instr_motors,
                cal=cal,
                gon_cal=gon_cal,
                mask=mask,
            )
        )
        worker.connect_result(self.calculate_single_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_single_complete(self, result):
        """
        Plot the single-reflection coverage on the instrument view.

        Parameters
        ----------
        result : tuple or None
            ``(gamma, nu, lamda, d)`` from
            :meth:`calculate_single_process`, giving the detector
            gamma/nu angles (degrees), wavelength, and d-spacing
            (Angstrom) of the matching goniometer settings, or None if
            the calculation was cancelled or invalid.
        """

        if result is not None:
            inst_background = self.model.get_instrument_background()
            gamma, nu, lamda, _ = result
            self.view.plot_instrument(inst_background, gamma, nu, lamda)

    def calculate_single_process(
        self,
        progress,
        stop_event=None,
        hkl_1=None,
        hkl_2=None,
        wavelength=None,
        equiv=None,
        pg=None,
        instrument=None,
        mode=None,
        limits=None,
        instr_motors=None,
        cal=None,
        gon_cal=None,
        mask=None,
    ):
        """
        Worker task: compute detector coverage for a single reflection.

        Uses ``hkl_1`` unless :attr:`alt_hkl` is set, in which case
        ``hkl_2`` is used.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        hkl_1 : array-like, optional
            Primary Miller index input.
        hkl_2 : array-like, optional
            Alternate Miller index input.
        wavelength : 2-tuple of float, optional
            Minimum and maximum wavelength (Angstrom).
        equiv : bool, optional
            Whether to include symmetry-equivalent HKLs in the search.
        pg : str, optional
            Point group symbol used to generate symmetry equivalents.
        instrument : str, optional
            Instrument identifier.
        mode : str, optional
            Goniometer mode name.
        limits : sequence of (float, float), optional
            Per-axis (min, max) goniometer angle limits, in degrees.
        instr_motors : list, optional
            Auxiliary motor ``(name, value)`` settings.
        cal : str, optional
            Detector calibration file path.
        gon_cal : str, optional
            Goniometer calibration file path.
        mask : str, optional
            Detector mask file path.

        Returns
        -------
        gamma, nu, lamda, d : ndarray or None
            Detector gamma/nu angles (degrees), wavelength (Angstrom),
            and d-spacing (Angstrom) of matching goniometer settings, or
            None if the HKL is invalid, no UB matrix is loaded, or the
            worker was stopped early.
        """

        if self.stop_processing(stop_event):
            return None

        hkl = hkl_1 if not self.alt_hkl else hkl_2

        axes, polarities = self.model.get_axes_polarities(instrument, mode)

        if hkl is not None and self.model.has_UB():
            progress("Initializing instrument", 5)

            if self.stop_processing(stop_event):
                return None

            self.model.initialize_instrument(
                instrument, instr_motors, cal, gon_cal, mask
            )

            progress("Instrument initialized! ", 10)

            progress("Calculating peak coverage", 15)

            if self.stop_processing(stop_event):
                return None

            gamma, nu, lamda, d = self.model.individual_peak(
                hkl,
                wavelength,
                axes,
                polarities,
                limits,
                equiv,
                pg,
            )

            progress("Peak calculated!", 0)

            return gamma, nu, lamda, d

        else:
            if hkl is None:
                progress("Invalid HKL input.", 0)
            elif not self.model.has_UB():
                progress("No UB matrix loaded.", 0)
            else:
                progress("Invalid parameters for single peak calculation.", 0)

    def calculate_double(self):
        """
        Launch a background worker to calculate simultaneous coverage of
        two reflections.

        Slot for the "Calculate Double" button. Gathers the two input
        HKLs, wavelength, symmetry, and instrument settings from the
        view and runs :meth:`calculate_double_process` in a worker
        thread, plotting the result via
        :meth:`calculate_double_complete` when finished.
        """

        hkl_1, hkl_2 = self.view.get_input_hkls()
        wavelength = self.view.get_wavelength()
        equiv = self.view.use_equivalents()
        pg = self.view.get_point_group()
        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        limits = self.view.get_goniometer_limits()
        instr_motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()

        worker = self.view.worker(
            functools.partial(
                self.calculate_double_process,
                hkl_1=hkl_1,
                hkl_2=hkl_2,
                wavelength=wavelength,
                equiv=equiv,
                pg=pg,
                instrument=instrument,
                mode=mode,
                limits=limits,
                instr_motors=instr_motors,
                cal=cal,
                gon_cal=gon_cal,
                mask=mask,
            )
        )
        worker.connect_result(self.calculate_double_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_double_complete(self, result):
        """
        Plot the simultaneous two-reflection coverage on the instrument
        view.

        Parameters
        ----------
        result : tuple or None
            ``(gamma_1, nu_1, lamda_1, d_1, gamma_2, nu_2, lamda_2,
            d_2)`` from :meth:`calculate_double_process`, giving the
            detector gamma/nu angles (degrees), wavelength, and
            d-spacing (Angstrom) for the first and second reflection at
            each matching goniometer setting, or None if the
            calculation was cancelled or invalid.
        """

        if result is not None:
            inst_background = self.model.get_instrument_background()
            gamma_1, nu_1, lamda_1, _, gamma_2, nu_2, lamda_2, _ = result
            self.view.plot_instrument_alternate(
                inst_background, gamma_1, nu_1, lamda_1, gamma_2, nu_2, lamda_2
            )

    def calculate_double_process(
        self,
        progress,
        stop_event=None,
        hkl_1=None,
        hkl_2=None,
        wavelength=None,
        equiv=None,
        pg=None,
        instrument=None,
        mode=None,
        limits=None,
        instr_motors=None,
        cal=None,
        gon_cal=None,
        mask=None,
    ):
        """
        Worker task: compute detector coverage for two simultaneous
        reflections.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        hkl_1 : array-like, optional
            Miller index of the first reflection.
        hkl_2 : array-like, optional
            Miller index of the second reflection.
        wavelength : 2-tuple of float, optional
            Minimum and maximum wavelength (Angstrom).
        equiv : bool, optional
            Whether to include symmetry-equivalent HKLs in the search.
        pg : str, optional
            Point group symbol used to generate symmetry equivalents.
        instrument : str, optional
            Instrument identifier.
        mode : str, optional
            Goniometer mode name.
        limits : sequence of (float, float), optional
            Per-axis (min, max) goniometer angle limits, in degrees.
        instr_motors : list, optional
            Auxiliary motor ``(name, value)`` settings.
        cal : str, optional
            Detector calibration file path.
        gon_cal : str, optional
            Goniometer calibration file path.
        mask : str, optional
            Detector mask file path.

        Returns
        -------
        gamma_1, nu_1, lamda_1, d_1 : ndarray
            Detector gamma/nu angles (degrees), wavelength (Angstrom),
            and d-spacing (Angstrom) for the first reflection.
        gamma_2, nu_2, lamda_2, d_2 : ndarray
            Same quantities for the second reflection, at goniometer
            settings where both reflections are simultaneously visible.
            The full 8-tuple is None instead if either HKL is missing,
            no UB matrix is loaded, or the worker was stopped early.
        """

        if self.stop_processing(stop_event):
            return None

        axes, polarities = self.model.get_axes_polarities(instrument, mode)

        if hkl_1 is not None and hkl_2 is not None and self.model.has_UB():
            progress("Initializing instrument", 5)

            if self.stop_processing(stop_event):
                return None

            self.model.initialize_instrument(
                instrument, instr_motors, cal, gon_cal, mask
            )

            progress("Instrument initialized! ", 10)

            progress("Calculating peaks coverage", 15)

            if self.stop_processing(stop_event):
                return None

            peak_1, peak_2 = self.model.simultaneous_peaks(
                hkl_1, hkl_2, wavelength, axes, polarities, limits, equiv, pg
            )

            gamma_1, nu_1, lamda_1, d_1 = peak_1
            gamma_2, nu_2, lamda_2, d_2 = peak_2

            progress("Peaks calculated!", 0)

            return gamma_1, nu_1, lamda_1, d_1, gamma_2, nu_2, lamda_2, d_2

        else:
            if hkl_1 is None:
                progress("Invalid first hkl input.", 0)
            elif hkl_2 is None:
                progress("Invalid second hkl input.", 0)
            elif not self.model.has_UB():
                progress("No UB matrix loaded.", 0)
            else:
                progress("Invalid parameters for double peak calculation.", 0)

    def update_peaks(self, visualize=True):
        """
        Refresh the peaks table and Laue plot for the selected orientation.

        Slot for the orientations combo box's ``activated`` signal, and
        called directly (with ``visualize=False``) after orientations
        are added, deleted, reordered, or reconfigured.

        Parameters
        ----------
        visualize : bool, optional
            If True (default), also trigger a full coverage
            re-visualization via :meth:`visualize`.
        """

        row = self.view.get_peak_list()
        if row is not None:
            peak_list = self.model.generate_table(row)
            self.view.update_peaks_table(peak_list)
            result = self.model.get_laue_info()
            if result is not None:
                self.view.plot_laue(self.model.gamma, self.model.nu, *result)
            if visualize:
                self.visualize()

    def lookup_angle(self, gamma, nu):
        """
        Look up and display the goniometer setting nearest a clicked
        instrument-view detector position.

        Slot for the view's ``roi_ready`` signal, emitted with the
        (gamma, nu) coordinates of a mouse click on the instrument
        coverage plot.

        Parameters
        ----------
        gamma : float
            Detector gamma angle (degrees) of the clicked position.
        nu : float
            Detector nu angle (degrees) of the clicked position.
        """

        vals = self.model.get_angles(gamma, nu)
        if vals is not None:
            (
                angles,
                gamma,
                nu,
                lamda,
                d,
                gamma_alt,
                nu_alt,
                lamda_alt,
                d_alt,
            ) = vals
            self.view.set_comment(self.model.comment)
            self.view.set_angles(angles)
            self.view.set_horizontal(gamma)
            self.view.set_vertical(nu)
            self.view.set_intersect(lamda)
            self.view.set_horizontal_alternate(gamma_alt)
            self.view.set_vertical_alternate(nu_alt)
            self.view.set_intersect_alternate(lamda_alt)
            self.view.set_d(d)
            self.view.set_d_alternate(d_alt)
            self.view.update_inst()

    def calculate_harmonics(self):
        """
        Plot harmonic reflections overlapping the selected peak(s).

        Slot for the view's ``harm_ready`` signal, emitted when the
        instrument plot is updated. Computes and plots harmonics for
        the primary reflection and, if present, the alternate
        (simultaneous) reflection.
        """

        band = self.view.get_wavelength()
        wavelength = self.view.get_intersect()
        hkl = self.model.hkl
        validate = [band, wavelength, hkl]
        if all(elem is not None for elem in validate):
            harmonics = self.model.calculate_harmonics(hkl, wavelength, band)
            self.view.plot_harmonics(*harmonics)
        wavelength_alt = self.view.get_intersect_alternate()
        hkl_alt = self.model.hkl_alt
        validate = [band, wavelength_alt, hkl_alt]
        if all(elem is not None for elem in validate):
            harmonics = self.model.calculate_harmonics(
                hkl_alt, wavelength_alt, band
            )
            self.view.plot_harmonics_alternate(*harmonics)

    def select_peak(self, gamma, nu):
        """
        Select and highlight the peak nearest a clicked Laue-plot position.

        Slot for the view's ``sel_ready`` signal, emitted with the
        (gamma, nu) coordinates of a mouse click on the Laue plot.

        Parameters
        ----------
        gamma : float
            Detector gamma angle (degrees) of the clicked position.
        nu : float
            Detector nu angle (degrees) of the clicked position.
        """

        vals = self.model.get_peak_selection(gamma, nu)
        if vals is not None:
            gamma, nu, lamdas, hkl, wl, row = vals
            self.view.update_laue(gamma, nu, lamdas, hkl, wl)
            self.view.highlight_peak(row)

    def highlight_peak(self):
        """
        Update the Laue plot for the peak selected in the peaks table.

        Slot for the peaks table's ``itemSelectionChanged`` signal.
        """

        row = self.view.get_peak()

        vals = self.model.get_peak_index(row)
        if vals is not None:
            gamma, nu, lamdas, hkl, wl, row = vals
            self.view.update_laue(gamma, nu, lamdas, hkl, wl)

    def move_orientation_up(self):
        """
        Move the selected orientation up one row in the plan table.

        Slot for the "Move Up" button. Swaps the selected orientation
        with the one above it, in both the model and the view, then
        refreshes the peaks table without a full re-visualization.
        """

        row = self.view.get_selected_angle()
        no = self.view.get_number_of_orientations()
        if row is None:
            return
        if row > 0:
            self.model.swap_angles([row, (row - 1) % no])
            self.view.swap_angles([row, (row - 1) % no])
            self.update_peaks(True)
        else:
            self.view.swap_angles([row, row])

    def move_orientation_down(self):
        """
        Move the selected orientation down one row in the plan table.

        Slot for the "Move Down" button. Swaps the selected orientation
        with the one below it, in both the model and the view, then
        refreshes the peaks table without a full re-visualization.
        """

        row = self.view.get_selected_angle()
        no = self.view.get_number_of_orientations()
        if row is None:
            return
        if row < no - 1:
            self.model.swap_angles([row, (row + 1) % no])
            self.view.swap_angles([row, (row + 1) % no])
            self.update_peaks(True)
        else:
            self.view.swap_angles([row, row])

    def delete_angles(self):
        """
        Launch a background worker to delete the selected orientations.

        Slot for the "Delete" button. Gathers the rows selected for
        deletion from the view and runs :meth:`delete_angles_process`
        in a worker thread, updating the view via
        :meth:`delete_angles_complete` when finished.
        """

        rows = self.view.get_angles_to_delete()

        worker = self.view.worker(
            functools.partial(self.delete_angles_process, rows=rows)
        )
        worker.connect_result(self.delete_angles_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def delete_angles_complete(self, rows):
        """
        Remove the deleted orientation rows from the plan table.

        Parameters
        ----------
        rows : list of int or None
            Row indices that were deleted, as returned by
            :meth:`delete_angles_process`, or None if no rows were
            selected.
        """

        if rows is not None:
            self.view.delete_angles(rows)
            self.update_peaks(False)

    def delete_angles_process(self, progress, stop_event=None, rows=None):
        """
        Worker task: delete orientation rows from the model.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        rows : list of int, optional
            Row indices to delete.

        Returns
        -------
        rows : list of int or None
            The same ``rows`` passed in, if deletion was performed, or
            None if no rows were selected.
        """

        if rows is not None:
            self.model.delete_angles(rows)

            progress("Angles deleted!", 0)

            return rows

        else:
            progress("No rows selected for deletion.", 0)

    def add_orientation(self):
        """
        Launch a background worker to add a manually entered orientation.

        Slot for the "Add" button. Gathers the manually entered angles
        and current instrument settings from the view and runs
        :meth:`add_orientation_process` in a worker thread, updating the
        plan table via :meth:`add_orientation_complete` when finished.
        """

        angles = self.view.get_angles()
        free_angles = self.view.get_free_angles()
        all_angles = self.view.get_all_angles()
        wavelength = self.view.get_wavelength()
        d_min = self.view.get_d_min()
        rows = self.view.get_number_of_orientations()

        worker = self.view.worker(
            functools.partial(
                self.add_orientation_process,
                angles=angles,
                free_angles=free_angles,
                all_angles=all_angles,
                wavelength=wavelength,
                d_min=d_min,
                rows=rows,
            )
        )
        worker.connect_result(self.add_orientation_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def add_orientation_complete(self, result):
        """
        Add the newly calculated orientation as a row in the plan table.

        Parameters
        ----------
        result : tuple or None
            ``(angles, all_angles, free_angles)`` from
            :meth:`add_orientation_process`, giving the full per-axis
            angle setting, the names of all goniometer axes, and the
            names of the free (variable) axes, or None if no
            orientation was added.
        """

        if result is None:
            return
        angles, all_angles, free_angles = result

        comment = self.model.comment
        update_angles = []
        for angle, angle_name in zip(angles, all_angles):
            if angle_name in free_angles:
                update_angles.append(angle)

        title = self.view.get_title()
        self.view.add_orientations(title, comment, [update_angles])
        self.update_peaks(False)

    def add_orientation_process(
        self,
        progress,
        stop_event=None,
        angles=None,
        free_angles=None,
        all_angles=None,
        wavelength=None,
        d_min=None,
        rows=None,
    ):
        """
        Worker task: predict reflections for a manually entered
        orientation.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        angles : list of float, optional
            Full per-axis goniometer angle setting.
        free_angles : list of str, optional
            Names of the goniometer axes that are free (variable).
        all_angles : list of str, optional
            Names of all goniometer axes.
        wavelength : 2-tuple of float, optional
            Minimum and maximum wavelength (Angstrom).
        d_min : float, optional
            Minimum d-spacing (Angstrom).
        rows : int, optional
            Row index at which to add the new orientation's predicted
            peaks workspace.

        Returns
        -------
        angles, all_angles, free_angles : tuple or None
            The same ``angles``, ``all_angles``, and ``free_angles``
            passed in, if reflections were predicted, or None if
            ``angles`` was empty.
        """

        if self.stop_processing(stop_event):
            return None

        if len(angles) > 0:
            progress("Calculating reflections", 5)

            if self.stop_processing(stop_event):
                return None

            self.model.add_orientation(angles, wavelength, d_min, rows)

            progress("Reflections calculated!", 0)

            return angles, all_angles, free_angles

        else:
            progress("No angles provided for orientation.", 0)

    def mesh_scan(self):
        """
        Launch a background worker to add orientations from a mesh scan.

        Slot for the "Mesh Scan" button. Gathers the mesh axis limits
        and step counts along with the current instrument settings from
        the view and runs :meth:`mesh_scan_process` in a worker thread,
        updating the plan table via :meth:`mesh_scan_complete` when
        finished.
        """

        mesh_angles = self.view.get_mesh_angles()
        free_angles = self.view.get_free_angles()
        all_angles = self.view.get_all_angles()
        wavelength = self.view.get_wavelength()
        d_min = self.view.get_d_min()
        rows = self.view.get_number_of_orientations()
        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        instr_motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()

        worker = self.view.worker(
            functools.partial(
                self.mesh_scan_process,
                mesh_angles=mesh_angles,
                free_angles=free_angles,
                all_angles=all_angles,
                wavelength=wavelength,
                d_min=d_min,
                rows=rows,
                instrument=instrument,
                mode=mode,
                instr_motors=instr_motors,
                cal=cal,
                gon_cal=gon_cal,
                mask=mask,
            )
        )
        worker.connect_result(self.mesh_scan_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def mesh_scan_complete(self, result):
        """
        Add the mesh-scan orientations as rows in the plan table.

        Parameters
        ----------
        result : list of ndarray or None
            Per-orientation free-axis angle values from
            :meth:`mesh_scan_process`, or None if the mesh scan was
            cancelled or invalid.
        """

        title = self.view.get_title()
        if result is not None:
            self.view.add_orientations(title, "Mesh Scan", result)
            self.update_peaks(False)

    def mesh_scan_process(
        self,
        progress,
        stop_event=None,
        mesh_angles=None,
        free_angles=None,
        all_angles=None,
        wavelength=None,
        d_min=None,
        rows=None,
        instrument=None,
        mode=None,
        instr_motors=None,
        cal=None,
        gon_cal=None,
        mask=None,
    ):
        """
        Worker task: predict reflections for each orientation in a mesh
        of goniometer settings.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        mesh_angles : tuple, optional
            ``(limits, ns)`` -- per-axis [min, max] angle limits and
            number of steps to sample for each goniometer axis.
        free_angles : list of str, optional
            Names of the goniometer axes that are free (variable).
        all_angles : list of str, optional
            Names of all goniometer axes.
        wavelength : 2-tuple of float, optional
            Minimum and maximum wavelength (Angstrom).
        d_min : float, optional
            Minimum d-spacing (Angstrom).
        rows : int, optional
            Row index at which to start adding the new orientations'
            predicted peaks workspaces.
        instrument : str, optional
            Instrument identifier.
        mode : str, optional
            Goniometer mode name.
        instr_motors : list, optional
            Auxiliary motor ``(name, value)`` settings.
        cal : str, optional
            Detector calibration file path.
        gon_cal : str, optional
            Goniometer calibration file path.
        mask : str, optional
            Detector mask file path.

        Returns
        -------
        angles : list of ndarray or None
            Per-orientation free-axis angle values, one entry per mesh
            grid point, or None if ``mesh_angles`` was not provided or
            the worker was stopped early.
        """

        if self.stop_processing(stop_event):
            return None

        axes, polarities = self.model.get_axes_polarities(instrument, mode)
        self.model.generate_axes(axes, polarities)

        progress("Initializing instrument", 5)

        if self.stop_processing(stop_event):
            return None

        self.model.initialize_instrument(
            instrument, instr_motors, cal, gon_cal, mask
        )

        if mesh_angles is not None:
            progress("Calculating reflections", 5)

            if self.stop_processing(stop_event):
                return None

            angles = self.model.add_mesh(
                mesh_angles, wavelength, d_min, rows, free_angles, all_angles
            )

            progress("Reflections calculated!", 0)

            return angles

        else:
            progress("No mesh angles provided for mesh scan.", 0)

    def calculate_plane(self):
        """
        Launch a background worker to preview coverage of a scattering
        plane.

        Slot for the "Calculate Plane" button. No-op if a previous
        HKL-conversion/plane calculation is still running (guarded by
        :attr:`convert_idle`). Gathers the plane's HKL vectors,
        projection, slice, and instrument settings from the view and
        runs :meth:`calculate_plane_process` in a worker thread,
        updating the slice view via :meth:`convert_to_hkl_complete`
        when finished.
        """

        if not self.convert_idle:
            return

        self.convert_idle = False

        hkl_1 = self.view.get_plane_hkl_1()
        hkl_2 = self.view.get_plane_hkl_2()
        max_deg = self.view.get_plane_max_angle()
        n_steps = self.view.get_plane_n_steps()
        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        limits = self.view.get_goniometer_limits()
        proj = self.view.get_projection_matrix()
        value = self.view.get_slice_value()
        thickness = self.view.get_slice_thickness()
        d_min = self.view.get_d_min()
        symm = self.view.use_symmetry_mesh()
        point_group = self.view.get_point_group()
        wavelength = self.view.get_wavelength()
        norm = self.get_normal()
        instr_motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()

        worker = self.view.worker(
            functools.partial(
                self.calculate_plane_process,
                hkl_1=hkl_1,
                hkl_2=hkl_2,
                max_deg=max_deg,
                n_steps=n_steps,
                instrument=instrument,
                mode=mode,
                limits=limits,
                proj=proj,
                value=value,
                thickness=thickness,
                d_min=d_min,
                symm=symm,
                point_group=point_group,
                wavelength=wavelength,
                norm=norm,
                instr_motors=instr_motors,
                cal=cal,
                gon_cal=gon_cal,
                mask=mask,
            )
        )
        worker.connect_result(self.convert_to_hkl_complete)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_plane_process(
        self,
        progress,
        stop_event=None,
        hkl_1=None,
        hkl_2=None,
        max_deg=None,
        n_steps=None,
        instrument=None,
        mode=None,
        limits=None,
        proj=None,
        value=None,
        thickness=None,
        d_min=None,
        symm=None,
        point_group=None,
        wavelength=None,
        norm=None,
        instr_motors=None,
        cal=None,
        gon_cal=None,
        mask=None,
    ):
        """
        Worker task: preview the coverage slice for a scattering plane.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        hkl_1, hkl_2 : array-like, optional
            Miller-index vectors defining the scattering plane.
        max_deg : float, optional
            Maximum rotation angle (degrees) to scan about the plane
            normal.
        n_steps : int, optional
            Number of steps to scan over ``max_deg``.
        instrument : str, optional
            Instrument identifier.
        mode : str, optional
            Goniometer mode name.
        limits : sequence of (float, float), optional
            Per-axis (min, max) goniometer angle limits, in degrees.
        proj : array-like, optional
            Flattened 3x3 HKL projection matrix.
        value : float, optional
            Slice position along the slice normal.
        thickness : float, optional
            Slice half-thickness.
        d_min : float, optional
            Minimum d-spacing (Angstrom).
        symm : bool, optional
            Whether to apply point-group symmetry when computing
            coverage.
        point_group : str, optional
            Point group symbol used when ``symm`` is True.
        wavelength : 2-tuple of float, optional
            Minimum and maximum wavelength (Angstrom).
        norm : list of int, optional
            One-hot vector selecting the slice normal axis.
        instr_motors : list, optional
            Auxiliary motor ``(name, value)`` settings.
        cal : str, optional
            Detector calibration file path.
        gon_cal : str, optional
            Goniometer calibration file path.
        mask : str, optional
            Detector mask file path.

        Returns
        -------
        result : dict or None
            Slice dictionary from ``model.calculate_rotations`` (with
            keys such as ``"x"``, ``"y"``, ``"signal"``, ``"transform"``)
            for the requested plane, or None if the HKL vectors,
            projection, or other parameters are invalid, no plane
            orientations were found, or the worker was stopped early.
        """

        if self.stop_processing(stop_event):
            return None

        if hkl_1 is None or hkl_2 is None:
            progress("Invalid HKL vectors for plane preview.", 0)
            return None

        axes, polarities = self.model.get_axes_polarities(instrument, mode)
        self.model.generate_axes(axes, polarities)

        validate = [proj, value, thickness, d_min]
        if not all(elem is not None for elem in validate):
            progress("Invalid parameters.", 0)
            return None

        U, V, W, invalid = self.model.validate_projection(proj)
        if invalid:
            progress("Invalid projections.", 0)
            return None

        progress("Computing plane orientations...", 5)

        if self.stop_processing(stop_event):
            return None

        angles = self.model.compute_plane_angles(
            hkl_1, hkl_2, axes, polarities, limits, max_deg, n_steps
        )

        if not angles:
            progress("No plane orientations found.", 0)
            return None

        progress("Initializing instrument...", 5)

        if self.stop_processing(stop_event):
            return None

        self.model.initialize_instrument(
            instrument, instr_motors, cal, gon_cal, mask
        )
        self.model.calculate_footprint(wavelength, d_min)

        progress("Calculating plane coverage...", 50)

        if self.stop_processing(stop_event):
            return None

        result = self.model.calculate_rotations(
            angles,
            U,
            V,
            W,
            norm,
            value,
            thickness,
            False,
            point_group,
            symm,
        )

        progress("Plane coverage calculated!", 0)

        return result

    def plane_scan(self):
        """
        Launch a background worker to add orientations spanning a plane.

        Slot for the "Add Plane" button. Gathers the plane's HKL
        vectors, scan range, and current instrument settings from the
        view and runs :meth:`plane_scan_process` in a worker thread,
        updating the plan table via :meth:`plane_scan_complete` when
        finished.
        """

        hkl_1 = self.view.get_plane_hkl_1()
        hkl_2 = self.view.get_plane_hkl_2()
        max_deg = self.view.get_plane_max_angle()
        n_steps = self.view.get_plane_n_steps()
        free_angles = self.view.get_free_angles()
        all_angles = self.view.get_all_angles()
        wavelength = self.view.get_wavelength()
        d_min = self.view.get_d_min()
        rows = self.view.get_number_of_orientations()
        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        limits = self.view.get_goniometer_limits()
        instr_motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()

        worker = self.view.worker(
            functools.partial(
                self.plane_scan_process,
                hkl_1=hkl_1,
                hkl_2=hkl_2,
                max_deg=max_deg,
                n_steps=n_steps,
                free_angles=free_angles,
                all_angles=all_angles,
                wavelength=wavelength,
                d_min=d_min,
                rows=rows,
                instrument=instrument,
                mode=mode,
                limits=limits,
                instr_motors=instr_motors,
                cal=cal,
                gon_cal=gon_cal,
                mask=mask,
            )
        )
        worker.connect_result(self.plane_scan_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def plane_scan_complete(self, result):
        """
        Add the plane-scan orientations as rows in the plan table.

        Parameters
        ----------
        result : list of ndarray or None
            Per-orientation free-axis angle values from
            :meth:`plane_scan_process`, or None if the plane scan was
            cancelled or invalid.
        """

        title = self.view.get_title()
        if result is not None:
            self.view.add_orientations(title, "Plane Scan", result)
            self.update_peaks(False)

    def plane_scan_process(
        self,
        progress,
        stop_event=None,
        hkl_1=None,
        hkl_2=None,
        max_deg=None,
        n_steps=None,
        free_angles=None,
        all_angles=None,
        wavelength=None,
        d_min=None,
        rows=None,
        instrument=None,
        mode=None,
        limits=None,
        instr_motors=None,
        cal=None,
        gon_cal=None,
        mask=None,
    ):
        """
        Worker task: add orientations spanning a scattering plane.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        hkl_1, hkl_2 : array-like, optional
            Miller-index vectors defining the scattering plane.
        max_deg : float, optional
            Maximum rotation angle (degrees) to scan about the plane
            normal.
        n_steps : int, optional
            Number of steps to scan over ``max_deg``.
        free_angles : list of str, optional
            Names of the goniometer axes that are free (variable).
        all_angles : list of str, optional
            Names of all goniometer axes.
        wavelength : 2-tuple of float, optional
            Minimum and maximum wavelength (Angstrom).
        d_min : float, optional
            Minimum d-spacing (Angstrom).
        rows : int, optional
            Row-number offset at which to start adding the new
            orientations' predicted peaks.
        instrument : str, optional
            Instrument identifier.
        mode : str, optional
            Goniometer mode name.
        limits : sequence of (float, float), optional
            Per-axis (min, max) goniometer angle limits, in degrees.
        instr_motors : list, optional
            Auxiliary motor ``(name, value)`` settings.
        cal : str, optional
            Detector calibration file path.
        gon_cal : str, optional
            Goniometer calibration file path.
        mask : str, optional
            Detector mask file path.

        Returns
        -------
        angles : list of ndarray or None
            Per-orientation free-axis angle values, one entry per
            selected plane orientation, or None if the HKL vectors are
            invalid, no plane orientations were found, or the worker
            was stopped early.
        """

        if self.stop_processing(stop_event):
            return None

        if hkl_1 is None or hkl_2 is None:
            progress("Invalid HKL vectors for plane scan.", 0)
            return None

        axes, polarities = self.model.get_axes_polarities(instrument, mode)
        self.model.generate_axes(axes, polarities)

        progress("Initializing instrument", 5)

        if self.stop_processing(stop_event):
            return None

        self.model.initialize_instrument(
            instrument, instr_motors, cal, gon_cal, mask
        )

        progress("Calculating plane orientations", 5)

        if self.stop_processing(stop_event):
            return None

        angles = self.model.add_plane(
            hkl_1,
            hkl_2,
            wavelength,
            d_min,
            rows,
            free_angles,
            all_angles,
            axes,
            polarities,
            limits,
            max_deg=max_deg,
            n_steps=n_steps,
        )

        progress("Plane orientations calculated!", 0)

        return angles

    def reslice(self):
        """Re-slice the existing coverage without rebuilding the footprint."""
        self.slice_only = True
        self.convert_to_hkl()

    def convert_mesh_to_hkl(self):
        """
        Recompute coverage from the mesh-scan angles and slice to HKL.

        Slot for the "Convert Mesh" button. Sets the presenter to use
        the mesh-angle grid (rather than the plan's discrete
        orientations) as the source of goniometer settings, then
        delegates to :meth:`convert_to_hkl`.
        """

        self.slice_only = False
        self.mesh = True
        self.convert_to_hkl()

    def convert_plan_to_hkl(self):
        """
        Recompute coverage from the plan's orientations and slice to HKL.

        Slot for the "Convert Plan" button. Sets the presenter to use
        the discrete orientations in the plan table (rather than the
        mesh-angle grid) as the source of goniometer settings, then
        delegates to :meth:`convert_to_hkl`.
        """

        self.slice_only = False
        self.mesh = False
        self.convert_to_hkl()

    def convert_to_hkl(self):
        """
        Launch a background worker to compute/re-slice HKL coverage.

        No-op if a previous HKL-conversion is still running (guarded by
        :attr:`convert_idle`). Gathers the projection, slice, and
        instrument settings from the view, along with either the mesh
        angles or the plan angles depending on :attr:`mesh`, and runs
        :meth:`convert_to_hkl_process` in a worker thread, updating the
        slice view via :meth:`convert_to_hkl_complete` when finished.
        """

        if self.convert_idle:
            instrument = self.view.get_instrument()
            mode = self.view.get_mode()
            proj = self.view.get_projection_matrix()
            value = self.view.get_slice_value()
            thickness = self.view.get_slice_thickness()
            d_min = self.view.get_d_min()
            symm = self.view.use_symmetry_mesh()
            point_group = self.view.get_point_group()
            norm = self.get_normal()
            if self.mesh:
                angles = self.view.get_mesh_angles()
            else:
                angles = self.view.get_plan_angles()
            wavelength = self.view.get_wavelength()
            instr_motors = self.view.get_motors()
            cal = self.view.get_detector_calibration()
            gon_cal = self.view.get_goniometer_calibration()
            mask = self.view.get_mask()

            worker = self.view.worker(
                functools.partial(
                    self.convert_to_hkl_process,
                    instrument=instrument,
                    mode=mode,
                    proj=proj,
                    value=value,
                    thickness=thickness,
                    d_min=d_min,
                    symm=symm,
                    point_group=point_group,
                    norm=norm,
                    angles=angles,
                    wavelength=wavelength,
                    instr_motors=instr_motors,
                    cal=cal,
                    gon_cal=gon_cal,
                    mask=mask,
                )
            )
            worker.connect_result(self.convert_to_hkl_complete)
            worker.connect_progress(self.update_processing)

            self.convert_idle = False
            self.view.start_worker_pool(worker)

    def convert_to_hkl_complete(self, result):
        """
        Update the slice view with the newly computed HKL coverage.

        Parameters
        ----------
        result : dict or None
            Slice dictionary from :meth:`convert_to_hkl_process`, or
            None if the conversion was invalid or produced no coverage.
        """

        if result is not None:
            self.view.update_slice(result)
        self.convert_idle = True

    def convert_to_hkl_process(
        self,
        progress,
        stop_event=None,
        instrument=None,
        mode=None,
        proj=None,
        value=None,
        thickness=None,
        d_min=None,
        symm=None,
        point_group=None,
        norm=None,
        angles=None,
        wavelength=None,
        instr_motors=None,
        cal=None,
        gon_cal=None,
        mask=None,
    ):
        """
        Worker task: compute or re-slice the HKL coverage for a plan.

        If :attr:`slice_only` is set, re-slices the most recently
        computed coverage without recomputing the footprint (see
        :meth:`reslice`). Otherwise, initializes the instrument, builds
        the footprint, and computes the coverage for ``angles`` (either
        a mesh grid or a list of plan orientations, depending on
        :attr:`mesh`), then slices it onto the requested HKL plane.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        instrument : str, optional
            Instrument identifier.
        mode : str, optional
            Goniometer mode name.
        proj : array-like, optional
            Flattened 3x3 HKL projection matrix.
        value : float, optional
            Slice position along the slice normal.
        thickness : float, optional
            Slice half-thickness.
        d_min : float, optional
            Minimum d-spacing (Angstrom).
        symm : bool, optional
            Whether to apply point-group symmetry when computing
            coverage.
        point_group : str, optional
            Point group symbol used when ``symm`` is True.
        norm : list of int, optional
            One-hot vector selecting the slice normal axis.
        angles : array-like, optional
            Either a list of goniometer angle tuples (plan
            orientations) or ``(limits, ns)`` describing a mesh grid,
            depending on :attr:`mesh`.
        wavelength : 2-tuple of float, optional
            Minimum and maximum wavelength (Angstrom).
        instr_motors : list, optional
            Auxiliary motor ``(name, value)`` settings.
        cal : str, optional
            Detector calibration file path.
        gon_cal : str, optional
            Goniometer calibration file path.
        mask : str, optional
            Detector mask file path.

        Returns
        -------
        result : dict or None
            Slice dictionary from ``model.calculate_rotations`` or
            ``model.reslice_last`` (with keys such as ``"x"``, ``"y"``,
            ``"signal"``, ``"transform"``), or None if the parameters
            or projection are invalid, no angles were provided, there
            is no coverage to reslice, or the worker was stopped early.
        """

        if self.stop_processing(stop_event):
            return None

        axes, polarities = self.model.get_axes_polarities(instrument, mode)
        self.model.generate_axes(axes, polarities)

        validate = [proj, value, thickness, d_min]

        if all(elem is not None for elem in validate):
            U, V, W, invalid = self.model.validate_projection(proj)

            if not invalid:

                if self.slice_only:
                    self.slice_only = False
                    progress("Recalculating slice...", 5)
                    result = self.model.reslice_last(
                        U, V, W, norm, value, thickness
                    )
                    if result is None:
                        progress("No coverage to reslice.", 0)
                    else:
                        progress("Slice updated!", 0)
                    return result

                if len(angles) > 0:

                    progress("Initializing instrument...", 5)

                    if self.stop_processing(stop_event):
                        return None

                    self.model.initialize_instrument(
                        instrument, instr_motors, cal, gon_cal, mask
                    )

                    self.model.calculate_footprint(wavelength, d_min)

                    progress("Calculating footprint...", 50)

                    if self.stop_processing(stop_event):
                        return None

                    result = self.model.calculate_rotations(
                        angles,
                        U,
                        V,
                        W,
                        norm,
                        value,
                        thickness,
                        self.mesh,
                        point_group,
                        symm,
                    )

                    progress("Footprint calculated!", 0)

                    return result

                else:
                    progress("No angles provided for orientation.", 0)

            else:
                progress("Invalid projections.", 0)

        else:
            progress("Invalid parameters.", 0)

    def get_normal(self):
        """
        Get the one-hot slice-normal vector for the selected slice plane.

        Returns
        -------
        norm : list of int
            One-hot vector selecting which of the projection basis
            vectors (U, V, W) is the slice normal: ``[0, 0, 1]`` for
            "Axis 1/2", ``[0, 1, 0]`` for "Axis 1/3", or ``[1, 0, 0]``
            for "Axis 2/3".
        """

        slice_plane = self.view.get_slice()

        if slice_plane == "Axis 1/2":
            norm = [0, 0, 1]
        elif slice_plane == "Axis 1/3":
            norm = [0, 1, 0]
        else:
            norm = [1, 0, 0]

        return norm

    def visualize(self):
        """
        Recompute statistics and coverage, and refresh the plots.

        Slot connected to several view signals (combined-peaks
        checkbox, color scheme, lattice centering, point group,
        visualization-ready) and called after orientations are added,
        deleted, or reconfigured. No-op if a previous visualization is
        still running (guarded by :attr:`draw_idle`). Computes
        completeness/redundancy statistics and, if a UB matrix is
        loaded, the reciprocal-space coverage point cloud, then updates
        the corresponding plots in the view.
        """

        if not self.draw_idle:
            return

        self.draw_idle = False

        point_group = self.view.get_point_group()
        lattice_centering = self.view.get_lattice_centering()
        use = self.view.get_orientations_to_use()
        d_min = self.view.get_d_min()
        draw_all = self.view.draw_all()
        row = self.view.get_peak_list()
        color = self.view.get_color_scheme()

        try:
            self.update_processing()

            self.update_processing("Calculating statistics...", 5)

            stats = self.model.calculate_statistics(
                point_group, lattice_centering, use, d_min
            )

            self.update_processing("Statistics calculated...", 30)

            if stats is not None and self.model.has_UB():
                self.view.plot_statistics(*stats)

                self.update_processing("Calculating coverage...", 50)

                peak_dict = self.model.get_coverage_info(
                    point_group, lattice_centering, draw_all, color, row
                )

                self.update_processing("Coverage calculated...", 80)

                if peak_dict is not None:
                    peak_dict["axis_limit"] = self.view.get_d_min()

                    self.view.add_peaks(peak_dict)

            else:
                self.view.add_peaks(None)

            self.update_complete("Data visualized!")

        finally:
            self.draw_idle = True

    def optimize_coverage(self):
        """
        Launch a background worker to optimize coverage via CrystalPlan.

        Slot for the "Optimize" button. Gathers the symmetry, coverage,
        and instrument settings from the view and runs
        :meth:`optimize_coverage_process` (a genetic-algorithm search
        over orientations) in a worker thread, updating the plan table
        via :meth:`optimize_coverage_complete` when finished.
        """

        point_group = self.view.get_point_group()
        lattice_centering = self.view.get_lattice_centering()
        use = self.view.get_orientations_to_use()
        opt = self.view.get_optimized_settings()
        d_min = self.view.get_d_min()
        wavelength = self.view.get_wavelength()
        n_orient = self.view.get_settings()
        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        limits = self.view.get_goniometer_limits()
        instr_motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()

        worker = self.view.worker(
            functools.partial(
                self.optimize_coverage_process,
                point_group=point_group,
                lattice_centering=lattice_centering,
                use=use,
                opt=opt,
                d_min=d_min,
                wavelength=wavelength,
                n_orient=n_orient,
                instrument=instrument,
                mode=mode,
                limits=limits,
                instr_motors=instr_motors,
                cal=cal,
                gon_cal=gon_cal,
                mask=mask,
            )
        )
        worker.connect_result(self.optimize_coverage_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def optimize_coverage_complete(self, result):
        """
        Add the CrystalPlan-optimized orientations as rows in the plan
        table.

        Parameters
        ----------
        result : list of ndarray or None
            Per-orientation free-axis angle values from
            :meth:`optimize_coverage_process`, or None if no UB matrix
            was loaded or the optimization was cancelled.
        """

        title = self.view.get_title()
        if result is not None:
            self.view.add_orientations(title, "CrystalPlan", result)
            self.update_peaks(False)

    def optimize_coverage_process(
        self,
        progress,
        stop_event=None,
        point_group=None,
        lattice_centering=None,
        use=None,
        opt=None,
        d_min=None,
        wavelength=None,
        n_orient=None,
        instrument=None,
        mode=None,
        limits=None,
        instr_motors=None,
        cal=None,
        gon_cal=None,
        mask=None,
    ):
        """
        Worker task: optimize orientation coverage with a genetic
        algorithm.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        point_group : str, optional
            Point group symbol.
        lattice_centering : str, optional
            Lattice centering symbol.
        use : list of bool, optional
            Per-row flag indicating whether that plan orientation
            should be included in the optimization.
        opt : list of bool, optional
            Per-row flag indicating whether that plan orientation is
            free to be optimized.
        d_min : float, optional
            Minimum d-spacing (Angstrom).
        wavelength : 2-tuple of float, optional
            Minimum and maximum wavelength (Angstrom).
        n_orient : int, optional
            Number of orientations to optimize.
        instrument : str, optional
            Instrument identifier.
        mode : str, optional
            Goniometer mode name.
        limits : sequence of (float, float), optional
            Per-axis (min, max) goniometer angle limits, in degrees.
        instr_motors : list, optional
            Auxiliary motor ``(name, value)`` settings.
        cal : str, optional
            Detector calibration file path.
        gon_cal : str, optional
            Goniometer calibration file path.
        mask : str, optional
            Detector mask file path.

        Returns
        -------
        values : list or None
            Best per-orientation goniometer settings found by the
            genetic algorithm (see ``CrystalPlan.optimize``), or None
            if no UB matrix is loaded or the worker was stopped early.
        """

        if self.stop_processing(stop_event):
            return None

        n_elite = 2
        n_gener = 10
        n_indiv = 10
        mutation_rate = 0.15

        axes = self.model.get_goniometer_axes(instrument, mode)

        if self.model.has_UB():
            progress("Initializing instrument", 5)

            if self.stop_processing(stop_event):
                return None

            self.model.initialize_instrument(
                instrument, instr_motors, cal, gon_cal, mask
            )

            progress("Instrument initialized! ", 10)

            cp = self.model.crystal_plan(
                use,
                opt,
                axes,
                limits,
                wavelength,
                d_min,
                point_group,
                lattice_centering,
            )

            progress("Optimizing peaks coverage", 15)

            values = cp.optimize(
                n_orient, n_indiv, n_gener, n_elite, mutation_rate
            )

            progress("Peaks coverage optimized!", 0)

            return values

        else:
            progress("No UB matrix loaded for optimization.", 0)

    def update_plan(self):
        """
        Push the current view state into the model's plan and sample.

        Gathers the full plan table, instrument/goniometer settings,
        UB matrix, and crystal symmetry from the view and passes them
        to :meth:`model.create_plan`, :meth:`model.create_sample`,
        :meth:`model.update_sample`, and
        :meth:`model.update_goniometer_motors` so they are recorded on
        the ``plan``/``sample`` workspaces before saving. Called by
        :meth:`save_CSV` and :meth:`save_experiment`.
        """

        instrument = self.view.get_instrument()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()
        mode = self.view.get_mode()
        settings = self.view.get_all_settings()
        comments = self.view.get_all_comments()
        counts = self.view.get_all_countings()
        values = self.view.get_all_values()
        use = self.view.get_orientations_to_use()
        names = self.view.get_free_angles()
        titles = self.view.get_all_titles()
        UB = self.model.get_UB()
        wavelength = self.view.get_wavelength()
        d_min = self.view.get_d_min()
        crysal_system = self.view.get_crystal_system()
        point_group = self.view.get_point_group()
        lattice_centering = self.view.get_lattice_centering()
        motors = self.view.get_motors()
        limits = self.view.get_goniometer_limits()
        pv = self.model.get_scan_log(instrument)
        table = pv, names, titles, settings, comments, counts, values, use
        self.model.create_plan(table)
        self.model.create_sample(instrument, mode, UB, wavelength, d_min)
        self.model.update_sample(crysal_system, point_group, lattice_centering)
        self.model.update_goniometer_motors(limits, motors, cal, gon_cal, mask)

    def save_CSV(self):
        """
        Save the active plan rows to a CSV file and copy it to the
        instrument PC.

        Slot for the "Save CSV" button. Prompts for a destination file,
        updates the model's plan/sample from the current view state,
        writes the active (used) rows to CSV, and copies the file to
        the instrument PC share if applicable (see
        :meth:`model.copy_to_instrument_pc`). No-op if the file dialog
        is cancelled.
        """

        filename = self.view.save_CSV_file_dialog()

        if filename:
            self.update_plan()
            self.model.save_plan(filename)
            self.model.copy_to_instrument_pc(filename)

    def save_experiment(self):
        """
        Save the full experiment plan and sample to a Nexus file.

        Slot for the "Save Experiment" button. Prompts for a
        destination file (defaulting to the current instrument's
        directory), updates the model's plan/sample from the current
        view state, saves them to Nexus, and remembers the file's
        directory as the default for future dialogs. No-op if the file
        dialog is cancelled.
        """

        instrument = self.view.get_instrument()
        path = self.model.get_instrument_directory(instrument)
        filename = self.view.save_experiment_file_dialog(path)

        if filename:
            self.update_plan()
            self.model.save_experiment(filename)
            self.model.set_path(filename)

    def load_experiment(self):
        """
        Load a previously saved experiment plan and sample from Nexus.

        Slot for the "Load Experiment" button. Prompts for a source
        file (defaulting to the current instrument's directory) and, if
        one is selected, restores the instrument, goniometer mode,
        wavelength, d-min, goniometer limits, motor values, detector
        calibration, goniometer calibration, and mask files, crystal
        system, point group, lattice centering, and plan table rows from
        the file, then re-adds the restored
        settings via :meth:`add_settings` and remembers the file's
        directory as the default for future dialogs. No-op if the file
        dialog is cancelled.
        """

        instrument = self.view.get_instrument()
        path = self.model.get_instrument_directory(instrument)
        filename = self.view.load_experiment_file_dialog(path)

        if filename:
            plan, config, symm = self.model.load_experiment(filename)

            titles, settings, comments, counts, values, use = plan
            instrument, mode, wl, d_min, lims, vals, cal, gon_cal, mask = (
                config
            )
            cs, pg, lc = symm

            table = titles, settings, comments, counts, values, use

            self.view.set_instrument(instrument)
            self.switch_instrument()
            self.view.set_mode(mode)
            self.update_goniometer()
            self.update_oriented_lattice()
            self.view.set_transform(self.model.get_transform())
            self.view.set_wavelength(wl)
            self.view.set_d_min(d_min)
            self.view.set_goniometer_limits(lims)
            self.view.set_motors(vals)
            self.view.set_detector_calibration(cal)
            self.view.set_goniometer_calibration(gon_cal)
            self.view.set_mask(mask)
            self.view.set_crystal_system(cs)
            self.switch_crystal()
            self.view.set_point_group(pg)
            self.switch_group()
            self.view.set_lattice_centering(lc)
            self.view.add_settings(*table)
            self.add_settings()
            self.model.set_path(filename)

    def add_settings(self):
        """
        Launch a background worker to recompute peaks for all plan rows.

        Slot for the "Reset" button, and called after loading an
        experiment. Gathers the per-row angle settings and current
        instrument settings from the view and runs
        :meth:`add_settings_process` in a worker thread, updating the
        peaks table via :meth:`add_settings_complete` when finished.
        """

        wavelength = self.view.get_wavelength()
        d_min = self.view.get_d_min()
        rows = self.view.get_number_of_orientations()
        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        limits = self.view.get_goniometer_limits()
        instr_motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()
        angle_settings = [
            self.view.get_angle_setting(row) for row in range(rows)
        ]

        worker = self.view.worker(
            functools.partial(
                self.add_settings_process,
                wavelength=wavelength,
                d_min=d_min,
                rows=rows,
                instrument=instrument,
                mode=mode,
                limits=limits,
                instr_motors=instr_motors,
                cal=cal,
                gon_cal=gon_cal,
                mask=mask,
                angle_settings=angle_settings,
            )
        )
        worker.connect_result(self.add_settings_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def add_settings_complete(self, result):
        """
        Refresh the peaks table after recomputing all plan row settings.

        Parameters
        ----------
        result : int or None
            Number of rows processed, as returned by
            :meth:`add_settings_process`, or None if the worker was
            stopped early.
        """

        if result is not None:
            self.update_peaks(False)

    def add_settings_process(
        self,
        progress,
        stop_event=None,
        wavelength=None,
        d_min=None,
        rows=None,
        instrument=None,
        mode=None,
        limits=None,
        instr_motors=None,
        cal=None,
        gon_cal=None,
        mask=None,
        angle_settings=None,
    ):
        """
        Worker task: recompute peaks for every orientation in the plan.

        Clears the ``combined`` peaks workspace and, for each plan row,
        combines its free-axis angle settings with the fixed-limit
        angles and predicts peaks for the resulting orientation.

        Parameters
        ----------
        progress : callable
            Callback ``progress(status, percent)`` used to report status
            messages and progress percentage back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that the worker should stop early.
        wavelength : 2-tuple of float, optional
            Minimum and maximum wavelength (Angstrom).
        d_min : float, optional
            Minimum d-spacing (Angstrom).
        rows : int, optional
            Number of plan rows to recompute.
        instrument : str, optional
            Instrument identifier.
        mode : str, optional
            Goniometer mode name.
        limits : sequence of (float, float), optional
            Per-axis (min, max) goniometer angle limits, in degrees.
        instr_motors : list, optional
            Auxiliary motor ``(name, value)`` settings.
        cal : str, optional
            Detector calibration file path.
        gon_cal : str, optional
            Goniometer calibration file path.
        mask : str, optional
            Detector mask file path.
        angle_settings : list, optional
            Per-row free-axis angle values, one entry per plan row.

        Returns
        -------
        rows : int or None
            The same ``rows`` passed in, if processing completed, or
            None if the worker was stopped early.
        """

        if self.stop_processing(stop_event):
            return None

        axes, polarities = self.model.get_axes_polarities(instrument, mode)
        self.model.generate_axes(axes, polarities)

        progress("Initializing instrument", 5)

        if self.stop_processing(stop_event):
            return None

        self.model.initialize_instrument(
            instrument, instr_motors, cal, gon_cal, mask
        )
        self.model.clear_combined()

        for row in range(rows):
            progress("Calculating settings", 90 // rows * (row + 1) + 5)

            if self.stop_processing(stop_event):
                return None

            angles = angle_settings[row]

            setting = self.model.get_setting(angles, limits)

            self.model.add_orientation(setting, wavelength, d_min, row)

        progress("Settings calculated!", 0)

        return rows
