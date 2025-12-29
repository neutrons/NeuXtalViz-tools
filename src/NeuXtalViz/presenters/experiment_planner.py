from NeuXtalViz.presenters.base_presenter import NeuXtalVizPresenter


class Experiment(NeuXtalVizPresenter):
    def __init__(self, view, model):
        super(Experiment, self).__init__(view, model)

        self.view.connect_load_UB(self.load_UB)
        self.view.connect_reset(self.add_settings)
        self.view.connect_switch_instrument(self.switch_instrument)
        self.view.connect_update_goniometer(self.update_goniometer)
        self.view.connect_switch_crystal(self.switch_crystal)
        self.view.connect_switch_point_group(self.switch_group)
        self.view.connect_switch_lattice_centering(self.switch_centering)
        self.view.connect_wavelength(self.update_wavelength)
        self.view.connect_optimize(self.optimize_coverage)
        self.view.connect_mesh(self.mesh_scan)
        self.view.connect_calculate_single(self.calculate_single)
        self.view.connect_calculate_double(self.calculate_double)
        self.view.connect_calculate_single_alt(self.calculate_single_alt)
        self.view.connect_add_orientation(self.add_orientation)
        self.view.connect_delete_angles(self.delete_angles)
        self.view.connect_save_CSV(self.save_CSV)
        self.view.connect_save_experiment(self.save_experiment)
        self.view.connect_load_experiment(self.load_experiment)
        self.view.connect_combined(self.visualize)
        self.view.connect_peak_table(self.update_peaks)
        self.view.connect_load_mask(self.load_mask)
        self.view.connect_load_detector(self.load_detector)
        self.view.connect_load_goniometer(self.load_goniometer)
        self.view.connect_convert_mesh_to_hkl(self.convert_mesh_to_hkl)
        self.view.connect_convert_plan_to_hkl(self.convert_plan_to_hkl)

        self.view.connect_slice_combo(self.convert_to_hkl)
        self.view.connect_slice_thickness_line(self.convert_to_hkl)

        self.view.connect_roi_ready(self.lookup_angle)
        self.view.connect_selection_ready(self.select_peak)
        self.view.connect_visualization_ready(self.visualize)
        self.view.connect_harmonic_ready(self.calculate_harmonics)

        self.view.connect_update(self.view.update_counting)
        self.view.connect_highlight_angles(self.view.highlight_angles)
        self.view.connect_peak_row_highlighter(self.highlight_peak)

        self.draw_idle = True

        self.switch_instrument()
        self.switch_crystal()

        self.view.set_default_symmetry()

        self.mesh = True
        self.convert_idle = True

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
        print(filename)

        if filename:
            self.view.set_mask(filename)

    def load_UB(self):
        """
        Load UB matrix from file and update the view and model.

        """
        filename = self.view.load_UB_file_dialog()

        if filename:
            self.model.load_UB(filename)

            self.update_oriented_lattice()

            self.view.set_transform(self.model.get_transform())

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
        cs = self.view.get_crystal_system()

        point_groups = self.model.get_crystal_system_point_groups(cs)

        self.view.set_point_groups(point_groups)

        self.switch_group()

    def switch_group(self):
        pg = self.view.get_point_group()

        centerings = self.model.get_point_group_centering(pg)

        self.view.set_lattice_centerings(centerings)

        self.visualize()

    def switch_centering(self):
        self.visualize()

    def update_goniometer(self):
        instrument = self.view.get_instrument()
        mode = self.view.get_mode()

        goniometers = self.model.get_goniometers(instrument, mode)
        motors = self.model.get_motors(instrument)
        title = self.model.get_scan_log(instrument)

        self.view.update_tables(title, goniometers, motors)

    def update_wavelength(self):
        wl_min, _ = self.view.get_wavelength()
        self.view.update_wavelength(wl_min)

    def create_instrument(self):
        instrument = self.view.get_instrument()
        motors = self.view.get_motors()
        cal = self.view.get_detector_calibration()
        gon = self.view.get_goniometer_calibration()
        mask = self.view.get_mask()

        self.model.initialize_instrument(instrument, motors, cal, gon, mask)

    def calculate_single(self):
        self.alt_hkl = False
        self.calculate_single_hkl()

    def calculate_single_alt(self):
        self.alt_hkl = True
        self.calculate_single_hkl()

    def calculate_single_hkl(self):
        worker = self.view.worker(self.calculate_single_process)
        worker.connect_result(self.calculate_single_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_single_complete(self, result):
        if result is not None:
            self.view.plot_instrument(self.model.gamma, self.model.nu, *result)

    def calculate_single_process(self, progress):
        hkl_1, hkl_2 = self.view.get_input_hkls()
        wavelength = self.view.get_wavelength()

        hkl = hkl_1 if not self.alt_hkl else hkl_2

        equiv = self.view.use_equivalents()
        pg = self.view.get_point_group()

        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        axes, polarities = self.model.get_axes_polarities(instrument, mode)

        limits = self.view.get_goniometer_limits()

        if hkl is not None and self.model.has_UB():
            progress("Initializing instrument", 5)

            self.create_instrument()

            progress("Instrument initialized! ", 10)

            progress("Calculating peak coverage", 15)

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
        worker = self.view.worker(self.calculate_double_process)
        worker.connect_result(self.calculate_double_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_double_complete(self, result):
        if result is not None:
            self.view.plot_instrument_alternate(
                self.model.gamma, self.model.nu, *result
            )

    def calculate_double_process(self, progress):
        hkl_1, hkl_2 = self.view.get_input_hkls()
        wavelength = self.view.get_wavelength()

        equiv = self.view.use_equivalents()
        pg = self.view.get_point_group()

        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        axes, polarities = self.model.get_axes_polarities(instrument, mode)

        limits = self.view.get_goniometer_limits()

        if hkl_1 is not None and hkl_2 is not None and self.model.has_UB():
            progress("Initializing instrument", 5)

            self.create_instrument()

            progress("Instrument initialized! ", 10)

            progress("Calculating peaks coverage", 15)

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

    def update_peaks(self):
        row = self.view.get_peak_list()
        if row is not None:
            peak_list = self.model.generate_table(row)
            self.view.update_peaks_table(peak_list)
            result = self.model.get_laue_info()
            if result is not None:
                self.view.plot_laue(self.model.gamma, self.model.nu, *result)
            # if not self.view.draw_all():
            self.visualize()

    def lookup_angle(self, gamma, nu):

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

        vals = self.model.get_peak_selection(gamma, nu)
        if vals is not None:
            gamma, nu, lamdas, row = vals
            self.view.update_laue(gamma, nu, lamdas)
            self.view.highlight_peak(row)

    def highlight_peak(self):
        row = self.view.get_peak()

        vals = self.model.get_peak_index(row)
        if vals is not None:
            gamma, nu, lamdas, row = vals
            self.view.update_laue(gamma, nu, lamdas)

    def delete_angles(self):
        worker = self.view.worker(self.delete_angles_process)
        worker.connect_result(self.delete_angles_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def delete_angles_complete(self, rows):
        if rows is not None:
            self.view.delete_angles(rows)
            self.update_peaks()

    def delete_angles_process(self, progress):

        rows = self.view.get_angles_to_delete()

        if rows is not None:
            self.model.delete_angles(rows)

            progress("Angles deleted!", 0)

            return rows

        else:
            progress("No rows selected for deletion.", 0)

    def add_orientation(self):
        worker = self.view.worker(self.add_orientation_process)
        worker.connect_result(self.add_orientation_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def add_orientation_complete(self, result):
        angles, all_angles, free_angles = result

        comment = self.model.comment
        update_angles = []
        for angle, angle_name in zip(angles, all_angles):
            if angle_name in free_angles:
                update_angles.append(angle)

        title = self.view.get_title()
        self.view.add_orientations(title, comment, [update_angles])
        self.update_peaks()

    def add_orientation_process(self, progress):
        angles = self.view.get_angles()
        free_angles = self.view.get_free_angles()
        all_angles = self.view.get_all_angles()

        wavelength = self.view.get_wavelength()
        d_min = self.view.get_d_min()
        rows = self.view.get_number_of_orientations()

        if len(angles) > 0:
            progress("Calculating reflections", 5)

            self.model.add_orientation(angles, wavelength, d_min, rows)

            progress("Reflections calculated!", 0)

            return angles, all_angles, free_angles

        else:
            progress("No angles provided for orientation.", 0)

    def mesh_scan(self):
        worker = self.view.worker(self.mesh_scan_process)
        worker.connect_result(self.mesh_scan_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def mesh_scan_complete(self, result):
        title = self.view.get_title()
        if result is not None:
            self.view.add_orientations(title, "Mesh Scan", result)
            self.update_peaks()

    def mesh_scan_process(self, progress):
        mesh_angles = self.view.get_mesh_angles()
        free_angles = self.view.get_free_angles()
        all_angles = self.view.get_all_angles()

        wavelength = self.view.get_wavelength()
        d_min = self.view.get_d_min()
        rows = self.view.get_number_of_orientations()

        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        axes, polarities = self.model.get_axes_polarities(instrument, mode)
        self.model.generate_axes(axes, polarities)

        progress("Initializing instrument", 5)

        self.create_instrument()

        if mesh_angles is not None:
            progress("Calculating reflections", 5)

            angles = self.model.add_mesh(
                mesh_angles, wavelength, d_min, rows, free_angles, all_angles
            )

            progress("Reflections calculated!", 0)

            return angles

        else:
            progress("No mesh angles provided for mesh scan.", 0)

    def convert_mesh_to_hkl(self):
        self.mesh = True
        self.convert_to_hkl()

    def convert_plan_to_hkl(self):
        self.mesh = False
        self.convert_to_hkl()

    def convert_to_hkl(self):
        if self.convert_idle:
            worker = self.view.worker(self.convert_to_hkl_process)
            worker.connect_result(self.convert_to_hkl_complete)
            worker.connect_progress(self.update_processing)

            self.view.start_worker_pool(worker)

    def convert_to_hkl_complete(self, result):
        if result is not None:
            self.view.update_slice(result)
        self.convert_idle = True

    def convert_to_hkl_process(self, progress):
        instrument = self.view.get_instrument()
        mode = self.view.get_mode()

        axes, polarities = self.model.get_axes_polarities(instrument, mode)
        self.model.generate_axes(axes, polarities)

        proj = self.view.get_projection_matrix()

        value = self.view.get_slice_value()

        thickness = self.view.get_slice_thickness()

        d_min = self.view.get_d_min()

        symm = self.view.use_symmetry_mesh()

        point_group = self.view.get_point_group()

        validate = [proj, value, thickness, d_min]

        if all(elem is not None for elem in validate):
            U, V, W, invalid = self.model.validate_projection(proj)

            if not invalid:

                norm = self.get_normal()

                if self.mesh:
                    angles = self.view.get_mesh_angles()
                else:
                    angles = self.view.get_plan_angles()

                wavelength = self.view.get_wavelength()

                if len(angles) > 0:

                    progress("Initializing instrument...", 5)

                    self.create_instrument()

                    self.model.calculate_footprint(wavelength, d_min)

                    progress("Calculating footprint...", 50)

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
        slice_plane = self.view.get_slice()

        if slice_plane == "Axis 1/2":
            norm = [0, 0, 1]
        elif slice_plane == "Axis 1/3":
            norm = [0, 1, 0]
        else:
            norm = [1, 0, 0]

        return norm

    def visualize(self):
        point_group = self.view.get_point_group()
        lattice_centering = self.view.get_lattice_centering()
        use = self.view.get_orientations_to_use()
        d_min = self.view.get_d_min()
        draw_all = self.view.draw_all()
        row = self.view.get_peak_list()

        if self.draw_idle:

            self.update_processing()

            self.update_processing("Calculating statistics...", 5)

            stats = self.model.calculate_statistics(
                point_group, lattice_centering, use, d_min
            )

            self.update_processing("Statistics calculated...", 30)

            if stats is not None and self.model.has_UB():
                self.draw_idle = False

                self.view.plot_statistics(*stats)

                self.update_processing("Calculating coverage...", 50)

                peak_dict = self.model.get_coverage_info(
                    point_group, lattice_centering, draw_all, row
                )

                self.update_processing("Coverage calculated...", 80)

                if peak_dict is not None:
                    peak_dict["axis_limit"] = self.view.get_d_min()

                    self.view.add_peaks(peak_dict)

                self.draw_idle = True

            else:

                self.view.add_peaks(None)

            self.update_complete("Data visualized!")

    def optimize_coverage(self):
        worker = self.view.worker(self.optimize_coverage_process)
        worker.connect_result(self.optimize_coverage_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def optimize_coverage_complete(self, result):
        title = self.view.get_title()
        if result is not None:
            self.view.add_orientations(title, "CrystalPlan", result)
            self.update_peaks()

    def optimize_coverage_process(self, progress):
        point_group = self.view.get_point_group()
        lattice_centering = self.view.get_lattice_centering()
        use = self.view.get_orientations_to_use()
        opt = self.view.get_optimized_settings()
        d_min = self.view.get_d_min()
        wavelength = self.view.get_wavelength()
        n_orient = self.view.get_settings()

        n_elite = 2
        n_gener = 10
        n_indiv = 10
        mutation_rate = 0.15

        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        axes = self.model.get_goniometer_axes(instrument, mode)
        limits = self.view.get_goniometer_limits()

        if self.model.has_UB():
            progress("Initializing instrument", 5)

            self.create_instrument()

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
        instrument = self.view.get_instrument()
        cal = self.view.get_detector_calibration()
        gon = self.view.get_goniometer_calibration()
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
        self.model.update_goniometer_motors(limits, motors, cal, gon, mask)

    def save_CSV(self):
        filename = self.view.save_CSV_file_dialog()

        if filename:
            self.update_plan()
            self.model.save_plan(filename)

    def save_experiment(self):
        filename = self.view.save_experiment_file_dialog()

        if filename:
            self.update_plan()
            self.model.save_experiment(filename)

    def load_experiment(self):
        filename = self.view.load_experiment_file_dialog()

        if filename:
            plan, config, symm = self.model.load_experiment(filename)

            titles, settings, comments, counts, values, use = plan
            instrument, mode, wl, d_min, lims, vals, cal, gon, mask = config
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
            self.view.set_goniometer_calibration(gon)
            self.view.set_mask(mask)
            self.view.set_crystal_system(cs)
            self.switch_crystal()
            self.view.set_point_group(pg)
            self.switch_group()
            self.view.set_lattice_centering(lc)
            self.view.add_settings(*table)
            self.add_settings()

    def add_settings(self):
        worker = self.view.worker(self.add_settings_process)
        worker.connect_result(self.add_settings_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def add_settings_complete(self, result):
        if result is not None:
            self.update_peaks()

    def add_settings_process(self, progress):
        wavelength = self.view.get_wavelength()
        d_min = self.view.get_d_min()
        rows = self.view.get_number_of_orientations()

        instrument = self.view.get_instrument()
        mode = self.view.get_mode()
        axes, polarities = self.model.get_axes_polarities(instrument, mode)
        self.model.generate_axes(axes, polarities)
        limits = self.view.get_goniometer_limits()

        progress("Initializing instrument", 5)

        self.create_instrument()
        self.model.clear_combined()

        for row in range(rows):
            progress("Calculating settings", 90 // rows * (row + 1) + 5)

            angles = self.view.get_angle_setting(row)

            setting = self.model.get_setting(angles, limits)

            self.model.add_orientation(setting, wavelength, d_min, row)

        progress("Settings calculated!", 0)

        return rows
