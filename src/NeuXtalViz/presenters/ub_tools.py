from NeuXtalViz.presenters.base_presenter import NeuXtalVizPresenter

import functools
import numpy as np


class UB(NeuXtalVizPresenter):
    def __init__(self, view, model):
        super(UB, self).__init__(view, model)

        self.view.connect_load_Q(self.load_Q)
        self.view.connect_save_Q(self.save_Q)
        self.view.connect_load_peaks(self.load_peaks)
        self.view.connect_save_peaks(self.save_peaks)
        self.view.connect_load_UB(self.load_UB)
        self.view.connect_save_UB(self.save_UB)
        self.view.connect_switch_instrument(self.switch_instrument)
        self.view.connect_wavelength(self.update_wavelength)

        self.view.connect_browse_calibration(self.load_detector_calibration)
        self.view.connect_browse_goniometer(self.load_goniometer_calibration)
        self.view.connect_browse_tube(self.load_tube_calibration)

        self.view.connect_convert_Q(self.convert_Q)
        self.view.connect_reload_convert_Q(self.convert_Q_reload)
        self.view.connect_find_peaks(self.find_peaks)
        self.view.connect_find_spacing(self.update_find_spacing)
        self.view.connect_find_distance(self.update_find_distance)
        self.view.connect_index_peaks(self.index_peaks)
        self.view.connect_predict_peaks(self.predict_peaks)
        self.view.connect_integrate_peaks(self.integrate_peaks)
        self.view.connect_filter_peaks(self.filter_peaks)
        self.view.connect_undo_filter_peaks(self.undo_filter_peaks)
        self.view.connect_find_conventional(self.find_conventional)
        self.view.connect_lattice_transform(self.lattice_transform)
        self.view.connect_symmetry_transform(self.symmetry_transform)
        self.view.connect_transform_UB(self.transform_UB)
        self.view.connect_optimize_UB(self.refine_UB)
        self.view.connect_find_niggli(self.find_niggli)
        self.view.connect_calculate_peaks(self.calculate_peaks)
        self.view.connect_highlight_1(self.add_highlight_1)
        self.view.connect_highlight_2(self.add_highlight_2)
        self.view.connect_calculate_highlight(self.calculate_highlight)
        self.view.connect_delete_peak(self.delete_peak)
        self.view.connect_cell_row_highligter(self.highlight_cell)
        self.view.connect_peak_row_highligter(self.highlight_peak)
        self.view.connect_select_cell(self.select_cell)
        self.view.connect_set_UB(self.set_UB)
        self.view.connect_set_UB_from_scattering_plane(
            self.set_UB_from_scattering_plane
        )

        self.switch_instrument()
        self.lattice_transform()

        self.view.connect_convert_to_hkl(self.convert_to_hkl)

        self.view.connect_data_combo(self.handle_data_combo_change)
        self.view.connect_diffraction(self.update_roi)
        self.view.connect_d_min(self.update_instrument_view)
        self.view.connect_d_max(self.update_instrument_view)
        self.view.connect_horizontal(self.update_roi)
        self.view.connect_vertical(self.update_roi)
        self.view.connect_horizontal_roi(self.update_roi)
        self.view.connect_vertical_roi(self.update_roi)
        self.view.connect_instrument_scale_combo(
            self.update_instrument_display
        )
        self.view.connect_vlim_combo(self.update_instrument_view_autoscaled)
        self.view.connect_vbar_combo(self.update_instrument_display)
        self.view.connect_inst_vmin_line(self.update_inst_cvals)
        self.view.connect_inst_vmax_line(self.update_inst_cvals)

        self.view.connect_add_peak(self.add_peak)
        self.view.connect_save_roi_mask(self.save_roi_mask)
        self.view.connect_slice_ready(self.add_slice_peak)
        self.view.connect_check_hkl(self.calculate_hkl)

        self.view.connect_roi_ready(self.update_scan)
        self.view.connect_scan_ready(self.update_check_hkl)

        self.view.connect_h_index(self.hand_index_fractional)
        self.view.connect_k_index(self.hand_index_fractional)
        self.view.connect_l_index(self.hand_index_fractional)

        self.view.connect_integer_h_index(self.hand_index_integer)
        self.view.connect_integer_k_index(self.hand_index_integer)
        self.view.connect_integer_l_index(self.hand_index_integer)

        self.view.connect_integer_m_index(self.hand_index_integer)
        self.view.connect_integer_n_index(self.hand_index_integer)
        self.view.connect_integer_p_index(self.hand_index_integer)

        self.view.connect_slice_combo(self.reslice, self.update_slice_extent)
        self.view.connect_slice_thickness_line(self.reslice)
        self.view.connect_slice_width_line(self.reslice)

        self.view.connect_clim_combo(self.update_slice_display)
        self.view.connect_cbar_combo(self.update_slice_display)

        self.view.connect_slice_scale_combo(self.update_slice_display)
        self.view.connect_slice_auto_limits(self.update_slice_display)
        self.view.connect_slice_line(self.reslice)
        self.view.connect_slice_slider(self.reslice)
        self.view.connect_vmin_line(self.update_cvals)
        self.view.connect_vmax_line(self.update_cvals)
        self.view.connect_instrument_auto_limits(
            self.update_instrument_display
        )

        self.slice_idle = True
        self.volume_idle = True
        self.instrument_view_idle = True
        self.slice_signal_cache = None
        self.inst_signal_cache = None

        self.view.connect_cluster(self.cluster)
        self.view.connect_calculate_alignment(self.calculate_alignment)

    def refresh_peak_views(self):
        peaks = self.model.get_peak_info() if self.model.has_peaks() else []
        self.view.update_peaks_table(peaks)
        self.view.update_alignment_runs(peaks)

    def _calculate_display_limits(self, data, method, scale):
        clip = self.model.calculate_clim(np.array(data, copy=True), method)

        vmin = np.nanmin(clip)
        vmax = np.nanmax(clip)

        if scale == "log" and (not np.isfinite(vmin) or vmin <= 0):
            signal = np.asarray(data)
            positive = signal[np.isfinite(signal) & (signal > 0)]
            if positive.size > 0:
                vmin = np.nanmin(positive)

        if np.isclose(vmin, vmax) or not np.isfinite([vmin, vmax]).all():
            return (0.1, 1) if scale == "log" else (0, 1)

        return vmin, vmax

    def _resolve_display_limits(
        self, data, method, scale, auto_limits, current_vmin, current_vmax
    ):
        if (
            not auto_limits
            and current_vmin is not None
            and current_vmax is not None
            and current_vmin < current_vmax
        ):
            if current_vmin <= 0 and scale == "log":
                current_vmin = current_vmax / 10
            return current_vmin, current_vmax

        return self._calculate_display_limits(data, method, scale)

    def update_cvals(self):
        vmin = self.view.get_vmin_value()
        vmax = self.view.get_vmax_value()

        if vmin is not None and vmax is not None:
            if vmin < vmax:
                if vmin <= 0 and self.view.get_slice_scale() == "log":
                    vmin = vmax / 10
                self.view.update_colorbar_vlims(vmin, vmax)

    def update_inst_cvals(self):
        vmin = self.view.get_inst_vmin_value()
        vmax = self.view.get_inst_vmax_value()

        if vmin is not None and vmax is not None:
            if vmin < vmax:
                if vmin <= 0 and self.view.get_instrument_scale() == "log":
                    vmin = vmax / 10
                self.view.update_instrument_colorbar_vlims(vmin, vmax)

    def update_instrument_view_autoscaled(self):
        if self.view.get_instrument_auto_limits():
            self.view.inst_vmin_line.blockSignals(True)
            self.view.inst_vmax_line.blockSignals(True)
            self.view.clear_inst_vlims()
            self.view.inst_vmin_line.blockSignals(False)
            self.view.inst_vmax_line.blockSignals(False)

        if self.inst_signal_cache is not None:
            self.update_instrument_display()
        else:
            self.update_instrument_view()

    def handle_data_combo_change(self):
        self.inst_signal_cache = None
        self.update_instrument_view_autoscaled()

    def update_instrument_clim(self):
        self.update_instrument_display()

    def update_instrument_display(self):
        if self.inst_signal_cache is None:
            self.update_instrument_view()
            return

        vmin, vmax = self._resolve_display_limits(
            self.inst_signal_cache,
            self.get_vlim_method(),
            self.view.get_instrument_scale(),
            self.view.get_instrument_auto_limits(),
            self.view.get_inst_vmin_value(),
            self.view.get_inst_vmax_value(),
        )

        self.view.update_instrument_display(
            self.view.get_instrument_colormap(),
            self.view.get_instrument_scale(),
            vmin,
            vmax,
        )

    def update_slice_clim(self):
        self.update_slice_display()

    def update_slice_display(self):
        if self.slice_signal_cache is None:
            self.reslice()
            return

        vmin, vmax = self._resolve_display_limits(
            self.slice_signal_cache,
            self.get_clim_method(),
            self.view.get_slice_scale(),
            self.view.get_slice_auto_limits(),
            self.view.get_vmin_value(),
            self.view.get_vmax_value(),
        )
        self.view.update_slice_display(
            self.view.get_colormap(),
            self.view.get_slice_scale(),
            vmin,
            vmax,
        )

    def update_find_spacing(self):
        """
        Update the find peaks spacing in the view using Q value.

        Parameters
        ----------
        None
        """
        d = self.view.get_find_peaks_spacing()
        Q = self.model.get_Q(d)
        self.view.set_find_peaks_distance(Q)

    def update_find_distance(self):
        """
        Update the find peaks distance in the view using d value.

        Parameters
        ----------
        None
        """
        Q = self.view.get_find_peaks_distance()
        d = self.model.get_d(Q)
        self.view.set_find_peaks_spacing(d)

    def hand_index_fractional(self):
        """
        Handle fractional indexing for peaks and update the view.

        Parameters
        ----------
        None
        """
        mod_info = self.get_modulation_info()
        hkl_info = self.view.get_indices()
        index_row = self.view.get_peak()

        if (
            mod_info is not None
            and hkl_info is not None
            and index_row is not None
        ):
            mod_vec_1, mod_vec_2, mod_vec_3, *_ = mod_info
            hkl, int_hkl, int_mnp = hkl_info

            int_hkl, int_mnp = self.model.calculate_integer(
                mod_vec_1, mod_vec_2, mod_vec_3, hkl
            )

            self.model.set_peak(index_row, hkl, int_hkl, int_mnp)

            self.view.update_table_index(index_row, hkl)

            self.view.set_indices(hkl, int_hkl, int_mnp)

    def hand_index_integer(self):
        """
        Handle integer indexing for peaks and update the view.

        """

        mod_info = self.get_modulation_info()
        hkl_info = self.view.get_indices()
        index_row = self.view.get_peak()

        if (
            mod_info is not None
            and hkl_info is not None
            and index_row is not None
        ):
            mod_vec_1, mod_vec_2, mod_vec_3, *_ = mod_info
            hkl, int_hkl, int_mnp = hkl_info

            hkl = self.model.calculate_fractional(
                mod_vec_1, mod_vec_2, mod_vec_3, int_hkl, int_mnp
            )

            self.model.set_peak(index_row, hkl, int_hkl, int_mnp)

            self.view.update_table_index(index_row, hkl)

            self.view.set_indices(hkl, int_hkl, int_mnp)

    def convert_Q(self, force_reload=False):
        self.update_data_status()

        instrument = self.view.get_instrument()
        wavelength = self.view.get_wavelength()
        tube_cal = self.view.get_tube_calibration()
        det_cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        IPTS = self.view.get_IPTS()
        runs = self.view.get_runs()
        exp = self.view.get_experiment()
        lorentz = self.view.get_lorentz()
        time_stop = self.view.get_time_stop()
        d_min = self.view.get_convert_min_d()

        worker = self.view.worker(
            functools.partial(
                self.convert_Q_process,
                instrument=instrument,
                wavelength=wavelength,
                tube_cal=tube_cal,
                det_cal=det_cal,
                gon_cal=gon_cal,
                IPTS=IPTS,
                runs=runs,
                exp=exp,
                lorentz=lorentz,
                time_stop=time_stop,
                d_min=d_min,
                force_reload=force_reload,
            )
        )
        worker.connect_result(self.convert_Q_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def convert_Q_reload(self):
        self.convert_Q(force_reload=True)

    def convert_Q_complete(self, result):
        if result is not None:
            self.view.set_data_list(result["runs"])
            self.view.update_diffraction_label(result["mono"])

            self.update_instrument_view()

    def convert_Q_process(
        self,
        progress=None,
        stop_event=None,
        instrument=None,
        wavelength=None,
        tube_cal=None,
        det_cal=None,
        gon_cal=None,
        IPTS=None,
        runs=None,
        exp=None,
        lorentz=None,
        time_stop=None,
        d_min=None,
        force_reload=False,
    ):
        if self.stop_processing(stop_event):
            return None

        validate = [IPTS, runs, wavelength]

        if instrument == "DEMAND":
            validate.append(exp)

        if all(elem is not None for elem in validate):
            mono = self.model.is_mono(wavelength)

            progress("Processing...", 1)

            if self.stop_processing(stop_event):
                return None

            progress("Data loading...", 10)

            data_load = self.model.load_data(
                instrument,
                IPTS,
                runs,
                exp,
                time_stop,
                force_reload=force_reload,
            )

            if self.stop_processing(stop_event):
                return None

            if not data_load:
                progress("Files do not exist.", 0)
                return None

            progress("Data loaded...", 40)

            if self.stop_processing(stop_event):
                return None

            progress("Data calibrating...", 50)

            self.model.calibrate_data(instrument, det_cal, gon_cal, tube_cal)

            if self.stop_processing(stop_event):
                return None

            progress("Data calibrated...", 60)

            progress("Data converting...", 70)

            self.model.convert_data(
                instrument,
                wavelength,
                lorentz,
                d_min,
                force_reload=force_reload,
            )

            if self.stop_processing(stop_event):
                return None

            progress("Data converted...", 99)

            progress("Data converted!", 0)

            return {
                "mono": mono,
                "runs": self.model.get_number_workspaces(),
            }

        else:
            if instrument is None:
                progress("Invalid instrument.", 0)
            elif IPTS is None:
                progress("Invalid IPTS.", 0)
            elif runs is None:
                progress("Invalid run(s).", 0)
            elif wavelength is None:
                progress("Invalid wavelength.", 0)
            elif exp is None and instrument == "DEMAND":
                progress("Invalid experiment for DEMAND.", 0)
            else:
                progress("Invalid parameters for Q conversion.", 0)

    def add_peak(self):
        if self.model.has_Q():
            ind = self.view.get_data_list()
            horz = self.view.get_horizontal()
            vert = self.view.get_vertical()
            val = self.view.get_diffraction()

            validate = [horz, vert, val]

            if all(elem is not None for elem in validate):
                self.model.add_peak(ind, val, horz, vert)
                self.visualize()
            else:
                if horz is None:
                    self.update_processing("Invalid horizontal value.", 0)
                elif vert is None:
                    self.update_processing("Invalid vertical value.", 0)
                elif val is None:
                    self.update_processing("Invalid diffraction value.", 0)
                else:
                    self.update_processing(
                        "Invalid parameters for add_peak.", 0
                    )

    def add_slice_peak(self, h, k, l):
        if self.model.has_Q() and self.model.has_UB():
            ind = self.view.get_data_list()

            if ind is not None:
                self.model.add_peak_from_hkl(ind, [h, k, l])
                self.visualize()
            else:
                self.update_processing("Invalid data list index.", 0)

    def delete_peak(self):
        peaks = self.view.get_peaks()

        if self.model.has_peaks() and len(peaks) > 0:
            self.model.delete_peak_rows(peaks)
            self.view.clear_peak_selection()
            self.visualize()
        else:
            self.update_processing("No highlighted peaks selected.", 0)

    def calculate_hkl(self):
        if self.model.has_Q():
            ind = self.view.get_data_list()
            hkl = self.view.get_check_hkl()

            validate = [ind, hkl]

            if all(elem is not None for elem in validate):
                vals = self.model.calculate_hkl_position(ind, *hkl)

                if vals is not None:
                    x, horz, vert = vals
                    self.view.set_diffraction(x)
                    self.view.set_horizontal(horz)
                    self.view.set_vertical(vert)
                    self.update_instrument_view()
            else:
                if ind is None:
                    self.update_processing("Invalid data list index.", 0)
                elif hkl is None:
                    self.update_processing("Invalid hkl value.", 0)
                else:
                    self.update_processing(
                        "Invalid parameters for calculating hkl.", 0
                    )

    def update_instrument_view(self):
        if not self.instrument_view_idle:
            return

        self.instrument_view_idle = False

        ind = self.view.get_data_list()
        d_min = self.view.get_d_min()
        d_max = self.view.get_d_max()
        horz = self.view.get_horizontal()
        vert = self.view.get_vertical()
        horz_roi = self.view.get_horizontal_roi()
        vert_roi = self.view.get_vertical_roi()
        val = self.view.get_diffraction()
        vlim_method = self.get_vlim_method()
        instrument_scale = self.view.get_instrument_scale()
        instrument_auto_limits = self.view.get_instrument_auto_limits()
        inst_vmin = self.view.get_inst_vmin_value()
        inst_vmax = self.view.get_inst_vmax_value()

        worker = self.view.worker(
            functools.partial(
                self.update_instrument_view_process,
                ind=ind,
                d_min=d_min,
                d_max=d_max,
                horz=horz,
                vert=vert,
                horz_roi=horz_roi,
                vert_roi=vert_roi,
                val=val,
                vlim_method=vlim_method,
                instrument_scale=instrument_scale,
                instrument_auto_limits=instrument_auto_limits,
                inst_vmin=inst_vmin,
                inst_vmax=inst_vmax,
            )
        )
        worker.connect_result(self.update_instrument_view_complete)
        # worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def update_instrument_view_complete(self, result):
        self.instrument_view_idle = True
        if result is not None:
            self.inst_signal_cache = np.array(result[0]["img"], copy=True)
            self.view.update_instrument_view(result[0])
            self.view.update_roi_view(result[1])
            self.view.update_scan_view(result[1])
            self.update_run_goniometer()

            self.update_check_hkl()

    def update_run_goniometer(self):
        ind = self.view.get_data_list()
        angles = self.model.get_run_goniometer(ind)
        self.view.set_instrument_goniometer_setting(angles)

    def update_instrument_view_process(
        self,
        progress=None,
        stop_event=None,
        ind=None,
        d_min=None,
        d_max=None,
        horz=None,
        vert=None,
        horz_roi=None,
        vert_roi=None,
        val=None,
        vlim_method=None,
        instrument_scale=None,
        instrument_auto_limits=None,
        inst_vmin=None,
        inst_vmax=None,
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_Q():
            validate = [d_min, d_max, horz, vert, horz_roi, vert_roi, val]

            if all(elem is not None for elem in validate):
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Detector viewing...", 10)

                self.model.calculate_instrument_view(ind, d_min, d_max)

                if self.stop_processing(stop_event):
                    return None

                progress("Detector viewed...", 50)

                self.model.extract_roi(horz, vert, horz_roi, vert_roi, val)

                if self.stop_processing(stop_event):
                    return None

                img = self.model.inst_view["img"]
                signal = img.ravel()

                self.model.inst_view["vmin"], self.model.inst_view["vmax"] = (
                    self._resolve_display_limits(
                        signal,
                        vlim_method,
                        instrument_scale,
                        instrument_auto_limits,
                        inst_vmin,
                        inst_vmax,
                    )
                )

                progress("ROI viewed...", 70)

                progress("Data/ROI viewed!", 0)

                return self.model.inst_view, self.model.roi_view
            else:
                missing = []
                if d_min is None:
                    missing.append("d_min")
                if d_max is None:
                    missing.append("d_max")
                if horz is None:
                    missing.append("horizontal")
                if vert is None:
                    missing.append("vertical")
                if horz_roi is None:
                    missing.append("horizontal ROI")
                if vert_roi is None:
                    missing.append("vertical ROI")
                if val is None:
                    missing.append("diffraction value")
                if missing:
                    progress(f"Invalid parameter(s): {', '.join(missing)}.", 0)
                else:
                    progress("Invalid parameters for instrument view.", 0)
        else:
            progress("Invalid parameters for instrument view.", 0)

    def update_roi(self):
        if not self.instrument_view_idle:
            return
        if self.model.has_Q():
            horz = self.view.get_horizontal()
            vert = self.view.get_vertical()
            horz_roi = self.view.get_horizontal_roi()
            vert_roi = self.view.get_vertical_roi()
            val = self.view.get_diffraction()

            validate = [horz, vert, horz_roi, vert_roi, val]

            if all(elem is not None for elem in validate):
                self.model.extract_roi(horz, vert, horz_roi, vert_roi, val)

                self.view.update_roi_view(self.model.roi_view)

                self.update_check_hkl()

    def update_scan(self):
        if not self.instrument_view_idle:
            return
        if self.model.has_Q():
            horz = self.view.get_horizontal()
            vert = self.view.get_vertical()
            horz_roi = self.view.get_horizontal_roi()
            vert_roi = self.view.get_vertical_roi()
            val = self.view.get_diffraction()

            validate = [horz, vert, horz_roi, vert_roi, val]

            if all(elem is not None for elem in validate):
                self.model.extract_roi(horz, vert, horz_roi, vert_roi, val)

                self.view.update_scan_view(self.model.roi_view)

                self.update_check_hkl()

    def update_check_hkl(self):
        ind = self.view.get_data_list()
        horz = self.view.get_horizontal()
        vert = self.view.get_vertical()
        val = self.view.get_diffraction()

        validate = [horz, vert, val]

        if all(elem is not None for elem in validate):
            ind = self.view.get_data_list()
            hkl = self.model.roi_scan_to_hkl(ind, val, horz, vert)
            if hkl is not None:
                self.view.set_check_hkl(*hkl)

    def update_data_status(self):
        instrument = self.view.get_instrument()

        IPTS = self.view.get_IPTS()
        runs = self.view.get_runs()
        exp = self.view.get_experiment()

        validate = [IPTS, runs]

        if instrument == "DEMAND":
            validate.append(exp)

        files = None
        if all(elem is not None for elem in validate):
            files, *_ = self.model.get_files(instrument, IPTS, runs, exp)

        self.view.set_Q_status(self.model.get_Q_status(files))
        self.view.set_peaks_status(self.model.get_peaks_status())
        self.view.set_UB_status(self.model.get_UB_status())
        self.view.set_undo_filter_enabled(self.model.can_undo_filter_peaks())

    def visualize(self):

        self.update_data_status()

        Q_hist = self.model.get_Q_info()

        if Q_hist is not None and self.volume_idle:
            self.volume_idle = False

            self.update_processing()

            self.update_processing("Updating view...", 50)

            self.view.add_Q_viz(Q_hist)

            if self.model.has_UB():
                self.model.update_UB()

                self.update_oriented_lattice()

                self.view.set_transform(self.model.get_transform())

                self.update_lattice_info()

            if self.model.has_peaks():
                self.refresh_peak_views()
            else:
                self.refresh_peak_views()

            self.update_complete("Data visualized!")

            self.volume_idle = True

    def update_lattice_info(self):
        params = self.model.get_lattice_constants()
        errors = self.model.get_lattice_constant_errors()

        if params is not None:
            self.view.set_lattice_constants(params, errors)

        params = self.model.get_sample_directions()

        if params is not None:
            self.view.set_sample_directions(params)

    def find_peaks(self):
        self.update_data_status()

        Q_min = self.view.get_find_peaks_distance()
        d_max = self.view.get_find_peaks_spacing()
        params = self.view.get_find_peaks_parameters()
        edge = self.view.get_find_peaks_edge()
        peak_width = self.view.get_peak_width()
        no_al = self.view.get_avoid_aluminum()
        no_cu = self.view.get_avoid_copper()
        no_fe = self.view.get_avoid_iron()

        worker = self.view.worker(
            functools.partial(
                self.find_peaks_process,
                Q_min=Q_min,
                d_max=d_max,
                params=params,
                edge=edge,
                peak_width=peak_width,
                no_al=no_al,
                no_cu=no_cu,
                no_fe=no_fe,
            )
        )
        worker.connect_result(self.find_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def find_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def find_peaks_process(
        self,
        progress=None,
        stop_event=None,
        Q_min=None,
        d_max=None,
        params=None,
        edge=None,
        peak_width=None,
        no_al=None,
        no_cu=None,
        no_fe=None,
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_Q():
            if (
                Q_min is not None
                and params is not None
                and peak_width is not None
            ):
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Finding peaks...", 10)

                self.model.find_peaks(Q_min, *params, edge)
                d_min = self.model.get_d_min()

                if self.stop_processing(stop_event):
                    return None

                if no_al and d_min < d_max:
                    self.model.avoid_aluminum_contamination(
                        d_min, d_max, peak_width
                    )
                if no_cu and d_min < d_max:
                    self.model.avoid_copper_contamination(
                        d_min, d_max, peak_width
                    )
                if no_fe and d_min < d_max:
                    self.model.avoid_iron_contamination(
                        d_min, d_max, peak_width
                    )

                progress("Peaks found...", 90)

                progress("Peaks found!", 100)

            else:
                progress("Invalid parameters.", 0)

    def find_conventional(self):
        self.update_data_status()

        params = self.view.get_lattice_constants()
        tol = self.view.get_calculate_UB_tol()

        worker = self.view.worker(
            functools.partial(
                self.find_conventional_process,
                params=params,
                tol=tol,
            )
        )
        worker.connect_result(self.find_conventional_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def find_conventional_complete(self, result=None):
        self.update_data_status()

    def find_conventional_process(
        self, progress=None, stop_event=None, params=None, tol=None
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_peaks():
            if params is not None and tol is not None:
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Finding UB...", 10)

                self.model.determine_UB_with_lattice_parameters(*params, tol)

                progress("UB found...", 90)

                progress("UB found!", 100)

            else:
                progress("Invalid parameters.", 0)

    def find_niggli(self):
        self.update_data_status()

        params = self.view.get_min_max_constants()
        tol = self.view.get_calculate_UB_tol()

        worker = self.view.worker(
            functools.partial(
                self.find_niggli_process,
                params=params,
                tol=tol,
            )
        )
        worker.connect_result(self.find_niggli_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def find_niggli_complete(self, result):
        self.show_cells()

    def find_niggli_process(
        self, progress=None, stop_event=None, params=None, tol=None
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_peaks():
            if params is not None and tol is not None:
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Finding UB...", 10)

                self.model.determine_UB_with_niggli_cell(*params, tol)

                progress("UB found...", 90)

                progress("UB found!", 100)

            else:
                progress("Invalid parameters.", 0)

    def show_cells(self):
        scalar = self.view.get_max_scalar_error()

        worker = self.view.worker(
            functools.partial(
                self.show_cells_process,
                scalar=scalar,
            )
        )
        worker.connect_result(self.show_cells_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def show_cells_complete(self, result):
        if result is not None:
            self.view.update_cell_table(result)

    def show_cells_process(self, progress=None, stop_event=None, scalar=None):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_peaks() and self.model.has_UB():
            if scalar is not None:
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Finding possible cells...", 50)

                cells = self.model.possible_conventional_cells(scalar)

                progress("Possible cells found!", 100)

                return cells

            else:
                progress("Invalid parameters.", 0)

    def set_UB(self):
        constants = self.view.get_lattice_constants()
        directions = self.view.get_sample_directions()

        if constants is not None and directions is not None:
            self.model.set_manual_UB(constants, directions)
            self.update_data_status()
            self.update_lattice_info()
            self.update_oriented_lattice()

    def set_UB_from_scattering_plane(self):
        self.update_data_status()

        constants = self.view.get_lattice_constants()
        directions = self.view.get_sample_directions()

        worker = self.view.worker(
            functools.partial(
                self.set_UB_from_scattering_plane_process,
                constants=constants,
                directions=directions,
            )
        )
        worker.connect_result(self.set_UB_from_scattering_plane_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def set_UB_from_scattering_plane_complete(self, result=None):
        self.update_data_status()

    def set_UB_from_scattering_plane_process(
        self, progress=None, stop_event=None, constants=None, directions=None
    ):
        if self.stop_processing(stop_event):
            return None

        if constants is None or directions is None:
            progress("Invalid parameters.", 0)
            return None

        if self.stop_processing(stop_event):
            return None

        progress("Processing...", 1)

        if self.stop_processing(stop_event):
            return None

        progress("Finding UB from scattering plane...", 10)

        try:
            self.model.find_UB_from_scattering_plane(constants, directions)
        except Exception as exc:
            progress(str(exc), 0)
            return None

        if self.stop_processing(stop_event):
            return None

        progress("UB found from scattering plane!", 100)

        return True

    def select_cell(self):
        form = self.view.get_form_number()
        tol = self.view.get_calculate_UB_tol()

        worker = self.view.worker(
            functools.partial(
                self.select_cell_process,
                form=form,
                tol=tol,
            )
        )
        worker.connect_result(self.select_cell_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def select_cell_complete(self, result):
        pass

    def select_cell_process(
        self, progress=None, stop_event=None, form=None, tol=None
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_peaks() and self.model.has_UB():
            if form is not None and tol is not None:
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Selecting cell...", 50)

                self.model.select_cell(form, tol)

                progress("Cell selected...", 99)

                progress("Cell selected!", 100)

            else:
                progress("Invalid parameters.", 0)

    def highlight_cell(self):
        form = self.view.get_form()
        self.view.set_cell_form(form)

    def highlight_peak(self):
        peaks = self.view.get_peaks()
        if len(peaks) > 0:
            peak = self.model.get_peak(peaks[0])
            if peak is not None:
                self.view.set_peak_info(peak)
                self.view.highlight_peaks([no + 1 for no in peaks])
                self.view.set_position(peak["Q"])
        else:
            self.view.highlight_peaks([])

    def lattice_transform(self):
        cell = self.view.get_lattice_transform()

        Ts = self.model.generate_lattice_transforms(cell)

        self.view.update_symmetry_symbols(list(Ts.keys()))

        self.symmetry_transform()

    def symmetry_transform(self):
        cell = self.view.get_lattice_transform()

        Ts = self.model.generate_lattice_transforms(cell)

        symbol = self.view.get_symmetry_symbol()

        if symbol in Ts.keys():
            T = Ts[symbol]

            self.view.set_transform_matrix(T)

    def transform_UB(self):
        params = self.view.get_transform_matrix()
        tol = self.view.get_transform_UB_tol()

        worker = self.view.worker(
            functools.partial(
                self.transform_UB_process,
                params=params,
                tol=tol,
            )
        )
        worker.connect_result(self.transform_UB_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def transform_UB_complete(self, result):
        self.model.copy_UB_from_peaks()

    def transform_UB_process(
        self, progress=None, stop_event=None, params=None, tol=None
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_peaks() and self.model.has_UB():
            if params is not None and tol is not None:
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Transforming UB...", 50)

                self.model.transform_lattice(params, tol)

                progress("UB transformed...", 99)

                progress("UB transformed!", 100)

            else:
                progress("Invalid parameters.", 0)

    def refine_UB(self):
        params = self.view.get_lattice_constants()
        tol = self.view.get_refine_UB_tol()
        option = self.view.get_refine_UB_option()

        worker = self.view.worker(
            functools.partial(
                self.refine_UB_process,
                params=params,
                tol=tol,
                option=option,
            )
        )
        worker.connect_result(self.refine_UB_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def refine_UB_complete(self, result):
        self.model.copy_UB_from_peaks()

    def refine_UB_process(
        self,
        progress=None,
        stop_event=None,
        params=None,
        tol=None,
        option=None,
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_peaks():
            if option == "Constrained" and params is not None:
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Refining orientation...", 50)

                self.model.refine_U_only(*params)

                progress("Orientation refined...", 99)

                progress("Orientation refined!", 100)

            elif tol is not None:
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Refining UB...", 50)

                if option == "Unconstrained":
                    self.model.refine_UB_without_constraints(tol)
                else:
                    self.model.refine_UB_with_constraints(option, tol)

                progress("UB refined...", 99)

                progress("UB refined!", 100)

            else:
                progress("Invalid parameters.", 0)

    def get_modulation_info(self):
        mod_info = self.view.get_max_order_cross_terms()
        if mod_info is not None:
            max_order, cross_terms = mod_info
        else:
            max_order, cross_terms = 0, False

        mod_vec = self.view.get_modulatation_offsets()
        if mod_vec is not None:
            mod_vec_1 = mod_vec[0:3]
            mod_vec_2 = mod_vec[3:6]
            mod_vec_3 = mod_vec[6:9]

        return mod_vec_1, mod_vec_2, mod_vec_3, max_order, cross_terms

    def index_peaks(self):
        mod_info = self.get_modulation_info()
        params = self.view.get_index_peaks_parameters()
        sat = self.view.get_index_satellite_peaks()
        round_hkl = self.view.get_index_peaks_round()

        worker = self.view.worker(
            functools.partial(
                self.index_peaks_process,
                mod_info=mod_info,
                params=params,
                sat=sat,
                round_hkl=round_hkl,
            )
        )
        worker.connect_result(self.index_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def index_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def index_peaks_process(
        self,
        progress=None,
        stop_event=None,
        mod_info=None,
        params=None,
        sat=None,
        round_hkl=None,
    ):
        if self.stop_processing(stop_event):
            return None

        mod_vec_1, mod_vec_2, mod_vec_3, max_order, cross_terms = mod_info

        if self.model.has_peaks() and self.model.has_UB():
            if params is not None:
                tol, sat_tol = params

                if sat == False:
                    max_order = 0

                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Indexing peaks...", 50)

                self.model.index_peaks(
                    tol,
                    sat_tol,
                    mod_vec_1,
                    mod_vec_2,
                    mod_vec_3,
                    max_order,
                    cross_terms,
                    round_hkl=round_hkl,
                )

                progress("Peaks indexed...", 99)

                progress("Peaks indexed!", 100)

            else:
                progress("Invalid parameters.", 0)

    def predict_peaks(self):
        mod_info = self.get_modulation_info()
        centering = self.view.get_predict_peaks_centering()
        wavelength = self.view.get_wavelength()
        params = self.view.get_predict_peaks_parameters()
        predict_sat = self.view.get_predict_satellite_peaks()
        edge = self.view.get_predict_peaks_edge()

        worker = self.view.worker(
            functools.partial(
                self.predict_peaks_process,
                mod_info=mod_info,
                centering=centering,
                wavelength=wavelength,
                params=params,
                predict_sat=predict_sat,
                edge=edge,
            )
        )
        worker.connect_result(self.predict_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def predict_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def predict_peaks_process(
        self,
        progress=None,
        stop_event=None,
        mod_info=None,
        centering=None,
        wavelength=None,
        params=None,
        predict_sat=None,
        edge=None,
    ):
        if self.stop_processing(stop_event):
            return None

        mod_vec_1, mod_vec_2, mod_vec_3, max_order, cross_terms = mod_info

        # sat = self.view.get_predict_satellite_peaks()

        if self.model.has_UB():
            if wavelength is not None and params is not None:
                d_min, sat_d_min = params

                if sat_d_min < d_min:
                    sat_d_min = d_min

                lamda_min, lamda_max = wavelength

                if self.model.is_mono(wavelength):
                    lamda_min, lamda_max = 0.97 * lamda_min, 1.03 * lamda_max

                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Predicting peaks...", 50)

                self.model.predict_peaks(
                    centering, d_min, lamda_min, lamda_max, edge
                )

                if self.stop_processing(stop_event):
                    return None

                if predict_sat:
                    progress("Predicting modulated...", 75)

                    self.model.predict_satellite_peaks(
                        sat_d_min,
                        lamda_min,
                        lamda_max,
                        mod_vec_1,
                        mod_vec_2,
                        mod_vec_3,
                        max_order,
                        cross_terms,
                    )

                progress("Peaks predicted...", 99)

                progress("Peaks predicted!", 100)

            else:
                progress("Invalid parameters.", 0)

    def integrate_peaks(self):
        params = self.view.get_integrate_peaks_parameters()
        ellipsoid = self.view.get_ellipsoid()
        centroid = self.view.get_centroid()

        worker = self.view.worker(
            functools.partial(
                self.integrate_peaks_process,
                params=params,
                ellipsoid=ellipsoid,
                centroid=centroid,
            )
        )
        worker.connect_result(self.integrate_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def integrate_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def integrate_peaks_process(
        self,
        progress=None,
        stop_event=None,
        params=None,
        ellipsoid=None,
        centroid=None,
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_peaks() and self.model.has_Q():
            if params is not None:
                method = "ellipsoid" if ellipsoid else "sphere"

                rad, inner_factor, outer_factor = params

                if inner_factor < 1:
                    inner_factor = 1
                if outer_factor < inner_factor:
                    outer_factor = inner_factor

                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Integrating peaks...", 50)

                self.model.integrate_peaks(
                    rad,
                    inner_factor,
                    outer_factor,
                    method=method,
                    centroid=centroid,
                )

                progress("Peaks integrated...", 99)

                progress("Peaks integrated!", 100)

        else:
            progress("Invalid parameters.", 0)

    def filter_peaks(self):
        name = self.view.get_filter_variable()
        operator = self.view.get_filter_comparison()
        value = self.view.get_filter_value()

        worker = self.view.worker(
            functools.partial(
                self.filter_peaks_process,
                name=name,
                operator=operator,
                value=value,
            )
        )
        worker.connect_result(self.filter_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def filter_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def filter_peaks_process(
        self,
        progress=None,
        stop_event=None,
        name=None,
        operator=None,
        value=None,
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_peaks() and value is not None:
            progress("Processing...", 1)

            if self.stop_processing(stop_event):
                return None

            progress("Filtering peaks...", 50)

            self.model.snapshot_filter_peaks()
            self.model.filter_peaks(name, operator, value)

            progress("Peaks filtered...", 99)

            progress("Peaks filtered!", 100)

        else:
            progress("Invalid parameters.", 0)

    def undo_filter_peaks(self):
        worker = self.view.worker(self.undo_filter_peaks_process)
        worker.connect_result(self.undo_filter_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def undo_filter_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def undo_filter_peaks_process(self, progress=None, stop_event=None):
        if self.stop_processing(stop_event):
            return None

        if self.model.can_undo_filter_peaks():
            progress("Processing...", 1)

            if self.stop_processing(stop_event):
                return None

            progress("Restoring peaks...", 50)

            self.model.undo_filter_peaks()

            progress("Peaks restored...", 99)

            progress("Peaks restored!", 100)
        else:
            progress("No filtered peaks backup available.", 0)

    def load_detector_calibration(self):
        inst = self.view.get_instrument()

        path = self.model.get_calibration_file_path(inst)

        filename = self.view.load_detector_cal_dialog(path)

        if filename:
            self.view.set_detector_calibration(filename)

    def load_goniometer_calibration(self):
        inst = self.view.get_instrument()

        path = self.model.get_calibration_file_path(inst)

        filename = self.view.load_goniometer_cal_dialog(path)

        if filename:
            self.view.set_goniometer_calibration(filename)

    def load_tube_calibration(self):
        inst = self.view.get_instrument()

        path = self.model.get_calibration_file_path(inst)

        filename = self.view.load_tube_cal_dialog(path)

        if filename:
            self.view.set_tube_calibration(filename)

    def load_Q(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.load_Q_file_dialog(path)

        if filename:
            self.model.load_Q(filename)

    def save_Q(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_Q_file_dialog(path)

        if filename:
            self.model.save_Q(filename)

    def load_peaks(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.load_peaks_file_dialog(path)

        if filename:
            self.model.load_peaks(filename)
            self.refresh_peak_views()

    def save_peaks(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_peaks_file_dialog(path)

        if filename:
            self.model.save_peaks(filename)

    def load_UB(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.load_UB_file_dialog(path)

        if filename:
            self.model.load_UB(filename)

            self.view.set_transform(self.model.get_transform())

    def save_UB(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_UB_file_dialog(path)

        if filename:
            self.model.save_UB(filename)

    def save_roi_mask(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_mask_file_dialog(path)

        if filename:
            success, message = self.model.save_roi_mask(inst, filename)
            self.update_processing(message, 0)

    def switch_instrument(self):
        instrument = self.view.get_instrument()

        wavelength = self.model.get_wavelength(instrument)
        self.view.set_wavelength(wavelength)

        filepath = self.model.get_raw_file_path(instrument)
        self.view.clear_run_info(filepath)

        goniometer = self.model.get_goniometer_axes(instrument)
        self.view.set_peak_goniometer_axes(goniometer)
        self.view.set_instrument_goniometer_axes(goniometer)
        self.view.set_peak_goniometer_setting(None)
        self.view.set_instrument_goniometer_setting(None)

        d_min = self.model.get_default_d_min(instrument)
        self.view.set_convert_min_d(d_min)
        self.view.set_d_min(d_min)

    def update_wavelength(self):
        wl_min, wl_max = self.view.get_wavelength()
        self.view.update_wavelength(wl_min)

    def calculate_peaks(self):
        hkl_1, hkl_2 = self.view.get_input_hkls()
        constants = self.view.get_lattice_constants()
        if constants is not None:
            d_phi = self.model.calculate_peaks(hkl_1, hkl_2, *constants)
            self.view.set_d_phi(*d_phi)

    def add_highlight_1(self):
        no = self.view.get_peak()
        if no is not None:
            peak = self.model.get_peak(no)
            if peak is not None:
                self.view.add_highlight_1(peak)

    def add_highlight_2(self):
        no = self.view.get_peak()
        if no is not None:
            peak = self.model.get_peak(no)
            if peak is not None:
                self.view.add_highlight_2(peak)

    def calculate_highlight(self):
        Qs = self.view.get_highlight()
        if Qs is not None:
            phi = self.model.calculate_highlight(*Qs)
            self.view.set_highlight_phi(phi)

    def get_normal(self):
        slice_plane = self.view.get_slice()

        if slice_plane == "Axis 1/2":
            norm = [0, 0, 1]
        elif slice_plane == "Axis 1/3":
            norm = [0, 1, 0]
        else:
            norm = [1, 0, 0]

        return norm

    def get_clim_method(self):
        ctype = self.view.get_clim_clip_type()

        if ctype == "μ±3×σ":
            method = "normal"
        elif ctype == "Q₃/Q₁±1.5×IQR":
            method = "boxplot"
        else:
            method = None

        return method

    def get_vlim_method(self):
        ctype = self.view.get_vlim_clip_type()

        if ctype == "μ±3×σ":
            method = "normal"
        elif ctype == "Q₃/Q₁±1.5×IQR":
            method = "boxplot"
        else:
            method = None

        return method

    def update_slice_extent(self):
        proj = self.view.get_projection_matrix()
        if proj is None:
            return
        U, V, W, invalid = self.model.validate_projection(proj)
        if invalid:
            return
        norm = self.get_normal()
        z_min, z_max = self.model.get_slice_z_extent(U, V, W, norm)
        if z_min is not None and z_max is not None and z_max > z_min:
            self.view.setup_slice_slider(z_min, z_max)

    def reslice(self):
        if self.model.is_sliced():
            self.convert_to_hkl()

    def convert_to_hkl(self):
        if self.slice_idle:
            self.slice_idle = False

            proj = self.view.get_projection_matrix()
            value = self.view.get_slice_value()
            thickness = self.view.get_slice_thickness()
            width = self.view.get_slice_width()
            norm = self.get_normal()
            clim_method = self.get_clim_method()
            slice_scale = self.view.get_slice_scale()
            slice_auto_limits = self.view.get_slice_auto_limits()
            vmin = self.view.get_vmin_value()
            vmax = self.view.get_vmax_value()

            worker = self.view.worker(
                functools.partial(
                    self.convert_to_hkl_process,
                    proj=proj,
                    value=value,
                    thickness=thickness,
                    width=width,
                    norm=norm,
                    clim_method=clim_method,
                    slice_scale=slice_scale,
                    slice_auto_limits=slice_auto_limits,
                    vmin=vmin,
                    vmax=vmax,
                )
            )
            worker.connect_result(self.convert_to_hkl_complete)
            worker.connect_finished(self.update_complete)
            worker.connect_progress(self.update_processing)

            self.view.start_worker_pool(worker)

    def convert_to_hkl_complete(self, result):
        if result is not None:
            self.slice_signal_cache = np.array(result["signal"], copy=True)
            self.view.update_slice(result)
        self.slice_idle = True

    def convert_to_hkl_process(
        self,
        progress=None,
        stop_event=None,
        proj=None,
        value=None,
        thickness=None,
        width=None,
        norm=None,
        clim_method=None,
        slice_scale=None,
        slice_auto_limits=None,
        vmin=None,
        vmax=None,
    ):
        if self.stop_processing(stop_event):
            return None

        validate = [proj, value, thickness, width]

        if all(elem is not None for elem in validate):
            U, V, W, invalid = self.model.validate_projection(proj)

            if not invalid:

                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                slice_histo = self.model.get_slice_info(
                    U, V, W, norm, value, thickness, width
                )

                progress("Updating slice...", 50)

                if slice_histo is not None:
                    signal = slice_histo["signal"]

                    slice_histo["vmin"], slice_histo["vmax"] = (
                        self._resolve_display_limits(
                            signal,
                            clim_method,
                            slice_scale,
                            slice_auto_limits,
                            vmin,
                            vmax,
                        )
                    )

                    progress("Slice drawn!", 100)

                    return slice_histo

                else:
                    progress("Invalid parameters.", 0)

        else:
            progress("Invalid parameters.", 0)

    def cluster(self):
        params = self.view.get_cluster_parameters()

        worker = self.view.worker(
            functools.partial(
                self.cluster_process,
                params=params,
            )
        )
        worker.connect_result(self.cluster_complete)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def cluster_complete(self, result):
        if result is not None:
            self.update_processing("Adding peaks.", 30)
            self.view.add_cluster_peaks(result)
            self.view.update_cluster_table(result)
            self.update_processing("Peaks added!", 0)

    def calculate_alignment(self):
        run_number = self.view.get_alignment_run()
        tilts = self.view.get_alignment_tilts()

        worker = self.view.worker(
            functools.partial(
                self.calculate_alignment_process,
                run_number=run_number,
                tilts=tilts,
            )
        )
        worker.connect_result(self.calculate_alignment_complete)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_alignment_complete(self, result):
        if result is not None:
            self.update_processing("Rendering alignment.", 75)
            self.view.add_alignment_peaks(result)
            self.update_processing("Alignment calculated!", 0)

    def calculate_alignment_process(
        self,
        progress=None,
        stop_event=None,
        run_number=None,
        tilts=None,
    ):
        if self.stop_processing(stop_event):
            return None

        if self.model.has_peaks() and self.model.has_UB():
            if run_number is not None and tilts is not None:
                progress("Calculating alignment.", 25)

                result = self.model.get_alignment_info(run_number, tilts)

                if result is not None:
                    progress("Alignment calculated.", 100)
                    return result

                progress("No indexed peaks for selected run.", 0)
            else:
                progress("Invalid alignment parameters.", 0)
        else:
            progress("Alignment requires indexed peaks and UB.", 0)

    def cluster_process(self, progress=None, stop_event=None, params=None):
        if self.stop_processing(stop_event):
            return None

        if params is not None:
            progress("Invalid parameters.", 0)

            peak_info = self.model.get_cluster_info()
            if peak_info is not None:
                if self.stop_processing(stop_event):
                    return None

                progress("Clustering peaks.", 25)

                success = self.model.cluster_peaks(peak_info, *params)

                if success:
                    progress("Peaks clustered!", 100)

                    return peak_info

                else:
                    progress("Invalid cluster.", 0)

        else:
            progress("Invalid parameters.", 0)
