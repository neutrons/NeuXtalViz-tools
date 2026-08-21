from NeuXtalViz.presenters.base_presenter import NeuXtalVizPresenter
from NeuXtalViz.config.instruments import LIVE_INSTRUMENTS

import functools
import numpy as np

LIVE_IDLE_TICK_LIMIT = 20
LIVE_IDLE_WARNING_SECONDS = 120


class UB(NeuXtalVizPresenter):
    """
    Presenter for the UB-matrix determination and refinement tools.

    Mediates between the UB tools view and model: wires the view's Qt
    signals to presenter handlers, converts UI events into model calls,
    and pushes model results back to the view. Covers Q-space
    conversion, peak finding/indexing/integration, UB matrix
    determination/transformation/refinement, lattice/cell selection,
    instrument and slice visualization, and peak alignment/clustering.
    """

    def __init__(self, view, model):
        """
        Initialize the presenter and connect all UB tools view signals
        to their corresponding presenter handler methods.

        Parameters
        ----------
        view : object
            The UB tools view/UI instance.
        model : object
            The UB tools model instance containing data and logic.
        """

        super(UB, self).__init__(view, model)

        self._live_signal = None
        self.live_convert_idle = True
        self.live_tick_count = 0
        self.live_idle_warning_active = False

        self.view.connect_live_toggle(self.toggle_live)

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
        self.view.connect_vlim_combo(self.update_instrument_clim)
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

        self.view.connect_clim_combo(self.update_slice_clim)
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
        """
        Refresh the peaks table and alignment run list in the view.

        Pulls the current peak information from the model (or an empty
        list if there are no peaks) and pushes it to the peaks table
        and the alignment run selector in the view.
        """

        peaks = self.model.get_peak_info() if self.model.has_peaks() else []
        self.view.update_peaks_table(peaks)
        self.view.update_alignment_runs(peaks)

    def _calculate_display_limits(self, data, method, scale):
        """
        Compute display color-limits for a signal array.

        Parameters
        ----------
        data : array_like
            Signal values to compute display limits for.
        method : str or None
            Clipping method passed to the model's ``calculate_clim``
            (e.g. ``"normal"``, ``"boxplot"``, or ``None``).
        scale : str
            Display scale, ``"log"`` or ``"linear"``. When ``"log"``
            and the computed minimum is non-positive or non-finite,
            the minimum is recomputed from the smallest positive value.

        Returns
        -------
        vmin : float
            Lower display limit.
        vmax : float
            Upper display limit.
        """

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
        """
        Resolve the display color-limits to use, honoring manual limits.

        If auto-limits are disabled and valid manual limits are
        supplied, those are used (adjusted for log scale if needed);
        otherwise the limits are recomputed from the data.

        Parameters
        ----------
        data : array_like
            Signal values to compute display limits for.
        method : str or None
            Clipping method passed to ``_calculate_display_limits``.
        scale : str
            Display scale, ``"log"`` or ``"linear"``.
        auto_limits : bool
            Whether display limits should be automatically computed.
        current_vmin : float or None
            Current manually-set lower limit, if any.
        current_vmax : float or None
            Current manually-set upper limit, if any.

        Returns
        -------
        vmin : float
            Resolved lower display limit.
        vmax : float
            Resolved upper display limit.
        """

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
        """
        Update the slice colorbar limits from the view's vmin/vmax
        line-edit values.

        Connected to the slice view's vmin/vmax line-edit signals.
        Manually editing either value disables automatic color
        limits. Adjusts the lower limit upward when it is
        non-positive and the slice scale is logarithmic.
        """

        self.view.slice_auto_limits_box.setChecked(False)

        vmin = self.view.get_vmin_value()
        vmax = self.view.get_vmax_value()

        if vmin is not None and vmax is not None:
            if vmin < vmax:
                if vmin <= 0 and self.view.get_slice_scale() == "log":
                    vmin = vmax / 10
                self.view.update_colorbar_vlims(vmin, vmax)

    def update_inst_cvals(self):
        """
        Update the instrument view colorbar limits from the view's
        vmin/vmax line-edit values.

        Connected to the instrument view's vmin/vmax line-edit
        signals. Manually editing either value disables automatic
        color limits. Adjusts the lower limit upward when it is
        non-positive and the instrument scale is logarithmic.
        """

        self.view.instrument_auto_limits_box.setChecked(False)

        vmin = self.view.get_inst_vmin_value()
        vmax = self.view.get_inst_vmax_value()

        if vmin is not None and vmax is not None:
            if vmin < vmax:
                if vmin <= 0 and self.view.get_instrument_scale() == "log":
                    vmin = vmax / 10
                self.view.update_instrument_colorbar_vlims(vmin, vmax)

    def update_instrument_view_autoscaled(self):
        """
        Refresh the instrument view, clearing manual color limits when
        auto-limits are enabled.

        Connected to the instrument view's vlim-combo signal. If
        auto-limits are enabled, clears the manual vmin/vmax line
        edits (blocking their signals while doing so), then redraws
        the instrument display from the cached signal, or recomputes
        the instrument view if no cache is available.
        """

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
        """
        Handle selection of a new run in the data list combo box.

        Connected to the view's data-list combo box signal. Invalidates
        the cached instrument signal so the instrument view is
        recomputed from scratch, then refreshes the (auto-scaled)
        instrument view.
        """

        self.inst_signal_cache = None
        self.update_instrument_view_autoscaled()

    def update_instrument_clim(self):
        """
        Refresh the instrument display using the current color limits.

        Connected to the instrument view's vlim-combo signal.
        Selecting a method re-enables automatic color limits.
        """

        self.view.instrument_auto_limits_box.blockSignals(True)
        self.view.instrument_auto_limits_box.setChecked(True)
        self.view.instrument_auto_limits_box.blockSignals(False)

        self.update_instrument_view_autoscaled()

    def update_instrument_display(self):
        """
        Redraw the instrument view display using the cached signal.

        Connected to the instrument scale/colormap/colorbar combo box
        and auto-limits signals. If no signal is cached yet, triggers
        a full instrument view recomputation instead. Otherwise
        resolves the display limits and updates the view's colormap,
        scale, and color limits.
        """

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
        """
        Refresh the slice display using the current color limits.

        Connected to the slice view's color-limit method combo box.
        Selecting a method re-enables automatic color limits.
        """

        self.view.slice_auto_limits_box.blockSignals(True)
        self.view.slice_auto_limits_box.setChecked(True)
        self.view.slice_auto_limits_box.blockSignals(False)

        self.update_slice_display()

    def update_slice_display(self):
        """
        Redraw the HKL slice display using the cached signal.

        Connected to the slice scale/colormap/colorbar combo box and
        auto-limits signals. If no signal is cached yet, triggers a
        full reslice instead. Otherwise resolves the display limits
        and updates the view's colormap, scale, and color limits.
        """

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
        Update the find peaks distance (Q) in the view from the
        d-spacing value.

        Connected to the find-peaks d-spacing line edit signal.
        """
        d = self.view.get_find_peaks_spacing()
        Q = self.model.get_Q(d)
        self.view.set_find_peaks_distance(Q)

    def update_find_distance(self):
        """
        Update the find peaks d-spacing in the view from the distance
        (Q) value.

        Connected to the find-peaks distance (Q) line edit signal.
        """
        Q = self.view.get_find_peaks_distance()
        d = self.model.get_d(Q)
        self.view.set_find_peaks_spacing(d)

    def hand_index_fractional(self):
        """
        Recompute integer (h,k,l,m,n,p) indices from a manually edited
        fractional hkl and update the selected peak and view.

        Connected to the h/k/l fractional index line edit signals.
        Reads the modulation vectors, the current hkl/integer-hkl/mnp
        indices, and the selected peak row from the view; if all are
        available, recalculates the integer indices, stores them on
        the selected peak in the model, and updates the peaks table
        and index fields in the view.
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
        Recompute a fractional hkl from manually edited integer
        (h,k,l,m,n,p) indices and update the selected peak and view.

        Connected to the integer h/k/l/m/n/p index line edit signals.
        Reads the modulation vectors, the current hkl/integer-hkl/mnp
        indices, and the selected peak row from the view; if all are
        available, recalculates the fractional hkl, stores it on the
        selected peak in the model, and updates the peaks table and
        index fields in the view.
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
        """
        Load raw event/histogram data and convert it to Q-space in a
        background worker.

        Connected to the "convert Q" button signal. Gathers the
        instrument, wavelength, calibration files, IPTS/run/experiment
        numbers, Lorentz-correction flag, time-stop cutoff, and
        minimum d-spacing from the view, then dispatches
        ``convert_Q_process`` to a worker thread whose result updates
        the run list and triggers a visualization refresh.

        Parameters
        ----------
        force_reload : bool, optional
            Whether to force reloading/reconverting data that has
            already been loaded (default is False). True when invoked
            via ``convert_Q_reload``.
        """

        if self.view.get_live():
            if self.model.is_live():
                self.stop_live()
                self.view.set_convert_button_text("Start Live")
            else:
                self.start_live()
                self.view.set_convert_button_text("Stop Live")
            return

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
        """
        Reconvert data to Q-space, forcing a reload of already-loaded
        runs.

        Connected to the "reload/reconvert Q" button signal.
        """

        self.convert_Q(force_reload=True)

    def toggle_live(self, checked):
        """
        React to the "Live" checkbox being toggled.

        Connected to the live checkbox's toggled signal. Only adjusts
        the run-entry fields and the Convert button's label; starting
        or stopping the listener itself happens on the next Convert
        button click (handled in ``convert_Q``).

        Parameters
        ----------
        checked : bool
            New checkbox state.
        """

        self.view.set_run_entry_enabled(not checked)

        if checked:
            self.view.set_convert_button_text("Start Live")
        else:
            if self.model.is_live():
                self.stop_live()
            self.view.set_convert_button_text("Convert")

    def start_live(self):
        """
        Start live-data streaming for the selected instrument.

        Dispatches ``model.start_live_data`` to a worker thread (the
        underlying Mantid call blocks briefly for the first chunk),
        passing a signal whose ``updated`` emission -- marshalled onto
        the GUI thread -- is connected to ``_on_live_update``.
        """

        instrument = self.view.get_instrument()

        self.live_tick_count = 0
        self.live_idle_warning_active = False

        live_signal = self.view.live_signal()
        self.view.connect_live_signal(live_signal, self._on_live_update)
        self._live_signal = live_signal

        worker = self.view.worker(
            functools.partial(
                self._start_live_process,
                instrument=instrument,
                on_update=live_signal.updated.emit,
            )
        )
        self.view.start_worker_pool(worker)

    def _start_live_process(
        self, progress=None, stop_event=None, instrument=None, on_update=None
    ):
        """
        Worker task that starts the live-data listener.

        Parameters
        ----------
        progress, stop_event
            Injected by the worker infrastructure; unused here since
            starting the listener is a single quick call.
        instrument : str
            Instrument identifier.
        on_update : callable
            Forwarded to ``model.start_live_data`` as the per-chunk
            callback.
        """

        self.model.start_live_data(
            instrument, update_every=120, on_update=on_update
        )

    def stop_live(self):
        """
        Stop live-data streaming.

        Leaves whatever has already been merged into "md"/"Q3D" in
        place; only the listener itself is cancelled.
        """

        self.model.stop_live_data()

        if self._live_signal is not None:
            self._live_signal.updated.disconnect(self._on_live_update)
            self._live_signal = None

    def _on_live_update(self, name):
        """
        Handle a live-data chunk landing.

        Connected to the live signal's ``updated`` emission (delivered
        on the GUI thread). Reconverts the in-progress live run and
        rebuilds "md"/"Q3D" in a worker thread, then refreshes the
        instrument view and visualization exactly as a manual Convert
        click does -- peaks and the UB matrix are only redisplayed, not
        recomputed. No-ops if the previous tick's conversion is still
        running, so ticks never overlap.

        Parameters
        ----------
        name : str
            Name of the snapshot workspace for this chunk (unused --
            the model tracks it internally; conversion always reads
            the current snapshot).
        """

        if not self.live_convert_idle:
            return

        self.live_convert_idle = False

        instrument = self.view.get_instrument()
        wavelength = self.view.get_wavelength()
        tube_cal = self.view.get_tube_calibration()
        det_cal = self.view.get_detector_calibration()
        gon_cal = self.view.get_goniometer_calibration()
        lorentz = self.view.get_lorentz()
        d_min = self.view.get_convert_min_d()

        worker = self.view.worker(
            functools.partial(
                self._live_convert_process,
                instrument=instrument,
                wavelength=wavelength,
                tube_cal=tube_cal,
                det_cal=det_cal,
                gon_cal=gon_cal,
                lorentz=lorentz,
                d_min=d_min,
            )
        )
        worker.connect_finished(self._live_convert_complete)
        self.view.start_worker_pool(worker)

    def _live_convert_process(
        self,
        progress=None,
        stop_event=None,
        instrument=None,
        wavelength=None,
        tube_cal=None,
        det_cal=None,
        gon_cal=None,
        lorentz=None,
        d_min=None,
    ):
        """
        Worker task that reconverts the live run into "md"/"Q3D".

        Applies the goniometer/calibration to the live workspace (as
        ``calibrate_data`` does for loaded files) before reconverting.
        """

        self.model.calibrate_data(instrument, det_cal, gon_cal, tube_cal)
        self.model.convert_data(
            instrument, wavelength, lorentz, d_min, reset_peaks=False
        )

    def _live_convert_complete(self):
        """
        Refresh the instrument view and visualization after a live tick.

        Populates the run-selector list first (a manual Convert does
        this via ``convert_Q_complete``; live ticks never go through
        that path) so ``update_instrument_view`` has a valid selection
        to render instead of ``None``.

        After every `LIVE_IDLE_TICK_LIMIT` ticks, warns the user that
        live data is still accumulating unattended -- left unbounded,
        a long-running/idle session can grow the live workspace large
        enough to strain memory -- and auto-stops it unless they choose
        to keep going (see `_warn_live_idle`).
        """

        self.live_convert_idle = True

        runs = self.model.get_number_workspaces()
        if runs is not None:
            self.view.set_data_list(runs)

        self.update_instrument_view()
        self.visualize(refresh_peaks=False)

        self.live_tick_count += 1
        if (
            self.live_tick_count >= LIVE_IDLE_TICK_LIMIT
            and not self.live_idle_warning_active
        ):
            self._warn_live_idle()

    def _warn_live_idle(self):
        """
        Warn that live data has run unattended and stop it if ignored.

        Shows a modal countdown dialog; if the user hasn't responded
        within `LIVE_IDLE_WARNING_SECONDS`, the listener is stopped
        automatically (whatever has already been merged into "md"/"Q3D"
        is kept, as with a manual stop). Choosing to keep it running
        resets the tick counter so the warning can fire again later.

        Set as a no-op re-entry guard (`live_idle_warning_active`)
        while showing, since further ticks keep landing and calling
        `_live_convert_complete` while the dialog is up.
        """

        self.live_idle_warning_active = True
        keep_going = self.view.show_live_idle_warning(
            LIVE_IDLE_WARNING_SECONDS
        )
        self.live_idle_warning_active = False

        if keep_going:
            self.live_tick_count = 0
        elif self.model.is_live():
            self.stop_live()
            self.view.set_live_checked(False)

    def convert_Q_complete(self, result):
        """
        Handle the result of the Q-space conversion worker.

        Parameters
        ----------
        result : dict or None
            Dictionary with keys ``"runs"`` (list of loaded/converted
            run workspace names) and ``"mono"`` (whether the data is
            monochromatic), or None if conversion failed or produced
            no output.
        """

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
        """
        Worker task that loads, calibrates, and converts data to
        Q-space.

        Parameters
        ----------
        progress : callable or None
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None
            Event used to signal that processing should stop early.
        instrument : str or None
            Instrument name.
        wavelength : float, tuple, or None
            Incident wavelength (or wavelength band for
            time-of-flight instruments).
        tube_cal : str or None
            Path to a tube calibration file.
        det_cal : str or None
            Path to a detector calibration file.
        gon_cal : str or None
            Path to a goniometer calibration file.
        IPTS : int or None
            IPTS proposal number.
        runs : str or None
            Run number(s) to load.
        exp : int or None
            Experiment number (required for DEMAND).
        lorentz : bool or None
            Whether to apply the Lorentz correction during conversion.
        time_stop : float or None
            Time cutoff used to filter events during loading.
        d_min : float or None
            Minimum d-spacing used when converting data to Q-space.
        force_reload : bool, optional
            Whether to force reloading/reconverting already-loaded
            data (default is False).

        Returns
        -------
        result : dict or None
            Dictionary with keys ``"mono"`` (bool, whether the data is
            monochromatic) and ``"runs"`` (int, number of loaded
            workspaces), or None if processing was stopped, the
            required data files do not exist, or the input parameters
            are invalid.
        """

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
                progress=progress,
                stop_event=stop_event,
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
                progress=progress,
                stop_event=stop_event,
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
        """
        Add a peak at the current detector position to the model.

        Connected to the "add peak" button signal. Reads the selected
        run, horizontal/vertical detector coordinates, and diffraction
        (e.g. time-of-flight or wavelength) value from the view, adds
        the corresponding peak via the model, and refreshes the
        visualization.
        """

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
        """
        Add a peak at the given hkl position from the slice view.

        Connected to the view's ``slice_ready`` signal, emitted when
        the user picks a point on the HKL slice plot.

        Parameters
        ----------
        h : float
            H index of the selected slice position.
        k : float
            K index of the selected slice position.
        l : float
            L index of the selected slice position.
        """

        if self.model.has_Q() and self.model.has_UB():
            ind = self.view.get_data_list()

            if ind is not None:
                self.model.add_peak_from_hkl(ind, [h, k, l])
                self.visualize()
            else:
                self.update_processing("Invalid data list index.", 0)

    def delete_peak(self):
        """
        Delete the currently selected peak(s) from the model.

        Connected to the "delete peak" button signal. Removes the
        highlighted peak rows from the model's peaks table, clears the
        selection in the view, and refreshes the visualization.
        """

        peaks = self.view.get_peaks()

        if self.model.has_peaks() and len(peaks) > 0:
            self.model.delete_peak_rows(peaks)
            self.view.clear_peak_selection()
            self.visualize()
        else:
            self.update_processing("No highlighted peaks selected.", 0)

    def calculate_hkl(self):
        """
        Compute the detector position corresponding to a given hkl.

        Connected to the "check hkl" signal. Reads the selected run
        and target hkl from the view, asks the model for the
        diffraction value and horizontal/vertical detector
        coordinates for that hkl, and updates the corresponding view
        fields and instrument display.
        """

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
        """
        Recompute and redraw the instrument detector view in a
        background worker.

        Connected to the d_min/d_max range signals (and invoked after
        Q conversion, run selection, and hkl lookups). No-ops if a
        prior instrument view computation is still in flight. Gathers
        the selected run, d-spacing range, horizontal/vertical
        position and ROI, diffraction value, and display-limit
        settings from the view, then dispatches
        ``update_instrument_view_process`` to a worker thread.
        """

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
        """
        Handle the result of the instrument view computation worker.

        Parameters
        ----------
        result : tuple or None
            Tuple of ``(inst_view, roi_view)`` dictionaries as produced
            by the model's ``inst_view``/``roi_view`` state, or None if
            the computation was stopped or the parameters were
            invalid. ``inst_view["img"]`` is cached for redisplaying
            without recomputation.
        """

        self.instrument_view_idle = True
        if result is not None:
            self.inst_signal_cache = np.array(result[0]["img"], copy=True)
            self.view.update_instrument_view(result[0])
            self.view.update_roi_view(result[1])
            self.view.update_scan_view(result[1])
            self.update_run_goniometer()

            self.update_check_hkl()

    def update_run_goniometer(self):
        """
        Update the displayed goniometer setting for the selected run.
        """

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
        """
        Worker task that computes the instrument detector view and ROI.

        Parameters
        ----------
        progress : callable or None
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None
            Event used to signal that processing should stop early.
        ind : int or None
            Index of the selected run in the data list.
        d_min : float or None
            Minimum d-spacing to include in the instrument view.
        d_max : float or None
            Maximum d-spacing to include in the instrument view.
        horz : float or None
            Horizontal detector coordinate for the ROI center.
        vert : float or None
            Vertical detector coordinate for the ROI center.
        horz_roi : float or None
            Horizontal ROI half-width.
        vert_roi : float or None
            Vertical ROI half-width.
        val : float or None
            Diffraction (time-of-flight/wavelength) value for the ROI.
        vlim_method : str or None
            Clipping method used to resolve display color limits.
        instrument_scale : str or None
            Display scale (``"log"`` or ``"linear"``) for the
            instrument view.
        instrument_auto_limits : bool or None
            Whether display limits should be automatically computed.
        inst_vmin : float or None
            Manually-set lower display limit.
        inst_vmax : float or None
            Manually-set upper display limit.

        Returns
        -------
        inst_view : dict
            The model's instrument view data (image, extents, and
            resolved display limits).
        roi_view : dict
            The model's extracted region-of-interest data, or None if
            processing was stopped or the parameters were invalid.
        """

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
        """
        Recompute and redraw the region-of-interest (ROI) view.

        Connected to the horizontal/vertical position and ROI-width
        line edit signals. No-ops while an instrument view computation
        is in flight. Extracts the ROI from the model using the
        current horizontal/vertical position and widths, updates the
        ROI view, and refreshes the checked hkl display.
        """

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
        """
        Recompute and redraw the ROI scan (integrated profile) view.

        Connected to the view's ``roi_ready`` signal, emitted after
        the ROI plot has been redrawn. No-ops while an instrument view
        computation is in flight. Extracts the ROI from the model and
        updates the scan view and checked hkl display.
        """

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
        """
        Recompute and display the hkl corresponding to the current ROI
        scan position.

        Connected to the view's ``scan_ready`` signal, emitted after
        the scan plot has been redrawn.
        """

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
        """
        Refresh the Q/peaks/UB status indicators and undo-filter
        availability in the view.

        Checks whether the expected data files exist for the current
        instrument/IPTS/run(s)/experiment selection and updates the
        Q-data, peaks, and UB status indicators, as well as whether
        the "undo filter peaks" action is enabled.
        """

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

    def visualize(self, refresh_peaks=True):
        """
        Refresh the 3D Q-space visualization and dependent UI state.

        Called after data-modifying operations complete (e.g. Q
        conversion, peak finding/indexing/integration, UB
        determination). No-ops if there is no Q data or a
        visualization update is already in progress. Updates the data
        status, redraws the Q-space volume, refreshes UB/lattice
        information if a UB matrix is set, and refreshes the peaks
        table.

        Parameters
        ----------
        refresh_peaks : bool, optional
            Whether to redisplay the peaks table (default True). Live
            ticks pass False -- peaks are only ever added by explicit
            find/index actions, not by a live tick, so repopulating the
            table every tick just resets the user's selection/scroll
            position for no reason.
        """

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

            if refresh_peaks:
                self.refresh_peak_views()

            self.update_complete("Data visualized!")

            self.volume_idle = True

    def update_lattice_info(self):
        """
        Refresh the lattice constants and sample-direction displays in
        the view from the current UB matrix.
        """

        params = self.model.get_lattice_constants()
        errors = self.model.get_lattice_constant_errors()

        if params is not None:
            self.view.set_lattice_constants(params, errors)

        params = self.model.get_sample_directions()

        if params is not None:
            self.view.set_sample_directions(params)

    def find_peaks(self):
        """
        Search for peaks in Q-space in a background worker.

        Connected to the "find peaks" button signal. Gathers the
        minimum Q, maximum d-spacing, peak-finding parameters, edge
        exclusion, peak width, and contamination-avoidance flags
        (aluminum/copper/iron) from the view, then dispatches
        ``find_peaks_process`` to a worker thread whose completion
        copies the resulting peaks into the UB and refreshes the
        visualization.
        """

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
        """
        Handle completion of the peak-finding worker.

        Parameters
        ----------
        result : None
            Unused; ``find_peaks_process`` reports progress but does
            not return a value.
        """

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
        """
        Worker task that finds peaks and optionally removes peaks from
        known contaminant phases.

        Parameters
        ----------
        progress : callable or None
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None
            Event used to signal that processing should stop early.
        Q_min : float or None
            Minimum |Q| distance between peaks used for peak finding.
        d_max : float or None
            Maximum d-spacing considered for contamination avoidance.
        params : tuple or None
            Peak-finding parameters passed to the model's
            ``find_peaks``.
        edge : int or float or None
            Number of pixels/bins to exclude from the detector edge.
        peak_width : float or None
            Peak width used for contamination avoidance.
        no_al : bool or None
            Whether to remove peaks matching aluminum contamination.
        no_cu : bool or None
            Whether to remove peaks matching copper contamination.
        no_fe : bool or None
            Whether to remove peaks matching iron contamination.
        """

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
        """
        Determine the UB matrix from known lattice parameters.

        Connected to the "find conventional cell" button signal.
        Gathers the lattice constants and indexing tolerance from the
        view, then dispatches ``find_conventional_process`` to a
        worker thread whose completion refreshes the data status and
        visualization.
        """

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
        """
        Handle completion of the conventional-cell UB worker.

        Parameters
        ----------
        result : None, optional
            Unused; ``find_conventional_process`` reports progress but
            does not return a value.
        """

        self.update_data_status()

    def find_conventional_process(
        self, progress=None, stop_event=None, params=None, tol=None
    ):
        """
        Worker task that determines the UB matrix from lattice
        parameters.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        params : tuple or None, optional
            Lattice constants ``(a, b, c, alpha, beta, gamma)`` as
            returned by the view, or None if the input fields are
            invalid.
        tol : float or None, optional
            Indexing tolerance, or None if the input field is invalid.
        """

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
        """
        Determine the UB matrix from a Niggli (reduced primitive) cell.

        Connected to the "find niggli cell" button signal. Gathers the
        minimum/maximum lattice constants and indexing tolerance from
        the view, then dispatches ``find_niggli_process`` to a worker
        thread whose completion lists the possible conventional cells
        and refreshes the visualization.
        """

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
        """
        Handle completion of the Niggli-cell UB worker.

        Parameters
        ----------
        result : None
            Unused; ``find_niggli_process`` reports progress but does
            not return a value.
        """

        self.show_cells()

    def find_niggli_process(
        self, progress=None, stop_event=None, params=None, tol=None
    ):
        """
        Worker task that determines the UB matrix from a Niggli cell.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        params : tuple or None, optional
            ``(min_d, max_d)`` minimum and maximum lattice constants as
            returned by the view, or None if the input fields are
            invalid.
        tol : float or None, optional
            Indexing tolerance, or None if the input field is invalid.
        """

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
        """
        List possible conventional cells compatible with the current
        Niggli cell.

        Gathers the maximum scalar error tolerance from the view, then
        dispatches ``show_cells_process`` to a worker thread whose
        completion populates the cell table and refreshes the
        visualization.
        """

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
        """
        Handle completion of the possible-cells worker.

        Parameters
        ----------
        result : list or None
            Possible conventional cells returned by
            ``show_cells_process``, or None if the operation could not
            be performed.
        """

        if result is not None:
            self.view.update_cell_table(result)

    def show_cells_process(self, progress=None, stop_event=None, scalar=None):
        """
        Worker task that lists possible conventional cells.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        scalar : float or None, optional
            Maximum scalar error tolerance, or None if the input field
            is invalid.

        Returns
        -------
        cells : list
            Possible conventional cells found, returned only if
            peaks and a UB matrix are present and ``scalar`` is valid.
        """

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
        """
        Set the UB matrix manually from lattice constants and sample
        directions.

        Connected to the "Set UB" button signal. Reads the lattice
        constants and sample directions from the view and, if both are
        valid, sets the UB matrix on the model and refreshes the data
        status, lattice information, and oriented-lattice display.
        """

        constants = self.view.get_lattice_constants()
        directions = self.view.get_sample_directions()

        if constants is not None and directions is not None:
            self.model.set_manual_UB(constants, directions)
            self.update_data_status()
            self.update_lattice_info()
            self.update_oriented_lattice()

    def set_UB_from_scattering_plane(self):
        """
        Determine the UB matrix from a scattering plane and lattice
        constants.

        Connected to the "Search U from Scattering Plane" button
        signal. Gathers the lattice constants and scattering-plane
        sample directions from the view, then dispatches
        ``set_UB_from_scattering_plane_process`` to a worker thread
        whose completion refreshes the data status and visualization.
        """

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
        """
        Handle completion of the scattering-plane UB worker.

        Parameters
        ----------
        result : bool or None, optional
            True if ``set_UB_from_scattering_plane_process`` succeeded,
            otherwise None.
        """

        self.update_data_status()

    def set_UB_from_scattering_plane_process(
        self, progress=None, stop_event=None, constants=None, directions=None
    ):
        """
        Worker task that determines the UB matrix from a scattering
        plane.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        constants : list or None, optional
            Lattice constants and angles ``[a, b, c, alpha, beta,
            gamma]``, or None if the input fields are invalid.
        directions : list or None, optional
            Two non-parallel vectors ``[u1, u2, u3, v1, v2, v3]``
            defining the scattering plane, or None if the input fields
            are invalid.

        Returns
        -------
        success : bool
            True if the UB matrix was determined successfully,
            returned only on success.
        """

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
        """
        Transform to the conventional cell selected in the cell table.

        Connected to the cell-selection button signal. Gathers the
        form number and indexing tolerance from the view, then
        dispatches ``select_cell_process`` to a worker thread whose
        completion refreshes the visualization.
        """

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
        """
        Handle completion of the cell-selection worker.

        Parameters
        ----------
        result : None
            Unused; ``select_cell_process`` reports progress but does
            not return a value.
        """

        pass

    def select_cell_process(
        self, progress=None, stop_event=None, form=None, tol=None
    ):
        """
        Worker task that transforms the lattice to a conventional cell.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        form : int or None, optional
            Conventional cell form number, or None if the input field
            is invalid.
        tol : float or None, optional
            Indexing tolerance, or None if the input field is invalid.
        """

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
        """
        Sync the selected cell-table row to the form-number field.

        Connected to the cell table's selection-changed signal. Reads
        the form number of the currently selected row and writes it
        into the form-number input field.
        """

        form = self.view.get_form()
        self.view.set_cell_form(form)

    def highlight_peak(self):
        """
        Highlight the peak(s) selected in the peaks table.

        Connected to the peaks table's selection-changed signal. If
        one or more rows are selected, looks up the first selected
        peak, updates the peak-information display, highlights all
        selected peaks in the 3D view, and centers the view on the
        first peak's position. Clears the highlight if no rows are
        selected.
        """

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
        """
        Populate the symmetry-operation choices for the selected
        lattice system.

        Connected to the lattice-system combo box signal. Generates
        the lattice transforms compatible with the selected lattice
        system, updates the symmetry-operation combo box with their
        symbols, and applies the currently selected symmetry
        transform.
        """

        cell = self.view.get_lattice_transform()

        Ts = self.model.generate_lattice_transforms(cell)

        self.view.update_symmetry_symbols(list(Ts.keys()))

        self.symmetry_transform()

    def symmetry_transform(self):
        """
        Display the transform matrix for the selected symmetry
        operation.

        Connected to the symmetry-operation combo box signal.
        Regenerates the lattice transforms for the selected lattice
        system and, if the selected symmetry symbol is among them,
        writes the corresponding transform matrix into the view.
        """

        cell = self.view.get_lattice_transform()

        Ts = self.model.generate_lattice_transforms(cell)

        symbol = self.view.get_symmetry_symbol()

        if symbol in Ts.keys():
            T = Ts[symbol]

            self.view.set_transform_matrix(T)

    def transform_UB(self):
        """
        Apply a lattice transformation matrix to the UB matrix.

        Connected to the "Transform" button signal. Gathers the
        transform matrix and indexing tolerance from the view, then
        dispatches ``transform_UB_process`` to a worker thread whose
        completion copies the resulting UB back from the peaks and
        refreshes the visualization.
        """

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
        """
        Handle completion of the UB-transformation worker.

        Parameters
        ----------
        result : None
            Unused; ``transform_UB_process`` reports progress but does
            not return a value.
        """

        self.model.copy_UB_from_peaks()

    def transform_UB_process(
        self, progress=None, stop_event=None, params=None, tol=None
    ):
        """
        Worker task that applies a lattice transformation to the UB
        matrix.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        params : 3x3 array-like or None, optional
            Transform matrix to apply to the hkl values, or None if
            the input fields are invalid.
        tol : float or None, optional
            Indexing tolerance, or None if the input field is invalid.
        """

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
        """
        Refine the UB matrix using the selected refinement option.

        Connected to the "optimize UB" button signal. Gathers the
        lattice constants, refinement tolerance, and refinement option
        (e.g. constrained/unconstrained/lattice-system-constrained)
        from the view, then dispatches ``refine_UB_process`` to a
        worker thread whose completion copies the resulting UB back
        from the peaks and refreshes the visualization.
        """

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
        """
        Handle completion of the UB-refinement worker.

        Parameters
        ----------
        result : None
            Unused; ``refine_UB_process`` reports progress but does
            not return a value.
        """

        self.model.copy_UB_from_peaks()

    def refine_UB_process(
        self,
        progress=None,
        stop_event=None,
        params=None,
        tol=None,
        option=None,
    ):
        """
        Worker task that refines the UB matrix.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        params : tuple or None, optional
            Lattice constants ``(a, b, c, alpha, beta, gamma)``, used
            only when ``option`` is ``"Constrained"``, or None if the
            input fields are invalid.
        tol : float or None, optional
            Indexing tolerance, or None if the input field is invalid.
        option : str or None, optional
            Refinement option: ``"Constrained"`` to refine orientation
            only from ``params``, ``"Unconstrained"`` to refine the UB
            without lattice-system constraints, or a lattice system
            name to refine the UB with that lattice-system constraint.
        """

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
        """
        Gather modulation-vector and satellite-indexing settings from
        the view.

        Returns
        -------
        mod_vec_1, mod_vec_2, mod_vec_3 : list
            Modulation offset vectors ``[dh, dk, dl]``.
        max_order : int
            Maximum satellite order, 0 if the max-order/cross-terms
            input is invalid.
        cross_terms : bool
            Whether to include modulation cross terms, False if the
            max-order/cross-terms input is invalid.
        """

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
        """
        Index the peaks table against the current UB matrix.

        Connected to the "index peaks" button signal. Gathers
        modulation info, indexing tolerances, whether to index
        satellite peaks, and whether to round hkl values from the
        view, then dispatches ``index_peaks_process`` to a worker
        thread whose completion copies the resulting UB back from the
        peaks and refreshes the visualization.
        """

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
        """
        Handle completion of the peak-indexing worker.

        Parameters
        ----------
        result : None
            Unused; ``index_peaks_process`` reports progress but does
            not return a value.
        """

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
        """
        Worker task that indexes the peaks table.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        mod_info : tuple or None, optional
            ``(mod_vec_1, mod_vec_2, mod_vec_3, max_order,
            cross_terms)`` as returned by ``get_modulation_info``.
        params : tuple or None, optional
            ``(tol, sat_tol)`` indexing and satellite-indexing
            tolerances, or None if the input fields are invalid.
        sat : bool or None, optional
            Whether to index satellite peaks; if False, ``max_order``
            is forced to 0.
        round_hkl : bool or None, optional
            Whether to round indexed hkl values to integers.
        """

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
        """
        Predict peak positions from the UB matrix and lattice
        centering.

        Connected to the "predict peaks" button signal. Gathers
        modulation info, lattice centering, wavelength band,
        d-spacing/edge parameters, and whether to predict satellite
        peaks from the view, then dispatches
        ``predict_peaks_process`` to a worker thread whose completion
        copies the resulting UB back from the peaks and refreshes the
        visualization.
        """

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
        """
        Handle completion of the peak-prediction worker.

        Parameters
        ----------
        result : None
            Unused; ``predict_peaks_process`` reports progress but
            does not return a value.
        """

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
        """
        Worker task that predicts peak positions from the UB matrix.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        mod_info : tuple or None, optional
            ``(mod_vec_1, mod_vec_2, mod_vec_3, max_order,
            cross_terms)`` as returned by ``get_modulation_info``.
        centering : str or None, optional
            Lattice centering symbol used to apply reflection
            conditions.
        wavelength : tuple or None, optional
            ``(lamda_min, lamda_max)`` wavelength band in angstroms,
            or None if the input fields are invalid.
        params : tuple or None, optional
            ``(d_min, sat_d_min)`` minimum d-spacing for fundamental
            and satellite peaks, or None if the input fields are
            invalid.
        predict_sat : bool or None, optional
            Whether to also predict satellite peak positions.
        edge : int or float or None, optional
            Number of pixels/bins to exclude from the detector edge.
        """

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
        """
        Integrate peak intensities in Q-space.

        Connected to the "integrate peaks" button signal. Gathers the
        integration radius/background factors, whether to use
        ellipsoidal integration, and whether to centroid peaks first
        from the view, then dispatches ``integrate_peaks_process`` to
        a worker thread whose completion copies the resulting UB back
        from the peaks and refreshes the visualization.
        """

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
        """
        Handle completion of the peak-integration worker.

        Parameters
        ----------
        result : None
            Unused; ``integrate_peaks_process`` reports progress but
            does not return a value.
        """

        self.model.copy_UB_from_peaks()

    def integrate_peaks_process(
        self,
        progress=None,
        stop_event=None,
        params=None,
        ellipsoid=None,
        centroid=None,
    ):
        """
        Worker task that integrates peak intensities.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        params : tuple or None, optional
            ``(rad, inner_factor, outer_factor)`` integration radius
            and background shell factors, or None if the input fields
            are invalid.
        ellipsoid : bool or None, optional
            Whether to use ellipsoidal (True) or spherical (False)
            integration regions.
        centroid : bool or None, optional
            Whether to shift peak positions to their centroid before
            integrating (spherical method only).
        """

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
        """
        Filter the peaks table by a variable, comparison operator, and
        value.

        Connected to the "filter peaks" button signal. Gathers the
        filter variable, comparison operator, and threshold value from
        the view, then dispatches ``filter_peaks_process`` to a worker
        thread whose completion copies the resulting UB back from the
        peaks and refreshes the visualization.
        """

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
        """
        Handle completion of the peak-filtering worker.

        Parameters
        ----------
        result : None
            Unused; ``filter_peaks_process`` reports progress but does
            not return a value.
        """

        self.model.copy_UB_from_peaks()

    def filter_peaks_process(
        self,
        progress=None,
        stop_event=None,
        name=None,
        operator=None,
        value=None,
    ):
        """
        Worker task that filters the peaks table.

        Snapshots the current peaks table before filtering so the
        operation can be undone.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        name : str or None, optional
            Name of the peak variable to filter on.
        operator : str or None, optional
            Comparison operator used to filter peaks.
        value : float or None, optional
            Threshold value peaks are compared against, or None if the
            input field is invalid.
        """

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
        """
        Restore the peaks table to its state before the last filter
        operation.

        Connected to the "undo filter" button signal. Dispatches
        ``undo_filter_peaks_process`` to a worker thread whose
        completion copies the resulting UB back from the peaks and
        refreshes the visualization.
        """

        worker = self.view.worker(self.undo_filter_peaks_process)
        worker.connect_result(self.undo_filter_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def undo_filter_peaks_complete(self, result):
        """
        Handle completion of the undo-filter worker.

        Parameters
        ----------
        result : None
            Unused; ``undo_filter_peaks_process`` reports progress but
            does not return a value.
        """

        self.model.copy_UB_from_peaks()

    def undo_filter_peaks_process(self, progress=None, stop_event=None):
        """
        Worker task that restores the peaks table from the
        filter-peaks backup.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        """

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
        """
        Prompt for and set a detector calibration file.

        Connected to the "Browse" button signal for the detector
        calibration field. Opens a file dialog rooted at the
        instrument's calibration directory and, if a file is chosen,
        writes its path into the view.
        """

        inst = self.view.get_instrument()

        path = self.model.get_calibration_file_path(inst)

        filename = self.view.load_detector_cal_dialog(path)

        if filename:
            self.view.set_detector_calibration(filename)

    def load_tube_calibration(self):
        """
        Prompt for and set a tube calibration file.

        Connected to the "Browse" button signal for the tube
        calibration field. Opens a file dialog rooted at the
        instrument's calibration directory and, if a file is chosen,
        writes its path into the view.
        """

        inst = self.view.get_instrument()

        path = self.model.get_calibration_file_path(inst)

        filename = self.view.load_tube_cal_dialog(path)

        if filename:
            self.view.set_tube_calibration(filename)

    def load_goniometer_calibration(self):
        """
        Prompt for and set a goniometer calibration file.

        Connected to the "Browse" button signal for the goniometer
        calibration field. Opens a file dialog rooted at the
        instrument's calibration directory and, if a file is chosen,
        writes its path into the view.
        """

        inst = self.view.get_instrument()

        path = self.model.get_calibration_file_path(inst)

        filename = self.view.load_goniometer_cal_dialog(path)

        if filename:
            self.view.set_goniometer_calibration(filename)

    def load_Q(self):
        """
        Load a Q-sample workspace from a file selected by the user.

        Connected to the "Load Q" button signal. Opens a file dialog
        rooted at the instrument's shared directory and, if a file is
        chosen, loads it into the model, updates the minimum
        d-spacing, wavelength, and instrument selection fields in the
        view, refreshes the lattice information, and redraws the
        visualization.
        """

        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.load_Q_file_dialog(path)

        if filename:
            self.update_processing("Loading Q...", 1)
            d_min, wavelength, runs = self.model.load_Q(filename)
            self.update_processing("Q loaded...", 40)
            self.view.set_convert_min_d(d_min)
            self.view.set_d_min(d_min)
            if wavelength is not None:
                self.view.set_wavelength(wavelength)
            self.view.set_data_list(runs)
            inst = self.model.get_instrument_from_Q()
            if inst is not None:
                self.view.set_instrument(inst)
                self.switch_instrument()
            self.update_processing("Updating lattice info...", 60)
            self.update_lattice_info()
            self.view.set_Q_status(3)
            if self.model.has_peaks():
                self.update_processing("Refreshing peaks...", 80)
                self.refresh_peak_views()
            self.update_processing("Rendering...", 90)
            self.visualize()
            self.update_complete("Q loaded!")

    def save_Q(self):
        """
        Save the current Q-sample workspace to a file selected by the
        user.

        Connected to the "Save Q" button signal. Opens a file dialog
        rooted at the instrument's shared directory and, if a file is
        chosen, saves the Q-sample workspace there.
        """

        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_Q_file_dialog(path)

        if filename:
            self.model.save_Q(filename)

    def load_peaks(self):
        """
        Load a peaks table from a file selected by the user.

        Connected to the "Load Peaks" button signal. Opens a file
        dialog rooted at the instrument's shared directory and, if a
        file is chosen, loads it into the model and refreshes the
        peaks table and 3D peak views.
        """

        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.load_peaks_file_dialog(path)

        if filename:
            self.model.load_peaks(filename)
            self.refresh_peak_views()

    def save_peaks(self):
        """
        Save the current peaks table to a file selected by the user.

        Connected to the "Save Peaks" button signal. Opens a file
        dialog rooted at the instrument's shared directory and, if a
        file is chosen, saves the peaks table there.
        """

        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_peaks_file_dialog(path)

        if filename:
            self.model.save_peaks(filename)

    def load_UB(self):
        """
        Load a UB matrix from a file selected by the user.

        Connected to the "Load UB" button signal. Opens a file dialog
        rooted at the instrument's shared directory and, if a file is
        chosen, loads it into the model and updates the transform
        display in the view.
        """

        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.load_UB_file_dialog(path)

        if filename:
            self.model.load_UB(filename)

            self.view.set_transform(self.model.get_transform())

    def save_UB(self):
        """
        Save the current UB matrix to a file selected by the user.

        Connected to the "Save UB" button signal. Opens a file dialog
        rooted at the instrument's shared directory and, if a file is
        chosen, saves the UB matrix there.
        """

        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_UB_file_dialog(path)

        if filename:
            self.model.save_UB(filename)

    def save_roi_mask(self):
        """
        Save a detector mask XML file for the current instrument-view
        ROI.

        Connected to the "Save ROI Mask" button signal. Opens a file
        dialog rooted at the instrument's shared directory and, if a
        file is chosen, saves the mask there and reports the outcome
        message to the view.
        """

        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_mask_file_dialog(path)

        if filename:
            success, message = self.model.save_roi_mask(inst, filename)
            self.update_processing(message, 0)

    def switch_instrument(self):
        """
        Refresh instrument-dependent view state after the instrument
        selection changes.

        Connected to the instrument combo box signal. Updates the
        wavelength, clears run information, sets the goniometer axes
        and default minimum d-spacing fields for the newly selected
        instrument.
        """

        instrument = self.view.get_instrument()

        if self.model.is_live() and instrument != self.model.live_instrument:
            self.stop_live()
            self.view.set_live_checked(False)
            self.view.set_convert_button_text("Convert")
            self.view.set_run_entry_enabled(True)

        self.view.set_live_enabled(instrument in LIVE_INSTRUMENTS)

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
        """
        Sync the minimum wavelength value into the wavelength display.

        Connected to editing-finished on the minimum wavelength field.
        """

        wl_min, wl_max = self.view.get_wavelength()
        self.view.update_wavelength(wl_min)

    def calculate_peaks(self):
        """
        Calculate d-spacings and the interplanar angle for two entered
        hkl values.

        Connected to the hkl-calculation "Calculate" button signal.
        Reads the two hkl values and lattice constants from the view
        and, if the lattice constants are valid, updates the
        d-spacing/angle display.
        """

        hkl_1, hkl_2 = self.view.get_input_hkls()
        constants = self.view.get_lattice_constants()
        if constants is not None:
            d_phi = self.model.calculate_peaks(hkl_1, hkl_2, *constants)
            self.view.set_d_phi(*d_phi)

    def add_highlight_1(self):
        """
        Add the currently selected peak as the first highlighted peak.

        Connected to the first "Add Highlighted" button signal. Looks
        up the peak selected in the peaks table and, if found, adds it
        to the view as highlighted peak 1.
        """

        no = self.view.get_peak()
        if no is not None:
            peak = self.model.get_peak(no)
            if peak is not None:
                self.view.add_highlight_1(peak)

    def add_highlight_2(self):
        """
        Add the currently selected peak as the second highlighted
        peak.

        Connected to the second "Add Highlighted" button signal. Looks
        up the peak selected in the peaks table and, if found, adds it
        to the view as highlighted peak 2.
        """

        no = self.view.get_peak()
        if no is not None:
            peak = self.model.get_peak(no)
            if peak is not None:
                self.view.add_highlight_2(peak)

    def calculate_highlight(self):
        """
        Calculate the angle between the two highlighted peaks.

        Connected to the highlighted-peaks "Calculate" button signal.
        Reads the Q vectors of the two highlighted peaks from the view
        and, if both are set, updates the angle display.
        """

        Qs = self.view.get_highlight()
        if Qs is not None:
            phi = self.model.calculate_highlight(*Qs)
            self.view.set_highlight_phi(phi)

    def get_normal(self):
        """
        Get the slice normal vector for the selected axis pair.

        Returns
        -------
        norm : list
            Unit normal vector ``[1, 0, 0]``, ``[0, 1, 0]``, or
            ``[0, 0, 1]`` corresponding to the "Axis 2/3", "Axis 1/3",
            or "Axis 1/2" slice-plane selection, respectively.
        """

        slice_plane = self.view.get_slice()

        if slice_plane == "Axis 1/2":
            norm = [0, 0, 1]
        elif slice_plane == "Axis 1/3":
            norm = [0, 1, 0]
        else:
            norm = [1, 0, 0]

        return norm

    def get_clim_method(self):
        """
        Get the color-limit clipping method for the slice view.

        Returns
        -------
        method : str or None
            ``"normal"`` for mean +/- 3 standard deviations,
            ``"boxplot"`` for the interquartile-range method, or None
            if neither is selected (no clipping).
        """

        ctype = self.view.get_clim_clip_type()

        if ctype == "μ±3×σ":
            method = "normal"
        elif ctype == "Q₃/Q₁±1.5×IQR":
            method = "boxplot"
        else:
            method = None

        return method

    def get_vlim_method(self):
        """
        Get the color-limit clipping method for the instrument view.

        Returns
        -------
        method : str or None
            ``"normal"`` for mean +/- 3 standard deviations,
            ``"boxplot"`` for the interquartile-range method, or None
            if neither is selected (no clipping).
        """

        ctype = self.view.get_vlim_clip_type()

        if ctype == "μ±3×σ":
            method = "normal"
        elif ctype == "Q₃/Q₁±1.5×IQR":
            method = "boxplot"
        else:
            method = None

        return method

    def update_slice_extent(self):
        """
        Update the slice slider range for the selected projection and
        axis pair.

        Connected to the slice axis-pair combo box signal. Validates
        the projection matrix and, if valid, computes the extent along
        the slice normal and reconfigures the slice slider's range.
        """

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
        """
        Redraw the HKL slice if sliced data already exists.

        Connected to the slice-position/thickness/width field and
        slider signals.
        """

        if self.model.is_sliced():
            self.convert_to_hkl()

    def convert_to_hkl(self):
        """
        Redraw the HKL slice from the current Q-sample data.

        Connected to the "Convert" button signal for HKL conversion.
        No-ops if a slice update is already in progress. Gathers the
        projection matrix, slice position/thickness/width, slice
        normal, color-limit method, and display-scaling parameters
        from the view, then dispatches ``convert_to_hkl_process`` to a
        worker thread whose completion updates the slice display.
        """

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
        """
        Handle completion of the HKL-slice conversion worker.

        Parameters
        ----------
        result : dict or None
            Slice information dictionary returned by
            ``convert_to_hkl_process``, or None if the slice could not
            be computed. If present, its "signal" array is cached and
            the slice display is updated.
        """

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
        """
        Worker task that computes an HKL slice from Q-sample data.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        proj : array_like or None, optional
            Projection matrix (9 values reshaped to 3x3) whose rows
            define the U, V, W basis vectors, or None if the input
            fields are invalid.
        value : float or None, optional
            Position along the slice normal at which to slice.
        thickness : float or None, optional
            Thickness of the slice.
        width : float or None, optional
            Width of the output histogram bins.
        norm : list or None, optional
            Normal vector for the slice, as returned by
            ``get_normal``.
        clim_method : str or None, optional
            Color-limit clipping method, as returned by
            ``get_clim_method``.
        slice_scale : str or None, optional
            Display scaling for the slice ("linear" or "log").
        slice_auto_limits : bool or None, optional
            Whether to automatically compute display limits.
        vmin, vmax : float or None, optional
            Manual display limits, used when ``slice_auto_limits`` is
            False.

        Returns
        -------
        slice_histo : dict
            Slice information dictionary (extents, bins, transform,
            signal, and resolved display limits), returned only if the
            projection is valid and slice data is available.
        """

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
        """
        Cluster peak hkl offsets to detect satellite modulation
        vectors.

        Connected to the "cluster" button signal. Gathers the DBSCAN
        ``(eps, min_samples)`` parameters from the view, then
        dispatches ``cluster_process`` to a worker thread whose
        completion renders the cluster peaks and populates the cluster
        table.
        """

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
        """
        Handle completion of the peak-clustering worker.

        Parameters
        ----------
        result : dict or None
            Updated cluster peak information dictionary returned by
            ``cluster_process``, or None if clustering did not
            succeed. If present, the 3D cluster view and cluster table
            are refreshed.
        """

        if result is not None:
            self.update_processing("Adding peaks.", 30)
            self.view.add_cluster_peaks(result)
            self.view.update_cluster_table(result)
            self.update_processing("Peaks added!", 0)

    def calculate_alignment(self):
        """
        Calculate observed-vs-predicted Q vectors for goniometer
        alignment.

        Connected to the "calculate alignment" button signal. Gathers
        the selected run number and goniometer tilt angles from the
        view, then dispatches ``calculate_alignment_process`` to a
        worker thread whose completion renders the alignment
        comparison.
        """

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
        """
        Handle completion of the alignment-calculation worker.

        Parameters
        ----------
        result : dict or None
            Alignment information dictionary returned by
            ``calculate_alignment_process``, or None if the
            calculation did not succeed. If present, the alignment
            comparison view is refreshed.
        """

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
        """
        Worker task that computes observed-vs-predicted Q vectors for
        goniometer alignment.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        run_number : int or None, optional
            Run number to select peaks from, or None if the input
            field is invalid.
        tilts : tuple or None, optional
            Goniometer tilt angles ``(yaw, pitch, roll)`` in degrees,
            or None if the input fields are invalid.

        Returns
        -------
        result : dict
            Alignment information dictionary from the model, returned
            only if peaks and a UB matrix are present, the parameters
            are valid, and indexed peaks exist for the selected run.
        """

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
        """
        Worker task that clusters peak positions using DBSCAN.

        Parameters
        ----------
        progress : callable or None, optional
            Callback ``progress(message, percent)`` used to report
            status back to the view.
        stop_event : threading.Event or None, optional
            Event used to signal that processing should stop early.
        params : tuple or None, optional
            ``(eps, min_samples)`` DBSCAN parameters as returned by
            the view, or None if the input fields are invalid.

        Returns
        -------
        peak_info : dict
            Updated cluster peak information dictionary from the
            model, returned only if clustering succeeds.
        """
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
