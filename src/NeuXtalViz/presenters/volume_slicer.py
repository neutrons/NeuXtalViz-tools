from NeuXtalViz.presenters.base_presenter import NeuXtalVizPresenter

import functools
import numpy as np


class VolumeSlicer(NeuXtalVizPresenter):
    """
    Presenter for the volume slicer tool.

    Connects the volume slicer view's signals to model-driven actions for
    loading a NeXus MDHisto workspace and interactively slicing and cutting
    through the resulting volume, including managing display/color limits
    and saving slices and cuts.

    Parameters
    ----------
    view : NeuXtalViz.views.volume_slicer.VolumeSlicerView
        View for the volume slicer tool.
    model : NeuXtalViz.models.volume_slicer.VolumeSlicer
        Model for the volume slicer tool.
    """

    def __init__(self, view, model):
        """
        Initialize the presenter, wire up view signals, and set state flags.

        Parameters
        ----------
        view : NeuXtalViz.views.volume_slicer.VolumeSlicerView
            View for the volume slicer tool.
        model : NeuXtalViz.models.volume_slicer.VolumeSlicer
            Model for the volume slicer tool.
        """
        super(VolumeSlicer, self).__init__(view, model)

        self.view.connect_load_NXS(self.load_NXS)

        self.view.connect_slice_combo(self.update_volume)
        self.view.connect_cut_combo(self.update_cut)

        self.view.connect_slice_thickness_line(self.update_slice)
        self.view.connect_cut_thickness_line(self.update_cut)

        self.view.connect_clim_combo(self.update_slice)
        self.view.connect_cbar_combo(self.update_volume)

        self.view.connect_vlim_combo(self.update_slice_clim)

        self.view.connect_slice_scale_combo(self.update_slice_display)
        self.view.connect_auto_limits(self.update_slice_display)
        self.view.connect_cut_scale_combo(self.update_cut)

        self.view.connect_slice_line(self.update_slice_value)
        self.view.connect_cut_line(self.update_cut)

        self.view.connect_slice_ready(self.update_slice)
        self.view.connect_cut_ready(self.update_cut)

        self.view.connect_vmin_line(self.update_cvals)
        self.view.connect_vmax_line(self.update_cvals)

        self.view.connect_xmin_line(self.update_lims)
        self.view.connect_xmax_line(self.update_lims)

        self.view.connect_ymin_line(self.update_lims)
        self.view.connect_ymax_line(self.update_lims)

        self.view.connect_vol_scale_combo(self.update_volume)
        self.view.connect_opacity_combo(self.update_volume)
        self.view.connect_range_combo(self.update_volume)

        self.view.connect_save_slice(self.save_slice)
        self.view.connect_save_cut(self.save_cut)

        self.draw_idle = True
        self.slice_idle = True
        self.cut_idle = True
        self.slice_signal_cache = None

    def _calculate_display_limits(self, data, method, scale):
        """
        Compute display color limits for the given data.

        Parameters
        ----------
        data : array-like
            Signal array to compute display limits for.
        method : str or None
            Clipping method passed to the model's `calculate_clim`
            (e.g. 'normal', 'boxplot', or None).
        scale : str
            Display scale, either 'log' or 'linear'. When 'log' and the
            computed minimum is non-positive or non-finite, the minimum is
            recomputed from the smallest positive finite value in `data`.

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
        Resolve the display color limits to use, preferring manual values.

        If auto limits are disabled and valid manual limits are supplied,
        those are used (adjusted for log scale if necessary); otherwise the
        limits are recomputed from `data`.

        Parameters
        ----------
        data : array-like
            Signal array to compute display limits for if needed.
        method : str or None
            Clipping method passed to `_calculate_display_limits`.
        scale : str
            Display scale, either 'log' or 'linear'.
        auto_limits : bool
            Whether automatic limit calculation is enabled.
        current_vmin : float or None
            User-supplied lower limit, if any.
        current_vmax : float or None
            User-supplied upper limit, if any.

        Returns
        -------
        vmin : float
            Lower display limit.
        vmax : float
            Upper display limit.
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

    def update_lims(self):
        """
        Update slice and cut limits in the view based on input.

        Uses xmin/max, ymin/max and sets slice/cut limits if valid.

        Parameters
        ----------
        None
        """
        xmin = self.view.get_xmin_value()
        xmax = self.view.get_xmax_value()
        ymin = self.view.get_ymin_value()
        ymax = self.view.get_ymax_value()
        if (
            xmin is not None
            and xmax is not None
            and ymin is not None
            and ymax is not None
        ):
            if xmin < xmax and ymin < ymax:
                xlim = [xmin, xmax]
                ylim = [ymin, ymax]
                self.view.set_slice_lim(xlim, ylim)
                line_cut = self.view.get_cut()
                lim = xlim if line_cut == "Axis 1" else ylim
                self.view.set_cut_lim(lim)

    def update_cvals(self):
        """
        Update colorbar value limits in the view based on user input.

        Uses vmin, vmax from the view and sets colorbar limits if valid.

        Parameters
        ----------
        None
        """
        vmin = self.view.get_vmin_value()
        vmax = self.view.get_vmax_value()
        if vmin is not None and vmax is not None:
            if vmin < vmax:
                if vmin <= 0 and self.view.get_slice_scale() == "log":
                    vmin = vmax / 10
                self.view.update_colorbar_vlims(vmin, vmax)

    def update_slice_value(self):
        """
        Redraw the volume without resetting the current slice/cut view.

        Parameters
        ----------
        None
        """
        self.redraw_data(False)

    def update_volume(self):
        """
        Redraw the volume and reset the slice/cut view.

        Parameters
        ----------
        None
        """
        self.redraw_data(True)

    def update_cut_value(self):
        """
        Update the cut using the current cut value.

        Parameters
        ----------
        None
        """

        self.update_cut()

    def update_slice(self):
        """
        Slice the loaded volume if a histogram workspace is available.

        Parameters
        ----------
        None
        """
        if self.model.is_histo_loaded():
            self.slice_data()

    def update_slice_clim(self):
        """
        Refresh the slice display using the current color limit settings.

        Parameters
        ----------
        None
        """
        self.update_slice_display()

    def update_slice_display(self):
        """
        Update the slice display's colormap, scale, and value limits.

        If no slice signal is cached yet, triggers a slice update instead.

        Parameters
        ----------
        None
        """
        if self.slice_signal_cache is None:
            self.update_slice()
            return

        vmin, vmax = self._resolve_display_limits(
            self.slice_signal_cache,
            self.get_vlim_method(),
            self.view.get_slice_scale(),
            self.view.get_auto_limits(),
            self.view.get_vmin_value(),
            self.view.get_vmax_value(),
        )
        self.view.update_slice_display(
            self.view.get_colormap(),
            self.view.get_slice_scale(),
            vmin,
            vmax,
        )

    def update_cut(self):
        """
        Cut the loaded volume if a histogram workspace is available.

        Parameters
        ----------
        None
        """
        if self.model.is_histo_loaded():
            self.cut_data()

    def load_NXS(self):
        """
        Prompt for a NeXus file and load it on a worker thread.

        Parameters
        ----------
        None
        """
        filename = self.view.load_NXS_file_dialog()

        if filename:
            self.nxs_file = filename
            worker = self.view.worker(self.load_NXS_process)
            worker.connect_result(self.load_NXS_complete)
            worker.connect_finished(self.redraw_data)
            worker.connect_progress(self.update_processing)

            self.view.start_worker_pool(worker)

    def load_NXS_complete(self, result):
        """
        Complete NeXus file loading by refreshing the oriented lattice display.

        Parameters
        ----------
        result : object
            Result returned by `load_NXS_process` (unused).
        """
        self.update_oriented_lattice()

    def load_NXS_process(self, progress, stop_event=None):
        """
        Worker task that loads a NeXus MDHisto workspace.

        Intended to run on a background worker thread, reporting progress
        and checking for a stop request between steps.

        Parameters
        ----------
        progress : callable
            Callback invoked as ``progress(status, value)`` to report
            status text and percent complete.
        stop_event : threading.Event, optional
            Event used to signal that the worker should stop early
            (default None).
        """
        if self.stop_processing(stop_event):
            return None

        progress("Processing...", 1)

        if self.stop_processing(stop_event):
            return None

        progress("Loading NeXus file...", 10)

        if self.stop_processing(stop_event):
            return None

        self.model.load_md_histo_workspace(self.nxs_file)

        progress("Loading NeXus file...", 50)

        if self.stop_processing(stop_event):
            return None

        progress("Loading NeXus file...", 80)

        if self.stop_processing(stop_event):
            return None

        progress("NeXus file loaded!", 100)

    def get_normal(self):
        """
        Get the normal vector for the currently selected slice plane.

        Returns
        -------
        norm : list of int
            Unit normal vector, e.g. [0, 0, 1] for "Axis 1/2", [0, -1, 0]
            for "Axis 1/3", or [1, 0, 0] otherwise.
        """
        slice_plane = self.view.get_slice()

        if slice_plane == "Axis 1/2":
            norm = [0, 0, 1]
        elif slice_plane == "Axis 1/3":
            norm = [0, -1, 0]
        else:
            norm = [1, 0, 0]

        return norm

    def get_axis(self):
        """
        Get the axis vector for the currently selected line cut.

        Derived from the in-plane directions of `get_normal`, with the
        component corresponding to the selected cut axis zeroed out.

        Returns
        -------
        axis : list of int
            Axis vector with a single non-zero component identifying the
            in-plane direction being cut along.
        """
        axis = [1 if not norm else 0 for norm in self.get_normal()]
        ind = [i for i, ax in enumerate(axis) if ax == 1]

        line_cut = self.view.get_cut()

        if line_cut == "Axis 1":
            axis[ind[0]] = 0
        else:
            axis[ind[1]] = 0

        return axis

    def get_clim_method(self):
        """
        Get the color-limit clipping method for the 3D volume display.

        Returns
        -------
        method : str or None
            'normal' for mean +/- 3 sigma, 'boxplot' for quartile-based
            IQR clipping, or None for no clipping.
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
        Get the color-limit clipping method for the 2D slice display.

        Returns
        -------
        method : str or None
            'normal' for mean +/- 3 sigma, 'boxplot' for quartile-based
            IQR clipping, or None for no clipping.
        """
        ctype = self.view.get_vlim_clip_type()

        if ctype == "μ±3×σ":
            method = "normal"
        elif ctype == "Q₃/Q₁±1.5×IQR":
            method = "boxplot"
        else:
            method = None

        return method

    def redraw_data(self, reset=True):
        """
        Redraw the 3D volume on a worker thread if not already drawing.

        Parameters
        ----------
        reset : bool, optional
            Whether to reset the current slice/cut view before redrawing
            (default True).
        """
        if self.draw_idle:
            self.draw_idle = False

            if reset:
                self.view.reset_slice_cut()

            norm = self.get_normal()
            clim_method = self.get_clim_method()
            slice_value = self.view.get_slice_value()

            worker = self.view.worker(
                functools.partial(
                    self.redraw_data_process,
                    norm=norm,
                    clim_method=clim_method,
                    slice_value=slice_value,
                )
            )
            worker.connect_result(self.redraw_data_complete)
            worker.connect_finished(self.slice_data)
            worker.connect_progress(self.update_processing)

            self.view.start_worker_pool(worker)

    def redraw_data_complete(self, result):
        """
        Complete volume redraw and update the view with the new histogram.

        Parameters
        ----------
        result : tuple or None
            Tuple of (histo, normal, norm, value, trans) from
            `redraw_data_process`, or None if the worker was stopped or
            the parameters were invalid.
        """
        if result is not None:
            histo, normal, norm, value, trans = result

            self.view.add_histo(histo, normal, norm, value)

            self.view.set_transform(trans)

        self.draw_idle = True

    def redraw_data_process(
        self,
        progress,
        stop_event=None,
        norm=None,
        clim_method=None,
        slice_value=None,
    ):
        """
        Worker task that prepares the 3D volume histogram for display.

        Intended to run on a background worker thread, reporting progress
        and checking for a stop request between steps.

        Parameters
        ----------
        progress : callable
            Callback invoked as ``progress(status, value)`` to report
            status text and percent complete.
        stop_event : threading.Event, optional
            Event used to signal that the worker should stop early
            (default None).
        norm : array-like, optional
            Normal vector for the current slice plane (default None).
        clim_method : str or None, optional
            Color-limit clipping method passed to the model's
            `calculate_clim` (default None).
        slice_value : float, optional
            Position along the normal for the current slice plane
            (default None).

        Returns
        -------
        histo : dict
            Histogram information dictionary with clipped signal data.
        normal : numpy.ndarray
            Negated normal plane vector from the model.
        norm : array-like
            Normal vector for the current slice plane, as passed in.
        slice_value : float
            Position along the normal for the current slice plane, as
            passed in.
        transform : numpy.ndarray
            Transform matrix from the model.
        """
        if self.stop_processing(stop_event):
            return None

        if self.model.is_histo_loaded():
            progress("Processing...", 1)

            if self.stop_processing(stop_event):
                return None

            progress("Updating volume...", 20)

            if self.stop_processing(stop_event):
                return None

            histo = self.model.get_histo_info(norm)

            data = histo["signal"]

            data = self.model.calculate_clim(data, clim_method)

            progress("Updating volume...", 50)

            if self.stop_processing(stop_event):
                return None

            histo["signal"] = data

            normal = -self.model.get_normal_plane(norm)

            if slice_value is not None:
                progress("Volume drawn!", 100)

                return (
                    histo,
                    normal,
                    norm,
                    slice_value,
                    self.model.get_transform(),
                )

            else:
                progress("Invalid parameters.", 0)

    def slice_data(self):
        """
        Slice the volume on a worker thread if not already slicing.

        Parameters
        ----------
        None
        """
        if self.slice_idle:
            self.slice_idle = False

            norm = self.get_normal()
            thick = self.view.get_slice_thickness()
            value = self.view.get_slice_value()
            vlim_method = self.get_vlim_method()
            scale = self.view.get_slice_scale()
            auto_limits = self.view.get_auto_limits()
            vmin = self.view.get_vmin_value()
            vmax = self.view.get_vmax_value()

            worker = self.view.worker(
                functools.partial(
                    self.slice_data_process,
                    norm=norm,
                    thick=thick,
                    value=value,
                    vlim_method=vlim_method,
                    scale=scale,
                    auto_limits=auto_limits,
                    vmin=vmin,
                    vmax=vmax,
                )
            )
            worker.connect_result(self.slice_data_complete)
            worker.connect_finished(self.cut_data)
            worker.connect_progress(self.update_processing)

            self.view.start_worker_pool(worker)

    def slice_data_complete(self, result):
        """
        Complete slice calculation and update the view with the result.

        Caches the sliced signal for later use by `update_slice_display`.

        Parameters
        ----------
        result : dict or None
            Slice information dictionary from `slice_data_process`, or
            None if the worker was stopped or the parameters were invalid.
        """
        if result is not None:
            self.slice_signal_cache = result["signal"].copy()
            self.view.add_slice(result)
        self.slice_idle = True

    def slice_data_process(
        self,
        progress,
        stop_event=None,
        norm=None,
        thick=None,
        value=None,
        vlim_method=None,
        scale=None,
        auto_limits=None,
        vmin=None,
        vmax=None,
    ):
        """
        Worker task that computes a 2D slice through the loaded volume.

        Intended to run on a background worker thread, reporting progress
        and checking for a stop request between steps.

        Parameters
        ----------
        progress : callable
            Callback invoked as ``progress(status, value)`` to report
            status text and percent complete.
        stop_event : threading.Event, optional
            Event used to signal that the worker should stop early
            (default None).
        norm : array-like, optional
            Normal vector for the slicing direction (default None).
        thick : float, optional
            Thickness of the slice (default None).
        value : float, optional
            Position along the normal to slice (default None).
        vlim_method : str or None, optional
            Color-limit clipping method used to resolve display limits
            (default None).
        scale : str, optional
            Display scale, either 'log' or 'linear' (default None).
        auto_limits : bool, optional
            Whether automatic limit calculation is enabled (default None).
        vmin : float, optional
            User-supplied lower display limit, if any (default None).
        vmax : float, optional
            User-supplied upper display limit, if any (default None).

        Returns
        -------
        slice_histo : dict or None
            Slice information dictionary with resolved vmin/vmax display
            limits, or None if stopped, the histogram is not loaded, or
            `thick`/`value` are missing.
        """
        if self.stop_processing(stop_event):
            return None

        if self.model.is_histo_loaded():
            if thick is not None and value is not None:
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Updating slice...", 50)

                if self.stop_processing(stop_event):
                    return None

                slice_histo = self.model.get_slice_info(norm, value, thick)

                data = slice_histo["signal"]

                vmin_out, vmax_out = self._resolve_display_limits(
                    data, vlim_method, scale, auto_limits, vmin, vmax
                )
                slice_histo["vmin"] = vmin_out
                slice_histo["vmax"] = vmax_out

                progress("Data sliced!", 100)

                return slice_histo

    def cut_data(self):
        """
        Cut the sliced volume on a worker thread if not already cutting.

        Parameters
        ----------
        None
        """
        if self.cut_idle:
            self.cut_idle = False

            axis = self.get_axis()
            cut_value = self.view.get_cut_value()
            cut_thick = self.view.get_cut_thickness()

            worker = self.view.worker(
                functools.partial(
                    self.cut_data_process,
                    axis=axis,
                    cut_value=cut_value,
                    cut_thick=cut_thick,
                )
            )
            worker.connect_result(self.cut_data_complete)
            worker.connect_finished(self.update_complete)
            worker.connect_progress(self.update_processing)

            self.view.start_worker_pool(worker)

    def cut_data_complete(self, result):
        """
        Complete cut calculation and update the view with the result.

        Parameters
        ----------
        result : dict or None
            Cut information dictionary from `cut_data_process`, or None
            if the worker was stopped or the parameters were invalid.
        """
        if result is not None:
            self.view.add_cut(result)
        self.cut_idle = True

    def cut_data_process(
        self,
        progress,
        stop_event=None,
        axis=None,
        cut_value=None,
        cut_thick=None,
    ):
        """
        Worker task that computes a 1D cut through the sliced volume.

        Intended to run on a background worker thread, reporting progress
        and checking for a stop request between steps.

        Parameters
        ----------
        progress : callable
            Callback invoked as ``progress(status, value)`` to report
            status text and percent complete.
        stop_event : threading.Event, optional
            Event used to signal that the worker should stop early
            (default None).
        axis : array-like, optional
            Axis along which to cut (default None).
        cut_value : float, optional
            Position along the axis to cut (default None).
        cut_thick : float, optional
            Thickness of the cut (default None).

        Returns
        -------
        cut_histo : dict or None
            Cut information dictionary, or None if stopped, the volume is
            not sliced, or `cut_value`/`cut_thick` are missing.
        """
        if self.stop_processing(stop_event):
            return None

        if self.model.is_sliced():
            if cut_value is not None and cut_thick is not None:
                progress("Processing...", 1)

                if self.stop_processing(stop_event):
                    return None

                progress("Updating cut...", 50)

                if self.stop_processing(stop_event):
                    return None

                progress("Data cut!", 100)

                cut_histo = self.model.get_cut_info(axis, cut_value, cut_thick)

                return cut_histo

    def save_slice(self):
        """
        Prompt for a filename and save the current slice, if one exists.

        Parameters
        ----------
        None
        """
        if self.model.is_sliced():
            filename = self.view.save_file_dialog()

            if filename:
                self.model.save_slice(filename)

    def save_cut(self):
        """
        Prompt for a filename and save the current cut, if one exists.

        Parameters
        ----------
        None
        """
        if self.model.is_cut():
            filename = self.view.save_file_dialog()

            if filename:
                self.model.save_cut(filename)
