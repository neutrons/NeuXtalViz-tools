import re

from mantid.simpleapi import (
    LoadMD,
    LoadCIF,
    SaveMD,
    IntegrateMDHistoWorkspace,
    CloneWorkspace,
    CreateMDWorkspace,
    CreateSampleWorkspace,
    CreateSingleValuedWorkspace,
    BinMD,
    AddSampleLog,
    CopySample,
    MultiplyMD,
    MinusMD,
    DeleteWorkspace,
    mtd,
)
from mantid.geometry import SpaceGroupFactory

import numpy as np
import scipy.linalg
from scipy.ndimage import gaussian_filter, median_filter

import skimage.measure

from NeuXtalViz.models.base_model import NeuXtalVizModel
from NeuXtalViz.models.utilities import SaveMDToAscii


class VolumeSlicerModel(NeuXtalVizModel):
    """
    Model for slicing and cutting reciprocal-space MD histogram volumes.

    Wraps Mantid MD histogram workspaces (``histo``/``volume``) to load
    and downsample 3D reciprocal-space data, and to produce 2D slices and
    1D cuts of that data along axes aligned with the UB/W orientation
    matrices for visualization.
    """

    def __init__(self):
        """
        Initialize the VolumeSlicerModel.
        """
        super(VolumeSlicerModel, self).__init__()

        self.W = np.eye(3)

        # Registry of workspaces the Transform tab has loaded/derived,
        # keyed by the user-facing display name:
        #   {display_name: {"ws_name": str, "space": "reciprocal"|"real"}}
        # Only one entry is ever "active" (cloned into "histo") at a
        # time -- see activate_workspace().
        self.workspace_registry = {}
        self.active_space = "reciprocal"
        self.active_display_name = None
        self._load_counter = 0

        # Bond-network viewer state (see the "Bond-network viewer"
        # section below) -- populated by load_bond_cif().
        self._bond_A = None
        self._bond_sg = None
        self._bond_sites = []

    def register_workspace(self, display_name, ws_name, space="reciprocal"):
        """
        Register a workspace so it can be selected via the workspace combo.

        Parameters
        ----------
        display_name : str
            User-facing name shown in the workspace combo box.
        ws_name : str
            Actual Mantid AnalysisDataService workspace name.
        space : str, optional
            Either ``"reciprocal"`` (default) for ordinary Q-space data,
            or ``"real"`` for a delta-PDF (direct-space) result. Drives
            which physical basis matrix and axis labeling
            (hkl vs. uvw) is used when this workspace is active.
        """
        self.workspace_registry[display_name] = {
            "ws_name": ws_name,
            "space": space,
        }

    def activate_workspace(
        self, display_name, progress=None, stop_event=None, **kwargs
    ):
        """
        Make a registered workspace the active one for slicing/cutting.

        Clones the registered workspace into the fixed internal
        ``"histo"`` slot (deleting any previous ``"histo"``/``"volume"``/
        ``"slice"``/``"cut"`` first to bound memory use) and re-runs the
        same setup ``load_md_histo_workspace`` performs after ``LoadMD``,
        so the existing slice/cut pipeline is otherwise untouched.

        Parameters
        ----------
        display_name : str
            Display name previously passed to :meth:`register_workspace`.
        progress : callable, optional
            ``progress(message, percent)`` callback (injected by the
            worker infrastructure) -- cloning/downsampling can take a
            while for a large workspace.
        stop_event : threading.Event, optional
            Unused; accepted because the presenter runs this on a
            worker thread, which always passes it (injected by the
            worker infrastructure).
        """
        entry = self.workspace_registry[display_name]

        if progress is not None:
            progress("Activating workspace", 0)

        for ws in ("histo", "volume", "slice", "cut"):
            if mtd.doesExist(ws):
                DeleteWorkspace(Workspace=ws)

        CloneWorkspace(
            InputWorkspace=entry["ws_name"], OutputWorkspace="histo"
        )

        self.active_space = entry["space"]
        self.active_display_name = display_name

        if progress is not None:
            progress("Cloning workspace", 20)

        self._activate_histo(progress=progress)

        if progress is not None:
            progress("Done", 100)

    def delete_workspace(self, display_name):
        """
        Remove a registered workspace and delete its backing workspace.

        If ``display_name`` is currently active, the active
        ``"histo"``/``"volume"``/``"slice"``/``"cut"`` scratch
        workspaces are independent clones and are left as they are --
        they simply won't correspond to any registry entry until a
        different workspace is activated.

        Parameters
        ----------
        display_name : str
            Display name to remove.
        """
        entry = self.workspace_registry.pop(display_name)

        if mtd.doesExist(entry["ws_name"]):
            DeleteWorkspace(Workspace=entry["ws_name"])

    def save_md_workspace(self, display_name, filename):
        """
        Save a registered workspace's backing MD workspace to a file.

        Parameters
        ----------
        display_name : str
            Display name previously passed to
            :meth:`register_workspace` (or returned by one of the
            load/derived-workspace steps below).
        filename : str
            Output file path (``.nxs``).
        """
        entry = self.workspace_registry[display_name]

        SaveMD(InputWorkspace=entry["ws_name"], Filename=filename)

        if self.active_display_name == display_name:
            self.active_display_name = None

    def load_md_histo_workspace(self, filename, display_name=None):
        """
        Load a Mantid MD histogram workspace from a file, register it,
        and make it the active workspace.

        The file is loaded into its own stable, registry-backed
        workspace (not directly into ``"histo"``), so it survives later
        calls to :meth:`activate_workspace` for *other* registered
        workspaces -- ``"histo"`` itself is just a scratch slot that
        gets deleted/recreated on every activation.

        Parameters
        ----------
        filename : str
            Path to the Mantid MD histogram file to load.
        display_name : str, optional
            User-facing name for the workspace combo. Defaults to a
            sequential ``"data"``, ``"data_2"``, ``"data_3"``, ... name
            (the counter never repeats, even across renamed/deleted
            entries), disambiguated further if it still collides with
            an already-registered name.
        """
        self._load_counter += 1
        if display_name is None:
            display_name = self._auto_load_name()
        display_name = self._unique_display_name(display_name)

        ws_name = self._unique_ws_name(display_name)

        LoadMD(Filename=filename, OutputWorkspace=ws_name)

        self.register_workspace(
            display_name, ws_name, space=self._detect_space(ws_name)
        )
        self.activate_workspace(display_name)

        return display_name

    @staticmethod
    def _detect_space(ws_name):
        """
        Infer whether a just-loaded MD workspace is real-space
        (3D-ΔPDF) or reciprocal-space, from its own dimension units --
        needed because a workspace loaded from a file (as opposed to
        one derived via :meth:`calculate_pdf`, which already knows) has
        no other indication of which space it's in, and re-loading a
        previously :meth:`save_md_workspace`-saved ΔPDF result must
        still be recognized as real-space (e.g. so it appears in the
        bond-network viewer's ΔPDF-workspace combo, and so the Slice
        tab still defaults to the diverging colormap for it).

        Parameters
        ----------
        ws_name : str
            Mantid workspace name to inspect.

        Returns
        -------
        space : str
            ``"real"`` if every dimension is tagged with this app's
            own real-space unit string (``"d.l.u."``, always used by
            :meth:`_transform`'s output and nothing else this app
            produces); ``"reciprocal"`` otherwise (the common case,
            and the safe default for externally-produced files).
        """
        ws = mtd[ws_name]
        units = [ws.getDimension(i).getUnits() for i in range(ws.getNumDims())]

        if units and all(u == "d.l.u." for u in units):
            return "real"

        return "reciprocal"

    def _auto_load_name(self):
        """Sequential ``"data"``/``"data_2"``/... name for ``self._load_counter``."""
        return (
            "data"
            if self._load_counter == 1
            else "data_{}".format(self._load_counter)
        )

    def next_load_display_name(self):
        """
        Preview the display name that would be auto-assigned to the
        next :meth:`load_md_histo_workspace` call, without consuming
        the load counter (so the preview stays accurate if the caller
        doesn't end up loading, e.g. the user cancels a naming prompt).

        Returns
        -------
        display_name : str
            The sequential ``"data"``/``"data_2"``/... name that would
            be used next, disambiguated against the current registry.
        """
        n = self._load_counter + 1
        base = "data" if n == 1 else "data_{}".format(n)
        return self._unique_display_name(base)

    def rename_workspace(self, old_display_name, new_display_name):
        """
        Rename a registered workspace's display name.

        Only the registry entry's key changes -- the underlying Mantid
        workspace name is left as-is, so this is safe to call whether
        or not the workspace is currently active.

        Parameters
        ----------
        old_display_name : str
            Current display name (registry key).
        new_display_name : str
            New display name. Disambiguated if it collides with
            another already-registered name.

        Returns
        -------
        new_display_name : str
            The (possibly disambiguated) display name actually used.
        """
        new_display_name = self._unique_display_name(
            new_display_name, exclude=old_display_name
        )
        self.workspace_registry[new_display_name] = (
            self.workspace_registry.pop(old_display_name)
        )
        return new_display_name

    def _unique_display_name(self, base, exclude=None):
        """Disambiguate a display name against the workspace registry."""
        name = base
        n = 1
        while name in self.workspace_registry and name != exclude:
            n += 1
            name = "{} ({})".format(base, n)
        return name

    def _replace_workspace_if_exists(self, display_name, keep_ws_name=None):
        """
        Drop any existing registry entry/backing workspace at this name.

        Used by the Bragg-punch/blur/3D-ΔPDF steps so re-running a step
        (e.g. after tweaking a parameter) overwrites its previous
        output in place, rather than accumulating "punched (2)",
        "punched (3)", ... entries.

        Parameters
        ----------
        display_name : str
            Display name about to be (re-)registered.
        keep_ws_name : str, optional
            Backing workspace name to leave alone even if it matches
            the existing entry's (e.g. because it's still needed as an
            input to the operation replacing it).
        """
        entry = self.workspace_registry.pop(display_name, None)
        if entry is None:
            return
        if entry["ws_name"] == keep_ws_name:
            return
        if mtd.doesExist(entry["ws_name"]):
            DeleteWorkspace(Workspace=entry["ws_name"])

    @staticmethod
    def _unique_ws_name(base):
        """Sanitize a display name into a fresh, valid ADS workspace name."""
        safe = re.sub(r"[^0-9A-Za-z_]", "_", base) or "workspace"
        name = safe
        n = 1
        while mtd.doesExist(name):
            n += 1
            name = "{}_{}".format(safe, n)
        return name

    def _activate_histo(self, progress=None):
        """
        Prepare the ``"histo"``/``"volume"`` workspaces for slicing.

        Clamps infinities to NaN, downsamples for the 3D volume
        render, and reads the UB/W matrices. Shared by both
        :meth:`load_md_histo_workspace` (after ``LoadMD``) and
        :meth:`activate_workspace` (after cloning a registered
        workspace into ``"histo"``).

        Genuine NaN (e.g. from :meth:`run_bragg_punch`'s masking, or
        unmeasured detector regions) is left as NaN rather than zeroed
        out -- zeroing it here would silently turn "no data" into a
        real (and wrong) zero-intensity value for both the 2D
        slice/cut (matplotlib renders NaN as a transparent "bad"
        color) and the 3D volume render (the block-reduce below and
        PyVista's volume mapper are both already NaN-aware). Only
        actual infinities are clamped, matching the convention used in
        :meth:`get_slice_info`.

        Parameters
        ----------
        progress : callable, optional
            ``progress(message, percent)`` callback -- compacting/
            downsampling can take a while for a large workspace.
        """
        if progress is not None:
            progress("Clamping infinities", 30)

        signal = mtd["histo"].getSignalArray().copy()
        signal_var = mtd["histo"].getErrorSquaredArray().copy()

        signal[np.isinf(signal)] = np.nan
        signal_var[np.isinf(signal_var)] = np.nan

        mtd["histo"].setSignalArray(signal)
        mtd["histo"].setErrorSquaredArray(signal_var)

        if progress is not None:
            progress("Preparing volume", 50)

        CloneWorkspace(InputWorkspace="histo", OutputWorkspace="volume")

        signal = mtd["volume"].getSignalArray()

        self.shape = signal.shape

        dims = [mtd["volume"].getDimension(i) for i in range(3)]

        self.min_lim = np.array(
            [dim.getMinimum() + dim.getBinWidth() * 0.5 for dim in dims]
        )

        self.max_lim = np.array(
            [dim.getMaximum() - dim.getBinWidth() * 0.5 for dim in dims]
        )

        self.labels = [
            "{} ({})".format(dim.name, dim.getUnits()) for dim in dims
        ]

        self.spacing = np.array([dim.getBinWidth() for dim in dims])

        max_dim = 128.0
        max_spacing = 0.5

        shape = np.array(self.shape, dtype=float)

        base = np.ceil(shape / max_dim).astype(int)
        base[base < 1] = 1

        with np.errstate(divide="ignore", invalid="ignore"):
            spacing_limit = np.floor(max_spacing / self.spacing)

        spacing_limit[~np.isfinite(spacing_limit)] = np.inf
        spacing_limit[spacing_limit < 1] = 1
        spacing_limit = spacing_limit.astype(int)

        base = np.minimum(base, spacing_limit)

        scale = base.copy()
        compress = np.maximum(base, np.minimum(spacing_limit, 2 * base))

        blocks = [
            (compress[0], scale[1], scale[2]),
            (scale[0], compress[1], scale[2]),
            (scale[0], scale[1], compress[2]),
        ]

        if progress is not None:
            progress("Downsampling for 3D view", 70)

        self.signals = []
        self.spacings = []
        for block in blocks:
            self.spacings.append(self.spacing * np.array(block))
            self.signals.append(
                skimage.measure.block_reduce(
                    signal, block_size=block, func=np.nanmean, cval=np.nan
                )
            )

        if progress is not None:
            progress("Reading orientation", 90)

        self.set_B()
        self.set_W()

    def save_slice(self, filename):
        """
        Save the current slice workspace to an ASCII file.

        Parameters
        ----------
        filename : str
            Output filename for the slice.
        """
        SaveMDToAscii("slice", filename)

    def save_cut(self, filename):
        """
        Save the current cut workspace to an ASCII file.

        Parameters
        ----------
        filename : str
            Output filename for the cut.
        """
        SaveMDToAscii("cut", filename)

    def is_histo_loaded(self):
        """
        Check if a histogram workspace is loaded in Mantid.

        Returns
        -------
        bool
            True if 'histo' workspace exists, False otherwise.
        """
        return mtd.doesExist("histo")

    def is_sliced(self):
        """
        Check if a slice workspace exists in Mantid.

        Returns
        -------
        bool
            True if 'slice' workspace exists, False otherwise.
        """
        return mtd.doesExist("slice")

    def is_cut(self):
        """
        Check if a cut workspace exists in Mantid.

        Returns
        -------
        bool
            True if 'cut' workspace exists, False otherwise.
        """
        return mtd.doesExist("cut")

    def set_B(self):
        """
        Set the B matrix from the UB of the loaded histogram workspace.
        """
        if self.has_UB("histo"):
            ei = mtd["histo"].getExperimentInfo(0)

            B = ei.sample().getOrientedLattice().getB().copy()

            self.set_UB(B)

    def set_W(self):
        """
        Set the W matrix from the workspace log if available, otherwise identity.
        """
        ei = mtd["histo"].getExperimentInfo(0)

        self.W = np.eye(3)

        if ei.run().hasProperty("W_MATRIX"):
            self.W = np.array(ei.run().getLogData("W_MATRIX").value).reshape(
                3, 3
            )

    def get_histo_info(self, normal):
        """
        Get histogram information for a given normal direction.

        Parameters
        ----------
        normal : array-like
            Normal vector for the slicing direction.

        Returns
        -------
        dict
            Dictionary containing signal, limits, spacing, labels, and transforms.
        """
        ind = np.abs(normal).tolist().index(1)

        histo_dict = {}

        histo_dict["signal"] = self.signals[ind].copy()

        histo_dict["min_lim"] = self.min_lim
        histo_dict["max_lim"] = self.max_lim
        histo_dict["spacing"] = self.spacings[ind]
        histo_dict["labels"] = self.labels

        P, T, S = self.get_transforms()

        histo_dict["transform"] = T
        histo_dict["projection"] = P
        histo_dict["scales"] = S

        return histo_dict

    def get_slice_info(
        self, normal, value, thickness=0.01, xlim=None, ylim=None
    ):
        """
        Get slice information for a given normal and value.

        Parameters
        ----------
        normal : array-like
            Normal vector for the slicing direction.
        value : float
            Position along the normal to slice.
        thickness : float, optional
            Thickness of the slice (default 0.01).
        xlim : list, optional
            X-axis limits for the slice (default None).
        ylim : list, optional
            Y-axis limits for the slice (default None).

        Returns
        -------
        dict
            Dictionary containing x, y, labels, signal, transform, aspect, value, and title.
        """

        self.normal = normal

        slice_dict = {}

        integrate = [value - thickness, value + thickness]

        xbin = None
        if xlim is not None:
            if xlim[1] > xlim[0]:
                xbin = [xlim[0], 0, xlim[1]]
        ybin = None
        if ylim is not None:
            if ylim[1] > ylim[0]:
                ybin = [ylim[0], 0, ylim[1]]

        slice_lims = [xbin, ybin]

        self.integrate = integrate

        pbin = []
        j = 0
        for i, norm in enumerate(normal):
            if norm == 0:
                pbin.append(slice_lims[j])
                j += 1
            else:
                pbin.append(integrate)

        IntegrateMDHistoWorkspace(
            InputWorkspace="volume",
            P1Bin=pbin[0],
            P2Bin=pbin[1],
            P3Bin=pbin[2],
            OutputWorkspace="slice",
        )

        self.slice_bin = pbin

        i = np.abs(normal).tolist().index(1)

        form = "{} = ({:.2f},{:.2f})"

        title = form.format(mtd["slice"].getDimension(i).name, *integrate)

        dims = mtd["slice"].getNonIntegratedDimensions()

        x, y = [
            np.linspace(
                dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
            )
            for dim in dims
        ]

        labels = ["{} ({})".format(dim.name, dim.getUnits()) for dim in dims]

        slice_dict["x"] = x
        slice_dict["y"] = y
        slice_dict["labels"] = labels

        signal = mtd["slice"].getSignalArray().T.copy().squeeze()

        signal[np.isinf(signal)] = np.nan

        slice_dict["signal"] = signal

        Bp = self._basis_matrix()

        Q, R = scipy.linalg.qr(Bp)

        ind = np.abs(normal) != 1
        i = ind.tolist().index(False)

        slice_dict["z"] = value
        slice_dict["space"] = self.active_space

        readout_matrix = (
            np.linalg.inv(self.W).T if self.active_space == "real" else self.W
        )
        slice_dict["W"] = np.column_stack(
            [readout_matrix[:, ind], readout_matrix[:, i]]
        )

        v = scipy.linalg.cholesky(np.dot(R.T, R)[ind][:, ind], lower=False)

        v /= v[0, 0]

        T = np.eye(3)
        T[:2, :2] = v

        s = np.diag(T).copy()
        T[1, 1] = 1

        T[0, 2] = -T[0, 1] * y.min()

        slice_dict["transform"] = T
        slice_dict["aspect"] = s[1]
        slice_dict["value"] = value
        slice_dict["title"] = title

        return slice_dict

    def get_cut_info(self, axis, value, thickness=0.01):
        """
        Get cut information for a given axis and value.

        Parameters
        ----------
        axis : array-like
            Axis along which to cut (e.g., [1,0,0]).
        value : float
            Position along the axis to cut.
        thickness : float, optional
            Thickness of the cut (default 0.01).

        Returns
        -------
        dict
            Dictionary containing x, y, e, label, value, and title.
        """
        cut_dict = {}

        integrate = [value - thickness, value + thickness]

        pbin = [None if ax == 0 else integrate for ax in axis]

        self.integrate = integrate

        IntegrateMDHistoWorkspace(
            InputWorkspace="slice",
            P1Bin=pbin[0],
            P2Bin=pbin[1],
            P3Bin=pbin[2],
            OutputWorkspace="cut",
        )

        self.cut_bin = pbin

        i = np.abs(self.normal).tolist().index(1)
        j = np.array(axis).tolist().index(1)

        form = "{} = ({:.2f},{:.2f})"

        title = form.format(mtd["slice"].getDimension(i).name, *self.integrate)
        title += " / "
        title += form.format(mtd["cut"].getDimension(j).name, *integrate)

        dim = mtd["cut"].getNonIntegratedDimensions()[0]

        x = np.linspace(
            dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
        )

        x = 0.5 * (x[1:] + x[:-1])

        label = "{} ({})".format(dim.name, dim.getUnits())

        cut_dict["x"] = x
        cut_dict["y"] = mtd["cut"].getSignalArray().squeeze()
        cut_dict["e"] = np.sqrt(mtd["cut"].getErrorSquaredArray().squeeze())
        cut_dict["label"] = label
        cut_dict["value"] = value
        cut_dict["title"] = title

        return cut_dict

    def calculate_clim(self, trans, method="normal", symmetric_zero=False):
        """
        Calculate color limits for visualization based on a method.

        Parameters
        ----------
        trans : np.ndarray
            Signal array to calculate color limits for (clipped in place
            and also returned).
        method : str, optional
            Method for calculation: 'normal' (mean +/- 3 sigma), 'boxplot'
            (quartiles +/- 1.5 IQR), or any other value for plain min/max
            (default 'normal').
        symmetric_zero : bool, optional
            If True, force the computed limits to be symmetric about
            zero (``cmax = max(abs(cmin), abs(cmax))``, ``cmin =
            -cmax``) before clipping -- for signed data (e.g. a
            delta-PDF result) displayed with a zero-aware transfer
            function (default False).

        Returns
        -------
        trans : np.ndarray
            The input array with values clipped to the calculated color
            limits.
        """
        trans[~np.isfinite(trans)] = np.nan

        vmin, vmax = np.nanmin(trans), np.nanmax(trans)

        if np.isclose(vmin, vmax) or not np.isfinite(vmin):
            vmin = vmax / 100
        elif not np.isfinite([vmin, vmax]).all():
            vmin, vmax = 1e-3, 1 + 3

        if method == "normal":
            mu, sigma = np.nanmean(trans), np.nanstd(trans)

            spread = 3 * sigma

            cmin, cmax = mu - spread, mu + spread

        elif method == "boxplot":
            Q1, Q3 = np.nanpercentile(trans, [25, 75])

            IQR = Q3 - Q1

            spread = 1.5 * IQR

            cmin, cmax = Q1 - spread, Q3 + spread

        else:
            cmin, cmax = vmin, vmax

        if cmax <= cmin:
            # 'normal'/'boxplot' can collapse to a zero-width range on
            # sparse data (e.g. a mostly-masked/zero-filled reciprocal-
            # space volume, where >25% exactly-zero voxels gives
            # Q1 == Q3 == 0) -- fall back to the true min/max rather
            # than clipping everything to a single flat value.
            cmin, cmax = vmin, vmax

        if symmetric_zero:
            cmax = max(abs(cmin), abs(cmax))
            cmin = -cmax

        clim = [cmin if cmin > vmin else vmin, cmax if cmax < vmax else vmax]

        trans[trans < clim[0]] = clim[0]
        trans[trans > clim[1]] = clim[1]

        return trans

    def _basis_matrix(self):
        """
        Physical basis matrix for the active workspace's space.

        Returns ``Bp = UB @ W`` (the reciprocal-space physical basis,
        Å⁻¹ per W-projected index) for ordinary reciprocal-space
        workspaces. For a real-space (delta-PDF) active workspace,
        returns its dual ``Ap = inv(Bp).T`` instead: the FFT used to
        produce a delta-PDF result lives on the grid conjugate to the
        input's ``Bp``-scaled index grid (``np.fft.fftfreq`` returns
        cycles/unit, matching this file's own no-2*pi convention for
        ``Bp``, so no extra factor is needed here), so by the standard
        DFT dual-basis relationship its physical basis satisfies
        ``Bp.T @ Ap = I``, i.e. ``Ap = inv(Bp).T``.

        This is fed into the same QR/Cholesky orthonormal-basis
        construction used everywhere else in this class -- that
        construction is invariant to any global scalar on its input
        (scaling ``Bp`` scales ``R``/the Cholesky factor `v`
        proportionally, and every consumer normalizes by `v` or
        `v[0,0]`), so no 2*pi reconciliation between the two
        conventions is needed here.

        Returns
        -------
        Bp : np.ndarray
            3x3 physical basis matrix for the active workspace's space.
        """
        Bp = np.dot(self.UB, self.W)

        if self.active_space == "real":
            return np.linalg.inv(Bp).T

        return Bp

    def orientation_matrix(self):
        """
        Compute the orientation matrix for the current UB and W matrices.

        Overrides :meth:`NeuXtalVizModel.orientation_matrix` to account
        for the additional W matrix (projection into the slicing basis).

        Returns
        -------
        U : np.ndarray
            3x3 orientation matrix combining the UB and W matrices.
        """
        Bp = self._basis_matrix()

        Q, R = scipy.linalg.qr(Bp)

        v = scipy.linalg.cholesky(np.dot(R.T, R), lower=False)

        Q = np.dot(Bp, np.linalg.inv(v))

        return np.dot(Q.T, self.UB)

    def get_transform(self, reciprocal=True):
        """
        Get the transformation matrix for the current UB and W matrices.

        Overrides :meth:`NeuXtalVizModel.get_transform` to account for
        the additional W matrix (projection into the slicing basis).
        Returns None if no UB matrix is set.

        Parameters
        ----------
        reciprocal : bool, optional
            If True, return reciprocal transformation; else real space
            (default True).

        Returns
        -------
        T : np.ndarray or None
            Normalized 3x3 transformation matrix, or None if no UB
            matrix has been set.
        """
        if self.UB is not None:
            b = self.UB / np.linalg.norm(self.UB, axis=0)

            Bp = self._basis_matrix()

            Q, R = scipy.linalg.qr(Bp)

            v = scipy.linalg.cholesky(np.dot(R.T, R), lower=False)

            Q = np.dot(Bp, np.linalg.inv(v))

            T = np.dot(Q.T, b)

            if not reciprocal:
                T = np.column_stack(
                    [
                        np.cross(T[:, 1], T[:, 2]),
                        np.cross(T[:, 2], T[:, 0]),
                        np.cross(T[:, 0], T[:, 1]),
                    ]
                )

            return T

    def get_transforms(self):
        """
        Get projection, transform, and scale matrices for the current UB and W.

        Returns
        -------
        p : np.ndarray
            Projection matrix, normalized so its first diagonal element
            is 1.
        t : np.ndarray
            Transform matrix, normalized by the column norms of the
            Cholesky factor.
        s : np.ndarray
            Scale factors (column norms of `p`).
        """
        Bp = self._basis_matrix()

        Q, R = scipy.linalg.qr(Bp)

        v = scipy.linalg.cholesky(np.dot(R.T, R), lower=False)

        s = np.linalg.norm(v, axis=0)
        t = v / s
        p = v / v[0, 0]

        s = np.linalg.norm(p, axis=0)

        return p, t, s

    def get_normal_plane(self, ind):
        """
        Get the normal vector for a plane given an index vector.

        Parameters
        ----------
        ind : array-like
            One-hot index vector (e.g. ``[1, 0, 0]``) selecting the
            slicing axis in the W-projected basis.

        Returns
        -------
        vec : np.ndarray or None
            Normal vector for the plane in Cartesian coordinates, or
            None if no UB matrix has been set.
        """
        if self.UB is not None:
            Bp = self._basis_matrix()

            Q, R = scipy.linalg.qr(Bp)

            v = scipy.linalg.cholesky(np.dot(R.T, R), lower=False)

            matrix = np.cross(
                np.dot(v, np.roll(np.eye(3), 2, 1)).T,
                np.dot(v, np.roll(np.eye(3), 1, 1)).T,
            ).T

            vec = np.dot(matrix, ind)

            return vec

    # ------------------------------------------------------------------
    # Delta-PDF pipeline
    # ------------------------------------------------------------------
    #
    # Punch/cut/blur/pad/FFT pipeline: mask out the allowed-Bragg-
    # reflection regions of reciprocal space (Bragg punch), remove the
    # low-Q region (inner cut), fill the resulting gaps with a NaN-aware
    # Gaussian blur, zero-pad to a larger Q-extent, then Fourier-
    # transform to real space. Ring removal and background subtraction,
    # present in some delta-PDF workflows, are intentionally not
    # implemented here.
    #
    # Each method below operates on/produces entries in
    # ``self.workspace_registry`` (by display name), so results can be
    # selected via ``activate_workspace`` like any other loaded
    # workspace.

    @staticmethod
    def _check_common_bins(display_a, ws_a_name, display_b, ws_b_name):
        """
        Verify two workspaces share common binning before combining them.

        Parameters
        ----------
        display_a, display_b : str
            Display names, used only to build a readable error message.
        ws_a_name, ws_b_name : str
            Mantid workspace names to compare.

        Raises
        ------
        ValueError
            If the workspaces have a different number of dimensions,
            or any dimension's bin count/extent differs beyond a
            small numerical tolerance.
        """
        ws_a, ws_b = mtd[ws_a_name], mtd[ws_b_name]

        if ws_a.getNumDims() != ws_b.getNumDims():
            raise ValueError(
                "'{}' and '{}' have a different number of dimensions "
                "({} vs. {}) and cannot be combined.".format(
                    display_a,
                    display_b,
                    ws_a.getNumDims(),
                    ws_b.getNumDims(),
                )
            )

        for d in range(ws_a.getNumDims()):
            dim_a, dim_b = ws_a.getDimension(d), ws_b.getDimension(d)

            if dim_a.getNBins() != dim_b.getNBins() or not (
                np.isclose(dim_a.getMinimum(), dim_b.getMinimum())
                and np.isclose(dim_a.getMaximum(), dim_b.getMaximum())
            ):
                raise ValueError(
                    "'{}' and '{}' do not share common binning along "
                    "dimension {} ({} bins over [{:.4g}, {:.4g}] vs. "
                    "{} bins over [{:.4g}, {:.4g}]) and cannot be "
                    "combined.".format(
                        display_a,
                        display_b,
                        d,
                        dim_a.getNBins(),
                        dim_a.getMinimum(),
                        dim_a.getMaximum(),
                        dim_b.getNBins(),
                        dim_b.getMinimum(),
                        dim_b.getMaximum(),
                    )
                )

    def combine_workspaces(
        self, ws_a, coeff_a, ws_b, coeff_b, output_display_name
    ):
        """
        Compute ``coeff_a * ws_a - coeff_b * ws_b`` and register the result.

        Uses explicit Mantid algorithms with explicit output names
        throughout (a scalar ``WorkspaceSingleValue`` multiplied via
        ``MultiplyMD``, then ``MinusMD``) rather than Python operator
        overloads, which infer their output name from the caller's
        frame and are unreliable to use inside a method.

        Parameters
        ----------
        ws_a, ws_b : str
            Display names of two already-registered workspaces.
        coeff_a, coeff_b : float
            Scale factors applied before subtracting.
        output_display_name : str
            Display name for the combined result (disambiguated if it
            collides with an existing one).

        Returns
        -------
        display_name : str
            The (possibly disambiguated) display name actually used.

        Raises
        ------
        ValueError
            If ``ws_a`` and ``ws_b`` do not share common binning (same
            number of dimensions, bins, and extents per dimension).
            ``MinusMD`` operates element-wise on the signal arrays, so
            mismatched binning would silently combine voxels that
            don't correspond to the same point in reciprocal space.
        """
        ws_a_name = self.workspace_registry[ws_a]["ws_name"]
        ws_b_name = self.workspace_registry[ws_b]["ws_name"]

        self._check_common_bins(ws_a, ws_a_name, ws_b, ws_b_name)

        display_name = self._unique_display_name(output_display_name)
        output_ws = self._unique_ws_name(display_name)

        scaled_a = self._unique_ws_name(output_ws + "_a")
        scaled_b = self._unique_ws_name(output_ws + "_b")
        coeff_a_ws = self._unique_ws_name(output_ws + "_coeff_a")
        coeff_b_ws = self._unique_ws_name(output_ws + "_coeff_b")

        CreateSingleValuedWorkspace(
            OutputWorkspace=coeff_a_ws, DataValue=coeff_a
        )
        CreateSingleValuedWorkspace(
            OutputWorkspace=coeff_b_ws, DataValue=coeff_b
        )

        MultiplyMD(
            LHSWorkspace=ws_a_name,
            RHSWorkspace=coeff_a_ws,
            OutputWorkspace=scaled_a,
        )
        MultiplyMD(
            LHSWorkspace=ws_b_name,
            RHSWorkspace=coeff_b_ws,
            OutputWorkspace=scaled_b,
        )
        MinusMD(
            LHSWorkspace=scaled_a,
            RHSWorkspace=scaled_b,
            OutputWorkspace=output_ws,
        )

        for ws in (scaled_a, scaled_b, coeff_a_ws, coeff_b_ws):
            DeleteWorkspace(Workspace=ws)

        self.register_workspace(display_name, output_ws, space="reciprocal")

        return display_name

    def _attach_ub_w(self, output_ws, source_ws):
        """
        Attach the sample UB and the ``W_MATRIX`` log onto a workspace.

        ``CreateMDWorkspace`` produces a workspace with no
        ``ExperimentInfo`` at all, so ``set_B()``/``set_W()`` (which
        read ``mtd["histo"].getExperimentInfo(0)``) would silently fall
        back to no-UB/identity-``W`` if this workspace were later
        activated. Mirrors the working pattern already used for this
        in garnet's ``reduction/data.py`` (``add_UBW``): seed an
        ``ExperimentInfo`` via ``AddSampleLog`` (a no-op MD algorithm
        that creates one if none exists), copy the sample/lattice from
        ``source_ws`` via ``CopySample``, then set ``W_MATRIX``
        directly from the model's own (already-known) ``self.W`` --
        there is no separate "real-space W", see :meth:`_basis_matrix`.

        Parameters
        ----------
        output_ws : str
            Workspace to attach UB/``W_MATRIX`` onto (e.g. one just
            built via ``CreateMDWorkspace``/``BinMD``).
        source_ws : str
            Workspace to copy the sample/lattice from.
        """
        AddSampleLog(
            Workspace=output_ws,
            LogName="_seed",
            LogText="0",
            LogType="String",
        )
        CopySample(
            InputWorkspace=source_ws,
            OutputWorkspace=output_ws,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyLattice=True,
            CopyOrientationOnly=False,
        )
        run = mtd[output_ws].getExperimentInfo(0).run()
        run.addProperty("W_MATRIX", list(self.W.flatten() * 1.0), True)

    def _pdf_grid_info(self, ws_name):
        """
        Grid geometry for the delta-PDF pipeline, read from a workspace.

        Mirrors the setup done once in the reference ``DeltaPDF``
        script's constructor, but computed fresh from whichever
        workspace is being operated on, since that can change between
        calls (a different active workspace, or a freshly padded one).

        Parameters
        ----------
        ws_name : str
            Mantid workspace name to read dimensions from.

        Returns
        -------
        dict
            ``xs`` (meshgrid of bin-center coordinates), ``mins``,
            ``maxs``, ``widths`` (per-dimension), and the Miller-index
            bounding box (``h_min``/``h_max``/``k_min``/``k_max``/
            ``l_min``/``l_max``) covering the workspace's extent.
        """
        ws = mtd[ws_name]
        dims = [ws.getDimension(i) for i in range(ws.getNumDims())]

        xs = np.meshgrid(
            *[
                np.linspace(
                    dim.getMinimum() + dim.getBinWidth() / 2,
                    dim.getMaximum() - dim.getBinWidth() / 2,
                    dim.getNBins(),
                )
                for dim in dims
            ],
            indexing="ij",
        )

        mins = [dim.getMinimum() for dim in dims]
        maxs = [dim.getMaximum() for dim in dims]
        widths = [dim.getBinWidth() for dim in dims]

        corners = np.array(
            np.meshgrid(
                [mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]]
            )
        ).reshape(3, -1)

        hkl = np.einsum("ij,jk->ik", self.W, corners)

        h_max, k_max, l_max = np.max(hkl, axis=1).astype(int).tolist()
        h_min, k_min, l_min = np.min(hkl, axis=1).astype(int).tolist()

        return {
            "xs": xs,
            "mins": mins,
            "maxs": maxs,
            "widths": widths,
            "h_min": h_min,
            "h_max": h_max,
            "k_min": k_min,
            "k_max": k_max,
            "l_min": l_min,
            "l_max": l_max,
        }

    def _low_q_mask(self, grid, Q_inner):
        """Boolean mask, True within ``Q_inner`` (Å⁻¹) of the origin."""
        Qx, Qy, Qz = np.einsum(
            "ij,j...->i...", 2 * np.pi * self.UB @ self.W, grid["xs"]
        )
        return (Qx**2 + Qy**2 + Qz**2) / Q_inner**2 < 1

    def run_bragg_punch(
        self,
        input_display_name,
        output_display_name,
        space_group,
        Q_size,
        Q_inner,
        outlier=1.5,
        progress=None,
        stop_event=None,
        **kwargs,
    ):
        """
        Punch Bragg-reflection outliers and mask the low-Q region.

        For every allowed reflection (per ``space_group``) within the
        workspace's extent, examines an ellipsoidal region of
        half-width ``Q_size`` (in Å⁻¹) around that reflection and
        removes (sets to NaN) only the voxels that are statistical
        outliers *relative to that local region* -- following the
        interquartile-range (Tukey's fences) approach used by
        rmc-discord's ``punch``: a voxel is rejected if it falls
        outside ``[Q1 - outlier*IQR, Q3 + outlier*IQR]``, where
        ``Q1``/``Q3`` are the 25th/75th percentiles of the ellipsoid's
        own values and ``IQR = Q3 - Q1``. Unlike unconditionally
        blanking the whole ellipsoid, this keeps genuine diffuse
        scattering that happens to sit near a Bragg peak. The low-Q
        region within ``Q_inner`` of the origin is masked separately
        (that's a beamstop/forward-scattering cut, not a peak). The
        result is registered as a new (still reciprocal-space)
        workspace -- a separate, inspectable step that can be
        sliced/viewed like any ordinary reciprocal-space workspace
        before running :meth:`run_blur` on it.

        Parameters
        ----------
        input_display_name : str
            Display name of the (already-registered) workspace to punch.
        output_display_name : str
            Display name for the punched result.
        space_group : str
            Space-group symbol understood by
            ``mantid.geometry.SpaceGroupFactory.createSpaceGroup``.
        Q_size : float
            Half-width, in Å⁻¹, of the ellipsoidal region examined
            around each allowed reflection.
        Q_inner : float
            Inner cut radius, in Å⁻¹.
        outlier : float, optional
            Tukey's-fences scale factor applied to each reflection's
            local IQR (default 1.5, the conventional value).
        progress : callable, optional
            ``progress(message, percent)`` callback (injected by the
            worker infrastructure).
        stop_event : threading.Event, optional
            Cooperative-cancellation event (injected by the worker
            infrastructure).

        Returns
        -------
        display_name : str
            ``output_display_name``, unchanged -- re-running this step
            with the same output name overwrites its previous result
            in place rather than accumulating new entries.
        """
        input_ws = self.workspace_registry[input_display_name]["ws_name"]

        grid = self._pdf_grid_info(input_ws)

        sg = SpaceGroupFactory.createSpaceGroup(space_group)

        # Physical punch radius (Å⁻¹) -> half-width in bins along each
        # workspace dimension -- same conversion used by
        # _nan_gaussian_blur/_pad for Q_blur/Q_outer.
        bw_inv = np.linalg.inv(2 * np.pi * self.UB @ self.W)
        box = np.maximum(
            np.ceil(
                np.linalg.norm(bw_inv, axis=1)
                * Q_size
                / np.array(grid["widths"])
            ).astype(int),
            1,
        )

        W_inv = np.linalg.inv(self.W)
        mins = np.array(grid["mins"])
        widths = np.array(grid["widths"])

        signal = mtd[input_ws].getSignalArray().copy()
        shape = signal.shape

        h_min, h_max = grid["h_min"], grid["h_max"]
        k_min, k_max = grid["k_min"], grid["k_max"]
        l_min, l_max = grid["l_min"], grid["l_max"]

        total = (h_max - h_min + 1) * (k_max - k_min + 1) * (l_max - l_min + 1)
        n = 0
        for hh in range(h_min, h_max + 1):
            for kk in range(k_min, k_max + 1):
                for ll in range(l_min, l_max + 1):
                    if stop_event is not None and stop_event.is_set():
                        return None

                    n += 1
                    if progress is not None and n % 200 == 0:
                        progress("Bragg punch", int(50 * n / total))

                    if not sg.isAllowedReflection([hh, kk, ll]):
                        continue

                    coord = W_inv @ [hh, kk, ll]
                    idx = np.round((coord - mins) / widths).astype(int)

                    lo = np.maximum(idx - box, 0)
                    hi = np.minimum(idx + box + 1, shape)
                    if np.any(lo >= hi):
                        continue

                    values = signal[
                        lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]
                    ]

                    x, y, z = np.meshgrid(
                        np.arange(lo[0], hi[0]) - idx[0],
                        np.arange(lo[1], hi[1]) - idx[1],
                        np.arange(lo[2], hi[2]) - idx[2],
                        indexing="ij",
                    )
                    outside = (
                        (x / box[0]) ** 2
                        + (y / box[1]) ** 2
                        + (z / box[2]) ** 2
                    ) > 1

                    roi = np.where(outside, np.nan, values)

                    with np.errstate(invalid="ignore"):
                        Q1 = np.nanpercentile(roi, 25)
                        Q3 = np.nanpercentile(roi, 75)
                    iqr = Q3 - Q1

                    with np.errstate(invalid="ignore"):
                        reject = (roi >= Q3 + outlier * iqr) | (
                            roi < Q1 - outlier * iqr
                        )
                    values[reject & ~outside] = np.nan

        if progress is not None:
            progress("Bragg punch", 50)

        signal[self._low_q_mask(grid, Q_inner)] = np.nan

        display_name = output_display_name
        self._replace_workspace_if_exists(display_name, keep_ws_name=input_ws)
        output_ws = self._unique_ws_name(display_name)

        if progress is not None:
            progress("Cloning workspace", 75)
        CloneWorkspace(InputWorkspace=input_ws, OutputWorkspace=output_ws)
        mtd[output_ws].setSignalArray(signal)

        self.register_workspace(display_name, output_ws, space="reciprocal")

        if progress is not None:
            progress("Done", 100)

        return display_name

    def run_blur(
        self,
        input_display_name,
        output_display_name,
        Q_blur,
        progress=None,
        stop_event=None,
        **kwargs,
    ):
        """
        NaN-Gaussian-blur the gaps (e.g. from a Bragg punch) closed.

        Runs, on ``input_display_name`` (typically the output of
        :meth:`run_bragg_punch`, though any registered workspace may be
        used directly): a NaN-aware Gaussian blur of scale ``Q_blur``
        filling the gaps left by the punch/cut. A separate, inspectable
        step -- the result can be sliced/viewed like any ordinary
        reciprocal-space workspace before running :meth:`calculate_pdf`
        on it.

        Parameters
        ----------
        input_display_name : str
            Display name of the (already-registered) workspace to
            blur.
        output_display_name : str
            Display name for the blurred result.
        Q_blur : float
            Gaussian blur scale, in Å⁻¹.
        progress : callable, optional
            ``progress(message, percent)`` callback (injected by the
            worker infrastructure).
        stop_event : threading.Event, optional
            Cooperative-cancellation event (injected by the worker
            infrastructure).

        Returns
        -------
        display_name : str
            ``output_display_name``, unchanged -- re-running this step
            with the same output name overwrites its previous result
            in place rather than accumulating new entries.
        """
        input_ws = self.workspace_registry[input_display_name]["ws_name"]

        if progress is not None:
            progress("Blurring gaps", 0)
        grid = self._pdf_grid_info(input_ws)
        signal = mtd[input_ws].getSignalArray().copy()
        signal = self._nan_gaussian_blur(signal, grid, Q_blur)

        if stop_event is not None and stop_event.is_set():
            return None

        display_name = output_display_name
        self._replace_workspace_if_exists(display_name, keep_ws_name=input_ws)
        output_ws = self._unique_ws_name(display_name)

        if progress is not None:
            progress("Cloning workspace", 80)
        CloneWorkspace(InputWorkspace=input_ws, OutputWorkspace=output_ws)
        mtd[output_ws].setSignalArray(signal)

        self.register_workspace(display_name, output_ws, space="reciprocal")

        if progress is not None:
            progress("Done", 100)

        return display_name

    def run_karen(
        self,
        input_display_name,
        output_display_name,
        width,
        z_score=3.0,
        progress=None,
        stop_event=None,
        **kwargs,
    ):
        """
        Remove Bragg-peak outliers with the KAREN algorithm.

        An alternative to :meth:`run_bragg_punch` + :meth:`run_blur`,
        following Mantid's ``DeltaPDF3D`` "KAREN" method: a moving-window
        median filter estimates a local median and median absolute
        deviation (MAD) at every voxel; voxels more than ``z_score``
        (robust, MAD-based) standard deviations from the local median
        are replaced with ``median + 2.2*MAD``. Since this replaces
        outliers in place rather than punching NaNs out, no separate
        fill/blur step is needed before :meth:`calculate_pdf`. A
        separate, inspectable step -- the result can be sliced/viewed
        like any ordinary reciprocal-space workspace.

        Parameters
        ----------
        input_display_name : str
            Display name of the (already-registered) workspace to
            filter.
        output_display_name : str
            Display name for the KAREN-filtered result.
        width : float
            Moving-window size, in Å⁻¹ -- the same meaning as
            :meth:`run_blur`'s ``Q_blur``, converted per-axis into
            filter-window bins.
        z_score : float, optional
            Outlier cutoff, in estimated standard deviations
            (1.4826*MAD) from the local median (default 3.0, matching
            Mantid's ``DeltaPDF3D`` "KAREN" method).
        progress : callable, optional
            ``progress(message, percent)`` callback (injected by the
            worker infrastructure).
        stop_event : threading.Event, optional
            Cooperative-cancellation event (injected by the worker
            infrastructure).

        Returns
        -------
        display_name : str
            ``output_display_name``, unchanged -- re-running this step
            with the same output name overwrites its previous result
            in place rather than accumulating new entries.
        """
        input_ws = self.workspace_registry[input_display_name]["ws_name"]

        if progress is not None:
            progress("Running KAREN", 5)
        grid = self._pdf_grid_info(input_ws)
        signal = mtd[input_ws].getSignalArray().copy()
        signal = self._karen_filter(signal, grid, width, z_score)

        if stop_event is not None and stop_event.is_set():
            return None

        display_name = output_display_name
        self._replace_workspace_if_exists(display_name, keep_ws_name=input_ws)
        output_ws = self._unique_ws_name(display_name)

        if progress is not None:
            progress("Cloning workspace", 80)
        CloneWorkspace(InputWorkspace=input_ws, OutputWorkspace=output_ws)
        mtd[output_ws].setSignalArray(signal)

        self.register_workspace(display_name, output_ws, space="reciprocal")

        if progress is not None:
            progress("Done", 100)

        return display_name

    def calculate_pdf(
        self,
        input_display_name,
        output_display_name,
        Q_outer,
        window="None",
        progress=None,
        stop_event=None,
        **kwargs,
    ):
        """
        Zero-pad and Fourier-transform a (typically blurred) workspace.

        Runs, on ``input_display_name`` (typically the output of
        :meth:`run_blur`, though any registered workspace may be used
        directly): a zero-pad out to ``Q_outer``, an optional
        apodization window to suppress FFT series-termination ripples,
        then an FFT to real space. Registers the result as a new
        **real-space** workspace (see ``self.workspace_registry``).

        Parameters
        ----------
        input_display_name : str
            Display name of the (already-registered) workspace to
            transform.
        output_display_name : str
            Display name for the delta-PDF result.
        Q_outer : float
            Zero-pad extent, in Å⁻¹.
        window : str, optional
            Apodization window applied before the FFT: ``"None"``
            (default), ``"Lorch"``, or ``"Hann"`` -- see
            :meth:`_apodization_window`.
        progress : callable, optional
            ``progress(message, percent)`` callback (injected by the
            worker infrastructure).
        stop_event : threading.Event, optional
            Cooperative-cancellation event (injected by the worker
            infrastructure).

        Returns
        -------
        display_name : str
            ``output_display_name``, unchanged -- re-running this step
            with the same output name overwrites its previous result
            in place rather than accumulating new entries.
        """
        input_ws = self.workspace_registry[input_display_name]["ws_name"]

        if progress is not None:
            progress("Padding", 0)
        grid = self._pdf_grid_info(input_ws)
        signal = mtd[input_ws].getSignalArray().copy()
        pad_ws = self._unique_ws_name(input_ws + "_pad")
        self._pad(input_ws, signal, grid, pad_ws, Q_outer)

        if stop_event is not None and stop_event.is_set():
            DeleteWorkspace(Workspace=pad_ws)
            return None

        if progress is not None:
            progress("Fourier transform", 50)
        display_name = output_display_name
        self._replace_workspace_if_exists(display_name, keep_ws_name=input_ws)
        output_ws = self._unique_ws_name(display_name)
        self._transform(pad_ws, input_ws, output_ws, window=window)

        DeleteWorkspace(Workspace=pad_ws)

        self.register_workspace(display_name, output_ws, space="real")

        if progress is not None:
            progress("Done", 100)

        return display_name

    def _nan_gaussian_blur(self, signal, grid, Q_size):
        """NaN-aware Gaussian blur, filling gaps (NaNs) from surrounding data."""
        bw_inv = np.linalg.inv(2 * np.pi * self.UB @ self.W)
        sizes = np.linalg.norm(bw_inv, axis=1) * Q_size
        sigma = np.ceil(sizes / np.array(grid["widths"])).astype(int)

        mask = np.isfinite(signal)
        filled = np.nan_to_num(signal, nan=0.0)
        smoothed = gaussian_filter(filled * mask, sigma=sigma)
        norm = gaussian_filter(mask.astype(float), sigma=sigma)
        interp = smoothed / np.maximum(norm, 1e-8)

        signal = signal.copy()
        signal[~mask] = interp[~mask]

        return signal

    def _karen_filter(self, signal, grid, width, z_score):
        """
        Moving-window median/MAD outlier filter (Mantid's KAREN).

        Same median-filter-based sigma estimate (1.4826*MAD) and
        outlier replacement (median + 2.2*MAD) as ``DeltaPDF3D``'s
        ``_karen``, but with ``width`` given in Å⁻¹ (like
        ``_nan_gaussian_blur``'s ``Q_size``) rather than a raw bin
        count, and ``z_score`` exposed instead of Mantid's hard-coded
        3.
        """
        bw_inv = np.linalg.inv(2 * np.pi * self.UB @ self.W)
        sizes = np.linalg.norm(bw_inv, axis=1) * width
        size = np.maximum(
            np.ceil(sizes / np.array(grid["widths"])).astype(int), 1
        )
        # Force odd window sizes -- an even window is asymmetric about
        # the voxel being filtered (see DeltaPDF3D's KAREN docstring).
        size += 1 - size % 2

        med = median_filter(signal, size=tuple(size), mode="nearest")
        mad = median_filter(
            np.abs(signal - med), size=tuple(size), mode="nearest"
        )
        asigma = np.abs(1.4826 * mad)
        mask = np.abs(signal - med) > z_score * asigma

        signal = signal.copy()
        signal[mask] = (med + 2.2 * mad)[mask]

        return signal

    def _pad(self, source_ws, signal, grid, output_ws, Q_outer):
        """Zero-pad ``signal`` out to ``Q_outer`` and build the padded workspace."""
        bw_inv = np.linalg.inv(2 * np.pi * self.UB @ self.W)
        lims = np.linalg.norm(bw_inv, axis=1) * Q_outer

        maxs, mins, widths = grid["maxs"], grid["mins"], grid["widths"]

        pad_width = []
        for i in range(3):
            before = (
                int((lims[i] - maxs[i]) / widths[i])
                if lims[i] > maxs[i]
                else 0
            )
            after = (
                int((lims[i] + mins[i]) / widths[i])
                if -lims[i] < mins[i]
                else 0
            )
            pad_width.append([before, after])

        padded = np.pad(
            signal, pad_width=pad_width, mode="constant", constant_values=0
        )

        ws = mtd[source_ws]

        names, extents, number_of_bins = [], [], []
        for d in range(ws.getNumDims()):
            dim = ws.getDimension(d)
            names.append(dim.name)
            number_of_bins.append(dim.getNBins() + sum(pad_width[d]))
            extents.append(
                dim.getMinimum() - dim.getBinWidth() * pad_width[d][0]
            )
            extents.append(
                dim.getMaximum() + dim.getBinWidth() * pad_width[d][1]
            )

        dim_bins = [
            "{},{},{},{}".format(
                names[d], extents[2 * d], extents[2 * d + 1], number_of_bins[d]
            )
            for d in range(3)
        ]

        CreateMDWorkspace(
            Dimensions=3,
            Extents=extents,
            Names=names,
            Units=3 * ["r.l.u."],
            OutputWorkspace=output_ws,
        )
        BinMD(
            InputWorkspace=output_ws,
            AlignedDim0=dim_bins[0],
            AlignedDim1=dim_bins[1],
            AlignedDim2=dim_bins[2],
            OutputWorkspace=output_ws,
        )

        self._attach_ub_w(output_ws, source_ws)

        mtd[output_ws].setSignalArray(padded)

    @staticmethod
    def _apodization_window(shape, window):
        """
        Separable 3D apodization window, tapering each axis to zero at
        its edges to suppress FFT series-termination ripples.

        Parameters
        ----------
        shape : tuple of int
            Shape of the (padded) array to build a window for.
        window : str
            ``"None"`` (no taper), ``"Lorch"`` (``sinc`` taper, the
            conventional PDF choice -- reaches exactly zero at the
            edge), or ``"Hann"`` (raised-cosine taper).

        Returns
        -------
        w : np.ndarray or float
            3D window array broadcastable against ``shape``, or
            ``1.0`` if ``window == "None"``.
        """
        if window == "None":
            return 1.0

        axes = []
        for n in shape:
            if n <= 1:
                axes.append(np.ones(n))
                continue
            # z runs from -1 (first bin) to +1 (last bin), 0 at center.
            z = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
            if window == "Lorch":
                axes.append(np.sinc(z))
            elif window == "Hann":
                axes.append(0.5 * (1 + np.cos(np.pi * z)))
            else:
                raise ValueError(
                    "Unknown apodization window: {}".format(window)
                )

        return (
            axes[0][:, None, None]
            * axes[1][None, :, None]
            * axes[2][None, None, :]
        )

    def _transform(self, pad_ws, source_ws, output_ws, window="None"):
        """Fourier-transform a padded workspace's signal to real space."""
        ws = mtd[pad_ws]

        signal = ws.getSignalArray().copy()
        signal[np.isnan(signal)] = 0
        signal[np.isinf(signal)] = 0

        signal = signal * self._apodization_window(signal.shape, window)

        signal = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(signal)))
        number_of_bins = signal.shape
        signal = signal.real

        extents = []
        for d in range(ws.getNumDims()):
            dim = ws.getDimension(d)
            if dim.getNBins() == 1:
                fft_dim = 0.5 / (dim.getMaximum() - dim.getMinimum())
                extents += [-fft_dim, fft_dim]
            else:
                step = (
                    dim.getMaximum() - dim.getMinimum() - dim.getBinWidth()
                ) / dim.getNBins()
                fft_dim = np.fft.fftshift(np.fft.fftfreq(dim.getNBins(), step))
                extents += [fft_dim[0], fft_dim[-1]]

        # Unitless naming factor for the real-space axes -- see
        # _basis_matrix for why this (rather than a separately-derived
        # "real-space W") is the correct dual of W.
        w = np.linalg.inv(self.W).T
        char_dict = {0: "0", 1: "{1}", -1: "-{1}"}
        chars = ["X", "Y", "Z"]
        names = [
            "["
            + ",".join(
                char_dict.get(j, "{0}{1}").format(
                    j, chars[np.argmax(np.abs(w[:, i]))]
                )
                for j in w[:, i]
            )
            + "]"
            for i in range(3)
        ]

        dim_bins = [
            "{},{},{},{}".format(
                names[d], extents[2 * d], extents[2 * d + 1], number_of_bins[d]
            )
            for d in range(3)
        ]

        CreateMDWorkspace(
            Dimensions=3,
            Extents=extents,
            Names=names,
            Units=3 * ["d.l.u."],
            OutputWorkspace=output_ws,
        )
        BinMD(
            InputWorkspace=output_ws,
            AlignedDim0=dim_bins[0],
            AlignedDim1=dim_bins[1],
            AlignedDim2=dim_bins[2],
            OutputWorkspace=output_ws,
        )

        self._attach_ub_w(output_ws, source_ws)

        mtd[output_ws].setSignalArray(signal)

    # ------------------------------------------------------------------
    # Bond-network viewer
    # ------------------------------------------------------------------
    #
    # Loads a CIF and builds the distinct bond (separation-vector) set
    # and unit-cell-edge wireframe for display in the shared 3D view.
    # Deliberately uses its own workspace ("pdf_crystal") rather than
    # CrystalStructureModel's "crystal" -- both models can be alive at
    # once in the same Mantid session (see application.py), and
    # CrystalStructureModel's own docstrings already flag "crystal" as
    # unsafe to share across tools.
    #
    # Bonds are not found via a covalent-radius distance cutoff (see
    # build_bond_network) -- this tool samples 3D-ΔPDF correlation
    # strength at a separation vector, so "is this a real chemical
    # bond" isn't a relevant question here. Instead they follow the
    # reference DiffuseRefinement script's construction: symmetry-
    # expand each site within the unit cell, take every pairwise
    # separation there, wrap/dedupe, then extend by periodic images.

    BOND_CIF_WORKSPACE = "pdf_crystal"

    def load_bond_cif(
        self, filename, progress=None, stop_event=None, **kwargs
    ):
        """
        Load a CIF for the bond-network viewer.

        Parameters
        ----------
        filename : str
            Path to the CIF file to load.
        progress : callable, optional
            ``progress(message, percent)`` callback (injected by the
            worker infrastructure).
        stop_event : threading.Event, optional
            Unused; accepted because the presenter runs this on a
            worker thread, which always passes it (injected by the
            worker infrastructure).

        Returns
        -------
        sites : list of dict
            One entry per asymmetric-unit site, each with keys
            ``"label"`` (an auto-generated ``"{element}{n}"`` label,
            since Mantid's `CrystalStructure` does not retain the
            CIF's own `_atom_site_label` values), ``"element"``, and
            fractional ``"x"``/``"y"``/``"z"``. Unchecking a site in
            the atom table excludes it from every pair considered by
            :meth:`build_bond_network`.
        """
        if progress is not None:
            progress("Loading CIF", 10)

        ws = self.BOND_CIF_WORKSPACE

        if mtd.doesExist(ws):
            DeleteWorkspace(Workspace=ws)

        CreateSampleWorkspace(OutputWorkspace=ws)
        LoadCIF(Workspace=ws, InputFile=filename)

        cs = mtd[ws].sample().getCrystalStructure()

        uc = cs.getUnitCell()
        G = uc.getG()
        self._bond_A = scipy.linalg.cholesky(G, lower=False)
        self._bond_sg = cs.getSpaceGroup()

        scatterers = [s.split(" ") for s in list(cs.getScatterers())]
        scatterers = [
            [val if val.isalpha() else float(val) for val in scatterer]
            for scatterer in scatterers
        ]

        element_counts = {}
        sites = []

        for element, x, y, z, occ, U in scatterers:
            element_counts[element] = element_counts.get(element, 0) + 1
            label = "{}{}".format(element, element_counts[element])

            sites.append(
                {"label": label, "element": element, "x": x, "y": y, "z": z}
            )

        self._bond_sites = sites

        if progress is not None:
            progress("Done", 100)

        return sites

    def build_bond_network(
        self,
        nx,
        ny,
        nz,
        site_enabled,
        tolerance=1e-3,
        progress=None,
        stop_event=None,
        **kwargs,
    ):
        """
        Build the distinct bond (separation-vector) set and
        unit-cell-edge wireframe for the bond-network viewer.

        Bonds are not found via a covalent-radius distance cutoff --
        that isn't a meaningful criterion here (this tool samples
        3D-ΔPDF correlation strength at a separation vector; it
        doesn't need "is this a real chemical bond"). Instead, this
        follows the same construction as the reference
        ``DiffuseRefinement`` script's ``initialize_unit_cell_atoms``
        + ``pair_expansion``:

        1. Every enabled site is expanded by the space group's
           symmetry operations into all of its equivalent positions
           within the unit cell, each wrapped into ``[0, 1)``.
        2. Every pairwise separation vector between these expanded
           positions is computed (the full N×N set, not just an
           upper triangle) -- these are the bonds *within the unit
           cell*.
        3. Each is wrapped into ``[0, 1)`` (a unit cell's own
           fractional coordinates always span that range) and
           deduplicated (within `tolerance`) to the set of distinct
           new positions.
        4. Periodic images of each are then generated by adding every
           requested lattice translation to it, and each resulting
           vector is explicitly unioned with its own negation before
           deduping again. This step is required, not optional:
           wrapping to ``[0, 1)`` in step 3 is one-sided (it shifts
           negative components up by 1, never shifts positive ones
           down), so a vector's negation is generally *not* produced
           naturally by this same translation loop at the mirror
           translation -- verified empirically (corundum): without
           this explicit union, 34-66% of the extended set was
           missing its negation partner, not a rare edge effect.
           Explicitly including it also has the side benefit of
           centering the final set on the origin regardless of where
           the wrapped `unit_cell_bonds` themselves happen to sit,
           matching the unit-cell wireframe (see below).

        Parameters
        ----------
        nx, ny, nz : int
            Number of unit cells to repeat along a, b, c in each
            direction (translations run ``-n...+n`` inclusive along
            each axis, i.e. ``2n+1`` cells -- an odd count is what
            keeps the translation range closed under negation, which
            is what makes the extended bond set centrosymmetric; see
            above).
        site_enabled : list of bool
            One flag per site returned by :meth:`load_bond_cif`
            (same order); disabled sites (and all of their
            symmetry-equivalent copies) are excluded.
        tolerance : float, optional
            Fractional-coordinate tolerance used to collapse
            separation vectors that are effectively the same
            crystallographic bond (default 1e-3).
        progress : callable, optional
            ``progress(message, percent)`` callback (injected by the
            worker infrastructure).
        stop_event : threading.Event, optional
            Cooperative-cancellation event (injected by the worker
            infrastructure).

        Returns
        -------
        geometry : dict
            Dictionary with keys ``"bond_vectors"`` ((N, 3) array of
            Cartesian separation vectors, Angstrom, each measured
            from an implicit origin) and ``"edges"`` (list of (p0,
            p1) Cartesian line segments for the repeated unit-cell
            wireframe). None if cancelled via `stop_event`.
        """
        A = self._bond_A
        sg = self._bond_sg

        translations = [
            np.array([i, j, k])
            for i in range(-nx, nx)
            for j in range(-ny, ny)
            for k in range(-nz, nz)
        ]

        if progress is not None:
            progress("Expanding unit cell", 10)

        # --- Step 1: symmetry-expand enabled sites within [0, 1)^3 ------
        locs = []
        for site_index, enabled in enumerate(site_enabled):
            if not enabled:
                continue

            if stop_event is not None and stop_event.is_set():
                return None

            site = self._bond_sites[site_index]
            xyz = np.mod([site["x"], site["y"], site["z"]], 1)

            equiv = np.mod(
                np.array(sg.getEquivalentPositions(xyz.tolist())), 1
            )
            equiv, _ = self._unique_rows(equiv, tol=1e-4)

            locs.append(equiv)

        locs = np.vstack(locs) if locs else np.empty((0, 3))

        if progress is not None:
            progress("Finding bonds within unit cell", 30)

        # --- Step 2: all pairwise separations within the unit cell ------
        n = len(locs)
        if n > 1:
            i, j = np.indices((n, n))
            dr = locs[i.ravel()] - locs[j.ravel()]
        else:
            dr = np.empty((0, 3))

        if stop_event is not None and stop_event.is_set():
            return None

        # --- Step 3: wrap to [0, 1)^3 and find unique new positions -----
        # A unit cell's own fractional coordinates always span [0, 1),
        # so bonds "within the unit cell" belong in that same range.
        dr = np.mod(dr, 1)
        unit_cell_bonds, _ = self._unique_rows(dr, tol=tolerance)

        if progress is not None:
            progress("Generating periodic images", 60)

        # --- Step 4: periodic images ±(nx, ny, nz) -----------------------
        n_done = 0
        seen = set()
        bond_vectors = []

        for v0 in unit_cell_bonds:
            if stop_event is not None and stop_event.is_set():
                return None

            n_done += 1
            if progress is not None:
                progress(
                    "Generating periodic images",
                    60 + int(20 * n_done / max(len(unit_cell_bonds), 1)),
                )

            for translation in translations:
                vec_frac = v0 + translation
                key = tuple(np.round(vec_frac / tolerance).astype(np.int64))
                if key in seen:
                    continue
                seen.add(key)

                bond_vectors.append(vec_frac @ A)

        bond_vectors = (
            np.array(bond_vectors) if bond_vectors else np.empty((0, 3))
        )

        if progress is not None:
            progress("Building unit-cell edges", 80)

        cell_corners = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        cell_edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        seen_edges = set()
        edges = []
        for translation in translations:
            if stop_event is not None and stop_event.is_set():
                return None

            corners_cart = (cell_corners + translation) @ A
            for i, j in cell_edges:
                a, b = corners_cart[i], corners_cart[j]
                ka = tuple(np.round(a / 1e-3).astype(np.int64))
                kb = tuple(np.round(b / 1e-3).astype(np.int64))
                key = tuple(sorted((ka, kb)))
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append((a, b))

        if progress is not None:
            progress("Done", 100)

        return {
            "bond_vectors": bond_vectors,
            "edges": edges,
        }

    @staticmethod
    def _unique_rows(xyz, tol=1e-4):
        """
        Deduplicate rows of `xyz` that agree within `tol`.

        Parameters
        ----------
        xyz : (N, 3) array-like
            Rows to deduplicate.
        tol : float, optional
            Absolute tolerance used to snap rows to a common grid
            before comparing them (default 1e-4).

        Returns
        -------
        unique : (M, 3) ndarray
            Deduplicated rows, in their first-occurrence order.
        index : (M,) ndarray
            Indices into the original `xyz` selected as `unique`.
        """
        xyz = np.asarray(xyz, dtype=float)

        if len(xyz) == 0:
            return xyz, np.array([], dtype=int)

        rounded = np.round(xyz / tol).astype(np.int64)
        _, index = np.unique(rounded, axis=0, return_index=True)
        index = np.sort(index)

        return xyz[index], index

    def _read_ub_w(self, ws_name):
        """
        Read the UB (really just B -- no orientation, matching
        :meth:`set_B`) and W matrices directly from a workspace's own
        `ExperimentInfo`, without disturbing `self.UB`/`self.W` (which
        track whichever workspace is currently *active* for the Slice
        tab). Lets :meth:`sample_bond_correlations` sample any
        registered real-space workspace the user selects, independent
        of what's active elsewhere.

        Parameters
        ----------
        ws_name : str
            Mantid workspace name (a registry entry's ``"ws_name"``,
            not its display name) to read from.

        Returns
        -------
        UB : (3, 3) ndarray
        W : (3, 3) ndarray
            Identity if the workspace has no ``W_MATRIX`` log.
        """
        ei = mtd[ws_name].getExperimentInfo(0)

        UB = ei.sample().getOrientedLattice().getB().copy()

        W = np.eye(3)
        if ei.run().hasProperty("W_MATRIX"):
            W = np.array(ei.run().getLogData("W_MATRIX").value).reshape(3, 3)

        return UB, W

    def sample_bond_correlations(
        self,
        geometry,
        display_name,
        tolerance=1e-3,
        progress=None,
        stop_event=None,
        **kwargs,
    ):
        """
        Sample a 3D-ΔPDF workspace at each bond's separation vector.

        Maps each bond's Cartesian separation vector to its nearest
        grid coordinate in the workspace's own real-space basis and
        reads that voxel's signed value directly -- no neighborhood
        search or shell integration is performed.

        A ΔPDF value at real-space vector ``r`` reflects the
        correlated pair-density for atom pairs separated by ``r`` --
        since this depends only on the separation (not the pair's
        absolute position), every translationally-equivalent copy of
        a bond across the supercell samples the identical grid
        location automatically, with no separate symmetry-family
        grouping step needed.

        Reads ``display_name``'s own UB/W directly (via
        :meth:`_read_ub_w`), not the CIF's own basis, nor
        `self.UB`/`self.W` (whatever's *active* in the Slice tab) --
        the workspace sampled here can be any registered real-space
        result the user selects, and need not be the active one. Its
        own UB/W need not agree with the CIF's basis either, if it was
        built with a non-identity `W_MATRIX` projection.

        Parameters
        ----------
        geometry : dict
            Geometry dictionary from :meth:`build_bond_network`
            (only ``"bond_vectors"`` is used).
        display_name : str
            Display name of a registered real-space (3D-ΔPDF)
            workspace -- see `self.workspace_registry`.
        tolerance : float, optional
            Per-axis slack (in grid-index units, applied
            independently to x, y, and z), allowed when checking that
            a bond's nearest grid coordinate falls within the
            workspace's extent -- guards against floating-point
            roundoff putting an on-boundary index just outside
            ``[0, n_bins)`` (default 1e-3).
        progress : callable, optional
            ``progress(message, percent)`` callback (injected by the
            worker infrastructure).
        stop_event : threading.Event, optional
            Cooperative-cancellation event (injected by the worker
            infrastructure).

        Returns
        -------
        values : (len(bond_vectors),) ndarray or None
            Signed ΔPDF value sampled for each bond (NaN where the
            vector falls outside the workspace's extent, or its
            nearest voxel is itself NaN). None if ``display_name``
            isn't a registered real-space workspace, or if cancelled
            via `stop_event`.
        """
        entry = self.workspace_registry.get(display_name)

        if entry is None or entry.get("space") != "real":
            return None

        ws_name = entry["ws_name"]

        if not self.has_UB(ws_name):
            return None

        if progress is not None:
            progress("Sampling ΔPDF", 10)

        UB, W = self._read_ub_w(ws_name)
        Bp = np.dot(UB, W)
        Ap = np.linalg.inv(Bp).T
        Ap_inv = np.linalg.inv(Ap)

        ws = mtd[ws_name]
        dims = [ws.getDimension(i) for i in range(3)]
        mins = np.array([dim.getMinimum() for dim in dims])
        widths = np.array([dim.getBinWidth() for dim in dims])
        shape = np.array([dim.getNBins() for dim in dims])

        signal = ws.getSignalArray()

        bond_vectors = geometry["bond_vectors"]

        values = np.full(len(bond_vectors), np.nan)

        for n, r in enumerate(bond_vectors):
            if stop_event is not None and stop_event.is_set():
                return None

            if progress is not None and n % 200 == 0:
                progress(
                    "Sampling ΔPDF",
                    10 + int(80 * n / max(len(bond_vectors), 1)),
                )

            frac = Ap_inv @ r

            idx = (frac - mins) / widths

            # A bond's ideal separation vector can easily fall well
            # outside the workspace's own real-space extent -- e.g. a
            # large supercell's longest separations vs. a modestly
            # padded ΔPDF result -- in which case it must be excluded
            # entirely (NaN), not silently clamped to the boundary.
            if np.any(idx < -tolerance) or np.any(idx > shape - 1 + tolerance):
                continue

            nearest = np.clip(np.round(idx), 0, shape - 1).astype(int)
            value = signal[nearest[0], nearest[1], nearest[2]]

            if not np.isfinite(value):
                continue

            values[n] = value

        if progress is not None:
            progress("Done", 100)

        return values
