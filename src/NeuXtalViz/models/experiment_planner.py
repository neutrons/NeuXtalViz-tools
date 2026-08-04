import os
import glob
import shutil

import csv
import itertools

from collections import defaultdict

from mantid.simpleapi import (
    CreatePeaksWorkspace,
    ConvertPeaksWorkspace,
    PredictPeaks,
    FilterPeaks,
    CombinePeaksWorkspaces,
    SortPeaksWorkspace,
    AddPeakHKL,
    CountReflections,
    HFIRCalculateGoniometer,
    SetUB,
    SetGoniometer,
    LoadNexus,
    SaveNexus,
    LoadIsawUB,
    LoadEmptyInstrument,
    LoadInstrument,
    LoadMask,
    LoadIsawDetCal,
    LoadParameterFile,
    MaskDetectors,
    MaskDetectorsIf,
    ExtractMonitors,
    PreprocessDetectorsToMD,
    CreateMDWorkspace,
    BinMD,
    GroupDetectors,
    MaskBTP,
    AddSampleLog,
    CreateSampleWorkspace,
    CreateEmptyTableWorkspace,
    DeleteTableRows,
    CloneWorkspace,
    DeleteWorkspace,
    RenameWorkspaces,
    HasUB,
    mtd,
)

from mantid.kernel import V3D
from mantid.geometry import PointGroupFactory

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import binned_statistic_2d
import scipy.linalg
import matplotlib.pyplot as plt

import skimage

from NeuXtalViz.models.base_model import NeuXtalVizModel
from NeuXtalViz.config.instruments import beamlines

point_group_centering = {
    "1": ["P"],
    "-1": ["P"],
    "2": ["P", "C"],
    "m": ["P", "C"],
    "2/m": ["P", "C"],
    "112": ["P", "C"],
    "11m": ["P", "C"],
    "112/m": ["P", "C"],
    "222": ["P", "I", "F", "C", "A", "B"],
    "mm2": ["P", "I", "F", "C", "A", "B"],
    "mmm": ["P", "I", "F", "C", "A", "B"],
    "4": ["P", "I"],
    "-4": ["P", "I"],
    "4/m": ["P", "I"],
    "422": ["P", "I"],
    "4mm": ["P", "I"],
    "-42m": ["P", "I"],
    "-4m2": ["P", "I"],
    "4/mmm": ["P", "I"],
    "3 r": ["P"],
    "-3 r": ["P"],
    "32 r": ["P"],
    "3m r": ["P"],
    "-3m r": ["P"],
    "3": ["P", "Robv", "Rrev"],
    "-3": ["P", "Robv", "Rrev"],
    "312": ["P", "Robv", "Rrev"],
    "31m": ["P", "Robv", "Rrev"],
    "32": ["P", "Robv", "Rrev"],
    "321": ["P", "Robv", "Rrev"],
    "3m": ["P", "Robv", "Rrev"],
    "-31m": ["P", "Robv", "Rrev"],
    "-3m": ["P", "Robv", "Rrev"],
    "-3m1": ["P", "Robv", "Rrev"],
    "6": ["P"],
    "-6": ["P"],
    "6/m": ["P"],
    "622": ["P"],
    "6mm": ["P"],
    "-62m": ["P"],
    "-6m2": ["P"],
    "6/mmm": ["P"],
    "23": ["P", "I", "F"],
    "m-3": ["P", "I", "F"],
    "432": ["P", "I", "F"],
    "-43m": ["P", "I", "F"],
    "m-3m": ["P", "I", "F"],
}

crystal_system_point_groups = {
    "Triclinic": ["1", "-1"],
    "Monoclinic": ["2", "m", "2/m", "112", "11m", "112/m"],
    "Orthorhombic": ["222", "mm2", "mmm"],
    "Tetragonal": ["4", "-4", "4/m", "422", "4mm", "-42m", "-4m2", "4/mmm"],
    "Trigonal/Rhombohedral": ["3 r", "-3 r", "32 r", "3m r", "-3m r"],
    "Trigonal/Hexagonal": [
        "3",
        "-3",
        "312",
        "31m",
        "32",
        "321",
        "3m",
        "-31m",
        "-3m",
        "-3m1",
    ],
    "Hexagonal": ["6", "-6", "6/m", "622", "6mm", "-62m", "-6m2", "6/mmm"],
    "Cubic": ["23", "m-3", "432", "-43m", "m-3m"],
}

centering_conditions = {
    "P": lambda h, k, l: True,
    "I": lambda h, k, l: (h + k + l) % 2 == 0,
    "F": lambda h, k, l: (h % 2 == k % 2 == l % 2),
    "C": lambda h, k, l: (h + k) % 2 == 0,
    "A": lambda h, k, l: (k + l) % 2 == 0,
    "B": lambda h, k, l: (h + l) % 2 == 0,
    "R": lambda h, k, l: True,
    "Robv": lambda h, k, l: (-h + k + l) % 3 == 0,
    "Rrev": lambda h, k, l: (h - k + l) % 3 == 0,
}

AUTOLITE = "/SNS/software/scd/lite/"


class ExperimentModel(NeuXtalVizModel):
    """
    Model for managing experiment planning, instrument setup, and
    crystallographic calculations in NeuXtalViz.

    This class provides methods for initializing instruments, handling
    calibration and mask files, managing sample and plan workspaces,
    performing peak prediction, and calculating experiment statistics.
    """

    def __init__(self):
        """
        Initialize the experiment model and create the ``coverage`` peaks
        workspace used to track the UB matrix and predicted coverage.
        """

        super(ExperimentModel, self).__init__()

        CreatePeaksWorkspace(
            NumberOfPeaks=0,
            OutputType="LeanElasticPeak",
            OutputWorkspace="coverage",
        )

        self.comment = ""
        self.hkl = None
        self.hkl_alt = None
        self.dirname = None
        self.instrument_background = None
        self.last_meshmap = None
        self._count_refl_cache = {}

    def get_instrument_directory(self, instrument):
        """
        Get the file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        filepath : str
            File path to instrument experiment data.
        """

        inst = beamlines[instrument]

        directory = os.path.join(
            "/",
            inst["Facility"],
            inst["InstrumentName"],
        )

        return directory if self.dirname is None else self.dirname

    def get_autoreduce_instrument(self, instrument):
        """
        Get the autoreduce instrument definition file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        str
            Instrument definition file path.
        """

        name = beamlines[instrument]["Name"]

        idf = glob.glob(
            os.path.join(AUTOLITE, "{}_Definition*.xml".format(name))
        )

        return os.path.join(AUTOLITE, idf[0]) if len(idf) > 0 else None

    def copy_to_instrument_pc(self, filename):
        """
        Copy a file to the TOPAZ instrument PC data share, if applicable.

        Only copies the file when it lives under an SNS TOPAZ IPTS
        directory; otherwise this is a no-op.

        Parameters
        ----------
        filename : str
            Path of the file to copy, expected to start with
            ``/SNS/TOPAZ/IPTS-<n>/...``.
        """

        split = filename.split("/")
        if len(split) > 4:
            _, facility, instrument, ipts, *_ = split
            if (
                facility == "SNS"
                and instrument == "TOPAZ"
                and ipts.startswith("IPTS-")
            ):
                output = "/SNS/groups/topaz/bl_12/{}".format(ipts)
                print("Copying {} to {}".format(filename, output))
                os.makedirs(output, exist_ok=True)
                try:
                    os.chmod(output, 0o777)
                except:
                    print("Failed to set permissions for {}".format(output))
                copy = os.path.join(output, os.path.basename(filename))
                shutil.copy(filename, copy)
                os.chmod(copy, 0o777)

    def set_path(self, filename):
        """
        Remember the directory of a file for use as the default directory.

        Parameters
        ----------
        filename : str
            File path whose containing directory should be stored.
        """

        self.dirname = os.path.dirname(filename)

    def initialize_instrument(self, instrument, logs, cal, mask):
        """
        Build the ``instrument`` and related workspaces.

        Loads the empty instrument definition (or autoreduce IDF), applies
        sample logs, calibration, and mask files, then groups detectors
        and precomputes bank-corner coordinates used for the instrument
        3D view and background occupancy plot.

        Parameters
        ----------
        instrument : str
            Instrument identifier (key into ``beamlines``).
        logs : dict
            Mapping of auxiliary motor names to numeric values (from
            :meth:`get_motors`/the view's motor table), added as sample
            logs on the instrument workspace before (re)loading the
            instrument definition, so motor-dependent geometry is applied.
        cal : str
            Path to a calibration file (``.xml`` parameter file or ISAW
            DetCal file), or an empty string if none.
        mask : str
            Path to a mask file, or an empty string if none.
        """

        inst = self.get_instrument_name(instrument)
        idf = self.get_autoreduce_instrument(instrument)
        beamline = beamlines[instrument]

        self.pixel_size = beamline["PixelSize"]
        self.grouping_c, self.grouping_r = [
            int(v) for v in beamline["Grouping"].split("x")
        ]

        if not mtd.doesExist("instrument"):
            LoadEmptyInstrument(
                InstrumentName=inst if idf is None else None,
                Filename=idf if idf is not None else None,
                OutputWorkspace="instrument",
            )

            for key in logs.keys():
                AddSampleLog(
                    Workspace="instrument",
                    LogName=key,
                    LogText=str(logs[key]),
                    LogType="Number Series",
                    NumberType="Double",
                )

            if len(logs.keys()) > 0:
                LoadInstrument(
                    Workspace="instrument",
                    RewriteSpectraMap=False,
                    InstrumentName=inst if idf is None else None,
                    Filename=idf if idf is not None else None,
                )

            if cal != "" and os.path.exists(cal):
                if os.path.splitext(cal)[1] == ".xml":
                    LoadParameterFile(Workspace="instrument", Filename=cal)
                else:
                    LoadIsawDetCal(InputWorkspace="instrument", Filename=cal)

            ExtractMonitors(
                InputWorkspace="instrument",
                MonitorWorkspace="monitors",
                DetectorWorkspace="instrument",
            )

            c, r = [int(val) for val in beamline["Grouping"].split("x")]
            cols, rows = beamline["BankPixels"]
            mask_cols, mask_rows = beamline["MaskEdges"]

            print(idf)

            if idf is not None:
                cols //= c
                rows //= r
                mask_cols //= c
                mask_rows //= r
                c, r = 1, 1

            print(mask_rows, rows - mask_rows, rows)

            MaskBTP(
                Workspace="instrument",
                Instrument=inst,
                Tube="0-{},{}-{}".format(mask_cols, cols - mask_cols, cols),
            )

            MaskBTP(
                Workspace="instrument",
                Instrument=inst,
                Pixel="0-{},{}-{}".format(mask_rows, rows - mask_rows, rows),
            )

            banks = beamlines[instrument]["MaskBanks"]

            for bank in banks:
                MaskBTP(
                    Workspace="instrument",
                    Instrument=inst,
                    Bank=bank,
                )

            if mask != "" and os.path.exists(mask):
                if not mtd.doesExist("mask"):
                    LoadMask(
                        Instrument=inst, InputFile=mask, OutputWorkspace="mask"
                    )
                    PreprocessDetectorsToMD(
                        InputWorkspace="mask", OutputWorkspace="detectors"
                    )
                    c, r = [
                        int(val) for val in beamline["Grouping"].split("x")
                    ]
                    cols, rows = beamline["BankPixels"]
                    if c > 1 or r > 1:
                        detector_list = self.grouping_list(
                            "detectors", cols, rows, c, r
                        )

                        GroupDetectors(
                            InputWorkspace="mask",
                            GroupingPattern=detector_list,
                            OutputWorkspace="mask",
                        )
                        ys = mtd["mask"].extractY().copy()

                        CloneWorkspace(
                            InputWorkspace="instrument", OutputWorkspace="mask"
                        )

                        for i, y in enumerate(ys):
                            mtd["mask"].setY(i, y)

                        if idf is not None:
                            cols //= c
                            rows //= r
                            mask_cols //= c
                            mask_rows //= r
                            c, r = 1, 1

                        MaskDetectorsIf(
                            InputWorkspace="mask",
                            Operator="Greater",
                            Value=0,
                            OutputWorkspace="mask",
                        )

                MaskDetectors(Workspace="instrument", MaskedWorkspace="mask")

            PreprocessDetectorsToMD(
                InputWorkspace="instrument", OutputWorkspace="detectors"
            )
            mask = np.array(mtd["detectors"].column(7)) != 0
            det_ID = np.array(mtd["detectors"].column(4))
            bad_ID = (
                np.insert(det_ID[mask], -1, -1)
                if np.sum(mask) != 0
                else np.array([-1])
            )

            detector_list = self.grouping_list("detectors", cols, rows, c, r)

            GroupDetectors(
                InputWorkspace="instrument",
                GroupingPattern=detector_list,
                OutputWorkspace="instrument",
            )

            CreatePeaksWorkspace(
                InstrumentWorkspace="instrument",
                NumberOfPeaks=0,
                OutputType="LeanElasticPeak",
                OutputWorkspace="peak",
            )

            CreatePeaksWorkspace(
                InstrumentWorkspace="instrument",
                NumberOfPeaks=0,
                OutputType="Peak",
                OutputWorkspace="peaks",
            )

            CreatePeaksWorkspace(
                InstrumentWorkspace="instrument",
                NumberOfPeaks=0,
                OutputType="Peak",
                OutputWorkspace="combined",
            )

            PreprocessDetectorsToMD(
                InputWorkspace="instrument", OutputWorkspace="detectors"
            )

            mask = np.array(mtd["detectors"].column(7)) == 0

            L2 = np.array(mtd["detectors"].column(1))
            tt = np.array(mtd["detectors"].column(2))
            az = np.array(mtd["detectors"].column(3))

            x = L2 * np.sin(tt) * np.cos(az)
            y = L2 * np.sin(tt) * np.sin(az)
            z = L2 * np.cos(tt)

            self.det_ID = bad_ID.copy()
            self.nu = np.rad2deg(np.arcsin(y / L2))[mask]
            self.gamma = np.rad2deg(np.arctan2(x, z))[mask]
            self.instrument_background = None

            det_ID = np.array(mtd["detectors"].column(4))
            mask = np.array(mtd["detectors"].column(7))

            x = x.reshape(-1, cols // c, rows // r)
            y = y.reshape(-1, cols // c, rows // r)
            z = z.reshape(-1, cols // c, rows // r)
            det_ID = det_ID.reshape(-1, cols // c, rows // r)
            mask = mask.reshape(-1, cols // c, rows // r).sum(axis=(1, 2))
            mask = mask != cols * rows // (c * r)

            self.xc = x[:, [0, 0, -1, -1], [0, -1, -1, 0]][mask]
            self.yc = y[:, [0, 0, -1, -1], [0, -1, -1, 0]][mask]
            self.zc = z[:, [0, 0, -1, -1], [0, -1, -1, 0]][mask]
            self.detc = det_ID[:, 0, 0][mask]

    def grouping_list(self, detectors, cols, rows, c, r):
        """
        Build a Mantid GroupingPattern string from a detectors table.

        Parameters
        ----------
        detectors : str
            Name of the preprocessed detectors table workspace.
        cols, rows : int
            Number of columns and rows per bank (before grouping).
        c, r : int
            Grouping factors along columns and rows.

        Returns
        -------
        str
            Comma-separated grouping pattern suitable for GroupDetectors.
        """

        det_map = np.asarray(mtd[detectors].column(5)).reshape(-1, cols, rows)

        nb, nc, nr = det_map.shape

        gc = (np.arange(nc) // c).astype(np.int32)
        gr = (np.arange(nr) // r).astype(np.int32)

        ngc = (nc + c - 1) // c
        ngr = (nr + r - 1) // r

        group_id = (
            (np.arange(nb, dtype=np.int32)[:, None, None] * (ngc * ngr))
            + (gc[None, :, None] * ngr)
            + gr[None, None, :]
        ).ravel()

        det_ids = det_map.ravel()

        order = np.argsort(group_id, kind="stable")
        g_sorted = group_id[order]
        d_sorted = det_ids[order]

        starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
        ends = np.r_[starts[1:], g_sorted.size]

        parts = [
            "+".join(map(str, d_sorted[s:e])) for s, e in zip(starts, ends)
        ]

        return ",".join(parts)

    def extract_instrument_view(self):
        """
        Build a mesh of detector-bank quadrilaterals for 3D rendering.

        Uses the bank-corner coordinates computed by
        :meth:`initialize_instrument` (``self.xc``, ``self.yc``, ``self.zc``).

        Returns
        -------
        inst_dict : dict
            Dictionary with keys ``"points"`` (Nx3 array of corner
            coordinates), ``"faces"`` (PyVista-style face connectivity
            array), and ``"radius"`` (max extent along each axis).
        """

        n = self.xc.shape[0]

        points = np.column_stack(
            [self.xc.reshape(-1), self.yc.reshape(-1), self.zc.reshape(-1)]
        )

        faces = np.empty(n * 5, dtype=np.int64)
        faces[0::5] = 4
        faces[1::5] = np.arange(0, 4 * n, 4) + 0
        faces[2::5] = np.arange(0, 4 * n, 4) + 1
        faces[3::5] = np.arange(0, 4 * n, 4) + 2
        faces[4::5] = np.arange(0, 4 * n, 4) + 3

        inst_dict = {}
        inst_dict["points"] = points
        inst_dict["faces"] = faces
        inst_dict["radius"] = np.max(points, axis=0)

        return inst_dict

    def get_instrument_background(self):
        """
        Compute (or return cached) 2D detector-coverage occupancy image.

        Bins the instrument's ``gamma``/``nu`` detector angles into a 2D
        histogram to show which regions of angular space are covered by
        active (unmasked) detectors. Result is cached on
        ``self.instrument_background``.

        Returns
        -------
        instrument_background : dict or None
            Dictionary with keys ``"img"`` (occupancy image, 1 where
            detectors exist), ``"xedges"``, ``"yedges"`` (bin edges in
            gamma/nu), or None if no detector angle data is available.
        """

        if self.instrument_background is not None:
            return self.instrument_background

        gamma = getattr(self, "gamma", np.array([]))
        nu = getattr(self, "nu", np.array([]))

        if len(gamma) == 0 or len(nu) == 0:
            self.instrument_background = None
            return None

        dx = self.pixel_size[0] * self.grouping_c
        dy = self.pixel_size[1] * self.grouping_r

        if not np.isfinite(dx) or dx <= 0:
            x_span = np.ptp(gamma)
            dx = x_span / 200 if x_span > 0 else 1.0

        if not np.isfinite(dy) or dy <= 0:
            y_span = np.ptp(nu)
            dy = y_span / 200 if y_span > 0 else 1.0

        xedges = np.arange(gamma.min(), gamma.max() + dx, dx)
        yedges = np.arange(nu.min(), nu.max() + dy, dy)

        if len(xedges) < 2:
            xedges = np.array([gamma.min() - 0.5, gamma.max() + 0.5])
        if len(yedges) < 2:
            yedges = np.array([nu.min() - 0.5, nu.max() + 0.5])

        occupancy, xedges, yedges = np.histogram2d(
            gamma, nu, bins=[xedges, yedges]
        )

        self.instrument_background = {
            "img": (occupancy > 0).astype(float),
            "xedges": xedges,
            "yedges": yedges,
        }

        return self.instrument_background

    def clear_combined(self):
        """
        Reset the ``combined`` peaks workspace to be empty.
        """

        CreatePeaksWorkspace(
            InstrumentWorkspace="instrument",
            NumberOfPeaks=0,
            OutputType="Peak",
            OutputWorkspace="combined",
        )

    def get_calibration_file_path(self, instrument):
        """
        Get the shared calibration directory for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        filepath : str
            File path to the instrument's shared calibration directory.
        """

        inst = beamlines[instrument]

        return os.path.join(
            "/",
            inst["Facility"],
            inst["InstrumentName"],
            "shared",
            "calibration",
        )

    def get_vanadium_file_path(self, instrument):
        """
        Get the shared vanadium directory for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        filepath : str
            File path to the instrument's shared Vanadium directory.
        """

        inst = beamlines[instrument]

        return os.path.join(
            "/", inst["Facility"], inst["InstrumentName"], "shared", "Vanadium"
        )

    def remove_instrument(self):
        """
        Delete the ``instrument``, ``combined``, ``filtered``, and
        ``footprint`` workspaces if they exist, and clear the cached
        instrument background image.
        """

        if mtd.doesExist("instrument"):
            DeleteWorkspace(Workspace="instrument")

        self.instrument_background = None

        if mtd.doesExist("cobmined"):
            DeleteWorkspace(Workspace="cobmined")

        if mtd.doesExist("filtered"):
            DeleteWorkspace(Workspace="filtered")

        if mtd.doesExist("footprint"):
            DeleteWorkspace(Workspace="footprint")

    def get_crystal_system_point_groups(self, crystal_system):
        """
        Get the point groups belonging to a crystal system.

        Parameters
        ----------
        crystal_system : str
            Crystal system name (e.g. "Cubic", "Tetragonal").

        Returns
        -------
        point_groups : list of str
            Point group symbols for the given crystal system.
        """

        return crystal_system_point_groups[crystal_system]

    def get_point_group_centering(self, point_group):
        """
        Get the allowed lattice centerings for a point group.

        Parameters
        ----------
        point_group : str
            Point group symbol.

        Returns
        -------
        centerings : list of str
            Allowed lattice centering symbols for the given point group.
        """

        return point_group_centering[point_group]

    def get_symmetry(self, point_group, centering):
        """
        Coerce point group and centering to plain strings.

        Parameters
        ----------
        point_group : str
            Point group symbol.
        centering : str
            Lattice centering symbol.

        Returns
        -------
        pg : str
            Point group symbol as a string.
        lc : str
            Lattice centering symbol as a string.
        """

        return str(point_group), str(centering)

    def calculate_hkl_limits(self, d_min):
        """
        Calculate maximum Miller indices reachable at a minimum d-spacing.

        Parameters
        ----------
        d_min : float
            Minimum d-spacing (in Angstrom).

        Returns
        -------
        h_max, k_max, l_max : int
            Maximum h, k, l indices consistent with ``d_min`` given the
            oriented lattice on the ``coverage`` workspace.
        """

        ol = mtd["coverage"].sample().getOrientedLattice()

        astar = ol.astar()
        bstar = ol.bstar()
        cstar = ol.cstar()

        h_max = int(round(1 / (d_min * astar)))
        k_max = int(round(1 / (d_min * bstar)))
        l_max = int(round(1 / (d_min * cstar)))

        return h_max, k_max, l_max

    def create_plan(self, table):
        """
        Build the ``plan`` table workspace from an experiment plan tuple.

        Parameters
        ----------
        table : tuple
            8-tuple ``(pv, names, titles, settings, comments, counts,
            values, use)`` where ``pv`` is the name of the process-variable
            (title) column, ``names`` are the motor/axis column names,
            ``titles`` are the row titles/labels, ``settings`` are the
            per-row lists of motor angles, ``comments`` are per-row comment
            strings, ``counts`` are per-row "wait for" counting-condition
            values, ``values`` are per-row target count values, and ``use``
            are per-row booleans indicating whether the row is active.
        """

        pv, names, titles, settings, comments, counts, values, use = table

        CreateEmptyTableWorkspace(OutputWorkspace="plan")

        mtd["plan"].addColumn("str", pv)

        for name in names:
            mtd["plan"].addColumn("float", name)

        mtd["plan"].addColumn("str", "Comment")
        mtd["plan"].addColumn("str", "Wait For")
        mtd["plan"].addColumn("float", "Value")
        mtd["plan"].addColumn("bool", "Use")

        for title, setting, comment, count, value, active in zip(
            titles, settings, comments, counts, values, use
        ):
            row = {}
            row[pv] = title
            for angle, name in zip(setting, names):
                row[name] = np.round(angle, 2)
            row["Comment"] = comment
            row["Wait For"] = count
            row["Value"] = value
            row["Use"] = active
            mtd["plan"].addRow(row)

    def create_sample(self, instrument, mode, UB, wavelength, d_min):
        """
        Create the ``sample`` workspace and record experiment settings.

        Stores the UB matrix and the instrument, mode, wavelength range,
        and minimum d-spacing as sample logs for later retrieval (e.g. by
        :meth:`load_experiment`).

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        mode : str
            Goniometer/experiment mode name.
        UB : 3x3 array-like
            UB matrix to set on the sample workspace.
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom).
        d_min : float
            Minimum d-spacing (Angstrom).
        """

        CreateSampleWorkspace(OutputWorkspace="sample")

        SetUB(Workspace="sample", UB=UB)

        AddSampleLog(
            Workspace="sample",
            LogName="instrument",
            LogText=instrument,
            LogType="String",
        )

        AddSampleLog(
            Workspace="sample",
            LogName="mode",
            LogText=mode,
            LogType="String",
        )

        AddSampleLog(
            Workspace="sample",
            LogName="lamda_min",
            LogText=str(wavelength[0]),
            LogType="Number",
            NumberType="Double",
        )

        AddSampleLog(
            Workspace="sample",
            LogName="lamda_max",
            LogText=str(wavelength[1]),
            LogType="Number",
            NumberType="Double",
        )

        AddSampleLog(
            Workspace="sample",
            LogName="d_min",
            LogText=str(d_min),
            LogType="Number",
            NumberType="Double",
        )

    def update_sample(self, crytsal_system, point_group, lattice_centering):
        """
        Record crystal-symmetry information as sample logs.

        Parameters
        ----------
        crytsal_system : str
            Crystal system name.
        point_group : str
            Point group symbol.
        lattice_centering : str
            Lattice centering symbol.
        """

        if mtd.doesExist("sample"):
            AddSampleLog(
                Workspace="sample",
                LogName="crystal_system",
                LogText=crytsal_system,
                LogType="String",
            )

            AddSampleLog(
                Workspace="sample",
                LogName="point_group",
                LogText=point_group,
                LogType="String",
            )

            AddSampleLog(
                Workspace="sample",
                LogName="lattice_centering",
                LogText=lattice_centering,
                LogType="String",
            )

    def update_goniometer_motors(self, limits, motors, cal, mask):
        """
        Record goniometer motor limits and auxiliary motor/file settings
        as sample logs so they can be restored by :meth:`load_experiment`.

        Parameters
        ----------
        limits : array-like
            Per-axis (min, max) angle limits, flattened and stored as the
            ``"limits"`` sample log.
        motors : dict
            Mapping of auxiliary motor names to their values.
        cal : str
            Calibration file path, stored as the ``"cal"`` sample log.
        mask : str
            Mask file path, stored as the ``"mask"`` sample log.
        """

        if mtd.doesExist("sample"):
            mtd["sample"].run()["limits"] = np.array(limits).flatten().tolist()

            values = []
            for key in motors.keys():
                values.append(motors[key])
            if len(values) > 0:
                mtd["sample"].run()["motors"] = values

            mtd["sample"].run()["cal"] = cal
            mtd["sample"].run()["mask"] = mask

    def load_UB(self, filename):
        """
        Load a UB matrix from an ISAW UB file onto the ``coverage``
        workspace and update the model's cached UB matrix.

        Parameters
        ----------
        filename : str
            Path to the ISAW UB matrix file.
        """

        LoadIsawUB(InputWorkspace="coverage", Filename=filename)

        self.copy_UB()

    def get_UB(self):
        """
        Get the UB matrix currently set on the ``coverage`` workspace.

        Returns
        -------
        UB : 3x3 ndarray or None
            UB matrix, or None if no oriented lattice is set.
        """

        if self.has_UB():
            return mtd["coverage"].sample().getOrientedLattice().getUB().copy()

    def copy_UB(self):
        """
        Copy the UB matrix from the ``coverage`` workspace into the
        model's cached UB matrix (:attr:`NeuXtalVizModel.UB`).
        """

        UB = self.get_UB()
        if UB is not None:
            self.set_UB(UB)

    def has_UB(self):
        """
        Check whether the ``coverage`` workspace has an oriented lattice.

        Returns
        -------
        has_ub : bool
            True if the ``coverage`` workspace has a UB matrix set.
        """

        if HasUB(Workspace="coverage"):
            return True
        else:
            return False

    def get_instrument_name(self, instrument):
        """
        Get the short Mantid instrument name for an instrument identifier.

        Parameters
        ----------
        instrument : str
            Instrument identifier (key into ``beamlines``).

        Returns
        -------
        name : str
            Mantid instrument name.
        """

        return beamlines[instrument]["Name"]

    def get_modes(self, instrument):
        """
        Get the available goniometer modes for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        modes : list of str
            Names of the goniometer modes defined for the instrument.
        """

        return list(beamlines[instrument]["Goniometer"].keys())

    def get_counting_options(self, instrument):
        """
        Get the counting condition options for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        counting : object
            The instrument's "Counting" configuration entry (e.g. a list of
            available counting/"wait for" conditions).
        """

        return beamlines[instrument]["Counting"]

    def get_scan_log(self, instrument):
        """
        Get the process-variable/title log name for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        title : str
            Name of the instrument's scan title/process-variable log.
        """

        return beamlines[instrument]["Title"]

    def get_axes_polarities(self, instrument, mode):
        """
        Get the goniometer rotation axes and their polarities for a mode.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        mode : str
            Goniometer mode name.

        Returns
        -------
        axes : list
            Per-axis (x, y, z) rotation-axis direction components.
        polarities : list
            Per-axis rotation sense/polarity (+1 or -1).
        """

        goniometers = beamlines[instrument]["Goniometer"][mode]

        axes = [goniometers[name][:-3] for name in goniometers.keys()]

        polarities = [goniometers[name][3] for name in goniometers.keys()]

        return axes, polarities

    def get_goniometer_axes(self, instrument, mode):
        """
        Get Mantid ``SetGoniometer`` axis-string templates for a mode.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        mode : str
            Goniometer mode name.

        Returns
        -------
        axes : list of str
            Per-axis template strings of the form
            ``"{},x,y,z,polarity,type"`` (with a ``{}`` placeholder for the
            angle value) suitable for formatting and passing to
            ``SetGoniometer``.
        """

        goniometers = beamlines[instrument]["Goniometer"][mode]

        axes = [
            "{}," + ",".join(np.array(goniometers[name][:-2]).astype(str))
            for name in goniometers.keys()
        ]

        return axes

    def get_goniometers(self, instrument, mode):
        """
        Get goniometer axis names with their angle limits for a mode.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        mode : str
            Goniometer mode name.

        Returns
        -------
        goniometers : list of tuple
            One ``(name, min_angle, max_angle)`` tuple per goniometer axis.
        """

        goniometers = beamlines[instrument]["Goniometer"][mode]

        return [(name, *goniometers[name][-2:]) for name in goniometers.keys()]

    def get_motors(self, instrument):
        """
        Get auxiliary motor names and default values for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        motors : list of tuple
            One ``(name, value)`` tuple per auxiliary motor, or an empty
            list if the instrument defines no motors.
        """

        motors = beamlines[instrument].get("Motor")

        if motors is not None:
            return [(name, motors[name]) for name in motors.keys()]
        else:
            return []

    def get_wavelength(self, instrument):
        """
        Get the default wavelength (range) for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        wavelength : float or list of float
            Single wavelength, or [min, max] wavelength range (Angstrom).
        """

        return beamlines[instrument]["Wavelength"]

    def save_plan(self, filename):
        """
        Write the active (used) rows of the ``plan`` table to a CSV file.

        Parameters
        ----------
        filename : str
            Output CSV file path.
        """

        plan_dict = mtd["plan"].toDict().copy()
        use_angle = plan_dict["Use"]

        for key in plan_dict.keys():
            items = plan_dict[key]
            items = [item for item, use in zip(items, use_angle) if use]
            plan_dict[key] = items

        plan_dict.pop("Use")

        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=plan_dict.keys())
            writer.writeheader()
            for row in zip(*plan_dict.values()):
                writer.writerow(dict(zip(plan_dict.keys(), row)))

    def save_experiment(self, filename):
        """
        Save the ``plan`` (and ``sample``, if present) workspaces to Nexus.

        Parameters
        ----------
        filename : str
            Output Nexus file path.
        """

        if mtd.doesExist("plan"):
            SaveNexus(InputWorkspace="plan", Filename=filename)
            if mtd.doesExist("sample"):
                SaveNexus(
                    InputWorkspace="sample", Filename=filename, Append=True
                )

    def load_experiment(self, filename):
        """
        Load a previously saved experiment (plan + sample) from Nexus.

        Restores the UB matrix and rebuilds the plan table contents and
        experiment configuration from the sample logs written by
        :meth:`create_sample`, :meth:`update_sample`, and
        :meth:`update_goniometer_motors`.

        Parameters
        ----------
        filename : str
            Path to the Nexus file to load.

        Returns
        -------
        plan : tuple
            ``(titles, settings, comments, counts, values, use)`` -- the
            plan table contents, in the same layout consumed by
            :meth:`create_plan`.
        config : tuple
            ``(instrument, mode, wl, d_min, lims, vals, cal, mask)``
            -- the instrument/goniometer configuration, where ``wl`` is
            either a single wavelength or a [min, max] pair, ``lims`` are
            the goniometer axis limits, and ``vals`` are the auxiliary
            motor values.
        symm : tuple
            ``(cs, pg, lc)`` -- crystal system, point group, and lattice
            centering.
        """

        LoadNexus(Filename=filename, OutputWorkspace="experiment")

        plan, sample = mtd["experiment"].getNames()

        UB = mtd[sample].sample().getOrientedLattice().getUB().copy()
        SetUB(Workspace="coverage", UB=UB)

        self.set_UB(UB)

        instrument = mtd[sample].run().getProperty("instrument").value
        mode = mtd[sample].run().getProperty("mode").value
        wl_min = mtd[sample].run().getProperty("lamda_min").value
        wl_max = mtd[sample].run().getProperty("lamda_max").value
        d_min = mtd[sample].run().getProperty("d_min").value
        cs = mtd[sample].run().getProperty("crystal_system").value
        pg = mtd[sample].run().getProperty("point_group").value
        lc = mtd[sample].run().getProperty("lattice_centering").value
        lims = mtd[sample].run().getProperty("limits").value
        mask = mtd[sample].run().getProperty("mask").value
        cal = mtd[sample].run().getProperty("cal").value
        lims = np.array(lims).reshape(-1, 2).tolist()
        vals = []
        if mtd[sample].run().hasProperty("motors"):
            vals = mtd[sample].run().getProperty("motors").value

        if np.isclose(wl_min, wl_max):
            wl = wl_min
        else:
            wl = [wl_min, wl_max]

        cols = mtd[plan].columnCount() - 5
        rows = mtd[plan].rowCount()

        titles = mtd[plan].column(0)
        comments = mtd[plan].column(cols + 1)
        counts = mtd[plan].column(cols + 2)
        values = mtd[plan].column(cols + 3)
        use = mtd[plan].column(cols + 4)

        settings = []
        for row in range(rows):
            angles = []
            for col in range(cols):
                angle = mtd[plan].cell(row, col + 1)
                angles.append(angle)
            settings.append(angles)

        plan = (titles, settings, comments, counts, values, use)
        config = (instrument, mode, wl, d_min, lims, vals, cal, mask)
        symm = (cs, pg, lc)

        return plan, config, symm

    def generate_axes(self, axes, polarities):
        """
        Build and cache ``SetGoniometer`` axis-string templates.

        Stores the result on ``self.axes`` for use by
        :meth:`add_orientation`, :meth:`calculate_rotations`, and the
        ``CrystalPlan`` genetic algorithm.

        Parameters
        ----------
        axes : list
            Per-axis (x, y, z) rotation-axis direction components, as
            returned by :meth:`get_axes_polarities`.
        polarities : list
            Per-axis rotation sense/polarity, as returned by
            :meth:`get_axes_polarities`.
        """

        self.axes = [None] * 6

        for i, (axis, polarity) in enumerate(zip(axes, polarities)):
            self.axes[i] = "{},"
            self.axes[i] += ",".join(np.array([*axis, polarity]).astype(str))

    def get_setting(self, free_angles, limits):
        """
        Combine free (variable) angles with fixed-limit angles into a
        full per-axis goniometer setting.

        Parameters
        ----------
        free_angles : sequence of float
            Values for the axes whose limits are not fixed (min != max),
            in axis order.
        limits : sequence of (float, float)
            Per-axis (min, max) angle limits.

        Returns
        -------
        setting : list of float
            Full per-axis angle setting: the fixed limit value for axes
            with ``min == max``, otherwise the next value from
            ``free_angles``.
        """

        setting = []
        col = 0
        for limit in limits:
            if np.isclose(limit[0], limit[1]):
                setting.append(limit[0])
            else:
                setting.append(free_angles[col])
                col += 1
        return setting

    def _calculate_matrices(self, axes, polarities, limits, step):
        """
        Enumerate goniometer rotation matrices over a grid of axis angles.

        Builds the Cartesian product of per-axis angle values (spaced by
        ``step``, doubled per additional free axis for finer sampling) and
        composes the corresponding rotation matrix for each combination.

        Parameters
        ----------
        axes : list
            Per-axis (x, y, z) rotation-axis direction components.
        polarities : list
            Per-axis rotation sense/polarity.
        limits : sequence of (float, float)
            Per-axis (min, max) angle limits, in degrees.
        step : float
            Base angular step size, in degrees.

        Returns
        -------
        Rs : list of ndarray, shape (3, 3)
            Composed rotation matrix for each sampled goniometer setting.
        angles : ndarray, shape (n_settings, n_axes)
            Per-axis angle values (in degrees, before polarity/sign
            adjustment) corresponding to each rotation matrix in ``Rs``.
        """

        self.generate_axes(axes, polarities)

        free = 0
        for limit in limits:
            free += 1 - np.isclose(limit[0], limit[1])
        step *= 2 ** (free - 1)

        angular_coverage = []
        for limit in limits:
            angular_coverage.append(np.arange(limit[0], limit[1] + step, step))

        axes = np.array(axes)
        polarities = np.array(polarities)

        angle_settings = np.meshgrid(*angular_coverage, indexing="ij")
        angle_settings = np.reshape(angle_settings, (len(polarities), -1)).T

        angles = angle_settings.copy()

        angle_settings = angle_settings * polarities
        angle_settings = np.deg2rad(angle_settings)

        rotation_vectors = angle_settings[..., None] * axes
        rotation_vectors = rotation_vectors.reshape(-1, 3)

        all_rotations = Rotation.from_rotvec(rotation_vectors).as_matrix()
        all_rotations = all_rotations.reshape(*angle_settings.shape, 3, 3)

        Rs = []
        for i in range(all_rotations.shape[0]):
            R = np.eye(3)
            for j in range(all_rotations.shape[1]):
                R = R @ all_rotations[i, j, :, :]
            Rs.append(R)

        return Rs, angles

    def individual_peak(
        self, hkl, wavelength, axes, polarities, limits, equiv, pg, step=1
    ):
        """
        Find goniometer settings that bring a peak (or its symmetry
        equivalents) into the detector range.

        Aggregates results from :meth:`calculate_individual_peak` over the
        requested HKL and (optionally) all of its point-group equivalents,
        and caches the combined results on ``self.angles*`` attributes for
        later lookup via :meth:`get_angles`.

        Parameters
        ----------
        hkl : array-like, shape (3,)
            Miller index of the target reflection.
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom).
        axes : list
            Per-axis (x, y, z) rotation-axis direction components.
        polarities : list
            Per-axis rotation sense/polarity.
        limits : sequence of (float, float)
            Per-axis (min, max) angle limits, in degrees.
        equiv : bool
            If True, also search settings for all symmetry-equivalent HKLs
            of ``hkl`` under point group ``pg``.
        pg : str
            Point group symbol used to generate symmetry equivalents.
        step : float, optional
            Base angular step size, in degrees. Default is 1.

        Returns
        -------
        gamma, nu, lamda, d : ndarray
            Detector gamma/nu angles (degrees), wavelength (Angstrom), and
            d-spacing (Angstrom) for each goniometer setting found that
            places the peak on an active detector.
        """

        pg = PointGroupFactory.createPointGroup(pg)

        hkls = pg.getEquivalents(hkl) if equiv else [hkl]

        indices, angles = [], []
        angles_gamma, angles_nu = [], []
        angles_lamda, angles_d = [], []

        for hkl in hkls:
            settings, values = self.calculate_individual_peak(
                hkl, wavelength, axes, polarities, limits, step
            )

            gamma, nu, lamda, d = values

            indices.append(
                [hkl] * len(lamda) if len(lamda) > 0 else np.zeros((0, 3))
            )
            angles.append(settings)
            angles_gamma.append(gamma)
            angles_nu.append(nu)
            angles_lamda.append(lamda)
            angles_d.append([d] * len(lamda))

        angles = np.vstack(angles)

        indices = np.vstack(indices)
        gamma = np.concatenate(angles_gamma)
        nu = np.concatenate(angles_nu)
        lamda = np.concatenate(angles_lamda)
        d = np.concatenate(angles_d)

        self.angles = angles

        self.angles_indices = indices
        self.angles_gamma = gamma
        self.angles_nu = nu
        self.angles_lamda = lamda
        self.angles_d = d

        self.angles_indices_alt = None
        self.angles_gamma_alt = None
        self.angles_nu_alt = None
        self.angles_lamda_alt = None
        self.angles_d_alt = None

        return gamma, nu, lamda, d

    def calculate_individual_peak(
        self, hkl, wavelength, axes, polarities, limits, step=1
    ):
        """
        Scan goniometer settings for a single HKL and keep valid ones.

        Adds a peak with the given HKL to the ``peak`` workspace, rotates
        its Q vector through every goniometer setting produced by
        :meth:`_calculate_matrices`, and keeps only settings where the
        resulting wavelength is in range and the scattered beam does not
        fall on an already-masked/dead detector.

        Parameters
        ----------
        hkl : array-like, shape (3,)
            Miller index of the target reflection.
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom). If min and max are
            equal, a +/-2.5% band is used instead.
        axes : list
            Per-axis (x, y, z) rotation-axis direction components.
        polarities : list
            Per-axis rotation sense/polarity.
        limits : sequence of (float, float)
            Per-axis (min, max) angle limits, in degrees.
        step : float, optional
            Base angular step size, in degrees. Default is 1.

        Returns
        -------
        settings : ndarray, shape (n_valid, n_axes)
            Per-axis angle values for each valid goniometer setting.
        values : tuple of ndarray
            ``(gamma, nu, lamda, d)`` -- detector gamma/nu angles
            (degrees), wavelength (Angstrom), and d-spacing (Angstrom;
            scalar, same for all settings) for each valid setting.
        """

        if np.isclose(wavelength[0], wavelength[1]):
            wavelength = [0.975 * wavelength[0], 1.025 * wavelength[1]]

        FilterPeaks(
            InputWorkspace="peak",
            OutputWorkspace="peak",
            FilterVariable="RunNumber",
            FilterValue=-1,
            Operator="=",
        )

        UB = mtd["coverage"].sample().getOrientedLattice().getUB().copy()

        SetUB(Workspace="peak", UB=UB)

        AddPeakHKL(Workspace="peak", HKL=hkl)

        Q_sample = mtd["peak"].getPeak(0).getQSampleFrame()

        Q = np.sqrt(np.dot(Q_sample, Q_sample))

        Rs, angles = self._calculate_matrices(axes, polarities, limits, step)

        mtd["peak"].run().getGoniometer().setR(np.eye(3))
        mtd["peaks"].run().getGoniometer().setR(np.eye(3))

        Q_lab = np.einsum("kij,j->ki", Rs, Q_sample)

        lamda = -4 * np.pi * Q_lab[:, 2] / Q**2
        d = 2 * np.pi / Q

        mask = (lamda > wavelength[0]) & (lamda < wavelength[1])

        k = 2 * np.pi / lamda

        ki = k[:, np.newaxis] * np.array([0, 0, 1])
        kf = Q_lab + ki

        gamma = np.rad2deg(np.arctan2(kf[:, 0], kf[:, 2]))[mask]
        nu = np.rad2deg(np.arcsin(kf[:, 1] / k))[mask]
        lamda = lamda[mask]

        settings = angles[mask]

        if len(lamda) > 0:
            k = 2 * np.pi / lamda
            Qx = k * np.cos(np.deg2rad(nu)) * np.sin(np.deg2rad(gamma))
            Qy = k * np.sin(np.deg2rad(nu))
            Qz = k * (np.cos(np.deg2rad(nu)) * np.cos(np.deg2rad(gamma)) - 1)

            mask = []
            for i in range(len(k)):
                peak = mtd["peaks"].createPeak(V3D(Qx[i], Qy[i], Qz[i]))
                mask.append(peak.getDetectorID() not in self.det_ID)

            mask = np.array(mask)

            gamma = gamma[mask]
            nu = nu[mask]
            lamda = lamda[mask]

            settings = settings[mask]

        return settings, (gamma, nu, lamda, d)

    def simultaneous_peaks(
        self,
        hkl_1,
        hkl_2,
        wavelength,
        axes,
        polarities,
        limits,
        equiv,
        pg,
        step=1,
    ):
        """
        Find goniometer settings placing two reflections on detectors
        simultaneously.

        Considers every distinct pair drawn from the point-group
        equivalents of ``hkl_1`` and ``hkl_2`` (or just the two HKLs
        themselves if ``equiv`` is False), and aggregates the valid
        settings found by :meth:`simultaneous_peaks_hkl` for each pair.
        Caches the combined results on ``self.angles*`` attributes for
        later lookup via :meth:`get_angles`.

        Parameters
        ----------
        hkl_1, hkl_2 : array-like, shape (3,)
            Miller indices of the two target reflections.
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom).
        axes : list
            Per-axis (x, y, z) rotation-axis direction components.
        polarities : list
            Per-axis rotation sense/polarity.
        limits : sequence of (float, float)
            Per-axis (min, max) angle limits, in degrees.
        equiv : bool
            If True, search over all symmetry-equivalent HKL pairs under
            point group ``pg``.
        pg : str
            Point group symbol used to generate symmetry equivalents.
        step : float, optional
            Base angular step size, in degrees. Default is 1.

        Returns
        -------
        values0 : tuple of ndarray
            ``(gamma, nu, lamda, d)`` for the ``hkl_1``-side reflection at
            each valid goniometer setting.
        values1 : tuple of ndarray
            ``(gamma, nu, lamda, d)`` for the ``hkl_2``-side reflection at
            each valid goniometer setting.
        """

        pg = PointGroupFactory.createPointGroup(pg)

        hkls_1 = pg.getEquivalents(hkl_1) if equiv else [hkl_1]
        hkls_2 = pg.getEquivalents(hkl_2) if equiv else [hkl_2]

        hkls_1 = [tuple(hkl) for hkl in hkls_1]
        hkls_2 = [tuple(hkl) for hkl in hkls_2]

        unique_pairs = set()
        for a, b in itertools.product(hkls_1, hkls_2):
            if a != b:
                unique_pairs.add((a, b))
        pairs = list(unique_pairs)

        angles = []
        indices, indices_alt = [], []

        angles_gamma, angles_nu = [], []
        angles_lamda, angles_d = [], []
        angles_gamma_alt, angles_nu_alt = [], []
        angles_lamda_alt, angles_d_alt = [], []

        for hkl_1, hkl_2 in pairs:
            settings, values0, values1 = self.simultaneous_peaks_hkl(
                hkl_1, hkl_2, wavelength, axes, polarities, limits, step
            )

            gamma0, nu0, lamda0, d0 = values0
            gamma1, nu1, lamda1, d1 = values1

            if len(lamda0) > 0 and len(lamda1) > 0:
                angles.append(settings)

                indices.append(
                    [hkl_1] * len(lamda0)
                    if len(lamda0) > 0
                    else np.zeros((0, 3))
                )
                angles_gamma.append(gamma0)
                angles_nu.append(nu0)
                angles_lamda.append(lamda0)
                angles_d.append([d0] * len(lamda0))

                indices_alt.append(
                    [hkl_2] * len(lamda1)
                    if len(lamda1) > 0
                    else np.zeros((0, 3))
                )
                angles_gamma_alt.append(gamma1)
                angles_nu_alt.append(nu1)
                angles_lamda_alt.append(lamda1)
                angles_d_alt.append([d1] * len(lamda1))

        angles = np.vstack(angles)

        indices = np.vstack(indices)
        gamma = np.concatenate(angles_gamma)
        nu = np.concatenate(angles_nu)
        lamda = np.concatenate(angles_lamda)
        d = np.concatenate(angles_d)

        indices_alt = np.vstack(indices_alt)
        gamma_alt = np.concatenate(angles_gamma_alt)
        nu_alt = np.concatenate(angles_nu_alt)
        lamda_alt = np.concatenate(angles_lamda_alt)
        d_alt = np.concatenate(angles_d_alt)

        self.angles = angles

        self.angles_indices = indices
        self.angles_gamma = gamma
        self.angles_nu = nu
        self.angles_lamda = lamda
        self.angles_d = d

        self.angles_indices_alt = indices_alt
        self.angles_gamma_alt = gamma_alt
        self.angles_nu_alt = nu_alt
        self.angles_lamda_alt = lamda_alt
        self.angles_d_alt = d_alt

        return (gamma, nu, lamda, d), (gamma_alt, nu_alt, lamda_alt, d_alt)

    def simultaneous_peaks_hkl(
        self, hkl_1, hkl_2, wavelength, axes, polarities, limits, step=1
    ):
        """
        Scan goniometer settings for a pair of HKLs and keep settings
        where both reflections simultaneously satisfy the wavelength
        range and land on active detectors.

        Parameters
        ----------
        hkl_1, hkl_2 : array-like, shape (3,)
            Miller indices of the two target reflections.
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom). If min and max are
            equal, a +/-2.5% band is used instead.
        axes : list
            Per-axis (x, y, z) rotation-axis direction components.
        polarities : list
            Per-axis rotation sense/polarity.
        limits : sequence of (float, float)
            Per-axis (min, max) angle limits, in degrees.
        step : float, optional
            Base angular step size, in degrees. Default is 1.

        Returns
        -------
        angles : ndarray, shape (n_valid, n_axes)
            Per-axis angle values for each valid goniometer setting.
        values0 : tuple of ndarray
            ``(gamma0, nu0, lamda0, d0)`` for the ``hkl_1`` reflection at
            each valid setting.
        values1 : tuple of ndarray
            ``(gamma1, nu1, lamda1, d1)`` for the ``hkl_2`` reflection at
            each valid setting.
        """

        if np.isclose(wavelength[0], wavelength[1]):
            wavelength = [0.975 * wavelength[0], 1.025 * wavelength[1]]

        FilterPeaks(
            InputWorkspace="peak",
            OutputWorkspace="peak",
            FilterVariable="RunNumber",
            FilterValue=-1,
            Operator="=",
        )

        UB = mtd["coverage"].sample().getOrientedLattice().getUB().copy()

        SetUB(Workspace="peak", UB=UB)

        AddPeakHKL(Workspace="peak", HKL=hkl_1)
        AddPeakHKL(Workspace="peak", HKL=hkl_2)

        Q0_sample = mtd["peak"].getPeak(0).getQSampleFrame()
        Q1_sample = mtd["peak"].getPeak(1).getQSampleFrame()

        Q0 = np.sqrt(np.dot(Q0_sample, Q0_sample))
        Q1 = np.sqrt(np.dot(Q1_sample, Q1_sample))

        Rs, angles = self._calculate_matrices(axes, polarities, limits, step)

        Q0_lab = np.einsum("kij,j->ki", Rs, Q0_sample)
        Q1_lab = np.einsum("kij,j->ki", Rs, Q1_sample)

        lamda0 = -4 * np.pi * Q0_lab[:, 2] / Q0**2
        lamda1 = -4 * np.pi * Q1_lab[:, 2] / Q1**2

        d0 = 2 * np.pi / Q0
        d1 = 2 * np.pi / Q1

        mask = (
            (lamda0 > wavelength[0])
            & (lamda0 < wavelength[1])
            & (lamda1 > wavelength[0])
            & (lamda1 < wavelength[1])
        )

        k0 = 2 * np.pi / lamda0
        k1 = 2 * np.pi / lamda1

        k0i = k0[:, np.newaxis] * np.array([0, 0, 1])
        k1i = k1[:, np.newaxis] * np.array([0, 0, 1])

        k0f = Q0_lab + k0i
        k1f = Q1_lab + k1i

        gamma0 = np.rad2deg(np.arctan2(k0f[:, 0], k0f[:, 2]))[mask]
        gamma1 = np.rad2deg(np.arctan2(k1f[:, 0], k1f[:, 2]))[mask]

        nu0 = np.rad2deg(np.arcsin(k0f[:, 1] / k0))[mask]
        nu1 = np.rad2deg(np.arcsin(k1f[:, 1] / k1))[mask]

        lamda0 = lamda0[mask]
        lamda1 = lamda1[mask]

        angles = angles[mask]

        if len(lamda0) > 0:
            k0 = 2 * np.pi / lamda0
            k1 = 2 * np.pi / lamda1

            Q0x = k0 * np.cos(np.deg2rad(nu0)) * np.sin(np.deg2rad(gamma0))
            Q1x = k1 * np.cos(np.deg2rad(nu1)) * np.sin(np.deg2rad(gamma1))

            Q0y = k0 * np.sin(np.deg2rad(nu0))
            Q1y = k1 * np.sin(np.deg2rad(nu1))

            Q0z = k0 * (
                np.cos(np.deg2rad(nu0)) * np.cos(np.deg2rad(gamma0)) - 1
            )
            Q1z = k1 * (
                np.cos(np.deg2rad(nu1)) * np.cos(np.deg2rad(gamma1)) - 1
            )

            mask = []
            for i in range(len(k1)):
                peak0 = mtd["peaks"].createPeak(V3D(Q0x[i], Q0y[i], Q0z[i]))
                peak1 = mtd["peaks"].createPeak(V3D(Q1x[i], Q1y[i], Q1z[i]))
                det_ID0 = peak0.getDetectorID()
                det_ID1 = peak1.getDetectorID()
                mask.append(
                    (det_ID0 not in self.det_ID) & (det_ID1 not in self.det_ID)
                )

            mask = np.array(mask)

            gamma0 = gamma0[mask]
            gamma1 = gamma1[mask]

            nu0 = nu0[mask]
            nu1 = nu1[mask]

            lamda0 = lamda0[mask]
            lamda1 = lamda1[mask]

            angles = angles[mask]

        return angles, (gamma0, nu0, lamda0, d0), (gamma1, nu1, lamda1, d1)

    def get_angles(self, gamma, nu):
        """
        Look up the cached goniometer setting nearest a detector position.

        Searches the settings previously computed by
        :meth:`individual_peak` or :meth:`simultaneous_peaks` for the one
        whose detector (gamma, nu) is closest to the given position, and
        records the corresponding HKL(s) in ``self.hkl``/``self.hkl_alt``
        and a descriptive comment in ``self.comment``.

        Parameters
        ----------
        gamma : float
            Detector gamma angle (degrees) to search near.
        nu : float
            Detector nu angle (degrees) to search near.

        Returns
        -------
        angles : ndarray
            Per-axis goniometer angle values for the matched setting.
        gamma, nu : float
            Detector gamma/nu angles of the matched primary reflection.
        lamda : float
            Wavelength (Angstrom) of the matched primary reflection.
        d : float
            d-spacing (Angstrom) of the matched primary reflection.
        gamma_alt, nu_alt : float or None
            Detector gamma/nu angles of the matched secondary reflection
            (for simultaneous-peaks results), or None if not applicable.
        lamda_alt : float or None
            Wavelength of the matched secondary reflection, or None.
        d_alt : float or None
            d-spacing of the matched secondary reflection, or None.
        """

        if len(self.angles_gamma) > 0:
            d2 = (self.angles_gamma - gamma) ** 2 + (self.angles_nu - nu) ** 2

            i = np.argmin(d2)

            angles = self.angles[i]

            gamma = self.angles_gamma[i]
            nu = self.angles_nu[i]
            lamda = self.angles_lamda[i]
            d = self.angles_d[i]

            gamma_alt = nu_alt = lamda_alt = d_alt = None

            self.hkl = self.angles_indices[i]
            self.hkl_alt = None

            self.comment = "#(" + ", ".join(self.hkl.astype(str)) + ")"

            if self.angles_lamda_alt is not None:
                gamma_alt = self.angles_gamma_alt[i]
                nu_alt = self.angles_nu_alt[i]
                lamda_alt = self.angles_lamda_alt[i]
                d_alt = self.angles_d_alt[i]

                self.hkl_alt = self.angles_indices_alt[i]

                self.comment += (
                    "_#(" + ", ".join(self.hkl_alt.astype(str)) + ")"
                )

            return (
                angles,
                gamma,
                nu,
                lamda,
                d,
                gamma_alt,
                nu_alt,
                lamda_alt,
                d_alt,
            )

    def calculate_harmonics(self, hkl, wavelength, wavelength_band):
        """
        Find harmonic reflections overlapping a given reflection's beam.

        Parameters
        ----------
        hkl : array-like, shape (3,)
            Miller index of the reference reflection.
        wavelength : float
            Wavelength (Angstrom) at which the reference reflection is
            observed.
        wavelength_band : 2-tuple of float
            Minimum and maximum wavelength (Angstrom) of the instrument's
            usable band.

        Returns
        -------
        hkl_harmonics : list
            For each harmonic order, the reduced-integer HKL if it is an
            exact (order-1) harmonic, otherwise None.
        lamda_harmonics : list of float
            Wavelength (Angstrom) at which each corresponding harmonic
            order is observed.
        """

        scale = (
            np.gcd.reduce(hkl.astype(int)) if not np.mod(hkl, 1).any() else 1
        )
        n_lo = int(max(1, np.ceil(wavelength / wavelength_band[1] * scale)))
        n_hi = int(np.floor(wavelength / wavelength_band[0] * scale))

        lamda_harmonics = []
        hkl_harmonics = []
        for n in range(n_lo, n_hi + 1):
            lamda_harmonics.append(wavelength * scale / n)
            hkl_harmonics.append(hkl if scale / n == 1 else None)
        return hkl_harmonics, lamda_harmonics

    def add_mesh(
        self, mesh_angles, wavelength, d_min, rows, free_angles, all_angles
    ):
        """
        Add one orientation per point of a regular goniometer-angle mesh.

        Builds an N-dimensional grid over the free goniometer axes and
        calls :meth:`add_orientation` for every grid point.

        Parameters
        ----------
        mesh_angles : tuple
            ``(limits, ns)`` where ``limits`` is the per-axis (min, max)
            angle range and ``ns`` is the per-axis number of grid points.
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom).
        d_min : float
            Minimum d-spacing (Angstrom).
        rows : int
            Row-number offset; grid point ``i`` is stored as run number
            ``rows + i`` in the ``combined`` workspace.
        free_angles : list of str
            Names of the axes that vary across the mesh, in the order
            their values should be reported.
        all_angles : list of str
            Names of all goniometer axes, in axis order (used to map
            ``free_angles`` names to grid-point indices).

        Returns
        -------
        values : list of ndarray
            For each mesh point, the values of the ``free_angles`` axes at
            that point.
        """

        limits, ns = mesh_angles

        mins, maxs = zip(*limits)

        axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(mins, maxs, ns)]

        grids = np.meshgrid(*axes, indexing="ij")
        points = np.stack(grids, axis=-1).reshape(-1, len(limits))

        indices = [all_angles.index(free) for free in free_angles]

        values = []
        for i, angles in enumerate(points):
            self.add_orientation(angles, wavelength, d_min, rows + i)
            values.append(angles[indices])

        return values

    def calculate_projections(self, hkl_1, hkl_2):
        """Compute orthonormal u, v, w axes for a scattering plane.

        Parameters
        ----------
        hkl_1, hkl_2 : array-like, shape (3,)
            Two in-plane Miller-index vectors defining the scattering plane.

        Returns
        -------
        u, v, w : ndarray, shape (3,)
            Orthonormal basis: u along hkl_1 in Q-space, w = u × v (normal),
            v recomputed to ensure right-handed orthonormal frame.
        """
        UB = mtd["coverage"].sample().getOrientedLattice().getUB()

        u = np.dot(UB, hkl_1)
        v = np.dot(UB, hkl_2)

        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)

        w = np.cross(u, v)
        w /= np.linalg.norm(w)

        v = np.cross(w, u)

        return u, v, w

    def select_scan_near_equator(
        self,
        Rs,
        motor_angles,
        u,
        v,
        w,
        max_deg=360,
        n_steps=60,
        n_eq=(0, 1, 0),
        normal_weight=10.0,
    ):
        """
        Select, for each of a set of scan angles about ``w``, the
        goniometer setting from ``Rs`` that best matches that orientation
        while keeping the scattering plane near the equatorial plane.

        Parameters
        ----------
        Rs : array-like, shape (n_settings, 3, 3)
            Candidate goniometer rotation matrices.
        motor_angles : array-like, shape (n_settings, n_axes)
            Per-axis motor angles corresponding to each matrix in ``Rs``.
        u, v, w : array-like, shape (3,)
            Orthonormal scattering-plane basis (``u``, ``v`` in-plane,
            ``w`` normal), as returned by :meth:`calculate_projections`.
        max_deg : float, optional
            Total scan range about the plane normal ``w``, in degrees.
            Default is 360.
        n_steps : int, optional
            Number of scan steps over ``max_deg``. Default is 60.
        n_eq : 3-tuple of float, optional
            Direction defining the equatorial plane normal. Default is
            ``(0, 1, 0)``.
        normal_weight : float, optional
            Weight penalizing deviation of the scattering plane from the
            equatorial plane relative to orientation match. Default is
            10.0.

        Returns
        -------
        selected_Rs : ndarray, shape (n_steps, 3, 3)
            Best-matching rotation matrix for each scan step.
        selected_angles : ndarray, shape (n_steps, n_axes)
            Motor angles corresponding to ``selected_Rs``.
        """

        Rs = np.asarray(Rs, dtype=float)
        motor_angles = np.asarray(motor_angles)

        n_eq = np.asarray(n_eq, dtype=float)
        n_eq /= np.linalg.norm(n_eq)

        # --------------------------------------------------
        Ru_all = np.einsum("rij,j->ri", Rs, u)
        Rv_all = np.einsum("rij,j->ri", Rs, v)

        du = np.einsum("ri,i->r", Ru_all, n_eq)
        dv = np.einsum("ri,i->r", Rv_all, n_eq)

        equator_score = du**2 + dv**2

        base_idx = np.argmin(equator_score)
        R_base = Rs[base_idx]

        scan_angles = np.deg2rad(
            np.linspace(0.0, max_deg, n_steps, endpoint=False)
        )

        Rw_scan = Rotation.from_rotvec(
            scan_angles[:, None] * w[None, :]
        ).as_matrix()

        R_targets = R_base[None, :, :] @ Rw_scan

        orient_score = -np.einsum("rij,tij->rt", Rs, R_targets)

        equator_penalty = normal_weight * equator_score[None, :]

        total_score = orient_score.T + equator_penalty

        best_idx = np.argmin(total_score, axis=1)

        selected_Rs = Rs[best_idx]
        selected_angles = motor_angles[best_idx]

        return selected_Rs, selected_angles

    def compute_plane_angles(
        self,
        hkl_1,
        hkl_2,
        axes,
        polarities,
        limits,
        max_deg=360,
        n_steps=60,
        step=1,
        n_eq=(0, 1, 0),
    ):
        """Return the deduplicated motor-angle array for the scattering plane
        defined by *hkl_1* / *hkl_2* without modifying any workspace.

        Settings are chosen to keep the scattering plane close to the
        equatorial plane (defined by *n_eq*) while scanning around the
        plane normal.

        Parameters
        ----------
        hkl_1, hkl_2 : array-like, shape (3,)
            Two in-plane Miller-index vectors defining the scattering
            plane.
        axes : list
            Per-axis (x, y, z) rotation-axis direction components.
        polarities : list
            Per-axis rotation sense/polarity.
        limits : sequence of (float, float)
            Per-axis (min, max) angle limits, in degrees.
        max_deg : float, optional
            Total scan range about the plane normal, in degrees. Default
            is 360.
        n_steps : int, optional
            Number of scan steps over ``max_deg``. Default is 60.
        step : float, optional
            Base angular step size for :meth:`_calculate_matrices`, in
            degrees. Default is 1.
        n_eq : 3-tuple of float, optional
            Direction defining the equatorial plane normal. Default is
            ``(0, 1, 0)``.

        Returns
        -------
        unique_angles : list of ndarray, shape (n_axes,)
            One entry per unique goniometer setting selected.
        """
        Rs, motor_angles = self._calculate_matrices(
            axes, polarities, limits, step
        )

        u, v, w = self.calculate_projections(hkl_1, hkl_2)

        _, selected_angles = self.select_scan_near_equator(
            Rs,
            motor_angles,
            u,
            v,
            w,
            max_deg=max_deg,
            n_steps=n_steps,
            n_eq=n_eq,
        )

        seen = set()
        unique_angles = []
        for ang in selected_angles:
            key = tuple(np.round(ang).astype(int))
            if key not in seen:
                seen.add(key)
                unique_angles.append(ang)

        return unique_angles

    def add_plane(
        self,
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
        max_deg=360,
        n_steps=360,
        step=1,
    ):
        """Add orientations that keep the scattering plane defined by
        hkl_1/hkl_2 accessible, spanning *max_deg* in rotation about
        the plane normal.

        Returns the list of free-angle values (same format as add_mesh).

        Parameters
        ----------
        hkl_1, hkl_2 : array-like, shape (3,)
            Two in-plane Miller-index vectors defining the scattering
            plane.
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom).
        d_min : float
            Minimum d-spacing (Angstrom).
        rows : int
            Row-number offset; the i-th selected orientation is stored as
            run number ``rows + i`` in the ``combined`` workspace.
        free_angles : list of str
            Names of the axes that vary, in the order their values should
            be reported.
        all_angles : list of str
            Names of all goniometer axes, in axis order.
        axes : list
            Per-axis (x, y, z) rotation-axis direction components.
        polarities : list
            Per-axis rotation sense/polarity.
        limits : sequence of (float, float)
            Per-axis (min, max) angle limits, in degrees.
        max_deg : float, optional
            Total scan range about the plane normal, in degrees. Default
            is 360.
        n_steps : int, optional
            Number of scan steps over ``max_deg``. Default is 360.
        step : float, optional
            Base angular step size for :meth:`_calculate_matrices`, in
            degrees. Default is 1.

        Returns
        -------
        values : list of ndarray
            For each selected orientation, the values of the
            ``free_angles`` axes at that orientation.
        """
        unique_angles = self.compute_plane_angles(
            hkl_1, hkl_2, axes, polarities, limits, max_deg, n_steps, step
        )

        indices = [all_angles.index(free) for free in free_angles]

        values = []
        for i, angles in enumerate(unique_angles):
            self.add_orientation(angles, wavelength, d_min, rows + i)
            values.append(angles[indices])

        return values

    def add_orientation(self, angles, wavelength, d_min, rows):
        """
        Predict and store peaks for a single goniometer orientation.

        Sets the goniometer to ``angles`` on the ``instrument`` workspace,
        predicts peaks with Mantid's ``PredictPeaks``, removes duplicate
        and dead/masked-detector peaks, tags them with run number
        ``rows``, and merges them into the ``combined`` workspace.

        Parameters
        ----------
        angles : sequence of float
            Per-axis goniometer angle values, formatted into the cached
            ``self.axes`` templates.
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom). If min and max are
            equal, a +/-2.5% band is used instead.
        d_min : float
            Minimum d-spacing (Angstrom).
        rows : int
            Run number to assign to the predicted peaks -- identifies this
            orientation/row in the ``combined`` workspace. May be an
            existing row index being recalculated, or one past the last
            existing row to append a new orientation.
        """

        if np.isclose(wavelength[0], wavelength[1]):
            wavelength = [0.975 * wavelength[0], 1.025 * wavelength[1]]

        axes = np.array(self.axes).copy().tolist()

        for i, angle in enumerate(angles):
            axes[i] = axes[i].format(angle)

        ol = mtd["coverage"].sample().getOrientedLattice()
        UB = ol.getUB().copy()

        SetUB(Workspace="instrument", UB=UB)

        SetGoniometer(
            Workspace="instrument",
            Axis0=axes[0],
            Axis1=axes[1],
            Axis2=axes[2],
            Axis3=axes[3],
            Axis4=axes[4],
            Axis5=axes[5],
        )

        d_max = float("inf")

        ws = "peaks_orientation_{}".format(rows)

        PredictPeaks(
            InputWorkspace="instrument",
            MinDSpacing=d_min,
            MaxDSpacing=d_max,
            WavelengthMin=wavelength[0],
            WavelengthMax=wavelength[1],
            ReflectionCondition="Primitive",
            OutputWorkspace=ws,
        )

        SortPeaksWorkspace(
            InputWorkspace=ws,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace=ws,
        )

        columns = ["l", "k", "h"]

        for col in columns:
            SortPeaksWorkspace(
                InputWorkspace=ws,
                ColumnNameToSortBy=col,
                SortAscending=False,
                OutputWorkspace=ws,
            )

        for no in range(mtd[ws].getNumberPeaks() - 1, 0, -1):
            if (
                mtd[ws].getPeak(no).getHKL() - mtd[ws].getPeak(no - 1).getHKL()
            ).norm2() == 0:
                DeleteTableRows(TableWorkspace=ws, Rows=no)

        for no in range(mtd[ws].getNumberPeaks() - 1, 0, -1):
            if mtd[ws].getPeak(no).getDetectorID() in self.det_ID:
                DeleteTableRows(TableWorkspace=ws, Rows=no)

        SortPeaksWorkspace(
            InputWorkspace=ws,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace=ws,
        )

        for peak in mtd[ws]:
            peak.setRunNumber(rows)

        mtd["instrument"].run().getGoniometer().setR(np.eye(3))

        SetUB(Workspace="combined", UB=UB)
        CombinePeaksWorkspaces(
            LHSWorkspace="combined",
            RHSWorkspace=ws,
            OutputWorkspace="combined",
        )

    def generate_table(self, row):
        """
        Build a sorted HKL/d-spacing/wavelength table for one plan row.

        Parameters
        ----------
        row : int
            Run number (plan row) to filter peaks for, taken from the
            ``combined`` workspace. If -1 and a ``missing`` workspace
            exists (from :meth:`calculate_statistics`), that workspace of
            missing reflections is used instead.

        Returns
        -------
        table : list of list
            One row per peak, each ``[h, k, l, d, lamda]`` sorted by
            descending d-spacing (and h, k, l as tie-breakers).
        """

        if row == -1 and mtd.doesExist("missing"):
            ws = "missing"
        else:
            ws = "table"
            FilterPeaks(
                InputWorkspace="combined",
                FilterVariable="RunNumber",
                FilterValue=str(row),
                Operator="=",
                OutputWorkspace=ws,
            )

        SortPeaksWorkspace(
            InputWorkspace=ws,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace=ws,
        )

        columns = ["l", "k", "h"]

        for col in columns:
            SortPeaksWorkspace(
                InputWorkspace=ws,
                ColumnNameToSortBy=col,
                SortAscending=False,
                OutputWorkspace=ws,
            )

        SortPeaksWorkspace(
            InputWorkspace=ws,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace=ws,
        )

        h = mtd[ws].column("h")
        k = mtd[ws].column("k")
        l = mtd[ws].column("l")

        d = mtd[ws].column("DSpacing")
        lamda = mtd[ws].column("Wavelength")

        return np.array([h, k, l, d, lamda]).T.tolist()

    def _cumulative_stats(
        self, filtered_ws, pg, lc, d_min, d_max, rows, total_sym, total_asym
    ):
        """
        Compute cumulative completeness/redundancy/unique for all rows without
        workspace cloning.  Peaks are extracted once from filtered_ws; HKLs are
        reduced to their canonical symmetry-equivalent using PointGroup, then a
        two-pointer scan over sorted run numbers gives O(N_peaks + N_rows) work.

        Parameters
        ----------
        filtered_ws : str
            Name of the peaks workspace to scan (already filtered to the
            rows/runs of interest).
        pg : str
            Point group symbol used to reduce HKLs to symmetry-equivalent
            classes for the "symmetric" statistics.
        lc : str
            Lattice centering symbol used to determine allowed
            reflections.
        d_min, d_max : float
            d-spacing range (Angstrom) to include.
        rows : sequence of int
            Run numbers (plan rows), in cumulative order, at which to
            report the running statistics.
        total_sym : int
            Total number of symmetry-inequivalent reflections possible in
            range (denominator for symmetric completeness).
        total_asym : int
            Total number of reflections possible in range without
            applying symmetry (denominator for asymmetric completeness).

        Returns
        -------
        c_sym, m_sym, r_sym : list of float
            Cumulative symmetric completeness (%), redundancy, and unique
            reflection count at each row in ``rows``.
        c_asym, m_asym, r_asym : list of float
            Cumulative asymmetric (P1) completeness (%), redundancy, and
            unique reflection count at each row in ``rows``.
        """
        pg_obj = PointGroupFactory.Instance().createPointGroup(pg)
        is_allowed = centering_conditions.get(lc, lambda h, k, l: True)
        ws = mtd[filtered_ws]

        first_sym, first_asym = {}, {}
        runs_sym, runs_asym = [], []

        for i in range(ws.getNumberPeaks()):
            peak = ws.getPeak(i)
            h = int(round(peak.getH()))
            k = int(round(peak.getK()))
            l = int(round(peak.getL()))
            run = peak.getRunNumber()
            if not (d_min <= peak.getDSpacing() <= d_max):
                continue
            if not is_allowed(h, k, l):
                continue

            key_asym = (h, k, l)
            if key_asym not in first_asym or run < first_asym[key_asym]:
                first_asym[key_asym] = run
            runs_asym.append(run)

            equivs = pg_obj.getEquivalents(V3D(h, k, l))
            key_sym = min(
                (int(round(v[0])), int(round(v[1])), int(round(v[2])))
                for v in equivs
            )
            if key_sym not in first_sym or run < first_sym[key_sym]:
                first_sym[key_sym] = run
            runs_sym.append(run)

        def scan(first_dict, all_runs, total_possible):
            """
            Compute cumulative completeness/redundancy/unique counts.

            Parameters
            ----------
            first_dict : dict
                Mapping of reflection key to the first run number that
                observed it.
            all_runs : list of int
                Run number of every observation (including repeats).
            total_possible : int
                Total number of reflections possible in range (used as
                the completeness denominator).

            Returns
            -------
            comp_out : list of float
                Cumulative completeness (%) at each row in ``rows``.
            mult_out : list of float
                Cumulative redundancy (average multiplicity) at each row
                in ``rows``.
            refl_out : list of int
                Cumulative number of unique reflections at each row in
                ``rows``.
            """

            sorted_firsts = sorted(first_dict.values())
            sorted_runs = sorted(all_runs)
            comp_out, mult_out, refl_out = [], [], []
            h_ptr = o_ptr = 0
            for row in rows:
                while (
                    h_ptr < len(sorted_firsts) and sorted_firsts[h_ptr] <= row
                ):
                    h_ptr += 1
                while o_ptr < len(sorted_runs) and sorted_runs[o_ptr] <= row:
                    o_ptr += 1
                unique = h_ptr
                total_obs = o_ptr
                comp = unique / total_possible if total_possible > 0 else 0
                mult = total_obs / unique if unique > 0 else 0
                comp_out.append(comp * 100)
                mult_out.append(mult)
                refl_out.append(unique)
            return comp_out, mult_out, refl_out

        c_sym, m_sym, r_sym = scan(first_sym, runs_sym, total_sym)
        c_asym, m_asym, r_asym = scan(first_asym, runs_asym, total_asym)
        return c_sym, m_sym, r_sym, c_asym, m_asym, r_asym

    def _count_reflections(self, ws_name, pg, lc, d_min, d_max):
        """
        Cached wrapper around Mantid's ``CountReflections`` algorithm.

        Results are memoized on ``self._count_refl_cache`` keyed by the
        peak content and parameters, to avoid recomputation for repeated
        (workspace, point group, centering, d-range) combinations.

        Parameters
        ----------
        ws_name : str
            Name of the peaks workspace to analyze.
        pg : str
            Point group symbol.
        lc : str
            Lattice centering symbol.
        d_min, d_max : float
            d-spacing range (Angstrom) to include.

        Returns
        -------
        unique : int
            Number of unique (symmetry-independent) reflections observed.
        completeness : float
            Fraction (0-1) of possible reflections observed.
        redundancy : float
            Average multiplicity of observed reflections.
        extra : object
            Fourth output value of ``CountReflections`` (unused by callers
            here).
        """

        ws = mtd[ws_name]
        n = ws.getNumberPeaks()
        key = (
            frozenset(
                (
                    ws.getPeak(i).getRunNumber(),
                    round(ws.getPeak(i).getH()),
                    round(ws.getPeak(i).getK()),
                    round(ws.getPeak(i).getL()),
                )
                for i in range(n)
            ),
            pg,
            lc,
            round(d_min, 8),
            round(d_max, 8),
        )
        if key not in self._count_refl_cache:
            self._count_refl_cache[key] = CountReflections(
                InputWorkspace=ws_name,
                PointGroup=pg,
                LatticeCentering=lc,
                MinDSpacing=d_min,
                MaxDSpacing=d_max,
                MissingReflectionsWorkspace="",
            )
        return self._count_refl_cache[key]

    def calculate_statistics(self, point_group, lattice_centering, use, d_min):
        """
        Compute per-shell and cumulative completeness/redundancy statistics.

        Filters the ``combined`` workspace down to the active rows,
        computes overall and per-resolution-shell statistics (both with
        and without point-group symmetry applied), and computes the
        row-by-row cumulative statistics via :meth:`_cumulative_stats`.
        Also builds the ``missing`` workspace of symmetry-predicted but
        unobserved reflections.

        Parameters
        ----------
        point_group : str
            Point group symbol.
        lattice_centering : str
            Lattice centering symbol.
        use : list of bool
            Per-row flag (one per plan row/run number) indicating whether
            that orientation's peaks should be included.
        d_min : float
            Minimum d-spacing (Angstrom).

        Returns
        -------
        sym : tuple
            ``(shel_sym, comp_sym, mult_sym, refl_sym)`` -- per-shell
            labels, completeness (%), redundancy, and unique-reflection
            counts with point-group symmetry applied (first entry is the
            "Overall" row).
        asym : tuple
            Same layout as ``sym`` but computed in point group "1"
            (no symmetry applied).
        cumsym : tuple
            ``(x, comp_cumsym, mult_cumsym, refl_cumsym)`` -- row numbers
            and cumulative symmetric completeness/redundancy/unique counts
            as a function of row.
        cumasym : tuple
            ``(x, comp_cumasym, mult_cumasym, refl_cumasym)`` -- same as
            ``cumsym`` but without symmetry applied.

            Returns None instead if the ``combined`` workspace does not
            exist or has no peaks after filtering to the active rows.
        """

        shel_sym, comp_sym, mult_sym, refl_sym = [], [], [], []
        shel_asym, comp_asym, mult_asym, refl_asym = [], [], [], []

        comp_cumsym, mult_cumsym, refl_cumsym = [], [], []
        comp_cumasym, mult_cumasym, refl_cumasym = [], [], []

        if not mtd.doesExist("combined"):
            return None

        CloneWorkspace(InputWorkspace="combined", OutputWorkspace="filtered")

        rows = np.arange(len(use)).tolist()
        for row in rows:
            if not use[row]:
                FilterPeaks(
                    InputWorkspace="filtered",
                    FilterVariable="RunNumber",
                    FilterValue=str(row),
                    Operator="!=",
                    OutputWorkspace="filtered",
                )

        if mtd["filtered"].getNumberPeaks() == 0:
            return None

        ol = mtd["combined"].sample().getOrientedLattice()
        d_max = np.max([ol.d(1, 0, 0), ol.d(0, 1, 0), ol.d(0, 0, 1)])

        d = 1 / np.sqrt(np.linspace(1 / d_max**2, 1 / d_min**2, 5))

        pg, lc = self.get_symmetry(point_group, lattice_centering)

        symmetric = CountReflections(
            InputWorkspace="filtered",
            PointGroup=pg,
            LatticeCentering=lc,
            MinDSpacing=d_min,
            MaxDSpacing=d_max,
            MissingReflectionsWorkspace="missing",
        )

        ConvertPeaksWorkspace(
            PeakWorkspace="missing", OutputWorkspace="missing"
        )

        for peak in mtd["missing"]:
            h, k, l = peak.getHKL()
            Q = ol.qFromHKL(V3D(h, k, l))
            peak.setGoniometerMatrix(np.eye(3))
            peak.setQSampleFrame(Q)
            peak.setQLabFrame(Q)

        HFIRCalculateGoniometer(Workspace="missing", Wavelength=1)

        unique, completeness, redundancy, _, _ = symmetric

        total_possible_sym = (
            round(unique / completeness) if completeness > 0 else 0
        )

        shel_sym = ["Overall"]
        comp_sym = [completeness * 100]
        mult_sym = [redundancy]
        refl_sym = [unique]

        for i in range(len(d) - 1):
            unique, completeness, redundancy, _ = self._count_reflections(
                "filtered", pg, lc, d[i + 1], d[i]
            )

            shel_sym.append("{:.2f}-{:.2f}".format(d[i], d[i + 1]))
            comp_sym.append(completeness * 100)
            mult_sym.append(redundancy)
            refl_sym.append(unique)

        unique, completeness, redundancy, _ = self._count_reflections(
            "filtered", "1", lc, d_min, d_max
        )

        total_possible_asym = (
            round(unique / completeness) if completeness > 0 else 0
        )

        shel_asym = ["Overall"]
        comp_asym = [completeness * 100]
        mult_asym = [redundancy]
        refl_asym = [unique]

        for i in range(len(d) - 1):
            unique, completeness, redundancy, _ = self._count_reflections(
                "filtered", "1", lc, d[i + 1], d[i]
            )

            shel_asym.append("{:.2f}-{:.2f}".format(d[i], d[i + 1]))
            comp_asym.append(completeness * 100)
            mult_asym.append(redundancy)
            refl_asym.append(unique)

        (
            comp_cumsym,
            mult_cumsym,
            refl_cumsym,
            comp_cumasym,
            mult_cumasym,
            refl_cumasym,
        ) = self._cumulative_stats(
            "filtered",
            pg,
            lc,
            d_min,
            d_max,
            rows,
            total_possible_sym,
            total_possible_asym,
        )

        x = rows

        sym = (shel_sym, comp_sym, mult_sym, refl_sym)
        asym = (shel_asym, comp_asym, mult_asym, refl_asym)
        cumsym = (x, comp_cumsym, mult_cumsym, refl_cumsym)
        cumasym = (x, comp_cumasym, mult_cumasym, refl_cumasym)

        return sym, asym, cumsym, cumasym

    def downsample(self, arr, n=14):
        """
        Reduce a sequence to at most ``n`` representative elements.

        Always keeps the first and last elements; for short sequences
        keeps every other element, for longer sequences picks
        approximately evenly-spaced indices.

        Parameters
        ----------
        arr : sequence
            Sequence of values to downsample.
        n : int, optional
            Target number of interior samples to keep. Default is 14.

        Returns
        -------
        sampled : list
            Downsampled list of elements from ``arr``.
        """

        m = len(arr)
        if m == 0:
            return []
        elif m <= n // 2 + 2:
            return arr[:]
        elif m <= n + 2:
            return [arr[0]] + [val for val in arr[1:-1:2]] + [arr[-1]]

        k = n

        step = (m - 1) / (k + 1)
        indices = [int(round(step * (i + 1))) for i in range(k)]

        return [arr[0]] + [arr[i] for i in indices] + [arr[-1]]

    def hsl_to_rgb(self, hue, saturation, lightness):
        """
        Convert HSL color values to RGB.

        Parameters
        ----------
        hue : array-like
            Hue values, in degrees (0-360).
        saturation : array-like
            Saturation values (0-1), broadcastable with ``hue``.
        lightness : array-like
            Lightness values (0-1), broadcastable with ``hue``.

        Returns
        -------
        rgb : ndarray, shape (..., 3)
            RGB values (0-1) corresponding to each input HSL triple.
        """

        h = np.array(hue)
        s = np.array(saturation)
        l = np.array(lightness)

        def f(h, s, l, n):
            """
            Compute one RGB channel from HSL per the CSS HSL-to-RGB formula.

            Parameters
            ----------
            h, s, l : ndarray
                Hue (degrees), saturation, and lightness values.
            n : float
                Channel offset (0, 8, or 4 for red, green, blue
                respectively).

            Returns
            -------
            channel : ndarray
                Computed channel value(s) (0-1).
            """

            k = (n + h / 30) % 12
            a = s * np.minimum(l, 1 - l)
            return l - a * np.maximum(
                -1, np.minimum(np.minimum(k - 3, 9 - k), 1)
            )

        rgb = np.stack((f(h, s, l, 0), f(h, s, l, 8), f(h, s, l, 4)), axis=-1)

        return rgb

    def delete_angles(self, rows):
        """
        Delete plan rows/orientations from the ``combined`` workspace and
        renumber the remaining runs to be contiguous.

        Parameters
        ----------
        rows : list of int
            Run numbers (plan rows) to delete.
        """

        for row in rows:
            FilterPeaks(
                InputWorkspace="combined",
                FilterVariable="RunNumber",
                FilterValue=str(row),
                Operator="!=",
                OutputWorkspace="combined",
            )

        runs = mtd["combined"].column(0)

        _, new_runs = np.unique(runs, return_index=True)

        for new_run, peak in zip(new_runs.tolist(), mtd["combined"]):
            peak.setRunNumber(new_run)

    def swap_angles(self, rows):
        """
        Swap the run numbers of two plan rows/orientations.

        Parameters
        ----------
        rows : 2-element sequence of int
            The pair of run numbers to swap in the ``combined`` workspace.
        """

        if rows[0] == rows[1]:
            return
        for peak in mtd["combined"]:
            run = peak.getRunNumber()
            if run == rows[0]:
                peak.setRunNumber(rows[1])
            elif run == rows[1]:
                peak.setRunNumber(rows[0])

    def get_coverage_info(
        self, point_group, lattice_centering, draw_all, color, row=None
    ):
        """
        Build reciprocal-space point-cloud data for the coverage plot.

        Reduces observed HKLs to symmetry-independent representatives
        (applying the lattice centering condition), counts redundancy per
        representative, and colors each point either by direction on a
        sphere or by redundancy.

        Parameters
        ----------
        point_group : str
            Point group symbol used to generate symmetry equivalents.
        lattice_centering : str
            Lattice centering symbol; only reflections satisfying its
            centering condition are counted.
        draw_all : bool
            If True, use all active peaks (``filtered`` workspace);
            if False, restrict to the currently selected row (``table``
            workspace).
        color : str
            Coloring scheme: "Sphere" to color by direction, or
            "Redundancy" to color by observation count.
        row : int or None, optional
            If -1, use the ``missing`` reflections workspace instead of
            ``filtered``/``table``. Default is None.

        Returns
        -------
        coverage_dict : dict or None
            Dictionary with keys ``"colors"`` (per-point RGB uint8),
            ``"sizes"`` (normalized redundancy per point), ``"coords"``
            (Cartesian reciprocal-space coordinates), ``"axis_coords"``
            and ``"axis_colors"`` (coordinates/colors for the (100),
            (010), (001) axis markers), and ``"type"`` (name of the
            workspace used). Returns None if no ``filtered`` workspace
            exists yet.
        """

        pg = PointGroupFactory.createPointGroup(point_group)

        coverage_dict = {}

        UB = mtd["coverage"].sample().getOrientedLattice().getUB().copy()
        # UB_inv = np.linalg.inv(UB)

        if mtd.doesExist("filtered"):

            ws = "filtered"
            if not draw_all:
                ws = "table"
            if row == -1:
                ws = "missing"

            h = mtd[ws].column("h")
            k = mtd[ws].column("k")
            l = mtd[ws].column("l")

            hkls = np.array([h, k, l]).T.astype(int).tolist()

            cond = centering_conditions[lattice_centering]
            rep_count = defaultdict(int)
            rep_members = {}

            def rep_of(hkl):
                """
                Map an HKL to its symmetry-equivalent representative.

                Computes the point-group equivalents of ``hkl``, picks
                the lexicographically smallest as the representative,
                and records the full equivalent set in ``rep_members``
                (from the enclosing scope) the first time it is seen.

                Parameters
                ----------
                hkl : tuple of int
                    Miller index to reduce to its representative.

                Returns
                -------
                rep : tuple of int
                    Symmetry-independent representative Miller index.
                """
                eq = tuple(map(tuple, pg.getEquivalents(hkl)))
                rep = min(eq)
                if rep not in rep_members:
                    rep_members[rep] = eq
                return rep

            for hkl in hkls:
                if cond(*hkl):
                    rep_count[rep_of(tuple(hkl))] += 1

            hkl_dict = {}
            for rep, cnt in rep_count.items():
                for m in rep_members[rep]:
                    hkl_dict[m] = hkl_dict.get(m, 0) + cnt

            hkl_dict[(0, 0, 0)] = 0

            nos = np.array([value for value in hkl_dict.values()])
            hkls = np.array([key for key in hkl_dict.keys()])

            if color == "Sphere":
                r = np.sqrt(
                    hkls[:, 0] ** 2 + hkls[:, 1] ** 2 + hkls[:, 2] ** 2
                )
                theta = np.arccos(hkls[:, 2] / r)
                phi = np.arctan2(hkls[:, 1], hkls[:, 0])

                hue = phi * 180 / np.pi + 180
                saturation = np.ones_like(hue)
                lightness = theta / np.pi

                rgb = self.hsl_to_rgb(hue, saturation, lightness)
            elif color == "Redundancy":
                redundancy_values = nos.astype(float)
                max_redundancy = (
                    redundancy_values.max()
                    if redundancy_values.max() > 0
                    else 1.0
                )

                norm_redundancy = redundancy_values / max_redundancy

                cmap = plt.cm.turbo
                rgb = cmap(norm_redundancy)[:, :3]  # Take RGB, drop alpha

            coords = np.einsum("ij,nj->ni", 2 * np.pi * UB, hkls)

            coverage_dict["colors"] = (rgb * 255).astype(np.uint8)
            coverage_dict["sizes"] = nos / nos.max()
            coverage_dict["coords"] = coords

            hkls = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            r = np.sqrt(hkls[:, 0] ** 2 + hkls[:, 1] ** 2 + hkls[:, 2] ** 2)
            theta = np.arccos(hkls[:, 2] / r)
            phi = np.arctan2(hkls[:, 1], hkls[:, 0])

            hue = phi * 180 / np.pi + 180
            saturation = np.ones_like(hue)
            lightness = theta / np.pi

            coverage_dict["axis_coords"] = coords

            rgb = self.hsl_to_rgb(hue, saturation, lightness)
            coords = np.einsum("ij,nj->ni", 2 * np.pi * UB, hkls)

            coverage_dict["axis_colors"] = (rgb * 255).astype(np.uint8)
            coverage_dict["axis_coords"] = coords
            coverage_dict["type"] = ws

            return coverage_dict

    def get_laue_info(self):
        """
        Extract detector-frame angles and HKLs for the current row's peaks.

        Computes gamma/nu detector angles, wavelength, and d-spacing for
        every peak in the ``table`` workspace, and caches the results as
        ``self.lamda_peaks``, ``self.gamma_peaks``, ``self.nu_peaks``,
        ``self.d_peaks``, and ``self.hkl_peaks`` for later lookup by
        :meth:`get_peak_index`/:meth:`get_peak_selection`.

        Returns
        -------
        gamma_peaks, nu_peaks : ndarray
            Detector gamma/nu angles (degrees) for each peak.
        lamda_peaks : ndarray
            Wavelength (Angstrom) for each peak.
        d_peaks : ndarray
            d-spacing (Angstrom) for each peak.

            Returns None if the ``table`` workspace does not exist.
        """

        if mtd.doesExist("table"):

            lamda_peaks = []
            gamma_peaks = []
            nu_peaks = []
            d_peaks = []
            hkl_peaks = []

            for peak in mtd["table"]:
                lamda = peak.getWavelength()
                Q_lab = peak.getQLabFrame()
                hkl = peak.getHKL()

                Q = np.linalg.norm(Q_lab)

                d = 2 * np.pi / Q
                k = 2 * np.pi / lamda

                ki = k * np.array([0, 0, 1])
                kf = Q_lab + ki

                gamma = np.rad2deg(np.arctan2(kf[0], kf[2]))
                nu = np.rad2deg(np.arcsin(kf[1] / k))

                lamda_peaks.append(lamda)
                gamma_peaks.append(gamma)
                nu_peaks.append(nu)
                d_peaks.append(d)
                hkl_peaks.append(np.array(hkl))

            lamda_peaks = np.array(lamda_peaks)
            gamma_peaks = np.array(gamma_peaks)
            nu_peaks = np.array(nu_peaks)
            d_peaks = np.array(d_peaks)
            hkl_peaks = np.array(hkl_peaks)

            self.lamda_peaks = lamda_peaks
            self.gamma_peaks = gamma_peaks
            self.nu_peaks = nu_peaks
            self.d_peaks = d_peaks
            self.hkl_peaks = hkl_peaks

            return gamma_peaks, nu_peaks, lamda_peaks, d_peaks

    def get_peak_index(self, i):
        """
        Look up a cached Laue peak (and any co-located peaks) by index.

        Uses the arrays cached by :meth:`get_laue_info`.

        Parameters
        ----------
        i : int
            Index into the cached peak arrays.

        Returns
        -------
        gamma, nu : float
            Detector gamma/nu angles (degrees) of peak ``i``.
        lamdas : ndarray
            Wavelengths (Angstrom) of all peaks at the same detector
            position as peak ``i`` (e.g. harmonics).
        hkl : ndarray, shape (3,)
            Miller index of peak ``i``.
        wl : float
            Wavelength (Angstrom) of peak ``i``.
        i : int
            The (unchanged) input index, echoed back for convenience.

            Returns None if no peaks are cached.
        """

        if len(self.lamda_peaks) > 0:
            gamma = self.gamma_peaks[i]
            nu = self.nu_peaks[i]

            d2 = (self.gamma_peaks - gamma) ** 2 + (self.nu_peaks - nu) ** 2
            lamdas = self.lamda_peaks[np.isclose(d2, d2[i])]

            hkl = self.hkl_peaks[i]
            wl = self.lamda_peaks[i]

            return gamma, nu, lamdas, hkl, wl, i

    def get_peak_selection(self, gamma, nu):
        """
        Find the cached Laue peak nearest a given detector position.

        Uses the arrays cached by :meth:`get_laue_info`.

        Parameters
        ----------
        gamma : float
            Detector gamma angle (degrees) to search near.
        nu : float
            Detector nu angle (degrees) to search near.

        Returns
        -------
        gamma, nu : float
            Detector gamma/nu angles (degrees) of the matched peak.
        lamdas : ndarray
            Wavelengths (Angstrom) of all peaks at the same detector
            position as the match (e.g. harmonics).
        hkl : ndarray, shape (3,)
            Miller index of the matched peak.
        wl : float
            Wavelength (Angstrom) of the matched peak.
        i : int
            Index of the matched peak in the cached arrays.

            Returns None if no peaks are cached.
        """

        if len(self.lamda_peaks) > 0:
            d2 = (self.gamma_peaks - gamma) ** 2 + (self.nu_peaks - nu) ** 2

            i = np.argmin(d2)

            gamma = self.gamma_peaks[i]
            nu = self.nu_peaks[i]

            lamdas = self.lamda_peaks[np.isclose(d2, d2[i])]

            hkl = self.hkl_peaks[i]
            wl = self.lamda_peaks[i]

            return gamma, nu, lamdas, hkl, wl, i

    def to_index(self, Q, Q_max, scale, n):
        """
        Convert Q-coordinate value(s) to a clipped voxel-grid index.

        Parameters
        ----------
        Q : array-like
            Q-coordinate value(s) (inverse Angstrom).
        Q_max : float
            Maximum Q extent of the grid (inverse Angstrom).
        scale : float
            Conversion factor from Q to index units, typically
            ``(n - 1) / (2 * Q_max)``.
        n : int
            Number of voxels along this axis.

        Returns
        -------
        idx : ndarray of int
            Voxel index (indices) clipped to ``[0, n - 1]``.
        """

        idx = np.round((Q + Q_max) * scale).astype(np.int32)
        return np.clip(idx, 0, n - 1)

    def extract_data(self, ws):
        """
        Extract bin-center coordinate grids, signal, and errors from an MD workspace.

        Parameters
        ----------
        ws : str
            Name of the MD workspace to extract from.

        Returns
        -------
        coords : list of ndarray
            Meshgrid (``indexing="ij"``) of bin-center coordinates for
            each non-integrated dimension of ``ws``.
        signal : ndarray
            Signal array of ``ws``, squeezed to drop singleton dimensions.
        errors : ndarray
            Standard errors (square root of the error-squared array) of
            ``ws``, squeezed to drop singleton dimensions.
        """

        dims = mtd[ws].getNonIntegratedDimensions()

        xs = [
            np.linspace(
                dim.getMinimum() + dim.getBinWidth() / 2,
                dim.getMaximum() - dim.getBinWidth() / 2,
                dim.getNBins(),
            )
            for dim in dims
        ]

        signal = mtd[ws].getSignalArray().squeeze().copy()
        errors = np.sqrt(mtd[ws].getErrorSquaredArray().squeeze())

        return np.meshgrid(*xs, indexing="ij"), signal, errors

    def calculate_footprint(self, wavelength, d_min, n=200):
        """
        Build the reciprocal-space "footprint" workspace covering the
        instrument's detector solid angle over the wavelength band.

        For every unmasked detector pixel, traces a line in Q-space
        (voxelized on an ``n``x``n``x``n`` grid out to ``Q_max``) from the
        minimum to maximum incident wavenumber, and marks the traversed
        voxels. Creates and populates the ``"footprint"`` MD workspace;
        does nothing if it already exists.

        Parameters
        ----------
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom).
        d_min : float
            Minimum d-spacing (Angstrom), used to set the Q-space extent
            (``Q_max = 2 * pi / d_min``).
        n : int, optional
            Number of voxels along each axis of the footprint grid.
            Default is 200.
        """
        if not mtd.doesExist("footprint"):
            lamda_min, lamda_max = wavelength
            k_min = 2 * np.pi / lamda_max
            k_max = 2 * np.pi / lamda_min

            two_theta = np.array(mtd["detectors"].column("TwoTheta"))
            azimuthal = np.array(mtd["detectors"].column("Azimuthal"))
            mask = np.array(mtd["detectors"].column("detMask")) == 0

            two_theta = two_theta[mask]
            azimuthal = azimuthal[mask]

            Q_max = 2 * np.pi / d_min

            hist = np.zeros((n, n, n))

            kx_hat = np.sin(two_theta) * np.cos(azimuthal)
            ky_hat = np.sin(two_theta) * np.sin(azimuthal)
            kz_hat = np.cos(two_theta) - 1

            scale = (n - 1) / (2 * Q_max)

            Qx_1 = k_min * kx_hat
            Qy_1 = k_min * ky_hat
            Qz_1 = k_min * kz_hat

            Qx_2 = k_max * kx_hat
            Qy_2 = k_max * ky_hat
            Qz_2 = k_max * kz_hat

            Q = np.sqrt(Qx_2**2 + Qy_2**2 + Qz_2**2)
            mask = Q > Q_max
            clip = np.ones_like(Q)

            clip[mask] = Q_max / Q[mask]

            Qx_2 = Qx_2 * clip
            Qy_2 = Qy_2 * clip
            Qz_2 = Qz_2 * clip

            Qx_1_ind = self.to_index(Qx_1, Q_max, scale, n)
            Qy_1_ind = self.to_index(Qy_1, Q_max, scale, n)
            Qz_1_ind = self.to_index(Qz_1, Q_max, scale, n)

            Qx_2_ind = self.to_index(Qx_2, Q_max, scale, n)
            Qy_2_ind = self.to_index(Qy_2, Q_max, scale, n)
            Qz_2_ind = self.to_index(Qz_2, Q_max, scale, n)

            pts = np.column_stack(
                [Qx_1_ind, Qy_1_ind, Qz_1_ind, Qx_2_ind, Qy_2_ind, Qz_2_ind]
            )
            _, indices = np.unique(pts, axis=0, return_index=True)

            for i in indices:
                q1 = Qx_1_ind[i], Qy_1_ind[i], Qz_1_ind[i]
                q2 = Qx_2_ind[i], Qy_2_ind[i], Qz_2_ind[i]
                ix, iy, iz = skimage.draw.line_nd(q1, q2, endpoint=False)
                hist[ix, iy, iz] = 1

            CreateMDWorkspace(
                Dimensions=3,
                Extents=3 * [-Q_max, Q_max],
                Names="Qx,Qy,Qz",
                Units=3 * ["inv. ang."],
                OutputWorkspace="footprint",
            )
            BinMD(
                InputWorkspace="footprint",
                AlignedDim0="Qx,-{},{},{}".format(Q_max, Q_max, n),
                AlignedDim1="Qy,-{},{},{}".format(Q_max, Q_max, n),
                AlignedDim2="Qz,-{},{},{}".format(Q_max, Q_max, n),
                OutputWorkspace="footprint",
            )
            mtd["footprint"].setSignalArray(hist)

    def validate_projection(self, proj):
        """
        Validate and unpack a 3x3 HKL projection matrix.

        Parameters
        ----------
        proj : array-like
            Flat or nested sequence of 9 values, reshaped to a 3x3
            projection matrix whose rows/columns are the U, V, W vectors.

        Returns
        -------
        U, V, W : ndarray, shape (3,)
            Rows of the reshaped projection matrix.
        invalid : bool
            True if the projection matrix is singular (determinant
            close to zero), meaning U, V, W do not form a valid basis.
        """
        proj = np.array(proj).reshape(3, 3)
        invalid = np.isclose(np.linalg.det(proj), 0)
        return *proj, invalid

    def calculate_rotations(
        self,
        angles,
        U,
        V,
        W,
        normal,
        value,
        thickness,
        mesh,
        point_group="1",
        use_symmetry=False,
        factor=2,
    ):
        """
        Compute (or reuse) reciprocal-space coverage over a set of
        goniometer orientations and slice it onto an HKL plane.

        Builds a coverage MD workspace (cached under a name derived from
        the goniometer axes/angles and, if symmetry is applied, the point
        group) by rotating the footprint into reciprocal space for each
        orientation in ``angles`` and, if ``use_symmetry`` is True,
        applying the Laue point-group symmetry equivalents. If a matching
        coverage workspace already exists, it is reused instead of being
        recomputed. The resulting 3D coverage is then sliced onto the
        requested HKL plane via :meth:`_slice_meshmap`.

        Parameters
        ----------
        angles : array-like
            Either a list of goniometer angle tuples (if ``mesh`` is
            False), or ``(limits, ns)`` describing a grid of angles to
            mesh over (if ``mesh`` is True).
        U, V, W : ndarray, shape (3,)
            HKL projection basis vectors defining the slice plane.
        normal : list of int
            One-hot vector selecting which of U, V, W is the slice
            normal.
        value : float
            Slice position along the normal direction.
        thickness : float
            Slice half-thickness along the normal direction.
        mesh : bool
            If True, ``angles`` is ``(limits, ns)`` and a meshgrid of
            goniometer angles is generated; if False, ``angles`` is
            already an explicit list of angle tuples.
        point_group : str, optional
            Point group symbol used to generate symmetry equivalents when
            ``use_symmetry`` is True. Default is "1".
        use_symmetry : bool, optional
            Whether to fold in Laue point-group symmetry equivalents when
            accumulating coverage. Default is False.
        factor : int, optional
            Upsampling factor applied to the coverage array along each
            axis before binning into the MD workspace. Default is 2.

        Returns
        -------
        slice_dict : dict
            Slice dictionary as returned by :meth:`_slice_meshmap`.
        """
        if mesh:
            limits, ns = angles

            mins, maxs = zip(*limits)

            axes = [
                np.linspace(lo, hi, n) for lo, hi, n in zip(mins, maxs, ns)
            ]

            grids = np.meshgrid(*axes, indexing="ij")
            points = np.stack(grids, axis=-1).reshape(-1, len(limits))
        else:
            points = np.array(angles)

        UB = mtd["coverage"].sample().getOrientedLattice().getUB()

        title = []
        for i, angles in enumerate(points):
            axes = np.array(self.axes).copy().tolist()
            for i, angle in enumerate(angles):
                axes[i] = axes[i].format(angle)
            title.append(str(axes))

        meshmap = ",".join(title)

        if use_symmetry:
            meshmap += "_" + point_group

        self.last_meshmap = meshmap

        if not mtd.doesExist(meshmap):
            UB_inv = np.linalg.inv(UB)

            (Qx, Qy, Qz), signal, _ = self.extract_data("footprint")

            coverage = np.zeros_like(signal)
            mask = signal > 0

            Q = [Qx[mask], Qy[mask], Qz[mask]]

            n = Qx.shape[0]
            Q_max = Qx[-1, 0, 0]
            scale = (n - 1) / (2 * Q_max)

            if use_symmetry:
                pg = PointGroupFactory.createPointGroup(point_group)
                laue_group = pg.getLauePointGroupSymbol()
                pg = PointGroupFactory.createPointGroup(laue_group)
            else:
                pg = PointGroupFactory.createPointGroup("1")

            symops = list(pg.getSymmetryOperations())

            for i, angles in enumerate(points):
                axes = np.array(self.axes).copy().tolist()
                for i, angle in enumerate(angles):
                    axes[i] = axes[i].format(angle)
                SetGoniometer(
                    Workspace="instrument",
                    Axis0=axes[0],
                    Axis1=axes[1],
                    Axis2=axes[2],
                    Axis3=axes[3],
                    Axis4=axes[4],
                    Axis5=axes[5],
                )
                R = mtd["instrument"].run().getGoniometer().getR()

                Q1, Q2, Q3 = np.einsum("ij,j...->i...", R.T, Q)

                h, k, l = np.einsum("ij,j...->i...", UB_inv, [Q1, Q2, Q3])

                for symop in symops:

                    T = np.column_stack(
                        [
                            symop.transformHKL([*hkl])
                            for hkl in np.eye(3).tolist()
                        ]
                    )

                    Q1, Q2, Q3 = np.einsum("ij,j...->i...", UB @ T, [h, k, l])

                    i1 = self.to_index(Q1, Q_max, scale, n)
                    i2 = self.to_index(Q2, Q_max, scale, n)
                    i3 = self.to_index(Q3, Q_max, scale, n)

                    coverage[i1, i2, i3] += 1

            coverage = (
                coverage.repeat(factor, axis=0)
                .repeat(factor, axis=1)
                .repeat(factor, axis=2)
            )

            CreateMDWorkspace(
                Dimensions=3,
                Extents=3 * [-Q_max, Q_max],
                Names="Qx,Qy,Qz",
                Units=3 * ["inv. ang."],
                OutputWorkspace=meshmap,
            )
            BinMD(
                InputWorkspace=meshmap,
                AlignedDim0="Qx,-{},{},{}".format(Q_max, Q_max, 2 * n),
                AlignedDim1="Qy,-{},{},{}".format(Q_max, Q_max, 2 * n),
                AlignedDim2="Qz,-{},{},{}".format(Q_max, Q_max, 2 * n),
                OutputWorkspace=meshmap,
            )
            mtd[meshmap].setSignalArray(coverage)

        return self._slice_meshmap(
            meshmap, UB, U, V, W, normal, value, thickness
        )

    def reslice_last(self, U, V, W, normal, value, thickness):
        """Re-slice the most-recently computed coverage workspace without
        rebuilding the footprint or coverage map.

        Returns the same ``slice_dict`` as :meth:`calculate_rotations`, or
        ``None`` if no coverage workspace is available yet.
        """
        if self.last_meshmap is None or not mtd.doesExist(self.last_meshmap):
            return None

        UB = mtd["coverage"].sample().getOrientedLattice().getUB()

        return self._slice_meshmap(
            self.last_meshmap, UB, U, V, W, normal, value, thickness
        )

    def _slice_meshmap(self, meshmap, UB, U, V, W, normal, value, thickness):
        """Project *meshmap* coverage onto the requested HKL slice plane."""
        (Qx, Qy, Qz), coverage, _ = self.extract_data(meshmap)

        n = Qx.shape[0]

        P = np.column_stack([U, V, W])
        ub_inv = np.linalg.inv(2 * np.pi * UB @ P)

        X, Y, Z = np.einsum("ij,j...->i...", ub_inv, [Qx, Qy, Qz])

        char_dict = {0: "0", 1: "{1}", -1: "-{1}"}
        chars = ["H", "K", "L"]
        names = [
            "["
            + ",".join(
                char_dict.get(j, "{0}{1}").format(
                    j, chars[np.argmax(np.abs(P[:, i]))]
                )
                for j in P[:, i]
            )
            + "]"
            for i in range(3)
        ]

        ind = normal.index(1)

        if ind == 2:
            x, y, z = X, Y, Z
            labels, name = (names[0], names[1]), names[2]
        elif ind == 1:
            x, y, z = X, Z, Y
            labels, name = (names[0], names[2]), names[1]
        else:
            x, y, z = Y, Z, X
            labels, name = (names[1], names[2]), names[0]

        form = "{} = ({:.2f},{:.2f})"

        title = form.format(name, value - thickness, value + thickness)

        mask = np.abs(z - value) < thickness

        x, y = x[mask], y[mask]

        cov = coverage[mask]

        mask = cov > 0

        if mask.sum() > 0:
            bins_x = np.linspace(x[mask].min(), x[mask].max(), 100)
            bins_y = np.linspace(y[mask].min(), y[mask].max(), 101)
        else:
            bins_x = np.linspace(-thickness, thickness, 100)
            bins_y = np.linspace(-thickness, thickness, 101)

        stats, x_edges, y_edges, _ = binned_statistic_2d(
            x, y, cov, statistic="mean", bins=[bins_x, bins_y]
        )

        stats[np.isclose(stats, 0)] = np.nan
        stats = np.ceil(stats)

        Bp = np.dot(UB, P)

        slice_dict = {}

        slice_dict["x"] = x_edges
        slice_dict["y"] = y_edges
        slice_dict["labels"] = labels

        slice_dict["signal"] = stats.T

        Q, R = scipy.linalg.qr(Bp)

        ind = np.array(normal) != 1
        i = ind.tolist().index(False)

        slice_dict["z"] = value
        slice_dict["W"] = np.column_stack([P[:, ind], P[:, i]])

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

    def crystal_plan(self, *args):
        """
        Construct a :class:`CrystalPlan` genetic-algorithm optimizer.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :class:`CrystalPlan`, i.e.
            ``use``, ``opt``, ``axes``, ``limits``, ``wavelength``,
            ``d_min``, ``point_group``, ``lattice_centering``.

        Returns
        -------
        crystal_plan : CrystalPlan
            The constructed optimizer instance.
        """
        return CrystalPlan(*args)


class CrystalPlan:
    """
    Genetic algorithm for optimizing experiment plans in NeuXtalViz.

    This class generates, recombines, and evaluates sets of orientations
    and settings to maximize experiment coverage and completeness.
    """

    def __init__(
        self,
        use,
        opt,
        axes,
        limits,
        wavelength,
        d_min,
        point_group,
        lattice_centering,
    ):
        """
        Initialize the optimizer from the current combined peaks/UB.

        Clones the ``"combined"`` peaks workspace to ``"crystal_plan"``,
        removes rows not marked for use, and caches the UB matrix,
        wavelength band, goniometer axes/limits, resolution range, point
        group, and lattice centering needed for fitness evaluation.

        Parameters
        ----------
        use : list of bool
            Per-row flag selecting which existing peaks/orientations to
            keep in the base ``"crystal_plan"`` workspace.
        opt : list of bool
            Per-row flag indicating which rows are being optimized (used
            only for status printing here).
        axes : list of str
            Goniometer axis format strings (as used by ``SetGoniometer``)
            for each of the (up to six) axes.
        limits : list of 2-tuple of float
            ``(min, max)`` angle limits for each goniometer axis.
        wavelength : 2-tuple of float
            Minimum and maximum wavelength (Angstrom). If both values are
            equal, they are widened slightly (+/-2.5%) to give a nonzero
            band.
        d_min : float
            Minimum d-spacing (Angstrom) used for peak prediction and
            fitness evaluation.
        point_group : str
            Point group symbol used when evaluating fitness/completeness.
        lattice_centering : str
            Lattice centering symbol used when evaluating
            fitness/completeness.
        """
        CloneWorkspace(
            InputWorkspace="combined", OutputWorkspace="crystal_plan"
        )

        self.instrument = "instrument"

        rows = np.arange(len(use)).tolist()

        for row in rows:
            if not use[row]:
                FilterPeaks(
                    InputWorkspace="crystal_plan",
                    FilterVariable="RunNumber",
                    FilterValue=str(row),
                    Operator="!=",
                    OutputWorkspace="crystal_plan",
                )
            if opt[row]:
                print("Row #{}: {}".format(row, use[row]))

        if np.isclose(wavelength[0], wavelength[1]):
            wavelength = [0.975 * wavelength[0], 1.025 * wavelength[1]]

        self.wavelength = wavelength

        self.axes = axes.copy()
        self.limits = limits.copy()

        ol = mtd["coverage"].sample().getOrientedLattice()
        UB = ol.getUB().copy()

        SetUB(Workspace="instrument", UB=UB)
        SetUB(Workspace="crystal_plan", UB=UB)

        self.UB = UB.copy()
        self.d_min = d_min
        self.d_max = 1.1 * np.max(
            [ol.d(1, 0, 0), ol.d(0, 1, 0), ol.d(0, 0, 1)]
        )
        self.offset = len(use)

        self.point_group = point_group
        self.lattice_centering = lattice_centering
        self.genes = {}

        # rng seed ---------#
        np.random.seed(13)  #
        #####################

    def generation(self, i, j):
        """
        Randomly generate one orientation "gene" and predict its peaks.

        Draws a random goniometer angle within ``self.limits`` for each
        non-fixed axis, records the drawn angles in ``self.genes`` under
        the key ``"peaks_{i}_{j}"``, sets the instrument goniometer to
        those angles, and predicts peaks into a workspace of that name
        (tagged with run number ``i + self.offset``).

        Parameters
        ----------
        i : int
            Orientation index within an individual.
        j : int
            Individual index within the population.
        """
        axes = self.axes.copy()
        limits = self.limits.copy()

        ax = [None] * 6
        angles = []
        for ind, (axis, limit) in enumerate(zip(axes, limits)):
            delta = limit[1] - limit[0]
            angle = limit[0] + delta * np.random.random()
            ax[ind] = axis.format(angle)
            if not np.isclose(delta, 0):
                angles.append(angle)

        outname = "peaks_{}_{}".format(i, j)

        self.genes[outname] = angles

        SetGoniometer(
            Workspace=self.instrument,
            Axis0=ax[0],
            Axis1=ax[1],
            Axis2=ax[2],
            Axis3=ax[3],
            Axis4=ax[4],
            Axis5=ax[5],
        )

        PredictPeaks(
            InputWorkspace=self.instrument,
            CalculateStructureFactors=False,
            MinDSpacing=self.d_min,
            MaxDSpacing=self.d_max,
            WavelengthMin=self.wavelength[0],
            WavelengthMax=self.wavelength[1],
            ReflectionCondition="Primitive",
            OutputWorkspace=outname,
        )

        for pk in mtd[outname]:
            pk.setRunNumber(i + self.offset)
            pk.setIntensity(100)
            pk.setSigmaIntensity(10)

    def initialization(self, n_orient, n_indiv):
        """
        Create the initial population of individuals for the genetic algorithm.

        For each of ``n_indiv`` individuals, randomly generates
        ``n_orient`` orientation genes (see :meth:`generation`), combines
        their predicted peaks into one workspace per individual (see
        :meth:`recombination`), and evaluates its fitness.

        Parameters
        ----------
        n_orient : int
            Number of orientations (goniometer settings) per individual.
        n_indiv : int
            Number of individuals in the population.

        Returns
        -------
        fit : ndarray
            Fitness score of each individual in the initial population.
        """
        fit = []
        for j in range(n_indiv):
            for i in range(n_orient):
                self.generation(i, j)
            self.recombination(n_orient, j)
            fit.append(self.fitness("peaks_{}".format(j)))

        return np.array(fit)

    def recombination(self, n_orient, j):
        """
        Combine an individual's per-orientation predicted peaks into one workspace.

        Sequentially combines the ``n_orient`` gene peak workspaces
        ``"peaks_{i}_{j}"`` (for ``i`` in ``range(n_orient)``), starting
        from the base ``"crystal_plan"`` peaks workspace, into a single
        peaks workspace named ``"peaks_{j}"``.

        Parameters
        ----------
        n_orient : int
            Number of orientations (goniometer settings) for individual
            ``j``.
        j : int
            Individual index within the population.
        """
        individuals = "peaks_{}".format(j)
        for i in range(n_orient):
            genes = "peaks_{}_{}".format(i, j)
            if i == 0:
                CombinePeaksWorkspaces(
                    LHSWorkspace="crystal_plan",
                    RHSWorkspace=genes,
                    OutputWorkspace=individuals,
                )
            else:
                CombinePeaksWorkspaces(
                    LHSWorkspace=individuals,
                    RHSWorkspace=genes,
                    OutputWorkspace=individuals,
                )

    def fitness(self, peaks, n=5):
        """
        Score a peaks workspace by resolution-weighted completeness.

        Splits the d-spacing range ``[self.d_min, self.d_max]`` into
        ``n - 1`` shells (evenly spaced in ``1/d**2``), computes the
        reflection completeness in each shell via ``CountReflections``,
        and sums the completeness values weighted so that
        higher-resolution (smaller d-spacing) shells count more.

        Parameters
        ----------
        peaks : str
            Name of the peaks workspace to evaluate.
        n : int, optional
            Number of d-spacing shell boundaries (giving ``n - 1``
            shells). Default is 5.

        Returns
        -------
        fit : float
            Resolution-weighted completeness fitness score.
        """
        d = 1 / np.sqrt(np.linspace(1 / self.d_max**2, 1 / self.d_min**2, n))

        fit = 0

        for i in range(n - 1):
            output = CountReflections(
                InputWorkspace=peaks,
                PointGroup=self.point_group,
                LatticeCentering=self.lattice_centering,
                MinDSpacing=d[i + 1],
                MaxDSpacing=d[i],
                MissingReflectionsWorkspace="",
            )

            _, completeness, _, _ = output

            fitness = completeness * (n - i - 1)

            fit += fitness

        return fit

    def crossover(self, n_orient, best, selection):
        """
        Produce the next generation's gene workspaces via elitism and crossover.

        Clones the gene workspaces of the elite individuals in ``best``
        unchanged, then for each parent pair in ``selection`` produces
        offspring by swapping orientation genes between the two parents
        (a single split point when ``n_orient > 1``, or independent
        copies of each parent when ``n_orient == 1``). Renames all
        resulting workspaces with the ``"peak"`` prefix and rebuilds
        ``self.genes`` to reflect the new generation's gene angles.

        Parameters
        ----------
        n_orient : int
            Number of orientations (goniometer settings) per individual.
        best : array-like of int
            Indices of the elite individuals to carry over unchanged.
        selection : list of array-like
            List of parent-index pairs selected for crossover.
        """
        j = 0
        genes = "peaks_{}_{}"
        genome = "s_{}_{}"
        workspaces = []
        new_genes = {}

        for elite in best:
            for i in range(n_orient):
                CloneWorkspace(
                    InputWorkspace=genes.format(i, elite),
                    OutputWorkspace=genome.format(i, j),
                )
                workspaces.append(genome.format(i, j))
                # Copy gene angles for elite
                new_genes[genome.format(i, j)] = self.genes[
                    genes.format(i, elite)
                ]
            j += 1

        for parents in selection:
            if n_orient == 1:
                for p in parents:
                    CloneWorkspace(
                        InputWorkspace=genes.format(0, p),
                        OutputWorkspace=genome.format(0, j),
                    )
                    workspaces.append(genome.format(0, j))
                    # Copy gene angles for single-orient
                    new_genes[genome.format(0, j)] = self.genes[
                        genes.format(0, p)
                    ]
                    j += 1
            else:
                k = np.random.randint(1, n_orient)
                for i in range(k):
                    CloneWorkspace(
                        InputWorkspace=genes.format(i, parents[0]),
                        OutputWorkspace=genome.format(i, j + 0),
                    )
                    CloneWorkspace(
                        InputWorkspace=genes.format(i, parents[1]),
                        OutputWorkspace=genome.format(i, j + 1),
                    )
                    workspaces.append(genome.format(i, j + 0))
                    workspaces.append(genome.format(i, j + 1))
                    # Copy gene angles for crossover
                    new_genes[genome.format(i, j + 0)] = self.genes[
                        genes.format(i, parents[0])
                    ]
                    new_genes[genome.format(i, j + 1)] = self.genes[
                        genes.format(i, parents[1])
                    ]
                for i in range(k, n_orient):
                    CloneWorkspace(
                        InputWorkspace=genes.format(i, parents[1]),
                        OutputWorkspace=genome.format(i, j + 0),
                    )
                    CloneWorkspace(
                        InputWorkspace=genes.format(i, parents[0]),
                        OutputWorkspace=genome.format(i, j + 1),
                    )
                    workspaces.append(genome.format(i, j + 0))
                    workspaces.append(genome.format(i, j + 1))
                    new_genes[genome.format(i, j + 0)] = self.genes[
                        genes.format(i, parents[1])
                    ]
                    new_genes[genome.format(i, j + 1)] = self.genes[
                        genes.format(i, parents[0])
                    ]
                j += 2

        RenameWorkspaces(InputWorkspaces=workspaces, Prefix="peak")

        updated_genes = {}
        for old_name, angles in new_genes.items():
            new_name = old_name.replace("s_", "peaks_")
            updated_genes[new_name] = angles
        self.genes = updated_genes

    def mutation(self, n_orient, n_indiv, mutation_rate):
        """
        Randomly mutate genes of the current generation and re-evaluate fitness.

        For each individual, each orientation gene has probability
        ``mutation_rate`` of being replaced by a freshly randomized
        orientation (see :meth:`generation`); unchanged genes are left in
        place. Recombines each individual's genes (see
        :meth:`recombination`) and recomputes its fitness.

        Parameters
        ----------
        n_orient : int
            Number of orientations (goniometer settings) per individual.
        n_indiv : int
            Number of individuals in the population.
        mutation_rate : float
            Probability, in ``[0, 1]``, that any given gene is mutated.

        Returns
        -------
        fit : ndarray
            Fitness score of each individual after mutation.
        """
        fit = []
        for j in range(n_indiv):
            for i in range(n_orient):
                if np.random.random() < mutation_rate:
                    self.generation(i, j)
            self.recombination(n_orient, j)
            for i in range(n_orient):
                gene_name = "peaks_{}_{}".format(i, j)
                if gene_name not in self.genes:
                    self.genes[gene_name] = None
            fit.append(self.fitness("peaks_{}".format(j)))

        return np.array(fit)

    def optimize(self, n_orient, n_indiv, n_gener, n_elite, mutation_rate):
        """
        Run the genetic algorithm to find an optimized set of orientations.

        Initializes a population (:meth:`initialization`), then for
        ``n_gener`` generations ranks individuals by fitness, carries the
        top ``n_elite`` forward unchanged, fills the remainder of the
        population via fitness-proportional selection and crossover
        (:meth:`crossover`), and applies random mutation
        (:meth:`mutation`). Returns the orientation angles of the
        best-scoring individual from the final generation, and clones its
        combined peaks workspace to ``"combined"``.

        Parameters
        ----------
        n_orient : int
            Number of orientations (goniometer settings) per individual.
        n_indiv : int
            Number of individuals in the population.
        n_gener : int
            Number of generations to evolve.
        n_elite : int
            Number of top individuals carried over unchanged each
            generation.
        mutation_rate : float
            Probability, in ``[0, 1]``, that any given gene is mutated
            each generation.

        Returns
        -------
        values : list
            Orientation angle list for each of the ``n_orient``
            goniometer settings of the best individual found.
        """
        fit = self.initialization(n_orient, n_indiv)

        ranking = np.argsort(fit)

        for _ in range(n_gener):
            ranking = np.argsort(fit)

            best = ranking[-n_elite:]

            fraction = fit / np.sum(fit)

            selection = []

            while len(selection) < (n_indiv - n_elite) // 2:
                selection.append(
                    np.random.choice(
                        np.arange(n_indiv), size=2, p=fraction, replace=False
                    )
                )

            self.crossover(n_orient, best, selection)

            fit = self.mutation(n_orient, n_indiv, mutation_rate)

        ranking = np.argsort(fit)

        j = ranking[-1]

        values = []
        for i in range(n_orient):
            genes = "peaks_{}_{}".format(i, j)
            values.append(self.genes[genes])

        CloneWorkspace(
            InputWorkspace="peaks_{}".format(j), OutputWorkspace="combined"
        )

        return values
