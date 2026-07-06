import os

from mantid.simpleapi import (
    SelectCellWithForm,
    ShowPossibleCells,
    TransformHKL,
    CalculatePeaksHKL,
    IndexPeaks,
    FindUBUsingFFT,
    FindUBUsingLatticeParameters,
    FindUBUsingIndexedPeaks,
    FindUBFromScatteringPlane,
    OptimizeLatticeForCellType,
    CalculateUMatrix,
    HasUB,
    SetUB,
    LoadIsawUB,
    SaveIsawUB,
    FindPeaksMD,
    PredictPeaks,
    PredictSatellitePeaks,
    CentroidPeaksMD,
    IntegratePeaksMD,
    FilterPeaks,
    SortPeaksWorkspace,
    LoadIsawPeaks,
    DeleteWorkspace,
    DeleteTableRows,
    CombinePeaksWorkspaces,
    CreatePeaksWorkspace,
    ConvertQtoHKLMDHisto,
    CompactMD,
    CopySample,
    CreateSampleWorkspace,
    CloneWorkspace,
    SaveNexus,
    LoadNexus,
    SaveIsawPeaks,
    LoadIsawPeaks,
    HB3AAdjustSampleNorm,
    LoadWANDSCD,
    LoadEventNexus,
    MaskBTP,
    Rebin,
    SetGoniometer,
    PreprocessDetectorsToMD,
    CompressEvents,
    GroupDetectors,
    GroupWorkspaces,
    UnGroupWorkspace,
    RenameWorkspace,
    ConvertToMD,
    ConvertHFIRSCDtoMDE,
    LoadMD,
    SaveMD,
    MergeMD,
    LoadIsawDetCal,
    LoadParameterFile,
    LoadEmptyInstrument,
    ApplyCalibration,
    BinMD,
    SliceMD,
    ConvertUnits,
    CropWorkspace,
    MaskDetectors,
    SaveMask,
    mtd,
)

from mantid import config

config["Q.convention"] = "Crystallography"

from mantid.geometry import (
    CrystalStructure,
    ReflectionGenerator,
    ReflectionConditionFilter,
    PointGroupFactory,
    UnitCell,
    Goniometer,
)

from mantid.kernel import V3D, FloatTimeSeriesProperty

from sklearn.cluster import DBSCAN

import numpy as np
import scipy
import glob
import json
import re

from NeuXtalViz.models.base_model import NeuXtalVizModel
from NeuXtalViz.config.instruments import beamlines

lattice_group = {
    "Triclinic": "-1",
    "Monoclinic": "2/m",
    "Orthorhombic": "mmm",
    "Tetragonal": "4/mmm",
    "Rhombohedral": "-3m",
    "Hexagonal": "6/mmm",
    "Cubic": "m-3m",
}

centering_reflection = {
    "P": "Primitive",
    "I": "Body centred",
    "F": "All-face centred",
    "R": "Primitive",  # rhomb axes
    "R(obv)": "Rhombohderally centred, obverse",  # hex axes
    "R(rev)": "Rhombohderally centred, reverse",  # hex axes
    "A": "A-face centred",
    "B": "B-face centred",
    "C": "C-face centred",
}

variable = {
    "I/σ": "Signal/Noise",
    "I": "Intensity",
    "d": "DSpacing",
    "λ": "Wavelength",
    "Q": "QMod",
    "h^2+k^2+l^2": "h^2+k^2+l^2",
    "m^2+n^2+p^2": "m^2+n^2+p^2",
    "Run #": "RunNumber",
}


class UBModel(NeuXtalVizModel):
    """
    Model for UB matrix and peak table operations in NeuXtalViz.

    Provides methods for loading, saving, and manipulating
    crystallographic data, including peak finding, UB matrix
    determination, lattice refinement, and clustering. Integrates with
    Mantid algorithms for data processing and supports both conventional
    and modulated structures.
    """

    def __init__(self):
        """
        Initialize the UBModel instance and set up internal state.
        """

        super(UBModel, self).__init__()

        self.Q = None
        self.conv = "YZY"
        self.table = "ub_peaks"
        self.filter_table_backup = self.table + "_filter_backup"
        self.cell = "ub_lattice"
        self.primitive_cell = "primitive_cell"

        self.peak_info = None
        self.loaded_data_key = None
        self.loaded_data_workspaces = {}
        self.loaded_md_key = None
        self.loaded_md_workspaces = {}
        self.loaded_convert_metadata = {}
        self.requested_filenames = []
        self.detector_grouping_key = None
        self.detector_grouping_pattern = None

        CreateSampleWorkspace(OutputWorkspace="ub_lattice")
        CreateSampleWorkspace(OutputWorkspace="primitive_cell")
        CreatePeaksWorkspace(NumberOfPeaks=0, OutputWorkspace="ub_peaks")

    def has_Q(self):
        """
        Check if the Q workspace exists in Mantid.

        Returns
        -------
        bool
            True if Q workspace exists, False otherwise.
        """

        if self.Q is None:
            return False
        elif mtd.doesExist(self.Q):
            return True
        else:
            return False

    def has_peaks(self):
        """
        Check if the peaks table exists in Mantid.

        Returns
        -------
        bool
            True if peaks table exists, False otherwise.
        """

        if mtd.doesExist(self.table):
            return True
        else:
            return False

    def can_undo_filter_peaks(self):
        """
        Check if a backup of the peaks table exists to undo a filter operation.

        Returns
        -------
        bool
            True if a filter-peaks backup workspace exists, False otherwise.
        """
        return mtd.doesExist(self.filter_table_backup)

    def clear_filter_peaks_backup(self):
        """
        Delete the backup workspace used to undo peak filtering, if present.
        """
        if self.can_undo_filter_peaks():
            DeleteWorkspace(Workspace=self.filter_table_backup)

    def snapshot_filter_peaks(self):
        """
        Save a backup copy of the current peaks table before filtering.

        Does nothing if there is no peaks table. Any existing backup is
        cleared first.
        """
        if not self.has_peaks():
            return

        self.clear_filter_peaks_backup()
        CloneWorkspace(
            InputWorkspace=self.table,
            OutputWorkspace=self.filter_table_backup,
        )

    def undo_filter_peaks(self):
        """
        Restore the peaks table from the filter-peaks backup, if available.

        Replaces the current peaks table with the backup snapshot taken by
        `snapshot_filter_peaks`. Does nothing if no backup exists.
        """
        if not self.can_undo_filter_peaks():
            return

        if self.has_peaks():
            DeleteWorkspace(Workspace=self.table)

        RenameWorkspace(
            InputWorkspace=self.filter_table_backup,
            OutputWorkspace=self.table,
        )

    def has_UB(self):
        """
        Check if the UB matrix is defined on the sample.

        Returns
        -------
        bool
            True if UB matrix is present, False otherwise.
        """

        if mtd.doesExist(self.cell):
            return HasUB(Workspace=self.cell)
        else:
            return False

    def get_Q_status(self, files=None):
        """
        Get a status code describing the state of the Q-sample data.

        Parameters
        ----------
        files : list, optional
            List of data file paths to check for existence. If None, the
            status is reported as if the files do not exist.

        Returns
        -------
        status : int
            1 if the files do not exist (or none were given), 2 if the
            files exist but Q-sample data has not yet been calculated,
            3 if Q-sample data is ready.
        """
        if files is not None:
            exist = self.files_exist(files)
            if not exist:
                return 1
            else:
                if self.has_Q():
                    return 3
                else:
                    return 2
        else:
            return 1

    def get_peaks_status(self):
        """
        Get a status code describing the state of the peaks table.

        Returns
        -------
        status : int
            0 if there is no peaks table or it is empty, 1 if peaks exist
            but none are indexed, 2 if at least one peak is indexed.
        """
        if not self.has_peaks():
            return 0
        elif mtd[self.table].getNumberPeaks() == 0:
            return 0
        else:
            for peak in mtd[self.table]:
                if peak.getHKL().norm2() > 0:
                    return 2
            return 1

    def get_UB_status(self):
        """
        Get a status code describing whether a UB matrix is available.

        Returns
        -------
        status : int
            0 if no UB matrix is defined, 1 if a UB matrix is available.
        """
        if not self.has_UB():
            return 0
        else:
            return 1

    def get_UB(self):
        """
        Retrieve the UB matrix from the oriented lattice.

        Returns
        -------
        UB : ndarray
            3x3 UB matrix if present, else None.
        """

        if self.has_UB():
            return mtd[self.cell].sample().getOrientedLattice().getUB().copy()

    def update_UB(self):
        """
        Update the UB matrix on the sample and synchronize with Mantid
        workspaces.
        """

        UB = self.get_UB()

        if UB is not None:
            self.set_UB(UB)

    def get_instrument_name(self, instrument):
        """
        Get the instrument name string for a given instrument identifier.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        name : str
            Instrument name.
        """

        return beamlines[instrument]["Name"]

    def get_goniometers(self, instrument):
        """
        Get goniometer settings for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        settings : list
            List of goniometer settings.
        """

        return beamlines[instrument]["Goniometers"]

    def get_wavelength(self, instrument):
        """
        Get the wavelength for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        wavelength : float
            Wavelength in angstroms.
        """

        return beamlines[instrument]["Wavelength"]

    def get_goniometer_axes(self, instrument):
        """
        Get goniometer axis names for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        axes : list
            List of goniometer axis names.
        """

        return beamlines[instrument]["GoniometerNames"]

    def get_default_d_min(self, instrument):
        """
        Get the default minimum d-spacing for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        d_min : float
            Minimum d-spacing in angstroms.
        """

        return beamlines[instrument]["MinD"]

    def get_raw_file_path(self, instrument):
        """
        Get the raw data file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        filepath : str
            File path to raw data.
        """

        inst = beamlines[instrument]

        return os.path.join(
            "/",
            inst["Facility"],
            inst["InstrumentName"],
            "IPTS-{}",
            inst["RawFile"],
        )

    def get_example_file_path(self, instrument):
        """
        Get the example data file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        filepath : str
            File path to raw data.
        """

        inst = beamlines[instrument]

        return os.path.join(
            "/SNS/EXAMPLES/",
            inst["InstrumentName"],
            "IPTS-{}",
            inst["RawFile"],
        )

    def get_lite_file_path(self, instrument):
        """
        Get the LITE file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        filepath : str
            File path to LITE file.
        """

        inst = beamlines[instrument]

        raw = inst["RawFile"].replace("nexus", "shared/autoreduce/")
        fname, *exts = raw.split(os.extsep)
        ext = ".lite." + ".".join(exts)
        autoreduce = fname + ext

        return os.path.join(
            "/",
            inst["Facility"],
            inst["InstrumentName"],
            "IPTS-{}",
            autoreduce,
        )

    def get_shared_file_path(self, instrument, ipts):
        """
        Get the shared file path for a given instrument and IPTS.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        ipts : str
            IPTS number.

        Returns
        -------
        str
            Shared file path.
        """

        inst = beamlines[instrument]

        if ipts is not None:
            filepath = os.path.join(
                "/",
                inst["Facility"],
                inst["InstrumentName"],
                "IPTS-{}".format(ipts),
                "shared",
            )
            if os.path.exists(filepath):
                return filepath

        filepath = os.path.join("/", inst["Facility"], inst["InstrumentName"])

        return filepath

    def get_calibration_file_path(self, instrument):
        """
        Get the calibration file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        str
            Calibration file path.
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
        Get the vanadium calibration file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        str
            Vanadium calibration file path.
        """

        inst = beamlines[instrument]

        return os.path.join(
            "/", inst["Facility"], inst["InstrumentName"], "shared", "Vanadium"
        )

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

        inst = beamlines[instrument]

        filepath = os.path.join(
            "/", inst["Facility"], inst["InstrumentName"], "shared/autoreduce/"
        )

        idf = glob.glob(
            os.path.join(
                filepath, f"{inst['InstrumentName']}_Definition_*.xml"
            )
        )

        return os.path.join(filepath, idf[0]) if len(idf) > 0 else None

    def get_files(self, instrument, IPTS, runs, exp):
        """
        Get the appropriate data files for a given instrument, IPTS, and run numbers.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        IPTS : str
            IPTS number.
        runs : list
            List of run numbers.
        exp : str
            Experiment identifier.

        Returns
        -------
        filenames : list
            List of file paths to load.
        idf : str or None
            Instrument definition file path if using raw files, else None.
        grouping : str
            Grouping pattern for the instrument.
        raw : bool or None
            True if using raw files, False if using LITE files.
        message : str
            Message indicating which files are being used or if files do not exist.
        """

        rawpath = self.get_raw_file_path(instrument)
        examplepath = self.get_example_file_path(instrument)
        litepath = self.get_lite_file_path(instrument)

        inst = beamlines[instrument]
        grouping = inst["Grouping"]

        idf = self.get_autoreduce_instrument(instrument)
        raw = True

        filenames = [rawpath.format(IPTS, run) for run in runs]
        examplefilenames = [examplepath.format(IPTS, run) for run in runs]
        tutorialfilenames = [examplepath.format(12345, run) for run in runs]
        if np.all([os.path.exists(filename) for filename in filenames]):
            litefilenames = [litepath.format(IPTS, run) for run in runs]
            if np.all(
                [os.path.exists(filename) for filename in litefilenames]
            ):
                filenames = litefilenames
                raw = False
                message = "Using auto-reduced lite files"
            else:
                idf = None
                message = "Using raw files"
        elif np.all(
            [os.path.exists(filename) for filename in examplefilenames]
        ):
            filenames = examplefilenames
            idf = None
            message = "Using example files since raw data files do not exist"
        elif np.all(
            [os.path.exists(filename) for filename in tutorialfilenames]
        ):
            filenames = tutorialfilenames
            idf = None
            message = "Using tutorial files since raw data files do not exist"
        else:
            message = "Files do not exist"
            filenames, raw = [], None

        return filenames, idf, grouping, raw, message

    def _can_cache_loaded_data(self, instrument):
        """
        Check whether loaded data for an instrument can be cached.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        bool
            True if loaded data can be cached, False for instruments whose
            data should always be reloaded (e.g. DEMAND, WAND2, or any
            HFIR instrument).
        """
        if instrument in ["DEMAND", "WAND²"]:
            return False

        return "HFIR" not in self.get_raw_file_path(instrument)

    def _loaded_data_cache_key(
        self, instrument, IPTS, exp, time_stop, idf, grouping, raw
    ):
        """
        Build a cache key identifying a particular loaded-data request.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        IPTS : str
            IPTS number.
        exp : str
            Experiment identifier.
        time_stop : str
            Time stop filter used when loading event data.
        idf : str or None
            Instrument definition file path, or None.
        grouping : str
            Detector grouping pattern.
        raw : bool
            True if raw event files are being loaded.

        Returns
        -------
        key : tuple
            Hashable key uniquely identifying this loaded-data request.
        """
        return (
            instrument,
            str(IPTS),
            exp,
            time_stop if raw else None,
            idf,
            grouping,
            raw,
        )

    def _loaded_md_cache_key(self, instrument, wavelength, lorentz, min_d):
        """
        Build a cache key identifying a particular converted MD request.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        wavelength : list
            Wavelength band [min, max] used for conversion.
        lorentz : bool
            Whether the Lorentz correction was applied.
        min_d : float
            Minimum d-spacing used for conversion.

        Returns
        -------
        key : tuple
            Hashable key uniquely identifying this converted-data request.
        """
        return (
            instrument,
            tuple(wavelength),
            lorentz,
            min_d,
            self.loaded_data_key,
        )

    def _delete_loaded_file_workspaces(self, filenames):
        """
        Delete the raw loaded workspaces associated with given filenames.

        Parameters
        ----------
        filenames : list
            List of data file paths whose loaded workspaces should be
            removed from Mantid.
        """
        for filename in filenames:
            workspace = self._loaded_workspace_name(filename)
            if mtd.doesExist(workspace):
                DeleteWorkspace(Workspace=workspace)

    def _loaded_workspace_name(self, filename):
        """
        Derive the Mantid workspace name used to cache a loaded raw file.

        Parameters
        ----------
        filename : str
            Data file path.

        Returns
        -------
        name : str
            Sanitized workspace name prefixed with ``loaded_``.
        """
        workspace = os.path.basename(filename)
        workspace = re.sub(r"[^0-9A-Za-z]+", "_", workspace).strip("_")
        return f"loaded_{workspace}"

    def _loaded_md_workspace_name(self, filename):
        """
        Derive the Mantid workspace name used to cache a converted MD file.

        Parameters
        ----------
        filename : str
            Data file path.

        Returns
        -------
        name : str
            Sanitized workspace name prefixed with ``md_``.
        """
        workspace = os.path.basename(filename)
        workspace = re.sub(r"[^0-9A-Za-z]+", "_", workspace).strip("_")
        return f"md_{workspace}"

    def _clear_converted_data_cache(self):
        """
        Delete all cached converted MD workspaces and reset the MD cache.
        """
        cached_workspaces = set(self.loaded_md_workspaces.values())
        for workspace in cached_workspaces:
            if mtd.doesExist(workspace):
                DeleteWorkspace(Workspace=workspace)

        self.loaded_md_key = None
        self.loaded_md_workspaces = {}
        self.loaded_convert_metadata = {}

        for workspace in ["md", "Q3D"]:
            if mtd.doesExist(workspace):
                DeleteWorkspace(Workspace=workspace)

    def _clear_loaded_data_cache(self):
        """
        Delete all cached loaded-data and converted-MD workspaces.

        Resets the loaded-data cache state, including requested filenames
        and detector grouping information.
        """
        self._clear_converted_data_cache()

        cached_workspaces = set(self.loaded_data_workspaces.values())
        for workspace in cached_workspaces:
            if mtd.doesExist(workspace):
                DeleteWorkspace(Workspace=workspace)

        self.loaded_data_key = None
        self.loaded_data_workspaces = {}
        self.requested_filenames = []
        self.detector_grouping_key = None
        self.detector_grouping_pattern = None

        for workspace in ["data", "detectors"]:
            if mtd.doesExist(workspace):
                DeleteWorkspace(Workspace=workspace)

    def _drop_unrequested_workspaces(self, requested_filenames):
        """
        Remove cached workspaces for files no longer part of the request.

        Parameters
        ----------
        requested_filenames : list
            List of data file paths that are still requested; any cached
            filename not in this list is dropped from the cache and its
            associated workspaces deleted.
        """
        requested = set(requested_filenames)

        stale_filenames = [
            filename
            for filename in self.loaded_data_workspaces
            if filename not in requested
        ]

        for filename in stale_filenames:
            raw_workspace = self.loaded_data_workspaces.pop(filename, None)
            if raw_workspace is not None and mtd.doesExist(raw_workspace):
                DeleteWorkspace(Workspace=raw_workspace)

            md_workspace = self.loaded_md_workspaces.pop(filename, None)
            if md_workspace is not None and mtd.doesExist(md_workspace):
                DeleteWorkspace(Workspace=md_workspace)

            self.loaded_convert_metadata.pop(filename, None)

    def _group_data_workspaces(self, filenames):
        """
        Combine cached raw workspaces for the given filenames into "data".

        Parameters
        ----------
        filenames : list
            List of data file paths whose cached raw workspaces should be
            grouped together. Any existing "data" workspace is deleted
            first.
        """
        workspaces = [
            self.loaded_data_workspaces[filename]
            for filename in filenames
            if filename in self.loaded_data_workspaces
            and mtd.doesExist(self.loaded_data_workspaces[filename])
        ]

        if mtd.doesExist("data"):
            DeleteWorkspace(Workspace="data")

        if len(workspaces) == 0:
            return

        CloneWorkspace(
            InputWorkspace=workspaces[0],
            OutputWorkspace="data",
        )

    def _get_requested_loaded_workspaces(self):
        """
        Get the cached raw workspace names for the requested filenames.

        Returns
        -------
        workspaces : list
            List of Mantid workspace names corresponding to
            ``self.requested_filenames`` that are currently cached and
            exist in Mantid.
        """
        return [
            self.loaded_data_workspaces[filename]
            for filename in self.requested_filenames
            if filename in self.loaded_data_workspaces
            and mtd.doesExist(self.loaded_data_workspaces[filename])
        ]

    def _get_detector_grouping_pattern(
        self, workspace, instrument, grouping, cols, rows, c, r
    ):
        """
        Compute (and cache) a detector grouping pattern for pixel binning.

        Parameters
        ----------
        workspace : str
            Name of the workspace used to look up detector geometry.
        instrument : str
            Instrument identifier.
        grouping : str
            Detector grouping descriptor (e.g. "2x2").
        cols, rows : int
            Number of columns and rows of pixels per detector bank.
        c, r : int
            Number of columns and rows of pixels to group together.

        Returns
        -------
        pattern : str or None
            Comma-separated grouping pattern string suitable for
            `GroupDetectors`, or None if the workspace does not exist.
        """
        key = (instrument, grouping, cols, rows, c, r)

        if (
            self.detector_grouping_key == key
            and self.detector_grouping_pattern is not None
        ):
            return self.detector_grouping_pattern

        if workspace is None or not mtd.doesExist(workspace):
            return None

        PreprocessDetectorsToMD(
            InputWorkspace=workspace, OutputWorkspace="detectors"
        )

        det_map = np.asarray(mtd["detectors"].column(5)).reshape(
            -1, cols, rows
        )

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

        parts = []
        for start, end in zip(starts, ends):
            parts.append("+".join(map(str, d_sorted[start:end])))

        self.detector_grouping_key = key
        self.detector_grouping_pattern = ",".join(parts)

        return self.detector_grouping_pattern

    def _prepare_white_beam_workspace(
        self, workspace, instrument, inst, scale_c, scale_r, idf, grouping
    ):
        """
        Mask edge/lost pixels and banks, then group detectors in-place.

        Parameters
        ----------
        workspace : str
            Name of the workspace to mask and group.
        instrument : str
            Instrument identifier.
        inst : dict
            Beamline configuration dictionary for the instrument.
        scale_c, scale_r : int
            Scale factors applied to the number of pixel columns and rows
            (e.g. to account for lite-mode binning) before computing masks.
        idf : str or None
            Instrument definition file path; if None, detector grouping is
            applied based on `grouping`.
        grouping : str
            Detector grouping descriptor (e.g. "2x2") used to build the
            grouping pattern when `idf` is None.
        """
        cols, rows = beamlines[instrument]["BankPixels"]
        mask_cols, mask_rows = beamlines[instrument]["MaskEdges"]
        cols //= scale_c
        rows //= scale_r
        mask_cols //= scale_c
        mask_rows //= scale_r

        MaskBTP(
            Workspace=workspace,
            Instrument=inst["Name"],
            Tube="0-{},{}-{}".format(mask_cols, cols - mask_cols, cols),
        )
        MaskBTP(
            Workspace=workspace,
            Instrument=inst["Name"],
            Pixel="0-{},{}-{}".format(mask_rows, rows - mask_rows, rows),
        )

        mask_lost = beamlines[instrument].get("MaskLost")
        if mask_lost is not None:
            for btp in mask_lost:
                bank, tube, pixel = btp
                tube = list(tube)
                pixel = list(pixel)
                tube[0] //= scale_c
                tube[1] //= scale_c
                pixel[0] //= scale_r
                pixel[1] //= scale_r
                MaskBTP(
                    Workspace=workspace,
                    Instrument=inst["Name"],
                    Bank=bank,
                    Tube="{}-{}".format(*tube),
                    Pixel="{}-{}".format(*pixel),
                )

        for bank in beamlines[instrument]["MaskBanks"]:
            MaskBTP(
                Workspace=workspace,
                Instrument=inst["Name"],
                Bank=bank,
            )

        if idf is None:
            c, r = [int(val) for val in grouping.split("x")]
            detector_list = self._get_detector_grouping_pattern(
                workspace, instrument, grouping, cols, rows, c, r
            )
            if detector_list is not None:
                GroupDetectors(
                    InputWorkspace=workspace,
                    GroupingPattern=detector_list,
                    OutputWorkspace=workspace,
                )

    def _convert_white_beam_run(
        self, raw_workspace, md_workspace, wavelength, lorentz, q_min, q_max
    ):
        """
        Convert a white-beam raw event workspace into Q-sample MD data.

        Crops to a wavelength band, converts to d-spacing to extract the
        per-detector spectrum, then converts the workspace to Q-sample MD.

        Parameters
        ----------
        raw_workspace : str
            Name of the input raw event workspace.
        md_workspace : str
            Name of the output Q-sample MD workspace to create.
        wavelength : list
            Wavelength band [min, max] in angstroms to crop to.
        lorentz : bool
            Whether to apply the Lorentz correction during MD conversion.
        q_min, q_max : float
            Minimum and maximum momentum transfer used for cropping and
            setting the MD conversion extents.

        Returns
        -------
        d_spacing : ndarray
            Bin-centered d-spacing values.
        counts : ndarray
            Counts per detector and d-spacing bin.
        two_theta : ndarray
            Scattering angle for each detector.
        az_phi : ndarray
            Azimuthal angle for each detector.
        """
        temp_workspace = str(raw_workspace) + "_convert"

        if mtd.doesExist(temp_workspace):
            DeleteWorkspace(Workspace=temp_workspace)

        CloneWorkspace(
            InputWorkspace=raw_workspace, OutputWorkspace=temp_workspace
        )

        ConvertUnits(
            InputWorkspace=temp_workspace,
            Target="MomentumTransfer",
            OutputWorkspace=temp_workspace,
        )

        CropWorkspace(
            InputWorkspace=temp_workspace,
            XMin=q_min,
            XMax=q_max,
            OutputWorkspace=temp_workspace,
        )

        ConvertUnits(
            InputWorkspace=temp_workspace,
            Target="Wavelength",
            OutputWorkspace=temp_workspace,
        )

        CropWorkspace(
            InputWorkspace=temp_workspace,
            XMin=wavelength[0],
            XMax=wavelength[1],
            OutputWorkspace=temp_workspace,
        )

        CompressEvents(
            InputWorkspace=temp_workspace,
            Tolerance=1e-4,
            OutputWorkspace=temp_workspace,
        )

        ConvertUnits(
            InputWorkspace=temp_workspace,
            Target="dSpacing",
            OutputWorkspace=temp_workspace,
        )

        Rebin(
            InputWorkspace=temp_workspace,
            OutputWorkspace=temp_workspace,
            Params=[2 * np.pi / q_max, -0.01, 2 * np.pi / q_min],
        )

        d_spacing = mtd[temp_workspace].extractX()[0]
        d_spacing = 0.5 * (d_spacing[1:] + d_spacing[:-1])
        counts = mtd[temp_workspace].extractY().copy()

        PreprocessDetectorsToMD(
            InputWorkspace=temp_workspace,
            OutputWorkspace="detectors",
        )

        two_theta = np.array(mtd["detectors"].column("TwoTheta"))
        az_phi = np.array(mtd["detectors"].column("Azimuthal"))

        ConvertToMD(
            InputWorkspace=temp_workspace,
            QDimensions="Q3D",
            dEAnalysisMode="Elastic",
            Q3DFrames="Q_sample",
            LorentzCorrection=lorentz,
            MinValues=[-q_max, -q_max, -q_max],
            MaxValues=[+q_max, +q_max, +q_max],
            PreprocDetectorsWS="detectors",
            OutputWorkspace=md_workspace,
        )

        return d_spacing, counts, two_theta, az_phi

    def load_data(
        self, instrument, IPTS, runs, exp, time_stop, force_reload=False
    ):
        """
        Load experimental data for a given instrument and run parameters.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        IPTS : str
            IPTS number.
        runs : list
            List of run numbers.
        exp : str
            Experiment identifier.
        time_stop : float
            Time to stop loading data.
        force_reload : bool, optional
            Force reloading already cached workspaces instead of reusing them.

        Returns
        -------
        bool
            True if the "data" workspace was successfully loaded/grouped,
            False if no files were found or loading failed.
        """

        filenames, idf, grouping, raw, message = self.get_files(
            instrument, IPTS, runs, exp
        )

        print(message)

        if filenames == []:
            return False

        requested_filenames = [filename for filename in filenames]

        self.runs = runs
        self.instrument = instrument

        inst = beamlines[instrument]

        self.pixel_size = inst["PixelSize"]
        self.grouping_c, self.grouping_r = [
            int(v) for v in inst["Grouping"].split("x")
        ]

        cache_enabled = self._can_cache_loaded_data(instrument)
        cache_key = self._loaded_data_cache_key(
            instrument, IPTS, exp, time_stop, idf, grouping, raw
        )

        if force_reload or self.loaded_data_key != cache_key:
            self._clear_loaded_data_cache()
        else:
            self._drop_unrequested_workspaces(requested_filenames)

        self.requested_filenames = requested_filenames

        LoadEmptyInstrument(
            InstrumentName=inst["Name"] if idf is None else None,
            Filename=idf if idf is not None else None,
            OutputWorkspace="goniometer",
        )

        filenames = ",".join(requested_filenames)

        if instrument == "DEMAND":
            HB3AAdjustSampleNorm(
                Filename=filenames,
                OutputType="Detector",
                NormaliseBy="None",
                Grouping=grouping,
                OutputWorkspace="data",
            )
            group = mtd["data"].isGroup()
            if not group:
                GroupWorkspaces(InputWorkspaces="data", OutputWorkspace="data")
            return True
        elif instrument == "WAND²":
            LoadWANDSCD(
                Filename=filenames,
                Grouping=grouping,
                OutputWorkspace="data",
            )
            group = mtd["data"].isGroup()
            if not group:
                GroupWorkspaces(InputWorkspaces="data", OutputWorkspace="data")
            return True
        else:
            c, r = [int(val) for val in grouping.split("x")]
            scale_c = 1 if idf is None else c
            scale_r = 1 if idf is None else r

            if cache_enabled:
                missing_filenames = [
                    filename
                    for filename in requested_filenames
                    if filename not in self.loaded_data_workspaces
                    or not mtd.doesExist(self.loaded_data_workspaces[filename])
                ]

                for filename in missing_filenames:
                    workspace = self._loaded_workspace_name(filename)

                    if raw:
                        LoadEventNexus(
                            Filename=filename,
                            FilterByTimeStop=time_stop,
                            NumberOfBins=1,
                            OutputWorkspace=workspace,
                            LoadNexusInstrumentXML=False,
                        )
                    else:
                        LoadNexus(
                            Filename=filename,
                            OutputWorkspace=workspace,
                        )

                    self._prepare_white_beam_workspace(
                        workspace,
                        instrument,
                        inst,
                        scale_c,
                        scale_r,
                        idf,
                        grouping,
                    )

                    self.loaded_data_workspaces[filename] = workspace

                self.loaded_data_key = cache_key
                self._group_data_workspaces(requested_filenames)
            else:
                self.loaded_data_workspaces = {}
                for filename in requested_filenames:
                    workspace = self._loaded_workspace_name(filename)

                    if raw:
                        LoadEventNexus(
                            Filename=filename,
                            FilterByTimeStop=time_stop,
                            NumberOfBins=1,
                            OutputWorkspace=workspace,
                            LoadNexusInstrumentXML=False,
                        )
                    else:
                        LoadNexus(
                            Filename=filename,
                            OutputWorkspace=workspace,
                        )

                    self._prepare_white_beam_workspace(
                        workspace,
                        instrument,
                        inst,
                        scale_c,
                        scale_r,
                        idf,
                        grouping,
                    )

                    self.loaded_data_workspaces[filename] = workspace

                self.loaded_data_key = cache_key
                self._group_data_workspaces(requested_filenames)

            if not mtd.doesExist("data"):
                return False

            return True

    def calibrate_data(self, instrument, det_cal, gon_cal, tube_cal):
        """
        Calibrate the loaded data using detector, goniometer, and tube
        calibration files.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        det_cal : str
            Detector calibration file.
        tube_cal : str
            Tube calibration file.
        gon_cal : str
            Goniometer calibration file.
        """

        filepath = self.get_raw_file_path(instrument)

        if mtd.doesExist("data"):
            goniometers = self.get_goniometers(instrument)
            while len(goniometers) < 6:
                goniometers.append(None)

            white_beam_workspaces = self._get_requested_loaded_workspaces()
            if "HFIR" in filepath or len(white_beam_workspaces) == 0:
                workspaces = (
                    list(mtd["data"].getNames())
                    if mtd["data"].isGroup()
                    else ["data"]
                )
            else:
                workspaces = white_beam_workspaces

            for workspace in workspaces:
                SetGoniometer(
                    Workspace=workspace,
                    Axis0=goniometers[0],
                    Axis1=goniometers[1],
                    Axis2=goniometers[2],
                    Average=False if "HFIR" in filepath else True,
                )

            if tube_cal != "" and os.path.exists(tube_cal):
                LoadNexus(Filename=tube_cal, OutputWorkspace="tube_table")
                for workspace in workspaces:
                    ApplyCalibration(
                        Workspace=workspace, CalibrationTable="tube_table"
                    )

            if det_cal != "" and os.path.exists(det_cal):
                if os.path.splitext(det_cal)[1] == ".xml":
                    for workspace in workspaces:
                        LoadParameterFile(
                            Workspace=workspace, Filename=det_cal
                        )
                else:
                    for workspace in workspaces:
                        LoadIsawDetCal(
                            InputWorkspace=workspace, Filename=det_cal
                        )

            if (
                gon_cal != ""
                and os.path.exists(gon_cal)
                and os.path.splitext(gon_cal)[1] == ".xml"
            ):
                LoadParameterFile(Workspace="goniometer", Filename=gon_cal)

                inst = mtd["goniometer"].getInstrument()

                for workspace in workspaces:
                    run = mtd[workspace].run()

                    params = ["omega-offset", "chi-offset", "phi-offset"]

                    for i, param in enumerate(params):
                        if inst.hasParameter(param):
                            val = inst.getNumberParameter(param)[0]
                            name = goniometers[i].split(",")[0]
                            values = run.getProperty(name).value
                            times = run.getProperty(name).times
                            log = FloatTimeSeriesProperty(name)
                            for t, v in zip(times, values):
                                log.addValue(t, v + val)
                            run[name] = log

                    SetGoniometer(
                        Workspace=workspace,
                        Axis0=goniometers[0],
                        Axis1=goniometers[1],
                        Axis2=goniometers[2],
                        Average=False if "HFIR" in filepath else True,
                    )

                    run = mtd[workspace].run()

                    param = "goniometer-tilt"
                    if inst.hasParameter(param):
                        v = inst.getStringParameter(param)[0]
                        G = np.array(v.split(",")).astype(float).reshape(3, 3)
                        gon = run.getGoniometer()
                        gon.setR(G @ gon.getR())

    def get_number_workspaces(self):
        """
        Get the run numbers associated with the currently loaded data.

        Returns
        -------
        runs : list or None
            List of run numbers for the loaded "data" workspace, or None
            if no data has been loaded.
        """

        if mtd.doesExist("data"):
            return self.runs

    def convert_data(
        self, instrument, wavelength, lorentz, min_d=None, force_reload=False
    ):
        """
        Convert loaded data to Q-space using Mantid algorithms.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        wavelength : float
            Wavelength in angstroms.
        lorentz : bool
            Whether to apply Lorentz correction.
        min_d : float, optional
            Minimum d-spacing. Default is None.
        force_reload : bool, optional
            Force reconversion of cached MD workspaces.
        """

        filepath = self.get_raw_file_path(instrument)

        if min_d is not None:
            Q_max = 2 * np.pi / min_d

        if mtd.doesExist("data"):
            self.detector_ids = np.array([], dtype=int)

            input_ws_names = (
                list(mtd["data"].getNames())
                if mtd["data"].isGroup()
                else ["data"]
            )
            input_ws = input_ws_names[0]

            Rs = []

            if "HFIR" in filepath:
                r = mtd[input_ws].getExperimentInfo(0).run()

                two_theta = r.getProperty("TwoTheta").value
                az_phi = r.getProperty("Azimuthal").value

                for ws in input_ws_names:
                    r = mtd[ws].getExperimentInfo(0).run()
                    R = []
                    n_gon = r.getNumGoniometers()
                    for i in range(n_gon):
                        gon = r.getGoniometer(i)
                        R.append(gon.getR())
                        conv = gon.getConventionFromMotorAxes()
                        if conv == "YYY":
                            conv = "YZY"
                    Rs.append(R)

                lamda = wavelength[0]

                counts = [
                    np.swapaxes(mtd[ws].getSignalArray().copy(), 0, 1)
                    for ws in input_ws_names
                ]

                counts = [c.reshape(-1, c.shape[2]) for c in counts]

                if min_d is None:
                    k = 2 * np.pi / wavelength[0]
                    Q_max = k * np.sin(0.5 * max(two_theta))

                ConvertHFIRSCDtoMDE(
                    InputWorkspace="data",
                    Wavelength=wavelength[0],
                    LorentzCorrection=lorentz,
                    MinValues=[-Q_max, -Q_max, -Q_max],
                    MaxValues=[+Q_max, +Q_max, +Q_max],
                    MaxRecursionDepth=5,
                    OutputWorkspace="md",
                )

                input_ws_names = mtd["md"].getNames()
                input_ws = input_ws_names[0]

                if len(input_ws_names) > 1:
                    MergeMD(InputWorkspaces="md", OutputWorkspace="md")

                else:
                    UnGroupWorkspace(InputWorkspace="md")

                    RenameWorkspace(
                        InputWorkspace=input_ws, OutputWorkspace="md"
                    )
            else:
                raw_workspaces = self._get_requested_loaded_workspaces()
                if len(raw_workspaces) == 0:
                    raw_workspaces = input_ws_names

                PreprocessDetectorsToMD(
                    InputWorkspace=raw_workspaces[0],
                    OutputWorkspace="detectors",
                )

                two_theta = mtd["detectors"].column("TwoTheta")
                az_phi = mtd["detectors"].column("Azimuthal")
                self.detector_ids = np.array(
                    mtd["detectors"].column("DetectorID"), dtype=int
                )

                if min_d is None:
                    k = 2 * np.pi / min(wavelength)
                    Q_max = k * np.sin(0.5 * max(two_theta))

                k = 2 * np.pi / max(wavelength)
                Q_min = k * np.sin(0.5 * min(two_theta))

                convert_key = self._loaded_md_cache_key(
                    instrument, wavelength, lorentz, min_d
                )

                if force_reload or self.loaded_md_key != convert_key:
                    self._clear_converted_data_cache()

                d = None
                conv = None
                counts = []
                Rs = []
                two_theta = None
                az_phi = None

                for filename in self.requested_filenames:
                    raw_workspace = self.loaded_data_workspaces.get(filename)
                    if raw_workspace is None or not mtd.doesExist(
                        raw_workspace
                    ):
                        continue

                    metadata = self.loaded_convert_metadata.get(filename)
                    md_workspace = self.loaded_md_workspaces.get(filename)

                    if (
                        metadata is None
                        or md_workspace is None
                        or not mtd.doesExist(md_workspace)
                    ):
                        md_workspace = self._loaded_md_workspace_name(filename)
                        if mtd.doesExist(md_workspace):
                            DeleteWorkspace(Workspace=md_workspace)

                        d, vals, two_theta, az_phi = (
                            self._convert_white_beam_run(
                                raw_workspace,
                                md_workspace,
                                wavelength,
                                lorentz,
                                Q_min,
                                Q_max,
                            )
                        )

                        run = mtd[raw_workspace].run()
                        gon = run.getGoniometer(0)
                        R = gon.getR()
                        conv = gon.getConventionFromMotorAxes()
                        if conv == "YYY":
                            conv = "YZY"

                        self.loaded_convert_metadata[filename] = (
                            d,
                            vals,
                            two_theta,
                            az_phi,
                            R,
                            conv,
                        )
                        self.loaded_md_workspaces[filename] = md_workspace

                    d, vals, two_theta, az_phi, R, conv = (
                        self.loaded_convert_metadata[filename]
                    )

                    Rs.append(R)
                    counts.append(vals)

                self.loaded_md_key = convert_key

                md_workspaces = [
                    self.loaded_md_workspaces[filename]
                    for filename in self.requested_filenames
                    if filename in self.loaded_md_workspaces
                    and mtd.doesExist(self.loaded_md_workspaces[filename])
                ]

                if len(md_workspaces) == 0:
                    return

                if len(md_workspaces) == 1:
                    if mtd.doesExist("md"):
                        DeleteWorkspace(Workspace="md")
                    CloneWorkspace(
                        InputWorkspace=md_workspaces[0], OutputWorkspace="md"
                    )
                else:
                    if mtd.doesExist("md"):
                        DeleteWorkspace(Workspace="md")
                    MergeMD(
                        InputWorkspaces=",".join(md_workspaces),
                        OutputWorkspace="md",
                    )

                lamda = None

            self.wavelength = wavelength
            self.counts = counts

            self.two_theta = np.array(two_theta)
            self.scan = lamda if "HFIR" in filepath else d
            self.Rs = Rs
            self.conv = conv

            kf_x = np.sin(two_theta) * np.cos(az_phi)
            kf_y = np.sin(two_theta) * np.sin(az_phi)
            kf_z = np.cos(two_theta)

            self.nu = np.rad2deg(np.arcsin(kf_y))
            self.gamma = np.rad2deg(np.arctan2(kf_x, kf_z))

            self.make_Q(Q_max)

    def make_Q(self, Q_max):
        """
        Bin the converted MD workspace into a 3D Q-sample volume for display.

        Creates the "Q3D" histogram volume and associated peaks/lattice
        workspaces, then computes a normalized, outlier-masked log signal
        array for rendering.

        Parameters
        ----------
        Q_max : float
            Maximum extent of the Q-sample volume along each axis, and the
            radius beyond which voxels are masked out.
        """

        self.Q_max_cut = Q_max

        self.Q = "md"

        if mtd.doesExist("Q3D"):
            DeleteWorkspace(Workspace="Q3D")

        BinMD(
            InputWorkspace=self.Q,
            AlignedDim0="Q_sample_x,{},{},256".format(-Q_max, Q_max),
            AlignedDim1="Q_sample_y,{},{},256".format(-Q_max, Q_max),
            AlignedDim2="Q_sample_z,{},{},256".format(-Q_max, Q_max),
            OutputWorkspace="Q3D",
        )

        CreatePeaksWorkspace(
            InstrumentWorkspace=self.Q,
            NumberOfPeaks=0,
            OutputWorkspace=self.table,
        )

        CopySample(
            InputWorkspace=self.Q,
            OutputWorkspace=self.cell,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

        CompactMD(InputWorkspace="Q3D", OutputWorkspace="Q3D")

        signal = mtd["Q3D"].getSignalArray().copy()

        mu = scipy.ndimage.gaussian_filter(signal, 4, mode="nearest")
        mu2 = scipy.ndimage.gaussian_filter(signal**2, 4, mode="nearest")
        var = np.maximum(mu2 - mu**2, 0.0)
        std = np.sqrt(var) + 1e-6

        Z = (signal - mu) / std

        mask = Z > 3

        signal[~mask] = np.nan

        signal[np.isclose(signal, 0)] = np.nan

        # threshold = np.nanpercentile(signal, 99)
        # signal[signal >= threshold] = threshold

        dims = [mtd["Q3D"].getDimension(i) for i in range(3)]

        x, y, z = [
            np.linspace(
                dim.getMinimum() + dim.getBinWidth() / 2,
                dim.getMaximum() - dim.getBinWidth() / 2,
                dim.getNBins(),
            )
            for dim in dims
        ]

        Qx, Qy, Qz = np.meshgrid(x, y, z, indexing="ij")

        mask = (Qx**2 + Qy**2 + Qz**2) > self.Q_max_cut**2
        signal[mask] = np.nan

        self.spacing = tuple([dim.getBinWidth() for dim in dims])

        self.min_lim = x[0], y[0], z[0]
        self.max_lim = x[-1], y[-1], z[-1]

        signal = np.log10(signal)

        smin = np.nanmin(signal)
        smax = np.nanmax(signal)

        self.signal = np.round(255 * (signal - smin) / (smax - smin)).astype(
            np.float32
        )

    def is_mono(self, wavelength):
        """
        Check whether a wavelength band corresponds to monochromatic beam.

        Parameters
        ----------
        wavelength : list
            Wavelength band [min, max] in angstroms.

        Returns
        -------
        bool
            True if the minimum and maximum wavelength are equal
            (monochromatic beam), False otherwise.
        """
        return np.isclose(wavelength[0], wavelength[1])

    def get_run_goniometer(self, ind):
        """
        Return goniometer Euler angles for the selected run.

        Parameters
        ----------
        ind : int
            Index of the run/goniometer setting.

        Returns
        -------
        angles : tuple or None
            Goniometer Euler angles (in degrees) for the given run index,
            or None if Q-sample data, wavelength info, or a valid index
            is not available.
        """

        if not self.has_Q() or ind is None:
            return None

        wavelength = getattr(self, "wavelength", None)
        if wavelength is None:
            return None

        if ind < 0 or ind >= len(self.Rs):
            return None

        R = self.Rs[ind]

        if np.ndim(R) == 3:
            if type(self.scan) is float:
                val = (
                    self.roi_view["val"] if hasattr(self, "roi_view") else None
                )
                if val is None:
                    R = R[0]
                else:
                    x = np.rad2deg(
                        np.arccos([0.5 * (np.trace(r) - 1) for r in R])
                    )
                    R = R[np.argmin(np.abs(x - val))]
            else:
                R = R[0]

        goniometer = Goniometer()
        goniometer.setR(R)

        return list(goniometer.getEulerAngles(self.conv))

    def add_peak(self, ind, val, horz, vert):
        """
        Add a peak to the peaks table.

        Parameters
        ----------
        ind : int
            Index of the run.
        val : float
            Peak value (e.g., d-spacing or angle).
        horz : float
            Horizontal coordinate (e.g., angle or hkl).
        vert : float
            Vertical coordinate (e.g., angle or hkl).
        """

        R = self.Rs[ind]

        gamma = np.deg2rad(horz)
        nu = np.deg2rad(vert)

        if type(self.scan) is float:
            wl = self.scan
            x = np.rad2deg(np.arccos([0.5 * (np.trace(r) - 1) for r in R]))
            R = R[np.argmin(np.abs(x - val))]
        else:
            d = val
            two_theta = np.arccos(np.clip(np.cos(gamma) * np.cos(nu), -1, 1))
            wl = 2 * d * np.sin(0.5 * two_theta)

        k = 2 * np.pi / wl

        Qx = k * np.cos(nu) * np.sin(gamma)
        Qy = k * np.sin(nu)
        Qz = k * (np.cos(nu) * np.cos(gamma) - 1)

        mtd["ub_peaks"].run().getGoniometer().setR(R)
        peak = mtd["ub_peaks"].createPeak([Qx, Qy, Qz])
        peak.setRunNumber(self.runs[ind])
        mtd["ub_peaks"].addPeak(peak)

    def add_peak_from_hkl(self, ind, hkl):
        """
        Add a peak to the peaks workspace using HKL coordinates.

        Parameters
        ----------
        ind : int
            Index of the run/goniometer setting to use.
        hkl : list
            Miller indices [h, k, l] of the peak to add.
        """

        if self.has_UB() and self.has_peaks():
            UB = self.get_UB()
            Q = 2 * np.pi * np.dot(UB, hkl)
            int_hkl = np.round(hkl).astype(int)

            for run_index, Rs in enumerate(self.Rs):
                matrices = Rs if np.ndim(Rs) == 3 else [Rs]

                for R in matrices:
                    mtd[self.table].run().getGoniometer().setR(R)

                    Q_lab = np.dot(R, Q)
                    lamda = -4 * np.pi * Q_lab[2] / np.dot(Q_lab, Q_lab)

                    if lamda <= 0:
                        continue

                    peak = mtd[self.table].createPeak(Q_lab.tolist())

                    peak.setRunNumber(self.runs[run_index])
                    peak.setHKL(*hkl)
                    peak.setIntHKL(
                        V3D(*np.array(int_hkl).astype(float).tolist())
                    )
                    peak.setIntMNP(V3D(0.0, 0.0, 0.0))

                    mtd[self.table].addPeak(peak)
                    return

    def delete_peak(self, no):
        """
        Delete a single peak row from the current peaks workspace.

        Parameters
        ----------
        no : int
            Row index of the peak to delete.
        """

        if self.has_peaks() and 0 <= no < mtd[self.table].getNumberPeaks():
            DeleteTableRows(TableWorkspace=self.table, Rows=no)

    def delete_peak_rows(self, numbers):
        """
        Delete multiple peak rows from the current peaks workspace.

        Parameters
        ----------
        numbers : list
            Row indices of the peaks to delete.
        """

        if not self.has_peaks():
            return

        row_count = mtd[self.table].getNumberPeaks()
        valid_rows = [no for no in numbers if 0 <= no < row_count]

        for no in sorted(set(valid_rows), reverse=True):
            DeleteTableRows(TableWorkspace=self.table, Rows=no)

    def calculate_hkl_position(self, ind, h, k, l):
        """
        Calculate the HKL position for a given peak index and Miller indices.

        Parameters
        ----------
        ind : int
            Index of the peak.
        h, k, l : int
            Miller indices.

        Returns
        -------
        tuple
            Calculated values (x, gamma, nu) for the given HKL.
        """

        if self.has_UB():
            UB = self.get_UB()
            Q = 2 * np.pi * UB @ np.array([h, k, l])

            R = self.Rs[ind]

            if type(self.scan) is float:
                wl = self.scan
                Q = np.einsum("kij,j->ki", R, Q)
                lamda = (
                    4 * np.pi * np.abs(Q[2]) / np.linalg.norm(Q, axis=0) ** 2
                )
                R = R[np.argmin(np.abs(lamda - wl))]
                Q = np.einsum("ij,j->i", R, Q)
                x = np.rad2deg(np.arccos(0.5 * (np.trace(R) - 1)))
            else:
                Q = np.einsum("ij,j->i", R, Q)
                d = 2 * np.pi / np.linalg.norm(Q)
                x = d

            az_phi = np.arctan2(Q[1], Q[0])
            two_theta = 2 * np.abs(np.arcsin(Q[2] / np.linalg.norm(Q)))

            kf_x = np.sin(two_theta) * np.cos(az_phi)
            kf_y = np.sin(two_theta) * np.sin(az_phi)
            kf_z = np.cos(two_theta)

            nu = np.rad2deg(np.arcsin(kf_y))
            gamma = np.rad2deg(np.arctan2(kf_x, kf_z))

            return x, gamma, nu

    def roi_scan_to_hkl(self, ind, val, horz, vert):
        """
        Convert a ROI scan position to HKL coordinates.

        Parameters
        ----------
        ind : int
            Index of the run.
        val : float
            Peak value (e.g., d-spacing or angle).
        horz : float
            Horizontal coordinate (e.g., angle or hkl).
        vert : float
            Vertical coordinate (e.g., angle or hkl).

        Returns
        -------
        np.ndarray
            Calculated HKL coordinates.
        """

        if self.has_UB():
            R = self.Rs[ind]

            gamma = np.deg2rad(horz)
            nu = np.deg2rad(vert)

            if type(self.scan) is float:
                wl = self.scan
                x = np.rad2deg(np.arccos([0.5 * (np.trace(r) - 1) for r in R]))
                R = R[np.argmin(np.abs(x - val))]
            else:
                d = val
                two_theta = np.arccos(
                    np.clip(np.cos(gamma) * np.cos(nu), -1, 1)
                )
                wl = 2 * d * np.sin(0.5 * two_theta)

            k = 2 * np.pi / wl

            Qx = k * np.cos(nu) * np.sin(gamma)
            Qy = k * np.sin(nu)
            Qz = k * (np.cos(nu) * np.cos(gamma) - 1)

            UB = self.get_UB()

            hkl = np.dot(np.linalg.inv(2 * np.pi * (R @ UB)), [Qx, Qy, Qz])

            return hkl

    def calculate_instrument_view(self, ind, d_min, d_max):
        """
        Calculate the instrument view for a given peak index and d-spacing range.

        Parameters
        ----------
        ind : int
            Index of the peak.
        d_min, d_max : float
            Minimum and maximum d-spacing values.

        Returns
        -------
        inst_view : dict
            Instrument view data including d, gamma, nu, and counts.
        """

        inst_view = {}

        R = self.Rs[ind]

        two_theta = self.two_theta
        gamma = self.gamma
        nu = self.nu

        if type(self.scan) is float:
            lamda = np.full(len(R), self.scan)
            d_spacing = 0.5 * lamda / np.sin(0.5 * two_theta[:, np.newaxis])
        else:
            d = self.scan
            d_spacing = d * np.ones_like(two_theta)[:, np.newaxis]

        if np.isclose(d_min, d_max) or d_max < d_min:
            d_min, d_max = 0, np.inf

        mask = (d_spacing > d_min) & (d_spacing < d_max)

        rows, cols = np.nonzero(mask)

        vals = self.counts[ind].copy()
        vals[~mask] = np.nan

        uni_rows = np.unique(rows)

        counts = np.nansum(vals[uni_rows], axis=1)

        sort = np.argsort(counts)

        gamma_arr = gamma[uni_rows][sort]
        nu_arr = nu[uni_rows][sort]
        counts_arr = counts[sort]

        detector_ids = getattr(self, "detector_ids", np.array([], dtype=int))
        if detector_ids.size == gamma.size:
            det_ids_arr = detector_ids[uni_rows][sort]
        else:
            det_ids_arr = np.array([], dtype=int)

        inst_view["d"] = d_spacing
        inst_view["d_min"] = d_min
        inst_view["d_max"] = d_max
        inst_view["gamma"] = gamma_arr
        inst_view["nu"] = nu_arr
        inst_view["counts"] = counts_arr
        inst_view["detector_ids"] = det_ids_arr
        inst_view["ind"] = ind

        if len(gamma_arr) == 0:
            img = np.full((1, 1), np.nan)
            xedges = np.array([-0.5, 0.5])
            yedges = np.array([-0.5, 0.5])
        else:
            dx = self.pixel_size[0] * self.grouping_c
            dy = self.pixel_size[1] * self.grouping_r

            if not np.isfinite(dx) or dx <= 0:
                x_span = np.ptp(gamma_arr)
                dx = x_span / 200 if x_span > 0 else 1.0

            if not np.isfinite(dy) or dy <= 0:
                y_span = np.ptp(nu_arr)
                dy = y_span / 200 if y_span > 0 else 1.0

            xedges = np.arange(gamma_arr.min(), gamma_arr.max() + dx, dx)
            yedges = np.arange(nu_arr.min(), nu_arr.max() + dy, dy)

            if len(xedges) < 2:
                xedges = np.array(
                    [gamma_arr.min() - 0.5, gamma_arr.max() + 0.5]
                )
            if len(yedges) < 2:
                yedges = np.array([nu_arr.min() - 0.5, nu_arr.max() + 0.5])

            sum_I, xedges, yedges = np.histogram2d(
                gamma_arr, nu_arr, bins=[xedges, yedges], weights=counts_arr
            )
            count_map, _, _ = np.histogram2d(
                gamma_arr, nu_arr, bins=[xedges, yedges]
            )
            img = sum_I / count_map
            img[count_map == 0] = np.nan
            img[sum_I == 0] = np.nan

        inst_view["img"] = img
        inst_view["xedges"] = xedges
        inst_view["yedges"] = yedges

        self.inst_view = inst_view

    def save_roi_mask(self, instrument, filename):
        """
        Save a detector mask XML file for the current instrument-view ROI.

        Loads an empty instrument definition, determines which detectors
        fall within the current ROI box (in gamma/nu), and writes a Mantid
        mask file selecting those detectors.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        filename : str
            Output path for the saved mask XML file.

        Returns
        -------
        success : bool
            True if the mask was saved successfully, False otherwise.
        message : str
            Description of the outcome or error.
        """
        if not hasattr(self, "inst_view") or not hasattr(self, "roi_view"):
            return False, "No instrument ROI is available to save."

        horz = float(self.roi_view.get("horz", 0.0))
        vert = float(self.roi_view.get("vert", 0.0))
        horz_roi = float(self.roi_view.get("horz_roi", 0.0))
        vert_roi = float(self.roi_view.get("vert_roi", 0.0))

        inst_name = self.get_instrument_name(instrument)

        mask_ws = "ub_roi_mask"
        det_ws = "ub_roi_mask_detectors"

        for ws in [mask_ws, det_ws]:
            if mtd.doesExist(ws):
                DeleteWorkspace(Workspace=ws)

        try:
            LoadEmptyInstrument(
                InstrumentName=inst_name,
                OutputWorkspace=mask_ws,
            )

            # Build gamma/nu on the full instrument pixel map.
            PreprocessDetectorsToMD(
                InputWorkspace=mask_ws,
                OutputWorkspace=det_ws,
            )

            detector_ids = np.asarray(
                mtd[det_ws].column("DetectorID"), dtype=int
            )
            two_theta = np.asarray(mtd[det_ws].column("TwoTheta"), dtype=float)
            az_phi = np.asarray(mtd[det_ws].column("Azimuthal"), dtype=float)

            kf_x = np.sin(two_theta) * np.cos(az_phi)
            kf_y = np.sin(two_theta) * np.sin(az_phi)
            kf_z = np.cos(two_theta)

            nu = np.rad2deg(np.arcsin(kf_y))
            gamma = np.rad2deg(np.arctan2(kf_x, kf_z))

            in_roi = (
                (gamma >= horz - horz_roi)
                & (gamma <= horz + horz_roi)
                & (nu >= vert - vert_roi)
                & (nu <= vert + vert_roi)
            )

            selected = np.unique(detector_ids[in_roi])
            selected = selected[selected >= 0]

            if selected.size == 0:
                return False, "No detectors were selected in the ROI box."

            MaskDetectors(
                Workspace=mask_ws,
                DetectorList=selected.tolist(),
            )

            SaveMask(InputWorkspace=mask_ws, OutputFile=filename)
        finally:
            for ws in [det_ws, mask_ws]:
                if mtd.doesExist(ws):
                    DeleteWorkspace(Workspace=ws)

        return (
            True,
            "Saved ROI mask XML ({} detectors from full-pixel instrument ROI).".format(
                selected.size
            ),
        )

    def extract_roi(self, horz, vert, horz_roi, vert_roi, val):
        """
        Extract a region of interest (ROI) from the instrument view.

        Parameters
        ----------
        horz, vert : float
            Horizontal and vertical coordinates for the ROI center.
        horz_roi, vert_roi : float
            Horizontal and vertical ROI sizes.
        val : float
            Value at the ROI center.

        Returns
        -------
        dict
            Extracted ROI data including label, x, and y values.
        """

        inst_view = self.inst_view

        roi_view = {}

        d = inst_view["d"]
        d_min = inst_view["d_min"]
        d_max = inst_view["d_max"]
        gamma = inst_view["gamma"]
        nu = inst_view["nu"]
        ind = inst_view["ind"]

        if horz_roi == 0:
            horz_roi = (gamma.max() - gamma.min()) / 2

        if vert_roi == 0:
            vert_roi = (nu.max() - nu.min()) / 2

        if horz < gamma.min() or val > gamma.max():
            val = (gamma.max() + gamma.min()) / 2

        if vert < nu.min() or val > nu.max():
            val = (nu.max() + nu.min()) / 2

        R = self.Rs[ind]
        gamma = self.gamma
        nu = self.nu

        if type(self.scan) is float:
            x = np.rad2deg(np.arccos([0.5 * (np.trace(r) - 1) for r in R]))
            label = "angle"
        else:
            x = self.scan
            label = "d"

        mask = (
            (d > d_min)
            & (d < d_max)
            & (gamma[:, np.newaxis] > horz - horz_roi)
            & (gamma[:, np.newaxis] < horz + horz_roi)
            & (nu[:, np.newaxis] > vert - vert_roi)
            & (nu[:, np.newaxis] < vert + vert_roi)
        )

        rows, cols = np.nonzero(mask)

        vals = self.counts[ind].copy()

        uni_cols, inv_ind = np.unique(cols, return_inverse=True)

        x = x[uni_cols]
        y = np.bincount(inv_ind, weights=vals[mask])

        if len(x) > 1:
            if val < x.min() or val > x.max():
                val = (x.max() + x.min()) / 2
        else:
            val = 0

        roi_view["horz"] = horz
        roi_view["vert"] = vert
        roi_view["horz_roi"] = horz_roi
        roi_view["vert_roi"] = vert_roi
        roi_view["val"] = val
        roi_view["label"] = label
        roi_view["x"] = x
        roi_view["y"] = y
        roi_view["label"] = label

        self.roi_view = roi_view

    def is_sliced(self):
        """
        Check if the data has been sliced.

        Returns
        -------
        sliced : bool
            True if data is sliced, False otherwise.
        """

        return mtd.doesExist("slice")

    def get_Q(self, d):
        """
        Convert to Q momentum transfer magnitude.

        Parameters
        ----------
        d : float
            Interplanar d-spacing.

        Returns
        -------
        Q : float
            Momentum transfer.
        """
        return 2 * np.pi / d

    def get_d(self, Q):
        """
        Convert to d-spacing.

        Parameters
        ----------
        Q : float
            Momentum transfer.

        Returns
        -------
        d : float
            Interplanar d-spacing.

        """
        return 2 * np.pi / Q

    def get_slice_z_extent(self, U, V, W, normal):
        """
        Get the minimum and maximum extent along the slice normal direction.

        Parameters
        ----------
        U, V, W : list
            Normalized direction cosines defining the slice basis vectors.
        normal : list
            Normal vector for the slice.

        Returns
        -------
        z_min : float or None
            Minimum extent along the normal direction, or None if no UB/Q data.
        z_max : float or None
            Maximum extent along the normal direction, or None if no UB/Q data.
        """
        if not (self.has_UB() and self.has_Q()):
            return None, None
        UB = self.get_UB()
        Wt = np.column_stack([U, V, W])
        Bp = np.dot(UB, Wt)
        bp_inv = np.linalg.inv(2 * np.pi * Bp)
        Qx_min, Qy_min, Qz_min = self.min_lim
        Qx_max, Qy_max, Qz_max = self.max_lim
        corners = np.array(
            [
                [Qx_min, Qy_min, Qz_min],
                [Qx_max, Qy_min, Qz_min],
                [Qx_min, Qy_max, Qz_min],
                [Qx_max, Qy_max, Qz_min],
                [Qx_min, Qy_min, Qz_max],
                [Qx_max, Qy_min, Qz_max],
                [Qx_min, Qy_max, Qz_max],
                [Qx_max, Qy_max, Qz_max],
            ]
        )
        trans_corners = np.einsum("ij,kj->ki", bp_inv, corners)
        i = np.array(normal).tolist().index(1)
        return float(np.min(trans_corners[:, i])), float(
            np.max(trans_corners[:, i])
        )

    def get_slice_info(self, U, V, W, normal, value, thickness, width):
        """
        Get the slicing information for a given normal and value.

        Parameters
        ----------
        U, V, W : list
            Normalized direction cosines for the slice.
        normal : list
            Normal vector for the slice.
        value : float
            Value at which to slice.
        thickness : float
            Thickness of the slice.
        width : float
            Width of the output histogram bins.

        Returns
        -------
        dict
            Dictionary with slice information including extents, bins, and transform.
        """

        UB = self.get_UB()

        if self.has_UB() and self.has_Q():
            Wt = np.column_stack([U, V, W])

            slice_dict = {}

            Bp = np.dot(UB, Wt)

            bp_inv = np.linalg.inv(2 * np.pi * Bp)

            Qx_min, Qy_min, Qz_min = self.min_lim
            Qx_max, Qy_max, Qz_max = self.max_lim

            corners = np.array(
                [
                    [Qx_min, Qy_min, Qz_min],
                    [Qx_max, Qy_min, Qz_min],
                    [Qx_min, Qy_max, Qz_min],
                    [Qx_max, Qy_max, Qz_min],
                    [Qx_min, Qy_min, Qz_max],
                    [Qx_max, Qy_min, Qz_max],
                    [Qx_min, Qy_max, Qz_max],
                    [Qx_max, Qy_max, Qz_max],
                ]
            )

            trans_corners = np.einsum("ij,kj->ki", bp_inv, corners)

            min_values = np.ceil(np.min(trans_corners, axis=0))
            max_values = np.floor(np.max(trans_corners, axis=0))

            bin_sizes = np.round((max_values - min_values) / width).astype(int)

            min_values = min_values.tolist()
            max_values = max_values.tolist()

            bin_sizes = bin_sizes.tolist()

            extents = []
            bins = []

            integrate = [value - thickness, value + thickness]

            for ind, i in enumerate(normal):
                if i == 0:
                    extents += [min_values[ind], max_values[ind]]
                    bins += [1 + bin_sizes[ind]]
                else:
                    extents += integrate
                    bins += [1]

            self.copy_UB_to_peaks()

            ConvertQtoHKLMDHisto(
                InputWorkspace=self.Q,
                PeaksWorkspace=self.table,
                UProj=U,
                VProj=V,
                WProj=W,
                Extents=extents,
                Bins=bins,
                OutputWorkspace="slice",
            )

            CompactMD(InputWorkspace="slice", OutputWorkspace="slice")

            i = np.array(normal).tolist().index(1)

            form = "{} = ({:.2f},{:.2f})"

            title = form.format(mtd["slice"].getDimension(i).name, *integrate)

            dims = mtd["slice"].getNonIntegratedDimensions()

            x, y = [
                np.linspace(
                    dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
                )
                for dim in dims
            ]

            labels = [
                "{} ({})".format(dim.name, dim.getUnits()) for dim in dims
            ]

            slice_dict["x"] = x
            slice_dict["y"] = y
            slice_dict["labels"] = labels

            signal = mtd["slice"].getSignalArray().T.copy().squeeze()

            # signal[signal <= 0] = np.nan
            # signal[np.isinf(signal)] = np.nan

            slice_dict["signal"] = signal

            Q, R = scipy.linalg.qr(Bp)

            ind = np.array(normal) != 1
            i = ind.tolist().index(False)

            W = np.column_stack([U, V, W])

            slice_dict["z"] = value
            slice_dict["z_min"] = float(min_values[i])
            slice_dict["z_max"] = float(max_values[i])
            slice_dict["W"] = np.column_stack([W[:, ind], W[:, i]])

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

    def validate_projection(self, proj):
        """
        Validate and unpack a 3x3 projection matrix into its row vectors.

        Parameters
        ----------
        proj : array_like
            Projection matrix, given as 9 values reshaped into 3x3, whose
            rows define the U, V, W basis vectors of the slice.

        Returns
        -------
        U, V, W : ndarray
            Rows of the projection matrix.
        invalid : bool
            True if the projection matrix is singular (determinant close
            to zero), False otherwise.
        """
        proj = np.array(proj).reshape(3, 3)
        invalid = np.isclose(np.linalg.det(proj), 0)
        return *proj, invalid

    def calculate_clim(self, data, method="normal"):
        """
        Calculate the color limits for the given data using the specified method.

        Parameters
        ----------
        data : np.ndarray
            Input data for which to calculate color limits.
        method : str, optional
            Method for calculating limits ('normal', 'boxplot', or 'min/max'). Default is 'normal'.

        Returns
        -------
        tuple
            Lower and upper color limits.
        """

        trans = data.copy()
        trans[~np.isfinite(trans)] = np.nan
        trans[np.isclose(trans, 0)] = np.nan

        vmin, vmax = np.nanmin(trans), np.nanmax(trans)

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

        if np.isclose(cmin, cmax) or cmax < cmin:
            cmin, cmax = vmin, vmax

        clim = [cmin if cmin > vmin else vmin, cmax if cmax < vmax else vmax]

        trans[trans < clim[0]] = clim[0]
        trans[trans > clim[1]] = clim[1]

        return trans

    def get_has_Q_vol(self):
        """
        Check if the Q volume data exists.

        Returns
        -------
        has_Q_vol : bool
            True if Q volume data exists, False otherwise.
        """

        return mtd.doesExist("Q3D")

    def get_Q_info(self):
        """
        Extract Q-space information from the model.

        Returns
        -------
        Q_info : dict
            Dictionary containing Q-space signal, min/max limits, spacing, and optionally x, y, z coordinates.
        """

        Q_dict = {}

        if self.get_has_Q_vol():
            Q_dict["signal"] = self.signal
            # Q_dict["opacity"] = self.opacity

            Q_dict["min_lim"] = self.min_lim
            Q_dict["max_lim"] = self.max_lim
            Q_dict["spacing"] = self.spacing

            # Q_dict["x"] = self.x
            # Q_dict["y"] = self.y
            # Q_dict["z"] = self.z

        if self.has_peaks():
            self.sort_peaks_by_hkl(self.table)

            self.sort_peaks_by_d(self.table)

            Qs, Is, inds, pk_nos, Ts, rows = [], [], [], [], [], []

            for j, peak in enumerate(mtd[self.table]):
                T = np.zeros((4, 4))

                I = peak.getIntensity()

                ind = (peak.getHKL().norm2() > 0) * 1.0

                shape = eval(peak.getPeakShape().toJSON())

                pk_no = j

                Q = peak.getQSampleFrame()

                if any(["radius" in key for key in shape.keys()]):
                    if shape.get("radius0") is not None:
                        r, v = [], []
                        for i in range(3):
                            r.append(shape["radius{}".format(i)])
                            v.append(shape["direction{}".format(i)].split(" "))
                        r = np.array(r)
                        v = np.array(v).T.astype(float)

                    else:
                        r = np.array([shape["radius"]] * 3)
                        v = np.eye(3)

                    P = np.dot(v, np.dot(np.diag(r), v.T))

                else:
                    P = np.eye(3) * 0.2

                T[:3, -1] = Q
                T[:3, :3] = P
                T[-1, -1] = 1

                Qs.append(Q)
                Is.append(I)
                inds.append(ind)
                pk_nos.append(pk_no)
                Ts.append(T)
                rows.append(j)

            Q_dict["coordinates"] = Qs
            Q_dict["intensities"] = Is
            Q_dict["indexings"] = inds
            Q_dict["numbers"] = pk_nos
            Q_dict["transforms"] = Ts
            Q_dict["rows"] = rows

        return Q_dict if len(Q_dict.keys()) > 0 else None

    def get_lattice_constants(self):
        """
        Get the lattice constants from the oriented lattice.

        Returns
        -------
        params : list
            List of lattice constants [a, b, c, alpha, beta, gamma].
        """

        if self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            params = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

            return np.array(params).round(8).tolist()

    def get_lattice_constant_errors(self):
        """
        Get the errors in the lattice constants from the oriented lattice.

        Returns
        -------
        errors : list
            List of errors in lattice constants.
        """

        if self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            params = (
                ol.errora(),
                ol.errorb(),
                ol.errorc(),
                ol.erroralpha(),
                ol.errorbeta(),
                ol.errorgamma(),
            )

            params = np.array(params)
            params[~np.isfinite(params)] = 0.0

            return params.round(8).tolist()

    def simplify_vector(self, vec):
        """
        Simplify a vector to its primitive integer form.

        Parameters
        ----------
        vec : np.ndarray
            Input vector.

        Returns
        -------
        int_vec : np.ndarray
            Simplified integer vector.
        """

        vec = vec / np.linalg.norm(vec)

        vec *= 10

        vec = np.round(vec).astype(int)

        return vec // np.gcd.reduce(vec)

    def get_sample_directions(self):
        """
        Get the sample directions from the UB matrix.

        Returns
        -------
        directions : list
            List of simplified sample direction vectors.
        """

        if self.has_UB():
            UB = mtd[self.cell].sample().getOrientedLattice().getUB()

            vecs = np.linalg.inv(UB).T

            return [self.simplify_vector(vec) for vec in vecs]

    def copy_UB_from_peaks(self):
        """
        Copy the UB matrix from the peaks workspace to the cell workspace.
        """

        CopySample(
            InputWorkspace=self.table,
            OutputWorkspace=self.cell,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

    def copy_UB_to_peaks(self):
        """
        Copy the UB matrix from the cell workspace to the peaks workspace.
        """

        CopySample(
            InputWorkspace=self.cell,
            OutputWorkspace=self.table,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

    def copy_UB_from_Q(self):
        """
        Copy the UB matrix from the Q workspace to the cell workspace.
        """

        CopySample(
            InputWorkspace=self.Q,
            OutputWorkspace=self.cell,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

    def copy_UB_to_Q(self):
        """
        Copy the UB matrix from the cell workspace to the Q workspace.
        """

        CopySample(
            InputWorkspace=self.cell,
            OutputWorkspace=self.Q,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

    def save_UB(self, filename):
        """
        Save UB to file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """

        SaveIsawUB(InputWorkspace=self.cell, Filename=filename)

    def load_UB(self, filename):
        """
        Load UB from file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """

        LoadIsawUB(InputWorkspace=self.cell, Filename=filename)

        self.copy_UB_to_peaks()

    def determine_UB_with_niggli_cell(self, min_d, max_d, tol=0.1):
        """
        Determine UB with primitive lattice using min/max lattice constant.

        Parameters
        ----------
        min_d : float
            Minimum lattice parameter in ansgroms.
        max_d : float
            Maximum lattice parameter in angstroms.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingFFT(
            PeaksWorkspace=self.table, MinD=min_d, MaxD=max_d, Tolerance=tol
        )

        self.copy_UB_from_peaks()

        self.update_UB()

        CloneWorkspace(
            InputWorkspace=self.table, OutputWorkspace=self.primitive_cell
        )

    def determine_UB_with_lattice_parameters(
        self, a, b, c, alpha, beta, gamma, tol=0.1
    ):
        """
        Determine UB with prior known lattice parameters.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingLatticeParameters(
            PeaksWorkspace=self.table,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            Tolerance=tol,
            NumInitial=100,
            FixParameters=False,
            Iterations=3,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def refine_UB_without_constraints(self, tol=0.1, sat_tol=None):
        """
        Refine UB with unconstrained lattice parameters.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        FindUBUsingIndexedPeaks(
            PeaksWorkspace=self.table,
            Tolerance=tol,
            ToleranceForSatellite=tol_for_sat,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def refine_UB_with_constraints(self, cell, tol=0.1):
        """
        Refine UB with constraints corresponding to lattice system.

        +----------------+---------------+----------------------+
        | Lattice system | Lengths       | Angles               |
        +================+===============+======================+
        | Cubic          | :math:`a=b=c` | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Hexagonal      | :math:`a=b`   | :math:`α=β=90,γ=120` |
        +----------------+---------------+----------------------+
        | Rhombohedral   | :math:`a=b=c` | :math:`α=β=γ`        |
        +----------------+---------------+----------------------+
        | Tetragonal     | :math:`a=b`   | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Orthorhombic   | None          | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Monoclinic     | None          | :math:`α=γ=90`       |
        +----------------+---------------+----------------------+
        | Triclinic      | None          | None                 |
        +----------------+---------------+----------------------+

        Parameters
        ----------
        cell : float
            Lattice system.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        self.copy_UB_to_peaks()

        OptimizeLatticeForCellType(
            PeaksWorkspace=self.table, CellType=cell, Apply=True, Tolerance=tol
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def refine_U_only(self, a, b, c, alpha, beta, gamma):
        """
        Refine the U orientation only.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        self.copy_UB_to_peaks()

        CalculateUMatrix(
            PeaksWorkspace=self.table,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def select_cell(self, number, tol=0.1):
        """
        Transform to conventional cell using form number.

        Parameters
        ----------
        number : int
            Form number.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        CopySample(
            InputWorkspace=self.primitive_cell,
            OutputWorkspace=self.table,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

        SelectCellWithForm(
            PeaksWorkspace=self.table,
            FormNumber=number,
            Apply=True,
            Tolerance=tol,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def possible_conventional_cells(self, max_error=0.2, permutations=True):
        """
        List possible conventional cells.

        Parameters
        ----------
        max_error : float, optional
            Max scalar error to report form numbers. The default is 0.2.
        permutations : bool, optional
            Allow permutations of the lattice. The default is True.

        Returns
        -------
        vals : list
            List of form results.

        """

        CopySample(
            InputWorkspace=self.primitive_cell,
            OutputWorkspace=self.table,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

        result = ShowPossibleCells(
            PeaksWorkspace=self.table,
            MaxScalarError=max_error,
            AllowPermutations=permutations,
            BestOnly=False,
        )

        vals = [json.loads(cell) for cell in result.Cells]

        cells = []
        for i, val in enumerate(vals):
            form = val["FormNumber"]
            error = val["Error"]
            cell = val["CellType"]
            centering = val["Centering"]
            bravais = cell, centering
            a = val["a"]
            b = val["b"]
            c = val["c"]
            alpha = val["alpha"]
            beta = val["beta"]
            gamma = val["gamma"]
            vol = val["volume"]
            params = a, b, c, alpha, beta, gamma, vol
            cell = form, error, bravais, params
            cells.append(cell)

        return cells

    def transform_lattice(self, transform, tol=0.1):
        """
        Apply a cell transformation to the lattice.

        Parameters
        ----------
        transform : 3x3 array-like
            Transform to apply to hkl values.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        hkl_trans = ",".join(9 * ["{}"]).format(*transform)

        TransformHKL(
            PeaksWorkspace=self.table,
            Tolerance=tol,
            HKLTransform=hkl_trans,
            FindError=False,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def generate_lattice_transforms(self, cell):
        """
        Obtain possible transforms compatabile with a unit cell lattice.

        Parameters
        ----------
        cell : str
            Latttice system.

        Returns
        -------
        transforms : dict
            Transform dictionary with symmetry operation as key.

        """

        symbol = lattice_group[cell]

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transform = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            if np.linalg.det(T) > 0:
                name = "{}: ".format(symop.getOrder()) + symop.getIdentifier()
                transform[name] = T

        return {key: transform[key] for key in sorted(transform.keys())}

    def set_manual_UB(self, constants, directions):
        """
        Set the unit cell parameters and directions manually.

        Parameters
        ----------
        constants : list
            List of unit cell parameters [a, b, c, alpha, beta, gamma].
        directions : list
            List of direction indices [u1, u2, u3, v1, v2, v3].
        """

        a, b, c, alpha, beta, gamma = constants
        u1, u2, u3, v1, v2, v3 = directions

        for ws in [self.cell, self.primitive_cell, self.table, self.Q]:
            if ws is not None and mtd.doesExist(ws):
                SetUB(
                    Workspace=ws,
                    a=a,
                    b=b,
                    c=c,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    u=[u1, u2, u3],
                    v=[v1, v2, v3],
                )

        self.update_UB()

    def find_UB_from_scattering_plane(self, constants, directions):
        """
        Calculate the UB matrix from a scattering plane and one peak.

        Parameters
        ----------
        constants : list
            Lattice constants and angles [a, b, c, alpha, beta, gamma]
            in angstroms and degrees.
        directions : list
            Two non-parallel vectors [u1, u2, u3, v1, v2, v3] defining the
            scattering plane.
        """

        if not self.has_peaks() or mtd[self.table].getNumberPeaks() == 0:
            print("No peaks available for scattering-plane UB.")
            return

        a, b, c, alpha, beta, gamma = constants
        u1, u2, u3, v1, v2, v3 = directions

        u = np.array([u1, u2, u3], dtype=float)
        v = np.array([v1, v2, v3], dtype=float)

        if np.isclose(np.linalg.norm(u), 0) or np.isclose(
            np.linalg.norm(v), 0
        ):
            print("Scattering plane vectors u and v must be non-zero.")
            return

        if np.isclose(np.linalg.norm(np.cross(u, v)), 0):
            print("Scattering plane vectors u and v must not be parallel.")
            return

        FilterPeaks(
            InputWorkspace=self.table,
            OutputWorkspace="tmp",
            FilterVariable="h^2+k^2+l^2",
            FilterValue=0,
            Operator=">",
            Criterion="!=",
            BankName="None",
        )

        if mtd["tmp"].getNumberPeaks() == 0:
            print(
                "No valid peaks available after filtering for scattering-plane UB."
            )
            mtd.remove("tmp")
            return

        FindUBFromScatteringPlane(
            PeaksWorkspace="tmp",
            Vector1=u.tolist(),
            Vector2=v.tolist(),
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        CopySample(
            InputWorkspace="tmp",
            OutputWorkspace=self.table,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

        mtd.remove("tmp")

        self.copy_UB_from_peaks()
        self.update_UB()

    def index_peaks(
        self,
        tol=0.1,
        sat_tol=None,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
        round_hkl=True,
    ):
        """
        Index the peaks and calculate the lattice parameter uncertainties.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.
        round_hkl : bool, optional
            Round integers to integer. The default is True.

        Returns
        -------
        indexing : list
            Result of indexing including number indexed and errors.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        save = True if max_order > 0 else False

        indexing = IndexPeaks(
            PeaksWorkspace=self.table,
            Tolerance=tol,
            ToleranceForSatellite=tol_for_sat,
            RoundHKLs=round_hkl,
            CommonUBForAll=True,
            ModVector1=mod_vec_1,
            ModVector2=mod_vec_2,
            ModVector3=mod_vec_3,
            MaxOrder=max_order,
            CrossTerms=cross_terms,
            SaveModulationInfo=save,
        )

        return indexing

    def calculate_hkl(self):
        """
        Calculate hkl values without rounding.

        """

        CalculatePeaksHKL(PeaksWorkspace=self.table, OverWrite=True)

    def find_peaks(self, min_dist, density=1000, max_peaks=50, edge_pixels=0):
        """
        Harvest strong peak locations from Q-sample into a peaks table.

        Parameters
        ----------
        min_dist : float
            Minimum distance enforcing lower limit of peak spacing.
        density : int, optional
            Threshold density. The default is 1000.
        max_peaks : int, optional
            Maximum number of peaks to find. The default is 50.
        edge_pixels: int, optional
            Nnumber of edge pixels to exclude. The default is 0.

        """

        FindPeaksMD(
            InputWorkspace=self.Q,
            PeakDistanceThreshold=min_dist,
            MaxPeaks=max_peaks,
            PeakFindingStrategy="VolumeNormalization",
            DensityThresholdFactor=density,
            EdgePixels=edge_pixels,
            OutputWorkspace=self.table,
        )

        self.integrate_peaks(min_dist, 1, 1, method="sphere", centroid=False)

        self.clear_intensity()

        self.copy_UB_to_peaks()

    def centroid_peaks(self, peak_radius):
        """
        Re-center peak locations using centroid within given radius

        Parameters
        ----------
        peak_radius : float
            Integration region radius.

        """

        CentroidPeaksMD(
            InputWorkspace=self.Q,
            PeakRadius=peak_radius,
            PeaksWorkspace=self.table,
            OutputWorkspace=self.table,
        )

    def integrate_peaks(
        self,
        peak_radius,
        background_inner_fact=1,
        background_outer_fact=1.5,
        method="sphere",
        centroid=True,
    ):
        """
        Integrate peaks using spherical or ellipsoidal regions.
        Ellipsoid integration adapts itself to the peak distribution.

        Parameters
        ----------
        peak_radius : float
            Integration region radius.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1.
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.
        method : str, optional
            Integration method. The default is 'sphere'.
        centroid : str, optional
            Shift peak position to centroid. The default is True.

        """

        background_inner_radius = peak_radius * background_inner_fact
        background_outer_radius = peak_radius * background_outer_fact

        if method == "sphere" and centroid:
            self.centroid_peaks(peak_radius)

        IntegratePeaksMD(
            InputWorkspace=self.Q,
            PeaksWorkspace=self.table,
            PeakRadius=peak_radius,
            BackgroundInnerRadius=background_inner_radius,
            BackgroundOuterRadius=background_outer_radius,
            Ellipsoid=True if method == "ellipsoid" else False,
            FixQAxis=False,
            FixMajorAxisLength=False,
            UseCentroid=True,
            MaxIterations=1,
            ReplaceIntensity=True,
            IntegrateIfOnEdge=True,
            AdaptiveQBackground=False,
            MaskEdgeTubes=False,
            OutputWorkspace=self.table,
        )

    def clear_intensity(self):
        """
        Clear the intensity values of all peaks in the peaks table.
        """

        for peak in mtd[self.table]:
            peak.setIntensity(0)
            peak.setSigmaIntensity(0)

    def get_max_d_spacing(self, ws):
        """
        Obtain the maximum d-spacing from the oriented lattice.

        Parameters
        ----------
        ws : str
            Workspace with UB defined on oriented lattice.

        Returns
        -------
        d_max : float
            Maximum d-spacing.

        """

        if HasUB(Workspace=ws):
            if hasattr(mtd[ws], "sample"):
                ol = mtd[ws].sample().getOrientedLattice()
            else:
                for i in range(mtd[ws].getNumExperimentInfo()):
                    sample = mtd[ws].getExperimentInfo(i).sample()
                    if sample.hasOrientedLattice():
                        ol = sample.getOrientedLattice()
                        SetUB(Workspace=ws, UB=ol.getUB())
                ol = mtd[ws].getExperimentInfo(i).sample().getOrientedLattice()

            return 1 / min([ol.astar(), ol.bstar(), ol.cstar()])

    def predict_peaks(
        self, centering, d_min, lamda_min, lamda_max, edge_pixels=0
    ):
        """
        Predict peak Q-sample locations with UB and lattice centering.

        +--------+-----------------------+
        | Symbol | Reflection condition  |
        +========+=======================+
        | P      | None                  |
        +--------+-----------------------+
        | I      | :math:`h+k+l=2n`      |
        +--------+-----------------------+
        | F      | :math:`h,k,l` unmixed |
        +--------+-----------------------+
        | R      | None                  |
        +--------+-----------------------+
        | R(obv) | :math:`-h+k+l=3n`     |
        +--------+-----------------------+
        | R(rev) | :math:`h-k+l=3n`      |
        +--------+-----------------------+
        | A      | :math:`k+l=2n`        |
        +--------+-----------------------+
        | B      | :math:`l+h=2n`        |
        +--------+-----------------------+
        | C      | :math:`h+k=2n`        |
        +--------+-----------------------+

        Parameters
        ----------
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.
        edge_pixels: int, optional
            Nnumber of edge pixels to exclude. The default is 0.

        """

        self.copy_UB_to_peaks()

        has_q = self.Q is not None and mtd.doesExist(self.Q)

        if has_q:
            self.copy_UB_to_Q()
            input_ws = self.Q
        else:
            input_ws = self.cell

        d_max = self.get_max_d_spacing(input_ws)

        PredictPeaks(
            InputWorkspace=input_ws,
            WavelengthMin=lamda_min,
            WavelengthMax=lamda_max,
            MinDSpacing=d_min,
            MaxDSpacing=d_max * 1.2,
            ReflectionCondition=centering_reflection[centering],
            EdgePixels=edge_pixels if has_q else 0,
            OutputWorkspace=self.table,
        )

        if has_q:
            self.integrate_peaks(0.1, 1, 1, method="sphere", centroid=False)

        self.clear_intensity()

    def predict_modulated_peaks(
        self,
        d_min,
        lamda_min,
        lamda_max,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
    ):
        """
        Predict the modulated peak positions based on main peaks.

        Parameters
        ----------
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        """

        self.copy_UB_to_peaks()

        d_max = self.get_max_d_spacing(self.table)

        sat_peaks = self.table + "_sat"

        PredictSatellitePeaks(
            Peaks=self.table,
            SatellitePeaks=sat_peaks,
            ModVector1=mod_vec_1,
            ModVector2=mod_vec_2,
            ModVector3=mod_vec_3,
            MaxOrder=max_order,
            CrossTerms=cross_terms,
            IncludeIntegerHKL=False,
            IncludeAllPeaksInRange=True,
            WavelengthMin=lamda_min,
            WavelengthMax=lamda_max,
            MinDSpacing=d_min,
            MaxDSpacing=d_max * 1.2,
        )

        CombinePeaksWorkspaces(
            LHSWorkspace=self.table,
            RHSWorkspace=sat_peaks,
            OutputWorkspace=self.table,
        )

        DeleteWorkspace(Workspace=sat_peaks)

    def set_goniometer(self, peaks, R):
        """
        Set the goniometer matrix for a peaks workspace.

        Parameters
        ----------
        peaks : str
            Name of peaks workspace.
        R : 3x3 array-like
            Goniometer rotation matrix.

        """

        SetGoniometer(Workspace=peaks, GoniometerMatrix=R.flatten().tolist())

    def predict_satellite_peaks(
        self,
        d_min,
        lamda_min,
        lamda_max,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
    ):
        """
        Locate satellite peaks from goniometer angles.

        Parameters
        ----------
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min : float
            Minimum wavelength.
        lamda_max : float
            Maximum wavelength.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        """

        Rs = self.get_all_goniometer_matrices(self.Q)

        for R in Rs:
            self.set_goniometer(self.table, R)

            self.predict_modulated_peaks(
                d_min,
                lamda_min,
                lamda_max,
                mod_vec_1,
                mod_vec_2,
                mod_vec_3,
                max_order,
                cross_terms,
            )

            self.remove_duplicate_peaks(self.table)

    def sort_peaks_by_hkl(self, peaks):
        """
        Sort peaks table by descending hkl values.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        columns = ["l", "k", "h"]

        for col in columns:
            SortPeaksWorkspace(
                InputWorkspace=peaks,
                ColumnNameToSortBy=col,
                SortAscending=False,
                OutputWorkspace=peaks,
            )

    def sort_peaks_by_d(self, peaks):
        """
        Sort peaks table by descending d-spacing.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        SortPeaksWorkspace(
            InputWorkspace=peaks,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace=peaks,
        )

    def remove_duplicate_peaks(self, peaks):
        """
        Omit duplicate peaks from different based on indexing.
        Table will be sorted.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        self.sort_peaks_by_hkl(peaks)

        for no in range(mtd[peaks].getNumberPeaks() - 1, 0, -1):
            hkl_1 = mtd[peaks].getPeak(no).getHKL()
            hkl_2 = mtd[peaks].getPeak(no - 1).getHKL()
            if (hkl_1 - hkl_2).norm2() == 0:
                DeleteTableRows(TableWorkspace=peaks, Rows=no)

    def get_all_goniometer_matrices(self, ws):
        """
        Extract all goniometer matrices.

        Parameters
        ----------
        ws : str
            Name of workspace with goniometer indexing.

        Returns
        -------
        Rs: list
            Goniometer matrices.

        """

        Rs = []

        for ei in range(mtd[ws].getNumExperimentInfo()):
            run = mtd[ws].getExperimentInfo(ei).run()

            n_gon = run.getNumGoniometers()

            Rs += [run.getGoniometer(i).getR() for i in range(n_gon)]

        return np.array(Rs)

    def renumber_runs_by_index(self, ws, peaks):
        """
        Re-label the runs by index based on goniometer setting.

        Parameters
        ----------
        ws : str
            Name of workspace with goniometer indexing.
        peaks : str
            Name of peaks table.

        """

        Rs = self.get_all_goniometer_matrices(ws)

        for no in range(mtd[peaks].getNumberPeaks()):
            peak = mtd[peaks].getPeak(no)

            R = peak.getGoniometerMatrix()

            ind = np.isclose(Rs, R).all(axis=(1, 2))
            i = -1 if not np.any(ind) else ind.tolist().index(True)

            peak.setRunNumber(i + 1)

    def load_Q(self, filename):
        """
        Load Q file.

        Parameters
        ----------
        filename : str
            Name of Q file with extension .nxs.

        Returns
        -------
        d_min : float
            Minimum d-spacing derived from Q_max.
        wavelength : list or None
            [lambda_min, lambda_max] for DGS data, None otherwise.

        """

        self.Q = "md"

        LoadMD(Filename=filename, OutputWorkspace=self.Q)

        ws = mtd[self.Q]

        wavelength = None

        if ws.getNumDims() == 4:
            dim_names = [ws.getDimension(i).getName() for i in range(4)]

            q_names = ["Q_sample_x", "Q_sample_y", "Q_sample_z"]
            if not all(dim_names[i] == q_names[i] for i in range(3)):
                raise ValueError(
                    "Expected Q_sample_x/y/z as first three dimensions, "
                    "got: {}".format(dim_names[:3])
                )

            aligned_dims = []
            for i in range(3):
                d = ws.getDimension(i)
                aligned_dims.append(
                    "{},{},{},{}".format(
                        d.getName(),
                        d.getMinimum(),
                        d.getMaximum(),
                        d.getNBins(),
                    )
                )

            SliceMD(
                InputWorkspace=self.Q,
                AlignedDim0=aligned_dims[0],
                AlignedDim1=aligned_dims[1],
                AlignedDim2=aligned_dims[2],
                OutputWorkspace=self.Q,
            )

            try:
                run = ws.getExperimentInfo(0).run()
                Ei = float(run.getProperty("Ei").value)  # meV
                lamda = 9.0445 / np.sqrt(Ei)
                wavelength = [round(lamda - 0.1, 4), round(lamda + 0.1, 4)]
                self.wavelength = wavelength
            except Exception:
                pass

        ws = mtd[self.Q]
        Q_max = max(
            max(
                abs(ws.getDimension(i).getMinimum()),
                abs(ws.getDimension(i).getMaximum()),
            )
            for i in range(3)
        )

        self.make_Q(Q_max)

        CopySample(
            InputWorkspace=self.Q,
            OutputWorkspace=self.cell,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
            CopyLattice=True,
        )

        d_min = round(2 * np.pi / Q_max, 4)

        return d_min, wavelength

    def get_instrument_from_Q(self):
        """
        Return the display-name key (e.g. 'TOPAZ') for the instrument embedded
        in the loaded Q workspace, or None if unrecognised.

        Returns
        -------
        key : str or None
            Instrument identifier key matching an entry in `beamlines`, or
            None if the instrument could not be determined.
        """
        try:
            ws = mtd[self.Q]
            inst_name = ws.getExperimentInfo(0).getInstrument().getName()
            for key, cfg in beamlines.items():
                if cfg["Name"].upper() == inst_name.upper():
                    return key
        except Exception:
            pass
        return None

    def save_Q(self, filename):
        """
        Save Q file.

        Parameters
        ----------
        filename : str
            Name of Q file with extension .nxs.

        """

        SaveMD(Filename=filename, InputWorkspace=self.Q)

    def load_peaks(self, filename):
        """
        Load peaks file.

        Parameters
        ----------
        filename : str
            Name of peaks file (can be .nxs or .integrate, .peaks, etc.).
        """

        if filename.endswith(".nxs"):
            LoadNexus(Filename=filename, OutputWorkspace=self.table)
        else:
            LoadIsawPeaks(Filename=filename, OutputWorkspace=self.table)

    def save_peaks(self, filename):
        """
        Save the current peaks table to a file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        """

        if filename.endswith(".nxs"):
            SaveNexus(Filename=filename, InputWorkspace=self.table)
        else:
            SaveIsawPeaks(Filename=filename, InputWorkspace=self.table)

    def delete_peaks(self, peaks):
        """
        Remove peaks.

        Parameters
        ----------
        peaks : str
            Name of peaks table to be added.

        """

        if mtd.doesExist(peaks):
            DeleteWorkspace(Workspace=peaks)

    def filter_peaks(self, name, operator, value):
        """
        Filter out peaks based on value and operator.

        Parameters
        ----------
        name : str
            Filter name.
        operator : float
            Filter operator.
        value : float
            The filter value.

        """

        FilterPeaks(
            InputWorkspace=self.table,
            OutputWorkspace=self.table,
            FilterVariable=variable[name],
            FilterValue=value,
            Operator=operator,
            Criterion="!=",
            BankName="None",
        )

    def get_d_min(self):
        """
        Get the minimum d-spacing among the current peaks, capped at 0.7.

        Returns
        -------
        d_min : float
            Smallest peak d-spacing found, or 0.7 angstroms if no peak has
            a smaller d-spacing (or no peaks are present).
        """
        d_min = 0.7
        if self.has_peaks():
            for peak in mtd[self.table]:
                d_spacing = peak.getDSpacing()
                if d_spacing < d_min:
                    d_min = d_spacing
        return d_min

    def avoid_aluminum_contamination(self, d_min, d_max, delta=0.1):
        """
        Flag peaks coincident with aluminum powder ring reflections.

        Arblaster, J. W. Selected Values of the Crystallographic
        Properties of Elements; ASM International, 2018

        Parameters
        ----------
        d_min, d_max : float
            Minimum and maximum d-spacing range over which to generate
            aluminum reflections.
        delta : float, optional
            Tolerance in momentum transfer used to match peaks against
            aluminum reflections. The default is 0.1.
        """

        aluminum = CrystalStructure(
            "4.05 4.05 4.05", "F m -3 m", "Al 0 0 0 1.0 0.005"
        )

        self.avoid_contamination(aluminum, d_min, d_max, delta)

    def avoid_copper_contamination(self, d_min, d_max, delta=0.1):
        """
        Flag peaks coincident with copper powder ring reflections.

        Arblaster, J. W. Selected Values of the Crystallographic
        Properties of Elements; ASM International, 2018

        Parameters
        ----------
        d_min, d_max : float
            Minimum and maximum d-spacing range over which to generate
            copper reflections.
        delta : float, optional
            Tolerance in momentum transfer used to match peaks against
            copper reflections. The default is 0.1.
        """

        copper = CrystalStructure(
            "3.61 3.61 3.61", "F m -3 m", "Cu 0 0 0 1.0 0.005"
        )

        self.avoid_contamination(copper, d_min, d_max, delta)

    def avoid_iron_contamination(self, d_min, d_max, delta=0.1):
        """
        Flag peaks coincident with iron powder ring reflections.

        Arblaster, J. W. Selected Values of the Crystallographic
        Properties of Elements; ASM International, 2018

        Parameters
        ----------
        d_min, d_max : float
            Minimum and maximum d-spacing range over which to generate
            iron reflections.
        delta : float, optional
            Tolerance in momentum transfer used to match peaks against
            iron reflections. The default is 0.1.
        """

        aluminum = CrystalStructure(
            "2.87 2.87 2.87", "I m -3 m", "Fe 0 0 0 1.0 0.005"
        )

        self.avoid_contamination(aluminum, d_min, d_max, delta)

    def avoid_contamination(self, sample, d_min, d_max, delta=0.1):
        """
        Remove peaks in the peaks table that coincide with powder rings.

        Peaks whose momentum transfer is within `delta` of a reflection
        generated from `sample`, or whose d-spacing exceeds `d_max`, are
        flagged and removed from the peaks table.

        Parameters
        ----------
        sample : CrystalStructure
            Crystal structure of the contaminant phase used to generate
            candidate powder-ring reflections.
        d_min, d_max : float
            Minimum and maximum d-spacing range over which to generate
            reflections.
        delta : float, optional
            Tolerance in momentum transfer used to match peaks against
            the generated reflections. The default is 0.1.
        """

        generator = ReflectionGenerator(sample)

        hkls = generator.getUniqueHKLsUsingFilter(
            d_min, d_max, ReflectionConditionFilter.StructureFactor
        )

        ds = list(generator.getDValues(hkls))

        if self.has_peaks():
            for peak in mtd[self.table]:
                d_spacing = peak.getDSpacing()
                Q_mod = 2 * np.pi / d_spacing
                for d in ds:
                    Q = 2 * np.pi / d
                    if (Q - delta < Q_mod < Q + delta) or d_spacing > d_max:
                        peak.setRunNumber(-1)

            FilterPeaks(
                InputWorkspace=self.table,
                OutputWorkspace=self.table,
                FilterVariable="RunNumber",
                FilterValue="-1",
                Operator="!=",
                Criterion="!=",
                BankName="None",
            )

    def get_modulation_info(self):
        """
        Get the modulation vectors stored on the oriented lattice.

        Returns
        -------
        mod_vecs : list or None
            List of the three modulation vectors (V3D), or None if there
            are no peaks or no UB matrix defined.
        """
        if self.has_peaks() and self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            return [ol.getModVec(i) for i in range(3)]

    def get_peak_info(self):
        """
        Extract detailed information for each peak in the current table.

        Returns
        -------
        list of dict
            List of dictionaries with peak properties (hkl, d-spacing, intensity, etc.).
        """

        if self.has_peaks():
            banks = mtd[self.table].column("BankName")

            peak_info = []
            for i, peak in enumerate(mtd[self.table]):
                two_theta = peak.getScattering()
                az_phi = peak.getAzimuthal()
                kf_x = np.sin(two_theta) * np.cos(az_phi)
                kf_y = np.sin(two_theta) * np.sin(az_phi)
                kf_z = np.cos(two_theta)
                nu = np.arcsin(kf_y)
                gamma = np.arctan2(kf_x, kf_z)
                R = peak.getGoniometerMatrix()
                g = Goniometer()
                g.setR(R)
                angs = g.getEulerAngles(self.conv)
                peak_data = {
                    "hkl": list(peak.getHKL()),
                    "d_spacing": peak.getDSpacing(),
                    "wavelength": peak.getWavelength(),
                    "intensity": peak.getIntensity(),
                    "signal_to_noise": peak.getIntensityOverSigma(),
                    "sigma": peak.getSigmaIntensity(),
                    "int_hkl": list(peak.getIntHKL()),
                    "int_mnp": list(peak.getIntMNP()),
                    "run_number": peak.getRunNumber(),
                    "two_theta": np.rad2deg(two_theta),
                    "gamma": np.rad2deg(gamma),
                    "nu": np.rad2deg(nu),
                    "angles": list(angs),
                    "bank": banks[i],
                    "row": peak.getRow(),
                    "col": peak.getCol(),
                    "ind": peak.getHKL().norm2() > 0,
                    "Q": list(peak.getQSampleFrame()),
                    "peak_no": i,
                }
                peak_info.append(peak_data)
                peak.setPeakNumber(i)

            self.peak_info = peak_info

            return peak_info

    def get_alignment_info(self, run_number, tilts=(0.0, 0.0, 0.0)):
        """
        Compare observed and UB-predicted Q vectors for peaks in a run.

        Applies an optional goniometer tilt correction (yaw, pitch, roll)
        and computes, for each indexed peak belonging to `run_number`, the
        observed Q-sample vector (tilt-corrected) and the corresponding
        UB-predicted Q vector, for use in goniometer alignment analysis.

        Parameters
        ----------
        run_number : int
            Run number to select peaks from.
        tilts : tuple, optional
            Goniometer tilt angles (yaw, pitch, roll) in degrees applied
            as an extra rotation before comparing to the goniometer
            matrix. The default is (0.0, 0.0, 0.0).

        Returns
        -------
        info : dict or None
            Dictionary with keys "run_number", "observed", "predicted",
            "observed_hkl", and "tilts", or None if there are no peaks,
            no UB matrix, or no indexed peaks for the given run.
        """
        if not (self.has_peaks() and self.has_UB()):
            return None

        UB = self.get_UB()
        B = 2 * np.pi * UB

        yaw, pitch, roll = tilts
        G = scipy.spatial.transform.Rotation.from_euler(
            "yxz", [yaw, pitch, roll], degrees=True
        ).as_matrix()

        observed, predicted, observed_hkl = [], [], []

        for peak in mtd[self.table]:
            if peak.getRunNumber() != run_number or peak.getHKL().norm2() <= 0:
                continue

            hkl = np.array(list(peak.getHKL()), dtype=float)
            q_sample = np.array(list(peak.getQSampleFrame()), dtype=float)
            R = np.array(peak.getGoniometerMatrix(), dtype=float)

            q_lab = R @ q_sample
            q_observed = np.linalg.solve(G @ R, q_lab)
            hkl_observed = np.linalg.solve(B, q_observed)

            observed.append(q_observed)
            predicted.append(B @ hkl)
            observed_hkl.append(hkl_observed)

        if len(observed) == 0:
            return None

        return {
            "run_number": run_number,
            "observed": np.asarray(observed),
            "predicted": np.asarray(predicted),
            "observed_hkl": np.asarray(observed_hkl),
            "tilts": np.asarray(tilts, dtype=float),
        }

    def get_peak(self, i):
        """
        Get a specific peak's information by index.

        Parameters
        ----------
        i : int
            Index of the peak.

        Returns
        -------
        dict or None
            Peak information dictionary or None if index is out of range.
        """

        if self.peak_info is not None:
            if i >= 0 and i < len(self.peak_info):
                return self.peak_info[i]

    def calculate_fractional(
        self, mod_vec_1, mod_vec_2, mod_vec_3, int_hkl, int_mnp
    ):
        """
        Calculate the fractional coordinates from the given modulation vectors and integer coordinates.

        Parameters
        ----------
        mod_vec_1, mod_vec_2, mod_vec_3 : list
            Modulation vectors.
        int_hkl : list
            Integer HKL coordinates.
        int_mnp : list
            Integer MNP coordinates.

        Returns
        -------
        np.ndarray
            Calculated fractional coordinates.
        """

        if self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            ol.setModVec1(V3D(*mod_vec_1))
            ol.setModVec2(V3D(*mod_vec_2))
            ol.setModVec3(V3D(*mod_vec_3))

        delta_hkl = np.column_stack([mod_vec_1, mod_vec_2, mod_vec_3])

        return np.array(int_hkl) + np.dot(delta_hkl, int_mnp)

    def calculate_integer(self, mod_vec_1, mod_vec_2, mod_vec_3, hkl):
        """
        Calculate the integer coordinates from the given modulation vectors and HKL coordinates.

        Parameters
        ----------
        mod_vec_1, mod_vec_2, mod_vec_3 : list
            Modulation vectors.
        hkl : list
            HKL coordinates.

        Returns
        -------
        tuple
            Best integer coordinates (int_hkl, int_mnp) that minimize the error.
        """

        if self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            ol.setModVec1(V3D(*mod_vec_1))
            ol.setModVec2(V3D(*mod_vec_2))
            ol.setModVec3(V3D(*mod_vec_3))

        delta_hkl = np.column_stack([mod_vec_1, mod_vec_2, mod_vec_3])

        bounds_m = range(-3, 4) if np.linalg.norm(mod_vec_1) > 0 else [0]
        bounds_n = range(-3, 4) if np.linalg.norm(mod_vec_2) > 0 else [0]
        bounds_p = range(-3, 4) if np.linalg.norm(mod_vec_3) > 0 else [0]

        min_error = np.inf

        for m in bounds_m:
            for n in bounds_n:
                for p in bounds_p:
                    int_mnp = np.array([m, n, p])
                    residual = np.array(hkl) - np.dot(delta_hkl, int_mnp)
                    int_hkl = np.round(residual).astype(int)
                    model = int_hkl + np.dot(delta_hkl, int_mnp)
                    error = np.linalg.norm(hkl - model)
                    if error < min_error:
                        min_error = error
                        best_solution = (int_hkl, int_mnp)

        return best_solution

    def set_peak(self, i, hkl, int_hkl, int_mnp):
        """
        Set the HKL, integer HKL, and integer MNP values for a specific peak.

        Parameters
        ----------
        i : int
            Index of the peak.
        hkl : list
            HKL coordinates.
        int_hkl : list
            Integer HKL coordinates.
        int_mnp : list
            Integer MNP coordinates.
        """

        peak = mtd[self.table].getPeak(i)

        peak.setHKL(*hkl)
        peak.setIntHKL(V3D(*np.array(int_hkl).astype(float).tolist()))
        peak.setIntMNP(V3D(*np.array(int_mnp).astype(float).tolist()))

    def calculate_peaks(self, hkl_1, hkl_2, a, b, c, alpha, beta, gamma):
        """
        Calculate d-spacing and angle between two HKL planes.

        Parameters
        ----------
        hkl_1, hkl_2 : list
            Miller indices for the two planes.
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        Returns
        -------
        tuple
            d-spacing for the two planes and the angle between them.
        """

        uc = UnitCell(a, b, c, alpha, beta, gamma)

        d_1 = d_2 = phi_12 = None
        if hkl_1 is not None:
            d_1 = uc.d(*hkl_1)
        if hkl_2 is not None:
            d_2 = uc.d(*hkl_2)
        if hkl_1 is not None and hkl_2 is not None:
            phi_12 = uc.recAngle(*hkl_1, *hkl_2)

        return d_1, d_2, phi_12

    def calculate_highlight(self, Q1, Q2):
        """
        Calculate the angle between two Q vectors.

        Parameters
        ----------
        Q1, Q2 : array_like
            Q vectors (in reciprocal space) to compare.

        Returns
        -------
        phi : float
            Angle between `Q1` and `Q2` in degrees.
        """
        q1 = np.array(Q1) / np.linalg.norm(Q1)
        q2 = np.array(Q2) / np.linalg.norm(Q2)
        return np.rad2deg(np.arccos(np.clip(np.dot(q1, q2), -1, 1)))

    def cluster_peaks(self, peak_info, eps=0.025, min_samples=15):
        """
        Cluster peaks using DBSCAN algorithm.

        Parameters
        ----------
        peak_info : dict
            Dictionary containing peak information (coordinates, inverse transform, etc.).
        eps : float, optional
            The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default is 0.025.
        min_samples : int, optional
            The number of samples in a neighborhood for a point to be considered as a core point. Default is 15.

        Returns
        -------
        success : bool
            True if clustering is successful, False otherwise.
        """

        T_inv = peak_info["inverse"]

        points = np.array(peak_info["coordinates"])

        clustering = DBSCAN(eps=eps, min_samples=min_samples)

        labels = clustering.fit_predict(points)

        uni_labels, inverse = np.unique(labels, return_inverse=True)

        centroids = []
        for label in uni_labels:
            if label >= 0:
                center = points[labels == label].mean(axis=0)
                centroids.append(np.dot(T_inv, center))
        centroids = np.array(centroids)

        success = False

        if centroids.shape[0] > 0 and len(centroids.shape) == 2:
            null = np.argmin(np.linalg.norm(centroids, axis=1))

            mask = np.ones_like(centroids[:, 0], dtype=bool)
            mask[null] = False

            peaks = np.arange(mask.size)[mask]

            satellites = centroids[mask]
            nuclear = centroids[null]

            clusters = labels.copy()
            clusters[labels == null] = 0

            peak_info["clusters"] = clusters
            peak_info["nuclear"] = nuclear
            peak_info["satellites"] = np.empty((0, 3))

            success = True

            dist = scipy.spatial.distance_matrix(satellites, -satellites)

            n = dist.shape[0]

            if n > 2:
                indices = np.column_stack(
                    [np.arange(n), np.argmin(dist, axis=0)]
                )
                indices = np.sort(indices, axis=1)
                indices = np.unique(indices, axis=0)

                mod = 1
                satellites = []
                for inds in indices:
                    i, j = peaks[inds[0]], peaks[inds[1]]
                    clusters[labels == i] = mod
                    clusters[labels == j] = mod
                    satellites.append(centroids[i])
                    mod += 1
                satellites = np.array(satellites)

                peak_info["clusters"] = clusters
                peak_info["nuclear"] = nuclear
                peak_info["satellites"] = satellites

        return success

    def get_cluster_info(self):
        """
        Get cluster information for peaks based on UB and peak table.

        Returns
        -------
        clusters : dict
            Dictionary with cluster coordinates, points, numbers, translation vectors, and transforms.
        """

        if self.has_UB() and self.has_peaks():
            UB = self.get_UB()
            peak_dict = {}

            Qs, HKLs, pk_nos = [], [], []

            for j, peak in enumerate(mtd[self.table]):
                pk_no = j + 1

                diff_HKL = peak.getHKL() - np.round(peak.getHKL())

                Q = 2 * np.pi * np.dot(UB, diff_HKL)

                Qs.append(Q)
                HKLs.append(diff_HKL)
                pk_nos.append(pk_no)

                diff_HKL = peak.getHKL() - np.round(peak.getHKL())

                Q = 2 * np.pi * np.dot(UB, -diff_HKL)

                Qs.append(Q)
                HKLs.append(diff_HKL)
                pk_nos.append(-pk_no)

            peak_dict["coordinates"] = Qs
            peak_dict["points"] = HKLs
            peak_dict["numbers"] = pk_nos

            translation = (
                2 * np.pi * UB[:, 0],
                2 * np.pi * UB[:, 1],
                2 * np.pi * UB[:, 2],
            )

            peak_dict["translation"] = translation

            T = np.column_stack(translation)

            peak_dict["transform"] = T
            peak_dict["inverse"] = np.linalg.inv(T)

            return peak_dict
