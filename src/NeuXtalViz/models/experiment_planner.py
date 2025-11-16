import os

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
    ExtractMonitors,
    PreprocessDetectorsToMD,
    CreateMDWorkspace,
    BinMD,
    SmoothNeighbours,
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
    "C": lambda h, k, l: k % 2 == 0 and l % 2 == 0,
    "A": lambda h, k, l: h % 2 == 0 and l % 2 == 0,
    "B": lambda h, k, l: h % 2 == 0 and k % 2 == 0,
    "R": lambda h, k, l: True,
    "Robv": lambda h, k, l: (-h + k + l) % 3 == 0,
    "Rrev": lambda h, k, l: (h - k + l) % 3 == 0,
}


class ExperimentModel(NeuXtalVizModel):
    """
    Model for managing experiment planning, instrument setup, and
    crystallographic calculations in NeuXtalViz.

    This class provides methods for initializing instruments, handling
    calibration and mask files, managing sample and plan workspaces,
    performing peak prediction, and calculating experiment statistics.
    """

    def __init__(self):
        super(ExperimentModel, self).__init__()

        CreatePeaksWorkspace(
            NumberOfPeaks=0,
            OutputType="LeanElasticPeak",
            OutputWorkspace="coverage",
        )

        self.comment = ""

    def initialize_instrument(self, instrument, logs, cal, gon, mask):
        inst = self.get_instrument_name(instrument)

        if not mtd.doesExist("instrument"):
            LoadEmptyInstrument(
                InstrumentName=inst, OutputWorkspace="instrument"
            )

            CloneWorkspace(
                InputWorkspace="instrument", OutputWorkspace="goniometer"
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
                    InstrumentName=inst,
                )

            if cal != "" and os.path.exists(cal):
                if os.path.splitext(cal)[1] == ".xml":
                    LoadParameterFile(Workspace="instrument", Filename=cal)
                else:
                    LoadIsawDetCal(InputWorkspace="instrument", Filename=cal)

            if gon != "" and os.path.exists(gon):
                if os.path.splitext(gon)[1] == ".xml":
                    LoadParameterFile(Workspace="goniometer", Filename=gon)

            if mask != "" and os.path.exists(mask):
                if not mtd.doesExist("mask"):
                    LoadMask(
                        Instrument=inst, InputFile=mask, OutputWOrkspace="mask"
                    )
                MaskDetectors(Workspace="instrument", MaskedWorkspace="mask")

            ExtractMonitors(
                InputWorkspace="instrument",
                MonitorWorkspace="monitors",
                DetectorWorkspace="instrument",
            )

            cols, rows = beamlines[instrument]["BankPixels"]
            mask_cols, mask_rows = beamlines[instrument]["MaskEdges"]

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

            PreprocessDetectorsToMD(
                InputWorkspace="instrument", OutputWorkspace="detectors"
            )
            mask = np.array(mtd["detectors"].column(7)) != 0
            det_ID = np.array(mtd["detectors"].column(4))[mask]
            det_ID = np.insert(det_ID, -1, -1)

            c, r = beamlines[instrument]["Grouping"].split("x")

            SmoothNeighbours(
                InputWorkspace="instrument",
                OutputWorkspace="instrument",
                SumPixelsX=c,
                SumPixelsY=r,
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

            L2 = np.array(mtd["detectors"].column(1))[mask]
            tt = np.array(mtd["detectors"].column(2))[mask]
            az = np.array(mtd["detectors"].column(3))[mask]

            x = L2 * np.sin(tt) * np.cos(az)
            y = L2 * np.sin(tt) * np.sin(az)
            z = L2 * np.cos(tt)

            self.det_ID = det_ID.copy()
            self.nu = np.rad2deg(np.arcsin(y / L2))
            self.gamma = np.rad2deg(np.arctan2(x, z))

    def get_calibration_file_path(self, instrument):
        inst = beamlines[instrument]

        return os.path.join(
            "/",
            inst["Facility"],
            inst["InstrumentName"],
            "shared",
            "calibration",
        )

    def get_vanadium_file_path(self, instrument):
        inst = beamlines[instrument]

        return os.path.join(
            "/", inst["Facility"], inst["InstrumentName"], "shared", "Vanadium"
        )

    def remove_instrument(self):
        if mtd.doesExist("instrument"):
            DeleteWorkspace(Workspace="instrument")

        if mtd.doesExist("cobmined"):
            DeleteWorkspace(Workspace="cobmined")

        if mtd.doesExist("filtered"):
            DeleteWorkspace(Workspace="filtered")

        if mtd.doesExist("footprint"):
            DeleteWorkspace(Workspace="footprint")

    def get_crystal_system_point_groups(self, crystal_system):
        return crystal_system_point_groups[crystal_system]

    def get_point_group_centering(self, point_group):
        return point_group_centering[point_group]

    def get_symmetry(self, point_group, centering):
        return str(point_group), str(centering)

    def create_plan(self, table):
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

    def update_goniometer_motors(self, limits, motors, cal, gon, mask):
        if mtd.doesExist("sample"):
            mtd["sample"].run()["limits"] = np.array(limits).flatten().tolist()

            values = []
            for key in motors.keys():
                values.append(motors[key])
            if len(values) > 0:
                mtd["sample"].run()["motors"] = values

            mtd["sample"].run()["cal"] = cal
            mtd["sample"].run()["gon"] = gon
            mtd["sample"].run()["mask"] = mask

    def load_UB(self, filename):
        LoadIsawUB(InputWorkspace="coverage", Filename=filename)

        self.copy_UB()

    def get_UB(self):
        if self.has_UB():
            return mtd["coverage"].sample().getOrientedLattice().getUB().copy()

    def copy_UB(self):
        UB = self.get_UB()
        if UB is not None:
            self.set_UB(UB)

    def has_UB(self):
        if HasUB(Workspace="coverage"):
            return True
        else:
            return False

    def get_instrument_name(self, instrument):
        return beamlines[instrument]["Name"]

    def get_modes(self, instrument):
        return list(beamlines[instrument]["Goniometer"].keys())

    def get_counting_options(self, instrument):
        return beamlines[instrument]["Counting"]

    def get_scan_log(self, instrument):
        return beamlines[instrument]["Title"]

    def get_axes_polarities(self, instrument, mode):
        goniometers = beamlines[instrument]["Goniometer"][mode]

        axes = [goniometers[name][:-3] for name in goniometers.keys()]

        polarities = [goniometers[name][3] for name in goniometers.keys()]

        return axes, polarities

    def get_goniometer_axes(self, instrument, mode):
        goniometers = beamlines[instrument]["Goniometer"][mode]

        axes = [
            "{}," + ",".join(np.array(goniometers[name][:-2]).astype(str))
            for name in goniometers.keys()
        ]

        return axes

    def get_goniometers(self, instrument, mode):
        goniometers = beamlines[instrument]["Goniometer"][mode]

        return [(name, *goniometers[name][-2:]) for name in goniometers.keys()]

    def get_motors(self, instrument):
        motors = beamlines[instrument].get("Motor")

        if motors is not None:
            return [(name, motors[name]) for name in motors.keys()]
        else:
            return []

    def get_wavelength(self, instrument):
        return beamlines[instrument]["Wavelength"]

    def save_plan(self, filename):
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
        if mtd.doesExist("plan"):
            SaveNexus(InputWorkspace="plan", Filename=filename)
            if mtd.doesExist("sample"):
                SaveNexus(
                    InputWorkspace="sample", Filename=filename, Append=True
                )

    def load_experiment(self, filename):
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
        gon = mtd[sample].run().getProperty("gon").value
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
        config = (instrument, mode, wl, d_min, lims, vals, cal, gon, mask)
        symm = (cs, pg, lc)

        return plan, config, symm

    def generate_axes(self, axes, polarities):
        self.axes = [None] * 6

        for i, (axis, polarity) in enumerate(zip(axes, polarities)):
            self.axes[i] = "{},"
            self.axes[i] += ",".join(np.array([*axis, polarity]).astype(str))

    def get_setting(self, free_angles, limits):
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
        self.generate_axes(axes, polarities)

        free = 0
        for limit in limits:
            free += 1 - np.isclose(limit[0], limit[1])
        step *= free

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
        pg = PointGroupFactory.createPointGroup(pg)

        hkls_1 = pg.getEquivalents(hkl_1) if equiv else [hkl_1]
        hkls_2 = pg.getEquivalents(hkl_2) if equiv else [hkl_2]

        hkls_1 = [tuple(hkl) for hkl in hkls_1]
        hkls_2 = [tuple(hkl) for hkl in hkls_2]

        unique_pairs = set()
        for a, b in itertools.product(hkls_1, hkls_2):
            if a != b:
                unique_pairs.add(tuple(sorted((a, b))))
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
        if len(self.angles_gamma) > 0:
            d2 = (self.angles_gamma - gamma) ** 2 + (self.angles_nu - nu) ** 2

            i = np.argmin(d2)

            angles = self.angles[i]

            gamma = self.angles_gamma[i]
            nu = self.angles_nu[i]
            lamda = self.angles_lamda[i]
            d = self.angles_d[i]

            gamma_alt = nu_alt = lamda_alt = d_alt = None

            self.comment = (
                "#(" + ", ".join(self.angles_indices[i].astype(str)) + ")"
            )

            if self.angles_lamda_alt is not None:
                gamma_alt = self.angles_gamma_alt[i]
                nu_alt = self.angles_nu_alt[i]
                lamda_alt = self.angles_lamda_alt[i]
                d_alt = self.angles_d_alt[i]

                self.comment += (
                    ")_#("
                    + ", ".join(self.angles_indices_alt[i].astype(str))
                    + ")"
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

    def add_mesh(
        self, mesh_angles, wavelength, d_min, rows, free_angles, all_angles
    ):
        limits, ns = mesh_angles

        mins, maxs = zip(*limits)

        axes = [
            np.linspace(lo, hi, n + 1)[:-1]
            for lo, hi, n in zip(mins, maxs, ns)
        ]

        grids = np.meshgrid(*axes, indexing="ij")
        points = np.stack(grids, axis=-1).reshape(-1, len(limits))

        indices = [all_angles.index(free) for free in free_angles]

        values = []
        for i, angles in enumerate(points):
            self.add_orientation(angles, wavelength, d_min, rows + i)
            values.append(angles[indices])

        return values

    def add_orientation(self, angles, wavelength, d_min, rows):
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

        d_max = 1.1 * np.max([ol.d(1, 0, 0), ol.d(0, 1, 0), ol.d(0, 0, 1)])

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
        if row == -1 and mtd.doesExist("missing"):
            CloneWorkspace(InputWorkspace="missing", OutputWorkspace="table")
        else:
            FilterPeaks(
                InputWorkspace="combined",
                FilterVariable="RunNumber",
                FilterValue=str(row),
                Operator="=",
                OutputWorkspace="table",
            )

        SortPeaksWorkspace(
            InputWorkspace="table",
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace="table",
        )

        columns = ["l", "k", "h"]

        for col in columns:
            SortPeaksWorkspace(
                InputWorkspace="table",
                ColumnNameToSortBy=col,
                SortAscending=False,
                OutputWorkspace="table",
            )

        SortPeaksWorkspace(
            InputWorkspace="table",
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace="table",
        )

        h = mtd["table"].column("h")
        k = mtd["table"].column("k")
        l = mtd["table"].column("l")

        d = mtd["table"].column("DSpacing")
        lamda = mtd["table"].column("Wavelength")

        return np.array([h, k, l, d, lamda]).T.tolist()

    def calculate_statistics(self, point_group, lattice_centering, use, d_min):
        shel_sym, comp_sym, mult_sym, refl_sym = [], [], [], []
        shel_asym, comp_asym, mult_asym, refl_asym = [], [], [], []

        comp_cumsym, mult_cumsym, refl_cumsym = [], [], []
        comp_cumasym, mult_cumasym, refl_cumasym = [], [], []

        if mtd.doesExist("combined"):
            CloneWorkspace(
                InputWorkspace="combined", OutputWorkspace="filtered"
            )

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

            if mtd["filtered"].getNumberPeaks() > 0:
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

                unique, completeness, redundancy, multiple, _ = symmetric

                shel_sym = ["Overall"]
                comp_sym = [completeness * 100]
                mult_sym = [redundancy]
                refl_sym = [unique]

                for i in range(len(d) - 1):
                    symmetric = CountReflections(
                        InputWorkspace="filtered",
                        PointGroup=pg,
                        LatticeCentering=lc,
                        MinDSpacing=d[i + 1],
                        MaxDSpacing=d[i],
                        MissingReflectionsWorkspace="",
                    )

                    unique, completeness, redundancy, multiple = symmetric

                    shel_sym.append("{:.2f}-{:.2f}".format(d[i], d[i + 1]))
                    comp_sym.append(completeness * 100)
                    mult_sym.append(redundancy)
                    refl_sym.append(unique)

                asymmetric = CountReflections(
                    InputWorkspace="filtered",
                    PointGroup="1",
                    LatticeCentering=lc,
                    MinDSpacing=d_min,
                    MaxDSpacing=d_max,
                    MissingReflectionsWorkspace="",
                )

                unique, completeness, redundancy, multiple = asymmetric

                shel_asym = ["Overall"]
                comp_asym = [completeness * 100]
                mult_asym = [redundancy]
                refl_asym = [unique]

                for i in range(len(d) - 1):
                    asymmetric = CountReflections(
                        InputWorkspace="filtered",
                        PointGroup="1",
                        LatticeCentering=lc,
                        MinDSpacing=d[i + 1],
                        MaxDSpacing=d[i],
                        MissingReflectionsWorkspace="",
                    )

                    unique, completeness, redundancy, multiple = asymmetric

                    shel_asym.append("{:.2f}-{:.2f}".format(d[i], d[i + 1]))
                    comp_asym.append(completeness * 100)
                    mult_asym.append(redundancy)
                    refl_asym.append(unique)

                rows = np.unique(mtd["filtered"].column("RunNumber")).tolist()
                for row in rows:
                    CloneWorkspace(
                        InputWorkspace="filtered", OutputWorkspace="cumulative"
                    )
                    for cumrow in rows:
                        if cumrow > row:
                            FilterPeaks(
                                InputWorkspace="cumulative",
                                FilterVariable="RunNumber",
                                FilterValue=str(cumrow),
                                Operator="!=",
                                OutputWorkspace="cumulative",
                            )

                    symmetric = CountReflections(
                        InputWorkspace="cumulative",
                        PointGroup=pg,
                        LatticeCentering=lc,
                        MinDSpacing=d_min,
                        MaxDSpacing=d_max,
                        MissingReflectionsWorkspace="",
                    )

                    unique, completeness, redundancy, multiple = symmetric

                    comp_cumsym.append(completeness * 100)
                    mult_cumsym.append(redundancy)
                    refl_cumsym.append(unique)

                    asymmetric = CountReflections(
                        InputWorkspace="cumulative",
                        PointGroup="1",
                        LatticeCentering=lc,
                        MinDSpacing=d_min,
                        MaxDSpacing=d_max,
                        MissingReflectionsWorkspace="",
                    )

                    unique, completeness, redundancy, multiple = asymmetric

                    comp_cumasym.append(completeness * 100)
                    mult_cumasym.append(redundancy)
                    refl_cumasym.append(unique)

            else:
                return None

        sym = (shel_sym, comp_sym, mult_sym, refl_sym)
        asym = (shel_asym, comp_asym, mult_asym, refl_asym)
        cumsym = (comp_cumsym, mult_cumsym, refl_cumsym)
        cumasym = (comp_cumasym, mult_cumasym, refl_cumasym)

        return sym, asym, cumsym, cumasym

    def hsl_to_rgb(self, hue, saturation, lightness):
        h = np.array(hue)
        s = np.array(saturation)
        l = np.array(lightness)

        def f(h, s, l, n):
            k = (n + h / 30) % 12
            a = s * np.minimum(l, 1 - l)
            return l - a * np.maximum(
                -1, np.minimum(np.minimum(k - 3, 9 - k), 1)
            )

        rgb = np.stack((f(h, s, l, 0), f(h, s, l, 8), f(h, s, l, 4)), axis=-1)

        return rgb

    def delete_angles(self, rows):
        for row in rows:
            FilterPeaks(
                InputWorkspace="combined",
                FilterVariable="RunNumber",
                FilterValue=str(row),
                Operator="!=",
                OutputWorkspace="combined",
            )

        # SortPeaksWorkspace(
        #     InputWorkspace="combined",
        #     ColumnNameToSortBy="RunNumber",
        #     SortAscending=True,
        #     OutputWorkspace="combined",
        # )

        runs = mtd["combined"].column(0)

        _, new_runs = np.unique(runs, return_index=True)

        for new_run, peak in zip(new_runs.tolist(), mtd["combined"]):
            peak.setRunNumber(new_run)

    def get_coverage_info(
        self, point_group, lattice_centering, draw_all, row=None
    ):
        pg = PointGroupFactory.createPointGroup(point_group)

        coverage_dict = {}

        UB = mtd["coverage"].sample().getOrientedLattice().getUB().copy()
        # UB_inv = np.linalg.inv(UB)

        if mtd.doesExist("filtered"):

            ws = "filtered"
            if not draw_all and row is not None:
                if row == -1:
                    ws = "missing"
                else:
                    ws = "individual"
                    FilterPeaks(
                        InputWorkspace="filtered",
                        FilterVariable="RunNumber",
                        FilterValue=str(row),
                        Operator="!=",
                        OutputWorkspace=ws,
                    )

            h = mtd[ws].column("h")
            k = mtd[ws].column("k")
            l = mtd[ws].column("l")

            hkls = np.array([h, k, l]).T.astype(int).tolist()

            # hkl_dict = {}
            # hkl_dict[(0, 0, 0)] = 0
            # for hkl in hkls:
            #     if centering_conditions[lattice_centering](*hkl):
            #         equiv_hkls = pg.getEquivalents(hkl)
            #         for equiv_hkl in equiv_hkls:
            #             key = tuple(equiv_hkl)
            #             no = hkl_dict.get(key)
            #             if no is None:
            #                 no = 1
            #             else:
            #                 no += 1
            #             hkl_dict[key] = no

            cond = centering_conditions[lattice_centering]
            rep_count = defaultdict(int)
            rep_members = {}

            def rep_of(hkl):
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

            r = np.sqrt(hkls[:, 0] ** 2 + hkls[:, 1] ** 2 + hkls[:, 2] ** 2)
            theta = np.arccos(hkls[:, 2] / r)
            phi = np.arctan2(hkls[:, 1], hkls[:, 0])

            hue = phi * 180 / np.pi + 180
            saturation = np.ones_like(hue)
            lightness = theta / np.pi

            rgb = self.hsl_to_rgb(hue, saturation, lightness)
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

        if mtd.doesExist("table"):

            lamda_peaks = []
            gamma_peaks = []
            nu_peaks = []
            d_peaks = []

            for peak in mtd["table"]:
                lamda = peak.getWavelength()
                Q_lab = peak.getQLabFrame()

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

            lamda_peaks = np.array(lamda_peaks)
            gamma_peaks = np.array(gamma_peaks)
            nu_peaks = np.array(nu_peaks)
            d_peaks = np.array(d_peaks)

            self.lamda_peaks = lamda_peaks
            self.gamma_peaks = gamma_peaks
            self.nu_peaks = nu_peaks
            self.d_peaks = d_peaks

            return gamma_peaks, nu_peaks, lamda_peaks, d_peaks

    def get_peak_index(self, i):

        if len(self.lamda_peaks) > 0:
            gamma = self.gamma_peaks[i]
            nu = self.nu_peaks[i]

            return gamma, nu, i

    def get_peak_selection(self, gamma, nu):

        if len(self.lamda_peaks) > 0:
            d2 = (self.gamma_peaks - gamma) ** 2 + (self.nu_peaks - nu) ** 2

            i = np.argmin(d2)

            gamma = self.gamma_peaks[i]
            nu = self.nu_peaks[i]

            return gamma, nu, i

    def to_index(self, Q, Q_max, scale, n):
        idx = np.round((Q + Q_max) * scale).astype(np.int32)
        return np.clip(idx, 0, n - 1)

    def extract_data(self, ws):

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
        if not mtd.doesExist("footprint"):
            lamda_min, lamda_max = wavelength
            k_min = 2 * np.pi / lamda_max
            k_max = 2 * np.pi / lamda_min

            two_theta = np.array(mtd["detectors"].column("TwoTheta"))
            azimthal = np.array(mtd["detectors"].column("Azimuthal"))

            Q_max = 2 * np.pi / d_min

            hist = np.zeros((n, n, n))

            kx_hat = np.sin(two_theta) * np.cos(azimthal)
            ky_hat = np.sin(two_theta) * np.sin(azimthal)
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
        proj = np.array(proj).reshape(3, 3)
        invalid = np.isclose(np.linalg.det(proj), 0)
        return *proj, invalid

    def calculate_rotations(
        self,
        mesh_angles,
        U,
        V,
        W,
        normal,
        value,
        thickness,
        point_group="1",
        use_symmetry=False,
        factor=2,
    ):
        limits, ns = mesh_angles

        mins, maxs = zip(*limits)

        axes = [
            np.linspace(lo, hi, n + 1)[:-1]
            for lo, hi, n in zip(mins, maxs, ns)
        ]

        UB = mtd["coverage"].sample().getOrientedLattice().getUB()

        grids = np.meshgrid(*axes, indexing="ij")
        points = np.stack(grids, axis=-1).reshape(-1, len(limits))

        for i, angles in enumerate(points):
            axes = np.array(self.axes).copy().tolist()
            for i, angle in enumerate(angles):
                axes[i] = axes[i].format(angle)

        meshmap = str(axes)

        if use_symmetry:
            meshmap += "_" + point_group

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
        fit = []
        for j in range(n_indiv):
            for i in range(n_orient):
                self.generation(i, j)
            self.recombination(n_orient, j)
            fit.append(self.fitness("peaks_{}".format(j)))

        return np.array(fit)

    def recombination(self, n_orient, j):
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
