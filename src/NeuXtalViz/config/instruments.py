"""
Static per-instrument (beamline) configuration for SNS and HFIR.

Provides the ``beamlines`` dictionary consumed by the UB and experiment
planning models/views to look up instrument-specific defaults such as
wavelength range, detector geometry, goniometer definitions, and data
file naming conventions.

Attributes
----------
beamlines : dict[str, dict]
    Mapping of instrument key (e.g. "SNAP", "CORELLI", "TOPAZ", "MANDI",
    "IMAGINE", "WAND²", "DEMAND") to a dictionary of settings. Common
    keys found in the per-instrument dictionaries:

    Name : str
        Short display/identifier name for the beamline.
    InstrumentName : str
        Mantid instrument name, used e.g. with ``MaskBTP``.
    Facility : str
        Facility the instrument belongs to ("SNS" or "HFIR").
    Wavelength : float or list[float]
        Incident wavelength in angstroms, or a
        ``[min, max]`` wavelength band for white-beam/Laue instruments.
    MinD : float
        Minimum d-spacing in angstroms usable for peak indexing/search.
    Grouping : str
        Default detector pixel grouping/binning as a "columns x rows"
        string (e.g. "2x2"), parsed and passed to Mantid's grouping
        algorithms.
    PixelSize : list[float]
        Detector pixel size in meters as ``[width, height]``.
    BankPixels : list[int]
        Number of pixels per detector bank as ``[columns, rows]``.
    MaskEdges : list[int]
        Number of pixels to mask at each tube/bank edge as
        ``[columns, rows]``, to exclude noisy edge pixels.
    MaskBanks : list[int]
        Bank numbers to mask entirely for this instrument.
    MaskLost : list[list], optional
        Additional bank/tube/pixel regions to mask, each entry as
        ``[bank, [tube_min, tube_max], [pixel_min, pixel_max]]``.
    Goniometers : list[str]
        Goniometer axis definitions as comma-separated strings
        ``"name,x,y,z,sense"`` (motor/log name and rotation axis
        vector components and sense), passed directly to Mantid's
        ``SetGoniometer`` algorithm.
    Goniometer : dict[str, dict]
        Named goniometer configurations (e.g. different sample
        environments or angle conventions) mapping each motor/log
        name to ``[x, y, z, sense, min, max]``: rotation axis vector
        components, rotation sense, and allowed angle range in
        degrees.
    GoniometerNames : str
        Display string of goniometer axis symbols (e.g. "ω,χ,φ").
    Motor : dict[str, float], optional
        Default fixed motor positions (e.g. detector arc/translation
        or two-theta) used when planning or simulating an experiment.
    RawFile : str
        Template file path (relative to an IPTS directory) for the
        instrument's raw event/histogram file, with a ``{}``
        placeholder for the run number.
    Counting : list[str]
        Available run-duration counting/normalization options: a
        proton-charge log/PV name and/or ``"seconds"`` for time-based
        counting.
    Title : str
        EPICS PV name providing the run/scan title, used to read or
        display the current run's title.
"""

LIVE_INSTRUMENTS = ("TOPAZ", "CORELLI", "MANDI")
"""Instruments with a Mantid SNSLiveEventDataListener entry, eligible for live data."""

beamlines = {
    "SNAP": {
        "Name": "SNAP",
        "InstrumentName": "SNAP",
        "Facility": "SNS",
        "Wavelength": [0.6, 3.7],
        "MinD": 0.7,
        "Grouping": "2x2",
        "PixelSize": [0.125, 0.125],
        "BankPixels": [256, 256],
        "MaskEdges": [16, 16],
        "MaskBanks": [],
        "Goniometers": [
            "BL3:Mot:omega,0,1,0,1",
            "BL3:Mot:chi,0,0,1,1",
            "BL3:Mot:phi,0,1,0,1",
        ],
        "Goniometer": {
            "chi-45": {
                "BL3:Mot:omega": [0, 1, 0, 1, -180, 180],
                "BL3:Mot:phi": [0.70710678, 0.70710678, 0, 1, -180, 180],
            },
            "chi-0": {
                "BL3:Mot:omega": [0, 1, 0, 1, -180, 180],
                "BL3:Mot:phi": [0, 1, 0, 1, -180, 180],
            },
        },
        "GoniometerNames": "ω,χ,φ",
        "Motor": {
            "det_lin1": 0,
            "det_lin2": 0,
            "det_arc1": -65,
            "det_arc2": 105,
        },
        "RawFile": "nexus/SNAP_{}.nxs.h5",
        "Counting": ["BL3:Det:PCharge:C", "seconds"],
        "Title": "BL3:SMS:RunInfo:RunTitle",
    },
    "CORELLI": {
        "Name": "CORELLI",
        "InstrumentName": "CORELLI",
        "Facility": "SNS",
        "Wavelength": [0.6, 2.5],
        "MinD": 0.7,
        "Grouping": "1x4",
        "PixelSize": [0.4, 0.125],
        "BankPixels": [16, 256],
        "MaskEdges": [0, 16],
        "MaskBanks": [1, 2, 3, 4, 5, 6, 29, 30, 62, 63, 64, 65, 66, 67, 91],
        "MaskLost": [[58, [13, 16], [80, 130]], [59, [1, 4], [80, 130]]],
        "Goniometers": [
            "BL9:Mot:Sample:Axis1,0,1,0,1",
            "BL9:Mot:Sample:Axis2,0,1,0,1",
            "BL9:Mot:Sample:Axis3,0,1,0,1",
        ],
        "Goniometer": {
            "Goniometer": {
                "BL9:Mot:Sample:Axis1": [0, 1, 0, 1, 0, 0],
                "BL9:Mot:Sample:Axis2": [0, 1, 0, 1, 0, 0],
                "BL9:Mot:Sample:Axis3": [0, 1, 0, 1, 0, 360],
            }
        },
        "GoniometerNames": "ω,χ,φ",
        "RawFile": "nexus/CORELLI_{}.nxs.h5",
        "Counting": ["BL9:Det:PCharge:C", "seconds"],
        "Title": "BL9:SMS:RunInfo:RunTitle",
    },
    "TOPAZ": {
        "Name": "TOPAZ",
        "InstrumentName": "TOPAZ",
        "Facility": "SNS",
        "Wavelength": [0.4, 3.5],
        "MinD": 0.7,
        "Grouping": "2x2",
        "PixelSize": [0.125, 0.125],
        "BankPixels": [256, 256],
        "MaskEdges": [16, 16],
        "MaskBanks": [],
        "Goniometers": [
            "BL12:Mot:omega,0,1,0,1",
            "BL12:Mot:chi,0,0,1,1",
            "BL12:Mot:phi,0,1,0,1",
        ],
        "Goniometer": {
            "Ambient": {
                "BL12:Mot:goniokm:omega": [0, 1, 0, 1, -180, 180],
                "BL12:Mot:goniokm:chi": [0, 0, 1, 1, 135, 135],
                "BL12:Mot:goniokm:phi": [0, 1, 0, 1, -180, 180],
            },
            "Cryogenic": {
                "BL12:Mot:Gonioc:Omega": [0, 1, 0, 1, -180, 180],
                "BL12:Mot:Gonioc:Chi": [0, 0, 1, 1, 0, 0],
                "BL12:Mot:Gonioc:Phi": [0, 1, 0, 1, 0, 0],
            },
        },
        "GoniometerNames": "ω,χ,φ",
        "RawFile": "nexus/TOPAZ_{}.nxs.h5",
        "Counting": ["BL12:Det:PCharge:C", "seconds"],
        "Title": "BL12:SMS:RunInfo:RunTitle",
    },
    "MANDI": {
        "Name": "MANDI",
        "InstrumentName": "MANDI",
        "Facility": "SNS",
        "Wavelength": [2, 4],
        "MinD": 2.0,
        "Grouping": "2x2",
        "PixelSize": [0.125, 0.125],
        "BankPixels": [256, 256],
        "MaskEdges": [16, 16],
        "MaskBanks": [],
        "Goniometers": [
            "BL11B:Mot:omega,0,1,0,1",
            "BL11B:Mot:chi,0,0,1,1",
            "BL11B:Mot:phi,0,1,0,1",
        ],
        "Goniometer": {
            "Goniometer": {
                "BL11B:Mot:omega": [0, 1, 0, 1, 0, 90],
                "BL11B:Mot:chi": [0, 0, 1, 1, 130, 130],
                "BL11B:Mot:phi": [0, 1, 0, 1, -180, 180],
            }
        },
        "GoniometerNames": "ω,χ,φ",
        "RawFile": "nexus/MANDI_{}.nxs.h5",
        "Counting": ["BL11B:Det:PCharge:C", "seconds"],
        "Title": "BL11B:SMS:RunInfo:RunTitle",
    },
    "IMAGINE": {
        "Name": "IMAGINE",
        "InstrumentName": "CG4D",
        "Facility": "HFIR",
        "Wavelength": [2, 4.5],
        "MinD": 2.0,
        "Grouping": "4x4",
        "PixelSize": [0.125, 0.125],
        "BankPixels": [512, 512],
        "MaskEdges": [32, 32],
        "MaskBanks": [],
        "Goniometers": [
            "CG4D:Mot:omega,0,1,0,1",
            "CG4D:Mot:alpha,0,0,1,1",
            "CG4D:Mot:kappa,0,1,0,1",
            "CG4D:Mot:beta,0,0,1,1",
            "CG4D:Mot:phi,0,1,0,1",
        ],
        "Goniometer": {
            "Goniometer": {
                "CG4D:Mot:omega": [0, 1, 0, 1, -180, 180],
                "CG4D:Mot:alpha": [0, 0, 1, 1, 45, 45],
                "CG4D:Mot:kappa": [0, 1, 0, 1, -180, 180],
                "CG4D:Mot:beta": [0, 0, 1, 1, -45, -45],
                "CG4D:Mot:phi": [0, 1, 0, 1, -180, 180],
            }
        },
        "GoniometerNames": "ω,κ,φ",
        "RawFile": "nexus/CG4D_{}.nxs.h5",
        "Counting": ["seconds"],
        "Title": "CG4D:SMS:RunInfo:RunTitle",
    },
    "WAND²": {
        "Name": "WAND",
        "InstrumentName": "HB2C",
        "Facility": "HFIR",
        "Wavelength": 1.486,
        "MinD": 0.7,
        "Grouping": "4x4",
        "PixelSize": [0.125, 0.125],
        "BankPixels": [480, 512],
        "MaskEdges": [0, 16],
        "MaskBanks": [],
        "Goniometers": ["s1,0,1,0,1"],
        "Goniometer": {
            "Goniometer": {
                "HB2C:Mot:sgl": [1, 0, 0, -1, 0, 0],
                "HB2C:Mot:sgu": [0, 0, 1, -1, 0, 0],
                "HB2C:Mot:s1": [0, 1, 0, 1, -180, 180],
            }
        },
        "Motor": {
            "HB2C:Mot:s2.RBV": 30,
            "HB2C:Mot:detz.RBV": 0,
        },
        "GoniometerNames": "l,u,s1",
        "RawFile": "nexus/HB2C_{}.nxs.h5",
        "Counting": ["seconds"],
        "Title": "HB2C:SMS:RunInfo:RunTitle",
    },
    "DEMAND": {
        "Name": "HB3A",
        "InstrumentName": "HB3A",
        "Facility": "HFIR",
        "Wavelength": 1.546,
        "MinD": 0.7,
        "Grouping": "4x4",
        "PixelSize": [0.125, 0.125],
        "BankPixels": [512, 512],
        "MaskEdges": [32, 32],
        "MaskBanks": [],
        "Goniometers": [
            "omega,0,1,0,-1",
            "chi,0,0,1,-1",
            "phi,0,1,0,-1",
        ],
        "Goniometer": {
            "Goniometer": {
                "omega": [0, 1, 0, -1, -13, 45],
                "chi": [0, 0, 1, -1, -91, 91],
                "phi": [0, 1, 0, -1, -180, 180],
            }
        },
        "Motor": {
            "2theta": 30,
            "det_trans": 410.38595,
        },
        "GoniometerNames": "ω,χ,φ",
        "RawFile": "shared/autoreduce/HB3A_exp{:04}_scan{:04}.nxs",
        "Counting": ["seconds"],
        "Title": "Title",
    },
    "SXD": {
        "Name": "SXD",
        "InstrumentName": "SXD",
        "Facility": "ISIS",
        "Wavelength": [0.4, 7.5],
        "MinD": 0.4,
        "Grouping": "1x1",
        "PixelSize": [0.125, 0.125],
        "BankPixels": [64, 64],
        "MaskEdges": [1, 1],
        "MaskBanks": [],
        "Goniometers": [
            "chi,0,1,0,1",
        ],
        "Goniometer": {
            "Goniometer": {
                "chi": [0, 1, 0, 1, 0, 360],
            }
        },
        "GoniometerNames": "χ",
        "RawFile": "nexus/SXD{}.raw",
        "Counting": ["seconds"],
        "Title": "Title",
    },
    "WISH": {
        "Name": "WISH",
        "InstrumentName": "WISH",
        "Facility": "ISIS",
        "Wavelength": [0.8, 10],
        "MinD": 1,
        "Grouping": "1x1",
        "PixelSize": [0.125, 0.125],
        "BankPixels": [152, 512],
        "MaskEdges": [0, 0],
        "MaskBanks": [],
        "Goniometers": [
            "omega,0,1,0,1",
            "chi,1,1,0,1",
        ],
        "Goniometer": {
            "Vertical": {
                "omega": [0, 1, 0, 1, 0, 360],
            },
            "2-axis": {
                "omega": [0, 1, 0, 1, 0, 360],
                "chi": [1, 1, 0, 1, 0, 360],
            },
            "Newport": {
                "omega": [0, 1, 0, -1, -180, 180],
            },
        },
        "GoniometerNames": "ω, χ",
        "RawFile": "nexus/WISH000{}.raw",
        "Counting": ["seconds"],
        "Title": "Title",
    },
}
