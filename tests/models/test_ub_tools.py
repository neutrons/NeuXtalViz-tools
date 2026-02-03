import os
import pytest

from mantid.simpleapi import mtd
from NeuXtalViz.models.ub_tools import UBModel

# Use absolute paths to ensure test data is found regardless of working directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
peaks_file = os.path.join(DATA_DIR, "36079_Niggli.integrate")
ub_file = os.path.join(DATA_DIR, "36079_Niggli.mat")


def test_load_peaks():
    """
    Test loading peaks into UBModel and check the number of peaks loaded.
    """
    ubm = UBModel()
    ubm.load_peaks(peaks_file)
    assert mtd["ub_peaks"].getNumberPeaks() > 0


def test_load_UB():
    """
    Test loading a UB matrix into UBModel and check its shape.
    """
    ubm = UBModel()
    ubm.load_UB(ub_file)
    assert mtd["ub_lattice"].sample().getOrientedLattice().getUB().shape == (
        3,
        3,
    )
