import numpy as np

from mantid.simpleapi import (
    LoadEmptyInstrument,
    LoadParameterFile,
    SetGoniometer,
    mtd,
)

_AXIS_LETTERS = "XYZ"

_loaded_calibration_files = {}


def _euler_convention(axis_strings):
    """
    Derive a Mantid Euler-angle convention (e.g. "YZY") from a list of
    ``SetGoniometer`` axis strings ("name,x,y,z,sense"), by mapping each
    axis vector to its dominant unit-vector component.

    Returns None if fewer than three axes are given, or if the resulting
    convention does not alternate (repeated axis letters cannot be
    decomposed/rebuilt as a proper Euler triple).

    """

    if len(axis_strings) < 3:
        return None

    convention = ""
    for axis_string in axis_strings[:3]:
        vec = np.array(axis_string.split(",")[1:4], dtype=float)
        convention += _AXIS_LETTERS[np.argmax(np.abs(vec))]

    if convention[0] == convention[1] or convention[1] == convention[2]:
        return None

    return convention


def setup_goniometer_calibration(
    ref_inst, calibration_file, calibration_ws="goniometer"
):
    """
    Create (or refresh) a tiny, data-free instrument workspace carrying
    the goniometer calibration parameters (``chi-offset``,
    ``goniometer-tilt``) from an .xml parameter file.

    Cheap by construction: ``LoadEmptyInstrument`` builds only the
    instrument geometry, never touching/cloning real event data. The
    parameter file is only (re-)loaded when ``calibration_file`` differs
    from what is already cached on ``calibration_ws``, so calling this
    repeatedly (e.g. once per run, or once per candidate orientation) is
    free after the first call.

    Parameters
    ----------
    ref_inst : str
        Mantid instrument name to build the dummy workspace for.
    calibration_file : str
        Path to the goniometer calibration .xml parameter file.
    calibration_ws : str, optional
        Name of the dummy calibration workspace. The default is
        "goniometer".

    """

    if _loaded_calibration_files.get(
        calibration_ws
    ) == calibration_file and mtd.doesExist(calibration_ws):
        return

    LoadEmptyInstrument(
        InstrumentName=ref_inst, OutputWorkspace=calibration_ws
    )
    LoadParameterFile(Workspace=calibration_ws, Filename=calibration_file)

    _loaded_calibration_files[calibration_ws] = calibration_file


def correct_goniometer(workspace, axis_strings, calibration_ws="goniometer"):
    """
    Apply a calibrated chi-offset and fixed goniometer tilt to the
    goniometer already set on ``workspace``.

    Extracts the Euler angles of the current goniometer matrix (using the
    convention implied by ``axis_strings``), adds the calibrated
    chi-offset to the middle angle, and rebuilds the rotation matrix
    through Mantid's own ``SetGoniometer`` so the rotation convention
    stays internal to Mantid. The calibrated fixed tilt, if present, is
    then left-multiplied onto the result.

    No-op if fewer than three axes are given, if the axes don't form a
    proper alternating Euler triple, or if the calibration instrument
    defines neither "chi-offset" nor "goniometer-tilt".

    Parameters
    ----------
    workspace : str
        Name of the workspace whose goniometer should be corrected.
    axis_strings : list of str
        The (up to) three "name,x,y,z,sense" axis strings already used to
        set ``workspace``'s current goniometer.
    calibration_ws : str, optional
        Name of the dummy calibration workspace set up by
        :func:`setup_goniometer_calibration`. The default is
        "goniometer".

    """

    convention = _euler_convention(axis_strings)
    if convention is None:
        return

    inst = mtd[calibration_ws].getInstrument()

    has_offset = inst.hasParameter("chi-offset")
    has_tilt = inst.hasParameter("goniometer-tilt")

    if not has_offset and not has_tilt:
        return

    run = mtd[workspace].run()
    gon = run.getGoniometer()

    if has_offset:
        angles = list(gon.getEulerAngles(convention))
        angles[1] += inst.getNumberParameter("chi-offset")[0]

        kwargs = {}
        for i, (angle, axis_string) in enumerate(zip(angles, axis_strings)):
            _, x, y, z, _ = axis_string.split(",")
            kwargs["Axis{}".format(i)] = "{},{},{},{},1".format(angle, x, y, z)

        SetGoniometer(Workspace=workspace, **kwargs)

        gon = mtd[workspace].run().getGoniometer()

    if has_tilt:
        tilt = np.array(
            inst.getStringParameter("goniometer-tilt")[0].split(","),
            dtype=float,
        ).reshape(3, 3)

        gon.setR(tilt @ gon.getR())
