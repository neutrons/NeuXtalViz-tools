import os

import numpy as np

from NeuXtalViz.models.experiment_planner import ExperimentModel


def test_save_load_experiment(tmp_path):
    """
    Round-trip a plan/sample through save_experiment/load_experiment and
    check that the plan table, instrument/goniometer settings (including
    the detector, goniometer, and mask calibration files), and crystal
    symmetry all come back unchanged.
    """

    model = ExperimentModel()

    UB = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
        ]
    )

    pv = "Title"
    names = ["omega", "chi", "phi"]
    titles = ["1", "2"]
    settings = [[0.0, 5.0, 10.0], [90.0, 5.0, 10.0]]
    comments = ["", "second setting"]
    counts = ["Time", "Time"]
    values = [60, 120]
    use = [True, False]

    table = (pv, names, titles, settings, comments, counts, values, use)

    model.create_plan(table)
    model.create_sample("TOPAZ", "omega", UB, [0.4, 3.5], 0.7)
    model.update_sample("Orthorhombic", "222", "P")

    limits = [[-180.0, 180.0], [-10.0, 10.0], [-180.0, 180.0]]
    motors = {}
    cal = "/SNS/TOPAZ/shared/calibration/det.xml"
    gon_cal = "/SNS/TOPAZ/shared/calibration/gon.xml"
    mask = "/SNS/TOPAZ/shared/calibration/mask.xml"

    model.update_goniometer_motors(limits, motors, cal, gon_cal, mask)

    filename = os.path.join(str(tmp_path), "test_experiment.nxs")
    model.save_experiment(filename)

    plan, config, symm = model.load_experiment(filename)

    (
        out_titles,
        out_settings,
        out_comments,
        out_counts,
        out_values,
        out_use,
    ) = plan
    (
        out_instrument,
        out_mode,
        out_wl,
        out_d_min,
        out_lims,
        out_vals,
        out_cal,
        out_gon_cal,
        out_mask,
    ) = config
    out_cs, out_pg, out_lc = symm

    assert list(out_titles) == titles
    assert np.allclose(out_settings, settings)
    assert list(out_comments) == comments
    assert list(out_counts) == counts
    assert list(out_values) == values
    assert [bool(u) for u in out_use] == use

    assert out_instrument == "TOPAZ"
    assert out_mode == "omega"
    assert np.allclose(out_wl, [0.4, 3.5])
    assert np.isclose(out_d_min, 0.7)
    assert np.allclose(out_lims, limits)
    assert out_cal == cal
    assert out_gon_cal == gon_cal
    assert out_mask == mask

    assert out_cs == "Orthorhombic"
    assert out_pg == "222"
    assert out_lc == "P"

    assert np.allclose(model.get_UB(), UB)
