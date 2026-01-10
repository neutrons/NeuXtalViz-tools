import os

from qtpy.QtTest import QTest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog

from utilities import run_qt_scenario, copy_generated_pngs

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def _activate_crystal_structure_tool(window):
    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(3)

    cs_presenter = window.cs
    return cs_presenter, cs_presenter.view


def _load_cif_and_compute_f2(
    app, window, instrument, cif_filename, screenshot_prefix
):
    directory = os.path.join(DIRECTORY, instrument)
    os.makedirs(directory, exist_ok=True)

    cs_presenter, cs_view = _activate_crystal_structure_tool(window)

    cif_path = os.path.join(DIRECTORY, cif_filename)

    original_get_open = QFileDialog.getOpenFileName

    def _fake_get_open(self, *args, **kwargs):  # noqa: ANN001
        return cif_path, "CIF files (*.cif)"

    QFileDialog.getOpenFileName = _fake_get_open

    try:
        cs_view.load_CIF_button.setStyleSheet("background-color: green;")
        QTest.mouseClick(cs_view.load_CIF_button, Qt.LeftButton)
        QTest.qWait(1000 * 5)
    finally:
        QFileDialog.getOpenFileName = original_get_open

    cs_view.tab_widget.setCurrentIndex(0)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, f"{screenshot_prefix}_structure.png"), "png"
    )

    cs_view.load_CIF_button.setStyleSheet("")

    cs_view.tab_widget.setCurrentIndex(1)

    if hasattr(cs_view, "dmin_line"):
        cs_view.dmin_line.setText("1")

    cs_view.calculate_button.setStyleSheet("background-color: green;")
    QTest.mouseClick(cs_view.calculate_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, f"{screenshot_prefix}_F2.png"), "png"
    )

    cs_view.calculate_button.setStyleSheet("")

    copy_generated_pngs(directory)


def MANDI_Mesolite_Structure(app, window):
    _load_cif_and_compute_f2(
        app,
        window,
        instrument="MANDI",
        cif_filename=os.path.join("../data", "mesolite.cif"),
        screenshot_prefix="Mesolite_structure",
    )


def CORELLI_Natrolite_Structure(app, window):
    _load_cif_and_compute_f2(
        app,
        window,
        instrument="CORELLI",
        cif_filename=os.path.join("../data", "natrolite.cif"),
        screenshot_prefix="Natrolite_structure",
    )


def TOPAZ_Scolecite_Structure(app, window):
    _load_cif_and_compute_f2(
        app,
        window,
        instrument="TOPAZ",
        cif_filename=os.path.join("../data", "scolecite.cif"),
        screenshot_prefix="Scolecite_structure",
    )


def TOPAZ_Si_Structure(app, window):
    _load_cif_and_compute_f2(
        app,
        window,
        instrument="TOPAZ",
        cif_filename=os.path.join("../data", "Si.cif"),
        screenshot_prefix="Si_structure",
    )


def SNAP_Si_Structure(app, window):
    _load_cif_and_compute_f2(
        app,
        window,
        instrument="SNAP",
        cif_filename=os.path.join("../data", "Si.cif"),
        screenshot_prefix="Si_structure",
    )


def TOPAZ_Bixbyite_Structure(app, window):
    _load_cif_and_compute_f2(
        app,
        window,
        instrument="TOPAZ",
        cif_filename=os.path.join("../data", "bixbyite.cif"),
        screenshot_prefix="Bixbyite_structure",
    )


if __name__ == "__main__":
    # run_qt_scenario(MANDI_Mesolite_Structure)
    # run_qt_scenario(CORELLI_Natrolite_Structure)
    # run_qt_scenario(TOPAZ_Scolecite_Structure)
    # run_qt_scenario(TOPAZ_Si_Structure)
    # run_qt_scenario(SNAP_Si_Structure)
    run_qt_scenario(TOPAZ_Bixbyite_Structure)
