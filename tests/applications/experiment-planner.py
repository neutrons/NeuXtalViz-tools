import os

from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from utilities import run_qt_scenario, copy_generated_pngs

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def TOPAZ_Si_plan(app, window):
    directory = os.path.join(DIRECTORY, "TOPAZ")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(1)

    ep_presenter = window.ep
    ep_view = ep_presenter.view

    index = ep_view.instrument_combo.findText("TOPAZ")
    ep_view.instrument_combo.setCurrentIndex(index)

    index = ep_view.mode_combo.findText("Ambient")
    ep_view.mode_combo.setCurrentIndex(index)

    ep_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ep_view.mode_combo.setStyleSheet("background-color: yellow;")
    ep_view.load_UB_button.setStyleSheet("background-color: yellow;")

    # QTest.mouseClick(ep_view.convert_to_q_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_plan_initialize.png"), "png"
    )

    ep_view.instrument_combo.setStyleSheet("")

    copy_generated_pngs(directory)


if __name__ == "__main__":
    run_qt_scenario(TOPAZ_Si_plan)
    # run_qt_scenario(CORELLI_Bixbyite_plan)
