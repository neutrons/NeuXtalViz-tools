import os

from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

from utilities import run_qt_scenario, copy_generated_pngs


DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def CORELLI_Bixbyite_UB(app, window):
    directory = os.path.join(DIRECTORY, "CORELLI")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(0)

    ub_presenter = window.ub
    ub_view = ub_presenter.view

    index = ub_view.instrument_combo.findText("CORELLI")
    ub_view.instrument_combo.setCurrentIndex(index)
    ub_view.ipts_line.setText("36170")
    ub_view.runs_line.setText("72185")

    ub_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ub_view.ipts_line.setStyleSheet("background-color: yellow;")
    ub_view.runs_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_q_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.convert_to_q_button, Qt.LeftButton)
    QTest.qWait(1000 * 60)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_convert_Q.png"), "png"
    )

    ub_view.instrument_combo.setStyleSheet("")
    ub_view.ipts_line.setStyleSheet("")
    ub_view.runs_line.setStyleSheet("")
    ub_view.convert_to_q_button.setStyleSheet("")

    ub_view.max_peaks_line.setStyleSheet("background-color: yellow;")
    ub_view.density_threshold_line.setStyleSheet("background-color: yellow;")
    ub_view.min_distance_line.setStyleSheet("background-color: yellow;")
    ub_view.find_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.find_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_find_peaks.png"), "png"
    )

    ub_view.max_peaks_line.setStyleSheet("")
    ub_view.density_threshold_line.setStyleSheet("")
    ub_view.min_distance_line.setStyleSheet("")
    ub_view.find_button.setStyleSheet("")

    ub_view.calculate_tolerance_line.setStyleSheet("background-color: yellow;")
    ub_view.min_const_line.setStyleSheet("background-color: yellow;")
    ub_view.max_const_line.setStyleSheet("background-color: yellow;")
    ub_view.niggli_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.niggli_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_primitive_cell.png"), "png"
    )

    ub_view.calculate_tolerance_line.setStyleSheet("")
    ub_view.min_const_line.setStyleSheet("")
    ub_view.max_const_line.setStyleSheet("")
    ub_view.niggli_button.setStyleSheet("")

    ub_view.cell_table.selectRow(0)
    ub_view.select_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.select_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_conventional_cell.png"), "png"
    )

    ub_view.cell_table.clearSelection()
    ub_view.select_button.setStyleSheet("")

    ub_view.ub_tab.setCurrentIndex(2)

    index = ub_view.optimize_combo.findText("Cubic")
    ub_view.optimize_combo.setCurrentIndex(index)

    ub_view.optimize_combo.setStyleSheet("background-color: yellow;")
    ub_view.refine_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.refine_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_refine_UB.png"), "png"
    )

    ub_view.optimize_combo.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(1)
    ub_view.peaks_table.selectRow(0)

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_view_peak.png"), "png"
    )

    ub_view.peaks_table.clearSelection()
    ub_view.tab_widget.setCurrentIndex(2)

    ub_view.slice_combo.setStyleSheet("background-color: yellow;")
    ub_view.slice_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_hkl_button.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.convert_to_hkl_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_slice_view.png"), "png"
    )

    copy_generated_pngs(directory)


if __name__ == "__main__":
    run_qt_scenario(CORELLI_Bixbyite_UB)
