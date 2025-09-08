import sys
import os
import threading
import glob
import shutil

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QTimer, QThread, QThreadPool

from NeuXtalViz.application import NeuXtalViz

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def run_qt_scenario(scenario):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    window = NeuXtalViz()
    window.show()

    def _wrapped():
        scenario(app, window)

        window.close()
        QApplication.closeAllWindows()

        for _ in range(3):
            app.processEvents()
            QTest.qWait(50)

        for t in app.findChildren(QTimer):
            t.stop()

        for th in app.findChildren(QThread):
            th.requestInterruption()
            th.quit()
        for th in app.findChildren(QThread):
            th.wait(3000)

        QThreadPool.globalInstance().waitForDone(5000)

    QTimer.singleShot(0, app.quit)

    QTimer.singleShot(0, _wrapped)
    rc = app.exec_()

    alive = [
        t for t in threading.enumerate() if t is not threading.main_thread()
    ]
    if alive:
        os._exit(0)

    return rc


def copy_generated_pngs(directory):
    static = os.path.abspath(os.path.join(DIRECTORY, "../../docs/source"))
    os.makedirs(static, exist_ok=True)
    for png in glob.glob(
        os.path.join(directory, "**", "*.png"), recursive=True
    ):
        shutil.copy2(png, static)


def TOPAZ_Si_UB(app, window):
    directory = os.path.join(DIRECTORY, "TOPAZ")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(0)

    ub_presenter = window.ub
    ub_view = ub_presenter.view

    index = ub_view.instrument_combo.findText("TOPAZ")
    ub_view.instrument_combo.setCurrentIndex(index)
    ub_view.ipts_line.setText("36169")
    ub_view.runs_line.setText("54145")

    ub_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ub_view.ipts_line.setStyleSheet("background-color: yellow;")
    ub_view.runs_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_q_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.convert_to_q_button, Qt.LeftButton)
    QTest.qWait(1000 * 60)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_convert_Q.png"), "png"
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
        os.path.join(directory, "Si_UB_find_peaks.png"), "png"
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
        os.path.join(directory, "Si_UB_primitive_cell.png"), "png"
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
        os.path.join(directory, "Si_UB_conventional_cell.png"), "png"
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
        os.path.join(directory, "Si_UB_refine_UB.png"), "png"
    )

    ub_view.optimize_combo.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(1)
    ub_view.peaks_table.selectRow(0)

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_view_peak.png"), "png"
    )

    ub_view.peaks_table.clearSelection()

    ub_view.tab_widget.setCurrentIndex(2)

    ub_view.slice_combo.setStyleSheet("background-color: yellow;")
    ub_view.slice_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_hkl_button.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.convert_to_hkl_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_slice_view.png"), "png"
    )

    copy_generated_pngs(directory)


def scenario_other_tab(app: QApplication, window: QWidget) -> None:
    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(1)
    app.primaryScreen().grabWindow(window.winId()).save("other_tab.png", "png")


if __name__ == "__main__":
    run_qt_scenario(TOPAZ_Si_UB)
