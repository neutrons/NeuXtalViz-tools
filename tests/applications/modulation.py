import os
import sys
import subprocess

from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

from utilities import run_qt_scenario, copy_generated_pngs

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def TOPAZ_MnCoGeAs_modulation(app, window):
    directory = os.path.join(DIRECTORY, "TOPAZ")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(0)

    ub_presenter = window.ub
    ub_view = ub_presenter.view

    index = ub_view.instrument_combo.findText("TOPAZ")
    ub_view.instrument_combo.setCurrentIndex(index)
    ub_view.ipts_line.setText("23996")
    ub_view.runs_line.setText("36079")
    ub_view.wl_min_line.setText("0.4")
    ub_view.wl_max_line.setText("3.5")

    ub_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ub_view.ipts_line.setStyleSheet("background-color: yellow;")
    ub_view.runs_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_q_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.convert_to_q_button, Qt.LeftButton)
    QTest.qWait(1000 * 90)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "MnCoGeAs_UB_convert_Q.png"), "png"
    )

    ub_view.instrument_combo.setStyleSheet("")
    ub_view.ipts_line.setStyleSheet("")
    ub_view.runs_line.setStyleSheet("")
    ub_view.convert_to_q_button.setStyleSheet("")

    ub_view.max_peaks_line.setText("1000")
    ub_view.density_threshold_line.setText("150")
    ub_view.min_distance_line.setText("0.05")

    ub_view.max_peaks_line.setStyleSheet("background-color: yellow;")
    ub_view.density_threshold_line.setStyleSheet("background-color: yellow;")
    ub_view.min_distance_line.setStyleSheet("background-color: yellow;")
    ub_view.find_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.find_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "MnCoGeAs_UB_find_peaks.png"), "png"
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
        os.path.join(directory, "MnCoGeAs_UB_primitive_cell.png"), "png"
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
        os.path.join(directory, "MnCoGeAs_UB_conventional_cell.png"), "png"
    )

    ub_view.cell_table.clearSelection()
    ub_view.select_button.setStyleSheet("")

    ub_view.ub_tab.setCurrentIndex(2)

    index = ub_view.optimize_combo.findText("Hexagonal")
    ub_view.optimize_combo.setCurrentIndex(index)

    ub_view.optimize_combo.setStyleSheet("background-color: yellow;")
    ub_view.refine_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.refine_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "MnCoGeAs_UB_refine_UB.png"), "png"
    )

    ub_view.optimize_combo.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    ub_view.refine_button.setStyleSheet("")

    ub_view.peaks_tab.setCurrentIndex(1)
    ub_view.round_box.setStyleSheet("background-color: yellow;")
    ub_view.index_tolerance_line.setStyleSheet("background-color: yellow;")
    ub_view.index_button.setStyleSheet("background-color: green;")

    ub_view.round_box.setChecked(False)
    ub_view.index_tolerance_line.setText("0.3")

    QTest.mouseClick(ub_view.index_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "MnCoGeAs_UB_index_peaks.png"), "png"
    )

    ub_view.round_box.setStyleSheet("")
    ub_view.index_tolerance_line.setStyleSheet("")
    ub_view.index_button.setStyleSheet("")
    ub_view.index_button.setStyleSheet("")

    # ---

    ub_view.tab_widget.setCurrentIndex(3)

    ub_view.param_eps_line.setStyleSheet("background-color: yellow;")
    ub_view.param_min_line.setStyleSheet("background-color: yellow;")
    ub_view.cluster_button.setStyleSheet("background-color: green;")

    ub_view.param_eps_line.setText("0.03")
    ub_view.param_min_line.setText("5")

    QTest.mouseClick(ub_view.cluster_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "MnCoGeAs_UB_view_cluster.png"), "png"
    )

    ub_view.param_eps_line.setStyleSheet("")
    ub_view.param_min_line.setStyleSheet("")
    ub_view.cluster_button.setStyleSheet("")

    # ub_view.peaks_table.clearSelection()

    # ub_view.tab_widget.setCurrentIndex(2)

    # ub_view.slice_combo.setStyleSheet("background-color: yellow;")
    # ub_view.slice_line.setStyleSheet("background-color: yellow;")
    # ub_view.convert_to_hkl_button.setStyleSheet("background-color: yellow;")

    # QTest.mouseClick(ub_view.convert_to_hkl_button, Qt.LeftButton)
    # QTest.qWait(1000 * 15)

    # app.primaryScreen().grabWindow(window.winId()).save(
    #     os.path.join(directory, "MnCoGeAs_UB_slice_view.png"), "png"
    # )

    # ub_view.slice_combo.setStyleSheet("")
    # ub_view.slice_line.setStyleSheet("")
    # ub_view.convert_to_hkl_button.setStyleSheet("")

    # ub_view.tab_widget.setCurrentIndex(0)

    # ub_view.save_ub_button.setStyleSheet("background-color: yellow;")

    # QTest.qWait(1000 * 1)

    # app.primaryScreen().grabWindow(window.winId()).save(
    #     os.path.join(directory, "MnCoGeAs_UB_save_UB.png"), "png"
    # )
    # ub_presenter.model.save_UB(
    #     os.path.join("/SNS/TOPAZ/IPTS-23996/shared/nxv/", "MnCoGeAs_UB.mat")
    # )

    copy_generated_pngs(directory)


SCENARIOS = {
    "TOPAZ_MnCoGeAs_modulation": TOPAZ_MnCoGeAs_modulation,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            print(f"Available: {', '.join(SCENARIOS)}")
            sys.exit(1)
        run_qt_scenario(SCENARIOS[name])
    else:
        script = os.path.abspath(__file__)
        failed = []
        for name in SCENARIOS:
            print(f"Running {name} ...")
            rc = subprocess.run([sys.executable, script, name]).returncode
            if rc != 0:
                failed.append(name)
        if failed:
            print(f"Failed: {', '.join(failed)}")
            sys.exit(1)
